# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve LTX 2.3 Outpaint IC-LoRA with one FlexTensor manager per diffusion stage (+ optional text encoder)."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import subprocess  # noqa: S404 - used for nvidia-smi diagnostics and the optional ffmpeg gamma round-trip.
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import av
import torch
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.iclora_utils import append_ic_lora_reference_video_conditionings
from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
from ltx_pipelines.utils.types import ModalitySpec

import flextensor
from flextensor import OffloadConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (
    decode_json_payload,
    gpu_memory_snapshot,
    payload_value,
    require_external_license_ack,
    resolve_file,
    resolve_snapshot,
)

if TYPE_CHECKING:
    from ltx_pipelines.utils.args import ImageConditioningInput

EXPECTED_STAGE_CALLS = 2
STAGE_MANAGER_NAMES = {
    1: "ltx_stage1",
    2: "ltx_stage2",
}
TEXT_MANAGER_NAME = "ltx_text"
LOGGER = logging.getLogger(__name__)
DEFAULT_INCLUDE_PATTERNS = [
    "velocity_model.transformer_blocks.*",
]
# Gemma decoder layers (the bulk of the text encoder's weights) live under
# ``model.model.language_model.layers.*``. Override with --text-include-pattern.
DEFAULT_TEXT_INCLUDE_PATTERNS = [
    "model.model.language_model.layers.*",
]


def _snap_frames_to_8k1(frames: int) -> int:
    """Snap a frame count to the nearest valid ``8 * k + 1`` (LTX 8x temporal latent compression)."""
    k = max(0, round((frames - 1) / 8))
    return 8 * k + 1


class RequestTransformerCache:
    """Cache stage transformers (and optionally the Gemma text encoder), each behind its own FlexTensor manager."""

    def __init__(
        self,
        config: OffloadConfig,
        profile_dir: Path,
        from_profile: bool,
        *,
        text_config: OffloadConfig | None = None,
    ) -> None:
        self.config = config
        self.text_config = text_config
        self.offload_text = text_config is not None
        self.profile_dir = profile_dir
        self.from_profile = from_profile
        self.cache: dict[int, torch.nn.Module] = {}
        self.text_model: torch.nn.Module | None = None
        self.stage_index = 0
        self.original_build = DiffusionStage._build_transformer  # noqa: SLF001 - example hooks into LTX lifecycle.

    def reset_request(self) -> None:
        self.stage_index = 0

    def validate_request_complete(self) -> None:
        if self.stage_index != EXPECTED_STAGE_CALLS:
            raise RuntimeError(f"Expected exactly {EXPECTED_STAGE_CALLS} DiffusionStage calls, got {self.stage_index}.")

    def install(self) -> None:
        cache = self

        def cached_call(self: DiffusionStage, *args: Any, **kwargs: Any) -> Any:
            cache.stage_index += 1
            stage_index = cache.stage_index
            if stage_index not in STAGE_MANAGER_NAMES:
                raise RuntimeError(
                    f"Expected at most {EXPECTED_STAGE_CALLS} DiffusionStage calls per request, got {stage_index}."
                )
            if stage_index not in cache.cache:
                cache.cache[stage_index] = cache.build_transformer(self, stage_index, **kwargs)
            return self.run(cache.cache[stage_index], *args, **kwargs)

        DiffusionStage.__call__ = cached_call

        if self.offload_text:
            self._install_text()

    def _install_text(self) -> None:
        cache = self

        def patched_text_encoder_ctx(pe_self: PromptEncoder):
            if cache.text_model is None:
                LOGGER.info("Building text encoder (CPU) for FlexTensor manager %s", TEXT_MANAGER_NAME)
                cpu_encoder = pe_self._text_encoder_builder.build(  # noqa: SLF001 - example hooks into LTX lifecycle.
                    device=torch.device("cpu"),
                    dtype=pe_self._dtype,  # noqa: SLF001
                ).eval()
                LOGGER.info("Text encoder offload include_patterns: %s", cache.text_config.include_patterns)

                text_profile_dir = cache.profile_dir / "text"
                if cache.from_profile:
                    LOGGER.info("Loading FlexTensor profile for %s from %s", TEXT_MANAGER_NAME, text_profile_dir)
                    cache.text_model = flextensor.offload_from_profile(
                        cpu_encoder,
                        str(text_profile_dir),
                        config=cache.text_config,
                        name=TEXT_MANAGER_NAME,
                    )
                else:
                    LOGGER.info("Profiling FlexTensor manager %s", TEXT_MANAGER_NAME)
                    cache.text_model = flextensor.offload(
                        cpu_encoder,
                        config=cache.text_config,
                        name=TEXT_MANAGER_NAME,
                    )
            # nullcontext yields the cached managed model and skips gpu_model's device move.
            return contextlib.nullcontext(cache.text_model)

        PromptEncoder._text_encoder_ctx = patched_text_encoder_ctx  # noqa: SLF001

    def stage_profile_dir(self, stage_index: int) -> Path:
        return self.profile_dir / f"stage{stage_index}"

    def build_transformer(self, stage: DiffusionStage, stage_index: int, **kwargs: Any) -> torch.nn.Module:
        manager_name = STAGE_MANAGER_NAMES[stage_index]
        stage_profile_dir = self.stage_profile_dir(stage_index)
        build_kwargs = dict(kwargs)
        build_kwargs["device"] = torch.device("cpu")
        LOGGER.info("Building stage%s CPU transformer", stage_index)
        transformer = self.original_build(stage, **build_kwargs)
        if self.from_profile:
            LOGGER.info("Loading FlexTensor profile for %s from %s", manager_name, stage_profile_dir)
            return flextensor.offload_from_profile(
                transformer,
                str(stage_profile_dir),
                config=self.config,
                name=manager_name,
            )

        LOGGER.info("Profiling FlexTensor manager %s", manager_name)
        return flextensor.offload(transformer, config=self.config, name=manager_name)

    def save_profile(self) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        for stage_index, manager_name in STAGE_MANAGER_NAMES.items():
            stage_profile_dir = self.stage_profile_dir(stage_index)
            stage_profile_dir.mkdir(parents=True, exist_ok=True)
            flextensor.save_profile(str(stage_profile_dir), name=manager_name)
            LOGGER.info("Saved FlexTensor profile %s -> %s", manager_name, stage_profile_dir)
        if self.offload_text:
            text_profile_dir = self.profile_dir / "text"
            text_profile_dir.mkdir(parents=True, exist_ok=True)
            flextensor.save_profile(str(text_profile_dir), name=TEXT_MANAGER_NAME)
            LOGGER.info("Saved FlexTensor profile %s -> %s", TEXT_MANAGER_NAME, text_profile_dir)


def apply_gamma(src_path: str, dst_path: str, gamma: float, *, keep_audio: bool) -> None:
    """Apply a per-channel RGB gamma curve (``out = (in/255) ** (1/gamma) * 255``).

    This mirrors the author's ComfyUI ``Color Correct (mtb)`` node, which operates in
    full-range RGB where pure black is an exact fixed point. That property is the whole
    point of the dark-scene trick: after the forward brighten the letterbox bars must
    stay pure black so they remain the model's unambiguous "generate here" sentinel.

    ``gamma`` and ``1/gamma`` are reciprocal exponents, so a forward pass at ``g`` and an
    inverse pass at ``1/g`` round-trip to within 8-bit quantization. The forward pass uses
    lossless video to preserve the sentinel exactly.

    NOTE: do *not* use ffmpeg's ``eq=gamma`` here. ``eq`` applies gamma to limited-range
    luma (black is ``Y=16``, not 0), so brightening lifts the bars to ~56/255 in the file
    the model consumes, silently destroying the sentinel. Decoding to ``rgb24`` first makes
    this range-robust: the input's range tag is honoured, so black maps to 0 either way.
    """
    exponent = 1.0 / gamma
    expr = f"pow(clip(val/255,0,1),{exponent:.6f})*255"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-vf",
        f"format=rgb24,lutrgb=r='{expr}':g='{expr}':b='{expr}',format=yuv444p",
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv444p",
    ]
    cmd += ["-c:a", "copy"] if keep_audio else ["-an"]
    cmd += [dst_path]
    subprocess.run(cmd, check=True)  # noqa: S603 - args are constructed locally, not from untrusted input.


def stream_duration_s(path: str) -> float | None:
    """Return the first video stream duration in seconds when available."""
    container = av.open(path)
    try:
        video_stream = next(stream for stream in container.streams if stream.type == "video")
        if video_stream.duration is not None and video_stream.time_base is not None:
            return float(video_stream.duration * video_stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        return None
    finally:
        container.close()


def packet_start_time_s(packet: av.Packet) -> float | None:
    """Return a demuxed packet start timestamp in seconds when available."""
    timestamp = packet.pts if packet.pts is not None else packet.dts
    if timestamp is None or packet.time_base is None:
        return None
    return float(timestamp * packet.time_base)


def mux_source_audio(
    video_path: str,
    audio_source_path: str,
    output_path: str,
    *,
    target_duration_s: float | None,
) -> None:
    """Mux generated video with the source video's audio stream when present.

    Outpaint changes only pixels, so generated audio is not useful.  Remuxing
    via PyAV keeps this path inside the Python media stack already used by LTX
    rather than shelling out to another ffmpeg process.  When a capped-frame
    request produces a shorter video than the source, audio packets are copied
    only up to the generated video duration.
    """
    video_input = av.open(video_path)
    audio_input = av.open(audio_source_path)
    output = av.open(output_path, mode="w")
    success = False
    try:
        video_stream = next(stream for stream in video_input.streams if stream.type == "video")
        audio_stream = next((stream for stream in audio_input.streams if stream.type == "audio"), None)
        output_video_stream = output.add_stream_from_template(video_stream)
        output_audio_stream = output.add_stream_from_template(audio_stream) if audio_stream is not None else None

        for packet in video_input.demux(video_stream):
            if packet.dts is None:
                continue
            packet.stream = output_video_stream
            output.mux(packet)

        if audio_stream is not None and output_audio_stream is not None:
            for packet in audio_input.demux(audio_stream):
                if packet.dts is None:
                    continue
                packet_start_s = packet_start_time_s(packet)
                if target_duration_s is not None and packet_start_s is not None and packet_start_s >= target_duration_s:
                    break
                packet.stream = output_audio_stream
                output.mux(packet)
        success = True
    finally:
        output.close()
        video_input.close()
        audio_input.close()
        if not success:
            Path(output_path).unlink(missing_ok=True)


def finalize_output_audio(
    encode_target: str,
    conditioning_video: str,
    output_path: str,
    *,
    audio_mode: str,
    num_frames: int,
    frame_rate: float,
) -> float | None:
    """Apply final audio handling and return the target video duration."""
    target_duration_s = stream_duration_s(encode_target)
    if audio_mode != "copy":
        return target_duration_s

    if target_duration_s is None and frame_rate > 0:
        target_duration_s = num_frames / frame_rate
    mux_source_audio(
        encode_target,
        conditioning_video,
        output_path,
        target_duration_s=target_duration_s,
    )
    return target_duration_s


def _create_tiled_video_conditionings(
    pipeline: ICLoraPipeline,
    *,
    images: list[ImageConditioningInput],
    video_conditioning: list[tuple[str, float]],
    height: int,
    width: int,
    num_frames: int,
    video_encoder: Any,
    conditioning_attention_strength: float,
    conditioning_attention_mask: torch.Tensor | None,
    tiling_config: TilingConfig,
) -> list[Any]:
    """Build IC-LoRA conditionings using a tiled VAE encode.

    ``ICLoraPipeline._create_conditionings`` hard-codes ``tiling_config=None``,
    so the VAE encodes the reference video in a single pass.  At Stage 2 full
    resolution that single ``conv_in`` conv3d needs tens of GiB and OOMs.  This
    mirrors ``_create_conditionings`` but forwards a real ``tiling_config`` so the
    reference video is routed through ``VideoEncoder.tiled_encode`` and encoded
    tile-by-tile instead.
    """
    conditionings = combined_image_conditionings(
        images=images,
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=pipeline.dtype,
        device=pipeline.device,
    )
    append_ic_lora_reference_video_conditionings(
        conditionings,
        video_conditioning,
        height=height,
        width=width,
        num_frames=num_frames,
        video_encoder=video_encoder,
        dtype=pipeline.dtype,
        device=pipeline.device,
        reference_downscale_factor=pipeline.reference_downscale_factor,
        reference_temporal_scale_factor=pipeline.reference_temporal_scale_factor,
        conditioning_attention_strength=conditioning_attention_strength,
        conditioning_attention_mask=conditioning_attention_mask,
        tiling_config=tiling_config,
    )
    return conditionings


def install_stage2_video_conditioning_patch() -> None:
    """Patch ICLoraPipeline so Stage 2 also sees the full video conditioning.

    Upstream LTX 2.3 IC-LoRA only applies ``video_conditioning`` in Stage 1.  For
    outpaint, the black-mask reference video is the source of truth for every
    frame, so Stage 2 must receive the same video conditioning rather than
    refining only from the Stage 1 latent.
    """
    if getattr(ICLoraPipeline.__call__, "_flextensor_stage2_video_conditioning", False):
        return

    def patched_call(
        self: ICLoraPipeline,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        skip_stage_2: bool = False,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Any, Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=1.0,
                conditioning_attention_mask=None,
            )
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        video_state, audio_state = self.stage_1(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=audio_context),
        )

        if skip_stage_2:
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio

        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        # Stage 2 runs at full resolution, so encode the reference video with VAE
        # tiling to avoid a single multi-tens-of-GiB conv3d activation (OOM).
        stage_2_tiling_config = tiling_config or TilingConfig.default()
        # No per-region attention mask in Stage 2: a conditioning attention mask
        # forces ``build_attention_mask`` to allocate a dense (B, T, T) self-attention
        # bias over the full-resolution sequence (generation + reference tokens),
        # which is hundreds of GiB and OOMs. Both the tensor and scalar mask paths
        # build the same dense tensor, so we drop masking here and condition Stage 2
        # on the reference video at full strength (Stage 1 still applies the mask).
        stage_2_conditionings = self.image_conditioner(
            lambda enc: _create_tiled_video_conditionings(
                self,
                images=images,
                video_conditioning=video_conditioning,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=1.0,
                conditioning_attention_mask=None,
                tiling_config=stage_2_tiling_config,
            )
        )

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    patched_call._flextensor_stage2_video_conditioning = True  # noqa: SLF001
    ICLoraPipeline.__call__ = patched_call  # type: ignore[method-assign]


def install_stage2_lora_patch() -> None:
    """Patch ICLoraPipeline so Stage 2 uses the same IC-LoRA list as Stage 1."""
    if getattr(ICLoraPipeline.__init__, "_flextensor_stage2_lora", False):
        return

    original_init = ICLoraPipeline.__init__

    def patched_init(self: ICLoraPipeline, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        stage_1_loras = self.stage_1._transformer_builder.loras  # noqa: SLF001
        if stage_1_loras:
            LOGGER.info("Applying %s IC-LoRA(s) to Stage 2 transformer builder", len(stage_1_loras))
            self.stage_2._transformer_builder = self.stage_2._transformer_builder.with_loras(stage_1_loras)  # noqa: SLF001
            if hasattr(self.stage_2, "_streaming_builder"):
                self.stage_2._streaming_builder = self.stage_2._streaming_builder.with_loras(stage_1_loras)  # noqa: SLF001

    patched_init._flextensor_stage2_lora = True  # noqa: SLF001
    ICLoraPipeline.__init__ = patched_init  # type: ignore[method-assign]


class OutpaintFlexTensorService:
    def __init__(self, args: argparse.Namespace, *, from_profile: bool) -> None:
        self.args = args
        self.profile_dir = Path(args.profile_dir)
        self.config = OffloadConfig(
            include_patterns=args.include_patterns or DEFAULT_INCLUDE_PATTERNS,
            discovery_iters=args.discovery_iters,
            profiling_iters=args.profiling_iters,
            min_blocks=args.min_blocks,
            num_blocks=args.num_blocks,
            max_gpu_mem_fraction=args.max_gpu_mem_fraction,
            profile_storage_dir=str(self.profile_dir),
            profile_read_only=from_profile,
            enable_diagnostics=True,
        )
        # Aggressive config for the one-shot Gemma encoder (tiny resident footprint,
        # slower encode), offloading the decoder layers.
        text_config = None
        if args.offload_text:
            text_config = OffloadConfig(
                include_patterns=args.text_include_patterns or DEFAULT_TEXT_INCLUDE_PATTERNS,
                discovery_iters=args.discovery_iters,
                profiling_iters=args.profiling_iters,
                min_blocks=2,
                num_blocks=2,
                max_gpu_mem_fraction=args.text_mem_fraction,
                profile_storage_dir=str(self.profile_dir),
                profile_read_only=from_profile,
                enable_diagnostics=True,
            )
        self.cache = RequestTransformerCache(
            self.config,
            self.profile_dir,
            from_profile=from_profile,
            text_config=text_config,
        )
        self.cache.install()
        self.request_lock = Lock()
        install_stage2_video_conditioning_patch()
        install_stage2_lora_patch()
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=args.distilled_checkpoint_path,
            spatial_upsampler_path=args.spatial_upsampler_path,
            gemma_root=args.gemma_root,
            loras=[LoraPathStrengthAndSDOps(args.lora_path, args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)],
        )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.request_lock:
            return self._generate_locked(payload)

    def _generate_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = payload_value(payload, "prompt", self.args.prompt)
        conditioning_video = payload_value(payload, "conditioning_video", self.args.conditioning_video)
        output_path = payload_value(payload, "output_path", self.args.output_path)
        height = int(payload_value(payload, "height", self.args.height))
        width = int(payload_value(payload, "width", self.args.width))
        seed = int(payload_value(payload, "seed", self.args.seed))
        conditioning_strength = float(payload_value(payload, "conditioning_strength", self.args.conditioning_strength))
        gamma = float(payload_value(payload, "gamma", self.args.gamma))
        use_gamma = abs(gamma - 1.0) > 1e-6
        audio_mode = str(payload_value(payload, "audio_mode", self.args.audio_mode))
        if audio_mode not in {"copy", "none"}:
            raise ValueError(f"audio_mode must be one of copy, none; got {audio_mode!r}")

        src = get_videostream_metadata(conditioning_video)
        num_frames = int(payload_value(payload, "num_frames", self.args.num_frames or _snap_frames_to_8k1(src.frames)))
        frame_rate = float(payload_value(payload, "frame_rate", self.args.frame_rate or src.fps))

        self.cache.reset_request()
        torch.cuda.reset_peak_memory_stats()
        before_memory = gpu_memory_snapshot()
        start = time.time()
        video = None
        audio = None
        tmp_paths: list[str] = []
        target_duration_s: float | None = None
        try:
            # Optional dark-scene gamma trick: brighten the (already letterboxed) input
            # so real content lifts away from the pure-black "generate here" bars.
            cond_path = conditioning_video
            if use_gamma:
                fd, cond_path = tempfile.mkstemp(suffix=".mp4", prefix="outpaint_gamma_in_")
                os.close(fd)
                tmp_paths.append(cond_path)
                apply_gamma(conditioning_video, cond_path, gamma, keep_audio=False)

            # Use no_grad for persistent serving.  Cached server tensors must remain
            # regular tensors across requests (inference-mode tensors can fail later).
            with torch.no_grad():
                pipeline_call = getattr(type(self.pipeline).__call__, "__wrapped__", type(self.pipeline).__call__)
                video, audio = pipeline_call(
                    self.pipeline,
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=[],
                    video_conditioning=[(cond_path, conditioning_strength)],
                    tiling_config=TilingConfig.default(),
                    enhance_prompt=False,
                )
                self.cache.validate_request_complete()

                # When post-processing is needed, encode to a temp file first. This
                # keeps visual transforms and source-audio muxing explicit.
                encode_target = output_path
                needs_postprocess = use_gamma or audio_mode == "copy"
                if needs_postprocess:
                    fd, encode_target = tempfile.mkstemp(suffix=".mp4", prefix="outpaint_gamma_out_")
                    os.close(fd)
                    tmp_paths.append(encode_target)

                encode_video(
                    video=video,
                    fps=int(frame_rate),
                    audio=None,
                    output_path=encode_target,
                    video_chunks_number=get_video_chunks_number(num_frames, TilingConfig.default()),
                )

                if use_gamma:
                    gamma_target = output_path
                    if audio_mode == "copy":
                        fd, gamma_target = tempfile.mkstemp(suffix=".mp4", prefix="outpaint_gamma_final_")
                        os.close(fd)
                        tmp_paths.append(gamma_target)
                    apply_gamma(encode_target, gamma_target, 1.0 / gamma, keep_audio=False)
                    encode_target = gamma_target

                target_duration_s = finalize_output_audio(
                    encode_target,
                    conditioning_video,
                    output_path,
                    audio_mode=audio_mode,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                )
        finally:
            self.cache.reset_request()
            del video
            del audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for tmp_path in tmp_paths:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()

        return {
            "ok": True,
            "output_path": str(output_path),
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "gamma": gamma if use_gamma else None,
            "audio_mode": audio_mode,
            "target_duration_s": target_duration_s,
            "wall_s": time.time() - start,
            "torch_cuda_max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_memory_before": before_memory,
            "gpu_memory_after_cleanup": gpu_memory_snapshot(),
        }


def drive_profile(args: argparse.Namespace) -> None:
    service = OutpaintFlexTensorService(args, from_profile=False)
    manager_names = list(STAGE_MANAGER_NAMES.values())
    if args.offload_text:
        manager_names.append(TEXT_MANAGER_NAME)
    managers = [flextensor.get_offload_manager(manager_name) for manager_name in manager_names]

    for index in range(args.discovery_iters):
        result = service.generate({"output_path": f"{args.output_path}.discovery_{index}.mp4"})
        LOGGER.info("Discovery result:\n%s", json.dumps(result, indent=2))
        for manager in managers:
            manager.update_state()
        LOGGER.info("Discovery pass %s complete", index)

    for index in range(args.profiling_iters):
        result = service.generate({"output_path": f"{args.output_path}.profiling_{index}.mp4"})
        LOGGER.info("Profiling result:\n%s", json.dumps(result, indent=2))
        for manager in managers:
            manager.update_state()
        LOGGER.info("Profiling pass %s complete", index)

    service.cache.save_profile()


def serve(args: argparse.Namespace) -> None:
    service = OutpaintFlexTensorService(args, from_profile=True)
    LOGGER.info("Starting warmup")
    LOGGER.info("Warmup result:\n%s", json.dumps(service.generate({"output_path": args.warmup_output_path}), indent=2))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802, RUF100 - BaseHTTPRequestHandler requires this method name.
            if self.path != "/healthz":
                self.send_error(404, "Not Found")
                return

            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802, RUF100 - BaseHTTPRequestHandler requires this method name.
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = decode_json_payload(self.rfile.read(length) if length > 0 else b"{}")
                result = service.generate(payload)
                status = 200
            except (json.JSONDecodeError, ValueError) as exc:
                result = {"ok": False, "error": f"Invalid JSON: {exc}"}
                status = 400
            except Exception as exc:
                traceback.print_exc()
                result = {"ok": False, "error": repr(exc)}
                status = 500
            body = json.dumps(result, indent=2).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    LOGGER.info("Serving on http://%s:%s", args.host, args.port)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
        p.add_argument("--distilled-checkpoint-path")
        p.add_argument("--distilled-checkpoint-repo", default="Lightricks/LTX-2.3")
        p.add_argument("--distilled-checkpoint-filename", default="ltx-2.3-22b-distilled-1.1.safetensors")
        p.add_argument("--spatial-upsampler-path")
        p.add_argument("--spatial-upsampler-repo", default="Lightricks/LTX-2.3")
        p.add_argument("--spatial-upsampler-filename", default="ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        p.add_argument("--gemma-root")
        p.add_argument("--gemma-repo", default="google/gemma-3-12b-it-qat-q4_0-unquantized")
        p.add_argument("--lora-path")
        p.add_argument("--lora-repo", default="oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint")
        p.add_argument("--lora-filename", default="ltx-2.3-22b-ic-lora-outpaint.safetensors")
        p.add_argument(
            "--accept-external-licenses",
            action="store_true",
            help="Acknowledge upstream terms before automatically downloading Hugging Face artifacts.",
        )
        p.add_argument("--lora-strength", type=float, default=1.0)
        # Letterboxed source video: the source content padded to the target canvas with
        # pure-black bars in the regions to outpaint (prepare it beforehand).
        p.add_argument("--conditioning-video", required=True)
        p.add_argument("--conditioning-strength", type=float, default=1.0)
        p.add_argument("--num-frames", type=int, default=None)
        p.add_argument("--frame-rate", type=float, default=None)
        # Optional dark-scene gamma round-trip; 1.0 disables it. 2.0 matches the model card.
        p.add_argument("--gamma", type=float, default=1.0)
        p.add_argument("--audio-mode", choices=("copy", "none"), default="copy")
        p.add_argument("--output-path", default="/workspace/outputs/ltx23_outpaint_out.mp4")
        p.add_argument("--prompt", default="extend the scene naturally, consistent with the original footage")
        p.add_argument("--height", type=int, default=704)
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--seed", type=int, default=171198)
        p.add_argument("--profile-dir", default="/workspace/outputs/ltx23_outpaint_profile")
        p.add_argument("--discovery-iters", type=int, default=1)
        p.add_argument("--profiling-iters", type=int, default=1)
        p.add_argument("--min-blocks", type=int, default=2)
        p.add_argument("--num-blocks", type=int, default=2)
        p.add_argument("--max-gpu-mem-fraction", type=float, default=0.15)
        p.add_argument("--include-pattern", action="append", dest="include_patterns")
        # Optional Gemma text-encoder offload
        p.add_argument("--offload-text", action="store_true")
        p.add_argument("--text-mem-fraction", type=float, default=0.05)
        p.add_argument("--text-include-pattern", action="append", dest="text_include_patterns")

    profile = subparsers.add_parser("profile", help="Create a FlexTensor profile and exit.")
    add_common(profile)

    server = subparsers.add_parser("serve", help="Start a from-profile HTTP server.")
    add_common(server)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8020)
    server.add_argument("--warmup-output-path", default="/workspace/outputs/ltx23_outpaint_warmup.mp4")

    args = parser.parse_args()
    repos_to_download = []
    if not args.distilled_checkpoint_path:
        repos_to_download.append(args.distilled_checkpoint_repo)
    if not args.spatial_upsampler_path:
        repos_to_download.append(args.spatial_upsampler_repo)
    if not args.lora_path:
        repos_to_download.append(args.lora_repo)
    if not args.gemma_root:
        repos_to_download.append(args.gemma_repo)
    require_external_license_ack(args, repos_to_download)

    args.distilled_checkpoint_path = resolve_file(
        args.distilled_checkpoint_path,
        args.distilled_checkpoint_repo,
        args.distilled_checkpoint_filename,
        args.cache_dir,
    )
    args.spatial_upsampler_path = resolve_file(
        args.spatial_upsampler_path,
        args.spatial_upsampler_repo,
        args.spatial_upsampler_filename,
        args.cache_dir,
    )
    args.lora_path = resolve_file(args.lora_path, args.lora_repo, args.lora_filename, args.cache_dir)
    args.gemma_root = resolve_snapshot(args.gemma_root, args.gemma_repo, args.cache_dir)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.command == "profile":
        drive_profile(args)
    elif args.command == "serve":
        serve(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
