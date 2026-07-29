# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve LTX 2.3 LipDub with one FlexTensor manager per diffusion stage (+ optional text encoder)."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.lipdub import LipDubPipeline, _snap_frames_to_8k1
from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata

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
        """Initialize per-stage transformer cache settings.

        Args:
            config: FlexTensor offload configuration used for each stage manager.
            profile_dir: Directory containing one profile subdirectory per stage.
            from_profile: Whether cached transformers should load existing profiles instead of profiling.
            text_config: Optional FlexTensor config for the Gemma text encoder manager.
        """
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
        """Reset stage-call accounting before or after one LipDub request."""
        self.stage_index = 0

    def validate_request_complete(self) -> None:
        """Verify that the request exercised the expected pair of diffusion stages."""
        if self.stage_index != EXPECTED_STAGE_CALLS:
            raise RuntimeError(f"Expected exactly {EXPECTED_STAGE_CALLS} DiffusionStage calls, got {self.stage_index}.")

    def install(self) -> None:
        """Install LTX hooks that reuse cached FlexTensor-managed components."""
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
        """Install a PromptEncoder hook that caches a FlexTensor-managed Gemma encoder."""
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
        """Return the profile directory for one diffusion stage."""
        return self.profile_dir / f"stage{stage_index}"

    def build_transformer(self, stage: DiffusionStage, stage_index: int, **kwargs: Any) -> torch.nn.Module:
        """Build or restore a stage transformer under that stage's FlexTensor manager."""
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
        """Persist one FlexTensor profile for each cached manager."""
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


class LipDubFlexTensorService:
    """Stateful LTX LipDub service backed by cached FlexTensor managers."""

    def __init__(self, args: argparse.Namespace, *, from_profile: bool) -> None:
        """Initialize the pipeline, offload configuration, and stage transformer cache."""
        self.args = args
        self.profile_dir = Path(args.profile_dir)
        self.config = OffloadConfig(
            include_patterns=args.include_patterns or DEFAULT_INCLUDE_PATTERNS,
            # ``skip_discovery=False`` because ``drive_profile`` below
            # explicitly loops over ``args.discovery_iters`` to build a
            # saved profile. Under the ``skip_discovery=True`` default,
            # the DISCOVERY phase would be short-circuited inside
            # ``offload()`` and the loop would execute in PROFILING /
            # INFERENCE instead of DISCOVERY.
            discovery_iters=args.discovery_iters,
            profiling_iters=args.profiling_iters,
            skip_discovery=False,
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
        self.pipeline = LipDubPipeline(
            distilled_checkpoint_path=args.distilled_checkpoint_path,
            spatial_upsampler_path=args.spatial_upsampler_path,
            gemma_root=args.gemma_root,
            ic_lora=LoraPathStrengthAndSDOps(args.lora_path, args.lora_strength, LTXV_LORA_COMFY_RENAMING_MAP),
        )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate one LipDub video from an HTTP-style request payload."""
        prompt = payload_value(payload, "prompt", self.args.prompt)
        reference_video = payload_value(payload, "reference_video", self.args.reference_video)
        output_path = payload_value(payload, "output_path", self.args.output_path)
        height = int(payload_value(payload, "height", self.args.height))
        width = int(payload_value(payload, "width", self.args.width))
        seed = int(payload_value(payload, "seed", self.args.seed))
        reference_strength = float(payload_value(payload, "reference_strength", self.args.reference_strength))

        self.cache.reset_request()
        torch.cuda.reset_peak_memory_stats()
        before_memory = gpu_memory_snapshot()
        start = time.time()
        video = None
        audio = None
        try:
            # Use no_grad for persistent serving.  LipDubPipeline.__call__ is
            # decorated with inference_mode for one-shot CLI usage, but cached
            # server tensors must remain regular tensors across requests.
            with torch.no_grad():
                pipeline_call = getattr(type(self.pipeline).__call__, "__wrapped__", type(self.pipeline).__call__)
                video, audio = pipeline_call(
                    self.pipeline,
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    images=[],
                    reference_video_path=reference_video,
                    reference_strength=reference_strength,
                    tiling_config=TilingConfig.default(),
                    enhance_prompt=False,
                )
                self.cache.validate_request_complete()
                src = get_videostream_metadata(reference_video)
                encode_video(
                    video=video,
                    fps=int(src.fps),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=get_video_chunks_number(
                        _snap_frames_to_8k1(src.frames),
                        TilingConfig.default(),
                    ),
                )
        finally:
            self.cache.reset_request()
            del video
            del audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "ok": True,
            "output_path": str(output_path),
            "wall_s": time.time() - start,
            "torch_cuda_max_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_memory_before": before_memory,
            "gpu_memory_after_cleanup": gpu_memory_snapshot(),
        }


def drive_profile(args: argparse.Namespace) -> None:
    """Run discovery/profiling passes and save FlexTensor profiles."""
    service = LipDubFlexTensorService(args, from_profile=False)
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
    """Start the from-profile HTTP service and process requests sequentially."""
    service = LipDubFlexTensorService(args, from_profile=True)
    LOGGER.info("Starting warmup")
    LOGGER.info("Warmup result:\n%s", json.dumps(service.generate({"output_path": args.warmup_output_path}), indent=2))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802, RUF100 - BaseHTTPRequestHandler requires this method name.
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = decode_json_payload(raw_body)
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

    server = HTTPServer((args.host, args.port), Handler)
    LOGGER.info("Serving on http://%s:%s", args.host, args.port)
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and resolve model artifact paths."""
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
        p.add_argument("--lora-repo", default="Lightricks/LTX-2.3-22b-IC-LoRA-LipDub")
        p.add_argument("--lora-filename", default="ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors")
        p.add_argument(
            "--accept-external-licenses",
            action="store_true",
            help="Acknowledge upstream terms before automatically downloading Hugging Face artifacts.",
        )
        p.add_argument("--lora-strength", type=float, default=1.0)
        p.add_argument("--reference-video", required=True)
        p.add_argument("--output-path", default="/workspace/outputs/ltx23_lipdub_out.mp4")
        p.add_argument("--prompt", default="a person speaking clearly")
        p.add_argument("--height", type=int, default=256)
        p.add_argument("--width", type=int, default=256)
        p.add_argument("--seed", type=int, default=171198)
        p.add_argument("--reference-strength", type=float, default=1.0)
        p.add_argument("--profile-dir", default="/workspace/outputs/ltx23_lipdub_profile")
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
    server.add_argument("--warmup-output-path", default="/workspace/outputs/ltx23_lipdub_warmup.mp4")

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
