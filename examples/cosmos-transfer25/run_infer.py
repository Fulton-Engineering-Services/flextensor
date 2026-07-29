# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Cosmos-Transfer2.5 inference with FlexTensor offloading.

Uses two offload managers so each module's own forward defines its iteration
boundary: ``cosmos_net`` for the DiT denoiser (re-invoked every diffusion step)
and ``cosmos_text`` for the Qwen text encoder (run once per generation).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from cosmos_transfer2.config import InferenceArguments, InferenceOverrides, SetupArguments
from cosmos_transfer2.inference import Control2WorldInference

import flextensor
from flextensor import OffloadConfig

LOGGER = logging.getLogger(__name__)

NET_MANAGER_NAME = "cosmos_net"
TEXT_MANAGER_NAME = "cosmos_text"

CACHE_ENV_DEFAULTS = {
    "UV_CACHE_DIR": ".cache/ft158/uv",
    "HF_HOME": ".cache/ft158/hf",
    "HUGGINGFACE_HUB_CACHE": ".cache/ft158/hf/hub",
    "MPLCONFIGDIR": ".cache/ft158/matplotlib",
}

# Patterns are relative to each manager's offload root (model.net /
# model.text_encoder), so the "net." / "text_encoder." prefixes used for a
# composite root are dropped here.
NET_INCLUDE_PATTERNS = [
    "class:ControlAwareDiTBlock",
    "class:ControlEncoderDiTBlock",
    "class:FinalLayer",
    "crossattn_proj",
    "control_embedder",
    "img_context_proj",
]

# Relative to model.text_encoder (FlexTensorTextEncoder.model == Cosmos Qwen wrapper).
TEXT_INCLUDE_PATTERNS = [
    "model.model.embed_tokens",
    "class:Qwen2_5_VLDecoderLayer",
    "class:Qwen2_5_VLVisionBlock",
    "model.model.norm",
    "model.lm_head",
    "model.visual.merger",
]


class FlexTensorTextEncoder(torch.nn.Module):
    """Expose Cosmos TextEncoder weights as a real nn.Module subtree for FlexTensor.

    Cosmos stores the Qwen stack on ``TextEncoder.model``, but ``TextEncoder`` itself is not a
    ``torch.nn.Module``. Registering ``model`` here makes Qwen parameters visible to
    ``model.named_parameters()`` when the text encoder is offloaded by FlexTensor.
    """

    def __init__(self, text_encoder: Any) -> None:
        super().__init__()
        self._text_encoder = text_encoder
        self.model = text_encoder.model

    def _compute_text_embeddings_online(self, *args: Any, **kwargs: Any) -> Any:
        # Keep the underlying TextEncoder pointing at this phase's Qwen module.
        # FlexTensor swaps model objects between discovery/profile/inference; a
        # closure over an earlier bound method would keep using a stale Qwen.
        # Resolving through self.model here keeps Cosmos' plain TextEncoder in
        # sync with the active FlexTensor phase model.
        #
        # Cosmos Qwen creates rope_deltas lazily during the first forward.  New
        # FlexTensor phase-model copies may not have seen that lazy attribute yet,
        # but Qwen's forward path reads it unconditionally.
        if not hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        self._text_encoder.model = self.model
        if hasattr(self, "output_device"):
            self._text_encoder.output_device = self.output_device
        return self._text_encoder.compute_text_embeddings_online(*args, **kwargs)

    def compute_text_embeddings_online(self, *args: Any, **kwargs: Any) -> Any:
        return self._compute_text_embeddings_online(*args, **kwargs)


def _configure_writable_caches() -> None:
    """Keep uv/HF/matplotlib caches off the read-only mounted /root/.cache."""
    root = Path.cwd()
    for key, relative in CACHE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, str(root / relative))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _latent_frames_from_pixel_chunk(pixel_frames: int) -> int:
    if pixel_frames < 1:
        raise ValueError("--num-video-frames-per-chunk must be positive.")
    if (pixel_frames - 1) % 4 != 0:
        raise ValueError("--num-video-frames-per-chunk must be 1 modulo 4 for the Wan2.1 tokenizer.")
    return ((pixel_frames - 1) // 4) + 1


def _apply_chunk_state_override(model: Any, num_video_frames_per_chunk: int | None) -> None:
    if num_video_frames_per_chunk is None:
        return
    state_t = _latent_frames_from_pixel_chunk(num_video_frames_per_chunk)
    if hasattr(model, "config") and hasattr(model.config, "state_t"):
        LOGGER.info(
            "COSMOS_CHUNK_STATE_OVERRIDE num_video_frames_per_chunk=%s state_t=%s old_state_t=%s",
            num_video_frames_per_chunk,
            state_t,
            model.config.state_t,
        )
        model.config.state_t = state_t
    if getattr(model, "net", None) is not None and hasattr(model.net, "max_frames"):
        LOGGER.info(
            "COSMOS_NET_MAX_FRAMES_OVERRIDE num_video_frames_per_chunk=%s old_max_frames=%s",
            num_video_frames_per_chunk,
            model.net.max_frames,
        )
        model.net.max_frames = num_video_frames_per_chunk


def _replace_submodule(parent: torch.nn.Module, name: str, value: Any) -> None:
    """Point ``parent.name`` at ``value`` (a non-Module FlexTensor proxy).

    ``nn.Module.__setattr__`` refuses to bind a non-Module to a registered
    submodule name, so deregister the child from ``_modules`` and store the proxy
    as a plain attribute. ``parent.name`` then routes through the proxy (which
    follows FlexTensor's per-phase module swaps).
    """
    if name in parent._modules:  # noqa: SLF001
        del parent._modules[name]  # noqa: SLF001
    object.__setattr__(parent, name, value)


def _run_phase_passes(
    inference: Control2WorldInference,
    inference_samples: Any,
    output_dir: Path,
    manager_name: str,
) -> None:
    """Run full Cosmos generations to advance FlexTensor phases.

    The net manager advances on its own ``forward`` hook (one call per diffusion
    step). The text encoder runs via ``compute_text_embeddings_online`` (not
    ``forward``), so its hook never fires -- advance it manually. The number of
    passes matches ``manager.iters_before_inference``, which is path-aware:
    under ``skip_discovery=True`` (with patched modules) DISCOVERY
    is short-circuited inside ``offload()`` and only ``profiling_iters`` passes
    are needed; under the default ``skip_discovery=False`` the full
    ``discovery_iters + profiling_iters`` sequence runs.
    """
    manager = flextensor.get_offload_manager(manager_name)
    phase_root = output_dir / "_flextensor_phase"
    phase_root.mkdir(parents=True, exist_ok=True)

    total_passes = manager.iters_before_inference
    for pass_index in range(total_passes):
        pass_dir = phase_root / f"pass_{pass_index}"
        pass_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "FLEXTENSOR_PHASE_PASS index=%s/%s output_dir=%s",
            pass_index,
            total_passes,
            pass_dir,
        )
        inference.generate(inference_samples, output_dir=pass_dir)
        manager.update_state()
        LOGGER.info("FLEXTENSOR_PHASE_PASS_DONE index=%s", pass_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("assets/robot_example/edge/robot_edge_spec.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cosmos_transfer"))
    parser.add_argument("--profile-dir", type=Path, default=Path("outputs/cosmos_transfer_profile/cosmos"))
    parser.add_argument("--model", default="edge", choices=("depth", "edge", "seg", "vis"))
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-video-frames-per-chunk", type=int, default=None)
    parser.add_argument("--max-gpu-mem-fraction", type=float, default=None)
    parser.add_argument("--profiling-iters", type=int, default=10)
    parser.add_argument("--min-blocks", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--skip-profile-save", action="store_true")
    parser.add_argument("--runtime-device", default="cuda")
    parser.add_argument("--from-profile", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _configure_writable_caches()
    if args.max_gpu_mem_fraction is None:
        args.max_gpu_mem_fraction = 0.2
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ["COSMOS_FT_CPU_FIRST"] = "1"
    os.environ.setdefault("COSMOS_FT_TEXT_ENCODER_CUDA", "1")
    os.environ["COSMOS_FT_TEXT_ENCODER_FLEXTENSOR"] = "1"
    init_environment()
    offload_started: list[str] = []
    try:
        inference_samples, batch_hint_keys = InferenceArguments.from_files(
            [args.input_file],
            overrides=InferenceOverrides(
                num_steps=args.num_steps,
                max_frames=args.max_frames,
                num_video_frames_per_chunk=args.num_video_frames_per_chunk,
            ),
        )
        init_output_dir(args.output_dir, profile=False)

        setup = SetupArguments(
            output_dir=args.output_dir,
            model=args.model,
            disable_guardrails=True,
            offload_guardrail_models=False,
        )
        inference = Control2WorldInference(setup, batch_hint_keys=batch_hint_keys)
        model = inference.inference_pipeline.model
        _apply_chunk_state_override(model, args.num_video_frames_per_chunk)
        model.net.to("cpu")
        has_text_encoder = getattr(model, "text_encoder", None) is not None
        if has_text_encoder:
            model.text_encoder.model.to("cpu")
            model.text_encoder = FlexTensorTextEncoder(model.text_encoder)
        torch.cuda.empty_cache()

        # Two managers with UNIQUE names, each offloading its own subtree so the
        # loader's per-iteration reset boundary == each module's own forward.
        net_config = OffloadConfig(
            min_blocks=args.min_blocks,
            num_blocks=args.num_blocks,
            profiling_iters=args.profiling_iters,
            max_gpu_mem_fraction=args.max_gpu_mem_fraction,
            enable_diagnostics=True,
            include_patterns=NET_INCLUDE_PATTERNS,
        )
        text_config = OffloadConfig(
            min_blocks=args.min_blocks,
            num_blocks=args.num_blocks,
            profiling_iters=args.profiling_iters,
            max_gpu_mem_fraction=args.max_gpu_mem_fraction,
            enable_diagnostics=True,
            include_patterns=TEXT_INCLUDE_PATTERNS,
        )
        net_profile_dir = str(Path(args.profile_dir) / "net")
        text_profile_dir = str(Path(args.profile_dir) / "text")

        if args.from_profile:
            net_proxy = flextensor.offload_from_profile(
                model.net, net_profile_dir, config=net_config, name=NET_MANAGER_NAME
            )
            offload_started.append(NET_MANAGER_NAME)
            _replace_submodule(model, "net", net_proxy)
            if has_text_encoder:
                text_proxy = flextensor.offload_from_profile(
                    model.text_encoder, text_profile_dir, config=text_config, name=TEXT_MANAGER_NAME
                )
                offload_started.append(TEXT_MANAGER_NAME)
                _replace_submodule(model, "text_encoder", text_proxy)
        else:
            net_proxy = flextensor.offload(model.net, config=net_config, name=NET_MANAGER_NAME)
            offload_started.append(NET_MANAGER_NAME)
            _replace_submodule(model, "net", net_proxy)
            if has_text_encoder:
                text_proxy = flextensor.offload(model.text_encoder, config=text_config, name=TEXT_MANAGER_NAME)
                offload_started.append(TEXT_MANAGER_NAME)
                _replace_submodule(model, "text_encoder", text_proxy)

        model.tensor_kwargs["device"] = args.runtime_device
        model.tensor_kwargs_fp32["device"] = args.runtime_device
        model.rectified_flow.device = torch.device(args.runtime_device)
        if has_text_encoder:
            model.text_encoder.output_device = args.runtime_device
        torch.cuda.empty_cache()

        if not args.from_profile and not args.skip_profile_save:
            # The net manager advances automatically (model.net.forward fires its
            # phase hook once per step); the text manager is driven manually, one
            # tick per full generation.
            _run_phase_passes(
                inference,
                inference_samples,
                args.output_dir,
                TEXT_MANAGER_NAME if has_text_encoder else NET_MANAGER_NAME,
            )
            flextensor.save_profile(net_profile_dir, name=NET_MANAGER_NAME)
            if has_text_encoder:
                flextensor.save_profile(text_profile_dir, name=TEXT_MANAGER_NAME)
            LOGGER.info("FLEXTENSOR_PROFILE_DIR %s", args.profile_dir)

        output_paths = inference.generate(inference_samples, output_dir=args.output_dir)
        LOGGER.info("OUTPUT_PATHS %s", output_paths)
    finally:
        for _name in offload_started:
            flextensor.release(_name)
        cleanup_environment()


if __name__ == "__main__":
    main()
