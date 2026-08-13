# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and refresh inference profiles for the FlexTensor vLLM v2 worker."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast, get_args

import torch
from vllm.logger import init_logger
from vllm.v1.utils import compute_iteration_details

import flextensor
from flextensor.config import OffloadConfig, _register_env_var
from flextensor.contrib.vllm.v2.errors import VllmFlexTensorV2Error
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler

LOGGER = init_logger("vllm.flextensor.v2.inference_profile")
TIMING_BATCH_ENV_VAR = "FT_VLLM_TIMING_BATCH"
TimingBatch: TypeAlias = Literal["decode", "prefill"]
PROFILE_FILENAME = "profile.json"
_REPLAY_GENERATION_ATTR = "_ft_vllm_replay_generation"
_MISSING_LABEL_SAMPLE_LIMIT = 10
_register_env_var(TIMING_BATCH_ENV_VAR)


@dataclass(frozen=True, slots=True)
class ReplayPatch:
    original: Any
    installed: Any


def timing_batch_from_env() -> TimingBatch | None:
    value = os.environ.get(TIMING_BATCH_ENV_VAR, "decode").strip().lower()
    if not value:
        return None
    allowed = get_args(TimingBatch)
    if value not in allowed:
        raise VllmFlexTensorV2Error(f"{TIMING_BATCH_ENV_VAR} must be one of {allowed!r}; got {value!r}")
    return cast("TimingBatch", value)


def current_cudagraph_replay_generation() -> int:
    replay = getattr(getattr(torch.cuda, "CUDAGraph", None), "replay", None)
    return int(getattr(replay, _REPLAY_GENERATION_ATTR, 0))


def patch_cudagraph_replay_counter() -> ReplayPatch | None:
    replay = getattr(getattr(torch.cuda, "CUDAGraph", None), "replay", None)
    if not callable(replay):
        return None
    if hasattr(replay, _REPLAY_GENERATION_ATTR):
        return ReplayPatch(original=replay, installed=replay)

    def replay_with_generation(graph: Any, *args: Any, **kwargs: Any) -> Any:
        result = replay(graph, *args, **kwargs)
        replay_with_generation.__dict__[_REPLAY_GENERATION_ATTR] += 1
        return result

    replay_with_generation.__dict__[_REPLAY_GENERATION_ATTR] = 0
    torch.cuda.CUDAGraph.replay = replay_with_generation  # type: ignore[assignment]
    return ReplayPatch(original=replay, installed=replay_with_generation)


def restore_cudagraph_replay(patch: ReplayPatch) -> None:
    cudagraph = getattr(torch.cuda, "CUDAGraph", None)
    if cudagraph is not None and getattr(cudagraph, "replay", None) is patch.installed:
        cudagraph.replay = patch.original


def classify_timing_batch(scheduler_output: Any) -> TimingBatch | None:
    try:
        details = compute_iteration_details(scheduler_output)
        num_ctx_requests = details.num_ctx_requests
        num_generation_requests = details.num_generation_requests
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if num_ctx_requests > 0 and num_generation_requests == 0:
        return "prefill"
    if num_generation_requests > 0 and num_ctx_requests == 0:
        return "decode"
    return None


def load_saved_profile(config: OffloadConfig) -> TensorManagerState | None:
    if config.profile_storage_dir is None:
        return None
    profile_file = Path(config.profile_storage_dir) / PROFILE_FILENAME
    try:
        state = TensorManagerStateHandler.load_from_file(profile_file)
    except FileNotFoundError:
        LOGGER.warning("saved profile not found at %s; using conservative statistics", profile_file)
        return None
    except (OSError, KeyError, TypeError, ValueError) as exc:
        LOGGER.warning(
            "saved profile at %s is unreadable; using conservative statistics: %s",
            profile_file,
            exc,
        )
        return None
    LOGGER.info("saved profile loaded path=%s", profile_file)
    return state


def save_refreshed_profile(
    *,
    config: OffloadConfig,
    state: TensorManagerState,
) -> None:
    if config.profile_storage_dir is None:
        LOGGER.warning("profile refresh save failed; profile storage directory is unavailable")
        return
    try:
        report = flextensor.collect_offload_timing()
        if report is None:
            raise ValueError("offload timing report is empty")
        offload_unit_labels = [layer.label for layer in state.stats]
        durations = report.compute_budgets_by_profile_label(offload_unit_labels, conservative=True)
        missing_labels = sorted(label for label in offload_unit_labels if label not in durations)
        if missing_labels:
            raise ValueError(
                "offload timing report is incomplete: "
                f"missing={len(missing_labels)}/{len(offload_unit_labels)} "
                f"sample={missing_labels[:_MISSING_LABEL_SAMPLE_LIMIT]}"
            )
        refreshed = replace(
            state,
            stats=[layer.model_copy(update={"duration": durations[layer.label]}) for layer in state.stats],
        )
        profile_file = Path(config.profile_storage_dir) / PROFILE_FILENAME
        TensorManagerStateHandler.save_to_file(profile_file, refreshed)
    except Exception as exc:
        LOGGER.warning("profile refresh save failed; keeping the previous profile: %s", exc)
        return
    LOGGER.info("refreshed profile saved path=%s samples=%d", profile_file, report.num_passes)
