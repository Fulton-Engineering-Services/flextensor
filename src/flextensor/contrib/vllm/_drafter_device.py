# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drafter-on-device helper for FlexTensorOffloadWorker.

vLLM's speculative-decoding path stores the drafter at
``model_runner.drafter.model``. FT's CPU-first loader populates those weights,
but ``flextensor.offload()`` is only applied to the main model — so the
drafter stays on CPU. vLLM's ``@torch.compile``-decorated layernorm helper
then sees a CPU weight against a CUDA input at warmup, and Dynamo raises a
device mismatch.

Ordering: callers must invoke this *before* ``flextensor.offload()``. Some
drafter submodules (e.g. ``embed_tokens``) can be identity-shared with the
main model; if FT installs forward patches first, they are registered
against the still-CPU tensor IDs, and the subsequent ``.to(cuda)`` swaps
those tensors out — the first drafter forward then raises ``KeyError`` in
FT's ``cpu_to_gpu_map`` on trap exit.
"""

import logging

import torch

LOGGER = logging.getLogger(__name__)


def ensure_drafter_on_device(model_runner: object, device: torch.device | str | int) -> None:
    """Move ``model_runner.drafter.model`` to ``device`` if reachable.

    No-op when the runner has no drafter (non-speculative run, logged at
    DEBUG). Logs ``WARNING`` when a drafter is present but its ``.model`` is
    missing or lacks a callable ``.to()`` — warmup is likely to crash with a
    CPU/CUDA mismatch in that case. Exceptions from ``.to()`` propagate to
    the caller.

    Must be called before ``flextensor.offload(...)``; see the module
    docstring for the ordering invariant.

    Args:
        model_runner: vLLM GPU model runner, possibly with a speculative-
            decoding drafter attached at ``.drafter``.
        device: Target device for the drafter; passed through to
            ``nn.Module.to()``.
    """
    drafter = getattr(model_runner, "drafter", None)
    if drafter is None:
        LOGGER.debug("no drafter on model_runner (non-speculative run); nothing to move")
        return
    model = getattr(drafter, "model", None)
    if model is None:
        LOGGER.warning(
            "drafter present on %s but drafter.model is missing/None — "
            "vLLM spec-decode API may have changed; the #140 workaround is "
            "a no-op and warmup may crash with a CPU/CUDA device mismatch.",
            type(drafter).__name__,
        )
        return
    to = getattr(model, "to", None)
    if not callable(to):
        LOGGER.warning(
            "drafter.model (%s) has no callable .to(); cannot push to %s. "
            "Warmup may crash with a CPU/CUDA device mismatch (#140).",
            type(model).__name__,
            device,
        )
        return
    LOGGER.debug("moving drafter.model (%s) to %s", type(model).__name__, device)
    to(device)
