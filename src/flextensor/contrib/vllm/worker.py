# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlexTensor-enabled vLLM Worker for offloading support.

Usage:
    FT_ENABLED=1 vllm serve model \\
        --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker

See flextensor.config.OffloadConfig for configuration options (FT_* env vars).
"""

import atexit
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psutil
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker.gpu_worker import Worker

import flextensor
import flextensor.contrib.vllm.loader
from flextensor.config import load_config
from flextensor.contrib.vllm._drafter_device import ensure_drafter_on_device
from flextensor.contrib.vllm._logging import safely_install_flextensor_logging_bridge

safely_install_flextensor_logging_bridge()

LOGGER = logging.getLogger(__name__)

# Default include patterns for per-layer offloading in decoder-only transformer models.
# Each transformer layer gets its own trap, enabling the prefetch pipeline to
# overlap CPU→GPU transfers with GPU compute for subsequent layers.
# Override via FT_INCLUDE_PATTERNS env var or OffloadConfig(include_patterns=[...]).
VLLM_DEFAULT_INCLUDE_PATTERNS: list[str] = [
    "model.embed_tokens",
    "model.layers.*",
    "model.norm",
    "lm_head",
    "logits_processor",
]


def _GiB(b: int) -> float:  # noqa: N802
    return b / GiB_bytes


@contextmanager
def vllm_model_context(model_runner: Any) -> Generator[list[Any], None, None]:
    """Context manager for unwrapping/re-wrapping vLLM model wrappers.

    Extracts actual model from UBatchWrapper/CUDAGraphWrapper on entry.
    Updates wrapper with modified model on exit.

    Args:
        model_runner: The model runner containing the model (wrapped or unwrapped).

    Yields:
        A single-element list containing the unwrapped model.
        Modify model_container[0] to update the wrapper on exit.

    Example:
        >>> with vllm_model_context(self.model_runner) as model_container:
        ...     model_container[0] = flextensor.offload(model_container[0], config)
    """
    model_or_wrapper = model_runner.model
    actual_model = model_or_wrapper
    wrapper_attrs = []  # Tracks which wrapper attributes to update

    try:
        from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

        if isinstance(model_or_wrapper, UBatchWrapper):
            actual_model = model_or_wrapper.runnable
            wrapper_attrs.append("runnable")
            # Also track 'model' attribute if it points to the same object
            # vLLM's UBatchWrapper may use both attributes
            if hasattr(model_or_wrapper, "model") and model_or_wrapper.model is model_or_wrapper.runnable:
                wrapper_attrs.append("model")
    except ImportError:
        pass

    try:
        from vllm.compilation.wrapper import CUDAGraphWrapper

        if isinstance(model_or_wrapper, CUDAGraphWrapper):
            actual_model = model_or_wrapper.model
            wrapper_attrs.append("model")
    except ImportError:
        pass

    model_container = [actual_model]
    yield model_container

    if wrapper_attrs:
        for attr in wrapper_attrs:
            setattr(model_or_wrapper, attr, model_container[0])
    else:
        model_runner.model = model_container[0]


class FlexTensorOffloadWorker(Worker):
    """vLLM Worker with FlexTensor offloading support.

    Applies tensor offloading after model loading when FT_ENABLED=1.
    """

    def load_model(self):
        """Load model with optional FlexTensor offloading."""
        # Create offload config using the device set by init_device()
        offload_config = load_config(gpu_device=self.device.index)
        config_updates: dict[str, Any] = {
            "discovery_iters": max(offload_config.discovery_iters, 3),
            "profiling_iters": max(offload_config.profiling_iters, 2),
        }
        if offload_config.include_patterns == ["*"]:
            config_updates["include_patterns"] = VLLM_DEFAULT_INCLUDE_PATTERNS
        self._offload_config = offload_config.model_copy(update=config_updates)
        if self._offload_config.enabled:
            LOGGER.info("FlexTensor offloading enabled with config: %s", self._offload_config)

        if self._offload_config.enabled:
            old_device = self.vllm_config.load_config.device or self.vllm_config.device_config.device
            old_load_format = self.vllm_config.load_config.load_format

            # Use FlexTensor's custom loader: loads on CPU, processes CUDA-only ops
            # (FP8 quant, MLA attention) on GPU layer-by-layer, returns model on CPU
            self.vllm_config.load_config.load_format = "flextensor"
            self.vllm_config.load_config.device = "cpu"

            LOGGER.info(
                "FlexTensor: Loading model with CPU-first strategy (was: %s); load_format: %s -> flextensor",
                old_device,
                old_load_format,
            )

        super().load_model()

        if not self._offload_config.enabled:
            return
        if not self.vllm_config.model_config.enforce_eager:
            LOGGER.warning("FlexTensor offloading requires eager mode. Add --enforce-eager flag.")

        # Speculative-decoding drafter (e.g. MTP / Eagle) lives outside the
        # main model and is not reached by flextensor.offload(). FT's
        # CPU-first loader leaves its weights on CPU; warmup then crashes
        # when vLLM's @torch.compile layernorm helper sees CPU weights
        # against CUDA inputs (flex-tensor #140). Push the drafter to GPU
        # BEFORE flextensor.offload() walks the main model — some drafter
        # submodules (e.g. embed_tokens) can be identity-shared with the
        # main model. If offload runs first, FT installs forward patches
        # against the still-CPU tensor IDs; the later .to(cuda) swaps
        # those tensors out, and the first drafter forward then hits a
        # KeyError in FT's cpu_to_gpu_map on trap exit.
        ensure_drafter_on_device(self.model_runner, self.device)

        atexit.register(flextensor.release)
        # Extract, offload, and update model using context manager
        with vllm_model_context(self.model_runner) as model_container:
            model_container[0] = flextensor.offload(model_container[0], self._offload_config)

        self.warmup_and_profile_model()
        self.model_runner.model_memory_usage = flextensor.get_gpu_memory_usage().total_bytes
        LOGGER.info(
            "FlexTensor offloading applied (GPU usage: %.2f GiB)",
            _GiB(self.model_runner.model_memory_usage),
        )

    def warmup_and_profile_model(self) -> None:
        """Run discovery and profiling iterations for FlexTensor offloading.

        Executes discovery iterations to map parameters to traps, then runs
        profiling iterations at max batch size to collect layer statistics
        for the offloading strategy. Finally switches to inference mode.
        """
        self.compile_or_warm_up_model()
        # vLLM's first compile_or_warm_up_model() runs 2 forward passes through
        # the offloaded model; each one counts as a discovery iteration in the
        # OffloadManager, so the loop below only tops up to discovery_iters.
        compile_warm_iters = 2

        discovery_iters = self._offload_config.discovery_iters - compile_warm_iters
        for i in range(discovery_iters):
            LOGGER.info("FlexTensor: Discovery iteration %d/%d (num_tokens=1)", i + 1, discovery_iters)
            self.model_runner._dummy_run(1, skip_eplb=True)  # noqa: SLF001

        # Pause profiling around vLLM's mixed-batch warmup: durations are suppressed
        # and the PROFILING counter is frozen, so only the explicit iterations below
        # feed the profile. The context manager also guarantees resume on exception.
        with flextensor.pause_profiling():
            self.compile_or_warm_up_model()
        max_num_tokens = min(self.model_runner.max_model_len, self.vllm_config.scheduler_config.max_num_batched_tokens)
        profiling_iters = self._offload_config.profiling_iters
        for i in range(profiling_iters):
            LOGGER.info(
                "FlexTensor: Profiling iteration %d/%d (max_num_tokens=%d)", i + 1, profiling_iters, max_num_tokens
            )
            self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

        LOGGER.info("FlexTensor: Switching to inference mode")
        # Informational-only host-memory hint; psutil probes /proc which can fail
        # on hardened containers (gVisor / distroless / restricted /proc).
        try:
            vm = psutil.virtual_memory()
            free_gib = vm.available / GiB_bytes
            if free_gib < 2.0:
                LOGGER.warning(
                    "FlexTensor: Low host memory (%.1f GiB free) — inference transition may OOM",
                    free_gib,
                )
        except (OSError, AttributeError, psutil.Error) as exc:
            LOGGER.debug(
                "FlexTensor: host-memory pre-check failed (%s); continuing without low-mem warning",
                exc,
            )
        self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001
        self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

    def shutdown(self) -> None:
        super().shutdown()
        flextensor.release()
