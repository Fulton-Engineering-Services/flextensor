# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlexTensor-enabled vLLM Worker for offloading support.

Usage:
    FT_ENABLED=1 vllm serve model \\
        --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker

See flextensor.config.OffloadConfig for configuration options (FT_* env vars).
"""

import atexit
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import psutil
import tqdm
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker.gpu_worker import Worker

import flextensor
import flextensor.contrib.vllm.loader
from flextensor.config import load_config

LOGGER = init_logger(__name__)

_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"

# Default module patterns for per-layer offloading in LLaMA-style models.
# Each transformer layer gets its own trap, enabling the prefetch pipeline to
# overlap CPU→GPU transfers with GPU compute for subsequent layers.
# Override via FT_MODULE_PATTERNS env var or OffloadConfig(module_patterns=[...]).
VLLM_DEFAULT_MODULE_PATTERNS: list[str] = [
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
            "warmup_iters": max(offload_config.warmup_iters, 3),
            "profile_iters": max(offload_config.profile_iters, 2),
        }
        if offload_config.module_patterns == ["*"]:
            config_updates["module_patterns"] = VLLM_DEFAULT_MODULE_PATTERNS
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
        """Run warmup and profiling iterations for FlexTensor offloading.

        Executes warmup iterations to stabilize memory patterns, then runs
        profiling iterations at max batch size to collect layer statistics
        for the offloading strategy. Finally switches to inference mode.
        """
        self.compile_or_warm_up_model()
        compile_warm_iters = 2  # _dummy_run doesn't include sampling, account for it

        for _ in range(self._offload_config.warmup_iters - compile_warm_iters):
            self.model_runner._dummy_run(1, skip_eplb=True)  # noqa: SLF001

        self.compile_or_warm_up_model()
        max_num_tokens = min(self.model_runner.max_model_len, self.vllm_config.scheduler_config.max_num_batched_tokens)
        for _ in tqdm.tqdm(
            range(self._offload_config.profile_iters - compile_warm_iters),
            desc=f"FlexTensor: Profiling model with max_num_tokens={max_num_tokens}",
            bar_format=_BAR_FORMAT,
        ):
            self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

        LOGGER.info("FlexTensor: Switching to inference mode")
        try:
            vm = psutil.virtual_memory()
            free_gib = vm.available / GiB_bytes
            if free_gib < 2.0:
                LOGGER.warning(
                    "FlexTensor: Low host memory (%.1f GiB free) — inference transition may OOM",
                    free_gib,
                )
        except Exception:  # noqa: S110
            pass
        self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001
        self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

    def shutdown(self) -> None:
        super().shutdown()
        flextensor.release()
