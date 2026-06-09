# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlexTensor-enabled vLLM Worker for offloading support.

Usage:
    FT_ENABLED=1 vllm serve model \\
        --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker

See flextensor.config.OffloadConfig for configuration options (FT_* env vars).
"""

import atexit
import importlib
import logging
from contextlib import contextmanager
from typing import Any, cast

import psutil
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker.gpu_worker import Worker

import flextensor
from flextensor.config import load_config
from flextensor.contrib.vllm._drafter_device import ensure_drafter_on_device
from flextensor.contrib.vllm._logging import safely_install_flextensor_logging_bridge
from flextensor.utils import config_field_was_set

safely_install_flextensor_logging_bridge()
# Register FlexTensor's vLLM load_format side effect.
importlib.import_module("flextensor.contrib.vllm.loader")

LOGGER = logging.getLogger(__name__)

# Default patterns for per-layer offloading in decoder-only and text-only
# hybrid-wrapper vLLM models.
# Each transformer layer gets its own trap, enabling the prefetch pipeline to
# overlap CPU→GPU transfers with GPU compute for subsequent layers.
# Override via FT_INCLUDE_PATTERNS env var or OffloadConfig(include_patterns=[...]).
VLLM_DEFAULT_INCLUDE_PATTERNS: list[str] = [
    # Prefer class-based layer selection so model path changes across vLLM
    # versions do not collapse the worker back to one coarse model trap.
    "class:*DecoderLayer",
    "class:*DecoderBlock",
    "class:*TransformerBlock",
    "model.embed_tokens",
    "model.norm",
    # Nemotron-H uses ``model.norm_f`` for the final decoder norm.
    "model.norm_f",
    "lm_head",
    "logits_processor",
    # Qwen3.5/3.6 hybrid wrappers keep the causal LM under
    # ``language_model`` even in text-only serving mode.
    "language_model.model.embed_tokens",
    "language_model.model.norm",
    "language_model.lm_head",
    "language_model.logits_processor",
]

# Some MoE sidecars can contain CUDA-only kernels or small router/gating tensors
# that are safer kept GPU-resident while the routed expert bulk remains
# available to FlexTensor. Use class-based excludes when the class identifies
# only the sidecar; keep name patterns for shared classes and parameter-level
# exclusions. Non-matching patterns are harmless for other architectures.
VLLM_DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "class:GateLinear",
    "model.layers.*.mixer.shared_experts",
    "model.layers.*.mixer.fc1_latent_proj",
    "model.layers.*.mixer.fc2_latent_proj",
    # Qwen3.5/3.6 shared experts are invoked by vLLM's FusedMoE runner
    # side path, including a separate CUDA stream, so keep them resident.
    "model.layers.*.mlp.shared_expert",
    "language_model.model.layers.*.mlp.shared_expert",
    # Qwen3.5/3.6 MoE router/gating tensors are tiny compared with expert
    # weights and are better kept resident while expert linears are offloaded.
    "language_model.model.layers.*.mlp.gate",
    "language_model.model.layers.*.mlp.shared_expert_gate",
    # Qwen3.5/3.6 GDN linear-attention kernels read these parameters inside
    # vLLM custom ops, outside the normal torch function argument rewrite path.
    "language_model.model.layers.*.linear_attn.A_log",
    "language_model.model.layers.*.linear_attn.dt_bias",
]

VLLM_COMPILE_WARMUP_FORWARD_COUNT = 2
VLLM_DISCOVERY_ITER_FLOOR = VLLM_COMPILE_WARMUP_FORWARD_COUNT + 1
VLLM_PROFILING_ITER_FLOOR = 2


def _GiB(b: int) -> float:  # noqa: N802
    return b / float(GiB_bytes)


_CUDA_GRAPH_WRAPPER_LOGGED = False


def _resolve_cuda_graph_wrapper() -> type[Any] | None:
    """Locate vLLM's ``CUDAGraphWrapper`` class.

    The primary import path logs at INFO the first time it succeeds.  The
    fallback import path logs at DEBUG so normal startup stays quiet while
    debug logs still show which wrapper was selected.  If both imports fail,
    return ``None`` and warn so callers can treat CUDA-graph wrapping as
    unavailable.
    """
    global _CUDA_GRAPH_WRAPPER_LOGGED
    primary_exc: ImportError | None = None
    try:
        from vllm.compilation.wrapper import CUDAGraphWrapper

        if not _CUDA_GRAPH_WRAPPER_LOGGED:
            LOGGER.info("FlexTensor: using CUDAGraphWrapper from vllm.compilation.wrapper (primary path)")
            _CUDA_GRAPH_WRAPPER_LOGGED = True
        return cast("type[Any]", CUDAGraphWrapper)
    except ImportError as exc:
        primary_exc = exc

    try:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper

        if not _CUDA_GRAPH_WRAPPER_LOGGED:
            LOGGER.debug(
                "FlexTensor: using CUDAGraphWrapper from vllm.compilation.cuda_graph (fallback path; "
                "primary import failed: %s)",
                primary_exc,
            )
            _CUDA_GRAPH_WRAPPER_LOGGED = True
        return cast("type[Any]", CUDAGraphWrapper)
    except ImportError as exc:
        LOGGER.warning(
            "FlexTensor: CUDAGraphWrapper not importable from either vllm.compilation.wrapper "
            "(%s) or vllm.compilation.cuda_graph (%s); CUDA-graph wrapping is unavailable in "
            "this vLLM version",
            primary_exc,
            exc,
        )
        return None


def _vllm_config_updates(offload_config: Any) -> dict[str, Any]:
    """Return vLLM-specific OffloadConfig updates.

    Args:
        offload_config: OffloadConfig-like object with discovery/profiling
            iteration counts and include/exclude pattern lists.

    Returns:
        Config update dictionary with minimum vLLM warmup iteration counts,
        default non-wildcard include patterns when the user did not customize
        includes, and MoE sidecar excludes when the user did not customize
        excludes.
    """
    # vLLM's first compile_or_warm_up_model() runs two forwards that already
    # count as FlexTensor discovery. Keep one additional explicit small-token
    # discovery pass, while using two max-token profiling passes to bound
    # startup cost.
    config_updates: dict[str, Any] = {
        "discovery_iters": max(offload_config.discovery_iters, VLLM_DISCOVERY_ITER_FLOOR),
        "profiling_iters": max(offload_config.profiling_iters, VLLM_PROFILING_ITER_FLOOR),
    }
    if offload_config.include_patterns == ["*"]:
        config_updates["include_patterns"] = VLLM_DEFAULT_INCLUDE_PATTERNS
    if offload_config.exclude_patterns == [] and not config_field_was_set(offload_config, "exclude_patterns"):
        config_updates["exclude_patterns"] = VLLM_DEFAULT_EXCLUDE_PATTERNS
    return config_updates


# Return type is `Any` for beartype + @contextmanager compatibility.
@contextmanager
def vllm_model_context(model_runner: Any) -> Any:
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
    wrapper_attrs: list[str] = []  # Tracks which wrapper attributes to update

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

    CUDAGraphWrapper = _resolve_cuda_graph_wrapper()  # noqa: N806 — class returned from helper, used as isinstance target

    if CUDAGraphWrapper is not None and isinstance(model_or_wrapper, CUDAGraphWrapper):
        actual_model = model_or_wrapper.model
        wrapper_attrs.append("model")

    model_container = [actual_model]
    try:
        yield model_container
    finally:
        if wrapper_attrs:
            for attr in wrapper_attrs:
                setattr(model_or_wrapper, attr, model_container[0])
        else:
            model_runner.model = model_container[0]


class FlexTensorOffloadWorker(Worker):
    """vLLM Worker with FlexTensor offloading support.

    Applies tensor offloading after model loading when FT_ENABLED=1.
    """

    def load_model(self) -> None:
        """Load model with optional FlexTensor offloading."""
        # Create offload config using the device set by init_device()
        offload_config = load_config(gpu_device=self.device.index)
        config_updates = _vllm_config_updates(offload_config)
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
        compile_warm_iters = VLLM_COMPILE_WARMUP_FORWARD_COUNT

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
            # First probe failure goes out at WARNING so an operator on a
            # locked-down container learns FlexTensor cannot pre-check OOM
            # risk; subsequent failures are demoted to DEBUG to avoid noise.
            if not getattr(self, "_ft_host_mem_probe_warned", False):
                LOGGER.warning(
                    "FlexTensor: host-memory pre-check failed (%s) — unable to warn about OOM risk "
                    "before inference transition; further probe failures will be logged at DEBUG",
                    exc,
                )
                self._ft_host_mem_probe_warned = True
            else:
                LOGGER.debug("FlexTensor: host-memory pre-check failed again (%s)", exc)
        self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001
        self.model_runner._dummy_run(max_num_tokens, skip_eplb=True)  # noqa: SLF001

    def shutdown(self) -> None:
        super().shutdown()
        flextensor.release()
