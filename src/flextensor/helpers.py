# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from torch.overrides import TorchFunctionMode

from flextensor.types import GPUMemoryUsage


class TrapNestingGuard:
    """Prevents nested trap entry on a shared pair of CUDA timing events.

    Traps execute sequentially (one per layer in forward()). A second
    ``acquire()`` before the matching ``release()`` would overwrite the
    start-event timestamp and silently corrupt timing, so we fail fast.

    One instance lives on ``TensorManager``; each trap calls
    ``acquire(trace_id)`` / ``release()`` via that shared instance.
    """

    def __init__(self) -> None:
        self._active = False

    def acquire(self, trace_id: str) -> None:
        """Mark a trap as active, or raise if one is already active.

        Args:
            trace_id: Name of the trap being entered (included in the error message).

        Raises:
            RuntimeError: If called while another trap is already active.
        """
        if self._active:
            raise RuntimeError(
                f"Nested traps are not supported: trap '{trace_id}' entered while another "
                f"trap is active. Shared CUDA events would produce incorrect timing."
            )
        self._active = True

    def release(self) -> None:
        self._active = False


class StatsTrap:
    """Context manager that measures layer execution time using shared CUDA events.

    Used during direct-mode profiling to accumulate per-layer durations
    on ``tensor_manager.traps_direct_stats``.

    Args:
        tensor_manager: The ``TensorManager`` instance owning the shared
            CUDA events and duration accumulators.
        name: Trace identifier for the layer being measured.
    """

    def __init__(self, tensor_manager: Any, name: str) -> None:
        self.tensor_manager = tensor_manager
        self.current_trace_id = name
        self.start_event = tensor_manager.trap_start_event
        self.end_event = tensor_manager.trap_end_event
        self._nesting_guard = tensor_manager.trap_nesting_guard

    def __enter__(self):
        self._nesting_guard.acquire(self.current_trace_id)
        self.start_event.record()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.end_event.record()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)
        self.tensor_manager.traps_direct_duration_ms += duration_ms
        self.tensor_manager.traps_direct_stats[self.current_trace_id] = duration_ms
        self._nesting_guard.release()
        return False


class NoOpTrap:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False


class EmptyFunctionModeTrap(TorchFunctionMode):
    def __init__(self):
        pass

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, _type, _value, _traceback):
        super().__exit__(_type, _value, _traceback)

    def __torch_function__(self, func, _types, args, kwargs=None):
        return func(*args, **(kwargs or {}))


class NoOpTensorManager:
    def __init__(
        self,
        device_gpu,
        benchmark_cls=None,
    ):
        self.device_gpu = device_gpu
        self.benchmark_cls = benchmark_cls
        self.traps_duration_ms = 0
        self.traps_direct_duration_ms = 0
        self.traps_direct_duration_ms = 0
        self.traps_direct_stats = {}
        self.tensor_statistics_map = {}
        self.tensors_map = {}
        self.traced_tensors = set()
        self.loader_type = ""

    def prepare_warmup_mode(self):
        pass

    def prepare_profile_mode(self):
        pass

    def prepare_profile_direct_mode(self):
        pass

    def prepare_infer_mode(self):
        pass

    def trap(self, _name):
        return NoOpTrap()

    def release_memory(self):
        self.traps_direct_duration_ms = 0

    def prepare_profile_direct_mode_model(self, model):
        return model

    def prepare_model(self, model):
        return model

    def prepare_final_model(self, model):
        return model

    def benchmark_context(self, _iterations: int = 10):
        return self.benchmark_cls(device_gpu=self.device_gpu)

    def run_profile_suite(self, _callback, _model=None, _direct_mode=True):
        pass

    def set_model(self, model):
        self.model = model

    def initialize_warmup(self):
        return self.model

    def initialize_profile(self):
        return self.model

    def initialize_inference(self):
        return self.model

    def shutdown(self):
        pass

    def get_gpu_memory_usage(self) -> GPUMemoryUsage:
        """Get GPU memory usage (returns zeros for disabled offload).

        When offload is disabled, no FlexTensor memory is allocated,
        so this returns a GPUMemoryUsage with all zero values.

        Returns:
            GPUMemoryUsage: All zero values
        """
        return GPUMemoryUsage(blocks_bytes=0, unmapped_tensors_bytes=0, total_bytes=0)
