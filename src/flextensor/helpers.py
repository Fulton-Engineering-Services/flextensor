# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import torch
from torch.overrides import TorchFunctionMode

from flextensor.types import GPUMemoryUsage


class StatsTrap:
    def __init__(self, tensor_manager, name):
        self.tensor_manager = tensor_manager
        self.current_trace_id = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.timer_start = time.time_ns()
        return self

    def __exit__(self, _type, _value, _traceback):
        _wait_time_start = time.time_ns()
        torch.cuda.synchronize()
        _wait_time_end = time.time_ns()
        self.timer_end = time.time_ns()
        duration_ms = (self.timer_end - self.timer_start) / 1e6
        self.tensor_manager.traps_direct_duration_ms += duration_ms
        self.tensor_manager.traps_direct_stats[self.current_trace_id] = duration_ms
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
