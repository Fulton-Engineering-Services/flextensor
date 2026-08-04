# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for strategy GPU budget reservation."""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from flextensor.collectors import LayerStatistics
from flextensor.tensor_processors import compute_reachable_tensor_map

LOGGER = logging.getLogger(__name__)
MIN_GPU_BUDGET_BYTES = 256 * 1024**2  # 256 MiB floor for strategy budget
_GIB = 1 << 30
_BLOCK_LOADER_TYPES = frozenset({"allocation_block_transfer", "raw_block_transfer"})


@dataclass(frozen=True)
class StrategyInvisibleGPUBudgetReservation:
    """Strategy budget after reserving permanent GPU tensors invisible to stats.

    Attributes:
        effective_budget: Budget passed to strategy computation after subtracting
            ``reserved_bytes``. ``None`` means no strategy memory constraint.
        reserved_bytes: Bytes reserved for reachable tensors that are absent from
            layer statistics but will remain on the target GPU for inference.
        reserved_count: Number of tensors contributing to ``reserved_bytes``.
    """

    effective_budget: int | None
    reserved_bytes: int
    reserved_count: int


@dataclass(frozen=True, slots=True)
class CUDAMemorySnapshot:
    """Raw CUDA/PyTorch memory counters and derived availability."""

    free_bytes: int
    total_bytes: int
    reserved_bytes: int
    allocated_bytes: int

    @classmethod
    def capture(cls, device_gpu: torch.device | str | int) -> "CUDAMemorySnapshot":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_gpu)
        return cls(
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            reserved_bytes=torch.cuda.memory_reserved(device_gpu),
            allocated_bytes=torch.cuda.memory_allocated(device_gpu),
        )

    @property
    def reusable_cache_bytes(self) -> int:
        return max(0, self.reserved_bytes - self.allocated_bytes)

    @property
    def available_bytes(self) -> int:
        return self.free_bytes + self.reusable_cache_bytes


def resolve_gpu_budget(
    max_gpu_mem_fraction: float | None,
    device_gpu: torch.device | str | int,
    *,
    min_gpu_budget_bytes: int = MIN_GPU_BUDGET_BYTES,
    logger: logging.Logger | None = None,
) -> int | None:
    """Resolve runtime GPU memory budget from fraction and current device state.

    Caps the fractional budget by actual available GPU memory to prevent OOM
    when CUDA context, KV cache, or framework buffers have already consumed memory.

    Available memory is computed as ``free_cuda + (reserved - allocated)``.
    The ``reserved - allocated`` term accounts for PyTorch allocator cache that
    CUDA reports as used but is actually reusable (reserved >= allocated is a
    PyTorch allocator invariant).
    """
    if max_gpu_mem_fraction is None:
        return None

    memory = CUDAMemorySnapshot.capture(device_gpu)
    budget = int(memory.total_bytes * max_gpu_mem_fraction)
    available = memory.available_bytes

    if available < min_gpu_budget_bytes:
        raise RuntimeError(
            f"Insufficient free GPU memory: {available / _GIB:.2f} GiB available "
            f"(free={memory.free_bytes / _GIB:.2f}, reserved={memory.reserved_bytes / _GIB:.2f}, "
            f"allocated={memory.allocated_bytes / _GIB:.2f}), "
            f"minimum required: {min_gpu_budget_bytes / _GIB:.2f} GiB"
        )

    if budget > available:
        budget_logger = logger or LOGGER
        budget_logger.warning(
            "Capping GPU memory budget from %.2f GiB to %.2f GiB "
            "(available: %.2f GiB free_cuda + %.2f GiB allocator cache)",
            budget / _GIB,
            available / _GIB,
            memory.free_bytes / _GIB,
            memory.reusable_cache_bytes / _GIB,
        )
        budget = available

    return budget


def resolve_gpu_mem_bytes(config: Any, *, context: str = "") -> int | None:
    """Resolve GPU memory limit from config to an absolute byte count.

    If ``max_gpu_mem_fraction`` is set, queries GPU device properties and returns
    ``int(total_memory * fraction)``. If ``None``, returns ``None`` for latency mode.
    """
    if config.max_gpu_mem_fraction is not None:
        try:
            props = torch.cuda.get_device_properties(config.gpu_device)
        except (RuntimeError, AssertionError) as e:
            ctx = f" while {context}" if context else ""
            raise RuntimeError(
                f"Failed to query GPU device {config.gpu_device}{ctx} "
                f"for max_gpu_mem_fraction={config.max_gpu_mem_fraction}. "
                f"Ensure CUDA is available and gpu_device={config.gpu_device} is valid."
            ) from e
        return int(props.total_memory * config.max_gpu_mem_fraction)

    return None


def tensor_needs_permanent_gpu_budget(tensor: torch.Tensor, device_gpu: torch.device) -> bool:
    """Whether inference setup may allocate new target-GPU storage for ``tensor``."""
    return not (device_gpu.type == "cuda" and tensor.device.type == "cuda" and tensor.device == device_gpu)


def compute_strategy_invisible_permanent_gpu_bytes(
    *,
    model: object | None,
    loader_type: str,
    device_gpu: torch.device,
    layer_stats: Iterable[LayerStatistics],
    tensors_map: Mapping[int, torch.Tensor],
) -> tuple[int, int]:
    """Return bytes/count for tensors absent from strategy stats but force-pinned to GPU."""
    if model is None:
        return 0, 0

    stats_tensor_ids = {tensor_info.tensor_id for layer in layer_stats for tensor_info in layer.tensors}
    reachable_tensors = compute_reachable_tensor_map(model)
    if not reachable_tensors:
        return 0, 0

    if loader_type == "strategy":
        # TensorStrategyLoader only force-pins traced tensors that are still
        # reachable from the model but absent from layer stats.
        candidate_tensor_ids = set(tensors_map).difference(stats_tensor_ids)
    elif loader_type in _BLOCK_LOADER_TYPES:
        # Block loaders finalize a view model and move every reachable
        # non-view-mapped tensor to GPU, including custom attributes that
        # are not present in tensors_map.
        candidate_tensor_ids = set(reachable_tensors).difference(stats_tensor_ids)
    else:
        return 0, 0

    total_bytes = 0
    count = 0
    for tensor_id in candidate_tensor_ids:
        tensor = reachable_tensors.get(tensor_id)
        if tensor is None:
            continue
        if not tensor_needs_permanent_gpu_budget(tensor, device_gpu):
            continue
        total_bytes += tensor.numel() * tensor.element_size()
        count += 1
    return total_bytes, count


def reserve_strategy_invisible_gpu_budget(
    max_gpu_mem_bytes: int | None,
    *,
    model: object | None,
    loader_type: str,
    device_gpu: torch.device,
    layer_stats: Iterable[LayerStatistics],
    tensors_map: Mapping[int, torch.Tensor],
    min_gpu_budget_bytes: int = MIN_GPU_BUDGET_BYTES,
) -> StrategyInvisibleGPUBudgetReservation:
    """Subtract permanent GPU residents from the strategy-compute memory budget."""
    if max_gpu_mem_bytes is None:
        return StrategyInvisibleGPUBudgetReservation(
            effective_budget=None,
            reserved_bytes=0,
            reserved_count=0,
        )

    invisible_bytes, invisible_count = compute_strategy_invisible_permanent_gpu_bytes(
        model=model,
        loader_type=loader_type,
        device_gpu=device_gpu,
        layer_stats=layer_stats,
        tensors_map=tensors_map,
    )
    if invisible_bytes == 0:
        return StrategyInvisibleGPUBudgetReservation(
            effective_budget=max_gpu_mem_bytes,
            reserved_bytes=0,
            reserved_count=0,
        )

    effective_budget = max_gpu_mem_bytes - invisible_bytes
    if effective_budget < min_gpu_budget_bytes:
        raise RuntimeError(
            "Insufficient strategy GPU budget after reserving strategy-invisible permanent GPU tensors: "
            f"original_budget={max_gpu_mem_bytes} bytes, "
            f"reserved_strategy_invisible_permanent_gpu={invisible_bytes} bytes "
            f"across {invisible_count} tensor(s), "
            f"effective_strategy_budget={max(0, effective_budget)} bytes, "
            f"minimum_required={min_gpu_budget_bytes} bytes. "
            "These tensors are reachable from the model but absent from layer statistics and would be "
            "moved or pinned on GPU during inference setup."
        )

    return StrategyInvisibleGPUBudgetReservation(
        effective_budget=effective_budget,
        reserved_bytes=invisible_bytes,
        reserved_count=invisible_count,
    )
