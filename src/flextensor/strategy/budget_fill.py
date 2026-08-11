# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-first offload strategy that fills a hard GPU budget.

:class:`BudgetFillStrategy` keeps estimated peak GPU memory at or under
``max_gpu_mem_bytes`` by spreading offload across pipelinable layers. For a
candidate per-layer byte cap ``C``, each layer (except layer 0) contributes
tensors until that layer's offload reaches ``C``. The smallest feasible ``C``
is found by binary search; lower ``C`` means smaller per-slot transfers and
typically smaller pipeline blocks.

Memory is the hard constraint. Within a layer, tensors are ordered so those
more likely to hide under the previous layer's compute window
(``duration * scale``) are taken first: fit before size, then lower transfer
time, then ``tensor_id``. Block packing for feasibility minimizes
``sum(per-block max transfer sizes)`` locally (without changing shared
:class:`~flextensor.memory_block_planner.MemoryBlockPlanner` behaviour used by
other strategies). Block count is searched exhaustively over
``min_blocks..n_blocks``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from flextensor.collectors import LayerStatistics, TensorStatistics  # noqa: TC001
from flextensor.memory_block_planner import MemoryBlockPlanner
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator
from flextensor.strategy.protocol import BlockStrategyData, StrategyComputeError, StrategyResult
from flextensor.strategy.utils import (
    _build_block_data,
    _prepare_label_to_block_id,
    calculate_transfer_to_compute_map,
    compute_label_to_size_map,
    validate_memory_params,
)


def _peak_memory(result: StrategyResult, layer_stats: list[LayerStatistics]) -> int:
    """Peak GPU ~ sum(block sizes) + (model - offloaded)."""
    block_memory = 0
    if result.block_data is not None:
        sizes = result.block_data.block_sizes
        block_memory = sum(sizes.values()) if isinstance(sizes, dict) else sum(sizes)
    total_model = sum(tensor.size_bytes for layer in layer_stats for tensor in layer.tensors)
    total_offloaded = sum(tensor.size_bytes for tensors in result.strategy_map.values() for tensor in tensors)
    return block_memory + total_model - total_offloaded


def _block_count(result: StrategyResult) -> int:
    if result.block_data is None:
        return 0
    sizes = result.block_data.block_sizes
    if isinstance(sizes, dict):
        return len(sizes)
    return len(sizes)


@dataclass(frozen=True, slots=True)
class _LayerTensors:
    """Eligible tensors for one pipelinable layer, fit-then-size ordered."""

    layer_index: int
    tensors: tuple[TensorStatistics, ...]
    total_bytes: int


def _transfer_ms(
    tensor: TensorStatistics,
    interpolator: MemoryTransferInterpolator | None,
) -> float:
    if interpolator is not None:
        return float(interpolator.bytes_to_duration(tensor.size_bytes))
    return float(tensor.load_time_ms)


def _eligible_by_layer(
    layer_stats: list[LayerStatistics],
    *,
    threshold_mb: float,
    scale: float,
    interpolator: MemoryTransferInterpolator | None,
) -> list[_LayerTensors]:
    """Group eligible tensors by layer; prefer transfer fit, then size."""
    threshold_bytes = threshold_mb * 1024 * 1024
    by_layer: list[_LayerTensors] = []
    for index, layer in enumerate(layer_stats):
        if index == 0:
            continue
        window_ms = max(0.0, layer_stats[index - 1].duration * scale)
        scored: list[tuple[tuple[float | int, ...], TensorStatistics]] = []
        for tensor in layer.tensors:
            if tensor.size_bytes <= threshold_bytes:
                continue
            transfer_ms = _transfer_ms(tensor, interpolator)
            fits = transfer_ms <= window_ms
            # Memory fills the hard budget via the per-layer byte cap; this key
            # only ranks which tensors to take first within that cap.
            scored.append((
                (
                    0 if fits else 1,
                    -tensor.size_bytes,
                    transfer_ms,
                    tensor.tensor_id,
                ),
                tensor,
            ))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0])
        tensors = tuple(tensor for _, tensor in scored)
        by_layer.append(
            _LayerTensors(
                layer_index=index,
                tensors=tensors,
                total_bytes=sum(tensor.size_bytes for tensor in tensors),
            )
        )
    return by_layer


def _strategy_map_from_selection(
    layer_stats: list[LayerStatistics],
    selected_ids: set[int],
) -> dict[str, list[TensorStatistics]]:
    """Build pipeline ``strategy_map``: transfer layer *i* tensors during *i-1*."""
    strategy: dict[str, list[TensorStatistics]] = {}
    for index in range(1, len(layer_stats)):
        tensors = [tensor for tensor in layer_stats[index].tensors if tensor.tensor_id in selected_ids]
        if tensors:
            strategy[layer_stats[index - 1].label] = tensors
    return strategy


def _select_ids_for_cap(by_layer: list[_LayerTensors], cap_bytes: int) -> set[int]:
    """From each layer, take ranked tensors until offload reaches ``cap_bytes``."""
    selected: set[int] = set()
    for layer in by_layer:
        accumulated = 0
        for tensor in layer.tensors:
            if accumulated >= cap_bytes:
                break
            selected.add(tensor.tensor_id)
            accumulated += tensor.size_bytes
    return selected


def _sum_block_maxes(label_sizes: dict[str, int], allocation: dict[int, list[str]]) -> int:
    total = 0
    for labels in allocation.values():
        if labels:
            total += max(label_sizes[label] for label in labels)
    return total


def _available_blocks(
    label: str,
    blocks: dict[int, list[str]],
    adjacency: dict[str, set[str]],
    num_blocks: int,
) -> list[int]:
    neighbors = adjacency.get(label, set())
    available: list[int] = []
    for block_num in range(num_blocks):
        if any(existing in neighbors for existing in blocks[block_num]):
            continue
        available.append(block_num)
    return available


def _allocate_blocks_min_bytes(
    label_sizes: dict[str, int],
    adjacency: dict[str, set[str]],
    num_blocks: int,
    seed: dict[int, list[str]],
) -> dict[int, list[str]]:
    """Minimize sum of per-block max transfer sizes (BudgetFill-local).

    Shared :class:`MemoryBlockPlanner` keeps count-balanced packing for other
    strategies; feasibility here needs size-aware coloring so a hard GPU budget
    is not rejected when a valid packing exists.
    """
    labels = list(label_sizes.keys())
    if not labels:
        return {}

    blocks: dict[int, list[str]] = {i: [] for i in range(num_blocks)}
    block_max = [0] * num_blocks
    best = {block_id: list(ls) for block_id, ls in seed.items() if ls}
    best_cost = _sum_block_maxes(label_sizes, best)

    def search(index: int, current_cost: int) -> None:
        nonlocal best_cost, best
        if current_cost >= best_cost:
            return
        if index >= len(labels):
            best_cost = current_cost
            best = {b: list(ls) for b, ls in blocks.items() if ls}
            return

        label = labels[index]
        size = label_sizes[label]
        available = _available_blocks(label, blocks, adjacency, num_blocks)
        if not available:
            return

        def placement_key(block_num: int) -> tuple[int, int]:
            old_max = block_max[block_num]
            new_max = max(old_max, size)
            return (current_cost + (new_max - old_max), -old_max)

        for block_num in sorted(available, key=placement_key):
            old_max = block_max[block_num]
            new_max = max(old_max, size)
            delta = new_max - old_max
            blocks[block_num].append(label)
            block_max[block_num] = new_max
            search(index + 1, current_cost + delta)
            blocks[block_num].pop()
            block_max[block_num] = old_max

    # Exact search for typical trap counts; fall back to seed for large maps.
    if len(labels) <= 20:
        search(0, 0)
    return {block_id: ls for block_id, ls in best.items() if ls}


def _build_budget_fill_block_data(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    n_blocks: int,
) -> BlockStrategyData | None:
    """Like ``_build_block_data``, but pack blocks to minimize sum of max sizes."""
    if n_blocks <= 0:
        return None

    label_to_size_map = compute_label_to_size_map(layer_stats, strategy_map)
    if not label_to_size_map:
        return _build_block_data(strategy_map, layer_stats, n_blocks)

    planner = MemoryBlockPlanner(label_to_size_map)
    seed = planner.optimize_block_distribution(n_blocks)
    allocation = _allocate_blocks_min_bytes(
        dict(label_to_size_map),
        planner.adjacency_graph,
        n_blocks,
        seed,
    )
    block_sizes = planner.find_minimum_block_sizes(allocation)
    return BlockStrategyData(
        label_to_size_map=dict(label_to_size_map),
        allocation_ordered=allocation,
        block_sizes=block_sizes,
        label_to_block_id=_prepare_label_to_block_id(allocation),
        transfer_to_compute_map=calculate_transfer_to_compute_map(layer_stats, strategy_map),
    )


def _result_for_cap(
    layer_stats: list[LayerStatistics],
    by_layer: list[_LayerTensors],
    cap_bytes: int,
    n_blocks: int,
) -> StrategyResult:
    selected_ids = _select_ids_for_cap(by_layer, cap_bytes)
    strategy_map = _strategy_map_from_selection(layer_stats, selected_ids)
    return StrategyResult(
        strategy_map=strategy_map,
        block_data=_build_budget_fill_block_data(strategy_map, layer_stats, n_blocks),
    )


def _best_result_for_cap(
    layer_stats: list[LayerStatistics],
    by_layer: list[_LayerTensors],
    cap_bytes: int,
    block_counts: list[int],
    max_gpu_mem_bytes: int,
) -> tuple[StrategyResult, int] | None:
    """Best feasible plan for a fixed per-layer cap across block counts.

    Prefer lower peak, then fewer blocks. Returns ``None`` when no block count fits.
    """
    best: tuple[StrategyResult, int] | None = None
    best_peak = 0
    best_blocks = 0
    for n_blocks in block_counts:
        result = _result_for_cap(layer_stats, by_layer, cap_bytes, n_blocks)
        peak = _peak_memory(result, layer_stats)
        if peak > max_gpu_mem_bytes:
            continue
        blocks = n_blocks if n_blocks > 0 else _block_count(result)
        if best is None or peak < best_peak or (peak == best_peak and blocks < best_blocks):
            best = (result, peak)
            best_peak = peak
            best_blocks = blocks
    return best


class BudgetFillStrategy:
    """Offload tensors until estimated peak fits the GPU budget.

    Spreads offload evenly across layers via a per-layer byte cap so pipeline
    blocks and transfer pressure stay smaller than concentrating on few layers.
    Requires ``max_gpu_mem_bytes``. Constructor args are local to the strategy
    instance (not ``OffloadConfig`` fields).
    """

    def __init__(
        self,
        n_blocks: int = 4,
        threshold_mb: float = 0.1,
        scale: float = 1.0,
        min_blocks: int | None = None,
    ):
        """Initialize BudgetFillStrategy.

        Args:
            n_blocks: Maximum pipeline block count. Use ``0`` to skip blocks;
                otherwise must be ``>= 2`` (a single block cannot pipeline).
            threshold_mb: Minimum tensor size (MiB) considered for offload.
            scale: Multiplier on previous-layer duration when ranking transfer
                fit (``window = duration * scale``). Does not change the hard
                memory objective; only which tensors are preferred within each
                layer's byte cap.
            min_blocks: Minimum block count to search (default ``2`` when
                ``n_blocks >= 2``, else ``0`` when ``n_blocks == 0``). Every count in
                ``min_blocks..n_blocks`` is tried; peak vs block count is not
                monotonic so this range is not binary-searched.
        """
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if n_blocks == 1:
            raise ValueError("n_blocks must be 0 (no blocks) or >= 2 (pipelined blocks), got 1")
        if threshold_mb < 0:
            raise ValueError(f"threshold_mb must be >= 0, got {threshold_mb}")
        validate_memory_params(scale)

        if n_blocks == 0:
            resolved_min = 0
        elif min_blocks is None:
            resolved_min = 2
        else:
            resolved_min = min_blocks

        if n_blocks >= 2:
            if resolved_min < 2:
                raise ValueError(f"min_blocks must be >= 2 when n_blocks >= 2, got {resolved_min}")
            if resolved_min > n_blocks:
                raise ValueError(f"min_blocks ({resolved_min}) must be <= n_blocks ({n_blocks})")
        elif resolved_min != 0:
            raise ValueError(f"min_blocks must be 0 when n_blocks == 0, got {resolved_min}")

        self.n_blocks = n_blocks
        self.min_blocks = resolved_min
        self.threshold_mb = threshold_mb
        self.scale = scale

    def _block_counts(self, layer_stats: list[LayerStatistics]) -> list[int]:
        if self.n_blocks <= 0:
            return [0]
        # Cap by number of transferring layers (same idea as OptimizedRoundRobin).
        transfer_layers = max(1, len(layer_stats) - 1)
        effective_max = min(self.n_blocks, transfer_layers)
        effective_min = min(self.min_blocks, effective_max)
        return list(range(effective_min, effective_max + 1))

    def compute(  # noqa: C901 - budget validation, spread search, and fallback.
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Select residency under ``max_gpu_mem_bytes``.

        Args:
            layer_stats: Trap ownership, sizes, and compute durations.
            memory_stats: Optional H2D size→ms map used to score transfer fit.
                When omitted, per-tensor ``load_time_ms`` is used.
            max_gpu_mem_bytes: Hard peak GPU limit in bytes (required).

        Returns:
            StrategyResult with pipeline ``strategy_map`` and optional ``block_data``.
        """
        if max_gpu_mem_bytes is None:
            raise StrategyComputeError("BudgetFillStrategy requires max_gpu_mem_bytes")
        if max_gpu_mem_bytes <= 0:
            raise StrategyComputeError(f"max_gpu_mem_bytes must be positive, got {max_gpu_mem_bytes}")
        if len(layer_stats) < 2:
            raise StrategyComputeError("BudgetFillStrategy requires at least two layers for pipelining")

        block_counts = self._block_counts(layer_stats)
        empty_blocks = block_counts[0]
        total_model = sum(tensor.size_bytes for layer in layer_stats for tensor in layer.tensors)
        empty = StrategyResult(
            strategy_map={},
            block_data=_build_budget_fill_block_data({}, layer_stats, empty_blocks),
        )
        if total_model <= max_gpu_mem_bytes:
            return empty

        interpolator = MemoryTransferInterpolator(memory_stats) if memory_stats else None
        by_layer = _eligible_by_layer(
            layer_stats,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            interpolator=interpolator,
        )
        if not by_layer:
            warnings.warn(
                "BudgetFillStrategy: no eligible tensors to offload; "
                f"peak remains {total_model / 1024**3:.2f} GB above budget "
                f"{max_gpu_mem_bytes / 1024**3:.2f} GB",
                stacklevel=2,
            )
            return empty

        max_cap = max(layer.total_bytes for layer in by_layer)

        # Binary-search the smallest per-layer offload cap that meets the budget
        # under any block count in min_blocks..n_blocks.
        lo, hi = 1, max_cap
        best: StrategyResult | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = _best_result_for_cap(
                layer_stats,
                by_layer,
                mid,
                block_counts,
                max_gpu_mem_bytes,
            )
            if candidate is not None:
                best = candidate[0]
                hi = mid - 1
            else:
                lo = mid + 1

        if best is not None:
            return best

        # Best-effort: full eligible offload, pick lowest peak across block counts.
        fallback = _result_for_cap(layer_stats, by_layer, max_cap, block_counts[0])
        fallback_peak = _peak_memory(fallback, layer_stats)
        for n_blocks in block_counts[1:]:
            result = _result_for_cap(layer_stats, by_layer, max_cap, n_blocks)
            peak = _peak_memory(result, layer_stats)
            if peak < fallback_peak:
                fallback = result
                fallback_peak = peak
        warnings.warn(
            "BudgetFillStrategy could not meet GPU memory limit "
            f"({max_gpu_mem_bytes / 1024**3:.2f} GB). Estimated peak: "
            f"{fallback_peak / 1024**3:.2f} GB after offloading all eligible tensors.",
            stacklevel=2,
        )
        return fallback
