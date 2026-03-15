# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy result evaluation for comparing offload strategies.

Provides scoring utilities that compute overhead, peak memory, and pipeline
violations from a :class:`StrategyResult`, enabling :class:`AdaptiveStrategy`
to rank candidates and select the best solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flextensor.collectors import LayerStatistics, TensorStatistics  # noqa: TC001
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator  # noqa: TC001
from flextensor.strategy.protocol import BlockStrategyData, StrategyResult  # noqa: TC001
from flextensor.strategy.utils import calculate_transfer_to_compute_map, reverse_transfer_to_compute_map


@dataclass
class StrategyScore:
    """Evaluation metrics for a single strategy result.

    Attributes:
        strategy_name: Human-readable name of the strategy that produced the result.
        peak_memory_bytes: Estimated peak GPU memory in bytes (blocks + non-offloaded tensors).
        estimated_overhead: Fractional overhead from synchronous transfers.
            ``0.0`` means all transfers fit within compute windows (perfect pipelining).
            ``0.05`` means 5% extra time beyond the baseline compute duration.
        consecutive_violations: Number of adjacent layer pairs assigned to the same block.
        is_valid: ``True`` when there are no violations and memory is within the limit.
    """

    strategy_name: str
    peak_memory_bytes: int
    estimated_overhead: float
    consecutive_violations: int
    is_valid: bool

    def __lt__(self, other: StrategyScore) -> bool:
        """Lower is better: valid beats invalid, then lower overhead, then lower peak memory.

        When both are invalid, prefer lower peak memory (closer to fitting
        within the constraint).  When both are valid with equal overhead,
        prefer lower peak memory (leaves more headroom).
        """
        if self.is_valid != other.is_valid:
            return self.is_valid
        if not self.is_valid:
            return self.peak_memory_bytes < other.peak_memory_bytes
        if not math.isclose(self.estimated_overhead, other.estimated_overhead, rel_tol=1e-9, abs_tol=1e-12):
            return self.estimated_overhead < other.estimated_overhead
        return self.peak_memory_bytes < other.peak_memory_bytes


def evaluate_strategy_result(
    result: StrategyResult,
    layer_stats: list[LayerStatistics],
    strategy_name: str,
    interpolator: MemoryTransferInterpolator | None = None,
    max_gpu_mem_bytes: int | None = None,
) -> StrategyScore:
    """Score a strategy result for comparison.

    Args:
        result: The :class:`StrategyResult` to evaluate.
        layer_stats: Layer statistics from profiling (used for durations and tensor sizes).
        strategy_name: Display name for this strategy candidate.
        interpolator: Transfer-time interpolator.  Required for overhead calculation.
            When ``None``, overhead is estimated as ``0.0`` (cannot be computed).
        max_gpu_mem_bytes: Hard GPU memory limit.  When set, results exceeding
            this limit are marked invalid.

    Returns:
        A :class:`StrategyScore` summarising the result quality.
    """
    peak_memory = _compute_peak_memory(result, layer_stats)
    overhead = _compute_overhead(result.strategy_map, layer_stats, interpolator) if interpolator else 0.0
    violations = (
        _count_consecutive_violations(result.block_data, result.strategy_map, layer_stats) if result.block_data else 0
    )

    memory_ok = max_gpu_mem_bytes is None or peak_memory <= max_gpu_mem_bytes
    is_valid = violations == 0 and memory_ok

    return StrategyScore(
        strategy_name=strategy_name,
        peak_memory_bytes=peak_memory,
        estimated_overhead=overhead,
        consecutive_violations=violations,
        is_valid=is_valid,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_peak_memory(result: StrategyResult, layer_stats: list[LayerStatistics]) -> int:
    """Peak GPU = block memory + (total model - total offloaded).

    For non-block strategies, block memory is zero and the peak is just
    total model - total offloaded.

    Pipeline model: ``strategy_map[label_i]`` holds tensors transferred during
    layer *i*'s compute for layer *i+1*.  The offloaded tensors reside on CPU
    and are loaded into blocks just before they are needed, so they reduce
    the resident GPU footprint.
    """
    block_memory = 0
    if result.block_data is not None:
        sizes = result.block_data.block_sizes
        block_memory = sum(sizes.values()) if isinstance(sizes, dict) else sum(sizes)

    total_model = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats)

    total_offloaded = sum(sum(t.size_bytes for t in tensors) for tensors in result.strategy_map.values())

    peak = block_memory + total_model - total_offloaded
    return peak


def _compute_overhead(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    interpolator: MemoryTransferInterpolator,
) -> float:
    """Estimate fractional overhead from synchronous (non-pipelined) transfers.

    Uses ``transfer_to_compute_map`` to correctly attribute transfers when
    strategies skip gap layers, and computes gap-aware transfer windows
    (sum of compute durations from transfer source to compute target).
    """
    t2c_map = calculate_transfer_to_compute_map(layer_stats, strategy_map)
    compute_to_transfer = reverse_transfer_to_compute_map(t2c_map)
    label_to_idx = {layer.label: idx for idx, layer in enumerate(layer_stats)}

    total_sync_overhead = 0.0
    total_compute = sum(layer.duration for layer in layer_stats)

    for i, layer in enumerate(layer_stats):
        transfer_label = compute_to_transfer.get(layer.label)
        if transfer_label is None:
            continue

        offloaded = strategy_map.get(transfer_label, [])
        transfer_bytes = sum(t.size_bytes for t in offloaded)
        if transfer_bytes <= 0:
            continue

        transfer_time = interpolator.bytes_to_duration(transfer_bytes)
        transfer_idx = label_to_idx.get(transfer_label)
        if transfer_idx is None:
            raise ValueError(f"Transfer label '{transfer_label}' not found in layer statistics")
        window = sum(layer_stats[j].duration for j in range(transfer_idx, i))
        if transfer_time > window:
            total_sync_overhead += transfer_time - window

    if total_compute <= 0:
        return 0.0
    return total_sync_overhead / total_compute


def _count_consecutive_violations(
    block_data: BlockStrategyData,
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
) -> int:
    """Count adjacent *transferring* layers assigned to the same block.

    Labels are checked in **execution order** (from ``layer_stats``), not in
    the block-grouped order stored in ``label_to_block_id``.  The latter
    groups labels by block (all block-0 labels first, then block-1, …),
    which would produce phantom violations between unrelated layers.

    Only layers that actually have offloaded tensors (non-empty entry in
    ``strategy_map``) are considered — sharing a block ID with a neighbour
    is harmless when no transfer occurs.

    Cyclic wrap-around is NOT counted — the loader handles it via
    cross-iteration synchronisation.
    """
    label_to_block = block_data.label_to_block_id
    transferring_labels = [
        layer.label for layer in layer_stats if layer.label in label_to_block and strategy_map.get(layer.label)
    ]
    if len(transferring_labels) < 2:
        return 0
    return sum(
        1
        for i in range(len(transferring_labels) - 1)
        if label_to_block[transferring_labels[i]] == label_to_block[transferring_labels[i + 1]]
    )
