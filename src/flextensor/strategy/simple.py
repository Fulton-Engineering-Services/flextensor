# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple offloading strategies for FlexTensor.

This module provides simple, heuristic-based offloading strategies that don't
require complex optimization algorithms.
"""

from flextensor.collectors import LayerStatistics, TensorStatistics

from .protocol import StrategyResult
from .utils import _build_block_data, validate_memory_params

# =============================================================================
# Simple Strategy Helper Functions
# =============================================================================


def _calculate_load_time(
    layer: LayerStatistics,
    threshold_mb: float = 0.1,
) -> tuple[float, list[TensorStatistics]]:
    """Calculate total load time for weights in a layer above threshold.

    Args:
        layer: Layer statistics containing tensor information.
        threshold_mb: Minimum tensor size threshold in MB.

    Returns:
        Tuple of (total_load_time_ms, list_of_tensors_above_threshold).
    """
    tensors_load_time = 0.0
    tensors = []
    for tensor in layer.tensors:
        if (tensor.size_bytes / 1024 / 1024) > threshold_mb:
            tensors_load_time += tensor.load_time_ms
            tensors.append(tensor)
    return tensors_load_time, tensors


def _compute_offload_tensors_greedy(
    layer_stats: list[LayerStatistics],
    threshold_mb: float = 0.1,
    scale: float = 1.0,
) -> dict[str, list[TensorStatistics]]:
    """Compute offload strategy using greedy heuristic.

    Offloads weights when scaled cumulative compute time exceeds their load time.

    Args:
        layer_stats: List of layer statistics.
        threshold_mb: Minimum tensor size threshold in MB.
        scale: Multiplier on the cumulative compute budget.

    Returns:
        Strategy dict mapping layer labels to lists of weights to offload.
    """
    strategy: dict[str, list[TensorStatistics]] = {}
    upload_layer = layer_stats[0]
    cumulative_time_ms = 0.0

    for layer in layer_stats:
        load_time_in_layer, tensors = _calculate_load_time(layer, threshold_mb)
        if load_time_in_layer < cumulative_time_ms * scale:
            strategy[upload_layer.label] = tensors
            cumulative_time_ms = 0
            upload_layer = layer
        cumulative_time_ms += layer.duration

    return strategy


def _compute_offload_tensors_nth_layer(
    layer_stats: list[LayerStatistics],
    nth_layer: int,
    threshold_mb: float = 0.1,
) -> dict[str, list[TensorStatistics]]:
    """Compute offload strategy by offloading every Nth layer.

    Args:
        layer_stats: List of layer statistics.
        nth_layer: Interval between offloaded layers.
        threshold_mb: Minimum weight size threshold in MB.

    Returns:
        Strategy dict mapping layer labels to lists of weights to offload.
    """
    strategy: dict[str, list[TensorStatistics]] = {}
    upload_layer = layer_stats[0]
    cumulative_time_ms = 0.0
    c_idx = 0

    for _idx, layer in enumerate(layer_stats):
        _load_time_in_layer, tensors = _calculate_load_time(layer, threshold_mb)
        if c_idx == nth_layer:
            strategy[upload_layer.label] = tensors
            cumulative_time_ms = 0
            upload_layer = layer
            c_idx = 0
        cumulative_time_ms += layer.duration
        c_idx += 1

    return strategy


# =============================================================================
# Strategy Classes
# =============================================================================


class GreedyStrategy:
    """Greedy offloading strategy based on cumulative compute time heuristic."""

    def __init__(self, threshold_mb: float = 0.1, n_blocks: int = 4, scale: float = 1.0):
        """Initialize GreedyStrategy.

        Args:
            threshold_mb: Minimum tensor size threshold in MB.
            n_blocks: Number of blocks for block-based loaders.
            scale: Multiplier on the cumulative compute budget
                (< 1 adds safety margin, > 1 allows more transfers).
        """
        validate_memory_params(scale)
        self.threshold_mb = threshold_mb
        self.n_blocks = n_blocks
        self.scale = scale

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Compute greedy offload strategy.

        Args:
            layer_stats: List of layer statistics.
            memory_stats: Unused, accepted for interface consistency.
            max_gpu_mem_bytes: Unused, accepted for interface consistency.

        Returns:
            StrategyResult containing strategy_map and optional block_data.
        """
        if not layer_stats:
            return StrategyResult(strategy_map={}, block_data=None)
        _ = memory_stats, max_gpu_mem_bytes  # Unused, but accepted for interface consistency
        strategy_map = _compute_offload_tensors_greedy(layer_stats, self.threshold_mb, self.scale)
        block_data = _build_block_data(strategy_map, layer_stats, self.n_blocks)
        return StrategyResult(strategy_map=strategy_map, block_data=block_data)


class NthLayerStrategy:
    """Offloading strategy that offloads every Nth layer."""

    def __init__(self, nth_layer: int, threshold_mb: float = 0.1, n_blocks: int = 4):
        """Initialize NthLayerStrategy.

        Args:
            nth_layer: Interval between offloaded layers.
            threshold_mb: Minimum tensor size threshold in MB.
            n_blocks: Number of blocks for block-based loaders.
        """
        self.threshold_mb = threshold_mb
        self.nth_layer = nth_layer
        self.n_blocks = n_blocks

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Compute Nth layer offload strategy.

        Args:
            layer_stats: List of layer statistics.
            memory_stats: Unused, accepted for interface consistency.
            max_gpu_mem_bytes: Unused, accepted for interface consistency.

        Returns:
            StrategyResult containing strategy_map and optional block_data.
        """
        if not layer_stats:
            return StrategyResult(strategy_map={}, block_data=None)
        _ = memory_stats, max_gpu_mem_bytes  # Unused, but accepted for interface consistency
        strategy_map = _compute_offload_tensors_nth_layer(layer_stats, self.nth_layer, self.threshold_mb)
        block_data = _build_block_data(strategy_map, layer_stats, self.n_blocks)
        return StrategyResult(strategy_map=strategy_map, block_data=block_data)
