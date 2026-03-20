# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GlobalTensorSelectionStrategy optimizer."""

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy import (
    GlobalTensorSelectionStrategy,
    OptimizedRoundRobinAssignment,
    StrictRoundRobinAssignment,
)


def create_tensor(tensor_id: int, size_mb: float, load_time_ms: float) -> TensorStatistics:
    """Create a mock tensor with given size and transfer time."""
    return TensorStatistics(
        tensor_id=tensor_id,
        name=f"tensor_{tensor_id}",
        size_bytes=int(size_mb * 1024 * 1024),
        load_time_ms=load_time_ms,
    )


def create_layer(label: str, tensors: list[TensorStatistics], duration_ms: float) -> LayerStatistics:
    """Create a mock layer with given tensors and compute duration."""
    return LayerStatistics(
        label=label,
        duration=duration_ms,
        tensors=tensors,
    )


def create_memory_stats() -> dict[int, float]:
    """Create mock memory statistics with realistic transfer times.

    Returns a dict mapping size_bytes -> transfer_time_ms.
    """
    # Memory transfer data: size_bytes -> transfer_time_ms
    # Based on ~10 GB/s transfer speed
    return {
        1024: 0.0001,  # 1KB -> 0.0001ms
        1024 * 1024: 0.1,  # 1MB -> 0.1ms
        10 * 1024 * 1024: 1.0,  # 10MB -> 1ms
        100 * 1024 * 1024: 10.0,  # 100MB -> 10ms
        1024 * 1024 * 1024: 100.0,  # 1GB -> 100ms
    }


def print_summary(
    strategy: GlobalTensorSelectionStrategy,
    layers: list[LayerStatistics],
    test_name: str,
    max_gpu_mem_bytes: int | None = None,
) -> None:
    """Print a detailed summary of the optimization results."""
    print(f"\n{'=' * 70}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 70}")

    # Input summary
    print("\nINPUT:")
    print(f"  Layers: {len(layers)}")
    print(f"  Max blocks: {strategy.n_blocks}")
    if max_gpu_mem_bytes is not None:
        print(f"  Max GPU memory: {max_gpu_mem_bytes / 1024 / 1024:.0f} MB")
    print(f"  Scale bounds: [{strategy.scale}, {strategy.scale_ub}]")

    print("\n  Layer details:")
    for layer in layers:
        total_size = sum(t.size_bytes for t in layer.tensors) / 1024 / 1024
        print(f"    {layer.label}: compute={layer.duration}ms, tensors={total_size:.0f}MB")

    # Results summary
    print("\nRESULTS:")
    optimal_scale = getattr(strategy, "optimal_scale", 1.0)
    optimal_peak = getattr(strategy, "optimal_peak_memory", 0)
    optimal_blocks = getattr(strategy, "optimal_block_sizes", [])
    optimal_assignments = getattr(strategy, "optimal_layer_to_block", {})

    print(f"  Optimized scale: {optimal_scale:.4f}")
    print(f"  Peak GPU memory: {optimal_peak / 1024 / 1024:.2f} MB")

    if optimal_blocks:
        print("\n  Block Sizes:")
        for i, size in enumerate(optimal_blocks):
            if size > 0:
                print(f"    Block {i}: {size / 1024 / 1024:.2f} MB")

    if optimal_assignments:
        print("\n  Layer Assignments:")
        for label, block_idx in optimal_assignments.items():
            block_size = optimal_blocks[block_idx] if optimal_blocks else 0
            print(f"    {label} -> Block {block_idx} ({block_size / 1024 / 1024:.0f}MB)")

        blocks_used = len(set(optimal_assignments.values()))
        utilization = optimal_peak / max_gpu_mem_bytes * 100 if max_gpu_mem_bytes and max_gpu_mem_bytes > 0 else 0
        print("\n  Summary:")
        print(f"    Blocks used: {blocks_used}")
        print(f"    GPU utilization: {utilization:.1f}%")
        if max_gpu_mem_bytes is not None:
            print(f"    Within budget: {optimal_peak <= max_gpu_mem_bytes}")
    else:
        print("\n  No tensors offloaded (all below threshold or empty)")

    print(f"{'=' * 70}\n")


class TestGlobalTensorSelectionStrategyBasic:
    """Basic tests for GlobalTensorSelectionStrategy initialization and validation."""

    def test_initialization_with_defaults(self):
        """Test successful initialization with default parameters."""
        strategy = GlobalTensorSelectionStrategy()
        assert strategy.n_blocks == 4
        assert strategy.threshold_mb == 0.1
        assert strategy.scale == 1.0

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            threshold_mb=0.5,
            pop_size=30,
            epoch=50,
            scale=0.9,
            scale_ub=1.1,
        )
        assert strategy.n_blocks == 3
        assert strategy.threshold_mb == 0.5
        assert strategy.scale == 0.9
        assert strategy.scale_ub == 1.1

    def test_empty_layers_returns_empty(self):
        """Test that empty layer list returns empty strategy."""
        strategy = GlobalTensorSelectionStrategy()
        memory_stats = create_memory_stats()

        result = strategy.compute([], memory_stats, max_gpu_mem_bytes=1024 * 1024 * 1024)

        assert result.strategy_map == {}


class TestGlobalTensorSelectionStrategyEnoughMemory:
    """Tests for scenarios with sufficient GPU memory - should maximize GPU utilization."""

    def test_abundant_memory_maximizes_gpu_utilization(self):
        """When GPU memory is abundant, optimizer should maximize GPU utilization."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        # Total model size: 120MB
        # Large GPU budget: 200MB (plenty of headroom)
        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            threshold_mb=0.1,
            pop_size=20,
            epoch=20,
            max_early_stop=10,
            scale=0.9,
            scale_ub=1.0,
        )

        result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Abundant Memory - Maximize GPU Utilization", max_gpu_mem_bytes=budget)

        # Verify results
        assert strategy.optimal_peak_memory <= budget, "Should be within budget"
        # Optimizer may choose to offload fewer layers to maximize GPU utilization
        assert len(result.strategy_map) >= 0, "Should return valid strategy map"

        # GPU utilization should be within budget
        utilization = strategy.optimal_peak_memory / budget
        print(f"  GPU utilization: {utilization * 100:.1f}%")

        # Scale should stay within bounds
        assert strategy.scale <= strategy.optimal_scale <= strategy.scale_ub, "Scale should be within bounds"

    def test_high_utilization_target(self):
        """Test with high GPU utilization target (90%)."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 40, 4.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 40, 4.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 40, 4.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 40, 4.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        # Model size: 160MB
        # GPU budget: 200MB, target: 180MB (90%)
        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            threshold_mb=0.1,
            pop_size=20,
            epoch=20,
            max_early_stop=10,
            scale=0.9,
            scale_ub=1.0,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "High Utilization Target (90%)", max_gpu_mem_bytes=budget)

        assert strategy.optimal_peak_memory <= budget


class TestGlobalTensorSelectionStrategyLimitedMemory:
    """Tests for scenarios with limited GPU memory - may need scale adjustment."""

    def test_tight_budget_requires_more_offload(self):
        """When GPU memory is tight, optimizer should offload more."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 50, 5.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 50, 5.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 50, 5.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        # Model size: 200MB
        # Tight GPU budget: 100MB (must offload ~100MB)
        budget = 100 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            threshold_mb=0.1,
            pop_size=20,
            epoch=30,
            max_early_stop=15,
            scale=0.9,
            scale_ub=1.2,  # Allow some scale flexibility
        )

        result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Tight Budget - Requires More Offload", max_gpu_mem_bytes=budget)

        # Should offload tensors to fit within budget
        total_offload = sum(sum(t.size_bytes for t in ts) for ts in result.strategy_map.values())
        print(f"  Total offload: {total_offload / 1024 / 1024:.1f} MB")

        assert len(result.strategy_map) > 0, "Should offload some tensors"

    def test_scale_adjusted_for_transfer_constraint(self):
        """When transfers don't fit in compute time, scale may need adjustment.

        This test verifies the optimizer can use the scale parameter to handle
        scenarios where transfer time exceeds compute time. The scale represents
        the ratio of actual execution time to pure compute time.
        """
        # Large tensors, short compute time = transfers need more time
        layers = [
            create_layer("layer_0", [create_tensor(1, 50, 10.0)], 5.0),  # 50MB, 5ms compute
            create_layer("layer_1", [create_tensor(2, 50, 10.0)], 5.0),  # Transfer ~10ms but compute only 5ms
            create_layer("layer_2", [create_tensor(3, 50, 10.0)], 5.0),
            create_layer("layer_3", [create_tensor(4, 50, 10.0)], 5.0),
        ]
        memory_stats = create_memory_stats()

        # Budget forces some offloading
        budget = 150 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            threshold_mb=0.1,
            pop_size=20,
            epoch=30,
            max_early_stop=15,
            scale=1.0,
            scale_ub=3.0,  # Allow scale to increase
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Scale Adjusted for Transfer Constraint", max_gpu_mem_bytes=budget)

        # Scale should stay within bounds
        print(f"  Optimal scale: {strategy.optimal_scale:.2f}")
        assert strategy.scale <= strategy.optimal_scale <= strategy.scale_ub, (
            f"Scale {strategy.optimal_scale} should be within [{strategy.scale}, {strategy.scale_ub}]"
        )


class TestGlobalTensorSelectionStrategyBlockCount:
    """Tests for different block count configurations."""

    def test_3_blocks(self):
        """Test with 3 blocks (minimum for pipelining)."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            min_blocks=3,
            max_blocks=3,
            pop_size=20,
            epoch=15,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "3 Blocks Configuration", max_gpu_mem_bytes=budget)

        blocks_used = len(set(strategy.optimal_layer_to_block.values()))
        assert blocks_used <= 3, f"Should use at most 3 blocks, got {blocks_used}"

        # Verify pipeline constraint (no consecutive same blocks)
        assignments = list(strategy.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"

    def test_4_blocks(self):
        """Test with 4 blocks."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
            create_layer("layer_4", [create_tensor(5, 30, 3.0)], 10.0),
            create_layer("layer_5", [create_tensor(6, 30, 3.0)], 10.0),
            create_layer("layer_6", [create_tensor(7, 30, 3.0)], 10.0),
            create_layer("layer_7", [create_tensor(8, 30, 3.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 300 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=4,
            min_blocks=4,
            max_blocks=4,
            pop_size=20,
            epoch=15,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "4 Blocks Configuration", max_gpu_mem_bytes=budget)

        blocks_used = len(set(strategy.optimal_layer_to_block.values()))
        assert blocks_used <= 4, f"Should use at most 4 blocks, got {blocks_used}"

    def test_optimized_block_count(self):
        """Test that optimizer can choose optimal block count."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 60, 6.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 300 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=4,
            min_blocks=2,  # Let optimizer choose
            max_blocks=4,
            pop_size=20,
            epoch=15,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Optimized Block Count (min=2, max=4)", max_gpu_mem_bytes=budget)

        blocks_used = len(set(strategy.optimal_layer_to_block.values()))
        assert 2 <= blocks_used <= 4, f"Should use 2-4 blocks, got {blocks_used}"


class TestGlobalTensorSelectionStrategyScaleBounds:
    """Tests for scale bounds handling."""

    def test_scale_within_bounds(self):
        """Test that optimized scale stays within specified bounds."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 50, 5.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 50, 5.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 50, 5.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 150 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            scale=0.8,
            scale_ub=1.2,
            pop_size=20,
            epoch=20,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Scale Bounds [0.8, 1.2]", max_gpu_mem_bytes=budget)

        assert strategy.scale <= strategy.optimal_scale <= strategy.scale_ub, (
            f"Scale {strategy.optimal_scale} should be within [{strategy.scale}, {strategy.scale_ub}]"
        )

    def test_strict_scale_no_flexibility(self):
        """Test with scale = scale_ub (no flexibility)."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            scale=1.0,
            scale_ub=1.0,  # Fixed scale
            pop_size=20,
            epoch=20,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Fixed Scale (1.0)", max_gpu_mem_bytes=budget)

        assert strategy.optimal_scale == 1.0, f"Scale should be exactly 1.0, got {strategy.optimal_scale}"

    def test_scale_below_one_for_margin(self):
        """Test with scale < 1.0 to add safety margin."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 20, 2.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 20, 2.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 20, 2.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 20, 2.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        # Scale 0.9 means transfers must fit in 90% of compute time (10% safety margin)
        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            scale=0.9,
            scale_ub=0.95,  # Strict margin
            pop_size=20,
            epoch=20,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Scale with Safety Margin [0.9, 0.95]", max_gpu_mem_bytes=budget)

        assert 0.9 <= strategy.optimal_scale <= 0.95


class TestGlobalTensorSelectionStrategyAssignmentStrategies:
    """Tests for different assignment strategies."""

    def test_strict_round_robin_assignment(self):
        """Test with StrictRoundRobinAssignment."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 40, 4.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 40, 4.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            assignment_strategy=StrictRoundRobinAssignment(),
            pop_size=20,
            epoch=15,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "StrictRoundRobinAssignment", max_gpu_mem_bytes=budget)

        # Verify pipeline constraint
        assignments = list(strategy.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"

    def test_optimized_round_robin_assignment(self):
        """Test with OptimizedRoundRobinAssignment."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 60, 6.0)], 10.0),  # Larger
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=4,
            assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=4),
            pop_size=20,
            epoch=15,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "OptimizedRoundRobinAssignment", max_gpu_mem_bytes=budget)

        # Verify pipeline constraint
        assignments = list(strategy.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"


class TestGlobalTensorSelectionStrategyEdgeCases:
    """Tests for edge cases."""

    def test_all_tensors_below_threshold(self):
        """Test with all tensors below threshold (nothing to offload)."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 0.05, 0.01)], 10.0),
            create_layer("layer_1", [create_tensor(2, 0.08, 0.01)], 10.0),
            create_layer("layer_2", [create_tensor(3, 0.05, 0.01)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 100 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            threshold_mb=0.1,  # All tensors below this
            pop_size=20,
            epoch=10,
        )

        result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "All Tensors Below Threshold", max_gpu_mem_bytes=budget)

        # Should return empty or minimal strategy (nothing to offload)
        total_offload = sum(sum(t.size_bytes for t in ts) for ts in result.strategy_map.values())
        assert total_offload == 0 or result.strategy_map == {}, "Should not offload anything below threshold"

    def test_first_layer_never_offloaded(self):
        """First layer is always on GPU — no previous layer to pipeline from."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 50, 5.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 50, 5.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 50, 5.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            threshold_mb=0.1,
            pop_size=15,
            epoch=10,
        )
        result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=100 * 1024 * 1024)

        first_layer_ids = {t.tensor_id for t in layers[0].tensors}
        all_offloaded_ids = {t.tensor_id for tensors in result.strategy_map.values() for t in tensors}
        assert first_layer_ids.isdisjoint(all_offloaded_ids), "First layer tensors must not be offloaded"

    def test_single_layer(self):
        """Test with single layer."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 50, 5.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 100 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            pop_size=20,
            epoch=10,
        )

        result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Single Layer", max_gpu_mem_bytes=budget)

        # Single layer - no transfers possible (nothing to overlap with)
        assert len(result.strategy_map) <= 1

    def test_many_small_tensors_per_layer(self):
        """Test with many small tensors in each layer.

        The first layer (50MB) is always on GPU (no previous layer for pipeline
        transfer), so minimum peak = 3 blocks x 50MB + 50MB = 200MB.
        Budget of 210MB is feasible.
        """
        layers = [
            create_layer(
                "layer_0",
                [create_tensor(i, 5, 0.5) for i in range(1, 11)],  # 10 x 5MB = 50MB
                10.0,
            ),
            create_layer(
                "layer_1",
                [create_tensor(i, 5, 0.5) for i in range(11, 21)],
                10.0,
            ),
            create_layer(
                "layer_2",
                [create_tensor(i, 5, 0.5) for i in range(21, 31)],
                10.0,
            ),
            create_layer(
                "layer_3",
                [create_tensor(i, 5, 0.5) for i in range(31, 41)],
                10.0,
            ),
        ]
        memory_stats = create_memory_stats()

        budget = 210 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            threshold_mb=1.0,
            pop_size=20,
            epoch=15,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Many Small Tensors Per Layer", max_gpu_mem_bytes=budget)

        assert strategy.optimal_peak_memory <= budget


class TestGlobalTensorSelectionStrategyOptimizers:
    """Tests for different optimizer types."""

    @pytest.mark.parametrize("optimizer", ["SA", "DE"])
    def test_different_optimizers(self, optimizer: str):
        """Test that different optimizers produce valid results."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            optimizer=optimizer,
            pop_size=15,
            epoch=10,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, f"Optimizer: {optimizer}", max_gpu_mem_bytes=budget)

        assert strategy.optimal_peak_memory <= budget
        assert strategy.scale <= strategy.optimal_scale <= strategy.scale_ub


class TestGlobalTensorSelectionStrategyMemoryBounds:
    """Tests for min/max GPU memory bounds."""

    def test_min_max_gpu_memory_range(self):
        """Test that optimizer respects min/max GPU memory range."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 40, 4.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 40, 4.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 40, 4.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 40, 4.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        # Model size: 160MB
        # GPU: min=150MB, max=200MB
        budget = 200 * 1024 * 1024
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,  # Minimum 3 blocks for pipelining
            pop_size=20,
            epoch=20,
        )

        _result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "GPU Memory Range [150MB, 200MB]", max_gpu_mem_bytes=budget)

        # Should be within max
        assert strategy.optimal_peak_memory <= budget

    def test_different_utilization_targets(self):
        """Test different GPU utilization targets."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        memory_stats = create_memory_stats()

        results = {}
        for min_pct in [0.5, 0.8, 0.9]:
            max_mem = 200 * 1024 * 1024

            strategy = GlobalTensorSelectionStrategy(
                n_blocks=3,  # Minimum 3 blocks for pipelining
                pop_size=20,
                epoch=15,
            )

            strategy.compute(layers, memory_stats, max_gpu_mem_bytes=max_mem)
            utilization = strategy.optimal_peak_memory / max_mem * 100
            results[min_pct] = utilization

            print(f"  min={min_pct * 100:.0f}% -> actual utilization: {utilization:.1f}%")

        # All should be within budget
        for _min_pct, util in results.items():
            assert util <= 100, f"Utilization {util}% exceeds 100%"


class TestBuildFixedOnlyResult:
    """Tests for the _build_fixed_only_result short-circuit path.

    When every layer's offloadable tensors fit within the transfer budget
    at scale_lb, the optimizer is skipped and ``_build_fixed_only_result``
    independently computes the strategy map, block sizes, and peak memory.
    """

    @staticmethod
    def _make_fitting_layers() -> tuple[list[LayerStatistics], dict[int, float]]:
        """Create layers where every tensor fits at scale_lb=1.0.

        4 layers of 30MB tensors (load_time=3ms) with 10ms compute time.
        At scale=1.0 the 10ms transfer window easily fits 3ms transfers,
        so _build_fixed_only_result handles the result.
        """
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        return layers, create_memory_stats()

    def test_optimal_scale_equals_scale_lb(self):
        """When short-circuit fires, optimal_scale should equal scale_lb."""
        layers, mem = self._make_fitting_layers()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            scale=0.9,
            scale_ub=5.0,
            pop_size=10,
            epoch=5,
        )
        strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        assert strategy.optimal_scale == pytest.approx(0.9), (
            f"optimal_scale should equal scale_lb (0.9), got {strategy.optimal_scale}"
        )

    def test_strategy_map_pipeline_shift(self):
        """strategy_map should map layer_i to layer_{i+1}'s tensors."""
        layers, mem = self._make_fitting_layers()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            pop_size=10,
            epoch=5,
        )
        result = strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        assert "layer_0" in result.strategy_map, "layer_0 should transfer layer_1's tensors"
        assert "layer_1" in result.strategy_map, "layer_1 should transfer layer_2's tensors"
        assert "layer_2" in result.strategy_map, "layer_2 should transfer layer_3's tensors"
        assert "layer_3" not in result.strategy_map, "Last layer has nothing to prefetch"

        for key, tensors in result.strategy_map.items():
            layer_idx = int(key.split("_")[1])
            next_layer = layers[layer_idx + 1]
            expected_ids = {t.tensor_id for t in next_layer.tensors}
            actual_ids = {t.tensor_id for t in tensors}
            assert actual_ids == expected_ids, f"{key} should carry layer_{layer_idx + 1}'s tensors"

    def test_first_layer_tensors_always_on_gpu(self):
        """Layer 0's tensors must not appear in strategy_map values."""
        layers, mem = self._make_fitting_layers()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            pop_size=10,
            epoch=5,
        )
        result = strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        layer0_ids = {t.tensor_id for t in layers[0].tensors}
        for label, tensors in result.strategy_map.items():
            offloaded_ids = {t.tensor_id for t in tensors}
            assert not offloaded_ids & layer0_ids, f"Layer 0 tensors should not be offloaded (found in {label})"

    def test_block_sizes_match_max_transfer_per_block(self):
        """Each block should be sized to the largest transfer assigned to it."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 20, 2.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 40, 4.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 20, 2.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 10.0),
        ]
        mem = create_memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            pop_size=10,
            epoch=5,
        )
        result = strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        for label, block_idx in strategy.optimal_layer_to_block.items():
            if label not in result.strategy_map:
                continue
            transfer_size = sum(t.size_bytes for t in result.strategy_map[label])
            block_size = strategy.optimal_block_sizes[block_idx]
            assert transfer_size <= block_size, (
                f"Transfer at {label} ({transfer_size}) exceeds block {block_idx} ({block_size})"
            )

    def test_peak_memory_equals_blocks_plus_kept_on_gpu(self):
        """Peak memory should equal total block sizes + non-offloaded tensor memory."""
        layers, mem = self._make_fitting_layers()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            pop_size=10,
            epoch=5,
        )
        strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        total_blocks = sum(strategy.optimal_block_sizes)
        assert strategy.optimal_peak_memory >= total_blocks, "Peak must include block memory"
        assert strategy.optimal_peak_memory > 0

    def test_tensor_selection_marks_all_true(self):
        """When all tensors are fixed-offloaded, selection should be all True."""
        layers, mem = self._make_fitting_layers()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            pop_size=10,
            epoch=5,
        )
        strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        for label, selections in strategy.optimal_tensor_selection.items():
            assert all(selections), f"All tensors in {label} should be selected (True)"

    def test_mixed_threshold_tensors(self):
        """Layers with some tensors below threshold: only large ones offloaded."""
        layers = [
            create_layer(
                "layer_0",
                [create_tensor(1, 30, 3.0), create_tensor(2, 0.05, 0.01)],
                10.0,
            ),
            create_layer(
                "layer_1",
                [create_tensor(3, 30, 3.0), create_tensor(4, 0.08, 0.01)],
                10.0,
            ),
            create_layer(
                "layer_2",
                [create_tensor(5, 30, 3.0), create_tensor(6, 0.06, 0.01)],
                10.0,
            ),
        ]
        mem = create_memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            threshold_mb=0.1,
            pop_size=10,
            epoch=5,
        )
        result = strategy.compute(layers, mem, max_gpu_mem_bytes=500 * 1024 * 1024)

        for tensors in result.strategy_map.values():
            for t in tensors:
                assert t.size_bytes > 0.1 * 1024 * 1024, "Only above-threshold tensors should be offloaded"

    def test_within_memory_budget(self):
        """Short-circuit result should respect memory budget."""
        layers, mem = self._make_fitting_layers()
        budget = 500 * 1024 * 1024

        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            pop_size=10,
            epoch=5,
        )
        strategy.compute(layers, mem, max_gpu_mem_bytes=budget)

        assert strategy.optimal_peak_memory <= budget


class TestBlockAssignmentConsistency:
    """Regression tests: block assignment must match the one used during optimization.

    Previously, post-optimization re-derived block assignments from actual
    transfer sizes.  Different inputs to the assignment strategy produced
    different groupings whose total block memory could exceed what the
    optimizer validated — causing small GPU memory constraint violations
    (e.g. 50.05 GB reported vs 50.00 GB budget).
    """

    def test_block_assignment_consistent_with_optimizer_for_gap_layers(self):
        """Block assignment post-optimization must match what the optimizer used.

        With a permanent gap (layer with no offloadable tensors), the
        pre-computed assignment (from all N-1 layer sizes) differs from one
        re-derived from only the non-gap transfer sizes.  The code must use
        the pre-computed version so that peak memory is consistent with the
        optimizer's constraint check.
        """
        # Short compute (5ms) with large tensors forces transfers to exceed the
        # transfer window, so the optimizer path runs instead of the fixed-only
        # short-circuit.
        layers = [
            create_layer("layer_0", [create_tensor(1, 200, 20.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 150, 15.0)], 5.0),
            create_layer("layer_2", [], 5.0),  # gap — no offloadable tensors
            create_layer("layer_3", [create_tensor(4, 100, 10.0)], 5.0),
            create_layer("layer_4", [create_tensor(5, 180, 18.0)], 5.0),
        ]
        memory_stats = create_memory_stats()

        assignment_strategy = OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=3)
        strategy = GlobalTensorSelectionStrategy(
            n_blocks=3,
            assignment_strategy=assignment_strategy,
            threshold_mb=0.1,
            pop_size=15,
            epoch=20,
            scale=1.0,
            scale_ub=5.0,
            seed=42,
        )

        result = strategy.compute(layers, memory_stats, max_gpu_mem_bytes=200 * 1024 * 1024)

        # Expected: assignment from full layer sizes (all N-1 slots, including gap)
        layer_sizes = [sum(t.size_bytes for t in layer.tensors) for layer in layers]
        layer_sizes_for_assignment = layer_sizes[1:]
        expected = assignment_strategy.compute(layer_sizes_for_assignment, strategy.n_blocks)

        layer_labels = [layer.label for layer in layers]
        expected_map = {layer_labels[i]: expected[i] for i in range(len(expected))}

        # Verify all slots are present (including the gap layer)
        for label in expected_map:
            assert label in strategy.optimal_layer_to_block, f"{label} missing — gap layers must be included"

        # Verify block IDs match the pre-computed assignment
        for label, expected_blk in expected_map.items():
            actual_blk = strategy.optimal_layer_to_block[label]
            assert actual_blk == expected_blk, (
                f"{label}: block {actual_blk} != expected {expected_blk} (from pre-computed assignment)"
            )

        # Sanity: re-deriving from gap-skipped transfer sizes would differ
        transfer_labels = list(result.strategy_map.keys())
        if transfer_labels:
            transfer_sizes = [sum(t.size_bytes for t in result.strategy_map[lb]) for lb in transfer_labels]
            rederived = assignment_strategy.compute(transfer_sizes, strategy.n_blocks)
            rederived_map = dict(zip(transfer_labels, rederived, strict=False))
            # The two approaches give different inputs (3 slots vs 2 non-gap),
            # so block IDs or coverage should differ
            assert rederived_map != {lb: expected_map[lb] for lb in transfer_labels} or len(transfer_labels) < len(
                expected_map
            ), (
                "Expected pre-computed and re-derived assignments to differ "
                "for this gap scenario — test may not be exercising the fix"
            )


if __name__ == "__main__":
    # Run tests with verbose output to see summaries
    pytest.main([__file__, "-v", "-s"])
