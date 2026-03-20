# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GlobalOffloadStrategy with pipelined block optimization."""

import warnings

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy import (
    GlobalOffloadStrategy,
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


def print_summary(
    strategy: GlobalOffloadStrategy,
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
    print(f"  Blocks: {strategy.n_blocks}")
    if max_gpu_mem_bytes is not None:
        print(f"  Memory budget: {max_gpu_mem_bytes / 1024 / 1024:.0f} MB")

    print("\n  Layer details:")
    for layer in layers:
        total_size = sum(t.size_bytes for t in layer.tensors) / 1024 / 1024
        print(f"    {layer.label}: compute={layer.duration}ms, tensors={total_size:.0f}MB")

    # Results summary
    print("\nRESULTS:")
    print("  Block Sizes:")
    for i, size in enumerate(strategy.optimal_block_sizes):
        print(f"    Block {i}: {size / 1024 / 1024:.2f} MB")

    print("\n  Layer Assignments:")
    for label, block_idx in strategy.optimal_layer_to_block.items():
        block_size = strategy.optimal_block_sizes[block_idx]
        print(f"    {label} -> Block {block_idx} ({block_size / 1024 / 1024:.0f}MB)")

    print("\n  Memory Summary:")
    total_blocks = sum(strategy.optimal_block_sizes)
    print(f"    Total blocks: {total_blocks / 1024 / 1024:.2f} MB")
    print(f"    Non-offloaded: {strategy.optimal_non_offloaded_memory / 1024 / 1024:.4f} MB")
    print(f"    Peak memory: {strategy.optimal_peak_memory / 1024 / 1024:.2f} MB")
    if max_gpu_mem_bytes is not None:
        print(f"    Budget: {max_gpu_mem_bytes / 1024 / 1024:.0f} MB")
        print(f"    Within budget: {strategy.optimal_peak_memory <= max_gpu_mem_bytes}")

    # Pipeline check
    print("\n  Pipeline Constraint:")
    assignments = list(strategy.optimal_layer_to_block.values())
    violations = 0
    for i in range(1, len(assignments)):
        prev, curr = assignments[i - 1], assignments[i]
        status = "OK" if prev != curr else "VIOLATION"
        if prev == curr:
            violations += 1
        print(f"    layer_{i - 1}(B{prev}) -> layer_{i}(B{curr}): {status}")
    print(f"    Total violations: {violations}")

    print(f"{'=' * 70}\n")


class TestGlobalOffloadStrategyBasic:
    """Basic tests for GlobalOffloadStrategy initialization and validation."""

    def test_minimum_blocks_required(self):
        """Test that n_blocks < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_blocks must be at least 2"):
            GlobalOffloadStrategy(
                n_blocks=1,
            )

    def test_initialization_with_valid_params(self):
        """Test successful initialization with valid parameters."""
        strategy = GlobalOffloadStrategy(
            n_blocks=3,
            threshold_mb=0.5,
        )
        assert strategy.n_blocks == 3
        assert strategy.threshold_mb == 0.5


class TestGlobalOffloadStrategyFeasible:
    """Tests with feasible optimization scenarios."""

    def test_simple_4_layers_2_blocks(self):
        """Test simple case: 4 layers with 2 blocks, all transfers feasible."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 25, 2.5)], 6.0),
            create_layer("layer_2", [create_tensor(3, 35, 3.5)], 5.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 4.0),
        ]

        budget = 100 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            threshold_mb=0.1,
            min_blocks=2,  # Explicitly test 2-block scenario
        )

        result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Simple 4 Layers with 2 Blocks", max_gpu_mem_bytes=budget)

        # Verify results
        # Pipelined: transfers happen during previous layer's compute, so N-1 entries for N layers
        assert len(result.strategy_map) == 3, "Should have N-1 transfers for N layers (pipelined)"
        assert strategy.optimal_peak_memory <= budget, "Should be within budget"

        # Verify pipeline constraint (no consecutive same blocks)
        assignments = list(strategy.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"

        # Verify all transfers fit in their assigned block.
        # Block assignment[i] holds the NEXT layer's data (transferred during
        # layer i's execution), so check strategy_map transfer sizes.
        for label, tensors in result.strategy_map.items():
            block_idx = strategy.optimal_layer_to_block[label]
            transfer_size = sum(t.size_bytes for t in tensors)
            block_size = strategy.optimal_block_sizes[block_idx]
            assert transfer_size <= block_size, f"Transfer overflow at {label}"

    def test_6_layers_3_blocks(self):
        """Test with 6 layers and 3 blocks for more flexibility."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 20, 2.0)], 4.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 25, 2.5)], 6.0),
            create_layer("layer_3", [create_tensor(4, 35, 3.5)], 5.0),
            create_layer("layer_4", [create_tensor(5, 20, 2.0)], 4.0),
            create_layer("layer_5", [create_tensor(6, 30, 3.0)], 5.0),
        ]

        budget = 150 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=3,
            threshold_mb=0.1,
        )

        result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "6 Layers with 3 Blocks", max_gpu_mem_bytes=budget)

        # Verify results
        assert len(result.strategy_map) == 5  # N-1 for pipelined transfers
        assert strategy.optimal_peak_memory <= budget

        # Verify pipeline constraint
        assignments = list(strategy.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1]

    def test_8_layers_4_blocks(self):
        """Test with 8 layers and 4 blocks for maximum flexibility."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 25, 2.5)], 4.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 20, 2.0)], 6.0),
            create_layer("layer_3", [create_tensor(4, 40, 4.0)], 5.0),
            create_layer("layer_4", [create_tensor(5, 35, 3.5)], 4.0),
            create_layer("layer_5", [create_tensor(6, 25, 2.5)], 5.0),
            create_layer("layer_6", [create_tensor(7, 30, 3.0)], 6.0),
            create_layer("layer_7", [create_tensor(8, 20, 2.0)], 4.0),
        ]

        # Use OptimizedRoundRobinAssignment for reliable pipeline constraint satisfaction
        budget = 200 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=4),
        )

        result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "8 Layers with 4 Blocks", max_gpu_mem_bytes=budget)

        # Verify results
        assert len(result.strategy_map) == 7  # N-1 for pipelined transfers
        assert strategy.optimal_peak_memory <= budget

        # Verify pipeline constraint (no consecutive same blocks)
        assignments = list(strategy.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"

        # Verify all 4 blocks are potentially used
        unique_blocks = set(assignments)
        print(f"  Unique blocks used: {len(unique_blocks)} out of 4")

    def test_layers_with_multiple_tensors(self):
        """Test layers containing multiple tensors."""
        layers = [
            create_layer(
                "layer_0",
                [
                    create_tensor(1, 15, 1.5),
                    create_tensor(2, 10, 1.0),
                ],
                5.0,
            ),
            create_layer(
                "layer_1",
                [
                    create_tensor(3, 20, 2.0),
                    create_tensor(4, 15, 1.5),
                ],
                6.0,
            ),
            create_layer(
                "layer_2",
                [
                    create_tensor(5, 25, 2.5),
                ],
                5.0,
            ),
            create_layer(
                "layer_3",
                [
                    create_tensor(6, 10, 1.0),
                    create_tensor(7, 10, 1.0),
                    create_tensor(8, 10, 1.0),
                ],
                4.0,
            ),
        ]

        budget = 100 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            threshold_mb=0.1,
            min_blocks=2,  # Explicitly test 2-block scenario
        )

        result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Layers with Multiple Tensors", max_gpu_mem_bytes=budget)

        assert len(result.strategy_map) == 3  # N-1 for pipelined transfers
        assert strategy.optimal_peak_memory <= budget

    def test_with_non_offloaded_tensors(self):
        """Test with small tensors that stay on GPU (below threshold)."""
        layers = [
            create_layer(
                "layer_0",
                [
                    create_tensor(1, 30, 3.0),  # Offloaded
                    create_tensor(2, 0.05, 0.01),  # NOT offloaded (< 0.1 MB)
                ],
                5.0,
            ),
            create_layer(
                "layer_1",
                [
                    create_tensor(3, 25, 2.5),
                    create_tensor(4, 0.08, 0.01),  # NOT offloaded
                ],
                6.0,
            ),
            create_layer(
                "layer_2",
                [
                    create_tensor(5, 35, 3.5),
                    create_tensor(6, 0.06, 0.01),  # NOT offloaded
                ],
                5.0,
            ),
            create_layer(
                "layer_3",
                [
                    create_tensor(7, 30, 3.0),
                    create_tensor(8, 0.07, 0.01),  # NOT offloaded
                ],
                4.0,
            ),
        ]

        budget = 150 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            threshold_mb=0.1,
            min_blocks=2,  # Explicitly test 2-block scenario
        )

        _result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Layers with Non-Offloaded Tensors", max_gpu_mem_bytes=budget)

        # Verify non-offloaded memory is accounted for
        assert strategy.optimal_non_offloaded_memory > 0, "Should have non-offloaded memory"
        assert strategy.optimal_peak_memory <= budget

        # Peak memory should include both blocks and non-offloaded
        total_blocks = sum(strategy.optimal_block_sizes)
        expected_peak = total_blocks + strategy.optimal_non_offloaded_memory
        assert abs(strategy.optimal_peak_memory - expected_peak) < 1, "Peak should equal blocks + non-offloaded"


class TestGlobalOffloadStrategyConstraints:
    """Tests for constraint handling and edge cases."""

    def test_tight_memory_budget(self):
        """Test with tight memory budget that forces optimal block sizing.

        The first layer is always on GPU (no previous layer for pipeline
        transfer), so with 4x30MB layers and 2 blocks, the minimum peak is
        60MB (blocks) + 30MB (layer_0) = 90MB.  A 100MB budget is feasible.
        """
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 5.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 5.0),
        ]

        budget = 100 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            threshold_mb=0.1,
            min_blocks=2,
        )

        _result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Tight Memory Budget", max_gpu_mem_bytes=budget)

        assert strategy.optimal_peak_memory <= budget

    def test_empty_layers_returns_empty(self):
        """Test that empty layer list returns empty strategy."""
        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            min_blocks=2,  # Explicitly test 2-block scenario
        )

        result = strategy.compute([], max_gpu_mem_bytes=100 * 1024 * 1024)
        assert result.strategy_map == {}

    def test_first_layer_never_offloaded(self):
        """First layer is always on GPU — no previous layer to pipeline from."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 5.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 5.0),
            create_layer("layer_4", [create_tensor(5, 30, 3.0)], 5.0),
        ]

        strategy = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
        )
        result = strategy.compute(layers, max_gpu_mem_bytes=200 * 1024 * 1024)

        first_layer_ids = {t.tensor_id for t in layers[0].tensors}
        all_offloaded_ids = {t.tensor_id for tensors in result.strategy_map.values() for t in tensors}
        assert first_layer_ids.isdisjoint(all_offloaded_ids), "First layer tensors must not be offloaded"

    def test_layers_below_threshold(self):
        """Test layers with all tensors below threshold."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 0.05, 0.01)], 5.0),
            create_layer("layer_1", [create_tensor(2, 0.08, 0.01)], 5.0),
        ]

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            threshold_mb=0.1,  # All tensors below this
            min_blocks=2,  # Explicitly test 2-block scenario
        )

        result = strategy.compute(layers, max_gpu_mem_bytes=100 * 1024 * 1024)
        assert result.strategy_map == {}, "Should return empty when all tensors below threshold"


class TestGlobalOffloadStrategyBlockRange:
    """Tests for min_blocks/max_blocks options."""

    def test_8_layers_4_blocks_force_all(self):
        """Test forcing optimizer to use all 4 blocks via min_blocks=max_blocks=n_blocks."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 25, 2.5)], 4.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 20, 2.0)], 6.0),
            create_layer("layer_3", [create_tensor(4, 40, 4.0)], 5.0),
            create_layer("layer_4", [create_tensor(5, 35, 3.5)], 4.0),
            create_layer("layer_5", [create_tensor(6, 25, 2.5)], 5.0),
            create_layer("layer_6", [create_tensor(7, 30, 3.0)], 6.0),
            create_layer("layer_7", [create_tensor(8, 20, 2.0)], 4.0),
        ]

        budget = 200 * 1024 * 1024
        # Default: optimizer chooses optimal block count (min_blocks=2, max_blocks=4)
        strategy_auto = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=4),
        )
        _result_auto = strategy_auto.compute(layers, max_gpu_mem_bytes=budget)
        blocks_used_auto = len(set(strategy_auto.optimal_layer_to_block.values()))

        print_summary(strategy_auto, layers, "8 Layers 4 Blocks - AUTO (find optimal)", max_gpu_mem_bytes=budget)
        print(f"  Blocks used (auto): {blocks_used_auto} out of 4")

        # Force all 4 blocks: min_blocks=max_blocks=n_blocks
        strategy_force = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=4, max_blocks=4),
        )
        _result_force = strategy_force.compute(layers, max_gpu_mem_bytes=budget)
        blocks_used_force = len(set(strategy_force.optimal_layer_to_block.values()))

        print_summary(strategy_force, layers, "8 Layers 4 Blocks - FORCED (use all)", max_gpu_mem_bytes=budget)
        print(f"  Blocks used (forced): {blocks_used_force} out of 4")

        # Verify constraints
        assert strategy_auto.optimal_peak_memory <= budget
        assert strategy_force.optimal_peak_memory <= budget

        # When forced, should use all 4 blocks (if enough layers)
        assert blocks_used_force == 4, f"Expected 4 blocks used, got {blocks_used_force}"

        # Auto mode may use fewer blocks
        print("\n  Comparison:")
        print(
            f"    Auto mode:   {blocks_used_auto} blocks, peak={strategy_auto.optimal_peak_memory / 1024 / 1024:.0f}MB"
        )
        print(
            f"    Forced mode: {blocks_used_force} blocks, "
            f"peak={strategy_force.optimal_peak_memory / 1024 / 1024:.0f}MB"
        )

    def test_min_blocks_not_enough_layers(self):
        """Test that min_blocks is adjusted when not enough layers."""
        # Only 3 layers but 4 blocks requested - can't use all blocks
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 25, 2.5)], 5.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 5.0),
        ]

        budget = 200 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            min_blocks=4,  # Request all 4, but only 3 layers available
            max_blocks=4,
        )

        _result = strategy.compute(layers, max_gpu_mem_bytes=budget)
        blocks_used = len(set(strategy.optimal_layer_to_block.values()))

        print_summary(strategy, layers, "3 Layers 4 Blocks - MIN=4 (not enough layers)", max_gpu_mem_bytes=budget)
        print(f"  Blocks used: {blocks_used} (max possible with 3 layers)")

        # With only 3 layers, can use at most 3 blocks (due to pipeline constraint)
        assert blocks_used <= 3, "Cannot use more blocks than layers"
        assert strategy.optimal_peak_memory <= budget

    def test_round_robin_mode(self):
        """Test round-robin block assignment mode with min_blocks=max_blocks=n_blocks."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 30, 3.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 5.0),
            create_layer("layer_3", [create_tensor(4, 30, 3.0)], 5.0),
            create_layer("layer_4", [create_tensor(5, 30, 3.0)], 5.0),
            create_layer("layer_5", [create_tensor(6, 30, 3.0)], 5.0),
            create_layer("layer_6", [create_tensor(7, 30, 3.0)], 5.0),
            create_layer("layer_7", [create_tensor(8, 30, 3.0)], 5.0),
        ]

        budget = 200 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(
                min_blocks=4,  # Force using all 4 blocks
                max_blocks=4,
                distribution_weight=1.0,  # Prefer cyclic pattern
            ),
        )

        _result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(
            strategy, layers, "8 Layers 4 Blocks - ROUND ROBIN (forced, dist_weight=1.0)", max_gpu_mem_bytes=budget
        )

        # Verify all 4 blocks are used
        assignments = list(strategy.optimal_layer_to_block.values())
        assert len(set(assignments)) == 4, "Should use all 4 blocks"

        # Verify pipeline constraint (no consecutive same blocks)
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"

        # Verify reasonable reuse distances (at least averaging close to 4)
        last_use: dict[int, int] = {}
        distances: list[int] = []
        for i, b in enumerate(assignments):
            if b in last_use:
                distances.append(i - last_use[b])
            last_use[b] = i

        if distances:
            avg_distance = sum(distances) / len(distances)
            print(f"  Reuse distances: {distances}")
            print(f"  Avg reuse distance: {avg_distance:.2f} (ideal=4)")
            # With distribution_weight=1.0, should be close to ideal
            assert avg_distance >= 3.0, f"Average reuse distance {avg_distance} should be >= 3"

        print(f"  Pattern: {' -> '.join([f'B{b}' for b in assignments])}")

    def test_round_robin_variable_sizes(self):
        """Test round-robin with variable layer sizes uses per-block sizing."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),  # 30 MB
            create_layer("layer_1", [create_tensor(2, 60, 6.0)], 5.0),  # 60 MB
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 5.0),  # 30 MB
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 5.0),  # 50 MB
            create_layer("layer_4", [create_tensor(5, 40, 4.0)], 5.0),  # 40 MB
            create_layer("layer_5", [create_tensor(6, 25, 2.5)], 5.0),  # 25 MB
            create_layer("layer_6", [create_tensor(7, 35, 3.5)], 5.0),  # 35 MB
            create_layer("layer_7", [create_tensor(8, 45, 4.5)], 5.0),  # 45 MB
        ]

        budget = 300 * 1024 * 1024
        # Test MEMORY OPTIMIZED mode (distribution_weight=0.0)
        strategy_mem = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(
                distribution_weight=0.0,  # Minimize memory only
            ),
        )
        strategy_mem.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(
            strategy_mem, layers, "Variable Sizes - MEMORY OPTIMIZED (dist_weight=0.0)", max_gpu_mem_bytes=budget
        )

        blocks_used_mem = len([s for s in strategy_mem.optimal_block_sizes if s > 0])
        total_mem_mem = sum(strategy_mem.optimal_block_sizes) / 1024 / 1024

        print(f"  Blocks used: {blocks_used_mem}")
        print(f"  Total memory: {total_mem_mem:.0f} MB")

        # Memory-optimized should use fewer blocks (more efficient)
        assert blocks_used_mem <= 3, f"Memory-optimized should use <=3 blocks, got {blocks_used_mem}"
        assert total_mem_mem <= 120, f"Expected <=120MB, got {total_mem_mem:.0f}MB"

        # Test DISTRIBUTION OPTIMIZED mode (distribution_weight=1.0, force 4 blocks)
        strategy_dist = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(
                min_blocks=4,  # Force using all 4 blocks
                max_blocks=4,
                distribution_weight=1.0,  # Prefer cyclic distribution
            ),
        )
        strategy_dist.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(
            strategy_dist, layers, "Variable Sizes - DISTRIBUTION OPTIMIZED (dist_weight=1.0)", max_gpu_mem_bytes=budget
        )

        blocks_used_dist = len([s for s in strategy_dist.optimal_block_sizes if s > 0])
        total_mem_dist = sum(strategy_dist.optimal_block_sizes) / 1024 / 1024

        print(f"  Blocks used: {blocks_used_dist}")
        print(f"  Total memory: {total_mem_dist:.0f} MB")

        assert blocks_used_dist == 4, "Distribution mode with min=max=4 should use all 4 blocks"

        # Verify pipeline constraint
        assignments = list(strategy_dist.optimal_layer_to_block.values())
        for i in range(1, len(assignments)):
            assert assignments[i] != assignments[i - 1], f"Pipeline violation at layer {i}"

        # Compare memory usage
        print()
        print("  Memory comparison:")
        print(f"    Memory-optimized: {total_mem_mem:.0f} MB ({blocks_used_mem} blocks)")
        print(f"    Distribution-optimized: {total_mem_dist:.0f} MB ({blocks_used_dist} blocks)")
        print(f"    Memory difference: {total_mem_dist - total_mem_mem:.0f} MB")

    def test_block_range_min_3_max_4(self):
        """Test min_blocks=3, max_blocks=4 range optimization."""
        layers = [
            create_layer("layer_0", [create_tensor(1, 30, 3.0)], 5.0),
            create_layer("layer_1", [create_tensor(2, 60, 6.0)], 5.0),
            create_layer("layer_2", [create_tensor(3, 30, 3.0)], 5.0),
            create_layer("layer_3", [create_tensor(4, 50, 5.0)], 5.0),
            create_layer("layer_4", [create_tensor(5, 40, 4.0)], 5.0),
            create_layer("layer_5", [create_tensor(6, 25, 2.5)], 5.0),
        ]

        # Test with range [3, 4]
        budget = 300 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=4,
            threshold_mb=0.1,
            assignment_strategy=OptimizedRoundRobinAssignment(
                min_blocks=3,  # At least 3 blocks
                max_blocks=4,
            ),  # At most 4 blocks
        )
        strategy.compute(layers, max_gpu_mem_bytes=budget)

        blocks_used = len([s for s in strategy.optimal_block_sizes if s > 0])
        total_mem = sum(strategy.optimal_block_sizes) / 1024 / 1024

        print_summary(strategy, layers, "6 Layers - min_blocks=3, max_blocks=4", max_gpu_mem_bytes=budget)
        print(f"  Blocks used: {blocks_used}")
        print(f"  Total memory: {total_mem:.0f} MB")

        # Should use either 3 or 4 blocks (within range)
        assert 3 <= blocks_used <= 4, f"Expected 3-4 blocks, got {blocks_used}"
        assert strategy.optimal_peak_memory <= budget

    def test_validation_min_greater_than_max(self):
        """Test that min_blocks > max_blocks raises ValueError."""
        with pytest.raises(ValueError, match=r"min_blocks.*cannot exceed.*max_blocks"):
            GlobalOffloadStrategy(
                n_blocks=4,
                min_blocks=4,
                max_blocks=2,  # Invalid: max < min
            )

    def test_validation_min_blocks_less_than_2(self):
        """Test that min_blocks < 2 raises ValueError."""
        with pytest.raises(ValueError, match="min_blocks must be at least 2"):
            GlobalOffloadStrategy(
                n_blocks=4,
                min_blocks=1,  # Invalid: < 2
            )

    def test_validation_max_blocks_exceeds_n_blocks(self):
        """Test that max_blocks > n_blocks raises ValueError."""
        with pytest.raises(ValueError, match=r"max_blocks.*cannot exceed.*n_blocks"):
            GlobalOffloadStrategy(
                n_blocks=4,
                max_blocks=5,  # Invalid: > n_blocks
            )


class TestGlobalOffloadStrategyTransferCapacity:
    """Tests for transfer capacity constraints."""

    def test_transfer_within_capacity(self):
        """Test that blocks are sized to fit within transfer capacity."""
        # Design layers where transfer capacity limits block size
        layers = [
            create_layer("layer_0", [create_tensor(1, 20, 2.0)], 3.0),  # 3ms compute
            create_layer("layer_1", [create_tensor(2, 20, 2.0)], 3.0),  # Can transfer ~30MB
            create_layer("layer_2", [create_tensor(3, 20, 2.0)], 3.0),
            create_layer("layer_3", [create_tensor(4, 20, 2.0)], 3.0),
        ]

        budget = 100 * 1024 * 1024
        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            threshold_mb=0.1,
            min_blocks=2,  # Explicitly test 2-block scenario
        )

        result = strategy.compute(layers, max_gpu_mem_bytes=budget)

        print_summary(strategy, layers, "Transfer Within Capacity", max_gpu_mem_bytes=budget)

        # Verify all constraints satisfied
        assert strategy.optimal_peak_memory <= budget
        assert len(result.strategy_map) == 3  # N-1 for pipelined transfers


class TestGlobalOffloadStrategyMemoryConstraint:
    """Tests for memory constraint behavior."""

    def test_memory_constraint_warning(self):
        """Test that memory constraint violation issues a warning."""
        import warnings

        # Large blocks that exceed memory budget
        layers = [
            create_layer("layer_0", [create_tensor(1, 90, 9.0)], 10.0),
            create_layer("layer_1", [create_tensor(2, 90, 9.0)], 10.0),
            create_layer("layer_2", [create_tensor(3, 90, 9.0)], 10.0),
            create_layer("layer_3", [create_tensor(4, 90, 9.0)], 10.0),
        ]

        budget = 100 * 1024 * 1024  # Budget too small (need ~180MB)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            strategy = GlobalOffloadStrategy(
                n_blocks=2,
                threshold_mb=0.1,
                min_blocks=2,  # Explicitly test 2-block scenario
                assignment_strategy=StrictRoundRobinAssignment(),
            )
            strategy.compute(layers, max_gpu_mem_bytes=budget)

            memory_warnings = [x for x in w if "memory constraint" in str(x.message).lower()]

        print_summary(strategy, layers, "Memory Constraint Warning", max_gpu_mem_bytes=budget)
        print(f"  Peak memory: {strategy.optimal_peak_memory / 1024 / 1024:.0f} MB")
        print(f"  Budget: {budget / 1024 / 1024:.0f} MB")
        print(f"  Memory warnings: {len(memory_warnings)}")

        assert strategy.optimal_peak_memory > budget, "Memory should exceed budget"
        assert len(memory_warnings) == 1, "Should have memory constraint warning"
        assert "tensor sizes" in str(memory_warnings[0].message).lower()


class TestAdjustScaleForMemory:
    """Tests for the _adjust_scale_for_memory binary search.

    This method is private but contains the core memory-fitting logic:
    a binary search (up to 40 iterations) that increases scale until
    peak GPU memory fits within the target budget.
    """

    @staticmethod
    def _make_constrained_layers() -> list[LayerStatistics]:
        """Create layers where scale=1.0 doesn't offload all large tensors.

        Alternates 15MB and 40MB tensors with 2ms compute time.  Two
        distinct sizes are needed so the MemoryTransferInterpolator can
        interpolate (a single size causes it to return a constant).

        At scale=1.0 the 2ms transfer window allows ~20MB, so 40MB
        tensors on layers 1+ can't be offloaded.  At scale>=2 the
        window opens to ~40MB and all tensors fit.
        """
        mib = 1024**2
        layers = []
        for i in range(8):
            if i % 2 == 0:
                t = TensorStatistics(tensor_id=i, name=f"t{i}", size_bytes=15 * mib, load_time_ms=1.5)
            else:
                t = TensorStatistics(tensor_id=i, name=f"t{i}", size_bytes=40 * mib, load_time_ms=4.0)
            layers.append(LayerStatistics(label=f"layer_{i}", tensors=[t], duration=2.0))
        return layers

    def test_scale_increases_when_initial_exceeds_target(self):
        """Binary search should raise scale so peak fits within max_gpu_mem_bytes."""
        layers = self._make_constrained_layers()
        target = 100 * 1024 * 1024

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            scale=1.0,
            threshold_mb=0.1,
        )
        strategy.compute(layers, max_gpu_mem_bytes=target)

        assert strategy.scale > 1.0, "Scale should have been increased by the binary search"
        assert strategy.optimal_peak_memory <= target, "Peak should be within target"

    def test_scale_unchanged_when_already_within_target(self):
        """When initial result is already within target, scale should not change."""
        layers = [
            LayerStatistics(
                label=f"layer_{i}",
                tensors=[TensorStatistics(tensor_id=i, name=f"t{i}", size_bytes=10 * 1024**2, load_time_ms=1.0)],
                duration=10.0,
            )
            for i in range(4)
        ]

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            scale=1.0,
            threshold_mb=0.1,
        )
        strategy.compute(layers, max_gpu_mem_bytes=500 * 1024**2)

        assert strategy.scale == pytest.approx(1.0), "Scale should remain 1.0 when target is already met"

    def test_warns_when_scale_adjusted(self):
        """A UserWarning should be emitted when scale is increased."""
        layers = self._make_constrained_layers()
        target = 100 * 1024 * 1024

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            scale=1.0,
            threshold_mb=0.1,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.compute(layers, max_gpu_mem_bytes=target)
            scale_warnings = [x for x in w if "insufficient to meet GPU memory target" in str(x.message)]

        assert len(scale_warnings) >= 1

    def test_warns_when_constraint_unsatisfiable(self):
        """When target is impossibly small, a 'Cannot satisfy' warning should be emitted."""
        layers = self._make_constrained_layers()

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            scale=1.0,
            threshold_mb=0.1,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.compute(layers, max_gpu_mem_bytes=1)
            constraint_warnings = [x for x in w if "Cannot satisfy GPU memory constraint" in str(x.message)]

        assert len(constraint_warnings) >= 1

    def test_scale_not_mutated_during_search(self):
        """self.scale should reflect the original value until compute() returns."""
        layers = self._make_constrained_layers()

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            scale=1.0,
            threshold_mb=0.1,
        )

        original_collect = strategy._collect_layer_tensors
        observed_self_scale: list[float] = []

        def spy_collect(*args, **kwargs):
            observed_self_scale.append(strategy.scale)
            return original_collect(*args, **kwargs)

        strategy._collect_layer_tensors = spy_collect  # type: ignore[method-assign]
        strategy.compute(layers, max_gpu_mem_bytes=500 * 1024 * 1024)

        pre_final = observed_self_scale[:-1]
        if pre_final:
            assert all(s == pytest.approx(1.0) for s in pre_final), f"self.scale was mutated mid-search: {pre_final}"

    def test_idempotent_compute(self):
        """Calling compute() twice should produce the same peak and strategy keys."""
        layers = self._make_constrained_layers()

        strategy = GlobalOffloadStrategy(
            n_blocks=2,
            scale=1.0,
            threshold_mb=0.1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result1 = strategy.compute(layers, max_gpu_mem_bytes=500 * 1024 * 1024)
            peak1 = strategy.optimal_peak_memory
            keys1 = set(result1.strategy_map.keys())

            result2 = strategy.compute(layers, max_gpu_mem_bytes=500 * 1024 * 1024)
            peak2 = strategy.optimal_peak_memory
            keys2 = set(result2.strategy_map.keys())

        assert peak1 == peak2
        assert keys1 == keys2


if __name__ == "__main__":
    # Run tests with verbose output to see summaries
    pytest.main([__file__, "-v", "-s"])
