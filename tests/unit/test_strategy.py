# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for strategy module functions."""

import warnings

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics

# Import private functions directly from their modules for testing internals
from flextensor.loaders import (
    _compute_peak_memory_from_strategy as compute_peak_memory_from_strategy,
)
from flextensor.memory_transfer_benchmark import extract_memory_transfers_from_layer_stats
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator
from flextensor.strategy import (
    AdaptiveKnapsackStrategy,
    GreedyStrategy,
    KnapsackBlockStrategy,
    KnapsackStrategy,
)
from flextensor.strategy.knapsack import (
    _compute_optimal_scale as compute_optimal_scale,
)
from flextensor.strategy.knapsack import (
    _compute_solution as compute_solution,
)
from flextensor.strategy.knapsack import (
    _estimate_required_scale as estimate_required_scale,
)
from flextensor.strategy.knapsack import (
    _prepare_merged_layers as prepare_merged_layers,
)


def create_tensor_stats(tensor_id: int, size_bytes: int, load_time_ms: float) -> TensorStatistics:
    """Helper to create TensorStatistics."""
    return TensorStatistics(
        tensor_id=tensor_id,
        name=f"tensor_{tensor_id}",
        size_bytes=size_bytes,
        load_time_ms=load_time_ms,
    )


def create_layer_stats(label: str, tensors: list[TensorStatistics], duration: float) -> LayerStatistics:
    """Helper to create LayerStatistics."""
    return LayerStatistics(label=label, tensors=tensors, duration=duration)


class TestExtractMemoryTransfersFromLayerStats:
    """Test cases for extract_memory_transfers_from_layer_stats function."""

    def test_extract_single_tensor(self):
        """Test extraction with single tensor."""
        tensor = create_tensor_stats(1, 1024 * 1024, 1.0)  # 1MB, 1ms
        layer = create_layer_stats("layer_0", [tensor], 10.0)

        result = extract_memory_transfers_from_layer_stats([layer])

        assert len(result) == 1
        assert result[1024 * 1024] == 1.0

    def test_extract_multiple_tensors_same_size(self):
        """Test that tensors with same size get averaged."""
        tensor1 = create_tensor_stats(1, 1024 * 1024, 1.0)
        tensor2 = create_tensor_stats(2, 1024 * 1024, 3.0)
        layer = create_layer_stats("layer_0", [tensor1, tensor2], 10.0)

        result = extract_memory_transfers_from_layer_stats([layer])

        assert len(result) == 1
        assert result[1024 * 1024] == 2.0  # Average of 1.0 and 3.0

    def test_extract_multiple_tensors_different_sizes(self):
        """Test extraction with different tensor sizes."""
        tensor1 = create_tensor_stats(1, 1024, 0.1)
        tensor2 = create_tensor_stats(2, 1024 * 1024, 1.0)
        layer = create_layer_stats("layer_0", [tensor1, tensor2], 10.0)

        result = extract_memory_transfers_from_layer_stats([layer])

        assert len(result) == 2
        assert result[1024] == 0.1
        assert result[1024 * 1024] == 1.0

    def test_extract_skips_zero_load_time(self):
        """Test that tensors with zero load time are skipped."""
        tensor1 = create_tensor_stats(1, 1024, 0.0)  # Zero load time
        tensor2 = create_tensor_stats(2, 1024 * 1024, 1.0)
        layer = create_layer_stats("layer_0", [tensor1, tensor2], 10.0)

        result = extract_memory_transfers_from_layer_stats([layer])

        assert len(result) == 1
        assert 1024 not in result
        assert result[1024 * 1024] == 1.0

    def test_extract_empty_layer_stats(self):
        """Test extraction with empty layer stats."""
        result = extract_memory_transfers_from_layer_stats([])
        assert len(result) == 0

    def test_extract_multiple_layers(self):
        """Test extraction across multiple layers."""
        tensor1 = create_tensor_stats(1, 1024, 0.1)
        tensor2 = create_tensor_stats(2, 1024 * 1024, 1.0)
        layer1 = create_layer_stats("layer_0", [tensor1], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor2], 20.0)

        result = extract_memory_transfers_from_layer_stats([layer1, layer2])

        assert len(result) == 2
        assert result[1024] == 0.1
        assert result[1024 * 1024] == 1.0


class TestEstimateRequiredScale:
    """Test cases for estimate_required_scale function."""

    @pytest.fixture
    def memory_transfers(self):
        """Create memory transfer data for tests."""
        return {
            1024: 0.001,  # 1KB -> 0.001ms
            1024 * 1024: 0.1,  # 1MB -> 0.1ms
            1024 * 1024 * 1024: 100.0,  # 1GB -> 100ms
        }

    @pytest.fixture
    def interpolator(self, memory_transfers):
        """Create interpolator for tests."""
        return MemoryTransferInterpolator(memory_transfers)

    def test_estimate_normal_case(self, interpolator):
        """Test estimation with normal inputs."""
        tensor = create_tensor_stats(1, 100 * 1024 * 1024, 10.0)  # 100MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)

        # Need to offload 100MB
        scale = estimate_required_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=float(100 * 1024 * 1024),
        )

        # Should return a reasonable scale >= 10.0 (minimum)
        assert scale >= 10.0

    def test_estimate_zero_duration_returns_fallback(self, interpolator):
        """Test that zero total duration returns fallback of 100.0."""
        tensor = create_tensor_stats(1, 100 * 1024 * 1024, 10.0)
        layer1 = create_layer_stats("layer_0", [tensor], 0.0)  # Zero duration
        layer2 = create_layer_stats("layer_1", [tensor], 0.0)  # Zero duration

        scale = estimate_required_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=float(100 * 1024 * 1024),
        )

        assert scale == 100.0

    def test_estimate_single_layer_uses_fallback_duration(self, interpolator):
        """Test that single layer uses fallback duration of 1.0."""
        tensor = create_tensor_stats(1, 100 * 1024 * 1024, 10.0)
        layer = create_layer_stats("layer_0", [tensor], 10.0)

        scale = estimate_required_scale(
            layer_stats=[layer],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=float(100 * 1024 * 1024),
        )

        # With only one layer, total_duration = 1.0 (fallback)
        # Should still return a valid scale
        assert scale >= 10.0

    def test_estimate_applies_safety_margin(self, interpolator):
        """Test that 50% safety margin is applied (result clamped to minimum)."""
        tensor = create_tensor_stats(1, 100 * 1024 * 1024, 10.0)
        layer1 = create_layer_stats("layer_0", [tensor], 100.0)  # 100ms
        layer2 = create_layer_stats("layer_1", [tensor], 100.0)

        # With 100ms duration at scale=1.0, interpolator gives capacity
        capacity_at_scale_1 = interpolator.duration_to_bytes(100.0)
        needed_bytes = float(capacity_at_scale_1)  # Exactly match capacity

        scale = estimate_required_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=needed_bytes,
        )

        # Without margin: scale = 1.0
        # With 50% margin: scale = 1.5
        # But minimum is 10.0, so should be max(1.5, 10.0) = 10.0
        assert scale >= 10.0

    def test_estimate_safety_margin_above_minimum(self, interpolator):
        """Test that 50% safety margin is actually applied when result > 10.0."""
        tensor = create_tensor_stats(1, 1024 * 1024 * 1024, 100.0)  # 1GB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)  # Short duration
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)

        # Calculate capacity at scale=1.0 with 10ms duration
        capacity_at_scale_1 = interpolator.duration_to_bytes(10.0)

        # Need 20x the capacity -> base scale would be 20
        # With 50% margin -> scale should be 30
        needed_bytes = float(capacity_at_scale_1 * 20)

        scale = estimate_required_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=needed_bytes,
        )

        # Expected scale = 30 ( 20 * 1.5 )
        expected_scale = 20 * 1.5
        assert abs(scale - expected_scale) < 0.1, f"Expected ~{expected_scale}, got {scale}"

    def test_estimate_enforces_minimum_of_10(self, interpolator):
        """Test that minimum of 10.0 is enforced."""
        tensor = create_tensor_stats(1, 1024, 0.001)  # Very small tensor
        layer1 = create_layer_stats("layer_0", [tensor], 1000.0)  # Long duration
        layer2 = create_layer_stats("layer_1", [tensor], 1000.0)

        # Very small offload need with long duration -> would give tiny scale
        scale = estimate_required_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=1.0,  # Just 1 byte
        )

        assert scale == 10.0  # Should be clamped to minimum

    def test_estimate_large_offload_need(self, interpolator):
        """Test estimation with large offload requirement."""
        tensor = create_tensor_stats(1, 1024 * 1024 * 1024, 100.0)  # 1GB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)  # Short duration
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)

        # Need to offload 10GB in very short time
        scale = estimate_required_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            needed_offload_bytes=float(10 * 1024 * 1024 * 1024),
        )

        # Should be a large scale
        assert scale > 10.0


class TestComputeOptimalScale:
    """Test cases for compute_optimal_scale function."""

    @pytest.fixture
    def memory_transfers(self):
        """Create memory transfer data for tests."""
        return {
            1024: 0.001,  # 1KB -> 0.001ms
            1024 * 1024: 0.1,  # 1MB -> 0.1ms
            1024 * 1024 * 1024: 100.0,  # 1GB -> 100ms
        }

    @pytest.fixture
    def interpolator(self, memory_transfers):
        """Create interpolator for tests."""
        return MemoryTransferInterpolator(memory_transfers)

    def test_optimal_scale_already_under_limit(self, interpolator):
        """Test when model already fits in GPU memory."""
        # Small tensors that fit easily
        tensor = create_tensor_stats(1, 1024 * 1024, 0.1)  # 1MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)

        scale, _total_offload, _per_layer = compute_optimal_scale(
            layer_stats=[layer1, layer2],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=1024 * 1024 * 1024,  # 1GB limit
            initial_scale=1.0,
        )

        # Should return initial scale since we're under limit
        assert scale == 1.0

    def test_optimal_scale_needs_adjustment(self, interpolator):
        """Test when scale needs to be increased to meet target."""
        # Large tensors that need offloading
        tensor = create_tensor_stats(1, 500 * 1024 * 1024, 50.0)  # 500MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)
        layer3 = create_layer_stats("layer_2", [tensor], 10.0)

        scale, _total_offload, _per_layer = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=800 * 1024 * 1024,  # 800MB limit (need to offload 700MB)
            initial_scale=1.0,
        )

        # Scale should be increased
        assert scale > 1.0

    def test_optimal_scale_returns_per_layer_offload(self, interpolator):
        """Test that per-layer offload list is returned."""
        tensor = create_tensor_stats(1, 100 * 1024 * 1024, 10.0)  # 100MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)
        layer3 = create_layer_stats("layer_2", [tensor], 10.0)

        _scale, _total_offload, per_layer = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=200 * 1024 * 1024,
            initial_scale=1.0,
        )

        # Should have per_layer entries for layers 1 and 2 (layer 0 is first, not offloaded)
        assert len(per_layer) == 2

    def test_optimal_scale_with_initial_scale_below_one(self, interpolator):
        """Test behavior when initial_scale is below 1.0."""
        # Large tensors that need offloading
        tensor = create_tensor_stats(1, 500 * 1024 * 1024, 50.0)  # 500MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)
        layer3 = create_layer_stats("layer_2", [tensor], 10.0)

        # Test with initial_scale = 0.5
        scale_low, _total_offload_low, _ = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=800 * 1024 * 1024,  # 800MB limit
            initial_scale=0.5,
        )

        # Test with initial_scale = 1.0 for comparison
        scale_normal, _total_offload_normal, _ = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=800 * 1024 * 1024,  # Same limit
            initial_scale=1.0,
        )

        # Binary search range is [initial_scale, initial_scale * 100]
        # So with initial_scale=0.5, range is [0.5, 50]
        # With initial_scale=1.0, range is [1.0, 100]
        # The optimal scale should still be found if it's within the range

        # Scale should be >= initial_scale since we need to offload
        assert scale_low >= 0.5
        assert scale_normal >= 1.0

    def test_optimal_scale_initial_scale_very_low(self, interpolator):
        """Test when initial_scale is very low (e.g., 0.1) and needs significant adjustment."""
        tensor = create_tensor_stats(1, 500 * 1024 * 1024, 50.0)  # 500MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)
        layer3 = create_layer_stats("layer_2", [tensor], 10.0)

        # Very low initial scale
        scale, _total_offload, _ = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=800 * 1024 * 1024,
            initial_scale=0.1,
        )

        # Scale should be adjusted upward from 0.1
        # The search range is [0.1, 10], so if optimal is within that range, it will be found
        assert scale >= 0.1

    def test_optimal_scale_edge_case_very_small_initial_scale(self, interpolator):
        """Test that min_upper_bound prevents small initial_scale from limiting search.

        With min_upper_bound=100 (default), even with initial_scale=0.01,
        the search range becomes [0.01, 100] which includes the optimal scale.
        """
        tensor = create_tensor_stats(1, 500 * 1024 * 1024, 50.0)  # 500MB
        layer1 = create_layer_stats("layer_0", [tensor], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor], 10.0)
        layer3 = create_layer_stats("layer_2", [tensor], 10.0)

        # Very small initial scale - but min_upper_bound ensures sufficient range
        scale_tiny, total_offload_tiny, _ = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=800 * 1024 * 1024,
            initial_scale=0.01,  # Range will be [0.01, 100] due to min_upper_bound
        )

        # Compare with normal initial_scale
        scale_normal, total_offload_normal, _ = compute_optimal_scale(
            layer_stats=[layer1, layer2, layer3],
            memory_transfer_interpolator=interpolator,
            max_gpu_mem_bytes=800 * 1024 * 1024,
            initial_scale=1.0,  # Range will be [1.0, 100]
        )

        # Both should find approximately the same optimal scale (~3.43)
        assert abs(scale_tiny - scale_normal) < 0.1  # Within 0.1 of each other

        # Both should achieve similar offload amounts
        offload_diff_mb = abs(total_offload_normal - total_offload_tiny) / 1024 / 1024
        assert offload_diff_mb < 10  # Within 10MB of each other


class TestPrepareMergedLayers:
    """Test cases for prepare_merged_layers function."""

    def test_merge_two_layers(self):
        """Test merging pairs of layers."""
        tensor1 = create_tensor_stats(1, 1024, 0.1)
        tensor2 = create_tensor_stats(2, 2048, 0.2)
        layer1 = create_layer_stats("layer_0", [tensor1], 10.0)
        layer2 = create_layer_stats("layer_1", [tensor2], 20.0)

        result = prepare_merged_layers([layer1, layer2], group_size=2)

        assert len(result) == 1
        assert result[0].label == "layer_0"
        assert result[0].duration == 30.0
        assert len(result[0].tensors) == 2

    def test_merge_odd_number_of_layers(self):
        """Test merging with odd number of layers."""
        layers = [create_layer_stats(f"layer_{i}", [create_tensor_stats(i, 1024, 0.1)], 10.0) for i in range(3)]

        result = prepare_merged_layers(layers, group_size=2)

        assert len(result) == 2  # Two groups: (0,1) and (2)

    def test_merge_empty_list(self):
        """Test merging empty list."""
        result = prepare_merged_layers([], group_size=2)
        assert result == []

    def test_merge_single_layer(self):
        """Test merging single layer."""
        layer = create_layer_stats("layer_0", [create_tensor_stats(1, 1024, 0.1)], 10.0)

        result = prepare_merged_layers([layer], group_size=2)

        assert len(result) == 1
        assert result[0].label == "layer_0"


class TestKnapsackStrategy:
    """Test cases for KnapsackStrategy class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        strategy = KnapsackStrategy(scale=1.0)

        assert strategy.scale == 1.0
        assert strategy.cyclic is True
        assert strategy.group_size == 1
        assert strategy.threshold_mb == 0.1

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        strategy = KnapsackStrategy(
            scale=2.0,
            cyclic=False,
            group_size=2,
            threshold_mb=0.5,
        )

        assert strategy.scale == 2.0
        assert strategy.cyclic is False
        assert strategy.group_size == 2
        assert strategy.threshold_mb == 0.5

    def test_compute_empty_layer_stats(self):
        """Test compute with empty layer stats returns StrategyResult with empty strategy_map."""
        strategy = KnapsackStrategy(scale=1.0)

        result = strategy.compute([])

        assert result.strategy_map == {}

    def test_compute_warns_when_scale_adjusted(self):
        """Test that warning is issued when scale is adjusted."""
        # Create tensors with valid load times for interpolation
        tensor1 = create_tensor_stats(1, 100 * 1024 * 1024, 10.0)  # 100MB
        tensor2 = create_tensor_stats(2, 100 * 1024 * 1024, 10.0)
        layer1 = create_layer_stats("layer_0", [tensor1], 1.0)  # Very short duration
        layer2 = create_layer_stats("layer_1", [tensor2], 1.0)

        strategy = KnapsackStrategy(
            scale=0.001,  # Very low scale that will need adjustment
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.compute([layer1, layer2], max_gpu_mem_bytes=50 * 1024 * 1024)  # 50MB limit

        scale_warnings = [x for x in w if "insufficient" in str(x.message)]
        assert scale_warnings, (
            "Expected an 'insufficient' scale warning when scale is auto-adjusted for max_gpu_mem_bytes"
        )


class TestKnapsackBlockStrategy:
    """Test cases for KnapsackBlockStrategy class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        strategy = KnapsackBlockStrategy(scale=1.0)

        assert strategy.scale == 1.0
        assert strategy.group_size == 1
        assert strategy.threshold_mb == 0.1

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        strategy = KnapsackBlockStrategy(
            scale=2.0,
            group_size=2,
            threshold_mb=0.5,
        )

        assert strategy.scale == 2.0
        assert strategy.group_size == 2
        assert strategy.threshold_mb == 0.5


class TestGreedyStrategy:
    """Test cases for GreedyStrategy class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        strategy = GreedyStrategy()
        assert strategy.scale == 1.0
        assert strategy.threshold_mb == 0.1
        assert strategy.n_blocks == 4

    def test_init_custom_scale(self):
        """Test initialization with custom scale."""
        strategy = GreedyStrategy(scale=0.8)
        assert strategy.scale == 0.8

    def test_invalid_scale_raises(self):
        """Test that non-positive scale raises ValueError."""
        with pytest.raises(ValueError, match="scale must be positive"):
            GreedyStrategy(scale=0.0)
        with pytest.raises(ValueError, match="scale must be positive"):
            GreedyStrategy(scale=-1.0)

    def test_compute_empty_layer_stats(self):
        """Test compute with empty layer stats."""
        strategy = GreedyStrategy()
        result = strategy.compute([])
        assert result.strategy_map == {}

    def test_scale_increases_offloading(self):
        """Higher scale makes offloading more aggressive (more layers offloaded)."""
        tensor = create_tensor_stats(1, 10 * 1024 * 1024, load_time_ms=5.0)
        layers = [create_layer_stats(f"layer_{i}", [tensor], duration=2.0) for i in range(6)]

        conservative = GreedyStrategy(scale=0.5)
        aggressive = GreedyStrategy(scale=2.0)

        result_conservative = conservative.compute(layers)
        result_aggressive = aggressive.compute(layers)

        assert len(result_aggressive.strategy_map) >= len(result_conservative.strategy_map)

    def test_scale_decreases_offloading(self):
        """Lower scale makes offloading more conservative (fewer layers offloaded)."""
        tensor = create_tensor_stats(1, 10 * 1024 * 1024, load_time_ms=5.0)
        layers = [create_layer_stats(f"layer_{i}", [tensor], duration=2.0) for i in range(6)]

        baseline = GreedyStrategy(scale=1.0)
        conservative = GreedyStrategy(scale=0.5)

        result_baseline = baseline.compute(layers)
        result_conservative = conservative.compute(layers)

        assert len(result_conservative.strategy_map) <= len(result_baseline.strategy_map)

    def test_scale_one_matches_default(self):
        """scale=1.0 produces the same result as omitting scale."""
        tensor = create_tensor_stats(1, 10 * 1024 * 1024, load_time_ms=3.0)
        layers = [create_layer_stats(f"layer_{i}", [tensor], duration=2.0) for i in range(5)]

        default_strategy = GreedyStrategy()
        explicit_strategy = GreedyStrategy(scale=1.0)

        assert default_strategy.compute(layers).strategy_map == explicit_strategy.compute(layers).strategy_map


class TestAdaptiveKnapsackStrategy:
    """Test cases for AdaptiveKnapsackStrategy class."""

    def test_init_with_strategy_loader_type(self):
        """Test initialization with strategy loader type."""
        strategy = AdaptiveKnapsackStrategy(
            scale=1.0,
            loader_type="strategy",
        )

        assert strategy.loader_type == "strategy"

    def test_init_with_block_loader_type(self):
        """Test initialization with block loader type."""
        strategy = AdaptiveKnapsackStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
        )

        assert strategy.loader_type == "allocation_block_transfer"

    def test_create_strategy_returns_knapsack_for_strategy_type(self):
        """Test that _create_strategy returns KnapsackStrategy for strategy type."""
        strategy = AdaptiveKnapsackStrategy(
            scale=1.0,
            loader_type="strategy",
        )

        impl = strategy._create_strategy()

        assert isinstance(impl, KnapsackStrategy)

    def test_create_strategy_returns_block_for_block_type(self):
        """Test that _create_strategy returns KnapsackBlockStrategy for block types."""
        strategy = AdaptiveKnapsackStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
        )

        impl = strategy._create_strategy()

        assert isinstance(impl, KnapsackBlockStrategy)

    def test_block_loader_types_constant(self):
        """Test that BLOCK_LOADER_TYPES contains expected values."""
        assert "allocation_block_transfer" in AdaptiveKnapsackStrategy.BLOCK_LOADER_TYPES
        assert "raw_block_transfer" in AdaptiveKnapsackStrategy.BLOCK_LOADER_TYPES
        assert "strategy" not in AdaptiveKnapsackStrategy.BLOCK_LOADER_TYPES

    def test_init_passes_all_params(self):
        """Test that all parameters are stored correctly."""
        strategy = AdaptiveKnapsackStrategy(
            scale=2.5,
            loader_type="raw_block_transfer",
            cyclic=False,
            group_size=3,
            threshold_mb=0.25,
        )

        assert strategy.scale == 2.5
        assert strategy.loader_type == "raw_block_transfer"
        assert strategy.cyclic is False
        assert strategy.group_size == 3
        assert strategy.threshold_mb == 0.25


class TestComputePeakMemoryFromStrategy:
    """This test suite validates the compute_peak_memory_from_strategy function
    by simulating the sliding-window pattern of tensor loading and releasing to
    compute peak GPU memory. It covers overlapping loads/releases, single-layer
    cases, duplicates, and edge conditions to ensure correctness.
    """

    def test_empty_inputs_returns_zero(self):
        """Test that empty inputs return 0 peak memory."""
        result = compute_peak_memory_from_strategy(
            strategy_map={},
            release_strategy_map={},
            layer_stats=[],
        )
        assert result == 0

    def test_empty_layer_stats_returns_zero(self):
        """Test that empty layer_stats returns 0 even with non-empty maps."""
        tensor_info = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1000,
            load_time_ms=0.1,
        )
        result = compute_peak_memory_from_strategy(
            strategy_map={"layer_0": [tensor_info]},
            release_strategy_map={"layer_1": [tensor_info]},
            layer_stats=[],
        )
        assert result == 0

    def test_single_layer_single_tensor(self):
        """Test peak memory with a single layer and single tensor."""
        tensor_info = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,  # 1 MB
            load_time_ms=0.1,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[tensor_info], duration=1.0),
        ]

        result = compute_peak_memory_from_strategy(
            strategy_map={"layer_0": [tensor_info]},
            release_strategy_map={"layer_0": [tensor_info]},
            layer_stats=layer_stats,
        )

        # Peak is 1 MB (tensor loaded then released in same layer)
        assert result == 1024 * 1024

    def test_sliding_window_with_same_layer_release(self):
        """Test peak memory when release is scheduled in the same layer as next load.

        Pattern:
        - layer_0: load tensor_1
        - layer_1: load tensor_2, release tensor_1 (release happens AFTER load)
        - layer_2: release tensor_2

        Since releases occur at the END of a layer (after loads complete), both
        tensors are simultaneously resident at layer_1. This creates an overlap
        where peak memory is the sum of both tensors.
        """
        tensor_1 = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,  # 1 MB
            load_time_ms=0.1,
        )
        tensor_2 = TensorStatistics(
            tensor_id=2,
            name="tensor_2",
            size_bytes=1024 * 1024,  # 1 MB
            load_time_ms=0.1,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[tensor_1], duration=1.0),
            LayerStatistics(label="layer_1", tensors=[tensor_2], duration=1.0),
            LayerStatistics(label="layer_2", tensors=[], duration=1.0),
        ]

        result = compute_peak_memory_from_strategy(
            strategy_map={"layer_0": [tensor_1], "layer_1": [tensor_2]},
            release_strategy_map={"layer_1": [tensor_1], "layer_2": [tensor_2]},
            layer_stats=layer_stats,
        )

        # At layer_1: tensor_1 is still loaded (released at end), tensor_2 is loaded
        # Peak is 2 MB because both tensors overlap in memory
        assert result == 2 * 1024 * 1024

    def test_sliding_window_with_overlap(self):
        """Test peak memory with overlapping load/release.

        Pattern:
        - layer_0: load tensor_1 (1 MB)
        - layer_1: load tensor_2 (2 MB)
        - layer_2: release tensor_1
        - layer_3: release tensor_2

        Peak should be 3 MB (both tensors loaded at layer_1 and layer_2).
        """
        tensor_1 = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,  # 1 MB
            load_time_ms=0.1,
        )
        tensor_2 = TensorStatistics(
            tensor_id=2,
            name="tensor_2",
            size_bytes=2 * 1024 * 1024,  # 2 MB
            load_time_ms=0.2,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[tensor_1], duration=1.0),
            LayerStatistics(label="layer_1", tensors=[tensor_2], duration=1.0),
            LayerStatistics(label="layer_2", tensors=[], duration=1.0),
            LayerStatistics(label="layer_3", tensors=[], duration=1.0),
        ]

        result = compute_peak_memory_from_strategy(
            strategy_map={"layer_0": [tensor_1], "layer_1": [tensor_2]},
            release_strategy_map={"layer_2": [tensor_1], "layer_3": [tensor_2]},
            layer_stats=layer_stats,
        )

        # Peak is 3 MB (layer_1 and layer_2: both tensors loaded)
        assert result == 3 * 1024 * 1024

    def test_multiple_tensors_in_single_layer(self):
        """Test peak memory when multiple tensors are loaded in a single layer."""
        tensor_1 = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,  # 1 MB
            load_time_ms=0.1,
        )
        tensor_2 = TensorStatistics(
            tensor_id=2,
            name="tensor_2",
            size_bytes=2 * 1024 * 1024,  # 2 MB
            load_time_ms=0.2,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[tensor_1, tensor_2], duration=1.0),
            LayerStatistics(label="layer_1", tensors=[], duration=1.0),
        ]

        result = compute_peak_memory_from_strategy(
            strategy_map={"layer_0": [tensor_1, tensor_2]},
            release_strategy_map={"layer_1": [tensor_1, tensor_2]},
            layer_stats=layer_stats,
        )

        # Peak is 3 MB (both tensors loaded at layer_0)
        assert result == 3 * 1024 * 1024

    def test_no_strategy_returns_zero(self):
        """Test that layers with no strategy return 0 peak memory."""
        tensor_info = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,
            load_time_ms=0.1,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[tensor_info], duration=1.0),
            LayerStatistics(label="layer_1", tensors=[], duration=1.0),
        ]

        # Empty strategy_map means no tensors are loaded via strategy
        result = compute_peak_memory_from_strategy(
            strategy_map={},
            release_strategy_map={},
            layer_stats=layer_stats,
        )

        assert result == 0

    def test_duplicate_tensor_ids_not_double_counted(self):
        """Test that duplicate tensor IDs in strategy are not double-counted."""
        tensor_info = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,  # 1 MB
            load_time_ms=0.1,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[tensor_info], duration=1.0),
            LayerStatistics(label="layer_1", tensors=[tensor_info], duration=1.0),
        ]

        # Same tensor_id loaded in both layers (e.g., shared tensor)
        result = compute_peak_memory_from_strategy(
            strategy_map={"layer_0": [tensor_info], "layer_1": [tensor_info]},
            release_strategy_map={"layer_1": [tensor_info]},
            layer_stats=layer_stats,
        )

        # Should be 1 MB, not 2 MB (same tensor, not counted twice)
        assert result == 1024 * 1024

    def test_release_without_load_is_safe(self):
        """Test that releasing a tensor that was never loaded doesn't crash."""
        tensor_info = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=1024 * 1024,
            load_time_ms=0.1,
        )
        layer_stats = [
            LayerStatistics(label="layer_0", tensors=[], duration=1.0),
        ]

        # Release map has tensor that's not in strategy_map
        result = compute_peak_memory_from_strategy(
            strategy_map={},
            release_strategy_map={"layer_0": [tensor_info]},
            layer_stats=layer_stats,
        )

        assert result == 0

    def test_realistic_transformer_pattern(self):
        """Test with a pattern similar to transformer layer offloading.

        Pattern simulates:
        - Load layer N weights during layer N-1 compute
        - Release layer N-1 weights after layer N-1 compute

        This creates a sliding window of 2 layers worth of memory.
        """
        # Create 4 layers worth of tensors (100 MB each)
        tensors = [
            TensorStatistics(
                tensor_id=i,
                name=f"layer_{i}_weights",
                size_bytes=100 * 1024 * 1024,  # 100 MB each
                load_time_ms=1.0,
            )
            for i in range(4)
        ]

        layer_stats = [LayerStatistics(label=f"layer_{i}", tensors=[tensors[i]], duration=10.0) for i in range(4)]

        # Load pattern: load next layer's weights during current layer
        # layer_0: load layer_0 weights
        # layer_1: load layer_1 weights, release layer_0
        # layer_2: load layer_2 weights, release layer_1
        # layer_3: load layer_3 weights, release layer_2
        strategy_map = {
            "layer_0": [tensors[0]],
            "layer_1": [tensors[1]],
            "layer_2": [tensors[2]],
            "layer_3": [tensors[3]],
        }
        release_strategy_map = {
            "layer_1": [tensors[0]],
            "layer_2": [tensors[1]],
            "layer_3": [tensors[2]],
            # tensors[3] released after last layer (not in this map)
        }

        result = compute_peak_memory_from_strategy(
            strategy_map=strategy_map,
            release_strategy_map=release_strategy_map,
            layer_stats=layer_stats,
        )

        # Peak is 200 MB (2 layers of weights simultaneously: e.g., at layer_1 both 0 and 1 loaded)
        assert result == 200 * 1024 * 1024


# =============================================================================
# Known-Optimal Knapsack Solver Validation
# =============================================================================


class TestKnapsackSolverOptimality:
    """Validates the scipy differential_evolution knapsack solver against problems with known optimal solutions.

    These tests verify that the solver finds correct or near-optimal solutions.
    """

    @staticmethod
    def _make_layer(tensors: list[TensorStatistics], duration: float) -> LayerStatistics:
        """Create a LayerStatistics with given tensors and duration."""
        return LayerStatistics(label="test_layer", tensors=tensors, duration=duration)

    @staticmethod
    def _make_tensor(tensor_id: int, size_mb: float, load_time_ms: float) -> TensorStatistics:
        """Create a TensorStatistics with given size and transfer time."""
        return TensorStatistics(
            tensor_id=tensor_id,
            name=f"tensor_{tensor_id}",
            size_bytes=int(size_mb * 1024 * 1024),
            load_time_ms=load_time_ms,
        )

    def test_all_items_fit(self):
        """When total weight fits in capacity, all tensors should be selected."""
        # Capacity: duration=100.0, scale=1.0 → capacity=100_000_000
        # 3 tensors with load_time=10.0 each → total weight=30_000_000 (fits)
        tensors = [
            self._make_tensor(1, 10.0, 10.0),
            self._make_tensor(2, 20.0, 10.0),
            self._make_tensor(3, 30.0, 10.0),
        ]
        layer = self._make_layer(tensors, duration=100.0)

        result = compute_solution(duration=100.0, layer=layer, scale=1.0, threshold_mb=0.1)

        assert len(result) == 3
        selected_ids = {t.tensor_id for t in result}
        assert selected_ids == {1, 2, 3}

    def test_single_item_fits(self):
        """Only the item that fits within capacity should be selected."""
        # Capacity: duration=10.0, scale=1.0 → capacity=10_000_000
        # Tensor A: load_time=5.0 (weight=5M, fits), size=10MB
        # Tensor B: load_time=20.0 (weight=20M, exceeds), size=50MB
        tensors = [
            self._make_tensor(1, 10.0, 5.0),
            self._make_tensor(2, 50.0, 20.0),
        ]
        layer = self._make_layer(tensors, duration=10.0)

        result = compute_solution(duration=10.0, layer=layer, scale=1.0, threshold_mb=0.1)

        selected_ids = {t.tensor_id for t in result}
        assert 1 in selected_ids
        assert 2 not in selected_ids

    def test_classic_knapsack_known_optimal(self):
        """Solver should find the optimal subset for a classic 0/1 knapsack.

        Problem:
            Capacity: 50ms (weight units)
            Item A: weight=10ms, profit=60MB
            Item B: weight=20ms, profit=100MB
            Item C: weight=30ms, profit=120MB

        All feasible subsets:
            {A}:       weight=10, profit=60
            {B}:       weight=20, profit=100
            {C}:       weight=30, profit=120
            {A,B}:     weight=30, profit=160
            {A,C}:     weight=40, profit=180
            {B,C}:     weight=50, profit=220  ← OPTIMAL
            {A,B,C}:   weight=60 > 50 (infeasible)

        Expected: Items B and C selected (profit=220MB).
        """
        tensors = [
            self._make_tensor(1, 60.0, 10.0),  # A
            self._make_tensor(2, 100.0, 20.0),  # B
            self._make_tensor(3, 120.0, 30.0),  # C
        ]
        layer = self._make_layer(tensors, duration=50.0)

        result = compute_solution(duration=50.0, layer=layer, scale=1.0, threshold_mb=0.1)

        selected_ids = {t.tensor_id for t in result}
        total_profit = sum(t.size_bytes for t in result)

        # Optimal: B+C = 220MB
        expected_profit = int((100.0 + 120.0) * 1024 * 1024)
        assert selected_ids == {2, 3}, f"Expected items B,C but got {selected_ids}"
        assert total_profit == expected_profit

    def test_greedy_tricky_case(self):
        """Solver should beat the greedy-by-ratio heuristic.

        Problem:
            Capacity: 10ms (weight units)
            Item A: weight=1ms, profit=2MB  → ratio=2.0 (greedy picks this first)
            Item B: weight=10ms, profit=15MB → ratio=1.5

        Greedy by value/weight ratio: picks A (w=1), then B doesn't fit (1+10=11>10).
            Greedy profit = 2MB.

        Optimal: B alone (w=10≤10, profit=15MB).
        Expected: Only Item B selected.
        """
        tensors = [
            self._make_tensor(1, 2.0, 1.0),  # A: high ratio but low absolute value
            self._make_tensor(2, 15.0, 10.0),  # B: lower ratio but high absolute value
        ]
        layer = self._make_layer(tensors, duration=10.0)

        result = compute_solution(duration=10.0, layer=layer, scale=1.0, threshold_mb=0.1)

        selected_ids = {t.tensor_id for t in result}
        total_profit = sum(t.size_bytes for t in result)

        # Optimal: B alone = 15MB (greedy would give A alone = 2MB)
        expected_profit = int(15.0 * 1024 * 1024)
        assert selected_ids == {2}, f"Expected only item B but got {selected_ids}"
        assert total_profit == expected_profit

    def test_no_items_fit(self):
        """When no individual item fits, result should be empty."""
        # Capacity: duration=1.0, scale=1.0 → capacity=1_000_000
        # Both tensors have load_time=5.0 (weight=5_000_000 each, both exceed capacity)
        tensors = [
            self._make_tensor(1, 10.0, 5.0),
            self._make_tensor(2, 20.0, 5.0),
        ]
        layer = self._make_layer(tensors, duration=1.0)

        result = compute_solution(duration=1.0, layer=layer, scale=1.0, threshold_mb=0.1)

        assert len(result) == 0

    def test_scale_increases_capacity(self):
        """Higher scale should allow more items to be selected."""
        # At scale=1.0, capacity=10M → only item A fits (weight=5M)
        # At scale=3.0, capacity=30M → both items fit (total weight=25M)
        tensors = [
            self._make_tensor(1, 10.0, 5.0),  # weight=5M
            self._make_tensor(2, 20.0, 20.0),  # weight=20M
        ]
        layer = self._make_layer(tensors, duration=10.0)

        result_low = compute_solution(duration=10.0, layer=layer, scale=1.0, threshold_mb=0.1)
        result_high = compute_solution(duration=10.0, layer=layer, scale=3.0, threshold_mb=0.1)

        assert len(result_low) == 1
        assert len(result_high) == 2
