# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AdaptiveStrategy and strategy evaluation."""

import warnings

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator
from flextensor.strategy.adaptive import AdaptiveStrategy
from flextensor.strategy.evaluation import (
    StrategyScore,
    _compute_overhead,
    _compute_peak_memory,
    _count_consecutive_violations,
    evaluate_strategy_result,
)
from flextensor.strategy.protocol import BlockStrategyData, StrategyComputeError, StrategyResult

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _tensor(tensor_id: int, size_bytes: int, load_time_ms: float = 1.0) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=tensor_id,
        name=f"tensor_{tensor_id}",
        size_bytes=size_bytes,
        load_time_ms=load_time_ms,
    )


def _layer(label: str, tensors: list[TensorStatistics], duration: float) -> LayerStatistics:
    return LayerStatistics(label=label, tensors=tensors, duration=duration)


def _make_simple_model(
    n_layers: int = 5,
    tensors_per_layer: int = 3,
    tensor_size: int = 10 * 1024**2,
    duration: float = 100.0,
) -> tuple[list[LayerStatistics], dict[int, float]]:
    """Create a simple model with uniform layers for testing."""
    layers = []
    tid = 0
    for i in range(n_layers):
        tensors = []
        for _ in range(tensors_per_layer):
            tensors.append(_tensor(tid, tensor_size, load_time_ms=tensor_size / (10 * 1024**2)))
            tid += 1
        layers.append(_layer(f"layer_{i}", tensors, duration))
    memory_stats = {tensor_size: tensor_size / (10 * 1024**2)}
    return layers, memory_stats


# ===========================================================================
# StrategyScore comparison tests
# ===========================================================================


class TestStrategyScoreComparison:
    def test_valid_beats_invalid(self):
        valid = StrategyScore("A", 100, 0.5, 0, True)
        invalid = StrategyScore("B", 50, 0.01, 1, False)
        assert valid < invalid

    def test_lower_overhead_wins_among_valid(self):
        low = StrategyScore("A", 100, 0.01, 0, True)
        high = StrategyScore("B", 50, 0.10, 0, True)
        assert low < high

    def test_equal_scores(self):
        a = StrategyScore("A", 100, 0.05, 0, True)
        b = StrategyScore("B", 100, 0.05, 0, True)
        assert not (a < b)
        assert not (b < a)

    def test_invalid_lower_peak_memory_wins(self):
        a = StrategyScore("A", 80, 0.10, 1, False)
        b = StrategyScore("B", 100, 0.05, 2, False)
        assert a < b

    def test_invalid_equal_peak_memory_is_tie(self):
        a = StrategyScore("A", 100, 0.05, 1, False)
        b = StrategyScore("B", 100, 0.10, 2, False)
        assert not (a < b)
        assert not (b < a)

    def test_near_equal_overhead_breaks_tie_on_memory(self):
        """Float noise in overhead should not affect ordering."""
        a = StrategyScore("A", 80, 0.1 + 1e-15, 0, True)
        b = StrategyScore("B", 100, 0.1 - 1e-15, 0, True)
        assert a < b, "Nearly equal overhead should fall through to peak memory comparison"
        assert not (b < a)

    def test_genuinely_different_overhead_respected(self):
        """Overheads that differ by more than tolerance should still compare on overhead."""
        a = StrategyScore("A", 200, 0.01, 0, True)
        b = StrategyScore("B", 50, 0.10, 0, True)
        assert a < b, "Lower overhead should win despite higher peak memory"


# ===========================================================================
# Peak memory computation
# ===========================================================================


class TestComputePeakMemory:
    def test_no_offload(self):
        """No offloading means peak = total model size."""
        layers, _ = _make_simple_model(n_layers=3, tensors_per_layer=2, tensor_size=1000)
        result = StrategyResult(strategy_map={})
        assert _compute_peak_memory(result, layers) == 6000

    def test_full_offload_with_blocks(self):
        """All tensors offloaded: peak = block memory (no resident tensors)."""
        t0 = _tensor(0, 1000)
        t1 = _tensor(1, 2000)
        layers = [_layer("L0", [t0], 10.0), _layer("L1", [t1], 10.0)]
        strategy_map = {"L0": [t1]}
        block_data = BlockStrategyData(
            label_to_size_map={"L0": 2000},
            allocation_ordered={0: ["L0"]},
            block_sizes={0: 2000},
            label_to_block_id={"L0": 0},
            transfer_to_compute_map={"L0": "L1"},
        )
        result = StrategyResult(strategy_map=strategy_map, block_data=block_data)
        # block_memory=2000, total_model=3000, total_offloaded=2000 → 2000+3000-2000=3000
        # But t0 (1000) is still on GPU, plus block (2000) = 3000
        assert _compute_peak_memory(result, layers) == 3000

    def test_dict_vs_list_block_sizes(self):
        """Block sizes can be dict or list."""
        layers = [_layer("L0", [_tensor(0, 500)], 10.0)]
        result_dict = StrategyResult(
            strategy_map={},
            block_data=BlockStrategyData(
                label_to_size_map={},
                allocation_ordered={},
                block_sizes={0: 100, 1: 200},
                label_to_block_id={},
                transfer_to_compute_map={},
            ),
        )
        result_list = StrategyResult(
            strategy_map={},
            block_data=BlockStrategyData(
                label_to_size_map={},
                allocation_ordered={},
                block_sizes=[100, 200],
                label_to_block_id={},
                transfer_to_compute_map={},
            ),
        )
        assert _compute_peak_memory(result_dict, layers) == _compute_peak_memory(result_list, layers)


# ===========================================================================
# Overhead computation
# ===========================================================================


class TestComputeOverhead:
    def test_zero_overhead_when_transfers_fit(self):
        """Transfers that fit in compute windows cause no overhead."""
        layers = [
            _layer("L0", [_tensor(0, 1000)], 100.0),
            _layer("L1", [_tensor(1, 1000)], 100.0),
        ]
        strategy_map = {"L0": [_tensor(1, 1000)]}
        interp = MemoryTransferInterpolator({1000: 1.0})
        assert _compute_overhead(strategy_map, layers, interp) == 0.0

    def test_positive_overhead_when_transfer_exceeds_compute(self):
        """Transfer slower than compute causes positive overhead."""
        layers = [
            _layer("L0", [_tensor(0, 1000)], 1.0),
            _layer("L1", [_tensor(1, 1000)], 1.0),
        ]
        strategy_map = {"L0": [_tensor(1, 1000)]}
        interp = MemoryTransferInterpolator({1000: 10.0})
        overhead = _compute_overhead(strategy_map, layers, interp)
        # Transfer 10ms, compute window 1ms → sync overhead 9ms, total compute 2ms
        assert overhead == pytest.approx(9.0 / 2.0)

    def test_no_offload_means_no_overhead(self):
        """Empty strategy_map → no transfers → zero overhead."""
        layers = [_layer("L0", [_tensor(0, 1000)], 10.0)]
        interp = MemoryTransferInterpolator({1000: 1.0})
        assert _compute_overhead({}, layers, interp) == 0.0


# ===========================================================================
# Consecutive violations
# ===========================================================================


class TestCountConsecutiveViolations:
    def _strategy_map_for(self, *labels: str) -> dict[str, list[TensorStatistics]]:
        """Build a strategy_map with non-empty entries for the given labels."""
        return {label: [_tensor(0, 100)] for label in labels}

    @staticmethod
    def _layers_for(*labels: str) -> list[LayerStatistics]:
        """Build minimal layer_stats in execution order."""
        return [_layer(label, [_tensor(0, 100)], 10.0) for label in labels]

    def test_no_violations(self):
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0, "L1": 1, "L2": 0},
            transfer_to_compute_map={},
        )
        strategy_map = self._strategy_map_for("L0", "L1", "L2")
        layers = self._layers_for("L0", "L1", "L2")
        assert _count_consecutive_violations(block_data, strategy_map, layers) == 0

    def test_one_violation(self):
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0, "L1": 0, "L2": 1},
            transfer_to_compute_map={},
        )
        strategy_map = self._strategy_map_for("L0", "L1", "L2")
        layers = self._layers_for("L0", "L1", "L2")
        assert _count_consecutive_violations(block_data, strategy_map, layers) == 1

    def test_cyclic_not_counted(self):
        """Last == first should NOT be counted (handled by loader sync)."""
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0, "L1": 1, "L2": 0},
            transfer_to_compute_map={},
        )
        strategy_map = self._strategy_map_for("L0", "L1", "L2")
        layers = self._layers_for("L0", "L1", "L2")
        assert _count_consecutive_violations(block_data, strategy_map, layers) == 0

    def test_single_layer(self):
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0},
            transfer_to_compute_map={},
        )
        strategy_map = self._strategy_map_for("L0")
        layers = self._layers_for("L0")
        assert _count_consecutive_violations(block_data, strategy_map, layers) == 0

    def test_non_transferring_layers_ignored(self):
        """Layers without strategy_map entries don't cause violations."""
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0, "L1": 0, "L2": 0, "L3": 1},
            transfer_to_compute_map={},
        )
        strategy_map = self._strategy_map_for("L0", "L3")
        layers = self._layers_for("L0", "L1", "L2", "L3")
        assert _count_consecutive_violations(block_data, strategy_map, layers) == 0

    def test_block_grouped_order_does_not_cause_phantom_violations(self):
        """Round-robin assignment should have 0 violations even when
        label_to_block_id keys are grouped by block."""
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0, "L2": 0, "L1": 1, "L3": 1},
            transfer_to_compute_map={},
        )
        strategy_map = self._strategy_map_for("L0", "L1", "L2", "L3")
        layers = self._layers_for("L0", "L1", "L2", "L3")
        assert _count_consecutive_violations(block_data, strategy_map, layers) == 0


# ===========================================================================
# evaluate_strategy_result
# ===========================================================================


class TestEvaluateStrategyResult:
    def test_valid_result(self):
        layers = [_layer("L0", [_tensor(0, 1000)], 100.0)]
        result = StrategyResult(strategy_map={})
        score = evaluate_strategy_result(result, layers, "test", max_gpu_mem_bytes=2000)
        assert score.is_valid
        assert score.strategy_name == "test"
        assert score.peak_memory_bytes == 1000

    def test_memory_exceeded_is_invalid(self):
        layers = [_layer("L0", [_tensor(0, 5000)], 100.0)]
        result = StrategyResult(strategy_map={})
        score = evaluate_strategy_result(result, layers, "test", max_gpu_mem_bytes=2000)
        assert not score.is_valid

    def test_no_memory_limit_always_valid(self):
        layers = [_layer("L0", [_tensor(0, 5000)], 100.0)]
        result = StrategyResult(strategy_map={})
        score = evaluate_strategy_result(result, layers, "test", max_gpu_mem_bytes=None)
        assert score.is_valid

    def test_violations_make_invalid(self):
        layers = [_layer("L0", [_tensor(0, 100)], 10.0), _layer("L1", [_tensor(1, 100)], 10.0)]
        block_data = BlockStrategyData(
            label_to_size_map={},
            allocation_ordered={},
            block_sizes={},
            label_to_block_id={"L0": 0, "L1": 0},
            transfer_to_compute_map={},
        )
        strategy_map: dict[str, list[TensorStatistics]] = {
            "L0": [_tensor(0, 50)],
            "L1": [_tensor(1, 50)],
        }
        result = StrategyResult(strategy_map=strategy_map, block_data=block_data)
        score = evaluate_strategy_result(result, layers, "test")
        assert not score.is_valid
        assert score.consecutive_violations == 1


# ===========================================================================
# AdaptiveStrategy
# ===========================================================================


class TestAdaptiveStrategy:
    def test_init_validation(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            AdaptiveStrategy(scale=0.0)

    def test_compute_returns_result(self):
        """Smoke test: compute() runs all candidates and returns a StrategyResult."""
        layers, mem_stats = _make_simple_model(n_layers=5, tensors_per_layer=3)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
        )
        result = strategy.compute(layers, mem_stats, max_gpu_mem_bytes=200 * 1024**2)
        assert isinstance(result, StrategyResult)
        assert strategy.selected_strategy_name != ""
        assert len(strategy.all_scores) > 0

    def test_non_block_loader_uses_knapsack(self):
        """Non-block loader should only run KnapsackStrategy."""
        layers, mem_stats = _make_simple_model(n_layers=3)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="strategy",
            n_blocks=4,
        )
        result = strategy.compute(layers, mem_stats)
        assert isinstance(result, StrategyResult)
        assert strategy.selected_strategy_name == "Knapsack"
        assert len(strategy.all_scores) == 1

    def test_selected_strategy_name_set_after_compute(self):
        """selected_strategy_name should raise before compute, be set after."""
        strategy = AdaptiveStrategy(scale=1.0, n_blocks=4)
        with pytest.raises(RuntimeError, match=r"compute\(\) has been called"):
            _ = strategy.selected_strategy_name

        layers, mem_stats = _make_simple_model(n_layers=3)
        strategy.compute(layers, mem_stats)
        assert strategy.selected_strategy_name != ""

    def test_all_scores_populated(self):
        """all_scores should contain one entry per candidate (default: 3 fast)."""
        layers, mem_stats = _make_simple_model(n_layers=3)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
        )
        strategy.compute(layers, mem_stats, max_gpu_mem_bytes=200 * 1024**2)
        assert len(strategy.all_scores) == 3
        names = {s.strategy_name for s in strategy.all_scores}
        assert "KnapsackBlock" in names
        assert "GlobalOffload(Optimized)" in names
        assert "GlobalOffload(Strict)" in names

    def test_all_scores_populated_extra_optimization(self):
        """extra_optimization=True should include TensorSelection candidates."""
        layers, mem_stats = _make_simple_model(n_layers=3)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
            extra_optimization=True,
        )
        strategy.compute(layers, mem_stats, max_gpu_mem_bytes=200 * 1024**2)
        assert len(strategy.all_scores) == 5
        names = {s.strategy_name for s in strategy.all_scores}
        assert "KnapsackBlock" in names
        assert "GlobalOffload(Optimized)" in names
        assert "GlobalOffload(Strict)" in names
        assert "TensorSelection(Optimized)" in names
        assert "TensorSelection(Strict)" in names

    def test_warns_on_invalid_best(self):
        """Should warn when the best result still violates constraints."""
        layers, mem_stats = _make_simple_model(n_layers=3, tensors_per_layer=3, tensor_size=100 * 1024**2)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.compute(layers, mem_stats, max_gpu_mem_bytes=1)  # impossibly small
            adaptive_warnings = [x for x in w if "AdaptiveStrategy" in str(x.message)]
            assert len(adaptive_warnings) >= 1

    def test_block_data_present_for_block_loader(self):
        """Block loader strategies should produce block_data."""
        layers, mem_stats = _make_simple_model(n_layers=5)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
        )
        result = strategy.compute(layers, mem_stats, max_gpu_mem_bytes=200 * 1024**2)
        assert result.block_data is not None

    def test_strategy_compute_error_is_caught(self, monkeypatch):
        """StrategyComputeError from a candidate is caught; remaining candidates still run."""
        from flextensor.strategy import knapsack

        layers, mem_stats = _make_simple_model(n_layers=5)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
        )

        def _failing_compute(self, *a, **kw):
            raise StrategyComputeError("cannot satisfy constraints")

        monkeypatch.setattr(knapsack.KnapsackBlockStrategy, "compute", _failing_compute)

        result = strategy.compute(layers, mem_stats, max_gpu_mem_bytes=200 * 1024**2)
        assert isinstance(result, StrategyResult)
        assert strategy.selected_strategy_name != "KnapsackBlock"

    def test_programming_error_propagates(self, monkeypatch):
        """TypeError (a programming bug) must NOT be caught — it should propagate."""
        from flextensor.strategy import knapsack

        layers, mem_stats = _make_simple_model(n_layers=5)
        strategy = AdaptiveStrategy(
            scale=1.0,
            loader_type="allocation_block_transfer",
            n_blocks=4,
        )

        def _buggy_compute(self, *a, **kw):
            raise TypeError("unexpected None for tensor_id")

        monkeypatch.setattr(knapsack.KnapsackBlockStrategy, "compute", _buggy_compute)

        with pytest.raises(TypeError, match="unexpected None"):
            strategy.compute(layers, mem_stats, max_gpu_mem_bytes=200 * 1024**2)
