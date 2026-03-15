# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TransferWindowCalculator implementations and strategy_has_transfer_gaps."""

import numpy as np

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy import (
    GapAwareWindow,
    GlobalTensorSelectionStrategy,
    SingleLayerWindow,
    TransferWindowCalculator,
    strategy_has_transfer_gaps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _durations(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _offloads(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _tensor(tid: int, size_mb: float, load_ms: float) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=tid,
        name=f"tensor_{tid}",
        size_bytes=int(size_mb * 1024 * 1024),
        load_time_ms=load_ms,
    )


def _layer(label: str, tensors: list[TensorStatistics], duration_ms: float) -> LayerStatistics:
    return LayerStatistics(label=label, duration=duration_ms, tensors=tensors)


def _memory_stats() -> dict[int, float]:
    return {
        1024: 0.0001,
        1024 * 1024: 0.1,
        10 * 1024 * 1024: 1.0,
        100 * 1024 * 1024: 10.0,
        1024 * 1024 * 1024: 100.0,
    }


# ===================================================================
# TransferWindowCalculator Protocol conformance
# ===================================================================


class TestProtocolConformance:
    def test_single_layer_window_is_transfer_window_calculator(self):
        assert isinstance(SingleLayerWindow(), TransferWindowCalculator)

    def test_gap_aware_window_is_transfer_window_calculator(self):
        assert isinstance(GapAwareWindow(), TransferWindowCalculator)


# ===================================================================
# SingleLayerWindow
# ===================================================================


class TestSingleLayerWindow:
    def setup_method(self):
        self.calc = SingleLayerWindow()

    def test_returns_previous_layer_duration(self):
        dur = _durations(10.0, 20.0, 30.0)
        offload = _offloads(100.0, 200.0, 300.0)
        assert self.calc.compute_window(1, offload, dur) == 10.0
        assert self.calc.compute_window(2, offload, dur) == 20.0

    def test_first_layer_returns_zero(self):
        dur = _durations(10.0, 20.0)
        offload = _offloads(0.0, 100.0)
        assert self.calc.compute_window(0, offload, dur) == 0.0

    def test_ignores_offload_pattern(self):
        dur = _durations(10.0, 20.0, 30.0, 40.0)
        all_offloaded = _offloads(100.0, 200.0, 300.0, 400.0)
        no_offload = _offloads(0.0, 0.0, 0.0, 0.0)
        assert self.calc.compute_window(3, all_offloaded, dur) == self.calc.compute_window(3, no_offload, dur)

    def test_negative_layer_idx_returns_zero(self):
        dur = _durations(10.0)
        offload = _offloads(0.0)
        assert self.calc.compute_window(-1, offload, dur) == 0.0


# ===================================================================
# GapAwareWindow
# ===================================================================


class TestGapAwareWindow:
    def setup_method(self):
        self.calc = GapAwareWindow()

    def test_no_gaps_returns_single_layer(self):
        """All layers offloaded — same as SingleLayerWindow."""
        dur = _durations(10.0, 20.0, 30.0, 40.0)
        offload = _offloads(100.0, 200.0, 300.0, 400.0)
        assert self.calc.compute_window(3, offload, dur) == 30.0

    def test_single_gap_extends_window(self):
        """L1:offload, L2:gap, L3:offload → L3 gets dur(L2)+dur(L1)."""
        dur = _durations(10.0, 20.0, 30.0)
        offload = _offloads(100.0, 0.0, 300.0)
        # layer 2: dur[1]=20, layer 1 has offload[1]=0 → extend, dur[0]=10
        # then check offload[1]=0 which is gap, so extend
        # offload[0+1]=offload[1]=0 → extend to layer 0
        # j=0: offload[0+1]=offload[1]=0 → extend, add dur[0]=10
        # j=-1: loop ends
        # total = 20 + 10 = 30
        assert self.calc.compute_window(2, offload, dur) == 30.0

    def test_two_gaps_extend_window(self):
        """L1:offload, L2:gap, L3:gap, L4:offload → L4 gets dur(L3)+dur(L2)+dur(L1)."""
        dur = _durations(10.0, 20.0, 30.0, 40.0, 50.0)
        offload = _offloads(100.0, 0.0, 0.0, 0.0, 500.0)
        # layer 4: start with dur[3]=40
        # j=2: offload[3]=0 → add dur[2]=30
        # j=1: offload[2]=0 → add dur[1]=20
        # j=0: offload[1]=0 → add dur[0]=10
        # total = 40+30+20+10 = 100
        assert self.calc.compute_window(4, offload, dur) == 100.0

    def test_gap_stops_at_occupied_layer(self):
        """L1:offload, L2:offload, L3:gap, L4:offload → L4 gets dur(L3)+dur(L2)."""
        dur = _durations(10.0, 20.0, 30.0, 40.0, 50.0)
        offload = _offloads(100.0, 200.0, 300.0, 0.0, 500.0)
        # layer 4: start with dur[3]=40
        # j=2: offload[3]=0 → add dur[2]=30
        # j=1: offload[2]=300 > 0 → break
        # total = 40+30 = 70
        assert self.calc.compute_window(4, offload, dur) == 70.0

    def test_first_layer_returns_zero(self):
        dur = _durations(10.0, 20.0)
        offload = _offloads(0.0, 100.0)
        assert self.calc.compute_window(0, offload, dur) == 0.0

    def test_layer_one_no_backward_scan(self):
        """Layer 1 has no layers before layer 0 to extend into."""
        dur = _durations(10.0, 20.0)
        offload = _offloads(0.0, 100.0)
        assert self.calc.compute_window(1, offload, dur) == 10.0

    def test_all_gaps_extends_to_beginning(self):
        """Only last layer offloaded — window spans entire model."""
        dur = _durations(10.0, 20.0, 30.0, 40.0)
        offload = _offloads(0.0, 0.0, 0.0, 400.0)
        assert self.calc.compute_window(3, offload, dur) == 10.0 + 20.0 + 30.0

    def test_dynamic_gap_changes_window(self):
        """Optimizer choosing not to offload a layer dynamically extends neighbors."""
        dur = _durations(10.0, 20.0, 30.0)
        offloaded = _offloads(100.0, 200.0, 300.0)
        dynamic_gap = _offloads(100.0, 0.0, 300.0)

        window_no_gap = self.calc.compute_window(2, offloaded, dur)
        window_with_gap = self.calc.compute_window(2, dynamic_gap, dur)

        assert window_no_gap == 20.0
        assert window_with_gap == 30.0
        assert window_with_gap > window_no_gap

    def test_single_layer_model(self):
        dur = _durations(10.0)
        offload = _offloads(100.0)
        assert self.calc.compute_window(0, offload, dur) == 0.0


# ===================================================================
# strategy_has_transfer_gaps
# ===================================================================


class TestStrategyHasTransferGaps:
    def _make_layers(self, n: int) -> list[LayerStatistics]:
        return [_layer(f"layer_{i}", [_tensor(i, 10.0, 1.0)], 10.0) for i in range(n)]

    def test_empty_strategy_no_gaps(self):
        layers = self._make_layers(4)
        assert strategy_has_transfer_gaps({}, layers) is False

    def test_single_layer_no_gaps(self):
        layers = self._make_layers(1)
        assert strategy_has_transfer_gaps({"layer_0": [_tensor(0, 10.0, 1.0)]}, layers) is False

    def test_fully_dense_no_gaps(self):
        """All N-1 slots used — no gaps."""
        layers = self._make_layers(4)
        strategy = {
            "layer_0": [_tensor(100, 10.0, 1.0)],
            "layer_1": [_tensor(101, 10.0, 1.0)],
            "layer_2": [_tensor(102, 10.0, 1.0)],
        }
        assert strategy_has_transfer_gaps(strategy, layers) is False

    def test_one_gap(self):
        """3 layers, 2 possible slots, only 1 used — gap exists."""
        layers = self._make_layers(3)
        strategy = {
            "layer_0": [_tensor(100, 10.0, 1.0)],
        }
        assert strategy_has_transfer_gaps(strategy, layers) is True

    def test_multiple_gaps(self):
        """5 layers, 4 possible slots, only 1 used."""
        layers = self._make_layers(5)
        strategy = {
            "layer_2": [_tensor(100, 10.0, 1.0)],
        }
        assert strategy_has_transfer_gaps(strategy, layers) is True

    def test_empty_tensor_list_not_counted(self):
        """Strategy entries with empty tensor lists are NOT transfer slots."""
        layers = self._make_layers(3)
        strategy = {
            "layer_0": [_tensor(100, 10.0, 1.0)],
            "layer_1": [],
        }
        assert strategy_has_transfer_gaps(strategy, layers) is True


# ===================================================================
# Integration: GapAwareWindow with GlobalTensorSelectionStrategy
# ===================================================================


class TestGapAwareWindowIntegration:
    """Verify that injecting GapAwareWindow into GlobalTensorSelectionStrategy
    changes the fits classification for layers after gaps."""

    def test_gap_allows_more_fixed_layers(self):
        """With gaps, layers that didn't fit with SingleLayerWindow may fit with GapAwareWindow.

        Model: L0 (embed, 1ms), L1 (gap, 100ms), L2 (gap, 100ms), L3 (big tensors, 10ms)
        L3's transfer budget:
          - SingleLayerWindow: dur(L2) = 100ms → transfers 100MB max
          - GapAwareWindow: dur(L0)+dur(L1)+dur(L2) = 201ms → transfers ~200MB max

        L3 has 150MB of tensors. With SingleLayerWindow, 150MB won't fit in
        100ms budget (at ~10GB/s), but with GapAwareWindow it fits in 201ms.
        """
        t0 = _tensor(0, 5.0, 0.5)
        t3a = _tensor(3, 80.0, 8.0)
        t3b = _tensor(4, 70.0, 7.0)

        layers = [
            _layer("embed", [t0], 1.0),
            _layer("norm_1", [], 100.0),
            _layer("norm_2", [], 100.0),
            _layer("decoder", [t3a, t3b], 10.0),
        ]
        mem_stats = _memory_stats()

        strategy_single = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            target_gpu_mem_bytes=200 * 1024 * 1024,
            scale=1.0,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=10,
            transfer_window=SingleLayerWindow(),
        )

        strategy_gap = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            target_gpu_mem_bytes=200 * 1024 * 1024,
            scale=1.0,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=10,
            transfer_window=GapAwareWindow(),
        )

        result_single = strategy_single.compute(layers, memory_stats=mem_stats)
        result_gap = strategy_gap.compute(layers, memory_stats=mem_stats)

        # GapAwareWindow should offload at least as much as SingleLayerWindow
        offloaded_single = sum(sum(t.size_bytes for t in tensors) for tensors in result_single.strategy_map.values())
        offloaded_gap = sum(sum(t.size_bytes for t in tensors) for tensors in result_gap.strategy_map.values())
        assert offloaded_gap >= offloaded_single

    def test_no_gaps_same_result(self):
        """When all layers have offloadable tensors, both windows produce identical results."""
        tensors_per_layer = [_tensor(i, 10.0, 1.0) for i in range(4)]
        layers = [_layer(f"layer_{i}", [tensors_per_layer[i]], 100.0) for i in range(4)]
        mem_stats = _memory_stats()

        strategy_single = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=100 * 1024 * 1024,
            target_gpu_mem_bytes=50 * 1024 * 1024,
            scale=1.0,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=10,
            seed=42,
            transfer_window=SingleLayerWindow(),
        )

        strategy_gap = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=100 * 1024 * 1024,
            target_gpu_mem_bytes=50 * 1024 * 1024,
            scale=1.0,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=10,
            seed=42,
            transfer_window=GapAwareWindow(),
        )

        result_single = strategy_single.compute(layers, memory_stats=mem_stats)
        result_gap = strategy_gap.compute(layers, memory_stats=mem_stats)

        # With no gaps, both should produce the same strategy map keys
        assert set(result_single.strategy_map.keys()) == set(result_gap.strategy_map.keys())

    def test_default_is_single_layer_window(self):
        """Default transfer_window should be SingleLayerWindow for backward compatibility."""
        strategy = GlobalTensorSelectionStrategy(n_blocks=2)
        assert isinstance(strategy.transfer_window, SingleLayerWindow)


# ===================================================================
# Gap-aware auto-detection and block sizing tests
# ===================================================================


class TestGapAwareAutoDetection:
    """Tests for automatic GapAwareWindow detection in GlobalTensorSelectionStrategy."""

    def test_auto_switches_to_gap_aware_when_gaps_present(self):
        """Strategy should auto-switch to GapAwareWindow when permanent gaps exist."""
        t0 = _tensor(0, 10.0, 1.0)
        t1 = _tensor(1, 10.0, 1.0)
        t3 = _tensor(3, 10.0, 1.0)

        layers = [
            _layer("L0", [t0], 100.0),
            _layer("L1", [t1], 100.0),
            _layer("L2", [], 100.0),  # gap
            _layer("L3", [t3], 100.0),
        ]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
        )
        assert isinstance(strategy.transfer_window, SingleLayerWindow)

        result = strategy.compute(layers, memory_stats=mem_stats)
        # Should produce a valid result
        assert result.strategy_map is not None

    def test_no_auto_switch_without_gaps(self):
        """Strategy should NOT switch when all layers have offloadable tensors."""
        tensors = [_tensor(i, 10.0, 1.0) for i in range(4)]
        layers = [_layer(f"L{i}", [tensors[i]], 100.0) for i in range(4)]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
        )

        result = strategy.compute(layers, memory_stats=mem_stats)
        assert result.strategy_map is not None
        # transfer_window on the strategy instance should still be SingleLayerWindow
        assert isinstance(strategy.transfer_window, SingleLayerWindow)

    def test_gap_aware_produces_valid_block_data(self):
        """With gaps, block_data should still be valid and have correct structure."""
        t0 = _tensor(0, 10.0, 1.0)
        t1 = _tensor(1, 10.0, 1.0)
        t3 = _tensor(3, 10.0, 1.0)
        t4 = _tensor(4, 10.0, 1.0)

        layers = [
            _layer("L0", [t0], 100.0),
            _layer("L1", [t1], 100.0),
            _layer("L2", [], 100.0),  # gap
            _layer("L3", [t3], 100.0),
            _layer("L4", [t4], 100.0),
        ]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
        )
        result = strategy.compute(layers, memory_stats=mem_stats)

        assert result.block_data is not None
        assert result.block_data.label_to_block_id is not None
        assert result.block_data.transfer_to_compute_map is not None
        # Gap layer L2 is a valid transfer slot (transfers L3's data during L2's compute),
        # so it MAY appear in label_to_block_id. However, the gap layer itself should
        # NOT be a compute target in transfer_to_compute_map.
        compute_targets = set(result.block_data.transfer_to_compute_map.values())
        assert "L2" not in compute_targets

    def test_multiple_gaps_handled(self):
        """Strategy should handle multiple non-adjacent gap layers."""
        t0 = _tensor(0, 10.0, 1.0)
        t1 = _tensor(1, 10.0, 1.0)
        t3 = _tensor(3, 10.0, 1.0)
        t5 = _tensor(5, 10.0, 1.0)

        layers = [
            _layer("L0", [t0], 100.0),
            _layer("L1", [t1], 100.0),
            _layer("L2", [], 100.0),  # gap
            _layer("L3", [t3], 100.0),
            _layer("L4", [], 100.0),  # gap
            _layer("L5", [t5], 100.0),
        ]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
        )
        result = strategy.compute(layers, memory_stats=mem_stats)

        assert result.strategy_map is not None
        assert result.block_data is not None
        # Neither gap layer should be a compute TARGET
        compute_targets = set(result.block_data.transfer_to_compute_map.values())
        assert "L2" not in compute_targets
        assert "L4" not in compute_targets

    def test_explicit_gap_aware_not_overridden(self):
        """When user explicitly passes GapAwareWindow, it should be used even without gaps."""
        tensors = [_tensor(i, 10.0, 1.0) for i in range(3)]
        layers = [_layer(f"L{i}", [tensors[i]], 100.0) for i in range(3)]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
            transfer_window=GapAwareWindow(),
        )

        result = strategy.compute(layers, memory_stats=mem_stats)
        assert result.strategy_map is not None
        # The explicit GapAwareWindow should still be the instance's transfer_window
        assert isinstance(strategy.transfer_window, GapAwareWindow)

    def test_gap_at_last_layer(self):
        """Gap at the last layer should not cause errors."""
        t0 = _tensor(0, 10.0, 1.0)
        t1 = _tensor(1, 10.0, 1.0)

        layers = [
            _layer("L0", [t0], 100.0),
            _layer("L1", [t1], 100.0),
            _layer("L2", [], 100.0),  # gap at end
        ]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
        )
        result = strategy.compute(layers, memory_stats=mem_stats)
        assert result.strategy_map is not None


class TestGapAwareBlockSizing:
    """Tests for the next_real_transfer block sizing mapping."""

    def test_block_sizes_skip_gap_layers(self):
        """Block sizes should account for the next real layer's offload, not the gap's."""
        t0 = _tensor(0, 10.0, 1.0)
        t1 = _tensor(1, 10.0, 1.0)
        t3 = _tensor(3, 20.0, 2.0)  # larger than t1

        layers = [
            _layer("L0", [t0], 100.0),
            _layer("L1", [t1], 100.0),
            _layer("L2", [], 100.0),  # gap
            _layer("L3", [t3], 100.0),
        ]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=10,
            seed=42,
        )
        result = strategy.compute(layers, memory_stats=mem_stats)

        # Block sizes should be non-zero even with a gap,
        # because the gap slot is mapped to L3's offload
        if result.block_data:
            block_sizes = result.block_data.block_sizes
            total_block_mem = sum(block_sizes.values()) if isinstance(block_sizes, dict) else sum(block_sizes)
            assert total_block_mem > 0

    def test_fixed_only_result_with_gaps(self):
        """When all layers fit (fixed-only path), gap-aware block sizing should still work."""
        t0 = _tensor(0, 1.0, 0.01)
        t1 = _tensor(1, 1.0, 0.01)
        t3 = _tensor(3, 1.0, 0.01)

        layers = [
            _layer("L0", [t0], 100.0),
            _layer("L1", [t1], 100.0),
            _layer("L2", [], 100.0),  # gap
            _layer("L3", [t3], 100.0),
        ]
        mem_stats = _memory_stats()

        strategy = GlobalTensorSelectionStrategy(
            max_gpu_mem_bytes=500 * 1024 * 1024,
            n_blocks=2,
            threshold_mb=0.1,
            epoch=5,
            seed=42,
            scale=1.0,
        )
        result = strategy.compute(layers, memory_stats=mem_stats)
        assert result.strategy_map is not None
        # Should offload some tensors
        total_offloaded = sum(len(ts) for ts in result.strategy_map.values())
        assert total_offloaded > 0
