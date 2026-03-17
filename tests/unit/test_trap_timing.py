# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for CUDA event timing in Trap, WarmupTrap, TrapDirect, and StatsTrap."""

from typing import ClassVar
from unittest.mock import Mock, call

import pytest
import torch

from flextensor.helpers import StatsTrap, TrapNestingGuard
from flextensor.tensor_manager import TensorManager
from flextensor.trap_tensor_mode import Trap, TrapDirect, WarmupTrap


def _make_mock_event(elapsed_time_ms: float = 2.5) -> Mock:
    event = Mock(spec=torch.cuda.Event)
    event.elapsed_time.return_value = elapsed_time_ms
    return event


def _make_mock_tensor_manager(
    *,
    elapsed_time_ms: float = 2.5,
    with_loader: bool = True,
    with_module_tracker: bool = False,
    use_spec: bool = False,
) -> Mock:
    start_event = _make_mock_event(elapsed_time_ms)
    end_event = _make_mock_event(elapsed_time_ms)
    start_event.elapsed_time.return_value = elapsed_time_ms

    tm = Mock(spec=TensorManager) if use_spec else Mock()
    tm.trap_start_event = start_event
    tm.trap_end_event = end_event
    tm.trap_nesting_guard = TrapNestingGuard()
    tm.layer_statistics_collector = Mock()
    tm.module_tracker = Mock() if with_module_tracker else None
    tm.is_traced.return_value = False

    if with_loader:
        tm.tensor_layer_loader = Mock()
        tm.tensor_layer_loader.get.return_value = None

    return tm


class TestTrapTiming:
    """Test CUDA event timing in Trap (TorchFunctionMode-based profile trap)."""

    def test_events_from_tensor_manager(self):
        tm = _make_mock_tensor_manager()
        trap = Trap(tm, "layer.0", torch.device("cuda:0"))

        assert trap.start_event is tm.trap_start_event
        assert trap.end_event is tm.trap_end_event

    def test_record_and_sync_ordering(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=3.0)
        trap = Trap(tm, "layer.0", torch.device("cuda:0"))

        call_order = []
        tm.trap_start_event.record.side_effect = lambda: call_order.append("start.record")
        tm.trap_end_event.record.side_effect = lambda: call_order.append("end.record")
        tm.trap_end_event.synchronize.side_effect = lambda: call_order.append("end.synchronize")
        tm.trap_start_event.elapsed_time.side_effect = lambda e: (call_order.append("elapsed_time"), 3.0)[1]

        trap.__enter__()
        trap.__exit__(None, None, None)

        assert call_order == ["start.record", "end.record", "end.synchronize", "elapsed_time"]

    def test_duration_flows_to_collector(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=4.2)
        trap = Trap(tm, "layer.5", torch.device("cuda:0"))

        trap.__enter__()
        trap.__exit__(None, None, None)

        tm.layer_statistics_collector.add_all.assert_called_once_with("layer.5", set(), 4.2)

    def test_uses_event_sync_not_global_sync(self):
        tm = _make_mock_tensor_manager()
        trap = Trap(tm, "layer.0", torch.device("cuda:0"))

        trap.__enter__()
        trap.__exit__(None, None, None)

        tm.trap_end_event.synchronize.assert_called_once()

    def test_tensor_layer_loader_enter_exit(self):
        tm = _make_mock_tensor_manager()
        trap = Trap(tm, "layer.0", torch.device("cuda:0"))

        trap.__enter__()
        tm.tensor_layer_loader.enter.assert_called_once_with("layer.0")

        trap.__exit__(None, None, None)
        tm.tensor_layer_loader.exit.assert_called_once_with("layer.0")


class TestWarmupTrapTiming:
    """Test CUDA event timing in WarmupTrap."""

    def test_events_from_tensor_manager(self):
        tm = _make_mock_tensor_manager(with_loader=False)
        trap = WarmupTrap(tm, "layer.0", torch.device("cuda:0"))

        assert trap.start_event is tm.trap_start_event
        assert trap.end_event is tm.trap_end_event

    def test_record_and_sync_ordering(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=1.0, with_loader=False)

        call_order = []
        tm.trap_start_event.record.side_effect = lambda: call_order.append("start.record")
        tm.trap_end_event.record.side_effect = lambda: call_order.append("end.record")
        tm.trap_end_event.synchronize.side_effect = lambda: call_order.append("end.synchronize")
        tm.trap_start_event.elapsed_time.side_effect = lambda e: (call_order.append("elapsed_time"), 1.0)[1]

        trap = WarmupTrap(tm, "layer.0", torch.device("cuda:0"))
        trap.__enter__()
        trap.__exit__(None, None, None)

        assert call_order == ["start.record", "end.record", "end.synchronize", "elapsed_time"]

    def test_duration_flows_to_collector(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=7.3, with_loader=False)
        trap = WarmupTrap(tm, "encoder.2", torch.device("cuda:0"))

        trap.__enter__()
        trap.__exit__(None, None, None)

        tm.layer_statistics_collector.add_all.assert_called_once_with("encoder.2", set(), 7.3)

    def test_module_tracker_enter_exit(self):
        tm = _make_mock_tensor_manager(with_loader=False, with_module_tracker=True)
        trap = WarmupTrap(tm, "layer.0", torch.device("cuda:0"))

        trap.__enter__()
        tm.module_tracker.enter_trap.assert_called_once_with("layer.0")

        trap.__exit__(None, None, None)
        tm.module_tracker.exit_trap.assert_called_once_with("layer.0")

    def test_no_module_tracker(self):
        tm = _make_mock_tensor_manager(with_loader=False, with_module_tracker=False)
        trap = WarmupTrap(tm, "layer.0", torch.device("cuda:0"))

        trap.__enter__()
        trap.__exit__(None, None, None)

        tm.layer_statistics_collector.add_all.assert_called_once()


class TestTrapDirectTiming:
    """Test CUDA event timing in TrapDirect (plain context manager profile trap)."""

    def test_events_from_tensor_manager(self):
        tm = _make_mock_tensor_manager()
        trap = TrapDirect(tm, "layer.0", torch.device("cuda:0"))

        assert trap.start_event is tm.trap_start_event
        assert trap.end_event is tm.trap_end_event

    def test_record_and_sync_ordering(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=5.0)

        call_order = []
        tm.trap_start_event.record.side_effect = lambda: call_order.append("start.record")
        tm.trap_end_event.record.side_effect = lambda: call_order.append("end.record")
        tm.trap_end_event.synchronize.side_effect = lambda: call_order.append("end.synchronize")
        tm.trap_start_event.elapsed_time.side_effect = lambda e: (call_order.append("elapsed_time"), 5.0)[1]

        trap = TrapDirect(tm, "layer.0", torch.device("cuda:0"))
        trap.__enter__()
        trap.__exit__(None, None, None)

        assert call_order == ["start.record", "end.record", "end.synchronize", "elapsed_time"]

    def test_duration_flows_to_add_duration(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=6.1)
        trap = TrapDirect(tm, "attn.3", torch.device("cuda:0"))

        trap.__enter__()
        trap.__exit__(None, None, None)

        tm.layer_statistics_collector.add_duration.assert_called_once_with("attn.3", 6.1)

    def test_tensor_layer_loader_enter_exit(self):
        tm = _make_mock_tensor_manager()
        trap = TrapDirect(tm, "layer.0", torch.device("cuda:0"))

        trap.__enter__()
        tm.tensor_layer_loader.enter.assert_called_once_with("layer.0")

        trap.__exit__(None, None, None)
        tm.tensor_layer_loader.exit.assert_called_once_with("layer.0")


class TestStatsTrapTiming:
    """Test CUDA event timing in StatsTrap (helpers.py)."""

    def test_events_from_tensor_manager(self):
        tm = _make_mock_tensor_manager(use_spec=True)
        tm.traps_direct_duration_ms = 0.0
        tm.traps_direct_stats = {}
        trap = StatsTrap(tm, "layer.0")

        assert trap.start_event is tm.trap_start_event
        assert trap.end_event is tm.trap_end_event

    def test_record_and_sync_ordering(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=2.0, use_spec=True)
        tm.traps_direct_duration_ms = 0.0
        tm.traps_direct_stats = {}

        call_order = []
        tm.trap_start_event.record.side_effect = lambda: call_order.append("start.record")
        tm.trap_end_event.record.side_effect = lambda: call_order.append("end.record")
        tm.trap_end_event.synchronize.side_effect = lambda: call_order.append("end.synchronize")
        tm.trap_start_event.elapsed_time.side_effect = lambda e: (call_order.append("elapsed_time"), 2.0)[1]

        trap = StatsTrap(tm, "layer.0")
        trap.__enter__()
        trap.__exit__(None, None, None)

        assert call_order == ["start.record", "end.record", "end.synchronize", "elapsed_time"]

    def test_duration_accumulates(self):
        tm = _make_mock_tensor_manager(elapsed_time_ms=3.5, use_spec=True)
        tm.traps_direct_duration_ms = 10.0
        tm.traps_direct_stats = {}

        trap = StatsTrap(tm, "mlp.1")
        trap.__enter__()
        trap.__exit__(None, None, None)

        assert tm.traps_direct_duration_ms == 13.5
        assert tm.traps_direct_stats["mlp.1"] == 3.5


class TestMultiLayerTraps:
    """Simulate sequential trap usage across multiple layers, verifying event reuse and per-layer duration."""

    LAYER_NAMES: ClassVar[list[str]] = ["model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3"]
    LAYER_DURATIONS: ClassVar[list[float]] = [1.1, 2.2, 3.3, 4.4]

    def _run_layers(self, trap_cls, tm, device):
        """Enter/exit a fresh trap for each layer, returning per-layer durations from elapsed_time."""
        durations = iter(self.LAYER_DURATIONS)

        def _next_elapsed(_end_event):
            return next(durations)

        tm.trap_start_event.elapsed_time.side_effect = _next_elapsed

        for name in self.LAYER_NAMES:
            trap = trap_cls(tm, name, device)
            trap.__enter__()
            trap.__exit__(None, None, None)

    def test_trap_direct_multi_layer(self):
        tm = _make_mock_tensor_manager()
        device = torch.device("cuda:0")

        self._run_layers(TrapDirect, tm, device)

        assert tm.trap_start_event.record.call_count == len(self.LAYER_NAMES)
        assert tm.trap_end_event.record.call_count == len(self.LAYER_NAMES)
        assert tm.trap_end_event.synchronize.call_count == len(self.LAYER_NAMES)

        expected_calls = [call(name, dur) for name, dur in zip(self.LAYER_NAMES, self.LAYER_DURATIONS, strict=False)]
        tm.layer_statistics_collector.add_duration.assert_has_calls(expected_calls)

    def test_warmup_trap_multi_layer(self):
        tm = _make_mock_tensor_manager(with_loader=False)
        device = torch.device("cuda:0")

        self._run_layers(WarmupTrap, tm, device)

        assert tm.trap_start_event.record.call_count == len(self.LAYER_NAMES)
        assert tm.trap_end_event.record.call_count == len(self.LAYER_NAMES)

        expected_calls = [
            call(name, set(), dur) for name, dur in zip(self.LAYER_NAMES, self.LAYER_DURATIONS, strict=False)
        ]
        tm.layer_statistics_collector.add_all.assert_has_calls(expected_calls)

    def test_stats_trap_multi_layer_accumulates(self):
        tm = _make_mock_tensor_manager(use_spec=True)
        tm.traps_direct_duration_ms = 0.0
        tm.traps_direct_stats = {}

        durations = iter(self.LAYER_DURATIONS)
        tm.trap_start_event.elapsed_time.side_effect = lambda _e: next(durations)

        for name in self.LAYER_NAMES:
            trap = StatsTrap(tm, name)
            trap.__enter__()
            trap.__exit__(None, None, None)

        assert tm.traps_direct_duration_ms == pytest.approx(sum(self.LAYER_DURATIONS))
        for name, dur in zip(self.LAYER_NAMES, self.LAYER_DURATIONS, strict=False):
            assert tm.traps_direct_stats[name] == dur

    def test_events_identity_preserved_across_layers(self):
        """All traps created for different layers reference the same two event objects."""
        tm = _make_mock_tensor_manager()
        device = torch.device("cuda:0")

        traps = [TrapDirect(tm, name, device) for name in self.LAYER_NAMES]

        for trap in traps:
            assert trap.start_event is tm.trap_start_event
            assert trap.end_event is tm.trap_end_event


class TestEventReuse:
    """Verify that all trap types share the same event objects from TensorManager."""

    def test_all_traps_share_events(self):
        tm = _make_mock_tensor_manager(use_spec=True)
        tm.traps_direct_duration_ms = 0.0
        tm.traps_direct_stats = {}

        trap = Trap(tm, "layer.0", torch.device("cuda:0"))
        warmup = WarmupTrap(tm, "layer.0", torch.device("cuda:0"))
        direct = TrapDirect(tm, "layer.0", torch.device("cuda:0"))
        stats = StatsTrap(tm, "layer.0")

        assert trap.start_event is warmup.start_event is direct.start_event is stats.start_event
        assert trap.end_event is warmup.end_event is direct.end_event is stats.end_event


class TestTrapNestingGuard:
    """Unit tests for the TrapNestingGuard utility itself."""

    def test_acquire_sets_active(self):
        guard = TrapNestingGuard()

        guard.acquire("layer.0")

        assert guard._active is True

    def test_release_clears_active(self):
        guard = TrapNestingGuard()
        guard._active = True

        guard.release()

        assert guard._active is False

    def test_acquire_raises_when_already_active(self):
        guard = TrapNestingGuard()
        guard.acquire("layer.0")

        with pytest.raises(RuntimeError, match="Nested traps are not supported"):
            guard.acquire("attention.q")

    def test_error_message_contains_trace_id(self):
        guard = TrapNestingGuard()
        guard.acquire("layer.0")

        with pytest.raises(RuntimeError, match=r"model\.layers\.7"):
            guard.acquire("model.layers.7")

    def test_acquire_release_cycle(self):
        guard = TrapNestingGuard()

        guard.acquire("layer.0")
        assert guard._active is True
        guard.release()
        assert guard._active is False

    def test_sequential_acquire_release(self):
        guard = TrapNestingGuard()

        for name in ["layer.0", "layer.1", "layer.2"]:
            guard.acquire(name)
            guard.release()

        assert guard._active is False

    def test_second_acquire_without_release_raises(self):
        guard = TrapNestingGuard()

        guard.acquire("layer.0")
        with pytest.raises(RuntimeError, match=r"layer\.1"):
            guard.acquire("layer.1")
        guard.release()


class TestNestingGuard:
    """Verify that nested traps raise RuntimeError instead of silently corrupting timing."""

    DEVICE: ClassVar[torch.device] = torch.device("cuda:0")

    def test_trap_rejects_nesting(self):
        tm = _make_mock_tensor_manager()
        outer = Trap(tm, "layer.0", self.DEVICE)
        inner = Trap(tm, "layer.1", self.DEVICE)

        outer.__enter__()
        with pytest.raises(RuntimeError, match="Nested traps are not supported"):
            inner.__enter__()
        outer.__exit__(None, None, None)

    def test_warmup_trap_rejects_nesting(self):
        tm = _make_mock_tensor_manager(with_loader=False)
        outer = WarmupTrap(tm, "layer.0", self.DEVICE)
        inner = WarmupTrap(tm, "layer.1", self.DEVICE)

        outer.__enter__()
        with pytest.raises(RuntimeError, match="Nested traps are not supported"):
            inner.__enter__()
        outer.__exit__(None, None, None)

    def test_trap_direct_rejects_nesting(self):
        tm = _make_mock_tensor_manager()
        outer = TrapDirect(tm, "layer.0", self.DEVICE)
        inner = TrapDirect(tm, "layer.1", self.DEVICE)

        outer.__enter__()
        with pytest.raises(RuntimeError, match="Nested traps are not supported"):
            inner.__enter__()
        outer.__exit__(None, None, None)

    def test_stats_trap_rejects_nesting(self):
        tm = _make_mock_tensor_manager(use_spec=True)
        tm.traps_direct_duration_ms = 0.0
        tm.traps_direct_stats = {}
        outer = StatsTrap(tm, "layer.0")
        inner = StatsTrap(tm, "layer.1")

        outer.__enter__()
        with pytest.raises(RuntimeError, match="Nested traps are not supported"):
            inner.__enter__()
        outer.__exit__(None, None, None)

    def test_cross_type_nesting_rejected(self):
        """Nesting different trap types still triggers the guard."""
        tm = _make_mock_tensor_manager()
        outer = TrapDirect(tm, "layer.0", self.DEVICE)
        inner = Trap(tm, "layer.1", self.DEVICE)

        outer.__enter__()
        with pytest.raises(RuntimeError, match="Nested traps are not supported"):
            inner.__enter__()
        outer.__exit__(None, None, None)

    def test_error_message_includes_trap_name(self):
        tm = _make_mock_tensor_manager()
        outer = TrapDirect(tm, "layer.0", self.DEVICE)
        inner = TrapDirect(tm, "attention.q", self.DEVICE)

        outer.__enter__()
        with pytest.raises(RuntimeError, match=r"attention\.q"):
            inner.__enter__()
        outer.__exit__(None, None, None)

    def test_guard_cleared_after_exit(self):
        tm = _make_mock_tensor_manager()
        trap = TrapDirect(tm, "layer.0", self.DEVICE)

        assert tm.trap_nesting_guard._active is False
        trap.__enter__()
        assert tm.trap_nesting_guard._active is True
        trap.__exit__(None, None, None)
        assert tm.trap_nesting_guard._active is False

    def test_sequential_traps_allowed(self):
        """Sequential (non-nested) traps must work without error."""
        tm = _make_mock_tensor_manager()

        for name in ["layer.0", "layer.1", "layer.2"]:
            trap = TrapDirect(tm, name, self.DEVICE)
            trap.__enter__()
            trap.__exit__(None, None, None)

        assert tm.trap_nesting_guard._active is False
        assert tm.trap_start_event.record.call_count == 3
        assert tm.trap_end_event.record.call_count == 3
