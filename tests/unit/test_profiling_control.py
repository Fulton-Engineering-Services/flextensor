# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiling data control API.

Tests the public API for controlling profiling duration recording:
- ``clear_profiling_durations()`` — wipe accumulated durations
- ``suspend_profiling()`` / ``resume_profiling()`` — suppress / resume recording

Coverage:
1. TensorManager: flag ownership, record_all / record_duration facade
2. OffloadManager integration (phase-dependent counter behaviour)
3. Module-level convenience functions
4. End-to-end workflows through TensorManager
"""

from unittest.mock import MagicMock, patch

import pytest

import flextensor
from flextensor.collectors import (
    IterativeLayerStatistics,
    IterativeLayerStatisticsCollector,
    TensorStatistics,
)
from flextensor.helpers import NoOpTensorManager, ProfilingSuspender
from flextensor.offload_manager import (
    OFFLOAD_MANAGER_MAP,
    OffloadConfig,
    OffloadManager,
    OffloadPhase,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_manager_map():
    """Ensure each test starts with a clean global manager map."""
    OFFLOAD_MANAGER_MAP.clear()
    yield
    OFFLOAD_MANAGER_MAP.clear()


def _make_tm():
    """Create a TensorManager with a real collector, bypassing __init__."""
    from flextensor.tensor_manager import TensorManager

    with patch.object(TensorManager, "__init__", lambda self, *a, **kw: None):
        tm = TensorManager.__new__(TensorManager)
    tm.layer_statistics_collector = IterativeLayerStatisticsCollector()
    tm._profiling_suspender = ProfilingSuspender()
    return tm


# ---------------------------------------------------------------------------
# 1. TensorManager: flag, record_all, record_duration
# ---------------------------------------------------------------------------


class TestTensorManagerProfilingControl:
    """TensorManager owns the suspend counter and the record_* facade."""

    def test_initial_state_not_suspended(self):
        tm = _make_tm()
        assert not tm.is_profiling_suspended()

    def test_suspend_sets_flag(self):
        tm = _make_tm()
        tm.suspend_profiling()
        assert tm.is_profiling_suspended()

    def test_resume_clears_flag(self):
        tm = _make_tm()
        tm.suspend_profiling()
        tm.resume_profiling()
        assert not tm.is_profiling_suspended()

    def test_suspend_resume_is_reference_counted(self):
        """Nested suspensions only lift once every outstanding suspend is released."""
        tm = _make_tm()
        tm.suspend_profiling()
        tm.suspend_profiling()
        assert tm.is_profiling_suspended()
        tm.resume_profiling()
        assert tm.is_profiling_suspended(), "inner resume must not lift an outer suspension"
        tm.resume_profiling()
        assert not tm.is_profiling_suspended()

    def test_resume_without_suspend_raises(self):
        tm = _make_tm()
        with pytest.raises(RuntimeError, match="unbalanced"):
            tm.resume_profiling()
        assert not tm.is_profiling_suspended()

    def test_resume_past_zero_raises(self):
        tm = _make_tm()
        tm.suspend_profiling()
        tm.resume_profiling()
        with pytest.raises(RuntimeError, match="unbalanced"):
            tm.resume_profiling()
        assert not tm.is_profiling_suspended()

    def test_pause_profiling_context_manager(self):
        tm = _make_tm()
        with tm.pause_profiling() as x:
            assert x is None, "pause_profiling must yield None (see ProfilingSuspender.suspended)"
            assert tm.is_profiling_suspended()
            tm.record_duration("layer0", 999.0)
        assert not tm.is_profiling_suspended()
        tm.record_duration("layer0", 3.0)
        assert tm.layer_statistics_collector.duration_measurements["layer0"] == [3.0]

    def test_pause_profiling_releases_on_exception(self):
        tm = _make_tm()
        with pytest.raises(RuntimeError), tm.pause_profiling():
            assert tm.is_profiling_suspended()
            raise RuntimeError("boom")
        assert not tm.is_profiling_suspended()

    def test_pause_profiling_nests_with_raw_calls(self):
        """Context manager and raw suspend/resume compose via the shared counter."""
        tm = _make_tm()
        tm.suspend_profiling()
        with tm.pause_profiling():
            assert tm.is_profiling_suspended()
        assert tm.is_profiling_suspended(), "outer raw suspension must still be active"
        tm.resume_profiling()
        assert not tm.is_profiling_suspended()

    # --- record_all ---

    def test_record_all_records_both(self):
        tm = _make_tm()
        tm.record_all("layer0", {1, 2}, 5.0)
        c = tm.layer_statistics_collector
        assert c.tensor_measurements["layer0"] == [{1, 2}]
        assert c.duration_measurements["layer0"] == [5.0]

    def test_record_all_suppresses_both_when_suspended(self):
        """Suspended PROFILING calls drop tensor IDs *and* duration.

        Suppressing only durations would let paused warmup passes widen
        per-label tensor unions on data-dependent models (MoE / conditional
        branches), silently changing the offload strategy.
        """
        tm = _make_tm()
        tm.suspend_profiling()
        tm.record_all("layer0", {1, 2}, 999.0)
        c = tm.layer_statistics_collector
        assert "layer0" not in c.tensor_measurements
        assert "layer0" not in c.duration_measurements

    def test_record_all_resumes_both_after_resume(self):
        tm = _make_tm()
        tm.suspend_profiling()
        tm.record_all("layer0", {1}, 999.0)
        tm.resume_profiling()
        tm.record_all("layer0", {1}, 3.0)
        c = tm.layer_statistics_collector
        assert c.duration_measurements["layer0"] == [3.0]
        assert c.tensor_measurements["layer0"] == [{1}]

    def test_record_all_does_not_widen_tensor_set_when_suspended(self):
        """Regression: paused PROFILING pass must not widen per-label tensor sets.

        On data-dependent models, vLLM-style mixed-batch warmups wrapped in
        ``pause_profiling()`` may exercise a different parameter subset than
        the profiling iterations. ``record_all`` must be a complete no-op
        while suspended so those passes cannot leak tensor IDs into the
        strategy via ``collector.tensor_measurements``.
        """
        tm = _make_tm()

        tm.record_all("layer0", {1, 2}, 5.0)

        tm.suspend_profiling()
        tm.record_all("layer0", {3, 4}, 999.0)
        tm.resume_profiling()

        c = tm.layer_statistics_collector
        union = c.get_union_tensor_ids()
        assert union["layer0"] == {1, 2}
        assert c.duration_measurements["layer0"] == [5.0]

    def test_record_all_noop_without_collector(self):
        tm = _make_tm()
        tm.layer_statistics_collector = None
        tm.record_all("layer0", {1}, 5.0)

    def test_record_tensors_records_ids_only(self):
        """``record_tensors`` is the DISCOVERY recorder: tensor IDs, no duration."""
        tm = _make_tm()
        tm.record_tensors("layer0", {1, 2})
        c = tm.layer_statistics_collector
        assert c.tensor_measurements["layer0"] == [{1, 2}]
        assert "layer0" not in c.duration_measurements

    def test_record_tensors_ignores_suspension_by_default(self):
        """Default ``respect_suspension=False`` preserves DISCOVERY semantics.

        Discovery's tensor-to-layer mapping is a hard prerequisite for
        every later phase, so the default call (used by ``WarmupTrap``)
        must record even while suspended.
        """
        tm = _make_tm()
        tm.suspend_profiling()
        tm.record_tensors("layer0", {1, 2})
        c = tm.layer_statistics_collector
        assert c.tensor_measurements["layer0"] == [{1, 2}]

    def test_record_tensors_respects_suspension_when_opted_in(self):
        """``respect_suspension=True`` mirrors ``record_all``'s suspension gate.

        Used by ``Trap.__exit__`` on the tainted branch (PROFILING phase):
        a paused iteration must not widen per-layer tensor sets via the
        rescue path either, otherwise the suspension contract leaks on
        data-dependent models (MoE / conditional branches / mixed-batch
        shapes).
        """
        tm = _make_tm()
        tm.suspend_profiling()
        tm.record_tensors("layer0", {1, 2}, respect_suspension=True)
        c = tm.layer_statistics_collector
        assert "layer0" not in c.tensor_measurements

    def test_record_tensors_respects_suspension_records_when_not_suspended(self):
        """``respect_suspension=True`` is a no-op outside suspension."""
        tm = _make_tm()
        tm.record_tensors("layer0", {1, 2}, respect_suspension=True)
        c = tm.layer_statistics_collector
        assert c.tensor_measurements["layer0"] == [{1, 2}]

    def test_record_tensors_respect_suspension_does_not_widen_tensor_set(self):
        """Regression: opt-in suspension respect must not widen per-label sets.

        Companion to ``test_record_all_does_not_widen_tensor_set_when_suspended`` —
        pins the same invariant for the tainted-exit call site, which
        ``Trap.__exit__`` reaches via ``respect_suspension=True``.
        """
        tm = _make_tm()

        tm.record_tensors("layer0", {1, 2}, respect_suspension=True)

        tm.suspend_profiling()
        tm.record_tensors("layer0", {3, 4}, respect_suspension=True)
        tm.resume_profiling()

        c = tm.layer_statistics_collector
        union = c.get_union_tensor_ids()
        assert union["layer0"] == {1, 2}

    def test_record_tensors_noop_without_collector(self):
        tm = _make_tm()
        tm.layer_statistics_collector = None
        tm.record_tensors("layer0", {1})

    def test_record_tensors_noop_without_collector_with_respect_suspension(self):
        """The collector-None guard runs before the suspension check."""
        tm = _make_tm()
        tm.layer_statistics_collector = None
        tm.suspend_profiling()
        tm.record_tensors("layer0", {1}, respect_suspension=True)

    # --- record_duration ---

    def test_record_duration_records_when_not_suspended(self):
        tm = _make_tm()
        tm.record_duration("layer0", 5.0)
        assert tm.layer_statistics_collector.duration_measurements["layer0"] == [5.0]

    def test_record_duration_suppressed_when_suspended(self):
        tm = _make_tm()
        tm.suspend_profiling()
        tm.record_duration("layer0", 999.0)
        assert "layer0" not in tm.layer_statistics_collector.duration_measurements

    def test_record_duration_noop_without_collector(self):
        tm = _make_tm()
        tm.layer_statistics_collector = None
        tm.record_duration("layer0", 5.0)

    # --- clear ---

    def test_clear_wipes_durations(self):
        tm = _make_tm()
        tm.layer_statistics_collector.add_duration("x", 1.0)
        tm.clear_profiling_durations()
        assert tm.layer_statistics_collector.duration_measurements == {}

    def test_clear_noop_without_collector(self):
        tm = _make_tm()
        tm.layer_statistics_collector = None
        tm.clear_profiling_durations()

    def test_clear_works_while_suspended(self):
        tm = _make_tm()
        tm.record_duration("layer0", 1.0)
        tm.suspend_profiling()
        tm.clear_profiling_durations()
        tm.resume_profiling()
        assert tm.layer_statistics_collector.duration_measurements == {}

    # --- flag survives collector recreation ---

    def test_flag_survives_collector_recreation(self):
        tm = _make_tm()
        tm.suspend_profiling()
        tm.layer_statistics_collector = IterativeLayerStatisticsCollector()
        assert tm.is_profiling_suspended()
        tm.record_duration("layer0", 999.0)
        assert "layer0" not in tm.layer_statistics_collector.duration_measurements

    # --- unsampled labels are omitted from duration maps ---

    def test_get_median_duration_omits_label_when_suppressed(self):
        tm = _make_tm()
        tm.layer_statistics_collector.add_tensors("layer0", {1})
        tm.suspend_profiling()
        tm.record_duration("layer0", 99.0)
        tm.resume_profiling()
        medians = tm.layer_statistics_collector.get_median_duration_ms()
        assert "layer0" not in medians

    def test_get_min_duration_omits_label_when_suppressed(self):
        tm = _make_tm()
        tm.layer_statistics_collector.add_tensors("layer0", {1})
        tm.suspend_profiling()
        tm.record_duration("layer0", 99.0)
        tm.resume_profiling()
        mins = tm.layer_statistics_collector.get_min_duration_ms()
        assert "layer0" not in mins

    def test_get_layer_stats_sets_duration_none_when_unsampled(self):
        """Discovered-but-unmeasured labels appear in iterative stats with
        ``duration=None``; :func:`compute_layer_statistics` drops them before
        the strict ``LayerStatistics`` reaches strategy consumers."""
        tm = _make_tm()
        tm.layer_statistics_collector.add_tensors("layer0", {1})
        tm.suspend_profiling()
        tm.record_duration("layer0", 99.0)
        tm.resume_profiling()
        stats = tm.layer_statistics_collector.get_layer_stats()
        layer0_stats = [s for s in stats if s.label == "layer0"]
        assert len(layer0_stats) == 1
        assert layer0_stats[0].duration is None


# ---------------------------------------------------------------------------
# 1b. NoOpTensorManager: profiling-control no-op contract
# ---------------------------------------------------------------------------


class TestNoOpTensorManagerProfilingControl:
    """``NoOpTensorManager`` (used when ``OffloadConfig(enabled=False)``) must
    expose the same profiling-control surface as ``TensorManager`` but as
    no-ops.

    ``is_profiling_suspended() == False`` is the load-bearing invariant: it
    guarantees ``OffloadManager.update_state()`` does not freeze the
    iteration counter when offloading is disabled. A future regression that
    accidentally wires the real ``ProfilingSuspender`` into ``NoOpTensorManager``
    would silently freeze profiling counters in ``enabled=False`` mode and
    must fail one of these tests.
    """

    def _make_no_op(self):
        return NoOpTensorManager(device_gpu=None)

    def test_is_profiling_suspended_always_false(self):
        tm = self._make_no_op()
        assert tm.is_profiling_suspended() is False

    def test_suspend_does_not_change_state(self):
        tm = self._make_no_op()
        tm.suspend_profiling()
        assert tm.is_profiling_suspended() is False, (
            "NoOpTensorManager.suspend_profiling must not flip is_profiling_suspended"
        )

    def test_resume_does_not_raise_or_change_state(self):
        """No-op ``resume`` must tolerate being called without a prior
        ``suspend`` (no refcount, no unbalanced-resume error)."""
        tm = self._make_no_op()
        tm.resume_profiling()
        assert tm.is_profiling_suspended() is False

    def test_clear_profiling_durations_is_noop(self):
        tm = self._make_no_op()
        tm.clear_profiling_durations()

    def test_pause_profiling_yields_and_does_not_change_state(self):
        tm = self._make_no_op()
        with tm.pause_profiling() as x:
            assert x is None, "pause_profiling must yield None (see ProfilingSuspender.suspended)"
            assert tm.is_profiling_suspended() is False
        assert tm.is_profiling_suspended() is False


# ---------------------------------------------------------------------------
# 2. OffloadManager integration
# ---------------------------------------------------------------------------


class TestOffloadManagerProfilingControl:
    """OffloadManager delegates profiling control to TensorManager."""

    def _make_om(self, name="test"):
        om = OffloadManager(name)
        om._tensor_manager = MagicMock()
        om._tensor_manager.is_profiling_suspended.return_value = False
        return om

    def test_suspend_delegates(self):
        om = self._make_om()
        om.suspend_profiling()
        om._tensor_manager.suspend_profiling.assert_called_once()

    def test_resume_delegates(self):
        om = self._make_om()
        om.suspend_profiling()
        om.resume_profiling()
        om._tensor_manager.resume_profiling.assert_called_once()

    def test_clear_delegates(self):
        om = self._make_om()
        om.clear_profiling_durations()
        om._tensor_manager.clear_profiling_durations.assert_called_once()

    def test_clear_raises_without_tensor_manager(self):
        om = OffloadManager("test_noop")
        with pytest.raises(RuntimeError, match=r"clear_profiling_durations\(\) called before flextensor\.offload"):
            om.clear_profiling_durations()

    def test_suspend_raises_without_tensor_manager(self):
        om = OffloadManager("test_suspend_noop")
        with pytest.raises(RuntimeError, match=r"suspend_profiling\(\) called before flextensor\.offload"):
            om.suspend_profiling()

    def test_resume_raises_without_tensor_manager(self):
        om = OffloadManager("test_resume_noop")
        with pytest.raises(RuntimeError, match=r"resume_profiling\(\) called before flextensor\.offload"):
            om.resume_profiling()

    def test_pause_delegates(self):
        om = self._make_om()
        with om.pause_profiling():
            pass
        om._tensor_manager.pause_profiling.assert_called_once()

    def test_pause_yields_none(self):
        """``OffloadManager.pause_profiling`` delegates to
        ``TensorManager.pause_profiling``, which yields ``None`` (see
        ``ProfilingSuspender.suspended``). Pinned here too — using a real
        ``NoOpTensorManager`` because the ``MagicMock`` in
        :meth:`_make_om` would yield a sentinel rather than ``None`` — so
        a future change that introduces a yielded value must update all
        four context managers in lockstep.
        """
        om = OffloadManager("test_pause_yields_none")
        om._tensor_manager = NoOpTensorManager(device_gpu=None)
        with om.pause_profiling() as x:
            assert x is None

    def test_pause_raises_without_tensor_manager(self):
        """Must raise before yielding — body of the ``with`` must not run."""
        om = OffloadManager("test_pause_noop")
        with (
            pytest.raises(RuntimeError, match=r"pause_profiling\(\) called before flextensor\.offload"),
            om.pause_profiling(),
        ):
            pytest.fail("with-block body must not execute when no TensorManager is active")

    # --- update_state interaction ---

    def test_update_state_skips_counter_in_profiling_when_suspended(self):
        om = self._make_om()
        om._current_phase = OffloadPhase.PROFILING
        om.config = OffloadConfig(profiling_iters=5)
        om._iteration_count = 0
        om._tensor_manager.is_profiling_suspended.return_value = True

        om.update_state()
        assert om._iteration_count == 0

    def test_update_state_advances_counter_in_discovery_when_suspended(self):
        om = self._make_om()
        om._current_phase = OffloadPhase.DISCOVERY
        om.config = OffloadConfig(discovery_iters=100)
        om._iteration_count = 0
        om._tensor_manager.is_profiling_suspended.return_value = True

        om.update_state()
        assert om._iteration_count == 1

    def test_update_state_normal_when_not_suspended(self):
        om = self._make_om()
        om._current_phase = OffloadPhase.PROFILING
        om.config = OffloadConfig(profiling_iters=100)
        om._iteration_count = 0
        om._tensor_manager.is_profiling_suspended.return_value = False

        om.update_state()
        assert om._iteration_count == 1

    def test_update_state_noop_in_inference_regardless_of_suspend(self):
        om = self._make_om()
        om._current_phase = OffloadPhase.INFERENCE
        om._tensor_manager.is_profiling_suspended.return_value = True
        om._iteration_count = 42

        om.update_state()
        assert om._iteration_count == 42

    @pytest.mark.parametrize("phase", [OffloadPhase.DISCOVERY, OffloadPhase.PROFILING])
    def test_update_state_raises_when_tensor_manager_missing_in_active_phase(self, phase):
        om = OffloadManager("test_update_state_no_tm")
        om._current_phase = phase
        om.config = OffloadConfig(discovery_iters=5, profiling_iters=5)
        om._iteration_count = 7
        with pytest.raises(RuntimeError, match=rf"phase {phase.name}.*TensorManager is not set"):
            om.update_state()
        assert om._iteration_count == 7

    def test_suspended_profiling_does_not_trigger_transition(self):
        """Suspended iterations should not trigger the profiling->inference transition."""
        om = self._make_om()
        om._current_phase = OffloadPhase.PROFILING
        om.config = OffloadConfig(profiling_iters=2)
        om._iteration_count = 1
        om._tensor_manager.is_profiling_suspended.return_value = True

        om.update_state()
        assert om._current_phase == OffloadPhase.PROFILING
        assert om._iteration_count == 1

    def test_discovery_transition_still_fires_when_suspended(self):
        """Discovery counter advances even when suspended, so transition fires normally."""
        om = self._make_om()
        om._current_phase = OffloadPhase.DISCOVERY
        om.config = OffloadConfig(discovery_iters=1)
        om._iteration_count = 0
        om._tensor_manager.is_profiling_suspended.return_value = True

        with patch.object(om, "_transition_to_profile") as mock_transition:
            om.update_state()
        mock_transition.assert_called_once()

    def test_update_state_counter_re_advances_after_resume(self):
        """Integration: tick → suspend → tick (frozen) → resume → tick (advances).

        Pins the single-depth suspend / resume contract at the
        ``OffloadManager.update_state()`` layer using a real
        ``ProfilingSuspender`` wired into the mock tensor manager.
        Catches regressions where ``OffloadManager.resume_profiling()`` fails
        to delegate to ``TensorManager.resume_profiling()``, or where
        ``update_state()`` stops consulting ``is_profiling_suspended()``.
        Multi-depth refcount behaviour is covered by
        :meth:`test_update_state_counter_only_re_advances_after_all_suspends_released`.
        """
        om = OffloadManager("test_re_advance")
        suspender = ProfilingSuspender()
        tm = MagicMock()
        tm.is_profiling_suspended.side_effect = suspender.is_suspended
        tm.suspend_profiling.side_effect = suspender.suspend
        tm.resume_profiling.side_effect = suspender.resume
        om._tensor_manager = tm
        om._current_phase = OffloadPhase.PROFILING
        om.config = OffloadConfig(profiling_iters=100)
        om._iteration_count = 0

        om.update_state()
        assert om._iteration_count == 1, "counter must advance when not suspended"

        om.suspend_profiling()
        om.update_state()
        assert om._iteration_count == 1, "counter must freeze while suspended"

        om.resume_profiling()
        om.update_state()
        assert om._iteration_count == 2, "counter must re-advance after resume"

    def test_update_state_counter_only_re_advances_after_all_suspends_released(self):
        """Nested suspend/resume: counter must stay frozen until refcount hits 0.

        Pins the ``count > 1`` resume path through
        ``OffloadManager.update_state()``. A regression where
        ``resume_profiling()`` cleared the suspender flag at any
        ``count > 0`` (e.g. a flag-not-counter implementation) would let the
        counter advance after the *first* of two nested resumes — this test
        fails fast on that bug.
        """
        om = OffloadManager("test_re_advance_nested")
        suspender = ProfilingSuspender()
        tm = MagicMock()
        tm.is_profiling_suspended.side_effect = suspender.is_suspended
        tm.suspend_profiling.side_effect = suspender.suspend
        tm.resume_profiling.side_effect = suspender.resume
        om._tensor_manager = tm
        om._current_phase = OffloadPhase.PROFILING
        om.config = OffloadConfig(profiling_iters=100)
        om._iteration_count = 0

        om.suspend_profiling()
        om.suspend_profiling()
        om.update_state()
        assert om._iteration_count == 0, "counter must be frozen while both suspends are active"

        om.resume_profiling()
        om.update_state()
        assert om._iteration_count == 0, (
            "counter must stay frozen until the last outstanding suspend is released "
            "(refcount must be > 0 after one of two resumes)"
        )

        om.resume_profiling()
        om.update_state()
        assert om._iteration_count == 1, "counter must advance once the last suspend is released"


# ---------------------------------------------------------------------------
# 3. Module-level convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    """Tests for flextensor.{clear,suspend,resume,pause}_profiling()."""

    def test_clear_profiling_durations(self):
        om = flextensor.get_offload_manager("test_clear_mod")
        om._tensor_manager = MagicMock()
        flextensor.clear_profiling_durations(name="test_clear_mod")
        om._tensor_manager.clear_profiling_durations.assert_called_once()

    def test_suspend_resume_profiling(self):
        om = flextensor.get_offload_manager("test_sr_mod")
        om._tensor_manager = MagicMock()
        flextensor.suspend_profiling(name="test_sr_mod")
        om._tensor_manager.suspend_profiling.assert_called_once()
        flextensor.resume_profiling(name="test_sr_mod")
        om._tensor_manager.resume_profiling.assert_called_once()

    def test_pause_profiling_delegates(self):
        om = flextensor.get_offload_manager("test_pause_mod")
        om._tensor_manager = MagicMock()
        with flextensor.pause_profiling(name="test_pause_mod"):
            pass
        om._tensor_manager.pause_profiling.assert_called_once()


# ---------------------------------------------------------------------------
# 4. End-to-end workflows through TensorManager
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration-style tests using a real TensorManager + collector."""

    def test_suspend_prevents_junk_during_warmup(self):
        """Simulates a profile flow where a paused warmup pass is fully ignored.

        Both the tensor-IDs and the duration of the suspended pass are
        dropped; only the unsuspended PROFILING iterations contribute.
        """
        tm = _make_tm()

        tm.record_all("layer0", {1, 2}, 100.0)

        tm.suspend_profiling()
        tm.record_all("layer0", {1, 2}, 999.0)
        tm.resume_profiling()

        tm.record_all("layer0", {1, 2}, 5.0)

        c = tm.layer_statistics_collector
        assert c.duration_measurements["layer0"] == [100.0, 5.0]
        assert len(c.tensor_measurements["layer0"]) == 2

    def test_clear_then_suspend_workflow_tm(self):
        """Simulates clear existing junk + suspend for future warmup."""
        tm = _make_tm()

        tm.record_duration("layer0", 100.0)
        tm.clear_profiling_durations()

        tm.suspend_profiling()
        tm.record_duration("layer0", 200.0)
        tm.resume_profiling()

        tm.record_duration("layer0", 3.0)
        assert tm.layer_statistics_collector.duration_measurements["layer0"] == [3.0]

    def test_clear_then_suspend_workflow(self):
        """Simulates clear + suspend via OffloadManager imperative API."""
        om = OffloadManager("test_e2e_clear_suspend")
        tm = _make_tm()
        om._tensor_manager = tm

        tm.record_duration("layer0", 100.0)

        om.clear_profiling_durations()
        om.suspend_profiling()
        tm.record_duration("layer0", 999.0)
        om.resume_profiling()

        tm.record_duration("layer0", 3.0)
        assert tm.layer_statistics_collector.duration_measurements["layer0"] == [3.0]

    def test_multi_component_safe(self):
        """Suspended profiling preserves data from previously profiled components."""
        tm = _make_tm()

        tm.suspend_profiling()
        tm.record_all("comp_a", {1}, 999.0)
        tm.resume_profiling()
        tm.record_all("comp_a", {1}, 5.0)

        tm.suspend_profiling()
        tm.record_all("comp_b", {2}, 999.0)
        tm.resume_profiling()
        tm.record_all("comp_b", {2}, 7.0)

        c = tm.layer_statistics_collector
        assert c.duration_measurements["comp_a"] == [5.0]
        assert c.duration_measurements["comp_b"] == [7.0]


# ---------------------------------------------------------------------------
# 5. compute_layer_statistics boundary behaviour
# ---------------------------------------------------------------------------


class TestComputeLayerStatistics:
    """Direct tests for the iterative->strict stats boundary.

    ``compute_layer_statistics`` is the phase boundary where collector output
    (tolerant, ``duration`` may be ``None``) is narrowed to the strict
    ``LayerStatistics`` shape consumed by strategy code. The existing tests in
    this module cover the collector side of this boundary; these tests
    exercise the filter itself with mixed inputs.
    """

    @staticmethod
    def _tstat(tensor_id: int, size_bytes: int = 1024, load_time_ms: float = 0.5) -> TensorStatistics:
        return TensorStatistics(tensor_id=tensor_id, name="", size_bytes=size_bytes, load_time_ms=load_time_ms)

    def test_drops_entries_with_none_duration_silently(self, caplog):
        """Unsampled labels (``duration=None``) must be dropped silently.

        Dropping here is a legitimate outcome whenever a label receives
        tensor IDs without a paired duration sample — for example, a
        DISCOVERY-only ``record_tensors`` call (see
        ``TensorManager.record_tensors``) that no later PROFILING iteration
        re-traps. A WARNING from this helper would therefore spam on every
        legitimate occurrence. Visibility of untimed traps is instead
        consolidated into a single ``UntimedTrapsReport`` warning emitted
        by ``TensorManager.prepare_infer_mode`` when the report is
        non-empty, regardless of ``enable_diagnostics``.
        """
        from flextensor.tensor_manager import compute_layer_statistics

        iterative = [
            IterativeLayerStatistics(label="kept", tensor_ids={1}, duration=2.0),
            IterativeLayerStatistics(label="dropped", tensor_ids={2}, duration=None),
        ]
        tensor_statistics_map = {1: self._tstat(1), 2: self._tstat(2)}

        with caplog.at_level("WARNING", logger="flextensor.tensor_manager"):
            result = compute_layer_statistics(iterative, tensor_statistics_map)

        labels = [s.label for s in result]
        assert labels == ["kept"], "only labels with real durations must survive"
        assert caplog.records == [], (
            "compute_layer_statistics must not warn on drops; pause-only traps "
            "are an expected source of None-duration entries"
        )

    def test_preserves_insertion_order_across_mixed_input(self):
        """Kept labels should appear in the same order as the input."""
        from flextensor.tensor_manager import compute_layer_statistics

        iterative = [
            IterativeLayerStatistics(label="a", tensor_ids={1}, duration=1.0),
            IterativeLayerStatistics(label="skip", tensor_ids={2}, duration=None),
            IterativeLayerStatistics(label="b", tensor_ids={3}, duration=2.0),
            IterativeLayerStatistics(label="c", tensor_ids={4}, duration=3.0),
        ]
        tensor_statistics_map = {i: self._tstat(i) for i in (1, 2, 3, 4)}

        result = compute_layer_statistics(iterative, tensor_statistics_map)

        assert [s.label for s in result] == ["a", "b", "c"]
        assert [s.duration for s in result] == [1.0, 2.0, 3.0]

    def test_skips_tensor_ids_missing_from_statistics_map(self):
        """Tensor IDs without a statistics entry must be silently skipped,
        but the surrounding ``LayerStatistics`` entry is still emitted."""
        from flextensor.tensor_manager import compute_layer_statistics

        iterative = [
            IterativeLayerStatistics(label="partial", tensor_ids=[1, 2, 3], duration=4.0),
        ]
        # IDs 1 and 3 are known; 2 is missing on purpose.
        tensor_statistics_map = {
            1: self._tstat(1, size_bytes=100, load_time_ms=0.1),
            3: self._tstat(3, size_bytes=300, load_time_ms=0.3),
        }

        result = compute_layer_statistics(iterative, tensor_statistics_map)

        assert len(result) == 1
        kept = result[0]
        assert kept.label == "partial"
        assert kept.duration == 4.0
        kept_ids = {t.tensor_id for t in kept.tensors}
        assert kept_ids == {1, 3}, "unknown tensor IDs must be skipped, known ones kept"
        sizes = {t.tensor_id: t.size_bytes for t in kept.tensors}
        load_times = {t.tensor_id: t.load_time_ms for t in kept.tensors}
        assert sizes == {1: 100, 3: 300}
        assert load_times == {1: 0.1, 3: 0.3}

    def test_empty_input_returns_empty_list(self):
        from flextensor.tensor_manager import compute_layer_statistics

        assert compute_layer_statistics([], {}) == []

    def test_all_entries_dropped_returns_empty_list(self):
        """If every iterative entry is unsampled, the result is empty (not an error)."""
        from flextensor.tensor_manager import compute_layer_statistics

        iterative = [
            IterativeLayerStatistics(label="x", tensor_ids={1}, duration=None),
            IterativeLayerStatistics(label="y", tensor_ids={2}, duration=None),
        ]
        result = compute_layer_statistics(iterative, {1: self._tstat(1), 2: self._tstat(2)})
        assert result == []
