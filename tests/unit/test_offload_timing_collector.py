# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`flextensor.offload_timing.OffloadTimingCollector`.

Focus: the capture-region branch of ``on_pass_start``. Inside a CUDA-graph
capture region, ``_finalize_pass()`` must be skipped because its internal
``torch.cuda.synchronize()`` is illegal under capture; outside capture, the
existing finalize-on-active-pass behaviour must continue to hold so the
non-graph timing pipeline keeps working.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from flextensor.offload_manager import OffloadManager
from flextensor.offload_timing import OffloadTimingCollector, OffloadTimingSnapshot, TrapTimingRecord


@pytest.fixture()
def collector() -> OffloadTimingCollector:
    """A collector with no GPU events — we only exercise the branch logic.

    Patches ``is_current_stream_capturing`` so the unit suite stays
    CUDA-context-independent; capture tests override the patch explicitly.
    """
    with patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False):
        c = OffloadTimingCollector(trap_labels=[])
        c._external_events = True
        yield c


@pytest.fixture()
def disabled_collector() -> OffloadTimingCollector:
    return OffloadTimingCollector(enabled=False)


class TestDisabledOffloadTimingCollector:
    def test_on_pass_start_is_noop(self, disabled_collector: OffloadTimingCollector) -> None:
        with patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing") as capturing:
            disabled_collector.on_pass_start()
        capturing.assert_not_called()
        assert disabled_collector._pass_active is False

    def test_disabled_log_window_is_empty(self, disabled_collector: OffloadTimingCollector) -> None:
        """Disabled collectors never accumulate the rolling log window."""
        report = OffloadTimingCollector._build_report(disabled_collector._snapshots)
        assert report.num_passes == 0
        assert disabled_collector._snapshots == []

    def test_disabled_with_labels_is_true_noop(self) -> None:
        """Labels + enabled=False must not KeyError on hook calls (no event maps)."""
        stream = MagicMock(name="transfer")
        collector = OffloadTimingCollector(["L0", "L1"], enabled=False)
        collector.record_transfer_start("L0", stream)
        collector.record_transfer_end("L0", stream)
        collector.record_compute_start("L0")
        collector.record_compute_end("L0")
        collector.record_wait_start("L0")
        collector.record_wait_end("L0")
        assert collector._label_set == set()
        assert collector._transfer_start == {}


class TestHasFlagsRequireCompletedPairs:
    def test_start_without_end_does_not_mark_ready(self) -> None:
        """Partial abort after start must not allow elapsed_time on a missing end."""
        stream = MagicMock(name="transfer")
        start_ev = MagicMock(name="start")
        end_ev = MagicMock(name="end")

        with patch("flextensor.offload_timing._make_timing_cuda_event", side_effect=[start_ev, end_ev] * 3):
            collector = OffloadTimingCollector(["L0"], enabled=True)

        collector.record_transfer_start("L0", stream)
        collector.record_compute_start("L0")
        collector.record_wait_start("L0")
        assert collector._has_transfer["L0"] is False
        assert collector._has_compute["L0"] is False
        assert collector._has_wait["L0"] is False

        with (
            patch("flextensor.offload_timing.torch.cuda.synchronize"),
            patch.object(start_ev, "elapsed_time") as elapsed,
        ):
            snap = collector._read_pass_snapshot()
        elapsed.assert_not_called()
        assert snap.per_trap[0].transfer_ms == 0.0
        assert snap.per_trap[0].compute_ms == 0.0
        assert snap.per_trap[0].wait_ms == 0.0

    def test_end_marks_pair_ready(self) -> None:
        stream = MagicMock(name="transfer")
        events = [MagicMock(name=f"e{i}") for i in range(6)]
        with patch("flextensor.offload_timing._make_timing_cuda_event", side_effect=events):
            collector = OffloadTimingCollector(["L0"], enabled=True)

        collector.record_transfer_start("L0", stream)
        collector.record_transfer_end("L0", stream)
        collector.record_compute_start("L0")
        collector.record_compute_end("L0")
        collector.record_wait_start("L0")
        collector.record_wait_end("L0")
        assert collector._has_transfer["L0"] is True
        assert collector._has_compute["L0"] is True
        assert collector._has_wait["L0"] is True

    def test_read_pass_snapshot_maps_elapsed_pairs_to_record(self) -> None:
        """Each column must come from the matching start.elapsed_time(end) pair."""
        stream = MagicMock(name="transfer")
        t_start, t_end = MagicMock(name="t_start"), MagicMock(name="t_end")
        c_start, c_end = MagicMock(name="c_start"), MagicMock(name="c_end")
        w_start, w_end = MagicMock(name="w_start"), MagicMock(name="w_end")
        t_start.elapsed_time.return_value = 1.5
        c_start.elapsed_time.return_value = 2.5
        w_start.elapsed_time.return_value = 3.5

        # Constructor allocates events as transfer, compute, wait (start then end).
        with patch(
            "flextensor.offload_timing._make_timing_cuda_event",
            side_effect=[t_start, t_end, c_start, c_end, w_start, w_end],
        ):
            collector = OffloadTimingCollector(["L0"], enabled=True)

        collector.record_transfer_start("L0", stream)
        collector.record_transfer_end("L0", stream)
        collector.record_compute_start("L0")
        collector.record_compute_end("L0")
        collector.record_wait_start("L0")
        collector.record_wait_end("L0")

        with patch("flextensor.offload_timing.torch.cuda.synchronize"):
            snap = collector._read_pass_snapshot()

        t_start.elapsed_time.assert_called_once_with(t_end)
        c_start.elapsed_time.assert_called_once_with(c_end)
        w_start.elapsed_time.assert_called_once_with(w_end)
        assert snap.per_trap == (TrapTimingRecord(label="L0", transfer_ms=1.5, compute_ms=2.5, wait_ms=3.5),)


class TestOnPassStartCaptureBranch:
    """``on_pass_start`` must defer finalize when stream is capturing."""

    def test_skips_finalize_when_capturing(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(collector, "_finalize_pass") as finalize,
        ):
            collector.on_pass_start()

        finalize.assert_not_called()
        assert collector._pass_active is True
        assert collector._pass_was_captured is True

    def test_calls_finalize_outside_capture(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_finalize_pass") as finalize,
        ):
            collector.on_pass_start()

        finalize.assert_called_once_with()
        assert collector._pass_active is True
        assert collector._pass_was_captured is False

    def test_first_pass_never_finalizes(self, collector: OffloadTimingCollector) -> None:
        """On the very first pass (``_pass_active=False``) finalize must not
        be called, regardless of capture state — there is no prior pass to
        finalize. Guards against a regression that always calls finalize."""
        collector._pass_active = False
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_finalize_pass") as finalize,
        ):
            collector.on_pass_start()

        finalize.assert_not_called()
        assert collector._pass_active is True

    def test_skips_finalize_when_prior_pass_was_captured(
        self, collector: OffloadTimingCollector, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The captured-then-eager interleaving (vLLM's warmup-and-capture
        loop) must not call ``_finalize_pass`` — its ``elapsed_time`` read
        on captured-and-replayed internal events returns ``cudaErrorInvalidValue``
        or stale capture values. Instead, ``_reset_pass_state`` clears
        bookkeeping and the new pass starts fresh."""
        collector._pass_active = True
        collector._pass_was_captured = True
        collector._external_events = False
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_finalize_pass") as finalize,
            patch.object(collector, "_reset_pass_state", wraps=collector._reset_pass_state) as reset,
            caplog.at_level("WARNING"),
        ):
            collector.on_pass_start()

        finalize.assert_not_called()
        reset.assert_called_once_with()
        # ``_reset_pass_state`` clears the flag, then ``on_pass_start`` sets
        # ``_pass_active=True`` again for the fresh pass.
        assert collector._pass_active is True
        assert collector._pass_was_captured is False
        assert any("dropping captured pass timings" in r.message for r in caplog.records)
        # Warn once — second drop must not spam.
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_finalize_pass"),
        ):
            collector._pass_was_captured = True
            collector.on_pass_start()
        assert sum(1 for r in caplog.records if "dropping captured pass timings" in r.message) == 1

    def test_finalizes_when_prior_pass_was_captured_with_external_events(
        self, collector: OffloadTimingCollector
    ) -> None:
        """External+captured uses finalize_replay_pass so _has_* survive replan."""
        collector._pass_active = True
        collector._pass_was_captured = True
        collector._external_events = True
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "finalize_replay_pass", return_value=True) as finalize_replay,
            patch.object(collector, "_finalize_pass") as finalize,
            patch.object(collector, "_reset_pass_state") as reset,
        ):
            collector.on_pass_start()

        finalize_replay.assert_called_once_with()
        finalize.assert_not_called()
        reset.assert_not_called()
        assert collector._pass_active is True

    def test_resets_when_external_captured_finalize_replay_refuses(self, collector: OffloadTimingCollector) -> None:
        """Refused replay finalize must not leave sticky captured+active state."""
        collector._pass_active = True
        collector._pass_was_captured = True
        collector._external_events = True
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "finalize_replay_pass", return_value=False) as finalize_replay,
            patch.object(collector, "_finalize_pass") as finalize,
            patch.object(collector, "_reset_pass_state", wraps=collector._reset_pass_state) as reset,
        ):
            collector.on_pass_start()

        finalize_replay.assert_called_once_with()
        finalize.assert_not_called()
        reset.assert_called_once_with()
        assert collector._pass_active is True
        assert collector._pass_was_captured is False

    def test_external_captured_on_pass_start_preserves_maps_through_reset(
        self, collector: OffloadTimingCollector
    ) -> None:
        """vLLM may enter traps after capture; maps must survive into replan."""
        collector._pass_active = True
        collector._pass_was_captured = True
        collector._external_events = True
        collector._has_compute["layer.0"] = True
        collector._has_transfer["layer.0"] = True
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_read_pass_snapshot") as read_snapshot,
        ):
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="layer.0", compute_ms=1.0, transfer_ms=0.5),),
            )
            collector.on_pass_start()

        assert collector._has_compute == {"layer.0": True}
        assert collector._has_transfer == {"layer.0": True}
        assert collector._pass_was_captured is True
        collector.reset()
        assert collector._has_compute == {"layer.0": True}
        collector.arm_replay_measure()
        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="layer.0", compute_ms=1.0),),
            )
            assert collector.finalize_replay_pass(replay_generation=1) is True


class TestResetForGraphReplan:
    def test_reset_clears_snapshots_but_keeps_replay_state(self, collector: OffloadTimingCollector) -> None:
        """Post-capture re-plan calls reset() then finalize_replay_pass()."""

        collector._pass_active = True
        collector._pass_was_captured = True
        collector._has_compute["layer.0"] = True
        collector._has_transfer["layer.0"] = True
        collector._has_wait["layer.0"] = True
        collector._last_finalized_replay_generation = 42
        collector._snapshots.append(OffloadTimingSnapshot(per_trap=()))

        collector.reset()

        assert collector._snapshots == []
        assert collector._last_finalized_replay_generation == -1
        assert collector._pass_active is True
        assert collector._pass_was_captured is True
        assert collector._has_compute == {"layer.0": True}
        assert collector._has_transfer == {"layer.0": True}
        assert collector._has_wait == {"layer.0": True}

    def test_finalize_replay_pass_works_after_reset(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        collector._pass_was_captured = True
        # Transfer-only proof that finalize accepts any recorded map, not just compute.
        collector._has_transfer["layer.0"] = True
        collector.reset()

        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="L0", transfer_ms=1.0),),
            )
            assert collector.finalize_replay_pass(replay_generation=1) is True

        read_snapshot.assert_called_once_with()
        assert len(collector._snapshots) == 1

    def test_finalize_replay_pass_returns_false_when_inactive(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = False
        assert collector.finalize_replay_pass(replay_generation=1) is False

    def test_finalize_replay_pass_same_gen_dedup_returns_true(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        collector._last_finalized_replay_generation = 3
        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            assert collector.finalize_replay_pass(replay_generation=3) is True
        read_snapshot.assert_not_called()

    def test_finalize_minus_one_dedups_after_gen_aware_publish(self, collector: OffloadTimingCollector) -> None:
        """Gen hook then update_state(-1) must not double-publish."""
        collector._pass_active = True
        collector._has_compute["layer.0"] = True
        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="L0", compute_ms=1.0),),
            )
            assert collector.finalize_replay_pass(replay_generation=5) is True
            assert collector.finalize_replay_pass(replay_generation=-1) is True
        assert read_snapshot.call_count == 1
        assert len(collector._snapshots) == 1

    def test_finalize_minus_one_still_publishes_without_prior_gen(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        collector._has_compute["layer.0"] = True
        collector._last_finalized_replay_generation = -1
        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="L0", compute_ms=1.0),),
            )
            assert collector.finalize_replay_pass(replay_generation=-1) is True
            assert collector.finalize_replay_pass(replay_generation=-1) is True
        # Pure -1 path (no gen hook): each call still publishes.
        assert read_snapshot.call_count == 2
        assert len(collector._snapshots) == 2

    def test_disarm_replay_measure_clears_only_pass_active(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        collector._pass_was_captured = True
        collector._has_compute["layer.0"] = True
        collector._has_transfer["layer.0"] = True
        collector._has_wait["layer.0"] = True
        collector.disarm_replay_measure()
        assert collector._pass_active is False
        # Keep recorded maps so re-arm + finalize can still publish real timings.
        assert collector._pass_was_captured is True
        assert collector._has_compute == {"layer.0": True}
        assert collector._has_transfer == {"layer.0": True}
        assert collector._has_wait == {"layer.0": True}

    def test_finalize_refuses_when_no_layer_events_recorded(
        self, collector: OffloadTimingCollector, caplog: pytest.LogCaptureFixture
    ) -> None:
        collector._pass_active = True
        collector._has_compute.clear()
        collector._has_transfer.clear()
        collector._has_wait.clear()
        with (
            patch.object(collector, "_read_pass_snapshot") as read_snapshot,
            caplog.at_level("WARNING"),
        ):
            assert collector.finalize_replay_pass(replay_generation=1) is False
        read_snapshot.assert_not_called()
        assert any("no trap timing events recorded" in r.message for r in caplog.records)

    def test_finalize_refuses_without_external_events(
        self, collector: OffloadTimingCollector, caplog: pytest.LogCaptureFixture
    ) -> None:
        collector._external_events = False
        collector._pass_active = True
        collector._has_compute["layer.0"] = True
        with (
            patch.object(collector, "_read_pass_snapshot") as read_snapshot,
            caplog.at_level("WARNING"),
        ):
            assert collector.finalize_replay_pass(replay_generation=1) is False
        read_snapshot.assert_not_called()
        assert collector._snapshots == []
        assert any("external CUDA timing events are unavailable" in r.message for r in caplog.records)

    def test_arm_after_disarm_can_publish_when_has_maps_kept(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        # Wait-only: finalize_replay_pass must accept non-compute recorded maps.
        collector._has_wait["layer.0"] = True
        collector.disarm_replay_measure()
        collector.arm_replay_measure()
        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="L0", wait_ms=1.0),),
            )
            assert collector.finalize_replay_pass(replay_generation=2) is True
        read_snapshot.assert_called_once_with()
        assert len(collector._snapshots) == 1


class TestPassSinkSurvivesLogClear:
    def test_periodic_log_clears_window_not_sink(self, collector: OffloadTimingCollector) -> None:
        """``_maybe_log_periodic`` must not drop durable ``on_pass`` measure."""

        sunk: list[OffloadTimingSnapshot] = []
        collector.set_pass_sink(sunk.append)
        collector._log_every = 2

        snap = OffloadTimingSnapshot(
            per_trap=(TrapTimingRecord(label="L0", wait_ms=1.0, transfer_ms=2.0, compute_ms=3.0),),
        )
        with patch.object(collector, "_read_pass_snapshot", return_value=snap):
            collector._pass_active = True
            collector._finalize_pass()
            collector._pass_active = True
            collector._finalize_pass()

        assert len(sunk) == 2
        assert collector._snapshots == []  # log window cleared at _log_every
        report = OffloadTimingCollector._build_report(sunk)
        assert report.num_passes == 2


class TestExternalEventsConfig:
    def test_external_events_parameter(self) -> None:
        from flextensor.offload_timing import _resolve_external_timing_events

        collector = OffloadTimingCollector([], external_events=True)
        # Flag must match what the probe actually got (may be False on old PyTorch).
        assert collector._external_events is _resolve_external_timing_events(True)

    def test_external_events_fallback_clears_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        """TypeError on Event(external=True) must not leave _external_events=True."""

        def _fake_make(*, enable_timing: bool = True, external: bool = False):
            if external:
                raise TypeError("Event() got unexpected keyword argument 'external'")
            return MagicMock(name="internal_event")

        with (
            patch("flextensor.offload_timing._make_timing_cuda_event", side_effect=_fake_make),
            caplog.at_level("WARNING"),
        ):
            collector = OffloadTimingCollector(["layer.0"], enabled=True, external_events=True)

        assert collector._external_events is False
        assert any("unsupported in this PyTorch build" in r.message for r in caplog.records)

    def test_log_every_parameter(self) -> None:
        collector = OffloadTimingCollector([], enabled=True, log_every=3)
        assert collector._log_every == 3

    def test_log_every_zero_disables_periodic(self) -> None:
        collector = OffloadTimingCollector([], enabled=True, log_every=0)
        assert collector._log_every == 0


class TestDerivedAggregates:
    def test_snapshot_totals_sum_per_trap(self) -> None:
        snap = OffloadTimingSnapshot(
            per_trap=(
                TrapTimingRecord(label="a", transfer_ms=1.0, compute_ms=2.0, wait_ms=3.0),
                TrapTimingRecord(label="b", transfer_ms=4.0, compute_ms=5.0, wait_ms=6.0),
            )
        )
        assert snap.total_transfer_ms == 5.0
        assert snap.total_compute_ms == 7.0
        assert snap.total_wait_ms == 9.0

    def test_report_aggregates_from_passes(self) -> None:
        from flextensor.offload_timing import OffloadTimingReport

        passes = (
            OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="a", wait_ms=1.0, transfer_ms=2.0, compute_ms=3.0),)
            ),
            OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="a", wait_ms=3.0, transfer_ms=4.0, compute_ms=5.0),)
            ),
        )
        report = OffloadTimingReport(passes=passes)
        assert report.num_passes == 2
        assert report.total_wait_sum == 4.0
        assert report.total_transfer_sum == 6.0
        assert report.total_compute_sum == 8.0
        assert report.total_wait_avg == 2.0
        assert report.total_transfer_avg == 3.0
        assert report.total_compute_avg == 4.0

    def test_layer_record_is_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        rec = TrapTimingRecord(label="a", wait_ms=1.0)
        with pytest.raises(FrozenInstanceError):
            rec.wait_ms = 2.0  # type: ignore[misc]


class TestBuildReportPerTrapStats:
    def test_build_report_aggregates_every_per_trap_statistic(self) -> None:
        """``_build_report`` wires min/max/avg/median/stdev for each column and trap order."""
        import statistics

        passes = (
            OffloadTimingSnapshot(
                per_trap=(
                    TrapTimingRecord(label="L0", compute_ms=1.0, transfer_ms=10.0, wait_ms=100.0),
                    TrapTimingRecord(label="L1", compute_ms=2.0, transfer_ms=20.0, wait_ms=200.0),
                )
            ),
            OffloadTimingSnapshot(
                per_trap=(
                    TrapTimingRecord(label="L0", compute_ms=3.0, transfer_ms=30.0, wait_ms=300.0),
                    TrapTimingRecord(label="L1", compute_ms=4.0, transfer_ms=40.0, wait_ms=400.0),
                )
            ),
            OffloadTimingSnapshot(
                per_trap=(
                    TrapTimingRecord(label="L0", compute_ms=5.0, transfer_ms=50.0, wait_ms=500.0),
                    TrapTimingRecord(label="L1", compute_ms=8.0, transfer_ms=80.0, wait_ms=800.0),
                )
            ),
        )
        report = OffloadTimingCollector._build_report(passes)

        assert [s.label for s in report.per_trap] == ["L0", "L1"]
        assert report.passes == passes
        assert report.num_passes == 3

        l0_c, l0_t, l0_w = [1.0, 3.0, 5.0], [10.0, 30.0, 50.0], [100.0, 300.0, 500.0]
        l1_c, l1_t, l1_w = [2.0, 4.0, 8.0], [20.0, 40.0, 80.0], [200.0, 400.0, 800.0]
        expected = {
            "L0": (l0_c, l0_t, l0_w),
            "L1": (l1_c, l1_t, l1_w),
        }
        for stats in report.per_trap:
            c, t, w = expected[stats.label]
            assert stats.compute_min == min(c)
            assert stats.compute_max == max(c)
            assert stats.compute_avg == sum(c) / len(c)
            assert stats.compute_median == statistics.median(c)
            assert stats.compute_std == statistics.stdev(c)
            assert stats.transfer_min == min(t)
            assert stats.transfer_max == max(t)
            assert stats.transfer_avg == sum(t) / len(t)
            assert stats.transfer_median == statistics.median(t)
            assert stats.transfer_std == statistics.stdev(t)
            assert stats.wait_min == min(w)
            assert stats.wait_max == max(w)
            assert stats.wait_avg == sum(w) / len(w)
            assert stats.wait_median == statistics.median(w)
            assert stats.wait_std == statistics.stdev(w)


class TestFlushPendingEagerPass:
    def test_flush_publishes_pending_eager_pass(self, collector: OffloadTimingCollector) -> None:
        sunk: list[OffloadTimingSnapshot] = []
        collector.set_pass_sink(sunk.append)
        collector._pass_active = True
        collector._pass_was_captured = False
        collector._has_compute["layer.0"] = True

        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            snap = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="L0", compute_ms=1.5),),
            )
            read_snapshot.return_value = snap
            assert collector.flush_pending_eager_pass() is True

        assert sunk == [snap]
        assert collector._pass_active is False
        assert len(collector._snapshots) == 1

    def test_repeated_flush_bounds_rolling_log_window(self, collector: OffloadTimingCollector) -> None:
        """Collect-after-each-forward must not grow ``_snapshots`` without bound."""
        collector._log_every = 10
        sunk: list[OffloadTimingSnapshot] = []
        collector.set_pass_sink(sunk.append)
        snap = OffloadTimingSnapshot(per_trap=(TrapTimingRecord(label="L0", compute_ms=1.0),))

        with patch.object(collector, "_read_pass_snapshot", return_value=snap):
            for _ in range(25):
                collector._pass_active = True
                collector._pass_was_captured = False
                collector._has_compute["layer.0"] = True
                assert collector.flush_pending_eager_pass() is True
                assert len(collector._snapshots) <= collector._log_every

        assert len(sunk) == 25
        assert len(collector._snapshots) == 5  # 25 % 10

    def test_flush_skips_captured_pass(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        collector._pass_was_captured = True
        with patch.object(collector, "_finalize_pass") as finalize:
            assert collector.flush_pending_eager_pass() is False
        finalize.assert_not_called()
        assert collector._pass_active is True

    def test_reset_discards_pending_eager_without_publish(self, collector: OffloadTimingCollector) -> None:
        sunk: list[OffloadTimingSnapshot] = []
        collector.set_pass_sink(sunk.append)
        collector._pass_active = True
        collector._pass_was_captured = False
        collector._has_compute["layer.0"] = True
        collector._snapshots.append(OffloadTimingSnapshot(per_trap=()))

        collector.reset()

        assert sunk == []
        assert collector._pass_active is False
        assert collector._has_compute == {}
        assert collector._snapshots == []

    def test_reset_keeps_captured_replay_state(self, collector: OffloadTimingCollector) -> None:
        collector._pass_active = True
        collector._pass_was_captured = True
        collector._has_compute["layer.0"] = True
        collector.reset()
        assert collector._pass_active is True
        assert collector._has_compute == {"layer.0": True}

    def test_record_under_capture_latches_pass_was_captured(self) -> None:
        """Capture may begin after on_pass_start; record_* must still latch."""
        with patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False):
            collector = OffloadTimingCollector(trap_labels=["layer.0"], enabled=True, external_events=True)
        collector._external_events = True
        collector._pass_active = True
        collector._pass_was_captured = False
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(collector._compute_end["layer.0"], "record"),
        ):
            collector.record_compute_end("layer.0")
        assert collector._pass_was_captured is True
        assert collector._has_compute["layer.0"] is True

        collector.reset()
        assert collector._pass_active is True
        assert collector._has_compute == {"layer.0": True}
        with patch.object(collector, "_read_pass_snapshot") as read_snapshot:
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="layer.0", compute_ms=1.0),),
            )
            assert collector.finalize_replay_pass(replay_generation=1) is True

    def test_record_under_capture_latches_even_without_external_events(self) -> None:
        """Internal events must latch too so on_pass_start drops instead of finalize."""
        with patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False):
            collector = OffloadTimingCollector(trap_labels=["layer.0"], enabled=True, external_events=False)
        collector._external_events = False
        collector._pass_active = True
        collector._pass_was_captured = False
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(collector._compute_end["layer.0"], "record"),
        ):
            collector.record_compute_end("layer.0")
        assert collector._pass_was_captured is True

        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_finalize_pass") as finalize,
            patch.object(collector, "_reset_pass_state", wraps=collector._reset_pass_state) as reset,
        ):
            collector.on_pass_start()
        finalize.assert_not_called()
        reset.assert_called_once_with()


class TestPublicCollectFlushBoundary:
    """Public collect/reset must see the last eager pass and not leak across reset."""

    def _manager_with_collector(self) -> tuple[OffloadManager, OffloadTimingCollector]:
        import torch

        from flextensor.tensor_manager import TensorManager

        tm = TensorManager(
            device_gpu=torch.device("cpu"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            _offload_timing="eager",
        )
        collector = OffloadTimingCollector(trap_labels=["L0"], enabled=True, log_every=0)
        collector.set_pass_sink(tm._record_offload_timing_pass)
        loader = MagicMock()
        loader.offload_timing_collector = collector
        tm.tensor_layer_loader = loader

        om = OffloadManager("test_eager_flush")
        om.config = SimpleNamespace(offload_timing="eager")
        om._tensor_manager = tm
        return om, collector

    def test_collect_after_one_eager_pass_returns_that_pass(self) -> None:
        om, collector = self._manager_with_collector()
        with (
            patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False),
            patch.object(collector, "_read_pass_snapshot") as read_snapshot,
        ):
            collector.on_pass_start()
            collector._has_compute["L0"] = True
            read_snapshot.return_value = OffloadTimingSnapshot(
                per_trap=(TrapTimingRecord(label="L0", compute_ms=2.0),),
            )
            report = om.collect_offload_timing()

        assert report is not None
        assert report.num_passes == 1
        assert report.total_compute_sum == 2.0
        assert len(om._tensor_manager._offload_timing_measure) == 0

    def test_reset_drops_pending_eager_so_next_window_stays_clean(self) -> None:
        om, collector = self._manager_with_collector()
        with patch("flextensor.offload_timing.torch.cuda.is_current_stream_capturing", return_value=False):
            collector.on_pass_start()
            collector._has_compute["L0"] = True
            om.reset_offload_timing()

            # Window is empty immediately after reset (pre-reset pass discarded, not drained).
            assert om.collect_offload_timing() is None

            # Pre-reset pass must not be published when the next forward starts.
            with patch.object(collector, "_finalize_pass") as finalize:
                collector.on_pass_start()
            finalize.assert_not_called()
