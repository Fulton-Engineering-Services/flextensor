# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trap transfer / compute / wait timing for pipelined offload loaders.

Opt-in via :attr:`~flextensor.config.OffloadConfig.offload_timing`
(``"eager"`` or ``"cuda_graph"``). Use ``"cuda_graph"`` so ``elapsed_time()``
reflects each replay rather than capture-time values.

Periodic logging cadence and durable measure retention are internal constants
(:data:`OFFLOAD_TIMING_LOG_EVERY`, :data:`OFFLOAD_TIMING_MEASURE_MAX_PASSES`).
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Callable, Sequence  # noqa: TC003
from dataclasses import dataclass
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)

# Internal defaults — not OffloadConfig knobs (setup-time choice is offload_timing mode only).
OFFLOAD_TIMING_LOG_EVERY: int = 10
OFFLOAD_TIMING_MEASURE_MAX_PASSES: int = 1024


@dataclass(frozen=True, slots=True)
class TrapTimingRecord:
    """Per-trap timing measurements from a single forward pass.

    All durations are in milliseconds.  A ``wait_ms`` value of zero means
    the transfer finished before compute needed the data (fully hidden).

    ``label`` is the stable trap name string (same concept as
    ``LayerStatistics.label`` until that legacy rename lands).
    """

    label: str
    transfer_ms: float = 0.0
    compute_ms: float = 0.0
    wait_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class OffloadTimingSnapshot:
    """Timing data from a single forward pass.

    Canonical data is :attr:`per_trap`. Aggregate ``total_*_ms`` values are
    derived from those records so they cannot drift.
    """

    per_trap: Sequence[TrapTimingRecord] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_trap", tuple(self.per_trap))

    @property
    def total_wait_ms(self) -> float:
        """Cumulative stall across all traps in this pass."""
        return sum(r.wait_ms for r in self.per_trap)

    @property
    def total_transfer_ms(self) -> float:
        """Cumulative H2D transfer time across all traps in this pass."""
        return sum(r.transfer_ms for r in self.per_trap)

    @property
    def total_compute_ms(self) -> float:
        """Cumulative compute / budget window across all traps in this pass."""
        return sum(r.compute_ms for r in self.per_trap)


@dataclass(frozen=True, slots=True)
class TrapTimingStats:
    """Aggregate timing statistics for one trap across multiple passes."""

    label: str
    compute_min: float = 0.0
    compute_max: float = 0.0
    compute_avg: float = 0.0
    compute_median: float = 0.0
    compute_std: float = 0.0
    transfer_min: float = 0.0
    transfer_max: float = 0.0
    transfer_avg: float = 0.0
    transfer_median: float = 0.0
    transfer_std: float = 0.0
    wait_min: float = 0.0
    wait_max: float = 0.0
    wait_avg: float = 0.0
    wait_median: float = 0.0
    wait_std: float = 0.0

    def compute_budget_ms(self, *, conservative: bool = False) -> float:
        """Positive compute budget (ms) for offload knapsack / graph re-plan.

        Under CUDA-graph replay the compute column is the per-trap hiding
        window. Prefer median for a typical budget; ``conservative=True``
        prefers min (tightest window across noisy probes).

        Only this column is consumed by replan. Measured ``transfer_ms`` /
        ``wait_ms`` stay diagnostic (see
        :meth:`OffloadTimingReport.compute_budgets_by_profile_label`).
        """
        if conservative:
            order = (self.compute_min, self.compute_median, self.compute_avg)
        else:
            order = (self.compute_median, self.compute_avg, self.compute_min)
        for budget_ms in order:
            value = float(budget_ms or 0.0)
            if value > 0:
                return value
        return 0.0


@dataclass(slots=True)
class OffloadTimingReport:
    """Aggregate timing report across multiple forward passes.

    Produced by :meth:`~flextensor.OffloadManager.collect_offload_timing`
    (drains TensorManager's durable measure store). The collector's rolling
    ``_snapshots`` window is for periodic logging only.

    Canonical pass data is :attr:`passes`. ``num_passes`` and pass-total
    sums/averages are derived from it. :attr:`per_trap` holds precomputed
    min/max/avg/median/std stats built once when the report is assembled.
    """

    per_trap: Sequence[TrapTimingStats] = ()
    passes: Sequence[OffloadTimingSnapshot] = ()

    def __post_init__(self) -> None:
        self.per_trap = tuple(self.per_trap)
        self.passes = tuple(self.passes)

    @property
    def num_passes(self) -> int:
        return len(self.passes)

    @property
    def total_compute_sum(self) -> float:
        return sum(s.total_compute_ms for s in self.passes)

    @property
    def total_transfer_sum(self) -> float:
        return sum(s.total_transfer_ms for s in self.passes)

    @property
    def total_wait_sum(self) -> float:
        return sum(s.total_wait_ms for s in self.passes)

    @property
    def total_compute_avg(self) -> float:
        n = self.num_passes
        return self.total_compute_sum / n if n else 0.0

    @property
    def total_transfer_avg(self) -> float:
        n = self.num_passes
        return self.total_transfer_sum / n if n else 0.0

    @property
    def total_wait_avg(self) -> float:
        n = self.num_passes
        return self.total_wait_sum / n if n else 0.0

    def compute_budgets_by_profile_label(
        self,
        profile_labels: Sequence[str],
        *,
        conservative: bool = False,
    ) -> dict[str, float]:
        """Map this report's **compute** budgets onto profile trap labels by index.

        Timing row *i* pairs with ``profile_labels[i]`` (report labels may be
        shortened relative to TensorManager stats / ``LayerStatistics.label``).

        Replan feeds only these compute budgets into
        :meth:`~flextensor.tensor_manager.TensorManager.replan_from_compiled_durations`
        (rewriting per-trap ``duration``). Measured ``transfer_ms`` and
        ``wait_ms`` are **not** strategy inputs:

        * **Transfer cost** stays on the profiling size→time curve
          (``memory_transfer_stats``). H2D volume and host↔device bandwidth are
          assumed unchanged under CUDA-graph replay, so the size-based model
          remains valid; runtime ``transfer_ms`` is a diagnostic check that the
          curve still matches.
        * **Wait** is an *outcome* of the current schedule (exposed stall when
          H2D was not hidden), not a planner cost. ``wait_ms > 0`` means the
          prior plan lost overlap; replan adjusts the compute/hiding window so
          the knapsack can re-pack. Feeding wait back as a cost would be
          circular.
        """
        if not self.per_trap or not profile_labels:
            return {}
        n_timing = len(self.per_trap)
        n_profile = len(profile_labels)
        if n_timing != n_profile:
            LOGGER.warning(
                "FlexTensor: offload-timing trap count (%d) != profile label "
                "count (%d); mapping budgets by index — extra profile traps get "
                "no budget, extra timing rows are dropped. Check offload unit "
                "order if replan looks wrong.",
                n_timing,
                n_profile,
            )
        budgets: dict[str, float] = {}
        for idx, label in enumerate(profile_labels):
            if idx >= n_timing:
                break
            budget_ms = self.per_trap[idx].compute_budget_ms(conservative=conservative)
            if budget_ms > 0:
                budgets[label] = budget_ms
        return budgets


def format_offload_timing_table(data: OffloadTimingSnapshot | OffloadTimingReport) -> str:
    """Format a timing snapshot or report as a human-readable table.

    Accepts either a single-pass :class:`OffloadTimingSnapshot` or a
    multi-pass :class:`OffloadTimingReport`.
    """
    if isinstance(data, OffloadTimingReport):
        return _format_report_table(data)
    return _format_snapshot_table(data)


def _format_snapshot_table(snapshot: OffloadTimingSnapshot) -> str:
    if not snapshot.per_trap:
        return "No offload timing data."

    max_label = max(len(r.label) for r in snapshot.per_trap)
    label_w = max(max_label, 5) + 2
    sep_w = label_w + 46

    header = "=" * sep_w
    col_header = f"{'Trap':<{label_w}} {'Compute (ms)':>13} {'Transfer (ms)':>14} {'Wait (ms)':>10}"
    divider = "-" * sep_w

    lines = [
        "Offload Timing (single pass)",
        header,
        col_header,
        divider,
    ]

    for rec in snapshot.per_trap:
        flag = "  \u2190" if rec.wait_ms > 0.01 else ""
        lines.append(
            f"{rec.label:<{label_w}} {rec.compute_ms:>13.3f} {rec.transfer_ms:>14.3f} {rec.wait_ms:>10.3f}{flag}"
        )

    lines.append(divider)
    lines.append(
        f"{'Total':<{label_w}} "
        f"{snapshot.total_compute_ms:>13.3f} "
        f"{snapshot.total_transfer_ms:>14.3f} "
        f"{snapshot.total_wait_ms:>10.3f}"
    )
    lines.append(header)
    return "\n".join(lines)


def _format_report_table(report: OffloadTimingReport) -> str:
    if not report.per_trap:
        return "No offload timing data."

    max_label = max(len(s.label) for s in report.per_trap)
    label_w = max(max_label, 5) + 2

    sep_w = label_w + 106
    header = "=" * sep_w
    col_header = (
        f"{'Trap':<{label_w}} "
        f"{'Compute avg':>12} {'med':>8} {'±std':>7} {'[min':>7} {'max]':>7}  "
        f"{'Transfer avg':>13} {'med':>8}  "
        f"{'Wait avg':>9} {'med':>8}"
    )
    divider = "-" * sep_w

    lines = [
        f"Offload Timing ({report.num_passes} passes)",
        header,
        col_header,
        divider,
    ]

    for s in report.per_trap:
        flag = "  \u2190" if s.wait_avg > 0.01 else ""
        lines.append(
            f"{s.label:<{label_w}} "
            f"{s.compute_avg:>12.3f} {s.compute_median:>8.3f} {s.compute_std:>7.3f} "
            f"{s.compute_min:>7.3f} {s.compute_max:>7.3f}  "
            f"{s.transfer_avg:>13.3f} {s.transfer_median:>8.3f}  "
            f"{s.wait_avg:>9.3f} {s.wait_median:>8.3f}"
            f"{flag}"
        )

    lines.append(divider)
    lines.append(
        f"{'Avg/pass':<{label_w}} "
        f"{report.total_compute_avg:>12.3f} {'':>8} {'':>7} {'':>7} {'':>7}  "
        f"{report.total_transfer_avg:>13.3f} {'':>8}  "
        f"{report.total_wait_avg:>9.3f}"
    )
    lines.append(
        f"{'Sum (all)':<{label_w}} "
        f"{report.total_compute_sum:>12.1f} {'':>8} {'':>7} {'':>7} {'':>7}  "
        f"{report.total_transfer_sum:>13.1f} {'':>8}  "
        f"{report.total_wait_sum:>9.1f}"
    )
    lines.append(header)
    return "\n".join(lines)


def _make_timing_cuda_event(*, enable_timing: bool = True, external: bool = False) -> torch.cuda.Event:
    """Create a CUDA event for :class:`OffloadTimingCollector`.

    When ``external=True``, captured graphs record ``cudaEventRecordExternal``
    nodes so ``elapsed_time()`` reflects each replay. Requires PyTorch with
    CUDA-graph external-event support (``Event(external=...)``); older builds
    raise :class:`TypeError` (caller must fall back and clear the external flag).
    """
    kwargs: dict[str, bool] = {"enable_timing": enable_timing}
    if external:
        kwargs["external"] = True
    return torch.cuda.Event(**kwargs)


def _resolve_external_timing_events(requested: bool) -> bool:
    """Return whether external timing events are actually available.

    When ``requested`` is True but this PyTorch build rejects
    ``Event(external=True)``, log a WARNING and return False so the collector
    does not claim replay-safe readback while holding internal events.
    """
    if not requested:
        return False
    try:
        _make_timing_cuda_event(external=True)
    except TypeError:
        LOGGER.warning(
            "OffloadTimingCollector: torch.cuda.Event(external=True) is "
            "unsupported in this PyTorch build; using internal events. "
            "CUDA-graph replay timing will be unreliable — upgrade PyTorch "
            "or run with cudagraph_mode=NONE.",
        )
        return False
    return True


class OffloadTimingCollector:
    """Opt-in collector that records transfer / compute / stall durations.

    Pre-allocates CUDA timing events for each trap when ``enabled`` is True.
    Construct with ``enabled=False`` for a no-op so loaders can call hooks
    unconditionally.

    Under **eager** execution, passes are auto-finalized at each forward-pass
    boundary (when the first trap is entered again via :meth:`on_pass_start`),
    so every denoising step is captured even when ``pipe()`` calls the model
    many times internally. **CUDA-graph replay** does not re-enter those hooks;
    call
    :meth:`~flextensor.OffloadManager.update_offload_timing` (or
    :meth:`~flextensor.OffloadManager.update_state` during a manual replan)
    after each ``graph.replay()`` so :meth:`finalize_replay_pass` publishes
    the captured events.

    ``external_events`` is derived from
    :attr:`~flextensor.config.OffloadConfig.offload_timing`
    (``"cuda_graph"`` → True). ``log_every`` defaults to
    :data:`OFFLOAD_TIMING_LOG_EVERY` (not an OffloadConfig knob).
    """

    def __init__(
        self,
        trap_labels: list[str] | None = None,
        *,
        enabled: bool = True,
        external_events: bool = False,
        log_every: int = OFFLOAD_TIMING_LOG_EVERY,
        on_pass: Callable[[OffloadTimingSnapshot], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self._labels: list[str] = list(trap_labels) if trap_labels else []
        # Disabled collectors must stay true no-ops even if labels were passed:
        # event maps are not allocated below, so an empty label set keeps hooks
        # from indexing missing events.
        self._label_set = set(self._labels) if enabled else set()
        self._transfer_start: dict[str, torch.cuda.Event] = {}
        self._transfer_end: dict[str, torch.cuda.Event] = {}
        self._compute_start: dict[str, torch.cuda.Event] = {}
        self._compute_end: dict[str, torch.cuda.Event] = {}
        self._wait_start: dict[str, torch.cuda.Event] = {}
        self._wait_end: dict[str, torch.cuda.Event] = {}
        self._has_wait: dict[str, bool] = {}
        self._has_transfer: dict[str, bool] = {}
        self._has_compute: dict[str, bool] = {}
        self._pass_active = False
        self._pass_was_captured = False
        # Rolling window for periodic logging only — cleared by ``_maybe_log_periodic``.
        # Durable measure accumulation lives on TensorManager via ``on_pass``.
        self._snapshots: list[OffloadTimingSnapshot] = []
        self._on_pass = on_pass
        self._log_every = 0
        self._last_finalized_replay_generation: int = -1
        self._external_events = False
        self._warned_drop_captured = False

        if not enabled:
            return

        # Probe once: never leave ``_external_events=True`` with internal events.
        self._external_events = _resolve_external_timing_events(external_events)
        self._log_every = log_every

        for label in self._labels:
            self._transfer_start[label] = _make_timing_cuda_event(
                external=self._external_events,
            )
            self._transfer_end[label] = _make_timing_cuda_event(
                external=self._external_events,
            )
            self._compute_start[label] = _make_timing_cuda_event(
                external=self._external_events,
            )
            self._compute_end[label] = _make_timing_cuda_event(
                external=self._external_events,
            )
            self._wait_start[label] = _make_timing_cuda_event(
                external=self._external_events,
            )
            self._wait_end[label] = _make_timing_cuda_event(
                external=self._external_events,
            )

    def set_pass_sink(self, on_pass: Callable[[OffloadTimingSnapshot], None] | None) -> None:
        """Wire/replace the durable measure sink (TensorManager)."""
        self._on_pass = on_pass

    # ------------------------------------------------------------------
    # Recording helpers (called from loader hot path — very cheap)
    # ------------------------------------------------------------------

    def _mark_captured_if_recording_under_capture(self) -> None:
        """Latch ``_pass_was_captured`` if recording under CUDA-graph capture.

        Covers capture that starts after ``on_pass_start``. Without the latch,
        a later eager ``on_pass_start`` would treat the pass as non-captured and
        call ``_finalize_pass()``, which is unsafe for internal events and
        clears ``_has_*`` needed for external replay measure.
        """
        if self._pass_was_captured:
            return
        if torch.cuda.is_current_stream_capturing():
            self._pass_was_captured = True

    def record_transfer_start(self, label: str, stream: Any) -> None:
        if label in self._label_set:
            self._transfer_start[label].record(stream)
            # Incomplete until record_transfer_end — avoid elapsed_time on a
            # missing/stale end if finalize runs after a partial abort.
            self._has_transfer[label] = False
            self._mark_captured_if_recording_under_capture()

    def record_transfer_end(self, label: str, stream: Any) -> None:
        if label in self._label_set:
            self._transfer_end[label].record(stream)
            self._has_transfer[label] = True
            self._mark_captured_if_recording_under_capture()

    def record_compute_start(self, label: str) -> None:
        if label in self._label_set:
            self._compute_start[label].record()
            self._has_compute[label] = False
            self._mark_captured_if_recording_under_capture()

    def record_compute_end(self, label: str) -> None:
        if label in self._label_set:
            self._compute_end[label].record()
            self._has_compute[label] = True
            self._mark_captured_if_recording_under_capture()

    def record_wait_start(self, label: str) -> None:
        if label in self._label_set:
            self._wait_start[label].record()
            self._has_wait[label] = False
            self._mark_captured_if_recording_under_capture()

    def record_wait_end(self, label: str) -> None:
        if label in self._label_set:
            self._wait_end[label].record()
            self._has_wait[label] = True
            self._mark_captured_if_recording_under_capture()

    def on_pass_start(self) -> None:
        """Called by the loader when the first trap is entered.

        Finalizes the previous pass's events (if any) before they get
        overwritten.  The previous pass is guaranteed to be complete at
        this point because the join event synced both streams.

        Branches cover the capture-mode permutations:

        1. **No pending pass** — first call after construction or after a
           finalize / reset. Nothing to finalize; just mark the new pass active.
        2. **Pending pass + currently capturing** — ``_finalize_pass`` is
           illegal (its ``torch.cuda.synchronize`` is rejected as
           ``cudaErrorStreamCaptureUnsupported``). Skip the finalize. The
           prior pass's events keep updating from any graph replays that
           target them; ``_pass_was_captured`` propagates the dirty-state
           flag through to the next eager boundary or
           :meth:`finalize_replay_pass`.
        3. **Pending captured pass + not capturing + internal events** —
           ``cudaEventElapsedTime`` on those events is unreliable. Drop the
           dirty state without reading; the new pass starts fresh.
        4. **Pending captured pass + not capturing + external events** —
           publish via :meth:`finalize_replay_pass` so ``_has_*`` stay set
           for post-capture CUDA-graph measure replan. ``_finalize_pass``
           would clear those maps and leave ``arm_replay_measure`` with an
           empty trap set.
        5. **Pending non-captured pass + not capturing** — normal
           :meth:`_finalize_pass`.
        """
        if not self.enabled:
            return

        capturing = torch.cuda.is_current_stream_capturing()
        if self._pass_active and not capturing:
            if self._pass_was_captured and not self._external_events:
                if not self._warned_drop_captured:
                    LOGGER.warning(
                        "OffloadTimingCollector: dropping captured pass timings — "
                        "elapsed_time() on internal CUDA-graph events is unreliable. "
                        "Set offload_timing='cuda_graph' for replay readback.",
                    )
                    self._warned_drop_captured = True
                self._reset_pass_state()
            elif self._pass_was_captured and self._external_events:
                # Keep _has_* / _pass_was_captured for graph-replay measure.
                # If finalize refuses (e.g. empty trap maps), drop pending state
                # so the new pass does not stick in a captured+active loop.
                if not self.finalize_replay_pass():
                    self._reset_pass_state()
            else:
                self._finalize_pass()
        self._pass_active = True
        if capturing:
            self._pass_was_captured = True

    def on_pass_end(self) -> None:
        """Optional end-of-pass hook; eager finalization is deferred.

        Publishing the completed pass is deferred to the next
        :meth:`on_pass_start` (streaming) or to
        :meth:`flush_pending_eager_pass` at collect / reset boundaries so a
        single-forward measure window is not left empty.
        """

    # ------------------------------------------------------------------
    # Internal finalization
    # ------------------------------------------------------------------

    def flush_pending_eager_pass(self) -> bool:
        """Publish a pending **non-captured** pass into the log + durable sink.

        :meth:`on_pass_start` only finalizes the *previous* pass, so after the
        last eager forward the newest pass is still pending. Call this before
        draining the durable measure store (or when discarding a window) so
        that pass is not lost / cannot leak into the next window via a later
        ``on_pass_start``.

        Captured passes are left alone: use :meth:`finalize_replay_pass` after
        replay (``offload_timing='cuda_graph'``), or let the next eager
        :meth:`on_pass_start` drop dirty internal-event state.

        Returns:
            ``True`` when a pass was finalized and published.
        """
        if not self.enabled or not self._pass_active:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if self._pass_was_captured:
            return False
        self._finalize_pass(periodic_log=True)
        return True

    def _finalize_pass(self, *, periodic_log: bool = True) -> None:
        """Read events from the completed pass and store a snapshot.

        Traps whose compute events were never recorded (e.g. the first
        partial pass after the collector is created mid-forward) are
        included with zero durations.
        """
        snapshot = self._read_pass_snapshot()
        self._has_wait.clear()
        self._has_transfer.clear()
        self._has_compute.clear()
        self._pass_active = False
        self._pass_was_captured = False

        self._publish_pass(snapshot, periodic_log=periodic_log)

    def finalize_replay_pass(self, replay_generation: int = -1) -> bool:
        """Finalize timing after a CUDA-graph replay forward.

        Under ``FULL_AND_PIECEWISE`` serving, vLLM replays captured graphs
        without re-entering the Python custom-op implementations, so
        :meth:`on_pass_start` never fires and :meth:`_finalize_pass` is
        never reached via the eager path. The timing events recorded into
        the graph during capture are still updated on each replay; this
        method syncs and reads them from host code after the forward
        returns.

        ``replay_generation`` is bumped by the worker's
        ``torch.cuda.CUDAGraph.replay`` hook once per actual graph replay.
        vLLM may call ``_model_forward`` several times per replay (piece
        dispatch, ubatch wrappers, etc.); passing the generation prevents
        reading and logging the same replay multiple times.

        Returns:
            ``True`` when a pass was published, or when this
            ``replay_generation`` was already finalized (same-gen dedup),
            or when ``replay_generation < 0`` after a gen-aware publish
            (``-1`` must not re-sink). ``False`` on an unexpected no-op
            (disabled, no external events, ``_pass_active`` false, currently
            capturing, or no trap events recorded) so callers can abort rather
            than advance an empty measure slot.
        """
        if not self.enabled:
            return False
        if not self._pass_active:
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if replay_generation >= 0 and replay_generation == self._last_finalized_replay_generation:
            return True  # same-gen dedup: already published this replay
        if replay_generation < 0 and self._last_finalized_replay_generation >= 0:
            # Gen-aware caller already published this replay; default ``-1``
            # (e.g. update_state after a serving hook) must not re-sink.
            return True
        if not self._external_events:
            LOGGER.warning(
                "OffloadTimingCollector: finalize_replay_pass refused — external CUDA "
                "timing events are unavailable; internal captured events are not "
                "replay-safe. Upgrade PyTorch or use offload_timing='eager'."
            )
            return False
        if not (self._has_transfer or self._has_wait or self._has_compute):
            # Arm without capture/record left no trap events — refuse zeros.
            LOGGER.warning(
                "OffloadTimingCollector: finalize_replay_pass refused — no trap "
                "timing events recorded (disarmed then re-armed without recapture?). "
                "Recapture the CUDA graph before measuring again."
            )
            return False
        snapshot = self._read_pass_snapshot()
        self._publish_pass(snapshot, periodic_log=True)
        if replay_generation >= 0:
            self._last_finalized_replay_generation = replay_generation
        # Keep ``_pass_active`` and the ``_has_*`` maps: the same captured
        # event handles are reused on every replay.
        return True

    def _publish_pass(self, snapshot: OffloadTimingSnapshot, *, periodic_log: bool) -> None:
        """Append to the log window, sink to durable measure, optionally log."""
        self._snapshots.append(snapshot)
        if self._on_pass is not None:
            self._on_pass(snapshot)
        if periodic_log:
            self._maybe_log_periodic()

    def _read_pass_snapshot(self) -> OffloadTimingSnapshot:
        """Synchronize and read per-trap CUDA event timings for one pass."""
        torch.cuda.synchronize()

        records: list[TrapTimingRecord] = []
        for label in self._labels:
            transfer_ms = 0.0
            wait_ms = 0.0
            compute_ms = 0.0

            if self._has_transfer.get(label, False):
                transfer_ms = self._transfer_start[label].elapsed_time(self._transfer_end[label])

            if self._has_wait.get(label, False):
                wait_ms = self._wait_start[label].elapsed_time(self._wait_end[label])

            if self._has_compute.get(label, False):
                compute_ms = self._compute_start[label].elapsed_time(self._compute_end[label])

            records.append(
                TrapTimingRecord(
                    label=label,
                    transfer_ms=transfer_ms,
                    compute_ms=compute_ms,
                    wait_ms=wait_ms,
                )
            )

        return OffloadTimingSnapshot(per_trap=tuple(records))

    def _maybe_log_periodic(self) -> None:
        """Log a rolling offload-timing table every ``_log_every`` passes.

        Clears only the **log** window (``_snapshots``). Durable measure
        passes live on TensorManager and are unaffected.
        """
        if self._log_every <= 0 or len(self._snapshots) < self._log_every:
            return
        report = self._build_report(self._snapshots)
        LOGGER.info(
            "FlexTensor: Offload timing (rolling window, %d pass(es), avg stall/pass: %.2f ms)\n%s",
            report.num_passes,
            report.total_wait_avg,
            format_offload_timing_table(report),
        )
        self._snapshots.clear()

    def _reset_pass_state(self) -> None:
        """Drop pending-pass state without reading events.

        Mirrors the bookkeeping done at the tail of ``_finalize_pass()``
        (clearing the ``_has_*`` maps and ``_pass_active``) without the
        ``torch.cuda.synchronize()`` + ``elapsed_time`` reads. Used when a
        pass was captured and its events can no longer be safely queried
        from host code.
        """
        self._has_wait.clear()
        self._has_transfer.clear()
        self._has_compute.clear()
        self._pass_active = False
        self._pass_was_captured = False

    def reset(self) -> None:
        """Clear the rolling **log** window; drop pending eager pass state.

        A pending **non-captured** pass is discarded (without publishing) so
        the next :meth:`on_pass_start` cannot publish a pre-reset forward into
        a fresh durable measure window.

        After CUDA-graph capture, keeps ``_pass_active`` / ``_has_*`` when the
        pending pass was captured so :meth:`finalize_replay_pass` can still
        read those event handles during post-capture replan. Call
        :meth:`OffloadManager.reset_offload_timing` to also clear the durable
        measure store.
        """
        if self._pass_active and not self._pass_was_captured:
            self._reset_pass_state()
        self._snapshots.clear()
        self._last_finalized_replay_generation = -1

    def arm_replay_measure(self) -> None:
        """Ensure :meth:`finalize_replay_pass` records after CUDA-graph capture.

        Capture should leave ``_pass_active`` / ``_has_*`` set; call this if a
        prior finalize cleared ``_pass_active`` so replay measure would
        otherwise no-op. Warns when trap maps are empty — measure cannot
        succeed until graphs are recaptured with timing events recorded.
        """
        if not self.enabled:
            return
        self._pass_active = True
        if not (self._has_transfer or self._has_wait or self._has_compute):
            LOGGER.warning(
                "OffloadTimingCollector: arm_replay_measure with no recorded trap "
                "events — finalize_replay_pass will refuse until graphs are "
                "recaptured (on_pass_start must not clear _has_* after capture)."
            )

    def disarm_replay_measure(self) -> None:
        """Clear only ``_pass_active`` after a measure window finishes.

        Keeps ``_has_*`` maps and event handles so a later
        :meth:`arm_replay_measure` can still publish real replay timings
        (or so recapture can re-record). Only ``_pass_active`` must drop so
        the next eager :meth:`on_pass_start` does not re-publish the last
        replay into a fresh durable store.
        """
        if self.enabled:
            self._pass_active = False

    @staticmethod
    def _build_report(snapshots: Sequence[OffloadTimingSnapshot]) -> OffloadTimingReport:
        if not snapshots:
            return OffloadTimingReport()

        n = len(snapshots)
        labels = [r.label for r in snapshots[0].per_trap]
        n_traps = len(labels)

        compute_by_trap: list[list[float]] = [[] for _ in range(n_traps)]
        transfer_by_trap: list[list[float]] = [[] for _ in range(n_traps)]
        wait_by_trap: list[list[float]] = [[] for _ in range(n_traps)]

        for snap in snapshots:
            for i, rec in enumerate(snap.per_trap):
                compute_by_trap[i].append(rec.compute_ms)
                transfer_by_trap[i].append(rec.transfer_ms)
                wait_by_trap[i].append(rec.wait_ms)

        per_trap: list[TrapTimingStats] = []
        for i, label in enumerate(labels):
            c = compute_by_trap[i]
            t = transfer_by_trap[i]
            w = wait_by_trap[i]
            c_avg = sum(c) / n
            t_avg = sum(t) / n
            w_avg = sum(w) / n
            per_trap.append(
                TrapTimingStats(
                    label=label,
                    compute_min=min(c),
                    compute_max=max(c),
                    compute_avg=c_avg,
                    compute_median=statistics.median(c) if c else 0.0,
                    compute_std=statistics.stdev(c) if len(c) >= 2 else 0.0,
                    transfer_min=min(t),
                    transfer_max=max(t),
                    transfer_avg=t_avg,
                    transfer_median=statistics.median(t) if t else 0.0,
                    transfer_std=statistics.stdev(t) if len(t) >= 2 else 0.0,
                    wait_min=min(w),
                    wait_max=max(w),
                    wait_avg=w_avg,
                    wait_median=statistics.median(w) if w else 0.0,
                    wait_std=statistics.stdev(w) if len(w) >= 2 else 0.0,
                )
            )

        return OffloadTimingReport(
            per_trap=tuple(per_trap),
            passes=tuple(snapshots),
        )
