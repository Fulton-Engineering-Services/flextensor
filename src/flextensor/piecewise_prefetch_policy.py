# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect H2D prefetches interrupted by PIECEWISE CUDA-graph joins.

Under PIECEWISE capture, each piece must join ``transfer_stream`` before
``capture_end``.  That forces any H2D started in the piece to complete at the
piece boundary, and loader map clears make a later ``wait_event`` skip.

That bites whenever schedule and wait are split across pieces, including:

* **Reordered / rearranged** transfers (schedule at *S*, wait at later *C*).
* **Same-label** enter/exit split across pieces (e.g. parent module ``1`` with
  PIECEWISE cuts at nested ``1.1`` / ``1.2`` inside its forward) — both the
  rolling-block and reordered loaders.

Correctness is usually preserved by the piece join; the lost property is
async overlap (H2D forced onto the critical path of the earlier piece).

**End-of-forward caveat:** the reordered loader's last-layer ``enter`` often
schedules H2D for a *next-iteration* compute label that was already waited
earlier in this forward. That outstanding pair is intentional (wait on next
``enter``); ``on_piece_join(at_last_layer=True)`` ignores those so strict mode
does not false-raise. Mid-forward ``join_after_forward``
(``at_last_layer=False``) still flags every outstanding prefetch.

Loaders always hold a :class:`PiecewisePrefetchPolicy` and call hooks
unconditionally. Construct with ``enabled=False`` for a no-op. Config default
is warn (:attr:`~flextensor.config.OffloadConfig.piecewise_prefetch`
``"warn"`` / ``"error"`` / ``"off"``; also ``FT_PIECEWISE_PREFETCH``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


class PiecewisePrefetchPolicyError(RuntimeError):
    """Raised when a piece join interrupts an outstanding H2D prefetch (strict mode)."""


@dataclass(frozen=True, slots=True)
class OutstandingPrefetch:
    """One in-flight schedule→compute prefetch that has not been waited yet."""

    schedule_label: str
    compute_label: str


class PiecewisePrefetchPolicy:
    """Track scheduled H2D and report when a piece join fires before the wait.

    When ``enabled`` is False (default), all hooks are no-ops so loaders can
    call unconditionally.

    Any outstanding prefetch at a mid-forward ``on_piece_join`` is a policy hit:
    remapped rearrange slots **and** same-label enter/exit pairs that straddle
    PIECEWISE boundaries (nested graph pieces under a parent offload unit).

    At ``at_last_layer=True``, pairs whose ``compute_label`` was already waited
    earlier this forward are treated as next-iteration prefetches and ignored.
    """

    def __init__(self, *, enabled: bool = False, strict: bool = False) -> None:
        self.enabled = enabled
        self.strict = strict
        # compute_label → schedule_label
        self._outstanding: dict[str, str] = {}
        self._warned_keys: set[tuple[str, str]] = set()
        # compute labels waited since the last :meth:`reset` (one forward pass).
        self._waited_this_pass: set[str] = set()

    def reset(self) -> None:
        """Drop outstanding state (e.g. aborted iteration / loader rebuild)."""
        self._outstanding.clear()
        self._waited_this_pass.clear()

    def on_schedule(self, schedule_label: str, compute_label: str) -> None:
        """Record that ``schedule_label`` started H2D for ``compute_label``."""
        if not self.enabled:
            return
        self._outstanding[compute_label] = schedule_label

    def on_wait(self, compute_label: str) -> None:
        """Record that compute for ``compute_label`` waited on its prefetch."""
        if not self.enabled:
            return
        self._outstanding.pop(compute_label, None)
        self._waited_this_pass.add(compute_label)

    def on_piece_join(self, *, at_last_layer: bool = False) -> list[OutstandingPrefetch]:
        """Inspect outstanding prefetches at a piece / forward join.

        Called from loader ``join_after_forward`` and last-layer ``exit``.
        Returns outstanding pairs that violate policy (may be empty).  Warns
        once per pair; raises in strict mode.

        When ``at_last_layer`` is True, skips pairs whose compute label was
        already waited this forward (reordered next-iter prefetch).
        """
        if not self.enabled:
            return []

        all_outstanding = [
            OutstandingPrefetch(schedule_label=sched, compute_label=comp) for comp, sched in self._outstanding.items()
        ]
        if at_last_layer:
            broken = [item for item in all_outstanding if item.compute_label not in self._waited_this_pass]
        else:
            broken = all_outstanding

        # Piece / end-of-forward join drains H2D; drop all outstanding bookkeeping
        # (including ignored next-iter pairs) so we do not re-warn later.
        self._outstanding.clear()

        if not broken:
            return []

        new_items = [item for item in broken if (item.schedule_label, item.compute_label) not in self._warned_keys]
        for item in new_items:
            self._warned_keys.add((item.schedule_label, item.compute_label))

        if not new_items and not self.strict:
            return broken

        detail = ", ".join(
            (
                f"{item.schedule_label!r}"
                if item.schedule_label == item.compute_label
                else f"{item.schedule_label!r}→{item.compute_label!r}"
            )
            for item in (new_items or broken)
        )
        where = "end-of-forward join" if at_last_layer else "PIECEWISE piece join"
        msg = (
            f"FlexTensor piecewise prefetch policy: {where} with outstanding "
            f"H2D prefetch(es) [{detail}]. Transfer was forced to finish at "
            f"the join before the matching wait (rearrange early-slot, or "
            f"enter/exit split across nested pieces such as parent '1' with "
            f"graphs on '1.1'/'1.2'). Async overlap across those pieces is "
            f"not preserved. Prefer cudagraph_mode=FULL, avoid piece splits "
            f"between schedule and wait, or leave piecewise_prefetch='warn' "
            f"and treat this as a performance warning."
        )
        if self.strict:
            raise PiecewisePrefetchPolicyError(msg)
        if new_items:
            LOGGER.warning(msg)
        return broken
