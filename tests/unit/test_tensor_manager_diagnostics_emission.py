# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anchor tests for the post-profiling diagnostics emission contract.

These tests pin the behaviour invoked by ``TensorManager.prepare_infer_mode``
when it calls ``report_profiling_quality``. The two emissions covered here
are independent and have different gating, so they are exercised separately:

* ``UntimedTrapsReport`` — emitted at WARNING on
  ``flextensor.layer_statistics_analyzer`` whenever any label has tensor
  IDs but no duration samples. Unconditional (not gated on
  ``enable_diagnostics``) because untimed labels are silently dropped from
  strategy input and prod users need to see which layers were excluded.
* Per-layer duration statistics table — emitted at WARNING on the
  diagnostics logger only when ``enable_diagnostics=True`` *and* the
  consistency check failed. The table's columns exist to troubleshoot
  variability/low-sample warnings; on consistent runs it would just be
  noise alongside the strategy table's median.

A future refactor that re-gates the trap report behind ``enable_diagnostics``,
or that drops the table emission entirely, will fail one of these tests.
"""

from __future__ import annotations

import logging

import pytest

from flextensor._logging import DIAGNOSTICS_LOGGER_NAME
from flextensor.collectors import IterativeLayerStatisticsCollector
from flextensor.layer_statistics_analyzer import report_profiling_quality

_ANALYZER_LOGGER = "flextensor.layer_statistics_analyzer"


@pytest.fixture
def consistent_collector() -> IterativeLayerStatisticsCollector:
    """Collector whose only label has 5 identical samples (CV=0, count>=3)."""
    collector = IterativeLayerStatisticsCollector()
    for _ in range(5):
        collector.add_all("layer_a", {1}, 10.0)
    return collector


@pytest.fixture
def inconsistent_collector() -> IterativeLayerStatisticsCollector:
    """Collector with too few samples — fails the consistency check."""
    collector = IterativeLayerStatisticsCollector()
    collector.add_all("layer_a", {1}, 10.0)
    collector.add_all("layer_a", {1}, 15.0)
    return collector


class TestUntimedTrapsEmission:
    """Untimed-trap warning fires unconditionally when any label is untimed."""

    def test_untimed_traps_warn_when_diagnostics_disabled(self, caplog) -> None:
        collector = IterativeLayerStatisticsCollector()
        for _ in range(5):
            collector.add_all("timed_layer", {1}, 10.0)
        collector.add_tensors("untimed_layer", {2})

        with caplog.at_level(logging.WARNING, logger=_ANALYZER_LOGGER):
            report_profiling_quality(collector, enable_diagnostics=False)

        warnings = [r for r in caplog.records if r.name == _ANALYZER_LOGGER and r.levelno == logging.WARNING]
        assert any("untimed_layer" in r.getMessage() for r in warnings), (
            "untimed traps must surface in prod (enable_diagnostics=False) — they cause "
            "silent strategy degradation otherwise"
        )

    def test_untimed_traps_warn_when_diagnostics_enabled(self, caplog) -> None:
        collector = IterativeLayerStatisticsCollector()
        for _ in range(5):
            collector.add_all("timed_layer", {1}, 10.0)
        collector.add_tensors("untimed_layer", {2})

        with caplog.at_level(logging.WARNING, logger=_ANALYZER_LOGGER):
            report_profiling_quality(collector, enable_diagnostics=True)

        warnings = [r for r in caplog.records if r.name == _ANALYZER_LOGGER and r.levelno == logging.WARNING]
        assert any("untimed_layer" in r.getMessage() for r in warnings)

    def test_no_warning_when_all_traps_timed(self, consistent_collector, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=_ANALYZER_LOGGER):
            report_profiling_quality(consistent_collector, enable_diagnostics=True)

        assert "no duration samples" not in caplog.text, "no untimed traps means no trap-report warning"


class TestStatisticsTableEmission:
    """The verbose stats table is gated on enable_diagnostics AND inconsistency."""

    def test_table_emitted_when_diagnostics_on_and_inconsistent(self, inconsistent_collector, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=DIAGNOSTICS_LOGGER_NAME):
            report_profiling_quality(inconsistent_collector, enable_diagnostics=True)

        diag_records = [r for r in caplog.records if r.name == DIAGNOSTICS_LOGGER_NAME]
        assert any("Layer Duration Statistics" in r.getMessage() for r in diag_records), (
            "inconsistent runs under enable_diagnostics=True must emit the troubleshooting table"
        )

    def test_table_suppressed_when_diagnostics_on_and_consistent(self, consistent_collector, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=DIAGNOSTICS_LOGGER_NAME):
            report_profiling_quality(consistent_collector, enable_diagnostics=True)

        assert "Layer Duration Statistics" not in caplog.text, (
            "consistent runs must not dump the table even with enable_diagnostics=True — "
            "the strategy table's median is sufficient"
        )

    def test_table_suppressed_when_diagnostics_off(self, inconsistent_collector, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=DIAGNOSTICS_LOGGER_NAME):
            report_profiling_quality(inconsistent_collector, enable_diagnostics=False)

        assert "Layer Duration Statistics" not in caplog.text, (
            "the verbose table belongs behind enable_diagnostics; the per-issue WARNINGs "
            "from check_measurement_consistency are the prod-visible signal"
        )
