# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anchor tests for profiling warnings and inference diagnostics.

These tests pin two independent contracts:

* ``UntimedTrapsReport`` — emitted at WARNING on
  ``flextensor.layer_statistics_analyzer`` whenever any label has tensor
  IDs but no duration samples. Unconditional (not gated on
  ``enable_diagnostics``) because untimed labels are silently dropped from
  strategy input and prod users need to see which layers were excluded.
* The inference diagnostics bundle — layer duration, memory transfer, and
  block assignment — emitted together whenever ``enable_diagnostics=True``.
  Fresh profiling retains its raw-sample distribution columns; restored state
  reports only its saved effective compute duration.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from flextensor._logging import DIAGNOSTICS_LOGGER_NAME
from flextensor.collectors import (
    IterativeLayerStatisticsCollector,
    LayerStatistics,
    TensorStatistics,
)
from flextensor.layer_statistics_analyzer import LayerStatisticsAnalyzer, report_profiling_quality
from flextensor.state_handler import TensorManagerState
from flextensor.strategy import GreedyStrategy
from flextensor.tensor_manager import TensorManager

_ANALYZER_LOGGER = "flextensor.layer_statistics_analyzer"


@pytest.fixture
def consistent_collector() -> IterativeLayerStatisticsCollector:
    """Collector whose only label has 5 identical samples (CV=0, count>=3)."""
    collector = IterativeLayerStatisticsCollector()
    for _ in range(5):
        collector.add_all("layer_a", {1}, 10.0)
    return collector


class TestUntimedTrapsEmission:
    """Untimed-trap warning fires unconditionally when any label is untimed."""

    def test_untimed_traps_warn_when_diagnostics_disabled(self, caplog) -> None:
        collector = IterativeLayerStatisticsCollector()
        for _ in range(5):
            collector.add_all("timed_layer", {1}, 10.0)
        collector.add_tensors("untimed_layer", {2})

        with caplog.at_level(logging.WARNING, logger=_ANALYZER_LOGGER):
            report_profiling_quality(collector)

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
            report_profiling_quality(collector)

        warnings = [r for r in caplog.records if r.name == _ANALYZER_LOGGER and r.levelno == logging.WARNING]
        assert any("untimed_layer" in r.getMessage() for r in warnings)

    def test_no_warning_when_all_traps_timed(self, consistent_collector, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=_ANALYZER_LOGGER):
            report_profiling_quality(consistent_collector)

        assert "no duration samples" not in caplog.text, "no untimed traps means no trap-report warning"


def _restored_manager(*, enable_diagnostics: bool) -> TensorManager:
    manager = TensorManager.__new__(TensorManager)
    manager.enable_diagnostics = enable_diagnostics
    manager.tensor_manager_load_strategy = GreedyStrategy()
    manager.loader_type = "allocation_block_transfer"
    manager._first_loader_non_destructive = False
    manager._replan_source_data = {}
    manager.tensors_map = {}
    tensor = TensorStatistics(tensor_id=1, name="weight", size_bytes=1024, load_time_ms=0.1)
    manager.tensor_manager_state = TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={1: "weight"},
        allocation_ordered={0: ["layer0"]},
        label_to_size_map={"layer0": 1024},
        block_sizes={0: 1024},
        load_strategy={"layer0": [tensor]},
        release_strategy={},
        label_to_block_id={"layer0": 0},
        stats=[LayerStatistics(label="layer0", tensors=[tensor], duration=1.0)],
        transfer_to_compute_map={"layer0": "layer0"},
        view_tensors_ids=[1],
        view_tensors_names=["weight"],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )
    manager._create_loader = MagicMock()
    return manager


def _diagnostics_manager(*, enable_diagnostics: bool) -> TensorManager:
    manager = _restored_manager(enable_diagnostics=enable_diagnostics)
    manager.stats = manager.tensor_manager_state.stats
    manager.load_strategy = manager.tensor_manager_state.load_strategy
    manager.memory_transfer_stats = {1024: 0.1}
    return manager


class TestFreshProfileTables:
    def test_tables_include_full_duration_distribution(self, consistent_collector, caplog) -> None:
        manager = _diagnostics_manager(enable_diagnostics=True)
        analyzer = LayerStatisticsAnalyzer(consistent_collector)

        with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER_NAME):
            manager._log_inference_diagnostics(block_data=None, duration_analyzer=analyzer)

        assert "Layer Duration Statistics (ms)" in caplog.text
        assert "Median" in caplog.text
        assert "Count" in caplog.text
        assert "Memory Transfer Statistics" in caplog.text
        assert "BLOCK ASSIGNMENT:" in caplog.text


class TestRestoredStateTables:
    def test_tables_emitted_when_diagnostics_enabled(self, caplog) -> None:
        manager = _restored_manager(enable_diagnostics=True)

        with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER_NAME):
            manager.prepare_infer_load_mode()

        assert manager.memory_transfer_stats == {1024: 0.1}
        assert "Layer Duration Statistics (ms)" in caplog.text
        assert "Compute" in caplog.text
        assert "Median" not in caplog.text
        assert "layer0" in caplog.text
        assert "1.000" in caplog.text
        assert "Memory Transfer Statistics" in caplog.text
        assert "BLOCK ASSIGNMENT: GreedyStrategy" in caplog.text

    def test_tables_suppressed_when_diagnostics_disabled(self, caplog) -> None:
        manager = _restored_manager(enable_diagnostics=False)

        with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER_NAME):
            manager.prepare_infer_load_mode()

        assert manager.memory_transfer_stats == {1024: 0.1}
        assert "Layer Duration Statistics" not in caplog.text
        assert "Memory Transfer Statistics" not in caplog.text
        assert "BLOCK ASSIGNMENT:" not in caplog.text
