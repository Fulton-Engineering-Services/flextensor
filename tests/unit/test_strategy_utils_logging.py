# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for block-table logging routed through flextensor.diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flextensor._logging import DIAGNOSTICS_LOGGER_NAME
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy.utils import log_block_table

if TYPE_CHECKING:
    import pytest


def _make_layer_stats() -> list[LayerStatistics]:
    tensor = TensorStatistics(tensor_id=1, name="t0", size_bytes=1024, load_time_ms=0.1)
    return [LayerStatistics(label="layer0", duration=1.0, tensors=[tensor])]


def test_block_table_uses_diagnostics_logger(caplog: pytest.LogCaptureFixture) -> None:
    layer_stats = _make_layer_stats()
    strategy_map = {"layer0": []}

    with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER_NAME):
        log_block_table(layer_stats, strategy_map, block_data=None, strategy_name="TestStrategy")

    records = [r for r in caplog.records if r.name == DIAGNOSTICS_LOGGER_NAME]
    assert records, "log_block_table did not emit on flextensor.diagnostics"
    assert any("BLOCK ASSIGNMENT: TestStrategy" in r.getMessage() for r in records)


def test_block_table_logger_has_no_private_handlers() -> None:
    # Regression guard: the old `flextensor.block_table` logger with a direct
    # StreamHandler must not be present.
    legacy = logging.getLogger("flextensor.block_table")
    assert not any(isinstance(h, logging.StreamHandler) for h in legacy.handlers), (
        "flextensor.block_table must not have a direct StreamHandler installed by FT"
    )
