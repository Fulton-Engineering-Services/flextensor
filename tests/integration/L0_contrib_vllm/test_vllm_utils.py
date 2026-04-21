# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style tests for the integration-test helpers in vllm_utils.py."""

from __future__ import annotations

import pytest
from vllm_utils import parse_block_assignment_layers, parse_log_level

# Piggyback on the GPU-memory tier the rest of this integration directory
# already requests so these fast string-parsing tests share the runner instead
# of defaulting to the highest-tier unmarked runner.
pytestmark = pytest.mark.gpu_mem_40g


def test_parse_log_level_info() -> None:
    assert parse_log_level("(EngineCore pid=123) INFO 04-18 22:28:47 [worker.py:131] hello") == "INFO"


def test_parse_log_level_warning() -> None:
    assert parse_log_level("(EngineCore pid=123) WARNING 04-18 22:28:47 [x.py:1] hi") == "WARNING"


def test_parse_log_level_without_prefix() -> None:
    assert parse_log_level("INFO 04-18 22:28:47 [x.py:1] hi") == "INFO"


def test_parse_log_level_with_vllm_tag() -> None:
    assert parse_log_level("[vLLM] (EngineCore pid=123) INFO foo") == "INFO"


def test_parse_log_level_none_on_plain_text() -> None:
    assert parse_log_level("(EngineCore pid=123) some message without a level") is None


_BLOCK_ASSIGNMENT_FIXTURE = """
some unrelated log line

====================
BLOCK ASSIGNMENT: KnapsackStrategy
====================
Layer        Layer Size    Offload   Transfer | C.Blk T.Blk   Blk Size | Pipeline                       Compute
--------------------------------------------------------------------------------
model.layers.0    123.45MB    -   45.00MB |     0     1    150.00MB | fill blk 1 (1st transfer)       4.20ms
model.layers.1    123.45MB  45.00MB   50.00MB |     1     2    150.00MB | read blk 1, fill blk 2       4.10ms
model.layers.2    123.45MB  50.00MB       -   |     2     -          - | read blk 2                    4.30ms
--------------------------------------------------------------------------------
Total: layer_size=370.34MB, offload=95.00MB, compute=12.60ms
Compute: min=4.10ms, max=4.30ms, avg=4.20ms
Block Sizes:
  Block 0:   150.00MB  (transfers=1: model.layers.0 | computes=1: model.layers.0)
  Block 1:   150.00MB  (transfers=1: model.layers.1 | computes=1: model.layers.1)

other log stuff after the table
"""


def test_parse_block_assignment_layers_returns_labels_in_order() -> None:
    layers = parse_block_assignment_layers(_BLOCK_ASSIGNMENT_FIXTURE.splitlines())
    assert layers == ["model.layers.0", "model.layers.1", "model.layers.2"]


def test_parse_block_assignment_layers_empty_when_no_table() -> None:
    assert parse_block_assignment_layers(["no table here", "just noise"]) == []


def test_parse_block_assignment_layers_tolerates_vllm_pid_prefix() -> None:
    prefixed = [f"(EngineCore pid=123) {line}" for line in _BLOCK_ASSIGNMENT_FIXTURE.splitlines()]
    assert parse_block_assignment_layers(prefixed) == ["model.layers.0", "model.layers.1", "model.layers.2"]


def test_parse_block_assignment_layers_stops_at_trailing_content() -> None:
    # "other log stuff after the table" must not be captured as a layer.
    layers = parse_block_assignment_layers(_BLOCK_ASSIGNMENT_FIXTURE.splitlines())
    assert "other" not in layers
