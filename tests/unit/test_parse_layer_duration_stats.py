# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for parse_layer_duration_stats in vllm_utils."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# requests is a runtime dependency of vllm_utils but unused by parse_layer_duration_stats;
# stub it out so the module loads without a network-capable environment.
sys.modules.setdefault("requests", MagicMock())

_vllm_utils_path = Path(__file__).parent.parent / "integration" / "_vllm_utils.py"
_spec = importlib.util.spec_from_file_location("_vllm_utils_for_test", _vllm_utils_path)
assert _spec is not None and _spec.loader is not None, f"Could not load {_vllm_utils_path}"
_vllm_utils_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vllm_utils_mod)  # type: ignore[union-attr]
parse_layer_duration_stats = _vllm_utils_mod.parse_layer_duration_stats


def _make_table(*data_rows: str, header: bool = True) -> list[str]:
    """Build a minimal log snippet containing a Layer Duration Statistics table."""
    lines: list[str] = ["Layer Duration Statistics", "=" * 60]
    if header:
        lines.append("Layer  min  max  median  avg  std  cv  count")
    lines.extend(data_rows)
    lines.append("")  # blank line terminates the table
    return lines


_ROW_A = "model.layers.0  1.20  3.40  2.10  2.20  0.50  22.7%  10"
_ROW_B = "model.layers.1  1.10  3.30  2.00  2.10  0.40  19.0%  10"
_ROW_NORM = "model.norm  0.50  1.00  0.70  0.70  0.10  14.3%  8"


class TestParseLayerDurationStats:
    """Tests for parse_layer_duration_stats."""

    def test_nominal_parses_all_rows(self):
        """Basic table with column header and multiple data rows is parsed correctly."""
        lines = _make_table(_ROW_A, _ROW_B, _ROW_NORM)
        result = parse_layer_duration_stats(lines)

        assert set(result) == {"model.layers.0", "model.layers.1", "model.norm"}
        assert result["model.layers.0"]["min"] == pytest.approx(1.20)
        assert result["model.layers.0"]["max"] == pytest.approx(3.40)
        assert result["model.layers.0"]["median"] == pytest.approx(2.10)
        assert result["model.layers.0"]["avg"] == pytest.approx(2.20)
        assert result["model.layers.0"]["std"] == pytest.approx(0.50)
        assert result["model.layers.0"]["cv"] == pytest.approx(22.7)
        assert result["model.layers.0"]["count"] == 10

    def test_cv_is_percentage_not_ratio(self):
        """cv field stores 22.7, not 0.227 — guards against unit confusion."""
        lines = _make_table(_ROW_A)
        result = parse_layer_duration_stats(lines)
        assert result["model.layers.0"]["cv"] == pytest.approx(22.7)

    def test_empty_log_returns_empty_dict(self):
        result = parse_layer_duration_stats(["Some unrelated output", "No table here"])
        assert result == {}

    def test_empty_list_returns_empty_dict(self):
        assert parse_layer_duration_stats([]) == {}

    def test_ansi_codes_stripped(self):
        """ANSI escape sequences in layer names and values are removed before parsing."""
        lines = [
            "\x1b[32mINFO\x1b[0m Layer Duration Statistics",
            "=" * 60,
            "\x1b[1mmodel.layers.0\x1b[0m  1.20  3.40  2.10  2.20  0.50  22.7%  10",
        ]
        result = parse_layer_duration_stats(lines)
        assert "model.layers.0" in result
        assert result["model.layers.0"]["count"] == 10

    def test_vllm_pid_prefix_stripped(self):
        """(ProcessName pid=N) prefix emitted by vLLM's logger is stripped."""
        lines = [
            "(WorkerProcess pid=12345) Layer Duration Statistics",
            "(WorkerProcess pid=12345) =" * 5,
            f"(WorkerProcess pid=12345) {_ROW_A}",
        ]
        result = parse_layer_duration_stats(lines)
        assert "model.layers.0" in result

    def test_ft_timestamp_prefix_stripped(self):
        """[YYYY-MM-DD ...] LEVEL module.py:N: prefix from FT logger is stripped."""
        lines = [
            "[2026-01-15 12:34:56,789] INFO vllm_utils.py:150: Layer Duration Statistics",
            "=" * 60,
            f"[2026-01-15 12:34:56,791] INFO vllm_utils.py:155: {_ROW_A}",
        ]
        result = parse_layer_duration_stats(lines)
        assert "model.layers.0" in result

    def test_vllm_record_prefix_stripped(self):
        """LEVEL MM-DD HH:MM:SS [file.py:LINE] prefix from vLLM's formatter is stripped.

        Records propagated through the vLLM bridge (both INFO from the diagnostics
        logger and WARNING direct from flextensor) arrive with this prefix; without
        stripping it the row regex never matches and the table comes back empty.
        """
        info_prefix = "(EngineCore_DP0 pid=951) INFO 04-21 10:22:35 [flextensor/tensor_manager.py:898] "
        warn_prefix = "(EngineCore_DP0 pid=951) WARNING 04-21 10:22:35 [flextensor/layer_statistics_analyzer.py:184] "
        lines = [
            f"{info_prefix}Layer Duration Statistics (ms)",
            info_prefix + "=" * 60,
            f"{info_prefix}{_ROW_A}",
            f"{warn_prefix}{_ROW_B}",
        ]
        result = parse_layer_duration_stats(lines)
        assert "model.layers.0" in result
        assert "model.layers.1" in result

    def test_layer_prefix_data_rows_not_skipped(self):
        """Data rows whose name begins with 'Layer' must not be treated as column headers."""
        lines = _make_table(
            "Layer  min  max  median  avg  std  cv  count",  # column header — skip
            "LayerNorm.0  1.00  2.00  1.50  1.50  0.20  13.3%  5",  # data row — keep
            "LayerScale.1  0.80  1.80  1.20  1.20  0.15  12.5%  5",  # data row — keep
            header=False,
        )
        result = parse_layer_duration_stats(lines)
        assert "LayerNorm.0" in result
        assert "LayerScale.1" in result
        assert result["LayerNorm.0"]["count"] == 5

    def test_table_ends_at_non_data_line(self):
        """A non-empty non-separator line terminates parsing; subsequent data rows are ignored."""
        lines = [
            "Layer Duration Statistics",
            "=" * 60,
            _ROW_A,
            "INFO Some other log message",  # non-data line ends the table
            "model.extra  1.0  2.0  1.5  1.5  0.2  13.3%  5",  # must be ignored
        ]
        result = parse_layer_duration_stats(lines)
        assert "model.layers.0" in result
        assert "model.extra" not in result
