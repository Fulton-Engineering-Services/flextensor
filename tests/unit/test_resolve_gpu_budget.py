# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for resolve_gpu_budget()."""

import logging
from unittest.mock import patch

import pytest

import flextensor.gpu_budget as gpu_budget
from flextensor.gpu_budget import resolve_gpu_budget

_GiB = 1 << 30


@patch("torch.cuda.memory_allocated", return_value=2 * _GiB)
@patch("torch.cuda.memory_reserved", return_value=7 * _GiB)
@patch("torch.cuda.mem_get_info", return_value=(20 * _GiB, 80 * _GiB))
def test_cuda_memory_snapshot_captures_raw_and_computed_counters(mock_info, mock_reserved, mock_allocated):
    snapshot = gpu_budget.CUDAMemorySnapshot.capture("cuda:2")

    assert snapshot.free_bytes == 20 * _GiB
    assert snapshot.total_bytes == 80 * _GiB
    assert snapshot.reserved_bytes == 7 * _GiB
    assert snapshot.allocated_bytes == 2 * _GiB
    assert snapshot.reusable_cache_bytes == 5 * _GiB
    assert snapshot.available_bytes == 25 * _GiB
    mock_info.assert_called_once_with("cuda:2")
    mock_reserved.assert_called_once_with("cuda:2")
    mock_allocated.assert_called_once_with("cuda:2")


def test_cuda_memory_snapshot_clamps_negative_reusable_cache() -> None:
    snapshot = gpu_budget.CUDAMemorySnapshot(
        free_bytes=20,
        total_bytes=80,
        reserved_bytes=2,
        allocated_bytes=7,
    )

    assert snapshot.reusable_cache_bytes == 0
    assert snapshot.available_bytes == 20


class TestResolveGpuBudget:
    """Tests for resolve_gpu_budget."""

    def test_returns_none_when_fraction_is_none(self):
        assert resolve_gpu_budget(None, "cpu") is None

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("torch.cuda.mem_get_info")
    def test_returns_fraction_of_total(self, mock_mem_get_info, _res, _alloc):
        mock_mem_get_info.return_value = (46 * 1024**3, 48 * 1024**3)  # 46 GiB free, 48 GiB total
        result = resolve_gpu_budget(0.9, "cpu")
        expected = int(48 * 1024**3 * 0.9)  # 43.2 GiB < 46 GiB available - no cap
        assert result == expected

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("torch.cuda.mem_get_info")
    def test_returns_int(self, mock_mem_get_info, _res, _alloc):
        mock_mem_get_info.return_value = (10 * 1024**3, 24 * 1024**3)
        result = resolve_gpu_budget(0.5, "cpu")
        assert isinstance(result, int)


class TestGpuBudgetCap:
    """Tests for GPU memory budget capping in resolve_gpu_budget()."""

    @patch("torch.cuda.memory_allocated", return_value=1 * _GiB)
    @patch("torch.cuda.memory_reserved", return_value=2 * _GiB)
    @patch("torch.cuda.mem_get_info", return_value=(38 * _GiB, 48 * _GiB))
    def test_budget_capped_when_available_less_than_fractional(self, _info, _res, _alloc):
        """Budget is reduced to available when available < total * fraction."""
        result = resolve_gpu_budget(0.9, "cpu")
        available = 38 * _GiB + (2 * _GiB - 1 * _GiB)
        assert result == available

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=1 * _GiB)
    @patch("torch.cuda.mem_get_info", return_value=(46 * _GiB, 48 * _GiB))
    def test_budget_unchanged_when_available_exceeds_fractional(self, _info, _res, _alloc):
        """Budget passes through when available >= total * fraction."""
        result = resolve_gpu_budget(0.5, "cpu")
        expected = int(48 * _GiB * 0.5)
        assert result == expected

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("torch.cuda.mem_get_info", return_value=(100 * 1024**2, 48 * _GiB))
    def test_runtime_error_when_available_below_minimum(self, _info, _res, _alloc):
        """RuntimeError raised when available GPU memory < 256 MiB."""
        with pytest.raises(RuntimeError, match="Insufficient free GPU memory"):
            resolve_gpu_budget(0.9, "cpu")

    @patch("torch.cuda.mem_get_info")
    def test_latency_mode_skips_cuda_queries(self, mock_info):
        """No CUDA memory query when fraction is None (latency mode)."""
        result = resolve_gpu_budget(None, "cpu")
        assert result is None
        mock_info.assert_not_called()

    @patch("torch.cuda.memory_allocated", return_value=1 * _GiB)
    @patch("torch.cuda.memory_reserved", return_value=2 * _GiB)
    @patch("torch.cuda.mem_get_info", return_value=(38 * _GiB, 48 * _GiB))
    def test_warning_logged_when_budget_capped(self, _info, _res, _alloc, caplog):
        """Warning with memory breakdown logged when budget is capped."""
        budget_logger = logging.getLogger("flextensor.tensor_manager")
        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_manager"):
            resolve_gpu_budget(0.9, "cpu", logger=budget_logger)
        assert "Capping GPU memory budget" in caplog.text
