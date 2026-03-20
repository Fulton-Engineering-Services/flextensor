# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TensorManager._resolve_gpu_budget()."""

import logging
from unittest.mock import MagicMock, patch

import pytest

_GiB = 1 << 30


class TestResolveGpuBudget:
    """Tests for TensorManager._resolve_gpu_budget."""

    def _make_tensor_manager(self, max_gpu_mem_fraction=None):
        """Create a minimal TensorManager with the given fraction."""
        from flextensor.tensor_manager import TensorManager

        strategy = MagicMock()
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.Event", return_value=MagicMock()),
        ):
            tm = TensorManager(
                device_gpu="cpu",
                pinned_memory=False,
                tensor_manager_load_strategy=strategy,
                max_gpu_mem_fraction=max_gpu_mem_fraction,
            )
        return tm

    def test_returns_none_when_fraction_is_none(self):
        tm = self._make_tensor_manager(max_gpu_mem_fraction=None)
        assert tm._resolve_gpu_budget() is None

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("torch.cuda.mem_get_info")
    def test_returns_fraction_of_total(self, mock_mem_get_info, _res, _alloc):
        mock_mem_get_info.return_value = (46 * 1024**3, 48 * 1024**3)  # 46 GiB free, 48 GiB total
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.9)
        result = tm._resolve_gpu_budget()
        expected = int(48 * 1024**3 * 0.9)  # 43.2 GiB < 46 GiB available — no cap
        assert result == expected

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("torch.cuda.mem_get_info")
    def test_returns_int(self, mock_mem_get_info, _res, _alloc):
        mock_mem_get_info.return_value = (10 * 1024**3, 24 * 1024**3)
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.5)
        result = tm._resolve_gpu_budget()
        assert isinstance(result, int)


class TestGpuBudgetCap:
    """Tests for GPU memory budget capping in _resolve_gpu_budget()."""

    def _make_tensor_manager(self, max_gpu_mem_fraction=None):
        """Create a minimal TensorManager with the given fraction."""
        from flextensor.tensor_manager import TensorManager

        strategy = MagicMock()
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.Event", return_value=MagicMock()),
        ):
            tm = TensorManager(
                device_gpu="cpu",
                pinned_memory=False,
                tensor_manager_load_strategy=strategy,
                max_gpu_mem_fraction=max_gpu_mem_fraction,
            )
        return tm

    @patch("torch.cuda.memory_allocated", return_value=1 * _GiB)
    @patch("torch.cuda.memory_reserved", return_value=2 * _GiB)
    @patch("torch.cuda.mem_get_info", return_value=(38 * _GiB, 48 * _GiB))
    def test_budget_capped_when_available_less_than_fractional(self, _info, _res, _alloc):
        """Budget is reduced to available when available < total * fraction."""
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.9)
        result = tm._resolve_gpu_budget()
        available = 38 * _GiB + (2 * _GiB - 1 * _GiB)
        assert result == available

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=1 * _GiB)
    @patch("torch.cuda.mem_get_info", return_value=(46 * _GiB, 48 * _GiB))
    def test_budget_unchanged_when_available_exceeds_fractional(self, _info, _res, _alloc):
        """Budget passes through when available >= total * fraction."""
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.5)
        result = tm._resolve_gpu_budget()
        expected = int(48 * _GiB * 0.5)
        assert result == expected

    @patch("torch.cuda.memory_allocated", return_value=0)
    @patch("torch.cuda.memory_reserved", return_value=0)
    @patch("torch.cuda.mem_get_info", return_value=(100 * 1024**2, 48 * _GiB))
    def test_runtime_error_when_available_below_minimum(self, _info, _res, _alloc):
        """RuntimeError raised when available GPU memory < 256 MiB."""
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.9)
        with pytest.raises(RuntimeError, match="Insufficient free GPU memory"):
            tm._resolve_gpu_budget()

    @patch("torch.cuda.mem_get_info")
    def test_latency_mode_skips_cuda_queries(self, mock_info):
        """No CUDA memory query when fraction is None (latency mode)."""
        tm = self._make_tensor_manager(max_gpu_mem_fraction=None)
        result = tm._resolve_gpu_budget()
        assert result is None
        mock_info.assert_not_called()

    @patch("torch.cuda.memory_allocated", return_value=1 * _GiB)
    @patch("torch.cuda.memory_reserved", return_value=2 * _GiB)
    @patch("torch.cuda.mem_get_info", return_value=(38 * _GiB, 48 * _GiB))
    def test_warning_logged_when_budget_capped(self, _info, _res, _alloc, caplog):
        """Warning with memory breakdown logged when budget is capped."""
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.9)
        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_manager"):
            tm._resolve_gpu_budget()
        assert "Capping GPU memory budget" in caplog.text
