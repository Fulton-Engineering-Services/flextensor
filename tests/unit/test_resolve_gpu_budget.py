# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TensorManager._resolve_gpu_budget()."""

from unittest.mock import MagicMock, patch


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

    @patch("torch.cuda.mem_get_info")
    def test_returns_fraction_of_total(self, mock_mem_get_info):
        mock_mem_get_info.return_value = (30 * 1024**3, 48 * 1024**3)  # 30 GiB free, 48 GiB total
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.9)
        result = tm._resolve_gpu_budget()
        expected = int(48 * 1024**3 * 0.9)
        assert result == expected

    @patch("torch.cuda.mem_get_info")
    def test_returns_int(self, mock_mem_get_info):
        mock_mem_get_info.return_value = (10 * 1024**3, 24 * 1024**3)
        tm = self._make_tensor_manager(max_gpu_mem_fraction=0.5)
        result = tm._resolve_gpu_budget()
        assert isinstance(result, int)
