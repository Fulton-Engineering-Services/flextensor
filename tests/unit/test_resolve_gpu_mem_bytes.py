# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resolve_gpu_mem_bytes budget helper."""

from unittest.mock import MagicMock, patch

import pytest

from flextensor.config import OffloadConfig
from flextensor.gpu_budget import resolve_gpu_mem_bytes


class TestResolveGpuMemBytes:
    """Tests for resolve_gpu_mem_bytes()."""

    def test_fraction_resolved_to_bytes(self):
        """Fraction is multiplied by total GPU memory from device properties."""
        config = OffloadConfig(max_gpu_mem_fraction=0.8)
        fake_props = MagicMock()
        fake_props.total_memory = 80 * 1024**3  # 80 GB

        with patch("flextensor.gpu_budget.torch.cuda.get_device_properties", return_value=fake_props) as mock_get:
            result = resolve_gpu_mem_bytes(config)

        assert result == int(0.8 * 80 * 1024**3)
        mock_get.assert_called_once_with(0)  # default gpu_device=0

    def test_fraction_with_custom_gpu_device(self):
        """gpu_device index from config is passed to get_device_properties."""
        config = OffloadConfig(max_gpu_mem_fraction=0.5, gpu_device=3)
        fake_props = MagicMock()
        fake_props.total_memory = 48 * 1024**3

        with patch("flextensor.gpu_budget.torch.cuda.get_device_properties", return_value=fake_props) as mock_get:
            result = resolve_gpu_mem_bytes(config)

        assert result == int(0.5 * 48 * 1024**3)
        mock_get.assert_called_once_with(3)

    def test_fraction_none_returns_none(self):
        """Fraction=None (latency mode) returns None without querying GPU."""
        config = OffloadConfig(max_gpu_mem_fraction=None)

        with patch("flextensor.gpu_budget.torch.cuda.get_device_properties") as mock_get:
            result = resolve_gpu_mem_bytes(config)

        assert result is None
        mock_get.assert_not_called()

    def test_deprecated_bytes_returned_when_fraction_none(self):
        """When fraction is None due to deprecated bytes path, returns the bytes value."""
        with pytest.warns(DeprecationWarning):
            config = OffloadConfig(max_gpu_mem_bytes=20 * 1024**3)

        with patch("flextensor.gpu_budget.torch.cuda.get_device_properties") as mock_get:
            result = resolve_gpu_mem_bytes(config)

        assert result == 20 * 1024**3
        mock_get.assert_not_called()

    def test_cuda_query_failure_raises_runtime_error(self):
        """RuntimeError from get_device_properties is re-raised with context."""
        config = OffloadConfig(max_gpu_mem_fraction=0.9)

        with (
            patch(
                "flextensor.gpu_budget.torch.cuda.get_device_properties",
                side_effect=RuntimeError("no CUDA"),
            ),
            pytest.raises(RuntimeError, match="Failed to query GPU device"),
        ):
            resolve_gpu_mem_bytes(config)

    def test_context_included_in_error_message(self):
        """The context string appears in the error message on failure."""
        config = OffloadConfig(max_gpu_mem_fraction=0.9)

        with (
            patch(
                "flextensor.gpu_budget.torch.cuda.get_device_properties",
                side_effect=RuntimeError("no CUDA"),
            ),
            pytest.raises(RuntimeError, match="computing SHM namespace"),
        ):
            resolve_gpu_mem_bytes(config, context="computing SHM namespace")
