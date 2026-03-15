# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for host resource snapshot capture.

This test suite validates:
1. capture_host_resources returns all expected keys
2. All returned values are integers
3. Mocked psutil produces exact expected values
"""

from unittest.mock import MagicMock, patch

from flextensor.instrumentation.host_resources import capture_host_resources

EXPECTED_KEYS = {
    "host_memory_total",
    "host_memory_used",
    "host_memory_available",
    "swap_total",
    "swap_used",
    "swap_free",
}


class TestCaptureHostResources:
    """Tests for capture_host_resources function."""

    def test_returns_all_expected_keys(self) -> None:
        """Test that the snapshot dictionary contains all expected keys."""
        result = capture_host_resources()
        assert set(result.keys()) == EXPECTED_KEYS

    def test_all_values_are_integers(self) -> None:
        """Test that every value in the snapshot is an integer."""
        result = capture_host_resources()
        for key, value in result.items():
            assert isinstance(value, int), f"Expected int for '{key}', got {type(value).__name__}"

    @patch("flextensor.instrumentation.host_resources.psutil")
    def test_exact_values_with_mocked_psutil(self, mock_psutil: MagicMock) -> None:
        """Test that mocked psutil values produce the exact expected output."""
        mock_vmem = MagicMock()
        mock_vmem.total = 34_359_738_368
        mock_vmem.used = 17_179_869_184
        mock_vmem.available = 17_179_869_184
        mock_psutil.virtual_memory.return_value = mock_vmem

        mock_smem = MagicMock()
        mock_smem.total = 8_589_934_592
        mock_smem.used = 1_073_741_824
        mock_smem.free = 7_516_192_768
        mock_psutil.swap_memory.return_value = mock_smem

        result = capture_host_resources()

        assert result["host_memory_total"] == 34_359_738_368
        assert result["host_memory_used"] == 17_179_869_184
        assert result["host_memory_available"] == 17_179_869_184
        assert result["swap_total"] == 8_589_934_592
        assert result["swap_used"] == 1_073_741_824
        assert result["swap_free"] == 7_516_192_768
