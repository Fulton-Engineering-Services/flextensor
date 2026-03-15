# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for host_resources integration in dump_to_file output.

This test suite validates:
1. dump_to_file includes a host_resources key in the JSON output
2. The host_resources values match the mocked psutil data
3. capture_host_resources is importable from the instrumentation package
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from flextensor.instrumentation.dumper import dump_to_file


class TestDumpToFileHostResources:
    """Tests that dump_to_file includes host_resources in JSON output."""

    @patch("flextensor.instrumentation.dumper.capture_host_resources")
    @patch("flextensor.instrumentation.dumper.InstrumentationRegistry")
    def test_dump_includes_host_resources_key(
        self,
        mock_registry_cls: MagicMock,
        mock_capture: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that the JSON output contains a host_memory key."""
        mock_registry = MagicMock()
        mock_registry.get_records.return_value = []
        mock_registry_cls.get_instance.return_value = mock_registry

        mock_capture.return_value = {
            "host_memory_total": 68_719_476_736,
            "host_memory_used": 34_359_738_368,
            "host_memory_available": 34_359_738_368,
            "swap_total": 0,
            "swap_used": 0,
            "swap_free": 0,
        }

        output_file = tmp_path / "test_dump.json"
        dump_to_file(output_file)

        data = json.loads(output_file.read_text())
        assert "host_memory" in data

    @patch("flextensor.instrumentation.dumper.capture_host_resources")
    @patch("flextensor.instrumentation.dumper.InstrumentationRegistry")
    def test_dump_host_resources_values_match_mock(
        self,
        mock_registry_cls: MagicMock,
        mock_capture: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that host_resources values in the JSON match the mocked data."""
        mock_registry = MagicMock()
        mock_registry.get_records.return_value = []
        mock_registry_cls.get_instance.return_value = mock_registry

        expected = {
            "host_memory_total": 137_438_953_472,
            "host_memory_used": 103_079_215_104,
            "host_memory_available": 34_359_738_368,
            "swap_total": 8_589_934_592,
            "swap_used": 2_147_483_648,
            "swap_free": 6_442_450_944,
        }
        mock_capture.return_value = expected

        output_file = tmp_path / "test_dump.json"
        dump_to_file(output_file)

        data = json.loads(output_file.read_text())
        assert data["host_memory"] == expected

    @patch("flextensor.instrumentation.dumper.capture_host_resources")
    @patch("flextensor.instrumentation.dumper.InstrumentationRegistry")
    def test_dump_retains_existing_keys(
        self,
        mock_registry_cls: MagicMock,
        mock_capture: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that timestamp, flextensor_version, and components are still present."""
        mock_registry = MagicMock()
        mock_registry.get_records.return_value = []
        mock_registry_cls.get_instance.return_value = mock_registry

        mock_capture.return_value = {
            "host_memory_total": 34_359_738_368,
            "host_memory_used": 17_179_869_184,
            "host_memory_available": 17_179_869_184,
            "swap_total": 0,
            "swap_used": 0,
            "swap_free": 0,
        }

        output_file = tmp_path / "test_dump.json"
        dump_to_file(output_file)

        data = json.loads(output_file.read_text())
        assert "timestamp" in data
        assert "flextensor_version" in data
        assert "components" in data
        assert "host_memory" in data


class TestCaptureHostResourcesExport:
    """Tests that capture_host_resources is exported from the instrumentation package."""

    def test_capture_host_resources_importable_from_package(self) -> None:
        """Test that capture_host_resources can be imported from flextensor.instrumentation."""
        from flextensor.instrumentation import capture_host_resources

        assert callable(capture_host_resources)

    def test_capture_host_resources_in_package_all(self) -> None:
        """Test that capture_host_resources is listed in __all__."""
        import flextensor.instrumentation as instr

        assert "capture_host_resources" in instr.__all__
