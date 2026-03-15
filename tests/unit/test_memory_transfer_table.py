# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for format_memory_transfer_table."""

from flextensor.memory_transfer_benchmark import format_memory_transfer_table


class TestFormatMemoryTransferTable:
    """Tests for format_memory_transfer_table()."""

    def test_empty_dict_returns_message(self):
        """Empty stats dict produces a 'no data' message."""
        result = format_memory_transfer_table({})
        assert result == "No memory transfer statistics available."

    def test_single_entry(self):
        """Single entry renders correctly with bandwidth."""
        # 1 MiB in 1.0 ms → bandwidth = 1048576 / (0.001) / 1e9 ≈ 1.049 GB/s
        result = format_memory_transfer_table({1048576: 1.0})
        assert "1048576" in result
        assert "1.0 MiB" in result
        assert "1.000" in result  # transfer time
        assert "1.049" in result  # bandwidth GB/s

    def test_multiple_entries_sorted_by_size(self):
        """Multiple entries are sorted by tensor size ascending."""
        stats = {4096: 0.01, 1024: 0.005, 1048576: 0.5}
        result = format_memory_transfer_table(stats)
        lines = result.split("\n")
        # Find data rows (skip header/separator lines)
        data_rows = [
            line
            for line in lines
            if line.strip()
            and not line.startswith("=")
            and not line.startswith("-")
            and "Size" not in line
            and "Memory" not in line
        ]
        # First data row should be 1024, last should be 1048576
        assert "1024" in data_rows[0]
        assert "1048576" in data_rows[-1]

    def test_table_has_header_and_footer(self):
        """Table has = header/footer lines matching existing diagnostic style."""
        result = format_memory_transfer_table({1024: 0.01})
        lines = result.split("\n")
        assert lines[0].startswith("=")
        assert lines[-1].startswith("=")

    def test_human_readable_sizes(self):
        """Sizes are formatted in human-readable units."""
        stats = {512: 0.01, 1024: 0.01, 1048576: 0.5, 1073741824: 50.0}
        result = format_memory_transfer_table(stats)
        assert "512 B" in result
        assert "1.0 KiB" in result
        assert "1.0 MiB" in result
        assert "1.0 GiB" in result
