# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for memory_transfer_interpolator module."""

import pytest

from flextensor.memory_transfer_interpolator import (
    MemoryTransferInterpolator,
    create_interpolator_from_benchmarks,
)


class TestMemoryTransferInterpolatorInit:
    """Test cases for MemoryTransferInterpolator initialization."""

    def test_init_with_valid_data(self):
        """Test initialization with valid memory transfer data."""
        memory_transfers = {
            1024: 0.1,  # 1KB -> 0.1ms
            1024 * 1024: 1.0,  # 1MB -> 1.0ms
            1024 * 1024 * 1024: 100.0,  # 1GB -> 100.0ms
        }
        interpolator = MemoryTransferInterpolator(memory_transfers)

        assert interpolator.min_bytes == 1024
        assert interpolator.max_bytes == 1024 * 1024 * 1024
        assert interpolator.min_duration == 0.1
        assert interpolator.max_duration == 100.0

    def test_init_with_empty_dict_raises_error(self):
        """Test that empty dictionary raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            MemoryTransferInterpolator({})

    def test_init_with_single_data_point(self):
        """Test initialization with single data point."""
        memory_transfers = {1024 * 1024: 1.0}
        interpolator = MemoryTransferInterpolator(memory_transfers)

        assert interpolator.min_bytes == 1024 * 1024
        assert interpolator.max_bytes == 1024 * 1024
        assert len(interpolator.bytes_array) == 1

    def test_init_sorts_by_bytes(self):
        """Test that data is sorted by bytes."""
        # Provide data in unsorted order
        memory_transfers = {
            1024 * 1024: 1.0,
            1024: 0.1,
            1024 * 1024 * 1024: 100.0,
        }
        interpolator = MemoryTransferInterpolator(memory_transfers)

        # Verify sorted order
        assert interpolator.bytes_array[0] == 1024
        assert interpolator.bytes_array[1] == 1024 * 1024
        assert interpolator.bytes_array[2] == 1024 * 1024 * 1024


class TestBytesToDuration:
    """Test cases for bytes_to_duration method."""

    @pytest.fixture
    def interpolator(self):
        """Create a standard interpolator for tests."""
        memory_transfers = {
            1024: 0.01,  # 1KB -> 0.01ms
            1024 * 1024: 1.0,  # 1MB -> 1.0ms
            1024 * 1024 * 1024: 1000.0,  # 1GB -> 1000ms
        }
        return MemoryTransferInterpolator(memory_transfers)

    def test_bytes_to_duration_exact_match(self, interpolator):
        """Test duration for exact data point."""
        duration = interpolator.bytes_to_duration(float(1024 * 1024))
        assert abs(duration - 1.0) < 0.01

    def test_bytes_to_duration_interpolation(self, interpolator):
        """Test interpolation between data points."""
        # Value between 1MB and 1GB
        duration = interpolator.bytes_to_duration(float(100 * 1024 * 1024))  # 100MB
        # Should be between 1.0ms and 1000ms
        assert 1.0 < duration < 1000.0

    def test_bytes_to_duration_extrapolation_below(self, interpolator):
        """Test extrapolation below minimum."""
        duration = interpolator.bytes_to_duration(512.0)  # Below 1KB
        # Should be less than minimum duration
        assert duration < 0.01

    def test_bytes_to_duration_extrapolation_above(self, interpolator):
        """Test extrapolation above maximum."""
        duration = interpolator.bytes_to_duration(float(2 * 1024 * 1024 * 1024))  # 2GB
        # Should be greater than maximum duration
        assert duration > 1000.0

    def test_bytes_to_duration_zero_raises_error(self, interpolator):
        """Test that zero bytes raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            interpolator.bytes_to_duration(0.0)

    def test_bytes_to_duration_negative_raises_error(self, interpolator):
        """Test that negative bytes raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            interpolator.bytes_to_duration(-1024.0)

    def test_bytes_to_duration_single_data_point(self):
        """Test bytes_to_duration with single data point returns constant."""
        interpolator = MemoryTransferInterpolator({1024 * 1024: 5.0})

        # Any size should return the single known duration
        assert interpolator.bytes_to_duration(512.0) == 5.0
        assert interpolator.bytes_to_duration(float(1024 * 1024)) == 5.0
        assert interpolator.bytes_to_duration(float(1024 * 1024 * 1024)) == 5.0


class TestDurationToBytes:
    """Test cases for duration_to_bytes method."""

    @pytest.fixture
    def interpolator(self):
        """Create a standard interpolator for tests."""
        memory_transfers = {
            1024: 0.01,  # 1KB -> 0.01ms
            1024 * 1024: 1.0,  # 1MB -> 1.0ms
            1024 * 1024 * 1024: 1000.0,  # 1GB -> 1000ms
        }
        return MemoryTransferInterpolator(memory_transfers)

    def test_duration_to_bytes_exact_match(self, interpolator):
        """Test bytes for exact data point."""
        size = interpolator.duration_to_bytes(1.0)
        assert abs(size - 1024 * 1024) < 1024  # Within 1KB tolerance

    def test_duration_to_bytes_interpolation(self, interpolator):
        """Test interpolation between data points."""
        # Value between 1.0ms and 1000ms
        size = interpolator.duration_to_bytes(100.0)
        # Should be between 1MB and 1GB
        assert 1024 * 1024 < size < 1024 * 1024 * 1024

    def test_duration_to_bytes_extrapolation_below(self, interpolator):
        """Test extrapolation below minimum duration."""
        size = interpolator.duration_to_bytes(0.005)  # Below 0.01ms
        # Should be less than minimum bytes
        assert size < 1024

    def test_duration_to_bytes_extrapolation_above(self, interpolator):
        """Test extrapolation above maximum duration."""
        size = interpolator.duration_to_bytes(2000.0)  # Above 1000ms
        # Should be greater than maximum bytes
        assert size > 1024 * 1024 * 1024

    def test_duration_to_bytes_zero_raises_error(self, interpolator):
        """Test that zero duration raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            interpolator.duration_to_bytes(0.0)

    def test_duration_to_bytes_negative_raises_error(self, interpolator):
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            interpolator.duration_to_bytes(-1.0)

    def test_duration_to_bytes_single_data_point(self):
        """Test duration_to_bytes with single data point returns constant."""
        interpolator = MemoryTransferInterpolator({1024 * 1024: 5.0})

        # Any duration should return the single known bytes value
        assert interpolator.duration_to_bytes(1.0) == 1024 * 1024
        assert interpolator.duration_to_bytes(5.0) == 1024 * 1024
        assert interpolator.duration_to_bytes(100.0) == 1024 * 1024


class TestBidirectionalConsistency:
    """Test that bytes_to_duration and duration_to_bytes are inverses."""

    @pytest.fixture
    def interpolator(self):
        """Create a standard interpolator for tests."""
        memory_transfers = {
            1024: 0.01,
            1024 * 1024: 1.0,
            1024 * 1024 * 1024: 1000.0,
        }
        return MemoryTransferInterpolator(memory_transfers)

    def test_round_trip_bytes(self, interpolator):
        """Test that bytes -> duration -> bytes returns approximately original."""
        original_bytes = float(50 * 1024 * 1024)  # 50MB
        duration = interpolator.bytes_to_duration(original_bytes)
        recovered_bytes = interpolator.duration_to_bytes(duration)

        # Should be within 1% tolerance
        relative_error = abs(recovered_bytes - original_bytes) / original_bytes
        assert relative_error < 0.01

    def test_round_trip_duration(self, interpolator):
        """Test that duration -> bytes -> duration returns approximately original."""
        original_duration = 50.0  # 50ms
        size = interpolator.duration_to_bytes(original_duration)
        recovered_duration = interpolator.bytes_to_duration(size)

        # Should be within 1% tolerance
        relative_error = abs(recovered_duration - original_duration) / original_duration
        assert relative_error < 0.01


class TestGetBandwidthMbps:
    """Test cases for get_bandwidth_mbps method."""

    def test_get_bandwidth_mbps(self):
        """Test bandwidth calculation."""
        # 1GB in 1 second = 1024 MB/s
        memory_transfers = {
            1024 * 1024 * 1024: 1000.0,  # 1GB in 1000ms (1 second)
        }
        interpolator = MemoryTransferInterpolator(memory_transfers)

        bandwidth = interpolator.get_bandwidth_mbps()
        assert abs(bandwidth - 1024.0) < 1.0  # Should be ~1024 MB/s

    def test_get_bandwidth_mbps_uses_largest_transfer(self):
        """Test that bandwidth uses the largest transfer for accuracy."""
        memory_transfers = {
            1024: 0.1,  # Small transfer (less accurate)
            1024 * 1024 * 1024: 1000.0,  # Large transfer (more accurate)
        }
        interpolator = MemoryTransferInterpolator(memory_transfers)

        bandwidth = interpolator.get_bandwidth_mbps()
        # Should be based on 1GB/1s = 1024 MB/s, not the small transfer
        assert abs(bandwidth - 1024.0) < 1.0


class TestRepr:
    """Test cases for __repr__ method."""

    def test_repr_format(self):
        """Test that repr returns expected format."""
        memory_transfers = {
            1024: 0.01,
            1024 * 1024 * 1024: 1000.0,
        }
        interpolator = MemoryTransferInterpolator(memory_transfers)

        repr_str = repr(interpolator)

        assert "MemoryTransferInterpolator" in repr_str
        assert "bytes_range" in repr_str
        assert "duration_range" in repr_str
        assert "bandwidth" in repr_str
        assert "MB/s" in repr_str


class TestCreateInterpolatorFromBenchmarks:
    """Test cases for create_interpolator_from_benchmarks function."""

    def test_create_interpolator_from_benchmarks(self):
        """Test convenience function creates valid interpolator."""
        memory_transfers = {
            1024 * 1024: 1.0,
            1024 * 1024 * 1024: 1000.0,
        }
        interpolator = create_interpolator_from_benchmarks(memory_transfers)

        assert isinstance(interpolator, MemoryTransferInterpolator)
        assert interpolator.min_bytes == 1024 * 1024
        assert interpolator.max_bytes == 1024 * 1024 * 1024

    def test_create_interpolator_empty_raises_error(self):
        """Test that empty dict raises error through convenience function."""
        with pytest.raises(ValueError, match="cannot be empty"):
            create_interpolator_from_benchmarks({})
