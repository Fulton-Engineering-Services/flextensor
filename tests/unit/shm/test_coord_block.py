# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coordination block header read/write/validate."""

import pytest

import flextensor
from flextensor.shm.coord_block import COORD_HEADER_SIZE, CoordBlockHeader


class TestCoordBlockHeader:
    """Tests for coordination block header read/write."""

    def test_header_roundtrip(self):
        """Write and read header produces same values."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        assert loaded.flextensor_version == flextensor.__version__
        assert loaded.protocol_version == 1

    def test_version_mismatch_raises(self):
        """Mismatched FlexTensor version raises clear error."""
        header = CoordBlockHeader(
            flextensor_version="0.0.0+fake",
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        with pytest.raises(RuntimeError, match="FlexTensor version mismatch"):
            loaded.validate(
                expected_version=flextensor.__version__,
                expected_protocol=1,
            )

    def test_protocol_mismatch_raises(self):
        """Mismatched protocol version raises clear error."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=99,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        with pytest.raises(RuntimeError, match="SHM protocol version mismatch"):
            loaded.validate(
                expected_version=flextensor.__version__,
                expected_protocol=1,
            )

    def test_matching_version_passes(self):
        """Matching versions do not raise."""
        header = CoordBlockHeader(
            flextensor_version=flextensor.__version__,
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE + 256)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        # Should not raise
        loaded.validate(
            expected_version=flextensor.__version__,
            expected_protocol=1,
        )

    def test_header_at_nonzero_offset(self):
        """Header can be written and read at an arbitrary offset."""
        header = CoordBlockHeader(
            flextensor_version="1.2.3",
            protocol_version=2,
        )
        buf = bytearray(512)
        header.write_to(buf, offset=64)
        loaded = CoordBlockHeader.read_from(buf, offset=64)
        assert loaded.flextensor_version == "1.2.3"
        assert loaded.protocol_version == 2

    def test_short_version_string(self):
        """Short version strings are handled correctly."""
        header = CoordBlockHeader(
            flextensor_version="0.1",
            protocol_version=1,
        )
        buf = bytearray(COORD_HEADER_SIZE)
        header.write_to(buf, offset=0)
        loaded = CoordBlockHeader.read_from(buf, offset=0)
        assert loaded.flextensor_version == "0.1"
