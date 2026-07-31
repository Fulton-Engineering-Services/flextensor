# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordination block header for SHM version gating."""

from __future__ import annotations

import struct
from dataclasses import dataclass

# Header layout: 4-byte version string length, up to 64 bytes version string,
# 4-byte protocol version, 8-byte creator PID = 80 bytes total
_VERSION_MAX_LEN = 64
COORD_HEADER_SIZE = 4 + _VERSION_MAX_LEN + 4 + 8  # 80 bytes
_COORD_HEADER_FORMAT = f"!I{_VERSION_MAX_LEN}sIQ"


@dataclass
class CoordBlockHeader:
    """Version and protocol header written at the start of the coordination block.

    Args:
        flextensor_version: FlexTensor version string (e.g., "0.3.3").
        protocol_version: SHM protocol version integer.
        creator_pid: PID whose heartbeat identifies the creator process.
    """

    flextensor_version: str
    protocol_version: int
    creator_pid: int = 0

    def write_to(self, buf: bytearray | memoryview, offset: int = 0) -> None:
        """Write header to buffer at offset.

        Args:
            buf: Target buffer (must have at least COORD_HEADER_SIZE bytes from offset).
            offset: Byte offset into the buffer.
        """
        version_bytes = self.flextensor_version.encode("utf-8")[:_VERSION_MAX_LEN]
        padded = version_bytes.ljust(_VERSION_MAX_LEN, b"\x00")
        struct.pack_into(
            _COORD_HEADER_FORMAT,
            buf,
            offset,
            len(version_bytes),
            padded,
            self.protocol_version,
            self.creator_pid,
        )

    @classmethod
    def read_from(cls, buf: bytes | bytearray | memoryview, offset: int = 0) -> CoordBlockHeader:
        """Read header from buffer at offset.

        Args:
            buf: Source buffer.
            offset: Byte offset into the buffer.

        Returns:
            Parsed CoordBlockHeader instance.
        """
        version_len, version_bytes, protocol, creator_pid = struct.unpack_from(_COORD_HEADER_FORMAT, buf, offset)
        version_str = version_bytes[:version_len].decode("utf-8")
        return cls(flextensor_version=version_str, protocol_version=protocol, creator_pid=creator_pid)

    def validate(self, expected_version: str, expected_protocol: int) -> None:
        """Validate header against expected values.

        Args:
            expected_version: Expected FlexTensor version string.
            expected_protocol: Expected SHM protocol version.

        Raises:
            RuntimeError: If version or protocol doesn't match.
        """
        if self.flextensor_version != expected_version:
            raise RuntimeError(
                f"FlexTensor version mismatch: SHM created by {self.flextensor_version}, "
                f"this process is {expected_version}. All processes must use the same version."
            )
        if self.protocol_version != expected_protocol:
            raise RuntimeError(
                f"SHM protocol version mismatch: SHM uses protocol {self.protocol_version}, "
                f"this process expects {expected_protocol}. All processes must use the same FlexTensor version."
            )
        if self.protocol_version >= 3 and self.creator_pid <= 0:
            raise RuntimeError("SHM protocol v3 requires a positive creator PID.")
