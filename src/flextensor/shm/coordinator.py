# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SHM coordinator for creator/follower weight sharing."""

from __future__ import annotations

import logging

from flextensor._version import __version__
from flextensor.shm.coord_block import CoordBlockHeader
from flextensor.shm.flexible_shm import FlexibleSharedMemory
from flextensor.shm.namespace import SHM_PROTOCOL_VERSION, coord_block_name, profile_block_name
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler

logger = logging.getLogger(__name__)

# Coordination block stores only the CoordBlockHeader (~80 B).
# KeepAlive and MultiprocessCondition are separate SHM segments managed by FlexibleSharedMemory.
_COORD_BLOCK_SIZE = 256  # bytes — header + room for future fields


def _require_block(shm: FlexibleSharedMemory | None, label: str) -> FlexibleSharedMemory:
    """Return *shm* after verifying it and its block are not None."""
    if shm is None or shm.block is None:
        msg = f"ShmCoordinator: {label} SHM block is not initialised"
        raise RuntimeError(msg)
    return shm


class ShmCoordinator:
    """Orchestrates creator/follower SHM sharing.

    On init, tries to connect to an existing coordination block:
    - If it exists: becomes follower, validates version gate.
    - If it doesn't exist: becomes creator, writes version header.

    Args:
        namespace: Rank-scoped SHM namespace (e.g., "ft_abc123_tp0_pp0").
        keep_alive_seconds: Timeout for process liveness detection.
    """

    def __init__(self, namespace: str, keep_alive_seconds: int | float = 30) -> None:
        self.namespace = namespace
        self._coord_shm: FlexibleSharedMemory | None = None
        self._profile_shm: FlexibleSharedMemory | None = None
        self.is_creator = False
        self._creator_pid: int | None = None

        crd_name = coord_block_name(namespace)

        try:
            self._coord_shm = FlexibleSharedMemory(
                name=crd_name,
                shm_size=0,
                keep_alive_seconds=keep_alive_seconds,
            )
            # Connected — follower path
            self.is_creator = False
            try:
                self._validate_version()
            except RuntimeError:
                self._coord_shm.close()
                self._coord_shm = None
                raise
            logger.info("ShmCoordinator: follower attached to %s", namespace)
        except FileNotFoundError:
            self._coord_shm = FlexibleSharedMemory(
                name=crd_name,
                shm_size=_COORD_BLOCK_SIZE,
                keep_alive_seconds=keep_alive_seconds,
            )
            # Check actual creation status to handle TOCTOU race
            self.is_creator = self._coord_shm.shm_creator
            if self.is_creator:
                self._write_version_header()
            else:
                self._validate_version()
            logger.info("ShmCoordinator: %s initialized %s", "creator" if self.is_creator else "follower", namespace)

    def _write_version_header(self) -> None:
        """Write version header to coordination block (creator only)."""
        shm = _require_block(self._coord_shm, "coord")
        header = CoordBlockHeader(
            flextensor_version=__version__,
            protocol_version=SHM_PROTOCOL_VERSION,
            creator_pid=shm.pid,
        )
        self._creator_pid = shm.pid
        # block guaranteed non-None by _require_block
        header.write_to(shm.block.buf, offset=0)  # type: ignore[arg-type, union-attr]

    def _validate_version(self) -> None:
        """Read and validate version header (follower only)."""
        shm = _require_block(self._coord_shm, "coord")
        header = CoordBlockHeader.read_from(shm.block.buf, offset=0)  # type: ignore[arg-type, union-attr]
        header.validate(
            expected_version=__version__,
            expected_protocol=SHM_PROTOCOL_VERSION,
        )
        self._creator_pid = header.creator_pid

    def write_profile(self, state: TensorManagerState) -> None:
        """Write profile to SHM (creator only).

        Args:
            state: The TensorManagerState to share with followers.

        Raises:
            RuntimeError: If called on a follower coordinator.
        """
        if not self.is_creator:
            raise RuntimeError("write_profile called on follower coordinator")
        serialized = TensorManagerStateHandler.save_state_to_bytes(state)
        prof_name = profile_block_name(self.namespace)
        self._profile_shm = FlexibleSharedMemory(name=prof_name, shm_size=len(serialized))
        shm = _require_block(self._profile_shm, "profile")
        shm.block.buf[: len(serialized)] = serialized  # type: ignore[index, union-attr]

    def read_profile(self) -> TensorManagerState:
        """Read profile from SHM (follower).

        Returns:
            The deserialized TensorManagerState.

        Raises:
            RuntimeError: If called on a creator coordinator or data is corrupt.
        """
        if self.is_creator:
            raise RuntimeError("read_profile called on creator coordinator")
        prof_name = profile_block_name(self.namespace)
        self._profile_shm = FlexibleSharedMemory(name=prof_name, shm_size=0)
        shm = _require_block(self._profile_shm, "profile")
        # Copy buffer to immutable bytes — prevents concurrent creator writes from
        # corrupting the JSON parse mid-read.
        data = bytes(shm.block.buf)  # type: ignore[arg-type, union-attr]
        try:
            return TensorManagerStateHandler.load_state_from_bytes(data)
        except Exception as exc:
            msg = f"ShmCoordinator: failed to deserialize profile from SHM block '{prof_name}'"
            raise RuntimeError(msg) from exc

    def wait_for_ready(self) -> None:
        """Block until creator calls notify_ready().

        Warning:
            Blocks on a semaphore and does not detect creator death.
            Callers requiring liveness checks should poll is_creator_alive()
            in a separate thread or use a timeout mechanism.
        """
        if self._coord_shm is None:
            msg = "ShmCoordinator: coord SHM is not initialised"
            raise RuntimeError(msg)
        self._coord_shm.wait_for_ready()

    def notify_ready(self) -> None:
        """Signal all waiting followers that SHM is ready."""
        if self._coord_shm is None:
            msg = "ShmCoordinator: coord SHM is not initialised"
            raise RuntimeError(msg)
        self._coord_shm.notify_ready()

    def is_creator_alive(self) -> bool:
        """Check if the creator process is still alive via heartbeat."""
        if self._coord_shm is None or self._coord_shm.keep_alive is None or self._creator_pid is None:
            return False
        return self._coord_shm.keep_alive.is_process_alive(self._creator_pid)

    def close(self) -> None:
        """Release SHM resources."""
        try:
            if self._profile_shm is not None:
                self._profile_shm.close()
                self._profile_shm = None
        finally:
            if self._coord_shm is not None:
                self._coord_shm.close()
                self._coord_shm = None
