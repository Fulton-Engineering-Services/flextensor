# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess condition variable and shared string buffer.

This module provides synchronization primitives for coordinating multiple
processes using shared memory and semaphore-based signaling.
"""

import contextlib
import logging
import os
import struct
from multiprocessing.resource_tracker import register as resource_tracker_register
from multiprocessing.resource_tracker import unregister as resource_tracker_unregister
from multiprocessing.shared_memory import SharedMemory

import posix_ipc

from flextensor.shm.sync_primitives import BaseLock, ProcessFileLock, SemaphoreLock

logger = logging.getLogger(__name__)

HEADER_SIZE = 5


class SharedMultiString:
    """
    Shared multi-string for multiprocess coordination.

    This class provides a shared memory buffer for storing a list of strings.
    The strings are stored in the shared memory buffer in a format that is easy to parse.
    The strings are separated by a pipe character.
    The first 4 bytes of the shared memory buffer are used to store the length of the list.
    The next byte is used to store a boolean value indicating if the list is ready.
    The rest of the shared memory buffer is used to store the strings.

    Note:
        This class is **not thread-safe**. If you need to use it from multiple threads,
        you must protect all accesses (reads and writes) with an external lock.
    """

    def __init__(self, name: str, size: int = 1024 * 128, create: bool = False) -> None:
        """Initialize the shared multi-string buffer.

        Args:
            name: Name for the shared memory segment.
            size: Size in bytes for the buffer.
            create: If True, create the shared memory; otherwise connect to existing.
        """
        # Respect the create parameter
        self.shm: SharedMemory | None = None
        if create:
            try:
                self.shm = SharedMemory(name=name, size=size, create=True)
                # Initialize the header for new shared memory
                self.shm.buf[0:4] = struct.pack("I", HEADER_SIZE)
                self.shm.buf[4:5] = struct.pack("?", False)
            except FileExistsError:
                # Already exists, connect to it
                self.shm = SharedMemory(name=name, create=False)
        else:
            # Connect to existing
            self.shm = SharedMemory(name=name, create=False)
        # Unregister from resource tracker to prevent spurious cleanup warnings in multiprocess scenarios
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.shm._name, "shared_memory")  # noqa: SLF001

    def append(self, string: str) -> None:
        """Append a string to the buffer."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        pos = struct.unpack("I", self.shm.buf[0:4])[0]
        if pos > HEADER_SIZE:
            string = "|" + string
        arr = string.encode("utf-8")
        if pos + len(arr) > self.shm.size:
            msg = "Shared memory is full"
            raise ValueError(msg)
        self.shm.buf[pos : pos + len(arr)] = arr
        pos += len(arr)
        self.shm.buf[0:4] = struct.pack("I", pos)

    def set_ready(self) -> None:
        """Mark the buffer as ready."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        self.shm.buf[4:5] = struct.pack("?", True)

    def is_ready(self) -> bool:
        """Check if the buffer has been marked as ready."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        return struct.unpack("?", self.shm.buf[4:5])[0]

    def get_list(self) -> list[str]:
        """Return all strings in the buffer as a list."""
        if self.shm is None:
            msg = "SharedMultiString is closed"
            raise ValueError(msg)
        # Read the current content length from shared memory header
        # (don't use self.pos as it may be stale in multiprocess scenarios)
        pos = struct.unpack("I", self.shm.buf[0:4])[0]
        # Filter out empty strings that can result from splitting
        s = bytes(self.shm.buf[HEADER_SIZE:pos]).decode("utf-8")
        return s.split("|") if s else []

    def close(self) -> None:
        """Close the shared memory without unlinking."""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink(self) -> None:
        """Unlink and close the shared memory, removing it from the system."""
        # Register back to resource tracker before unlink, then unlink before close
        # Use _name (with / prefix) to match what resource_tracker expects
        if self.shm is not None:
            resource_tracker_register(self.shm._name, "shared_memory")  # noqa: SLF001
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()
        self.close()


class MultiprocessCondition:
    """
    Multiprocess condition for multiprocess coordination.

    This class provides a mechanism for processes to wait for a condition to be met.
    The condition is set by one process and other processes can wait for it to be met.
    """

    def __init__(self, name: str, is_creator: bool = False, lock_class: type[BaseLock] = ProcessFileLock) -> None:
        """Initialize the multiprocess condition.

        Args:
            name: Name for shared resources.
            is_creator: If True, create shared resources; otherwise connect.
            lock_class: Lock class to use (default: ProcessFileLock).
        """
        self.name = name
        self.notification_list: SharedMultiString | None = SharedMultiString(name=name, create=is_creator)
        self.lock_class = lock_class
        self.lock: BaseLock | None = self.lock_class(name, locked=False)

    def wait_for_ready(self) -> None:
        """Block until notify_ready() is called."""
        lock = self.lock
        notification_list = self.notification_list
        if lock is None or notification_list is None:
            raise RuntimeError("MultiprocessCondition is closed")
        pid = os.getpid()
        notification_lock = None
        with lock:
            if notification_list.is_ready():
                return
            lock_name = f"{self.name}_{pid}"
            # Always use SemaphoreLock for notifications, create and lock it
            notification_lock = SemaphoreLock(lock_name, locked=True, non_blocking=False)
            notification_list.append(f"{pid}")

        try:
            while not notification_list.is_ready():
                notification_lock.acquire(non_blocking=False)
        finally:
            if notification_lock is not None:
                with contextlib.suppress(FileNotFoundError, AttributeError, OSError):
                    notification_lock.unlink()
                with contextlib.suppress(FileNotFoundError, AttributeError, OSError):
                    notification_lock.close()

    def notify_ready(self) -> None:
        """Signal all waiting processes that the condition is ready."""
        lock = self.lock
        notification_list = self.notification_list
        if lock is None or notification_list is None:
            raise RuntimeError("MultiprocessCondition is closed")
        with lock:
            notification_list.set_ready()
            pids = notification_list.get_list()
            for waiter_pid in pids:
                if not waiter_pid:
                    continue
                lock_name = f"{self.name}_{waiter_pid}"
                try:
                    # Always use SemaphoreLock for notifications, connect without locking
                    notification_lock = SemaphoreLock(lock_name, locked=False)
                    notification_lock.signal()
                    notification_lock.close()
                except posix_ipc.ExistentialError:
                    # Lock doesn't exist - process may not be waiting (expected case)
                    pass
                except OSError as e:
                    # System error (PermissionError, ENOMEM, EMFILE) - log as warning
                    # If notification delivery fails, the waiter hangs forever
                    logger.warning(
                        "Failed to deliver notification to process %s (lock_name=%s): %s",
                        waiter_pid,
                        lock_name,
                        e,
                    )

    def close(self) -> None:
        """Close the notification list without unlinking."""
        if self.notification_list is not None:
            self.notification_list.close()

    def unlink(self) -> None:
        """Unlink and close all shared resources."""
        if self.notification_list is not None:
            self.notification_list.unlink()
            self.notification_list = None
        if self.lock is not None:
            self.lock.unlink()

    def close_lock(self) -> None:
        """Close only the lock resource."""
        if self.lock is not None:
            self.lock.close()
            self.lock = None
