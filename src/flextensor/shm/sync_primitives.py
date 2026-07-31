# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronization primitives for multiprocess coordination.

This module provides an abstraction layer for different locking mechanisms
to enable comparison and selection of the most reliable implementation.

Thread Safety:
    SemaphoreLock: Thread-safe. The underlying POSIX semaphores are managed by
    the kernel, which handles synchronization correctly across threads and processes.

    ProcessFileLock: Thread-safe. Uses a class-level registry that maps file paths
    to threading.Lock instances, ensuring all instances for the same path share
    the same thread lock. The underlying fcntl.flock provides inter-process
    synchronization, while the shared threading.Lock provides inter-thread
    synchronization.

    The release() method is idempotent - calling it when the lock is not held
    will silently succeed (no exception raised).
"""

import _thread
import contextlib
import fcntl
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import ClassVar

import posix_ipc


class BaseLock(ABC):
    """Abstract base class for lock implementations.

    Thread-safe base class that provides common context manager implementation.
    Subclasses must implement acquire(), release(), close(), and unlink().

    Args:
        name: Name/identifier for the lock (or path for file-based locks)
        locked: If True, acquire the lock immediately upon creation.
        non_blocking: Behavior when locked=True:
            - True: try to acquire non-blocking, raise exception if fails
            - False: blocking acquire (wait for lock)

    Attributes:
        _init_non_blocking: Stored non_blocking setting for context manager use.
    """

    _init_non_blocking: bool

    @abstractmethod
    def __init__(self, name: str, locked: bool = True, non_blocking: bool = False) -> None: ...

    def __enter__(self):
        """Enter context manager - acquire lock."""
        self.acquire(non_blocking=self._init_non_blocking)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - release lock."""
        self.release()
        return False

    @abstractmethod
    def acquire(self, non_blocking: bool = False, timeout: float | None = None):
        """Acquire the lock.

        Args:
            non_blocking: If True, try non-blocking acquire. If False, block until acquired.
            timeout: Maximum time in seconds to wait (only used if non_blocking=False).
                    None means wait indefinitely.

        Raises:
            BlockingIOError or posix_ipc.BusyError: If non_blocking=True and lock is busy,
                or if timeout expires.
        """

    @abstractmethod
    def release(self):
        """Release the lock.

        This method is idempotent - calling it when the lock is not held
        will silently succeed (no exception raised).
        """

    @abstractmethod
    def close(self):
        """Close the lock resource."""

    @abstractmethod
    def unlink(self):
        """Remove the lock resource from the system."""


class SemaphoreLock(BaseLock):
    """Lock implementation using POSIX semaphores.

    Thread-safe lock that wraps posix_ipc.Semaphore to provide the BaseLock interface.
    The underlying POSIX semaphore is managed by the kernel for thread/process safety.
    """

    def __init__(self, name: str, locked: bool = True, non_blocking: bool = False) -> None:
        """Initialize semaphore lock.

        Args:
            name: Name of the semaphore (must start with /)
            locked: If True, acquire lock immediately after creation.
            non_blocking: Only used when locked=True:
                - True: non-blocking acquire (raises error if busy)
                - False: blocking acquire (waits)
        """
        self.name = name if name.startswith("/") else f"/{name}"
        self._init_non_blocking = non_blocking
        self._held = False

        # Create semaphore if it doesn't exist, or open if it does
        # O_CREAT flag will create or open existing
        self.lock = posix_ipc.Semaphore(self.name, flags=posix_ipc.O_CREAT, initial_value=1)

        # Acquire lock if requested
        if locked:
            self.acquire(non_blocking=non_blocking)

    def acquire(self, non_blocking: bool = False, timeout: float | None = None):
        """Acquire the lock.

        Args:
            non_blocking: If True, try non-blocking acquire. If False, block until acquired.
            timeout: Maximum time in seconds to wait (only used if non_blocking=False).
                    None means wait indefinitely. If timeout expires, raises posix_ipc.BusyError.

        Raises:
            posix_ipc.BusyError: If non_blocking=True and lock is busy, or if timeout expires.
        """
        if non_blocking:
            # Non-blocking acquire
            self.lock.acquire(timeout=0)
        elif timeout is not None:
            # Blocking acquire with timeout
            self.lock.acquire(timeout=timeout)
        else:
            # Blocking acquire, wait indefinitely
            self.lock.acquire()
        self._held = True

    def release(self):
        """Release the lock.

        This method is idempotent - calling it when the lock is not held
        will silently succeed (no-op if not held).
        """
        if not self._held:
            return
        self._held = False
        with contextlib.suppress(ValueError):
            # ValueError can occur if semaphore value would exceed max
            # This happens when releasing an already-released lock
            self.lock.release()

    def signal(self) -> None:
        """Increment the semaphore from another process (e.g. to wake a waiter).

        Use this when a different process needs to \"release\" the semaphore to
        unblock a waiter that is blocked on acquire(). Unlike release(), this
        does not check _held and always increments the underlying semaphore.
        """
        with contextlib.suppress(ValueError):
            self.lock.release()

    def close(self):
        """Close the semaphore."""
        if self._held:
            self.release()
        self.lock.close()

    def unlink(self):
        """Remove the semaphore from the system."""
        with contextlib.suppress(posix_ipc.ExistentialError):
            self.lock.unlink()


class ProcessFileLock(BaseLock):
    """File-based lock using fcntl.flock (Linux/Unix) for inter-process and inter-thread synchronization.

    Thread Safety:
        This lock is thread-safe. It uses a class-level registry that maps file paths to
        threading.Lock instances. Multiple ProcessFileLock instances for the same path
        (even created separately) will share the same thread lock, ensuring proper
        synchronization between threads within the same process.

        The underlying fcntl.flock() provides inter-process synchronization, while the
        shared threading.Lock provides inter-thread synchronization.

    Args:
        path: Path to the lock file.
        locked: If True, acquire lock immediately in constructor.
        non_blocking: Only used when locked=True:
            - True: tries LOCK_EX | LOCK_NB (immediate exception if busy)
            - False: tries LOCK_EX (waits until available)
    """

    # Class-level registry for thread locks - maps normalized paths to threading.Lock instances
    # Note: Use _thread.LockType for type hints because threading.Lock is a factory function,
    # not a type class. This is needed for beartype compatibility with Python 3.12.
    _registry_lock: ClassVar[_thread.LockType] = threading.Lock()
    _thread_locks: ClassVar[dict[str, _thread.LockType]] = {}

    def __init__(self, path: str, locked: bool = True, non_blocking: bool = False) -> None:
        """Initialize file lock.

        Args:
            path: Path/name for the lock file
            locked: If True, acquire lock immediately
            non_blocking: Behavior when locked=True (see class docstring)
        """
        from pathlib import Path

        # Handle path construction
        if Path(path).is_absolute():
            self.path = path
        else:
            self.path = str(Path(tempfile.gettempdir()) / f"{path.lstrip('/')}.lock")

        # Ensure directory exists
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # Get or create thread lock for this path from the class-level registry
        # Uses double-check locking pattern for thread safety
        self._thread_lock = self._get_or_create_thread_lock(self.path)

        self.fd: object | None = None
        self._init_non_blocking = non_blocking

        # Open file (or create) - kept open for the lifetime of the lock
        self.fd = Path(self.path).open("a+")  # noqa: SIM115

        if locked:
            self.acquire(non_blocking=non_blocking)

    @classmethod
    def _get_or_create_thread_lock(cls, path: str) -> _thread.LockType:
        """Get or create a thread lock for the given path.

        Uses double-check locking pattern for thread-safe lazy initialization.

        Args:
            path: The normalized path to the lock file.

        Returns:
            The shared threading.Lock instance for this path.
        """
        # Fast path: lock already exists
        if path in cls._thread_locks:
            return cls._thread_locks[path]

        # Slow path: need to create lock
        with cls._registry_lock:
            # Double-check after acquiring registry lock
            if path not in cls._thread_locks:
                cls._thread_locks[path] = threading.Lock()
            return cls._thread_locks[path]

    def _acquire_thread_lock(self, non_blocking: bool, timeout: float | None) -> None:
        """Acquire the thread lock for inter-thread synchronization.

        Args:
            non_blocking: If True, try non-blocking acquire.
            timeout: Maximum time in seconds to wait (only if non_blocking=False).

        Raises:
            BlockingIOError: If lock cannot be acquired.
        """
        if non_blocking:
            if not self._thread_lock.acquire(blocking=False):
                raise BlockingIOError("Thread lock is busy")
        elif timeout is not None:
            if not self._thread_lock.acquire(blocking=True, timeout=timeout):
                raise BlockingIOError(f"Failed to acquire thread lock within {timeout} seconds")
        else:
            self._thread_lock.acquire()

    def _acquire_file_lock(self, non_blocking: bool, timeout: float | None) -> None:
        """Acquire the file lock for inter-process synchronization.

        Args:
            non_blocking: If True, try non-blocking acquire.
            timeout: Maximum time in seconds to wait (only if non_blocking=False).

        Raises:
            BlockingIOError: If lock cannot be acquired.
        """
        if non_blocking or timeout is None:
            flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if non_blocking else 0)
            fcntl.flock(self.fd.fileno(), flags)
        else:
            self._acquire_file_lock_with_timeout(timeout)

    def _acquire_file_lock_with_timeout(self, timeout: float) -> None:
        """Acquire file lock with polling timeout.

        Args:
            timeout: Maximum time in seconds to wait.

        Raises:
            BlockingIOError: If timeout expires.
        """
        import time

        start_time = time.time()
        while True:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.time() - start_time >= timeout:
                    raise BlockingIOError(f"Failed to acquire lock within {timeout} seconds") from None
                time.sleep(0.01)

    def acquire(self, non_blocking: bool = False, timeout: float | None = None):
        """Acquire the lock.

        This method is thread-safe: it first acquires a shared thread lock for this path,
        then acquires the file lock for inter-process synchronization.

        Args:
            non_blocking:
                - False: blocking LOCK_EX (waits)
                - True: LOCK_EX | LOCK_NB (BlockingIOError if busy)
            timeout: Maximum time in seconds to wait. Only used if non_blocking=False.
                    Note: fcntl.flock doesn't support native timeout, so we use polling.
                    None means wait indefinitely.

        Raises:
            BlockingIOError: If non_blocking=True and lock is busy, or if timeout expires.
            ValueError: If the lock has been closed.
        """
        if self.fd is None:
            msg = "ProcessFileLock is closed"
            raise ValueError(msg)

        self._acquire_thread_lock(non_blocking, timeout)
        try:
            self._acquire_file_lock(non_blocking, timeout)
        except BaseException:
            self._thread_lock.release()
            raise

    def release(self):
        """Release the lock without closing the file.

        This method releases both the file lock (inter-process) and the thread lock
        (inter-thread). It is idempotent - calling it when the lock is not held
        will silently succeed (LOCK_UN on an unlocked file is a no-op, and releasing
        an unheld thread lock is also safe).
        """
        if self.fd is None:
            return

        # Release file lock first
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)

        # Then release thread lock (suppress error if not held)
        # RuntimeError: release unlocked lock - ignore for idempotency
        with contextlib.suppress(RuntimeError):
            self._thread_lock.release()

    def close(self):
        """Close the lock file."""
        self.release()
        if self.fd is not None:
            self.fd.close()
            self.fd = None

    def unlink(self):
        """Remove the lock file from the system."""
        from pathlib import Path

        self.close()
        path_obj = Path(self.path)
        if path_obj.exists():
            with contextlib.suppress(OSError):
                path_obj.unlink()
