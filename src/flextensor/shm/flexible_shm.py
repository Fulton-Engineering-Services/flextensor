# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flexible shared memory management with keep-alive tracking.

This module provides classes for managing shared memory segments across multiple
processes with automatic liveness tracking and coordinated cleanup.
"""

import contextlib
import math
import os
import threading
import time
from collections.abc import Iterator
from multiprocessing.resource_tracker import register as resource_tracker_register
from multiprocessing.resource_tracker import unregister as resource_tracker_unregister
from multiprocessing.shared_memory import SharedMemory
from typing import ClassVar

import posix_ipc
from shared_memory_dict import SharedMemoryDict

from flextensor.shm.multiprocess_condition import MultiprocessCondition
from flextensor.shm.sync_primitives import BaseLock, ProcessFileLock

LOCK_ACQUIRE_TIMEOUT = 5.0  # Must be float for beartype type checking


class KeepAliveDict:
    """Shared memory dictionary mapping process IDs to timestamps for liveness tracking.

    Combines SharedMemoryDict for process ID to slot mapping with a SharedMemory
    buffer for storing timestamps. Allows efficient liveness checks across processes.

    Note:
        This dictionary is **not thread-safe** or multi-process safe on its own.
        All modifications (e.g., add, remove, update) must be externally synchronized
        using appropriate locking to avoid races and corruption.
        The caller is responsible for using a multiprocessing lock or similar mechanism
        around all modifying operations.
    """

    def __init__(self, name: str, size: int, is_creator: bool, meta_name: str | None = None) -> None:
        """Initialize the keep-alive dictionary.

        Args:
            name: Name for the SharedMemoryDict.
            size: Size in bytes for both the dict and timestamp buffer.
            is_creator: If True, create the shared memory; otherwise connect to existing.
            meta_name: Optional separate name for the timestamp SharedMemory buffer.
        """
        self.name = name
        self.size = size

        # Allow separate name for metadata to control exact names (avoid length limits)
        if meta_name is None:
            meta_name = name + "_m"

        # SharedMemoryDict handles both creation and connection automatically
        # Note: SharedMemoryDict will add "sm_" prefix internally
        self.dict = SharedMemoryDict(name=name, size=size)
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.dict.shm._name, "shared_memory")  # noqa: SLF001

        # For SharedMemory, respect is_creator but handle already-exists gracefully
        if is_creator:
            try:
                self.shm = SharedMemory(name=meta_name, create=True, size=size)
            except FileExistsError:
                # Creator expected to create, but it already exists - connect to it
                self.shm = SharedMemory(name=meta_name, create=False)
        else:
            # Not creator - just connect
            self.shm = SharedMemory(name=meta_name, create=False)
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.shm._name, "shared_memory")  # noqa: SLF001

        self.shm_view = self.shm.buf.cast("d")
        self._closed = False

    def prepare_slot(self, process_id: int) -> int:
        """Allocate a slot for a process. It is not thread-safe or multi-process safe on its own."""
        if process_id in self.dict:
            return self.dict[process_id]

        # Always compute the next available slot dynamically based on current dict state
        # This ensures correct behavior across multiple process instances
        used_slots = set()
        max_slot = -1
        for _, existing_slot in self.dict.items():
            used_slots.add(existing_slot)
            if existing_slot > max_slot:
                max_slot = existing_slot

        # Find a free slot (either reuse a gap or allocate new)
        slot = None
        for candidate in range(max_slot + 1):
            if candidate not in used_slots:
                slot = candidate
                break
        if slot is None:
            slot = max_slot + 1

        capacity = len(self.shm_view)
        if slot >= capacity:
            msg = (
                f"KeepAliveDict timestamp slots exhausted: process_id={process_id}, slot={slot}, "
                f"timestamp_capacity={capacity}, keep_alive_dict_size={self.size}"
            )
            raise RuntimeError(msg)

        try:
            self.dict[process_id] = slot
        except ValueError as e:
            msg = (
                f"KeepAliveDict mapping storage exhausted: process_id={process_id}, "
                f"slot={slot}, keep_alive_dict_size={self.size}"
            )
            raise RuntimeError(msg) from e
        self.shm_view[slot] = time.time()
        return slot

    def set_by_slot(self, slot: int, timestamp: int | float) -> None:
        """Set the timestamp for a given slot index."""
        self.shm_view[slot] = float(timestamp)

    def pop(self, process_id: int) -> float:
        """Remove a process from the dict. Must be called while holding the keep-alive lock."""
        slot = self.dict.pop(process_id)
        val = self.shm_view[slot]
        # Clear the slot timestamp (slot will be reused by prepare_slot if needed)
        self.shm_view[slot] = 0.0
        return val

    def items(self) -> Iterator[tuple[int, float]]:
        """Yield (process_id, timestamp) pairs for all registered processes."""
        for pid, slot in self.dict.items():
            yield pid, self.shm_view[slot]

    def __contains__(self, process_id: int) -> bool:
        """Check if a process_id is in the dictionary."""
        return process_id in self.dict

    def close(self) -> None:
        """Close shared memory resources without unlinking."""
        if self._closed:
            return
        self._closed = True
        # Release the memoryview first to allow proper cleanup
        if self.shm_view is not None:
            with contextlib.suppress(BufferError):
                self.shm_view.release()
            self.shm_view = None

        # Close SharedMemoryDict's shared memory
        if self.dict is not None:
            self.dict.shm.close()

        # Close our shared memory
        if self.shm is not None:
            self.shm.close()

    def unlink(self) -> None:
        """Unlink and close shared memory resources, removing them from the system."""
        # Register back to resource tracker before unlink, then unlink before close
        # Unlink our SharedMemory (for timestamps)
        # Use _name (with / prefix) to match what resource_tracker expects
        if self.shm is not None:
            resource_tracker_register(self.shm._name, "shared_memory")  # noqa: SLF001
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()
        # Register and unlink SharedMemoryDict's shared memory
        # Use _name (with / prefix) to match what resource_tracker expects
        if self.dict is not None:
            resource_tracker_register(self.dict.shm._name, "shared_memory")  # noqa: SLF001
            with contextlib.suppress(AttributeError, FileNotFoundError):
                self.dict.shm.unlink()
        # Now close after unlink
        self.close()
        self.shm = None
        self.dict = None


class KeepAlive:
    """Background thread that periodically updates process timestamp to signal liveness.

    Registers the current process in a KeepAliveDict and spawns a daemon thread
    that periodically updates the timestamp. Used to detect dead processes.
    """

    _local_registrations: ClassVar[dict[tuple[str, int], int]] = {}
    _local_registrations_lock: ClassVar = threading.Lock()

    def __init__(
        self,
        name: str,
        process_id: int,
        keep_alive_dict_size: int = 128 * 1024,
        keep_alive_seconds: int | float = 30,
        is_creator: bool = False,
        lock_class: type[BaseLock] = ProcessFileLock,
        dict_name: str | None = None,
        dict_meta_name: str | None = None,
        lock_name: str | None = None,
        unlink_on_init_failure: bool = False,
    ) -> None:
        """Initialize keep-alive tracking for the current process.

        Args:
            name: Base name for shared resources.
            process_id: Identifier for this process (typically PID).
            keep_alive_dict_size: Size in bytes for the keep-alive dictionary.
            keep_alive_seconds: Timeout after which a process is considered dead.
            is_creator: If True, create shared resources; otherwise connect.
            lock_class: Lock class to use (default: ProcessFileLock).
            dict_name: Optional custom name for the dictionary.
            dict_meta_name: Optional custom name for the metadata buffer.
            lock_name: Optional custom name for the lock.
            unlink_on_init_failure: Remove auxiliary resources if initialization fails.
        """
        if keep_alive_seconds <= 0 or not math.isfinite(keep_alive_seconds):
            raise ValueError("keep_alive_seconds must be finite and positive")

        self.name = name

        # Allow caller to specify exact names to avoid excessive nesting
        dict_name = dict_name if dict_name is not None else f"{name}_d"
        dict_meta_name = dict_meta_name if dict_meta_name is not None else f"{name}_m"
        lock_name = lock_name if lock_name is not None else f"{name}_s"

        self.process_id = process_id
        self.keep_alive_seconds = keep_alive_seconds
        self._registration_key = (dict_name, process_id)
        self._registered: bool = False
        self.keep_alive_dict = KeepAliveDict(
            name=dict_name, size=keep_alive_dict_size, is_creator=is_creator, meta_name=dict_meta_name
        )
        self.create_lock = is_creator
        self.stop_event = threading.Event()  # Event to signal thread to stop
        self.keep_alive_thread: threading.Thread | None = None
        self.lock_class = lock_class

        try:
            # Create or connect to lock, then acquire it for initialization
            self.lock = self.lock_class(lock_name, locked=False)
            self.lock.acquire(non_blocking=False, timeout=LOCK_ACQUIRE_TIMEOUT)
        except BaseException as e:
            self._cleanup_init_failure(unlink=unlink_on_init_failure)
            if isinstance(e, (posix_ipc.BusyError, RuntimeError, BlockingIOError)):
                msg = f"KeepAlive lock '{lock_name}' initialization failed (is_creator={is_creator})"
                raise RuntimeError(msg) from e
            raise

        registration_created = self.process_id not in self.keep_alive_dict
        try:
            self.slot = self.keep_alive_dict.prepare_slot(self.process_id)
            self._retain_registration()
            self._start_keep_alive_thread()
            if not registration_created:
                self.keep_alive_dict.set_by_slot(self.slot, time.time())
        except BaseException:
            self.stop_event.set()
            keep_alive_thread = getattr(self, "keep_alive_thread", None)
            if keep_alive_thread is not None and keep_alive_thread.is_alive():
                keep_alive_thread.join(timeout=2)
            self.keep_alive_thread = None
            self._release_registration()
            with contextlib.suppress(Exception):
                if registration_created and self.process_id in self.keep_alive_dict:
                    self.keep_alive_dict.pop(self.process_id)
            self._cleanup_init_failure(unlink=unlink_on_init_failure)
            raise
        finally:
            self.lock.release()

    def _cleanup_init_failure(self, *, unlink: bool) -> None:
        with contextlib.suppress(Exception):
            if unlink:
                self.keep_alive_dict.unlink()
            else:
                self.keep_alive_dict.close()

        lock = getattr(self, "lock", None)
        if lock is not None:
            if unlink:
                with contextlib.suppress(Exception):
                    lock.unlink()
            with contextlib.suppress(Exception):
                lock.close()

    def _retain_registration(self) -> None:
        with self._local_registrations_lock:
            count = self._local_registrations.get(self._registration_key, 0)
            self._local_registrations[self._registration_key] = count + 1
            self._registered = True

    def _release_registration(self) -> bool:
        if not self._registered:
            return False
        with self._local_registrations_lock:
            count = self._local_registrations.get(self._registration_key, 1)
            if count > 1:
                self._local_registrations[self._registration_key] = count - 1
                last_registration = False
            else:
                self._local_registrations.pop(self._registration_key, None)
                last_registration = True
            self._registered = False
            return last_registration

    def stop(self) -> None:
        """Stop the keep-alive thread and unregister this process."""
        self.stop_event.set()  # Signal thread to stop
        if self.keep_alive_thread is not None:
            self.keep_alive_thread.join(timeout=2)  # Wait max 2 seconds for thread to finish
            self.keep_alive_thread = None
        with self.lock:
            if self._release_registration() and self.process_id in self.keep_alive_dict:
                self.keep_alive_dict.pop(self.process_id)

    def close(self, any_process_alive: bool = True) -> None:
        """Close keep-alive resources.

        Args:
            any_process_alive: If False, unlink shared resources; otherwise just close.
        """
        if self._registered or self.keep_alive_thread is not None:
            self.stop()
        if not any_process_alive:
            # unlink() does: register → unlink → close
            self.keep_alive_dict.unlink()
            self.lock.unlink()
        else:
            self.keep_alive_dict.close()
        self.lock.close()

        self.keep_alive_dict = None
        self.lock = None

    def any_process_alive(self) -> bool:
        """Check if any registered process is still alive and remove stale entries."""
        current_time = time.time()
        with self.lock:
            alive_items = list(self.keep_alive_dict.items())
            alive = False
            for pid, timestamp in alive_items:
                if timestamp < current_time - self.keep_alive_seconds:
                    self.keep_alive_dict.pop(pid)
                else:
                    alive = True
            return alive

    def is_process_alive(self, process_id: int) -> bool:
        """Check one registered process heartbeat, removing it if stale."""
        current_time = time.time()
        with self.lock:
            for pid, timestamp in self.keep_alive_dict.items():
                if pid != process_id:
                    continue
                if timestamp < current_time - self.keep_alive_seconds:
                    self.keep_alive_dict.pop(pid)
                    return False
                return True
        return False

    def _start_keep_alive_thread(self) -> None:
        self.keep_alive_thread = threading.Thread(target=self._keep_alive_thread)
        self.keep_alive_thread.daemon = True
        self.keep_alive_thread.start()

    def _keep_alive_thread(self) -> None:
        while not self.stop_event.is_set():
            # Use event.wait() instead of time.sleep() so we can be interrupted quickly
            if self.stop_event.wait(timeout=self.keep_alive_seconds / 2):
                break  # Event was set, stop immediately
            self.keep_alive_dict.set_by_slot(self.slot, time.time())


class FlexibleSharedMemory:
    """Manages a shared memory segment with keep-alive tracking and notification.

    Provides coordinated access to shared memory across multiple processes with
    automatic liveness detection and cleanup of resources when the last process exits.
    """

    def __init__(
        self,
        name: str,
        shm_size: int,
        keep_alive_dict_size: int = 128 * 1024,
        keep_alive_seconds: int | float = 30,
        pinned_memory: bool = False,
        lock_class: type[BaseLock] = ProcessFileLock,
        lock_acquire_timeout: float | None = LOCK_ACQUIRE_TIMEOUT,
    ) -> None:
        """Initialize flexible shared memory.

        Args:
            name: Name for the shared memory segment.
            shm_size: Size in bytes. If > 0, creates or connects; if 0, connects only.
            keep_alive_dict_size: Size for the keep-alive dictionary.
            keep_alive_seconds: Timeout for process liveness detection.
            pinned_memory: If True, register memory with CUDA for faster transfers.
            lock_class: Lock class to use (default: ProcessFileLock).
            lock_acquire_timeout: Maximum seconds to wait for the main lock, or None to wait indefinitely.
        """
        self.name = name
        self.shm_size = shm_size
        self.pid = os.getpid()
        self.pinned_memory = pinned_memory
        self.shm: SharedMemory | None = None
        self.c_buf = None
        self.shm_ptr = None
        self.main_lock: BaseLock | None = None
        self.keep_alive: KeepAlive | None = None
        self.shm_creator = False
        self.lock_class = lock_class

        if shm_size < 0:
            msg = "shm_size cannot be negative"
            raise ValueError(msg)

        lock_name = name
        # Create or connect to lock and acquire it immediately with timeout
        self.main_lock = self.lock_class(lock_name, locked=False)
        try:
            self.main_lock.acquire(non_blocking=False, timeout=lock_acquire_timeout)
        except BaseException as e:
            with contextlib.suppress(Exception):
                self.main_lock.close()
            self.main_lock = None
            if isinstance(e, (posix_ipc.BusyError, BlockingIOError)):
                msg = f"Failed to acquire main lock for shared memory '{name}' within {lock_acquire_timeout} seconds"
                raise BlockingIOError(msg) from e
            raise

        try:
            self.shm_creator = self._create_memory_block()

            # Use compact naming to avoid exceeding POSIX name length limits (31 chars on macOS)
            # Instead of nested suffixes like name_ka_dict_d, use flat structure: name_kd, name_km, etc.
            keep_alive_dict_name = name + "_kd"  # Keep-alive dict data (will get sm_ prefix from SharedMemoryDict)
            keep_alive_meta_name = name + "_km"  # Keep-alive dict metadata
            keep_alive_lock_name = name + "_ks"  # Keep-alive lock
            condition_name = name + "_c"  # Notification condition

            # All processes with shm_size > 0 should try to create resources (with fallback to connect).
            # This ensures late-joining processes can still find/create the resources they need.
            # The KeepAliveDict and MultiprocessCondition handle FileExistsError gracefully.
            resources_creator = shm_size > 0
            self.keep_alive = KeepAlive(
                name=name + "_k",  # Keep-alive base name
                process_id=self.pid,
                keep_alive_dict_size=keep_alive_dict_size,
                keep_alive_seconds=keep_alive_seconds,
                is_creator=resources_creator,
                lock_class=self.lock_class,
                dict_name=keep_alive_dict_name,
                dict_meta_name=keep_alive_meta_name,
                lock_name=keep_alive_lock_name,
                unlink_on_init_failure=self.shm_creator,
            )
            self.notification_condition = MultiprocessCondition(
                condition_name,
                is_creator=resources_creator,
                lock_class=self.lock_class,
            )

        except Exception:
            # Best-effort cleanup on initialization failure
            self._cleanup_on_init_failure()
            raise
        finally:
            if self.main_lock is not None:
                self.main_lock.release()

    def _cleanup_on_init_failure(self) -> None:
        """Best-effort cleanup when __init__ fails after partial resource creation.

        Only unlinks resources if we created them (shm_creator=True), otherwise just closes.
        This prevents destroying shared memory segments that other processes are using.
        """
        # Use getattr for safe access since exception may occur before attributes are assigned
        notification_condition = getattr(self, "notification_condition", None)
        keep_alive = getattr(self, "keep_alive", None)
        shm = getattr(self, "shm", None)
        shm_creator = getattr(self, "shm_creator", False)

        # Clean up notification_condition if it was created
        if notification_condition is not None:
            with contextlib.suppress(Exception):
                notification_condition.close_lock()
            self.notification_condition = None

        # Clean up keep_alive if it was created
        if keep_alive is not None:
            with contextlib.suppress(Exception):
                keep_alive.stop()
            with contextlib.suppress(Exception):
                # Only unlink if no other process is alive (which shouldn't happen during init failure,
                # but be safe)
                keep_alive.close(any_process_alive=not shm_creator)
            self.keep_alive = None

        # Clean up shared memory - only unlink if WE created it
        if shm is not None:
            with contextlib.suppress(Exception):
                if shm_creator:
                    # We created this segment and failed before any other process could register,
                    # so it's safe to unlink
                    resource_tracker_register(shm._name, "shared_memory")  # noqa: SLF001
                    shm.unlink()
                shm.close()
            self.shm = None

    def _unregister_pinned_memory(self) -> None:
        """Unregister pinned memory if it was used."""
        if self.shm_ptr is not None and self.pinned_memory:
            import cupy
            import cupy.cuda.runtime

            cupy.cuda.runtime.hostUnregister(self.shm_ptr.value)
            self.shm_ptr = None
            if self.c_buf is not None:
                del self.c_buf
                self.c_buf = None

    def _try_acquire_lock(self, max_attempts: int = 50) -> bool:
        """Try to acquire the main lock with retries.

        Returns:
            bool: True if lock was acquired, False otherwise
        """
        lock = self.main_lock
        if lock is None:
            return False

        for attempt in range(max_attempts):
            try:
                lock.acquire(non_blocking=True)  # Non-blocking
            except (posix_ipc.BusyError, BlockingIOError):
                # Lock is busy, wait a bit and retry
                if attempt < max_attempts - 1:
                    time.sleep(0.1)
                continue
            except (OSError, posix_ipc.Error):
                # Other errors - unable to acquire
                break
            else:
                return True
        return False

    def _cleanup_without_coordination(self) -> None:
        """Clean up resources when lock cannot be acquired."""
        if self.keep_alive is not None:
            self.keep_alive.stop()
            # Close keep_alive resources - use any_process_alive=True because we couldn't
            # acquire the lock, so we must assume other processes may still be alive
            with contextlib.suppress(OSError, AttributeError):
                self.keep_alive.close(any_process_alive=True)
        self.notification_condition.close_lock()
        with contextlib.suppress(OSError, AttributeError):
            if self.shm is not None:
                self.shm.close()
        with contextlib.suppress(OSError, AttributeError):
            if self.main_lock is not None:
                self.main_lock.close()
        self.main_lock = None
        self.shm = None

    def _cleanup_with_coordination(self) -> None:
        """Clean up resources after acquiring lock."""
        if self.keep_alive is None or self.shm is None or self.main_lock is None:
            raise RuntimeError("FlexibleSharedMemory is closed")
        # Stop our keep_alive so any_process_alive doesn't count us
        self.keep_alive.stop()

        any_process_alive = self.keep_alive.any_process_alive()
        self.keep_alive.close(any_process_alive=any_process_alive)

        if not any_process_alive:
            # Register back to resource tracker before unlink, then unlink before close
            # Use _name (with / prefix) to match what resource_tracker expects
            resource_tracker_register(self.shm._name, "shared_memory")  # noqa: SLF001
            with contextlib.suppress(FileNotFoundError):
                self.shm.unlink()
            self.shm.close()
            self.main_lock.unlink()
            self.notification_condition.unlink()
        else:
            # Not last process - just close (no unlink)
            self.shm.close()

        self.notification_condition.close_lock()

        if any_process_alive:
            self.main_lock.release()

    def close(self) -> None:
        """Close and clean up all resources."""
        # Don't stop keep_alive yet - keep ourselves marked as alive until we're done with shared resources
        self._unregister_pinned_memory()
        self.notification_condition.close()

        # Try to acquire lock with retries to avoid deadlock
        acquired = self._try_acquire_lock()

        if not acquired:
            # Could not acquire lock, just close without cleanup coordination
            self._cleanup_without_coordination()
            return

        # Perform coordinated cleanup
        try:
            self._cleanup_with_coordination()
        finally:
            if self.main_lock is not None:
                with contextlib.suppress(OSError, AttributeError):
                    self.main_lock.close()
            self.main_lock = None
            self.shm = None

    def wait_for_ready(self) -> None:
        """Block until notify_ready() is called by another process."""
        self.notification_condition.wait_for_ready()

    def notify_ready(self) -> None:
        """Signal all waiting processes that the shared memory is ready."""
        self.notification_condition.notify_ready()

    def _create_memory_block(self) -> bool:
        """Create or connect to the shared memory block.

        Returns:
            True if this call created the memory, False if connected to existing.
        """
        # If shm_size is 0, we're connecting to existing memory, not creating
        # If shm_size > 0, we're creating (or trying to)
        is_created = False
        if self.shm_size > 0:
            # Try to create first
            try:
                self.shm = SharedMemory(name=self.name, size=self.shm_size, create=True)
                # We created it - update shm_creator to True
                is_created = True
            except FileExistsError:
                # Already exists, connect to it
                self.shm = SharedMemory(name=self.name, create=False)
                # We didn't create it
                is_created = False
        else:
            # Connecting to existing memory (shm_size == 0)
            self.shm = SharedMemory(name=self.name, create=False)
            is_created = False
        # Unregister from resource tracker to prevent spurious cleanup warnings in multiprocess scenarios
        # Use _name (with / prefix) to match what resource_tracker registered
        resource_tracker_unregister(self.shm._name, "shared_memory")  # noqa: SLF001
        self.shm_size = self.shm.size

        if self.pinned_memory:
            import ctypes

            import cupy
            import cupy.cuda.runtime

            self.c_buf = (ctypes.c_byte * self.shm_size).from_buffer(self.shm.buf)
            host_ptr = ctypes.addressof(self.c_buf)
            self.shm_ptr = ctypes.c_void_p(host_ptr)
            cupy.cuda.runtime.hostRegister(self.shm_ptr.value, self.shm_size, 0)
        return is_created

    @property
    def block(self) -> SharedMemory | None:
        """The underlying SharedMemory object."""
        return self.shm
