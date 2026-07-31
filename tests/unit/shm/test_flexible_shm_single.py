# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process unit tests for FlexibleSharedMemory class."""

import contextlib
import os
import time
from unittest import mock

import pytest

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock
from flextensor.shm.flexible_shm import KeepAlive


class TestFlexibleSharedMemorySingle:
    """Single-process tests for FlexibleSharedMemory class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "fsm_test"  # Short name to avoid POSIX name limits

        # Shared memory segments created by FlexibleSharedMemory
        # Must match the compact naming in FlexibleSharedMemory.__init__
        self.shm_names = [
            self.test_name,  # Main shared memory
            "sm_" + self.test_name + "_kd",  # Keep alive dict (SharedMemoryDict adds sm_ prefix)
            self.test_name + "_km",  # Keep alive timestamps array
            self.test_name + "_c",  # Condition notification list
        ]

        # Semaphores / lock names created by FlexibleSharedMemory
        self.semaphore_names = [
            self.test_name,  # Main lock
            self.test_name + "_ks",  # Keep alive lock
            self.test_name + "_c",  # Condition lock
        ]

        self.test_size = 1024 * 4  # 4KB for tests
        self._cleanup_resources()

    def teardown_method(self):
        """Cleanup after each test method."""
        self._cleanup_resources()

    def _cleanup_resources(self):
        """Helper to cleanup test resources."""
        try:
            from multiprocessing.shared_memory import SharedMemory

            import posix_ipc

            # Clean up semaphores (POSIX)
            for sem_name in self.semaphore_names:
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.close()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.unlink()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass

            # Clean up file locks
            import tempfile
            from pathlib import Path

            for lock_name in self.semaphore_names:
                lock_path = Path(tempfile.gettempdir()) / f"{lock_name.lstrip('/')}.lock"
                if lock_path.exists():
                    with contextlib.suppress(FileNotFoundError, OSError):
                        lock_path.unlink()

            # Clean up shared memory segments
            for shm_name in self.shm_names:
                try:
                    shm = SharedMemory(name=shm_name, create=False)
                    shm.close()
                    shm.unlink()
                except (FileNotFoundError, FileExistsError):
                    pass
                # Try to unlink even if we couldn't open it
                with contextlib.suppress(FileNotFoundError, FileExistsError):
                    SharedMemory(name=shm_name, create=False).unlink()
        except ImportError:
            pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_flexible_shared_memory_creation(self, lock_class):
        """Test FlexibleSharedMemory creation."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,  # Avoid cupy dependency in basic test
            lock_class=lock_class,
        )

        assert fsm.name == self.test_name
        assert fsm.shm_size >= self.test_size  # Actual size may be rounded up
        assert fsm.pid == os.getpid()
        assert fsm.pinned_memory is False
        assert fsm.shm_creator is True  # Should be creator in single process
        assert fsm.shm is not None
        assert fsm.keep_alive is not None
        assert fsm.notification_condition is not None
        assert fsm.main_lock is not None

        fsm.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_memory_block_access(self, lock_class):
        """Test accessing the memory block."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should be able to access block property
        block = fsm.block
        assert block is not None
        assert block is fsm.shm
        # Note: block.size might not match self.test_size if connecting to existing memory
        # When shm_creator is False, the size parameter is ignored and existing size is used
        assert block.size >= self.test_size  # At least as large as requested

        # Should be able to write to memory
        test_data = b"Hello, FlexibleSharedMemory!"
        block.buf[: len(test_data)] = test_data

        # Should be able to read back
        read_data = bytes(block.buf[: len(test_data)])
        assert read_data == test_data

        fsm.close()

    def test_keep_alive_integration(self):
        """Test integration with KeepAlive functionality."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,
            pinned_memory=False,
        )

        # Keep alive should be running
        assert fsm.keep_alive is not None
        assert fsm.keep_alive.any_process_alive() is True

        # Process should be in keep alive dictionary (process_id is an integer)
        with fsm.keep_alive.lock:
            assert fsm.pid in fsm.keep_alive.keep_alive_dict

        fsm.close()

    def test_try_acquire_lock_keeps_reference_across_retries(self):
        """A concurrent close cannot replace the lock between retry attempts."""
        fsm = FlexibleSharedMemory.__new__(FlexibleSharedMemory)

        class ClosingRaceLock:
            attempts = 0

            def acquire(self, *, non_blocking):
                assert non_blocking
                self.attempts += 1
                fsm.main_lock = None
                if self.attempts == 1:
                    raise BlockingIOError

        fsm.main_lock = ClosingRaceLock()

        assert fsm._try_acquire_lock(max_attempts=2)

    def test_notification_condition_integration(self):
        """Test integration with MultiprocessCondition."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should initially not be ready
        assert not fsm.notification_condition.notification_list.is_ready()

        # Should be able to notify ready
        fsm.notify_ready()
        assert fsm.notification_condition.notification_list.is_ready()

        # wait_for_ready should return immediately now
        start_time = time.time()
        fsm.wait_for_ready()
        wait_time = time.time() - start_time
        assert wait_time < 0.1  # Should be very fast

        fsm.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_protection(self, lock_class):
        """Test main semaphore protection."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should be able to acquire and release main semaphore
        fsm.main_lock.acquire()

        # Do some work while holding semaphore
        test_data = b"Protected operation"
        fsm.block.buf[: len(test_data)] = test_data

        fsm.main_lock.release()

        # Verify data was written
        read_data = bytes(fsm.block.buf[: len(test_data)])
        assert read_data == test_data

        fsm.close()

    def test_pinned_memory_unavailable(self):
        """Test behavior when cupy is not available for pinned memory."""
        # Test that we can create FSM even if pinned memory is requested but cupy fails
        # This will attempt to use pinned memory, but if cupy is not available it should handle gracefully
        # For this test, we just verify basic creation works with pinned_memory=False
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,  # Use False for this test to avoid cupy dependency
        )

        # Should create successfully without pinned memory
        assert fsm.shm is not None
        assert fsm.shm_ptr is None  # Should be None without pinned memory
        assert fsm.c_buf is None

        fsm.close()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_functionality(self, lock_class):
        """Test close functionality."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Should be functional before close
        assert fsm.keep_alive.any_process_alive()

        fsm.close()

        # Resources should be cleaned up
        assert fsm.main_lock is None
        assert fsm.shm is None

    def test_write_read_and_close(self):
        """Test that data can be written, read back, and the instance closed cleanly."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Write some data
        test_data = b"Shared data test"
        fsm.block.buf[: len(test_data)] = test_data

        # Verify we can read it back
        read_data = bytes(fsm.block.buf[: len(test_data)])
        assert read_data == test_data

        # Close the instance (single-process close unlinks the segment)
        fsm.close()

    def test_different_sizes(self):
        """Test with different memory sizes."""
        sizes = [1024, 4096]  # Reduced sizes for faster testing

        for size in sizes:
            test_name = f"{self.test_name}_{size}"
            # Clean up any leftover resources first
            self._cleanup_test_instance(test_name)
            time.sleep(0.1)  # Small delay to ensure cleanup completes

            fsm = FlexibleSharedMemory(
                name=test_name,
                shm_size=size,
                keep_alive_seconds=2,  # Short interval for faster test cleanup
                pinned_memory=False,
            )

            # Shared memory sizes are typically rounded up to page boundaries by the OS,
            # so we should verify it's at least as large as requested
            assert fsm.shm.size >= size
            assert fsm.shm_creator is True
            assert fsm.shm_size >= size  # Actual size may be rounded up

            # Should be able to use the memory
            test_pattern = b"x" * min(100, size)  # Don't fill entire memory for performance
            fsm.block.buf[: len(test_pattern)] = test_pattern
            read_pattern = bytes(fsm.block.buf[: len(test_pattern)])
            assert read_pattern == test_pattern

            fsm.close()
            time.sleep(0.1)  # Small delay before cleanup
            # Clean up after each iteration
            self._cleanup_test_instance(test_name)

    def _cleanup_test_instance(self, instance_name):
        """Helper to clean up a specific test instance."""
        try:
            from multiprocessing.shared_memory import SharedMemory

            import posix_ipc

            # Shared memory segments for this instance (must match FlexibleSharedMemory naming)
            shm_names = [
                instance_name,
                "sm_" + instance_name + "_kd",
                instance_name + "_km",
                instance_name + "_c",
            ]

            # Semaphores / lock names for this instance
            sem_names = [
                instance_name,
                instance_name + "_ks",
                instance_name + "_c",
            ]

            # Clean up semaphores
            for sem_name in sem_names:
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.close()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass
                try:
                    sem = posix_ipc.Semaphore(sem_name)
                    sem.unlink()
                except (posix_ipc.ExistentialError, FileNotFoundError):
                    pass

            # Clean up shared memory segments
            for shm_name in shm_names:
                try:
                    shm = SharedMemory(name=shm_name, create=False)
                    shm.close()
                    shm.unlink()
                except (FileNotFoundError, FileExistsError):
                    pass
        except ImportError:
            pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_keep_alive_parameters(self, lock_class):
        """Test different keep alive parameters."""
        fsm = FlexibleSharedMemory(
            name=self.test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_dict_size=256 * 1024,  # 256KB
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        assert fsm.keep_alive.keep_alive_seconds == 2
        # Note: keep_alive_dict_size is used internally but not directly accessible

        fsm.close()

    def test_error_handling_invalid_size(self):
        """Test error handling with invalid sizes."""
        # Test with zero size
        with pytest.raises((ValueError, OSError)):
            FlexibleSharedMemory(
                name=self.test_name,
                shm_size=0,
                pinned_memory=False,
            )

        # Test with negative size
        with pytest.raises((ValueError, OSError)):
            FlexibleSharedMemory(
                name=self.test_name,
                shm_size=-1,
                pinned_memory=False,
            )

    def test_memory_isolation(self):
        """Test that different named memories are isolated."""
        name1 = self.test_name + "_1"
        name2 = self.test_name + "_2"

        # Clean up any leftover resources
        self._cleanup_test_instance(name1)
        self._cleanup_test_instance(name2)

        fsm1 = FlexibleSharedMemory(
            name=name1,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        fsm2 = FlexibleSharedMemory(
            name=name2,
            shm_size=self.test_size,
            keep_alive_seconds=2,  # Short interval for faster test cleanup
            pinned_memory=False,
        )

        # Write different data to each
        data1 = b"Memory 1 data"
        data2 = b"Memory 2 data"

        fsm1.block.buf[: len(data1)] = data1
        fsm2.block.buf[: len(data2)] = data2

        # Each should retain its own data
        read_data1 = bytes(fsm1.block.buf[: len(data1)])
        read_data2 = bytes(fsm2.block.buf[: len(data2)])

        assert read_data1 == data1
        assert read_data2 == data2
        assert read_data1 != read_data2

        fsm2.close()
        fsm1.close()

        # Clean up after test
        self._cleanup_test_instance(name1)
        self._cleanup_test_instance(name2)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_init_failure_cleanup_creator_unlinks_segment(self, lock_class):
        """Test that when creator fails during init, the shared memory segment is unlinked.

        This prevents orphaned shared memory segments when initialization fails after
        the segment is created but before KeepAlive registration completes.
        """
        from multiprocessing.shared_memory import SharedMemory

        test_name = self.test_name + "_init_fail"
        self._cleanup_test_instance(test_name)

        # Mock KeepAlive to raise an exception after shared memory is created
        def failing_keepalive_init(self, *args, **kwargs):
            raise RuntimeError("Simulated KeepAlive initialization failure")

        with (
            mock.patch.object(KeepAlive, "__init__", failing_keepalive_init),
            pytest.raises(RuntimeError, match="Simulated KeepAlive initialization failure"),
        ):
            FlexibleSharedMemory(
                name=test_name,
                shm_size=self.test_size,
                lock_class=lock_class,
                pinned_memory=False,
            )

        # Verify the shared memory segment was cleaned up (unlinked)
        # Attempting to connect should fail with FileNotFoundError
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=test_name, create=False)

        # Clean up any remaining resources
        self._cleanup_test_instance(test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_init_failure_cleanup_connector_preserves_segment(self, lock_class):
        """Test that when connector fails during init, the shared memory segment is NOT unlinked.

        This ensures that if a process connecting to existing shared memory fails,
        it doesn't destroy the segment that other processes are using.
        """
        from multiprocessing.shared_memory import SharedMemory

        test_name = self.test_name + "_conn_fail"
        self._cleanup_test_instance(test_name)

        # First, create the shared memory segment successfully
        fsm_creator = FlexibleSharedMemory(
            name=test_name,
            shm_size=self.test_size,
            lock_class=lock_class,
            keep_alive_seconds=30,
            pinned_memory=False,
        )

        # Write test data to verify segment integrity later
        test_data = b"Creator data"
        fsm_creator.block.buf[: len(test_data)] = test_data

        # Now try to connect with a connector that will fail during KeepAlive init
        # We use shm_size=0 to indicate we're connecting (not creating)
        # But we need to make KeepAlive fail for the connector

        original_init = KeepAlive.__init__

        call_count = [0]

        def failing_on_second_keepalive_init(self_ka, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:  # Fail on second call (connector)
                raise RuntimeError("Simulated connector KeepAlive failure")
            return original_init(self_ka, *args, **kwargs)

        # Reset call count and patch
        call_count[0] = 1  # Set to 1 so next call (connector) will fail

        with (
            mock.patch.object(KeepAlive, "__init__", failing_on_second_keepalive_init),
            pytest.raises(RuntimeError, match="Simulated connector KeepAlive failure"),
        ):
            # Try to connect to existing memory (shm_size > 0 but segment already exists)
            FlexibleSharedMemory(
                name=test_name,
                shm_size=self.test_size,  # Will connect since segment exists
                lock_class=lock_class,
                pinned_memory=False,
            )

        # Verify the original shared memory segment still exists and has correct data
        # The creator's segment should NOT have been unlinked
        assert fsm_creator.shm is not None
        read_data = bytes(fsm_creator.block.buf[: len(test_data)])
        assert read_data == test_data

        # Also verify we can still connect to the segment
        try:
            check_shm = SharedMemory(name=test_name, create=False)
            check_shm.close()
        except FileNotFoundError:
            pytest.fail("Shared memory segment was incorrectly unlinked by failed connector")

        # Clean up
        fsm_creator.close()
        self._cleanup_test_instance(test_name)
