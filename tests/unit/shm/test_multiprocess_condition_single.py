# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-process unit tests for MultiprocessCondition class."""

import os
import time

import pytest

from flextensor.shm import MultiprocessCondition, ProcessFileLock, SemaphoreLock


class TestMultiprocessConditionSingle:
    """Single-process tests for MultiprocessCondition class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "mpc_s"  # Short name to avoid POSIX name limits
        # Clean up any leftovers before starting
        self._cleanup_resources(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        # Cleanup any leftover resources
        self._cleanup_resources(self.test_name)

    def _cleanup_resources(self, name):
        """Helper to cleanup test resources."""
        try:
            import posix_ipc

            # Cleanup main semaphore (same name as the condition)
            try:
                sem = posix_ipc.Semaphore(name)
                sem.unlink()
                sem.close()
            except (posix_ipc.ExistentialError, FileNotFoundError):
                pass

            # Cleanup any process-specific semaphores
            pid = os.getpid()
            try:
                sem_name = f"/{name}_{pid}"
                sem = posix_ipc.Semaphore(sem_name)
                sem.unlink()
                sem.close()
            except (posix_ipc.ExistentialError, FileNotFoundError):
                pass
        except ImportError:
            pass

        try:
            from multiprocessing.shared_memory import SharedMemory

            # Cleanup shared memory for SharedMultiString (same name as the condition)
            try:
                shm = SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        except ImportError:
            pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiprocess_condition_creation(self, lock_class):
        """Test MultiprocessCondition creation."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        assert condition.name == self.test_name
        assert condition.notification_list is not None
        assert condition.lock is not None

        # Should initially not be ready
        assert not condition.notification_list.is_ready()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_notify_ready_functionality(self, lock_class):
        """Test notify_ready functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Initially not ready
        assert not condition.notification_list.is_ready()

        # Notify ready
        condition.notify_ready()

        # Should now be ready
        assert condition.notification_list.is_ready()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_wait_for_ready_immediate(self, lock_class):
        """Test wait_for_ready when already ready."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Set ready first
        condition.notify_ready()

        # wait_for_ready should return immediately
        start_time = time.time()
        condition.wait_for_ready()
        wait_time = time.time() - start_time

        # Should be very fast since it's already ready
        assert wait_time < 0.1

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_protection(self, lock_class):
        """Test that semaphore properly protects critical sections."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Test acquiring semaphore
        condition.lock.acquire()

        # Should be able to access notification_list while holding semaphore
        condition.notification_list.append("test_entry")
        strings = condition.notification_list.get_list()
        assert "test_entry" in strings

        condition.lock.release()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_notification_list_interaction(self, lock_class):
        """Test interaction with underlying notification list."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Add process ID to notification list (simulating wait_for_ready behavior)
        with condition.lock:
            condition.notification_list.append(str(os.getpid()))
            strings = condition.notification_list.get_list()
            assert str(os.getpid()) in strings

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_connect_to_existing_condition(self, lock_class):
        """Test connecting to existing multiprocess condition."""
        # Create first condition
        condition1 = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
        condition1.notification_list.append("test_data")

        # Connect to existing condition
        condition2 = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)

        # Should see the same data
        strings = condition2.notification_list.get_list()
        assert "test_data" in strings

        condition2.close()
        condition1.close()
        condition1.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_functionality(self, lock_class):
        """Test close functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Verify it's working
        condition.notification_list.append("test")
        assert "test" in condition.notification_list.get_list()

        # Close should clean up
        condition.close()

        # notification_list should be closed
        with pytest.raises((ValueError, AttributeError)):
            condition.notification_list.append("should_fail")

        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_unlink_functionality(self, lock_class):
        """Test unlink functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # unlink() before close() should work and prevent reconnection
        condition.unlink()

        with pytest.raises((FileNotFoundError, ValueError)):
            MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_unlink_after_close_is_noop(self, lock_class):
        """Test that unlink after close is a no-op (does not raise)."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        condition.close()
        condition.unlink()  # Should not raise exception, but is a no-op

        # Shared memory still exists, so reconnection should succeed
        condition2 = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)
        condition2.close()
        condition2.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_close_lock_functionality(self, lock_class):
        """Test close_lock functionality."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        assert condition.lock is not None

        # Close semaphore specifically
        condition.close_lock()

        assert condition.lock is None

        # Should not be able to use semaphore after closing
        with pytest.raises(AttributeError):
            condition.lock.acquire()

        condition.close()
        condition.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    @pytest.mark.parametrize("create_flag", [True, False])
    def test_different_creation_modes(self, create_flag, lock_class):
        """Test different creation modes."""
        if create_flag:
            # Creating new condition
            condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)
            assert condition.notification_list is not None
            assert condition.lock is not None
            condition.close()
            condition.unlink()
        else:
            # First create a condition to connect to
            condition1 = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

            # Then connect without creating
            condition2 = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)
            assert condition2.notification_list is not None
            assert condition2.lock is not None

            condition2.close()
            condition1.close()
            condition1.unlink()

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_ready_state_persistence(self, lock_class):
        """Test that ready state persists across operations."""
        condition = MultiprocessCondition(name=self.test_name, is_creator=True, lock_class=lock_class)

        # Set ready
        condition.notify_ready()
        assert condition.notification_list.is_ready()

        # Add more data - ready state should persist
        with condition.lock:
            condition.notification_list.append("additional_data")

        # Should still be ready
        assert condition.notification_list.is_ready()

        condition.close()
        condition.unlink()
