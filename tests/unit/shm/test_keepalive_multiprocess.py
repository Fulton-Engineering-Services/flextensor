# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for KeepAlive class."""

import multiprocessing
import os
import time
from multiprocessing.shared_memory import SharedMemory

import posix_ipc
import pytest

from flextensor.shm import KeepAlive, ProcessFileLock, SemaphoreLock


def create_keepalive_process(
    name,
    label,
    keep_alive_seconds,
    create_semaphore,
    result_queue,
    duration,
    lock_class,
):
    """Helper function to create KeepAlive in a separate process."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),  # Use actual integer PID
            keep_alive_seconds=keep_alive_seconds,
            is_creator=create_semaphore,
            lock_class=lock_class,
        )

        # Let it run for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.5)
            if not keep_alive.any_process_alive():
                break

        keep_alive.close(any_process_alive=True)
        result_queue.put({"success": True, "process_id": label})

    except Exception as e:  # noqa: BLE001  # broad Exception intentional to tolerate transient IPC failures
        result_queue.put({"success": False, "error": str(e), "process_id": label})


def monitor_keepalive_process(name, label, keep_alive_seconds, result_queue, monitor_duration, lock_class):
    """Helper function to monitor an existing KeepAlive in a separate process."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),  # Use actual integer PID
            keep_alive_seconds=keep_alive_seconds,
            is_creator=False,  # Connect to existing
            lock_class=lock_class,
        )

        alive_checks = []
        start_time = time.time()
        while time.time() - start_time < monitor_duration:
            alive_checks.append(keep_alive.any_process_alive())
            time.sleep(0.5)

        keep_alive.close(any_process_alive=True)
        result_queue.put(
            {
                "success": True,
                "process_id": label,
                "alive_checks": alive_checks,
            },
        )

    except Exception as e:  # noqa: BLE001  # broad Exception intentional to tolerate transient IPC failures
        result_queue.put({"success": False, "error": str(e), "process_id": label})


def cleanup_resources(name):
    """Helper function to clean up test resources."""
    # Give time for any background processes to finish cleanup
    time.sleep(0.5)

    # Clean up semaphore (name) - retry multiple times if busy
    for _ in range(3):
        try:
            sem = posix_ipc.Semaphore(name)
            sem.close()
            sem.unlink()
            break
        except (posix_ipc.ExistentialError, FileNotFoundError):
            break
        except (posix_ipc.BusyError, OSError):
            time.sleep(0.3)
            continue

    # Clean up KeepAliveDict shared memory resources
    dict_name = f"{name}_dict"

    # Clean up _d (SharedMemoryDict) - retry if needed
    for _ in range(3):
        try:
            shm_d = SharedMemory(name=f"{dict_name}_d")
            shm_d.close()
            shm_d.unlink()
            break
        except FileNotFoundError:
            break
        except (OSError, Exception):  # noqa: BLE001  # broad Exception intentional to tolerate transient IPC failures
            time.sleep(0.1)
            continue

    # Clean up _m (SharedMemory) - retry if needed
    for _ in range(3):
        try:
            shm_m = SharedMemory(name=f"{dict_name}_m")
            shm_m.close()
            shm_m.unlink()
            break
        except FileNotFoundError:
            break
        except (OSError, Exception):  # noqa: BLE001  # broad Exception intentional to tolerate transient IPC failures
            time.sleep(0.1)
            continue

    # Give time for OS to fully release resources
    time.sleep(0.5)


def quick_cleanup_process(name, label, result_queue, lock_class, is_creator=False):
    """Process that quickly creates and cleans up KeepAlive."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),  # Use actual integer PID
            keep_alive_seconds=1,
            is_creator=is_creator,
            lock_class=lock_class,
        )
        time.sleep(0.5)  # Brief operation
        keep_alive.close(any_process_alive=False)  # Force cleanup
        result_queue.put({"success": True, "process_id": label})
    except Exception as e:  # noqa: BLE001  # broad Exception intentional to tolerate transient IPC failures
        result_queue.put({"success": False, "error": str(e), "process_id": label})


def long_holding_process(name, label, result_queue, lock_class):
    """Process that holds semaphore for a longer time."""
    try:
        keep_alive = KeepAlive(
            name=name,
            process_id=os.getpid(),  # Use actual integer PID
            keep_alive_seconds=1,
            is_creator=True,
            lock_class=lock_class,
        )

        # Hold semaphore for longer time
        with keep_alive.lock:
            time.sleep(2)  # Hold for 2 seconds
            # Do some work while holding semaphore (but don't call any_process_alive
            # here as it would try to acquire the semaphore again, causing deadlock)

        # Check alive status after releasing the semaphore
        alive = keep_alive.any_process_alive()

        keep_alive.close(any_process_alive=False)
        result_queue.put({"success": True, "process_id": label, "alive": alive})
    except Exception as e:  # noqa: BLE001  # broad Exception intentional to tolerate transient IPC failures
        result_queue.put({"success": False, "error": str(e), "process_id": label})


class TestKeepAliveMultiprocess:
    """Multiprocess tests for KeepAlive class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "ka_mp"  # Short name to avoid POSIX name limits
        self.keep_alive_seconds = 2
        # Clean up any leftover resources from previous runs
        cleanup_resources(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_resources(self.test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiprocess_keepalive_basic(self, lock_class):
        """Test basic multiprocess KeepAlive functionality."""
        result_queue = multiprocessing.Queue()

        # Start two processes with KeepAlive
        process1 = multiprocessing.Process(
            target=create_keepalive_process,
            args=(self.test_name, "proc1", self.keep_alive_seconds, True, result_queue, 3, lock_class),
        )
        process2 = multiprocessing.Process(
            target=create_keepalive_process,
            args=(self.test_name, "proc2", self.keep_alive_seconds, False, result_queue, 3, lock_class),
        )

        try:
            process1.start()
            time.sleep(0.5)  # Let first process create semaphore
            process2.start()

            # Wait for both processes to complete
            process1.join(timeout=10)
            process2.join(timeout=10)

            # Check results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            assert len(results) == 2
            for result in results:
                assert result["success"] is True, (
                    f"Process {result['process_id']} failed: {result.get('error', 'Unknown error')}"
                )

        finally:
            # Cleanup processes
            for proc in [process1, process2]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_process_detection_and_cleanup(self, lock_class):
        """Test that processes can detect each other and cleanup dead processes."""
        result_queue = multiprocessing.Queue()

        # Create first process (will be the creator)
        creator_proc = multiprocessing.Process(
            target=create_keepalive_process,
            args=(self.test_name, "creator", self.keep_alive_seconds, True, result_queue, 6, lock_class),
        )

        # Create monitor process that will check for other processes
        monitor_proc = multiprocessing.Process(
            target=monitor_keepalive_process,
            args=(self.test_name, "monitor", self.keep_alive_seconds, result_queue, 4, lock_class),
        )

        try:
            creator_proc.start()
            time.sleep(1)  # Let creator establish itself
            monitor_proc.start()

            # Wait for monitor to complete first
            monitor_proc.join(timeout=10)
            creator_proc.join(timeout=10)

            # Check results
            results = {}
            while not result_queue.empty():
                result = result_queue.get()
                results[result["process_id"]] = result

            assert len(results) == 2
            assert results["creator"]["success"] is True
            assert results["monitor"]["success"] is True

            # Monitor should have detected other processes alive
            alive_checks = results["monitor"]["alive_checks"]
            assert any(alive_checks), "Monitor should have detected other processes alive"

        finally:
            for proc in [creator_proc, monitor_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_race_condition_protection(self, lock_class):
        """Test that semaphores protect against race conditions."""
        result_queue = multiprocessing.Queue()
        processes = []

        # Start multiple processes simultaneously
        for i in range(4):
            create_semaphore = i == 0  # Only first process creates semaphore
            proc = multiprocessing.Process(
                target=create_keepalive_process,
                args=(
                    self.test_name,
                    f"proc_{i}",
                    self.keep_alive_seconds,
                    create_semaphore,
                    result_queue,
                    3,
                    lock_class,
                ),
            )
            processes.append(proc)

        try:
            # Start all processes quickly to test race conditions
            for proc in processes:
                proc.start()
                time.sleep(0.1)  # Small delay to avoid overwhelming

            # Wait for all processes
            for proc in processes:
                proc.join(timeout=15)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            assert len(results) == 4
            success_count = sum(1 for r in results if r["success"])

            # At least the creator should succeed, others might fail due to timing
            assert success_count >= 1, f"At least one process should succeed, got {success_count} successes"

            # Check for any specific errors that indicate race conditions
            for result in results:
                if not result["success"]:
                    error = result.get("error", "")
                    # These are acceptable errors due to cleanup timing
                    assert any(
                        acceptable in error.lower()
                        for acceptable in [
                            "existential",
                            "not found",
                            "no such file",
                            "already exists",
                        ]
                    ), f"Unexpected error that might indicate race condition: {error}"

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_no_deadlock_on_cleanup(self, lock_class):
        """Test that cleanup operations don't cause deadlocks."""
        result_queue = multiprocessing.Queue()

        processes = []
        for i in range(3):
            label = "creator" if i == 0 else f"proc_{i}"
            is_creator = i == 0
            proc = multiprocessing.Process(
                target=quick_cleanup_process,
                args=(self.test_name, label, result_queue, lock_class, is_creator),
            )
            processes.append(proc)

        try:
            # Start processes with small delays
            for i, proc in enumerate(processes):
                proc.start()
                if i == 0:
                    time.sleep(0.2)  # Let creator establish first

            # All should complete within reasonable time (no deadlocks)
            for proc in processes:
                proc.join(timeout=10)
                assert not proc.is_alive(), "Process should have completed (no deadlock)"

            # Check results - at least creator should succeed
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            success_count = sum(1 for r in results if r["success"])
            assert success_count >= 1, "At least creator process should succeed"

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_timeout_behavior(self, lock_class):
        """Test semaphore behavior under timeout conditions."""
        result_queue = multiprocessing.Queue()

        process = multiprocessing.Process(
            target=long_holding_process,
            args=(self.test_name, "holder", result_queue, lock_class),
        )

        try:
            process.start()
            process.join(timeout=15)  # Give enough time but not infinite

            assert not process.is_alive(), "Process should complete without hanging"

            # Check result
            assert not result_queue.empty()
            result = result_queue.get()
            assert result["success"] is True
            assert result["alive"] is True

        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
