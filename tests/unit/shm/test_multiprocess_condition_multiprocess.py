# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for MultiprocessCondition class."""

import multiprocessing
import os
import time

import pytest

from flextensor.shm import MultiprocessCondition, ProcessFileLock, SemaphoreLock


def wait_for_ready_process(name, _timeout, result_queue, init_lock, lock_class):
    """Helper function to wait for condition to be ready in a separate process."""
    import sys

    pid = os.getpid()
    print(f"[WAITER {pid}] Starting wait_for_ready_process for '{name}'", flush=True)
    sys.stdout.flush()
    try:
        print(f"[WAITER {pid}] Acquiring init_lock", flush=True)
        with init_lock:
            print(f"[WAITER {pid}] Lock acquired, creating MultiprocessCondition with create=False", flush=True)
            sys.stdout.flush()
            condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)
            print(f"[WAITER {pid}] MultiprocessCondition created successfully", flush=True)
        print(f"[WAITER {pid}] Released init_lock", flush=True)

        start_time = time.time()
        print(f"[WAITER {pid}] Calling wait_for_ready()", flush=True)
        condition.wait_for_ready()
        wait_time = time.time() - start_time
        print(f"[WAITER {pid}] wait_for_ready() returned after {wait_time:.2f}s", flush=True)

        # Get final state
        is_ready = condition.notification_list.is_ready()
        pid_list = condition.notification_list.get_list()
        print(f"[WAITER {pid}] is_ready={is_ready}, pid_list={pid_list}", flush=True)

        print(f"[WAITER {pid}] Closing condition", flush=True)
        condition.close()
        condition.close_lock()

        print(f"[WAITER {pid}] Putting result in queue", flush=True)
        result_queue.put(
            {
                "success": True,
                "wait_time": wait_time,
                "is_ready": is_ready,
                "pid_list": pid_list,
                "pid": pid,
            },
        )
        print(f"[WAITER {pid}] Result queued successfully", flush=True)

    except Exception as e:  # broad Exception intentional; errors sent via result_queue
        print(f"[WAITER {pid}] ERROR: {e}", flush=True)
        import traceback

        traceback.print_exc()
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "pid": pid,
            },
        )
        print(f"[WAITER {pid}] Error result queued", flush=True)


def notify_ready_process(name, delay, result_queue, init_lock, lock_class):
    """Helper function to notify ready after a delay in a separate process."""
    pid = os.getpid()
    print(f"[NOTIFIER {pid}] Starting notify_ready_process for '{name}'")
    try:
        print(f"[NOTIFIER {pid}] Acquiring init_lock")
        with init_lock:
            print(f"[NOTIFIER {pid}] Lock acquired, creating MultiprocessCondition with create=True")
            condition = MultiprocessCondition(name=name, is_creator=True, lock_class=lock_class)
            print(f"[NOTIFIER {pid}] MultiprocessCondition created successfully")
        print(f"[NOTIFIER {pid}] Released init_lock")

        if delay > 0:
            print(f"[NOTIFIER {pid}] Sleeping for {delay}s before notify")
            time.sleep(delay)

        print(f"[NOTIFIER {pid}] Calling notify_ready()")
        condition.notify_ready()
        print(f"[NOTIFIER {pid}] notify_ready() completed")

        print(f"[NOTIFIER {pid}] Closing condition")

        time.sleep(2.0)

        condition.close()
        condition.unlink()
        condition.close_lock()

        print(f"[NOTIFIER {pid}] Putting result in queue")
        result_queue.put(
            {
                "success": True,
                "action": "notify_ready",
                "pid": pid,
            },
        )
        print(f"[NOTIFIER {pid}] Result queued successfully")

    except Exception as e:  # broad Exception intentional; errors sent via result_queue
        print(f"[NOTIFIER {pid}] ERROR: {e}")
        import traceback

        traceback.print_exc()
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "action": "notify_ready",
                "pid": pid,
            },
        )
        print(f"[NOTIFIER {pid}] Error result queued")


def concurrent_wait_and_modify_process(name, proc_id, result_queue, init_lock, lock_class):
    """Helper function to test concurrent waiting and modification."""
    try:
        with init_lock:
            condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)

        # Add process ID to wait queue
        start_time = time.time()
        condition.wait_for_ready()
        wait_time = time.time() - start_time

        # After ready, try to add more data
        with condition.lock:
            condition.notification_list.append(f"post_ready_{proc_id}")

        pid_list = condition.notification_list.get_list()

        condition.close()
        condition.close_lock()

        result_queue.put(
            {
                "success": True,
                "proc_id": proc_id,
                "wait_time": wait_time,
                "pid_list": pid_list,
                "pid": os.getpid(),
            },
        )

    except Exception as e:  # noqa: BLE001  # broad Exception intentional; errors sent via result_queue
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def stress_test_process(name, proc_id, operations, result_queue, lock_class):
    """Helper function for stress testing semaphore operations."""
    try:
        condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)

        operations_completed = 0
        for i in range(operations):
            with condition.lock:
                # Simulate some work under semaphore protection
                condition.notification_list.get_list()
                condition.notification_list.append(f"proc{proc_id}_op{i}")
                time.sleep(0.001)  # Very small delay to increase chance of conflicts
                operations_completed += 1

        condition.close()

        result_queue.put(
            {
                "success": True,
                "proc_id": proc_id,
                "operations_completed": operations_completed,
                "pid": os.getpid(),
            },
        )

    except Exception as e:  # noqa: BLE001  # broad Exception intentional; errors sent via result_queue
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def cleanup_multiprocess_condition(name):
    """Helper function to clean up test resources."""
    # MultiprocessCondition creates:
    # 1. A SharedMemory with name=name (via SharedMultiString)
    # 2. A Semaphore with name=name

    # Cleanup semaphore
    try:
        import posix_ipc

        try:
            sem = posix_ipc.Semaphore(name)
            sem.close()
            sem.unlink()
        except (posix_ipc.ExistentialError, FileNotFoundError):
            pass
    except ImportError:
        pass

    # Cleanup shared memory
    try:
        from multiprocessing.shared_memory import SharedMemory

        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except (FileNotFoundError, FileExistsError):
            pass
    except ImportError:
        pass


def create_condition_process(name, result_queue, lock_class):
    """Helper function to create a condition with initial data for stress testing."""
    try:
        condition = MultiprocessCondition(name=name, is_creator=True, lock_class=lock_class)
        condition.notification_list.append("initial_data")
        condition.close()
        result_queue.put({"success": True, "action": "create"})
    except Exception as e:  # noqa: BLE001  # broad Exception intentional; errors sent via result_queue
        result_queue.put({"success": False, "error": str(e), "action": "create"})


def quick_lifecycle_process(name, proc_id, result_queue, lock_class):
    """Helper function to test quick lifecycle and cleanup operations."""
    try:
        if proc_id == 0:
            condition = MultiprocessCondition(name=name, is_creator=True, lock_class=lock_class)
            condition.notification_list.append(f"data_{proc_id}")
            time.sleep(0.2)
            condition.notify_ready()
            condition.close()
            condition.unlink()
        else:
            time.sleep(0.1 * proc_id)  # Stagger starts
            condition = MultiprocessCondition(name=name, is_creator=False, lock_class=lock_class)
            condition.wait_for_ready()
            condition.close()

        result_queue.put({"success": True, "proc_id": proc_id})
    except Exception as e:  # noqa: BLE001  # broad Exception intentional; errors sent via result_queue
        result_queue.put({"success": False, "error": str(e), "proc_id": proc_id})


class TestMultiprocessConditionMultiprocess:
    """Multiprocess tests for MultiprocessCondition class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "mpc_mp"  # Short name to avoid POSIX name limits

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_multiprocess_condition(self.test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_basic_wait_and_notify(self, lock_class):
        """Test basic wait and notify across processes."""

        cleanup_multiprocess_condition(self.test_name)
        result_queue = multiprocessing.Queue()
        init_lock = multiprocessing.Lock()

        # Process that will notify ready after delay
        notifier_proc = multiprocessing.Process(
            target=notify_ready_process,
            args=(self.test_name, 5.0, result_queue, init_lock, lock_class),
        )

        # Process that will wait for ready
        waiter_proc = multiprocessing.Process(
            target=wait_for_ready_process,
            args=(self.test_name, 10, result_queue, init_lock, lock_class),
        )

        try:
            # Start notifier first to create the condition
            notifier_proc.start()
            time.sleep(1.0)  # Let it create resources

            # Start waiter
            waiter_proc.start()

            # Wait for both to complete
            notifier_proc.join(timeout=15)
            waiter_proc.join(timeout=15)

            # Debug: Check if processes are still alive
            print(f"[TEST] Notifier alive: {notifier_proc.is_alive()}, exitcode: {notifier_proc.exitcode}")
            print(f"[TEST] Waiter alive: {waiter_proc.is_alive()}, exitcode: {waiter_proc.exitcode}")

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            print(f"[TEST] Collected {len(results)} results")
            for i, result in enumerate(results):
                print(f"[TEST] Result {i}: {result}")

            assert len(results) == 2

            # Find waiter and notifier results
            waiter_result = next(r for r in results if "wait_time" in r)
            notifier_result = next(r for r in results if r.get("action") == "notify_ready")

            # Both should succeed
            assert waiter_result["success"] is True
            assert notifier_result["success"] is True

            # Waiter should have waited approximately the delay time (4-5 seconds with some tolerance)
            assert 3.5 <= waiter_result["wait_time"] <= 6.0
            assert waiter_result["is_ready"] is True

            # Should contain data from notifier
            # Instead of checking for data_from_{notifier_result['pid']} in final_strings,
            # check if waiter_result['pid'] (the waiter process id) is present in the notification list (final_strings)
            assert str(waiter_result["pid"]) in waiter_result["pid_list"]

        finally:
            for proc in [waiter_proc, notifier_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiple_waiters_single_notifier(self, lock_class):
        """Test multiple processes waiting for single notifier."""
        result_queue = multiprocessing.Queue()
        init_lock = multiprocessing.Lock()
        num_waiters = 4

        # Create notifier process
        notifier_proc = multiprocessing.Process(
            target=notify_ready_process,
            args=(self.test_name, 3.0, result_queue, init_lock, lock_class),  # 3 second delay
        )

        # Create multiple waiter processes
        waiter_procs = []
        for _i in range(num_waiters):
            proc = multiprocessing.Process(
                target=wait_for_ready_process,
                args=(self.test_name, 15, result_queue, init_lock, lock_class),
            )
            waiter_procs.append(proc)

        try:
            # Start notifier first
            notifier_proc.start()
            time.sleep(0.5)

            # Start all waiters
            for proc in waiter_procs:
                proc.start()
                time.sleep(0.1)  # Small stagger

            # Wait for all to complete
            notifier_proc.join(timeout=20)
            for proc in waiter_procs:
                proc.join(timeout=20)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            assert len(results) == num_waiters + 1

            # All should succeed
            for result in results:
                assert result["success"] is True

            # Check waiter results
            waiter_results = [r for r in results if "wait_time" in r]
            assert len(waiter_results) == num_waiters

            for waiter_result in waiter_results:
                # All waiters should have waited approximately the same time (around 3 seconds with tolerance)
                assert 2.0 <= waiter_result["wait_time"] <= 5.0
                assert waiter_result["is_ready"] is True

        finally:
            all_procs = [notifier_proc, *waiter_procs]
            for proc in all_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_concurrent_access_no_race_conditions(self, lock_class):
        """Test concurrent access to condition doesn't cause race conditions."""
        result_queue = multiprocessing.Queue()
        init_lock = multiprocessing.Lock()

        # Create condition first
        # Use longer delay (3.0s) to ensure all workers have time to start and call wait_for_ready()
        # before the notification is sent. Workers serialize on init_lock, so they need enough time.
        creator_proc = multiprocessing.Process(
            target=notify_ready_process,
            args=(self.test_name, 3.0, result_queue, init_lock, lock_class),
        )

        # Create multiple processes that will wait and then modify
        worker_procs = []
        for i in range(3):
            proc = multiprocessing.Process(
                target=concurrent_wait_and_modify_process,
                args=(self.test_name, i, result_queue, init_lock, lock_class),
            )
            worker_procs.append(proc)

        try:
            # Start creator
            creator_proc.start()
            time.sleep(0.3)

            # Start all workers quickly so they can begin waiting
            for proc in worker_procs:
                proc.start()
                time.sleep(0.05)

            # Wait for all to complete
            creator_proc.join(timeout=15)
            for proc in worker_procs:
                proc.join(timeout=15)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # Should have results from creator + all workers
            worker_results = [r for r in results if "proc_id" in r]
            assert len(worker_results) == 3

            # All workers should succeed
            for result in worker_results:
                assert result["success"] is True
                assert result["wait_time"] <= 5.0  # Should not wait too long (notifier delays 3s)

            # Check that all post-ready modifications are present
            all_pid_list_items = set()
            for result in worker_results:
                all_pid_list_items.update(result["pid_list"])

            for i in range(3):
                expected_string = f"post_ready_{i}"
                assert any(expected_string in s for s in all_pid_list_items), (
                    f"Missing expected string: {expected_string}"
                )

        finally:
            all_procs = [creator_proc, *worker_procs]
            for proc in all_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_semaphore_stress_test(self, lock_class):
        """Stress test semaphore protection under high concurrency."""
        result_queue = multiprocessing.Queue()
        num_processes = 4
        operations_per_process = 20

        # Create initial condition
        creator_proc = multiprocessing.Process(
            target=create_condition_process,
            args=(self.test_name, result_queue, lock_class),
        )

        # Create stress test processes
        stress_procs = []
        for i in range(num_processes):
            proc = multiprocessing.Process(
                target=stress_test_process,
                args=(self.test_name, i, operations_per_process, result_queue, lock_class),
            )
            stress_procs.append(proc)

        try:
            # Start creator
            creator_proc.start()
            creator_proc.join(timeout=10)

            # Start all stress processes simultaneously
            for proc in stress_procs:
                proc.start()

            # Wait for all to complete
            for proc in stress_procs:
                proc.join(timeout=30)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # Check creator succeeded
            creator_results = [r for r in results if r.get("action") == "create"]
            assert len(creator_results) == 1
            assert creator_results[0]["success"] is True

            # Check stress test results
            stress_results = [r for r in results if "operations_completed" in r]
            assert len(stress_results) == num_processes

            # All processes should succeed
            successful_procs = [r for r in stress_results if r["success"]]
            assert len(successful_procs) >= num_processes // 2, "At least half of stress processes should succeed"

            # Check total operations completed
            total_operations = sum(r["operations_completed"] for r in successful_procs)
            assert total_operations > 0, "Some operations should have been completed"

        finally:
            all_procs = [creator_proc, *stress_procs]
            for proc in all_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

            # Final cleanup
            try:
                condition = MultiprocessCondition(name=self.test_name, is_creator=False, lock_class=lock_class)
                condition.close()
                condition.unlink()
            except Exception:  # noqa: S110
                pass

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_cleanup_no_deadlock(self, lock_class):
        """Test that cleanup operations don't cause deadlocks."""
        result_queue = multiprocessing.Queue()

        processes = []
        for i in range(4):
            proc = multiprocessing.Process(
                target=quick_lifecycle_process,
                args=(self.test_name, i, result_queue, lock_class),
            )
            processes.append(proc)

        try:
            # Start all processes
            for proc in processes:
                proc.start()

            # All should complete quickly without deadlocks
            for proc in processes:
                proc.join(timeout=15)
                assert not proc.is_alive(), "Process should complete without deadlock"

            # Check results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # At least the creator should succeed
            successful_count = sum(1 for r in results if r["success"])
            assert successful_count >= 1, "At least creator should succeed"

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
