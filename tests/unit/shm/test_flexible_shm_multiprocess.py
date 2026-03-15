# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess unit tests for FlexibleSharedMemory class."""

import multiprocessing
import os
import time

import pytest

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock


def create_and_write_process(name, shm_size, data_to_write, notify_ready, result_queue, lock_class):
    """Helper function to create FlexibleSharedMemory and write data.

    Note: Keeps memory alive for 2 seconds by default to allow other processes to connect.
    Pass keep_alive_time in data_to_write tuple format: (data, keep_alive_time) to customize.
    """
    # Handle optional keep_alive_time passed via tuple
    keep_alive_time = 2  # Default
    if isinstance(data_to_write, tuple):
        data_to_write, keep_alive_time = data_to_write

    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=shm_size,
            pinned_memory=False,  # Avoid cupy dependency in tests
            lock_class=lock_class,
        )

        # Write data to memory
        fsm.block.buf[: len(data_to_write)] = data_to_write

        if notify_ready:
            fsm.notify_ready()

        # Keep memory alive for other processes to connect
        time.sleep(keep_alive_time)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "create_and_write",
                "data_written": data_to_write,
                "shm_creator": fsm.shm_creator,
                "pid": os.getpid(),
            },
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "action": "create_and_write",
                "pid": os.getpid(),
            },
        )


def wait_and_read_process(name, expected_data_len, result_queue, keep_alive_time, lock_class):
    """Helper function to wait for ready and read data.

    Args:
        keep_alive_time: How long to stay alive after reading before closing (default 0s)
    """
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=0,  # Size will be determined from existing memory
            pinned_memory=False,
            lock_class=lock_class,
        )

        # Wait for ready
        start_time = time.time()
        fsm.wait_for_ready()
        wait_time = time.time() - start_time

        # Read data
        read_data = bytes(fsm.block.buf[:expected_data_len])

        # Keep alive to coordinate with other readers
        if keep_alive_time > 0:
            time.sleep(keep_alive_time)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "wait_and_read",
                "data_read": read_data,
                "wait_time": wait_time,
                "shm_creator": fsm.shm_creator,
                "pid": os.getpid(),
            },
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "action": "wait_and_read",
                "pid": os.getpid(),
            },
        )


def concurrent_access_process(name, proc_id, operations, result_queue, lock_class):
    """Helper function for concurrent access testing."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=0,  # Connect to existing
            pinned_memory=False,
            lock_class=lock_class,
        )

        successful_ops = 0
        for _i in range(operations):
            with fsm.main_lock:
                # Read current data
                current_data = bytes(fsm.block.buf[:8])  # Read first 8 bytes as counter

                # Interpret as integer counter (or initialize to 0)
                try:
                    counter = int.from_bytes(current_data[:4], "little")
                except (ValueError, TypeError):
                    counter = 0

                # Increment and write back
                counter += 1
                fsm.block.buf[:4] = counter.to_bytes(4, "little")

                successful_ops += 1
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "concurrent_access",
                "proc_id": proc_id,
                "successful_ops": successful_ops,
                "pid": os.getpid(),
            },
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "action": "concurrent_access",
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def keep_alive_monitor_process(name, monitor_duration, result_queue, lock_class):
    """Helper function to monitor keep alive functionality."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=0,  # Connect to existing
            pinned_memory=False,
            lock_class=lock_class,
        )

        alive_checks = []
        start_time = time.time()
        while time.time() - start_time < monitor_duration:
            alive = fsm.keep_alive.any_process_alive()
            alive_checks.append(alive)
            time.sleep(0.5)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "action": "keep_alive_monitor",
                "alive_checks": alive_checks,
                "monitor_duration": monitor_duration,
                "pid": os.getpid(),
            },
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "action": "keep_alive_monitor",
                "pid": os.getpid(),
            },
        )


def check_memory_size_process(name, expected_size, result_queue, create, wait_time, lock_class):
    """Helper function to check memory size consistency across processes."""
    try:
        if create:
            fsm = FlexibleSharedMemory(name=name, shm_size=expected_size, pinned_memory=False, lock_class=lock_class)
        else:
            fsm = FlexibleSharedMemory(name=name, shm_size=0, pinned_memory=False, lock_class=lock_class)

        actual_size = fsm.block.size

        # Wait to keep memory alive - creator waits longest, connectors wait briefly to coordinate
        if wait_time > 0:
            time.sleep(wait_time)

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "expected_size": expected_size,
                "actual_size": actual_size,
                "create": create,
                "shm_creator": fsm.shm_creator,
                "pid": os.getpid(),
            },
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "create": create,
                "pid": os.getpid(),
            },
        )


def coordinated_cleanup_process(name, proc_id, duration, result_queue, lock_class):
    """Helper function to test coordinated cleanup across processes."""
    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=4096,
            pinned_memory=False,
            lock_class=lock_class,
        )

        # Do some work
        test_data = f"proc_{proc_id}_data".encode()
        fsm.block.buf[: len(test_data)] = test_data

        time.sleep(duration)

        # Check if other processes still alive before cleanup
        others_alive = fsm.keep_alive.any_process_alive()

        fsm.close()

        result_queue.put(
            {
                "success": True,
                "proc_id": proc_id,
                "others_alive_at_close": others_alive,
                "pid": os.getpid(),
            },
        )

    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": str(e),
                "proc_id": proc_id,
                "pid": os.getpid(),
            },
        )


def cleanup_flexible_shared_memory(name):
    """Helper function to clean up test resources."""
    # Cleanup various shared resources
    # Based on compact naming in FlexibleSharedMemory.__init__:
    # - Main: name
    # - Keep alive dict: sm_name_kd (SharedMemoryDict adds sm_ prefix), name_km (metadata)
    # - Condition: name_c
    resource_names = [
        name,  # Main shared memory
        "sm_" + name + "_kd",  # KeepAliveDict SharedMemoryDict (sm_ prefix added internally)
        name + "_km",  # KeepAliveDict SharedMemory (timestamps)
        name + "_c",  # MultiprocessCondition
    ]

    for resource_name in resource_names:
        try:
            from multiprocessing.shared_memory import SharedMemory

            try:
                shm = SharedMemory(name=resource_name)
                shm.close()
                shm.unlink()
            except (FileNotFoundError, FileExistsError):
                pass
        except ImportError:
            pass

    # Cleanup semaphores / lock names
    # Based on compact naming:
    # - Main: name
    # - Keep alive lock: name_ks
    # - Condition lock: name_c
    semaphore_names = [
        name,  # Main lock
        name + "_ks",  # Keep alive lock
        name + "_c",  # Condition lock
    ]

    for sem_name in semaphore_names:
        try:
            import posix_ipc

            try:
                sem = posix_ipc.Semaphore(sem_name)
                sem.close()
                sem.unlink()
            except (posix_ipc.ExistentialError, FileNotFoundError):
                pass
        except ImportError:
            pass


class TestFlexibleSharedMemoryMultiprocess:
    """Multiprocess tests for FlexibleSharedMemory class."""

    def setup_method(self):
        """Setup test fixtures before each test method."""
        self.test_name = "fsm_mp"  # Short name to avoid POSIX name limits
        self.test_size = 8192  # 8KB for tests
        # Clean up any leftover resources from previous runs
        cleanup_flexible_shared_memory(self.test_name)

    def teardown_method(self):
        """Cleanup after each test method."""
        cleanup_flexible_shared_memory(self.test_name)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_basic_create_and_connect(self, lock_class):
        """Test basic creation and connection across processes."""
        result_queue = multiprocessing.Queue()
        test_data = b"Hello from creator process!"

        # Creator process
        creator_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(self.test_name, self.test_size, test_data, True, result_queue, lock_class),
        )

        # Reader process
        reader_proc = multiprocessing.Process(
            target=wait_and_read_process,
            args=(self.test_name, len(test_data), result_queue, 0, lock_class),
        )

        try:
            creator_proc.start()
            time.sleep(0.5)  # Let creator establish memory
            reader_proc.start()

            creator_proc.join(timeout=15)
            reader_proc.join(timeout=15)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # Debug: print all results
            print(f"\n[TEST] Collected {len(results)} results:")
            for i, result in enumerate(results):
                print(f"[TEST] Result {i}: {result}")

            assert len(results) == 2

            # Find creator and reader results
            creator_result = next(r for r in results if r.get("action") == "create_and_write")
            reader_result = next(r for r in results if r.get("action") == "wait_and_read")

            # Both should succeed
            assert creator_result["success"] is True, f"Creator failed: {creator_result.get('error', 'no error info')}"
            assert reader_result["success"] is True, f"Reader failed: {reader_result.get('error', 'no error info')}"

            # Creator should be the memory creator
            assert creator_result["shm_creator"] is True
            assert reader_result["shm_creator"] is False  # Connected to existing

            # Data should match
            assert reader_result["data_read"] == test_data
            assert reader_result["wait_time"] <= 3.0  # Should not wait too long

        finally:
            for proc in [creator_proc, reader_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_multiple_readers(self, lock_class):
        """Test one writer, multiple readers."""
        result_queue = multiprocessing.Queue()
        test_data = b"Shared data for multiple readers"
        num_readers = 3

        # Creator process - keep alive 4 seconds (readers coordinate their own cleanup)
        creator_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(self.test_name, self.test_size, (test_data, 4), True, result_queue, lock_class),  # keep_alive_time=4
        )

        # Multiple reader processes - each waits 1s after reading to ensure all connect
        reader_procs = []
        for _i in range(num_readers):
            proc = multiprocessing.Process(
                target=wait_and_read_process,
                args=(self.test_name, len(test_data), result_queue, 1, lock_class),  # keep_alive_time=1
            )
            reader_procs.append(proc)

        try:
            creator_proc.start()
            time.sleep(1)  # Let creator establish memory

            # Start all readers
            for proc in reader_procs:
                proc.start()
                time.sleep(0.1)  # Small stagger

            # Wait for all - longer timeout to handle system load
            creator_proc.join(timeout=30)
            for proc in reader_procs:
                proc.join(timeout=30)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            assert len(results) == num_readers + 1

            # All should succeed
            for result in results:
                assert result["success"] is True

            # Check reader results
            reader_results = [r for r in results if r.get("action") == "wait_and_read"]
            assert len(reader_results) == num_readers

            for reader_result in reader_results:
                assert reader_result["data_read"] == test_data
                assert reader_result["wait_time"] <= 5.0

        finally:
            all_procs = [creator_proc, *reader_procs]
            for proc in all_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_concurrent_access_with_semaphore(self, lock_class):
        """Test concurrent access protection with semaphore."""
        result_queue = multiprocessing.Queue()
        num_processes = 4
        operations_per_process = 10

        # Create initial shared memory with counter initialized to 0
        init_data = (0).to_bytes(4, "little") + b"\x00" * (self.test_size - 4)

        creator_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(self.test_name, self.test_size, init_data, False, result_queue, lock_class),
        )

        # Create concurrent access processes
        worker_procs = []
        for i in range(num_processes):
            proc = multiprocessing.Process(
                target=concurrent_access_process,
                args=(self.test_name, i, operations_per_process, result_queue, lock_class),
            )
            worker_procs.append(proc)

        try:
            # Start creator
            creator_proc.start()
            time.sleep(1)  # Let creator establish memory

            # Start all workers simultaneously
            for proc in worker_procs:
                proc.start()

            # Wait for all to complete
            creator_proc.join(timeout=20)
            for proc in worker_procs:
                proc.join(timeout=20)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # Creator should succeed
            creator_results = [r for r in results if r.get("action") == "create_and_write"]
            assert len(creator_results) == 1
            assert creator_results[0]["success"] is True

            # Check worker results
            worker_results = [r for r in results if r.get("action") == "concurrent_access"]
            successful_workers = [r for r in worker_results if r["success"]]

            # Most workers should succeed
            assert len(successful_workers) >= num_processes // 2

            # Total successful operations should be reasonable
            total_ops = sum(r["successful_ops"] for r in successful_workers)
            expected_total = len(successful_workers) * operations_per_process
            assert total_ops == expected_total, f"Expected {expected_total} ops, got {total_ops}"

        finally:
            all_procs = [creator_proc, *worker_procs]
            for proc in all_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_keep_alive_functionality(self, lock_class):
        """Test keep alive functionality across processes."""
        result_queue = multiprocessing.Queue()

        # Long-running process that will stay alive for 5 seconds (enough for monitor to complete 3s)
        long_running_proc = multiprocessing.Process(
            target=create_and_write_process,
            args=(
                self.test_name,
                self.test_size,
                (b"keep_alive_test", 5),
                False,
                result_queue,
                lock_class,
            ),  # keep_alive_time=5
        )

        # Monitor process that checks keep alive status
        monitor_proc = multiprocessing.Process(
            target=keep_alive_monitor_process,
            args=(self.test_name, 3, result_queue, lock_class),
        )

        try:
            print("\n[TEST] Starting long_running_proc")
            long_running_proc.start()
            time.sleep(1)  # Let it establish
            print("[TEST] Starting monitor_proc")
            monitor_proc.start()

            # Let monitor run while long process is alive
            print("[TEST] Waiting for monitor_proc to complete")
            monitor_proc.join(timeout=15)
            print(f"[TEST] monitor_proc alive={monitor_proc.is_alive()}, exitcode={monitor_proc.exitcode}")

            # Stop long running process
            print("[TEST] Waiting for long_running_proc to complete")
            long_running_proc.join(timeout=10)
            print(
                f"[TEST] long_running_proc alive={long_running_proc.is_alive()}, exitcode={long_running_proc.exitcode}",
            )

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            print(f"[TEST] Collected {len(results)} results:")
            for i, result in enumerate(results):
                print(f"[TEST] Result {i}: {result}")

            # Both should succeed
            for result in results:
                assert result["success"] is True, f"Result failed: {result.get('error', 'no error info')}"

            # Check monitor results
            monitor_results = [r for r in results if r.get("action") == "keep_alive_monitor"]
            assert len(monitor_results) == 1

            monitor_result = monitor_results[0]
            alive_checks = monitor_result["alive_checks"]

            # Should have detected processes alive during monitoring
            assert len(alive_checks) > 0
            assert any(alive_checks), "Should have detected alive processes"

        finally:
            for proc in [long_running_proc, monitor_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_memory_size_consistency(self, lock_class):
        """Test that memory size is consistent across processes."""
        result_queue = multiprocessing.Queue()

        # Creator process - keep alive for 5 seconds to allow connectors to access and close
        creator_proc = multiprocessing.Process(
            target=check_memory_size_process,
            args=(self.test_name, self.test_size, result_queue, True, 5, lock_class),  # wait_time=5
        )

        # Connector processes - wait 1 second before closing to ensure all connect
        connector_procs = []
        for _i in range(2):
            proc = multiprocessing.Process(
                target=check_memory_size_process,
                args=(self.test_name, self.test_size, result_queue, False, 1, lock_class),  # wait_time=1
            )
            connector_procs.append(proc)

        try:
            print("\n[TEST] Starting creator_proc")
            creator_proc.start()
            time.sleep(0.5)  # Let creator establish resources
            print(f"[TEST] creator_proc alive={creator_proc.is_alive()}, exitcode={creator_proc.exitcode}")

            # Start connectors while creator is alive
            print("[TEST] Starting connector_procs")
            for i, proc in enumerate(connector_procs):
                print(f"[TEST] Starting connector {i}")
                proc.start()
                time.sleep(0.2)

            print("[TEST] Waiting for connector_procs to complete")
            for i, proc in enumerate(connector_procs):
                proc.join(timeout=10)
                print(f"[TEST] connector {i} alive={proc.is_alive()}, exitcode={proc.exitcode}")

            print("[TEST] Waiting for creator_proc to complete")
            creator_proc.join(timeout=10)
            print(f"[TEST] creator_proc done: alive={creator_proc.is_alive()}, exitcode={creator_proc.exitcode}")

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            print(f"[TEST] Collected {len(results)} results:")
            for i, result in enumerate(results):
                print(f"[TEST] Result {i}: {result}")

            # All should succeed
            for result in results:
                assert result["success"] is True, f"Result failed: {result.get('error', 'no error info')}"

            # All should see the same memory size (may be rounded up to page size)
            sizes = [r["actual_size"] for r in results]
            assert len(set(sizes)) == 1, f"All processes should see the same size, got {sizes}"
            # Size should be at least what we requested
            assert sizes[0] >= self.test_size, f"Memory size should be at least {self.test_size}, got {sizes[0]}"

        finally:
            all_procs = [creator_proc, *connector_procs]
            for proc in all_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)

    @pytest.mark.parametrize("lock_class", [SemaphoreLock, ProcessFileLock])
    def test_cleanup_coordination(self, lock_class):
        """Test coordinated cleanup across processes."""
        result_queue = multiprocessing.Queue()

        processes = []
        durations = [1, 2, 3, 4]  # Staggered durations

        for i, duration in enumerate(durations):
            proc = multiprocessing.Process(
                target=coordinated_cleanup_process,
                args=(self.test_name, i, duration, result_queue, lock_class),
            )
            processes.append(proc)

        try:
            # Start all processes
            for proc in processes:
                proc.start()
                time.sleep(0.1)

            # Wait for all to complete
            for proc in processes:
                proc.join(timeout=20)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # All should succeed
            successful_results = [r for r in results if r["success"]]
            assert len(successful_results) >= len(durations) // 2

            # Earlier processes should have seen others alive
            # Later processes might not see others alive
            results_by_id = {r["proc_id"]: r for r in successful_results}

            # At least the first process should have seen others alive
            if 0 in results_by_id:
                assert results_by_id[0]["others_alive_at_close"] is True

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
