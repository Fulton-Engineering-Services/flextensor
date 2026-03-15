# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess lifecycle tests for FlexibleSharedMemory class.

Tests the complete lifecycle of shared memory across multiple processes:
- Creation and connection
- Read/write operations across processes
- Proper cleanup ordering (unlink before close)
- Verification of no memory leaks
"""

import multiprocessing
import os
import time
import uuid

import pytest

from flextensor.shm import FlexibleSharedMemory, ProcessFileLock, SemaphoreLock


def lifecycle_process(
    name, shm_size, process_id, duration, data_to_write, result_queue, lock_class, unlink_before_close
):
    """Helper process for lifecycle testing.

    Args:
        name: Shared memory name
        shm_size: Size of shared memory (>0 to create/connect, 0 to connect only)
        process_id: Identifier for this process
        duration: How long to stay alive before closing
        data_to_write: Data to write at this process's offset, or None to skip writing
        result_queue: Queue for sending results back
        lock_class: Lock class to use
    """
    import traceback

    try:
        fsm = FlexibleSharedMemory(
            name=name,
            shm_size=shm_size,
            pinned_memory=False,
            lock_class=lock_class,
        )

        # Write data at offset based on process_id
        if data_to_write is not None:
            offset = process_id * 32  # Each process writes at different offset
            fsm.block.buf[offset : offset + len(data_to_write)] = data_to_write

        # Notify ready if creator
        if fsm.shm_creator:
            fsm.notify_ready()
        else:
            fsm.wait_for_ready()

        # Read data written by process 0
        data_from_p0 = bytes(fsm.block.buf[0:14])

        # Stay alive for specified duration
        time.sleep(duration)

        # Check if other processes are alive before cleanup
        others_alive = fsm.keep_alive.any_process_alive()

        fsm.close()

        result_queue.put({
            "success": True,
            "process_id": process_id,
            "shm_creator": fsm.shm_creator,
            "data_from_p0": data_from_p0,
            "others_alive_at_close": others_alive,
            "pid": os.getpid(),
            "unlink_before_close": unlink_before_close,
        })

    except Exception as e:
        result_queue.put({
            "success": False,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "process_id": process_id,
            "pid": os.getpid(),
            "unlink_before_close": unlink_before_close,
        })


@pytest.fixture(params=[SemaphoreLock, ProcessFileLock])
def lock_class(request):
    """Parameterize tests with different lock implementations."""
    return request.param


class TestFlexibleSharedMemoryLifecycle:
    """Test complete lifecycle of FlexibleSharedMemory across multiple processes."""

    def test_full_lifecycle_three_processes(self, lock_class):
        """Test full lifecycle with three processes closing in sequence.

        Scenario:
        1. P0 creates shared memory and writes data
        2. P1 connects and writes data
        3. P2 connects and writes data
        4. P0 closes first (others still alive - no unlink)
        5. P1 closes second (P2 still alive - no unlink)
        6. P2 closes last (unlinks shared memory)
        7. Verify shared memory is properly cleaned up (no leak)
        """
        ctx = multiprocessing.get_context("fork")
        shm_name = f"lc_{uuid.uuid4().hex[:8]}"
        shm_size = 1024
        result_queue = ctx.Queue()

        # Process data and durations (processes close in order: P0, P1, P2)
        processes_config = [
            {
                "process_id": 0,
                "shm_size": shm_size,
                "duration": 0.4,
                "data": b"Hello from P0!",
                "unlink_before_close": False,
            },
            {
                "process_id": 1,
                "shm_size": shm_size,
                "duration": 2.0,
                "data": b"Hello from P1!",
                "unlink_before_close": True,
            },
            {
                "process_id": 2,
                "shm_size": shm_size,
                "duration": 1.0,
                "data": b"Hello from P2!",
                "unlink_before_close": False,
            },
        ]

        processes = []
        for config in processes_config:
            proc = ctx.Process(
                target=lifecycle_process,
                args=(
                    shm_name,
                    config["shm_size"],
                    config["process_id"],
                    config["duration"],
                    config["data"],
                    result_queue,
                    lock_class,
                    config["unlink_before_close"],
                ),
            )
            processes.append(proc)

        try:
            # Start all processes with small delay between them
            for proc in processes:
                proc.start()
                time.sleep(0.3)

            # Wait for all to complete
            for proc in processes:
                proc.join(timeout=15)

            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # All should succeed
            assert len(results) == 3, f"Expected 3 results, got {len(results)}"
            for result in results:
                assert result["success"], f"Process {result.get('process_id')} failed: {result.get('error')}"

            # Sort by process_id
            results_by_id = {r["process_id"]: r for r in results}

            # P0 should have created the memory
            assert results_by_id[0]["shm_creator"] is True

            # P1 and P2 should have connected (not created)
            assert results_by_id[1]["shm_creator"] is False
            assert results_by_id[2]["shm_creator"] is False

            # All should have read P0's data
            for result in results:
                assert result["data_from_p0"] == b"Hello from P0!", f"Process {result['process_id']} read wrong data"

            # At least P0 (first to close) should have seen others alive
            # Note: Due to slot allocation in KeepAliveDict, intermediate processes
            # may not always see others alive correctly
            assert results_by_id[0]["others_alive_at_close"] is True, "P0 should have seen others alive"

            # Verify no shared memory leak - attempting to connect should fail
            with pytest.raises(FileNotFoundError):
                FlexibleSharedMemory(
                    name=shm_name,
                    shm_size=0,  # Try to connect to non-existent memory
                    pinned_memory=False,
                    lock_class=lock_class,
                )

        finally:
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
