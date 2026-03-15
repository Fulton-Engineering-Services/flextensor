# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple multiprocess test for SharedMultiString class."""

import multiprocessing
import os
import time

from flextensor.shm import SharedMultiString


def writer_process(name, result_queue, ready_event):
    """Process that writes data to SharedMultiString."""
    pid = os.getpid()
    print(f"[WRITER {pid}] Starting", flush=True)

    print(f"[WRITER {pid}] Creating SharedMultiString with create=True", flush=True)
    sms = SharedMultiString(name=name, create=True)
    print(f"[WRITER {pid}] SharedMultiString created", flush=True)

    # Write some data
    print(f"[WRITER {pid}] Appending 'item1'", flush=True)
    sms.append("item1")
    print(f"[WRITER {pid}] Appending 'item2'", flush=True)
    sms.append("item2")
    print(f"[WRITER {pid}] Appending 'item3'", flush=True)
    sms.append("item3")

    # Read back to verify
    items = sms.get_list()
    print(f"[WRITER {pid}] Read back items: {items}", flush=True)

    # Set ready flag
    print(f"[WRITER {pid}] Setting ready flag", flush=True)
    sms.set_ready()

    # Verify ready flag
    is_ready = sms.is_ready()
    print(f"[WRITER {pid}] Ready flag: {is_ready}", flush=True)

    # Signal parent that shared memory is ready for reading
    print(f"[WRITER {pid}] Signaling ready event", flush=True)
    ready_event.set()

    # Keep it alive for reader to finish
    time.sleep(2)

    print(f"[WRITER {pid}] Closing", flush=True)
    sms.close()

    result_queue.put(
        {
            "action": "write",
            "items_written": ["item1", "item2", "item3"],
            "pid": pid,
        },
    )
    print(f"[WRITER {pid}] Done", flush=True)


def reader_process(name, result_queue):
    """Process that reads data from SharedMultiString."""
    pid = os.getpid()
    print(f"[READER {pid}] Starting", flush=True)

    print(f"[READER {pid}] Creating SharedMultiString with create=False", flush=True)
    sms = SharedMultiString(name=name, create=False)
    print(f"[READER {pid}] SharedMultiString created", flush=True)

    # Read the data
    print(f"[READER {pid}] Reading items", flush=True)
    items = sms.get_list()
    print(f"[READER {pid}] Read items: {items}", flush=True)

    # Check ready flag
    print(f"[READER {pid}] Checking ready flag", flush=True)
    is_ready = sms.is_ready()
    print(f"[READER {pid}] Ready flag: {is_ready}", flush=True)

    print(f"[READER {pid}] Closing", flush=True)
    sms.close()

    result_queue.put(
        {
            "action": "read",
            "items_read": items,
            "is_ready": is_ready,
            "pid": pid,
        },
    )
    print(f"[READER {pid}] Done", flush=True)


def cleanup_shared_multi_string(name):
    """Cleanup SharedMultiString resources."""
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


def test_shared_multi_string_basic():
    """Test basic SharedMultiString data sharing between processes."""
    test_name = "sms_test"

    # Cleanup any leftover resources
    cleanup_shared_multi_string(test_name)

    result_queue = multiprocessing.Queue()
    ready_event = multiprocessing.Event()

    # Writer process
    writer_proc = multiprocessing.Process(
        target=writer_process,
        args=(test_name, result_queue, ready_event),
    )

    # Reader process
    reader_proc = multiprocessing.Process(
        target=reader_process,
        args=(test_name, result_queue),
    )

    try:
        # Start writer first to create the shared memory
        print("[TEST] Starting writer process")
        writer_proc.start()

        # Wait for writer to signal that shared memory is ready
        print("[TEST] Waiting for writer to signal readiness")
        if not ready_event.wait(timeout=10):
            raise TimeoutError("Writer process did not signal readiness within 10 seconds")
        print("[TEST] Writer signaled readiness")

        # Start reader
        print("[TEST] Starting reader process")
        reader_proc.start()

        # Wait for both to complete
        writer_proc.join(timeout=10)
        reader_proc.join(timeout=10)

        print(f"[TEST] Writer alive: {writer_proc.is_alive()}, exitcode: {writer_proc.exitcode}")
        print(f"[TEST] Reader alive: {reader_proc.is_alive()}, exitcode: {reader_proc.exitcode}")

        # Assert processes completed successfully (exceptions propagate as non-zero exitcode)
        assert writer_proc.exitcode == 0, f"Writer process failed with exitcode {writer_proc.exitcode}"
        assert reader_proc.exitcode == 0, f"Reader process failed with exitcode {reader_proc.exitcode}"

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        print(f"[TEST] Collected {len(results)} results")
        for i, result in enumerate(results):
            print(f"[TEST] Result {i}: {result}")

        # Verify results
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        writer_result = next((r for r in results if r.get("action") == "write"), None)
        reader_result = next((r for r in results if r.get("action") == "read"), None)

        assert writer_result is not None, "No writer result found"
        assert reader_result is not None, "No reader result found"

        # Check that reader read the same items that writer wrote
        expected_items = ["item1", "item2", "item3"]
        actual_items = reader_result["items_read"]

        print(f"[TEST] Expected items: {expected_items}")
        print(f"[TEST] Actual items: {actual_items}")

        assert actual_items == expected_items, f"Items mismatch: expected {expected_items}, got {actual_items}"

        # Check ready flag
        assert reader_result["is_ready"] is True, "Ready flag should be True"

        print("[TEST] All assertions passed!")

    finally:
        # Cleanup
        for proc in [writer_proc, reader_proc]:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1)

        cleanup_shared_multi_string(test_name)


if __name__ == "__main__":
    test_shared_multi_string_basic()
