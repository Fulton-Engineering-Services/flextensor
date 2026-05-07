# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiprocess tests for ShmCoordinator creator/follower flow."""

from __future__ import annotations

import multiprocessing
import time
from typing import TYPE_CHECKING

import pytest

# pytest puts the test directory on sys.path when there is no __init__.py, so a
# top-level `from conftest import ...` resolves both at collection time and in
# spawn-launched worker processes (which inherit sys.path).
from conftest import EVENT_TIMEOUT, assert_clean_exit, drain_results, format_failed_results, wait_for_event

from flextensor.shm.coordinator import ShmCoordinator
from flextensor.state_handler import TensorManagerState

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType


def _make_minimal_state(loader_type: str = "allocation_block_transfer") -> TensorManagerState:
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map={1: "layer.weight", 2: "layer.bias"},
        allocation_ordered={0: ["layer.weight", "layer.bias"]},
        label_to_size_map={"layer": 1024},
        block_sizes={0: 2048},
        load_strategy={},
        release_strategy={},
        label_to_block_id={"layer.weight": 0, "layer.bias": 0},
        stats=[],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map={"layer.weight": "ft_test_w0", "layer.bias": "ft_test_w0"},
    )


def creator_process(
    namespace: str,
    result_queue: multiprocessing.Queue,
    ready_event: EventType,
    close_event: EventType,
) -> None:
    """Creator: write profile, notify, stay alive until ``close_event`` is set.

    Catches expected IPC errors and reports via the result queue. TimeoutError
    from event waits is *not* caught — it propagates to a non-zero exitcode so
    the parent's assert_clean_exit fires loudly instead of silently dropping
    the failure into a result-queue slot the parent might never drain.
    """
    try:
        coord = ShmCoordinator(namespace)
        state = _make_minimal_state()
        coord.write_profile(state)
        coord.notify_ready()

        ready_event.set()

        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to allow coordinator close")
        coord.close()
        result_queue.put({"success": True, "is_creator": True})
    except (FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "is_creator": True})


def follower_process(namespace: str, result_queue: multiprocessing.Queue) -> None:
    """Follower: wait for ready, read profile, verify data matches."""
    try:
        coord = ShmCoordinator(namespace)
        assert coord.is_creator is False, "Expected follower role"

        coord.wait_for_ready()
        state = coord.read_profile()

        coord.close()
        result_queue.put({
            "success": True,
            "is_creator": False,
            "loader_type": state.loader_type,
            "tensor_count": len(state.tensor_id_to_name_map),
            "block_count": len(state.allocation_ordered),
        })
    except (FileNotFoundError, FileExistsError, OSError) as e:
        result_queue.put({"success": False, "error": f"{type(e).__name__}: {e}", "is_creator": False})


def short_lived_creator_process(ns: str, rq: multiprocessing.Queue, ready_event, close_event, closed_event) -> None:
    """Creator that exits after the checker has connected."""
    try:
        coord = ShmCoordinator(ns)
        coord.write_profile(_make_minimal_state())
        coord.notify_ready()
        ready_event.set()
        if not close_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for parent to release short-lived creator")
        coord.close()
        rq.put({"success": True, "role": "creator"})
        closed_event.set()
    except (FileNotFoundError, FileExistsError, OSError) as e:
        rq.put({"success": False, "error": f"{type(e).__name__}: {e}", "role": "creator"})
        closed_event.set()


def liveness_checker_process(
    ns: str,
    rq: multiprocessing.Queue,
    connected_event: EventType,
    creator_closed_event: EventType,
) -> None:
    """Connect, signal ``connected_event``, wait on ``creator_closed_event``, then check liveness.

    Two-event handshake: ``connected_event`` lets the parent release the creator
    only after this process has attached, and ``creator_closed_event`` ensures
    ``is_creator_alive()`` is called *after* the creator has exited.
    """
    try:
        coord = ShmCoordinator(ns)
        coord.wait_for_ready()
        connected_event.set()
        if not creator_closed_event.wait(timeout=EVENT_TIMEOUT):
            raise TimeoutError("Timed out waiting for creator to close")
        alive = coord.is_creator_alive()
        coord.close()
        rq.put({"success": True, "role": "checker", "creator_alive": alive})
    except (FileNotFoundError, FileExistsError, OSError) as e:
        rq.put({"success": False, "error": f"{type(e).__name__}: {e}", "role": "checker"})


class TestShmCoordinatorMultiprocess:
    """Multiprocess tests for ShmCoordinator."""

    @pytest.fixture
    def namespace(self):
        return f"ft_mp_{int(time.time() * 1000) % 100000}"

    def test_creator_follower_flow(self, namespace):
        """Full creator/follower flow across two OS processes."""
        result_queue = multiprocessing.Queue()
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=creator_process,
            args=(namespace, result_queue, creator_ready, creator_close),
        )
        follower_proc = multiprocessing.Process(target=follower_process, args=(namespace, result_queue))

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "coordinator creator startup", proc=creator_proc)
            follower_proc.start()

            follower_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(follower_proc, "follower")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=2)

            for result in results:
                assert result["success"], f"Process failed: {result.get('error', 'unknown')}"

            follower_result = next(r for r in results if r["is_creator"] is False)
            assert follower_result["loader_type"] == "allocation_block_transfer"
            assert follower_result["tensor_count"] == 2
            assert follower_result["block_count"] == 1
        finally:
            creator_close.set()
            for proc in [creator_proc, follower_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    def test_multiple_followers(self, namespace):
        """One creator, multiple followers all read the same profile."""
        result_queue = multiprocessing.Queue()
        num_followers = 3
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=creator_process,
            args=(namespace, result_queue, creator_ready, creator_close),
        )
        follower_procs = [
            multiprocessing.Process(target=follower_process, args=(namespace, result_queue))
            for _ in range(num_followers)
        ]

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "coordinator creator startup", proc=creator_proc)

            for proc in follower_procs:
                proc.start()

            for i, proc in enumerate(follower_procs):
                proc.join(timeout=EVENT_TIMEOUT)
                assert_clean_exit(proc, f"follower {i}")
            creator_close.set()
            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "creator")

            results = drain_results(result_queue, expected_count=1 + num_followers)

            for result in results:
                assert result["success"], f"Process failed: {result.get('error', 'unknown')}"

            follower_results = [r for r in results if r["is_creator"] is False]
            assert len(follower_results) == num_followers
            for fr in follower_results:
                assert fr["loader_type"] == "allocation_block_transfer"
                assert fr["tensor_count"] == 2
        finally:
            creator_close.set()
            for proc in [creator_proc, *follower_procs]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)

    def test_is_creator_alive_returns_bool_after_close(self, namespace):
        """Smoke test: is_creator_alive() returns a bool after creator close, no crash.

        This test does NOT exercise stale-timeout detection — the default
        keep-alive window is much longer than this test's wall time, so
        ``creator_alive`` is typically True when the follower checks it.
        Until ``ShmCoordinator`` exposes a ``keep_alive_seconds`` knob, this
        test cannot exercise real stale-detection (where the API returns False
        because the heartbeat has gone stale); it only guards that the
        coordinator survives a creator exit and the follower's API call returns
        a bool without raising.
        """
        result_queue = multiprocessing.Queue()
        creator_ready = multiprocessing.Event()
        creator_close = multiprocessing.Event()
        creator_closed = multiprocessing.Event()
        checker_connected = multiprocessing.Event()

        creator_proc = multiprocessing.Process(
            target=short_lived_creator_process,
            args=(namespace, result_queue, creator_ready, creator_close, creator_closed),
        )
        checker_proc = multiprocessing.Process(
            target=liveness_checker_process,
            args=(namespace, result_queue, checker_connected, creator_closed),
        )

        try:
            creator_proc.start()
            wait_for_event(creator_ready, "short-lived coordinator creator startup", proc=creator_proc)
            checker_proc.start()
            wait_for_event(checker_connected, "coordinator liveness checker connection", proc=checker_proc)
            creator_close.set()

            creator_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(creator_proc, "short-lived creator")
            checker_proc.join(timeout=EVENT_TIMEOUT)
            assert_clean_exit(checker_proc, "liveness checker")

            results = drain_results(result_queue, expected_count=2)
            assert all(r["success"] for r in results), f"Failed: {format_failed_results(results)}"

            checker_result = next(r for r in results if r.get("role") == "checker")
            assert isinstance(checker_result["creator_alive"], bool)
        finally:
            creator_close.set()
            for proc in [creator_proc, checker_proc]:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=EVENT_TIMEOUT)
