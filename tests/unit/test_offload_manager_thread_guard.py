# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per-name thread-ownership guard on OffloadManager registry."""

import threading

import pytest

from flextensor.offload_manager import OFFLOAD_MANAGER_MAP, get_offload_manager


@pytest.fixture(autouse=True)
def _clean_manager_map():
    """Clear the global manager map before and after each test."""
    OFFLOAD_MANAGER_MAP.clear()
    yield
    OFFLOAD_MANAGER_MAP.clear()


class TestThreadOwnershipGuard:
    """Verify that each named manager is bound to the thread that created it."""

    def test_same_thread_access_works(self):
        """Repeated access from the creating thread succeeds."""
        om1 = get_offload_manager("test_same_thread")
        om2 = get_offload_manager("test_same_thread")
        assert om1 is om2

    def test_different_thread_raises_runtime_error(self):
        """Accessing a manager from a different thread raises RuntimeError."""
        get_offload_manager("test_cross_thread")

        error = None

        def access_from_other_thread():
            nonlocal error
            try:
                get_offload_manager("test_cross_thread")
            except RuntimeError as exc:
                error = exc

        t = threading.Thread(target=access_from_other_thread)
        t.start()
        t.join()

        assert error is not None
        assert "test_cross_thread" in str(error)
        assert "thread" in str(error).lower()

    def test_different_names_on_different_threads(self):
        """Two different names on two different threads both work fine."""
        get_offload_manager("thread_main_manager")

        result = {}

        def create_in_other_thread():
            try:
                om = get_offload_manager("thread_other_manager")
                result["manager"] = om
                result["error"] = None
            except RuntimeError as exc:
                result["error"] = exc

        t = threading.Thread(target=create_in_other_thread)
        t.start()
        t.join()

        assert result["error"] is None
        assert result["manager"] is not None

    def test_error_message_includes_thread_ids(self):
        """The RuntimeError message includes both thread IDs for debugging."""
        get_offload_manager("test_error_msg")
        owner_thread = threading.get_ident()

        captured = {}

        def access_from_other_thread():
            captured["accessor_thread"] = threading.get_ident()
            try:
                get_offload_manager("test_error_msg")
            except RuntimeError as exc:
                captured["error"] = str(exc)

        t = threading.Thread(target=access_from_other_thread)
        t.start()
        t.join()

        assert str(owner_thread) in captured["error"]
        assert str(captured["accessor_thread"]) in captured["error"]

    def test_concurrent_creation_same_name_one_wins(self):
        """When two threads race to create the same name, one succeeds and the other gets RuntimeError."""
        results = {"errors": [], "successes": []}
        barrier = threading.Barrier(2)

        def try_create(thread_results_key):
            barrier.wait()
            try:
                get_offload_manager("race_test")
                results["successes"].append(threading.get_ident())
            except RuntimeError:
                results["errors"].append(threading.get_ident())

        t1 = threading.Thread(target=try_create, args=("t1",))
        t2 = threading.Thread(target=try_create, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results["successes"]) == 1
        assert len(results["errors"]) == 1
