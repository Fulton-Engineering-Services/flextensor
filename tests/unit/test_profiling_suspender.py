# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`flextensor.helpers.ProfilingSuspender`.

The suspender owns the refcount invariant behind ``TensorManager``'s
``suspend_profiling`` / ``resume_profiling`` API.  These tests exercise the
class in isolation; integration with ``TensorManager`` is covered by
``test_profiling_control.py``.
"""

import pytest

from flextensor.helpers import ProfilingSuspender


class TestProfilingSuspender:
    def test_initial_state_is_not_suspended(self):
        s = ProfilingSuspender()
        assert not s.is_suspended()

    def test_suspend_then_resume_round_trip(self):
        s = ProfilingSuspender()
        s.suspend()
        assert s.is_suspended()
        s.resume()
        assert not s.is_suspended()

    def test_nested_suspend_requires_equal_number_of_resumes(self):
        s = ProfilingSuspender()
        s.suspend()
        s.suspend()
        s.suspend()
        assert s.is_suspended()
        s.resume()
        assert s.is_suspended()
        s.resume()
        assert s.is_suspended()
        s.resume()
        assert not s.is_suspended()

    def test_resume_without_suspend_raises(self):
        s = ProfilingSuspender()
        with pytest.raises(RuntimeError, match="unbalanced"):
            s.resume()
        assert not s.is_suspended()

    def test_resume_past_zero_raises(self):
        s = ProfilingSuspender()
        s.suspend()
        s.resume()
        with pytest.raises(RuntimeError, match="unbalanced"):
            s.resume()
        assert not s.is_suspended()

    def test_context_manager_suspends_inside_and_releases_outside(self):
        s = ProfilingSuspender()
        with s.suspended() as x:
            assert x is None, (
                "suspended() must yield None — beartype + @contextmanager "
                "force a `-> Any` annotation that loses static enforcement, "
                "so the contract has to be pinned at runtime by tests."
            )
            assert s.is_suspended()
        assert not s.is_suspended()

    def test_context_manager_releases_on_exception(self):
        s = ProfilingSuspender()
        with pytest.raises(RuntimeError), s.suspended():
            assert s.is_suspended()
            raise RuntimeError("boom")
        assert not s.is_suspended()

    def test_context_manager_nests_with_raw_calls(self):
        s = ProfilingSuspender()
        s.suspend()
        with s.suspended():
            assert s.is_suspended()
        assert s.is_suspended(), "outer raw suspension must still be active"
        s.resume()
        assert not s.is_suspended()

    def test_context_managers_nest_with_each_other(self):
        s = ProfilingSuspender()
        with s.suspended():
            with s.suspended():
                assert s.is_suspended()
            assert s.is_suspended(), "outer context must survive inner exit"
        assert not s.is_suspended()

    def test_instances_are_independent(self):
        """Two suspenders keep separate counters (no shared module-level state)."""
        a = ProfilingSuspender()
        b = ProfilingSuspender()
        a.suspend()
        assert a.is_suspended()
        assert not b.is_suspended()
