# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for flextensor._logging: diagnostics logger + ensure_diagnostics_visible."""

from __future__ import annotations

import logging

import pytest

from flextensor._logging import (
    _BRIDGE_MARKER,
    _DIAGNOSTICS_MARKER,
    _HELPER_HANDLER_MARKER,
    DIAGNOSTICS_LOGGER_NAME,
    ensure_diagnostics_visible,
    get_diagnostics_logger,
)


@pytest.fixture(autouse=True)
def _reset_logger_state() -> None:
    """Reset flextensor and flextensor.diagnostics logger state around each test.

    Snapshots existing state for restoration on teardown, then forces a clean
    slate (no handlers, NOTSET level, propagate=True, no markers) before
    yielding. The clean-slate step matters because Python's ``logging``
    registry is process-global: prior test modules can install handlers or
    raise the level on the ``flextensor`` logger (e.g. via the vLLM logging
    bridge) and those side effects would otherwise leak into preconditions
    here like ``assert not ft.handlers``.
    """
    ft = logging.getLogger("flextensor")
    diag = logging.getLogger(DIAGNOSTICS_LOGGER_NAME)

    ft_before = (list(ft.handlers), ft.level, ft.propagate, getattr(ft, _BRIDGE_MARKER, None))
    diag_before = (list(diag.handlers), diag.level, diag.propagate, getattr(diag, _DIAGNOSTICS_MARKER, None))

    ft.handlers[:] = []
    ft.setLevel(logging.NOTSET)
    ft.propagate = True
    if hasattr(ft, _BRIDGE_MARKER):
        delattr(ft, _BRIDGE_MARKER)

    diag.handlers[:] = []
    diag.setLevel(logging.NOTSET)
    diag.propagate = True
    if hasattr(diag, _DIAGNOSTICS_MARKER):
        delattr(diag, _DIAGNOSTICS_MARKER)

    yield

    ft.handlers[:] = ft_before[0]
    ft.setLevel(ft_before[1])
    ft.propagate = ft_before[2]
    if ft_before[3] is None:
        if hasattr(ft, _BRIDGE_MARKER):
            delattr(ft, _BRIDGE_MARKER)
    else:
        setattr(ft, _BRIDGE_MARKER, ft_before[3])

    diag.handlers[:] = diag_before[0]
    diag.setLevel(diag_before[1])
    diag.propagate = diag_before[2]
    if diag_before[3] is None:
        if hasattr(diag, _DIAGNOSTICS_MARKER):
            delattr(diag, _DIAGNOSTICS_MARKER)
    else:
        setattr(diag, _DIAGNOSTICS_MARKER, diag_before[3])


def test_diagnostics_logger_name() -> None:
    assert DIAGNOSTICS_LOGGER_NAME == "flextensor.diagnostics"
    assert get_diagnostics_logger() is logging.getLogger(DIAGNOSTICS_LOGGER_NAME)


def test_ensure_diagnostics_visible_sets_info_level_explicitly() -> None:
    diag = get_diagnostics_logger()
    ensure_diagnostics_visible()
    assert diag.level == logging.INFO
    assert diag.propagate is True


def test_ensure_diagnostics_visible_installs_fallback_handler_when_standalone() -> None:
    ft = logging.getLogger("flextensor")
    assert not ft.handlers  # precondition: clean state from fixture
    ensure_diagnostics_visible()
    assert len(ft.handlers) == 1
    [handler] = ft.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == logging.INFO
    assert getattr(handler, _HELPER_HANDLER_MARKER, False) is True


def test_ensure_diagnostics_visible_noop_when_bridge_active() -> None:
    ft = logging.getLogger("flextensor")
    setattr(ft, _BRIDGE_MARKER, True)
    # Put a pretend bridge handler on flextensor.
    bridge_handler = logging.NullHandler()
    ft.addHandler(bridge_handler)

    ensure_diagnostics_visible()

    # Helper must not add a fallback handler.
    assert ft.handlers == [bridge_handler]
    # But diagnostics level is still set.
    assert get_diagnostics_logger().level == logging.INFO


def test_ensure_diagnostics_visible_respects_user_handlers() -> None:
    ft = logging.getLogger("flextensor")
    user_handler = logging.NullHandler()
    ft.addHandler(user_handler)

    ensure_diagnostics_visible()

    # User's handler is untouched; no fallback is appended.
    assert ft.handlers == [user_handler]


def test_ensure_diagnostics_visible_idempotent() -> None:
    ensure_diagnostics_visible()
    ft = logging.getLogger("flextensor")
    handlers_after_first = list(ft.handlers)

    ensure_diagnostics_visible()

    assert ft.handlers == handlers_after_first  # no duplicate fallback handler


def test_diagnostics_visible_despite_flextensor_warning_level(caplog: pytest.LogCaptureFixture) -> None:
    ft = logging.getLogger("flextensor")
    ft.setLevel(logging.WARNING)

    ensure_diagnostics_visible()

    # caplog captures at the WARNING level by default; raise to INFO just for the test scope.
    with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER_NAME):
        get_diagnostics_logger().info("diagnostic-probe")

    messages = [r.getMessage() for r in caplog.records if r.name == DIAGNOSTICS_LOGGER_NAME]
    assert "diagnostic-probe" in messages


def test_diagnostics_visible_despite_root_warning_level(caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger().setLevel(logging.WARNING)

    ensure_diagnostics_visible()

    with caplog.at_level(logging.INFO, logger=DIAGNOSTICS_LOGGER_NAME):
        get_diagnostics_logger().info("diagnostic-probe-root")

    messages = [r.getMessage() for r in caplog.records if r.name == DIAGNOSTICS_LOGGER_NAME]
    assert "diagnostic-probe-root" in messages


def test_generic_flextensor_info_still_filtered_when_user_raised_flextensor_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ft = logging.getLogger("flextensor")
    ft.setLevel(logging.WARNING)

    ensure_diagnostics_visible()

    # Capture at the root (where records land after propagation). Crucially we do
    # NOT target `logger="flextensor"` — that would bump the flextensor logger
    # level to INFO for the with-block's duration, defeating the premise of the
    # test (user set flextensor=WARNING). Root-level capture leaves the
    # originating-logger level checks intact.
    with caplog.at_level(logging.INFO):
        logging.getLogger("flextensor.tensor_manager").info("should-be-filtered")
        get_diagnostics_logger().info("should-pass")

    messages = {r.getMessage() for r in caplog.records if r.name.startswith("flextensor")}
    assert "should-pass" in messages
    assert "should-be-filtered" not in messages
