# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the flextensor.contrib.vllm._logging bridge."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import TYPE_CHECKING

import pytest

from flextensor._logging import (
    _BRIDGE_MARKER,
    _DEGRADED_NOTIFIED_MARKER,
    _DIAGNOSTICS_MARKER,
    _HELPER_HANDLER_MARKER,
    DIAGNOSTICS_LOGGER_NAME,
    ensure_diagnostics_visible,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def stub_vllm_logger() -> Callable[..., list[logging.Handler]]:
    """Install a stub `vllm.logger` module with a configurable init_logger.

    Returns a factory that the test can call to customise the stub's behaviour.
    The factory returns the handler(s) attached to the stub `vllm` logger so the
    test can assert routing.
    """
    handlers: list[logging.Handler] = []

    def factory(*, attach_on: str = "vllm.flextensor", n_handlers: int = 1) -> list[logging.Handler]:
        handlers.clear()
        stub_logger = logging.getLogger(attach_on)
        # Tidy pre-existing state on this logger
        stub_logger.handlers[:] = []
        stub_logger.setLevel(logging.INFO)
        for _ in range(n_handlers):
            h = logging.Handler()
            stub_logger.addHandler(h)
            handlers.append(h)

        def init_logger(name: str) -> logging.Logger:
            return logging.getLogger(name)

        stub_module = types.ModuleType("vllm.logger")
        stub_module.init_logger = init_logger  # type: ignore[attr-defined]
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.logger"] = stub_module
        return handlers

    yield factory

    # Teardown: clear stub modules + any stub-owned loggers
    for mod_name in ("vllm", "vllm.logger"):
        sys.modules.pop(mod_name, None)
    for name in ("vllm", "vllm.flextensor"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def _reset_flextensor_logger_state() -> None:
    ft = logging.getLogger("flextensor")
    diag = logging.getLogger(DIAGNOSTICS_LOGGER_NAME)
    before = (
        list(ft.handlers),
        ft.level,
        ft.propagate,
        getattr(ft, _BRIDGE_MARKER, None),
        list(diag.handlers),
        diag.level,
        diag.propagate,
        getattr(diag, _DIAGNOSTICS_MARKER, None),
        getattr(ft, _DEGRADED_NOTIFIED_MARKER, None),
    )
    ft.handlers[:] = []
    ft.setLevel(logging.NOTSET)
    ft.propagate = True
    if hasattr(ft, _BRIDGE_MARKER):
        delattr(ft, _BRIDGE_MARKER)
    if hasattr(ft, _DEGRADED_NOTIFIED_MARKER):
        delattr(ft, _DEGRADED_NOTIFIED_MARKER)
    diag.handlers[:] = []
    diag.setLevel(logging.NOTSET)
    diag.propagate = True
    if hasattr(diag, _DIAGNOSTICS_MARKER):
        delattr(diag, _DIAGNOSTICS_MARKER)

    yield

    ft.handlers[:] = before[0]
    ft.setLevel(before[1])
    ft.propagate = before[2]
    if before[3] is None:
        if hasattr(ft, _BRIDGE_MARKER):
            delattr(ft, _BRIDGE_MARKER)
    else:
        setattr(ft, _BRIDGE_MARKER, before[3])
    diag.handlers[:] = before[4]
    diag.setLevel(before[5])
    diag.propagate = before[6]
    if before[7] is None:
        if hasattr(diag, _DIAGNOSTICS_MARKER):
            delattr(diag, _DIAGNOSTICS_MARKER)
    else:
        setattr(diag, _DIAGNOSTICS_MARKER, before[7])
    if before[8] is None:
        if hasattr(ft, _DEGRADED_NOTIFIED_MARKER):
            delattr(ft, _DEGRADED_NOTIFIED_MARKER)
    else:
        setattr(ft, _DEGRADED_NOTIFIED_MARKER, before[8])


def _fresh_bridge_module():
    sys.modules.pop("flextensor.contrib.vllm._logging", None)
    return importlib.import_module("flextensor.contrib.vllm._logging")


def test_bridge_copies_handlers_to_flextensor_logger(stub_vllm_logger) -> None:
    [handler] = stub_vllm_logger()
    bridge = _fresh_bridge_module()

    bridge.install_flextensor_logging_bridge()

    ft = logging.getLogger("flextensor")
    assert handler in ft.handlers
    assert ft.propagate is False
    assert getattr(ft, _BRIDGE_MARKER, False) is True


def test_bridge_is_idempotent(stub_vllm_logger) -> None:
    stub_vllm_logger()
    bridge = _fresh_bridge_module()

    bridge.install_flextensor_logging_bridge()
    first = list(logging.getLogger("flextensor").handlers)
    bridge.install_flextensor_logging_bridge()
    second = list(logging.getLogger("flextensor").handlers)

    assert first == second


def test_bridge_walks_up_to_find_handlers(stub_vllm_logger) -> None:
    # Attach handler only on the root `vllm` logger, not `vllm.flextensor`.
    [handler] = stub_vllm_logger(attach_on="vllm", n_handlers=1)
    bridge = _fresh_bridge_module()

    bridge.install_flextensor_logging_bridge()

    assert handler in logging.getLogger("flextensor").handlers


def test_bridge_copies_all_vllm_handlers_when_multiple(stub_vllm_logger) -> None:
    handlers = stub_vllm_logger(n_handlers=3)
    bridge = _fresh_bridge_module()

    bridge.install_flextensor_logging_bridge()

    ft = logging.getLogger("flextensor")
    for h in handlers:
        assert h in ft.handlers


def test_bridge_evicts_prior_helper_handler(stub_vllm_logger) -> None:
    # Helper runs first (standalone path), then bridge.
    ensure_diagnostics_visible()
    ft = logging.getLogger("flextensor")
    assert len(ft.handlers) == 1
    assert getattr(ft.handlers[0], _HELPER_HANDLER_MARKER, False)

    [vllm_handler] = stub_vllm_logger()
    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()

    assert vllm_handler in ft.handlers
    assert all(not getattr(h, _HELPER_HANDLER_MARKER, False) for h in ft.handlers)


def test_helper_noop_when_bridge_already_installed(stub_vllm_logger) -> None:
    [vllm_handler] = stub_vllm_logger()
    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()

    ensure_diagnostics_visible()

    ft = logging.getLogger("flextensor")
    assert ft.handlers == [vllm_handler]
    assert logging.getLogger(DIAGNOSTICS_LOGGER_NAME).level == logging.INFO


def test_diagnostics_reaches_all_bridge_handlers(stub_vllm_logger) -> None:
    handlers = stub_vllm_logger(n_handlers=2)
    # Spy on emit so we can count invocations without relying on caplog propagation.
    received: list[logging.LogRecord] = []
    for h in handlers:
        h.emit = received.append  # type: ignore[method-assign]
    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()
    ensure_diagnostics_visible()

    logging.getLogger(DIAGNOSTICS_LOGGER_NAME).info("multi-handler-probe")

    messages = [r.getMessage() for r in received]
    assert messages.count("multi-handler-probe") == 2  # both handlers saw it exactly once


def test_bridge_import_requires_vllm() -> None:
    # NOTE: Assumes vllm is NOT installed in the test venv. If vllm becomes a
    # test dep, this test will pass for the wrong reason (real module imports).
    # No stub installed → bridge's lazy import of vllm.logger must fail cleanly.
    sys.modules.pop("vllm", None)
    sys.modules.pop("vllm.logger", None)
    bridge = _fresh_bridge_module()

    with pytest.raises(ModuleNotFoundError):
        bridge.install_flextensor_logging_bridge()


def test_bridge_degraded_mode_installs_fallback_when_no_ancestor_has_handlers(
    stub_vllm_logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard for issue 139: no-ancestor-with-handlers must not silently drop FT records.

    When the walk up ``vllm.flextensor → … → root`` finds no handlers, the
    bridge must install a fallback so FT diagnostics stay visible. It must
    leave ``propagate=True`` and must NOT stamp ``_BRIDGE_MARKER`` — a later
    call with real vLLM handlers should still be able to swap them in.
    """
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()

    ft = logging.getLogger("flextensor")
    assert len(ft.handlers) == 1
    [handler] = ft.handlers
    assert getattr(handler, _HELPER_HANDLER_MARKER, False) is True
    assert ft.propagate is True
    assert getattr(ft, _BRIDGE_MARKER, None) is None


def test_bridge_degraded_mode_is_idempotent(stub_vllm_logger, monkeypatch: pytest.MonkeyPatch) -> None:
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()
    first = list(logging.getLogger("flextensor").handlers)
    bridge.install_flextensor_logging_bridge()
    second = list(logging.getLogger("flextensor").handlers)

    assert first == second


def test_bridge_degraded_fallback_delivers_diagnostic_records(
    stub_vllm_logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under degraded mode, FT diagnostic records still reach a sink."""
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()
    ensure_diagnostics_visible()

    ft = logging.getLogger("flextensor")
    received: list[logging.LogRecord] = []
    ft.handlers[0].emit = received.append  # type: ignore[method-assign]

    logging.getLogger(DIAGNOSTICS_LOGGER_NAME).info("diag-probe-degraded")

    assert any(r.getMessage() == "diag-probe-degraded" for r in received)


def test_safe_install_returns_normally_when_bridge_succeeds(stub_vllm_logger) -> None:
    stub_vllm_logger()
    bridge = _fresh_bridge_module()

    bridge.safely_install_flextensor_logging_bridge()

    assert getattr(logging.getLogger("flextensor"), _BRIDGE_MARKER, False) is True


def test_safe_install_swallows_and_logs_when_bridge_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bridge failures must not propagate — registration in loader/worker is critical."""
    bridge = _fresh_bridge_module()

    def _boom() -> None:
        raise RuntimeError("simulated bridge failure")

    monkeypatch.setattr(bridge, "install_flextensor_logging_bridge", _boom)

    with caplog.at_level(logging.ERROR, logger="flextensor.contrib.vllm._logging"):
        bridge.safely_install_flextensor_logging_bridge()

    assert any("simulated bridge failure" in r.getMessage() or r.exc_info for r in caplog.records), (
        "safely_install_flextensor_logging_bridge did not log the bridge failure"
    )


def test_safe_install_writes_to_stderr_when_bridge_raises(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bridge failures must reach stderr even when the logger chain is broken.

    Regression guard for issue 139: when the bridge itself raises, the stdlib
    logger used to report the failure may have no reachable handler (the
    bridge was supposed to install them) and ``logging.lastResort`` can be
    disabled by the host process. stderr is the only sink guaranteed to be
    visible under any host logging configuration.
    """
    bridge = _fresh_bridge_module()

    def _boom() -> None:
        raise RuntimeError("simulated bridge failure XYZ")

    monkeypatch.setattr(bridge, "install_flextensor_logging_bridge", _boom)

    # Break every stdlib sink available to the bridge's error logger so only a
    # direct stderr write can deliver: clear all handler chains AND disable
    # `lastResort`, mimicking a host process that has nulled it out.
    for name in ("flextensor.contrib.vllm._logging", "flextensor.contrib.vllm", "flextensor", ""):
        logging.getLogger(name).handlers[:] = []
    monkeypatch.setattr(logging, "lastResort", None)

    bridge.safely_install_flextensor_logging_bridge()

    captured = capsys.readouterr()
    assert "FlexTensor" in captured.err
    assert "simulated bridge failure XYZ" in captured.err


def test_degraded_mode_emits_single_stderr_notification(
    stub_vllm_logger, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Degraded-mode notification must appear exactly once on stderr.

    The fallback ``StreamHandler`` (default target: stderr) and a
    ``logger.warning()`` call that propagates up to ``flextensor`` (which
    now has the fallback) would otherwise both emit the same message,
    producing duplicate operator-facing output under an already-rare
    condition.
    """
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()

    captured = capsys.readouterr()
    marker = "no ancestor with handlers"
    assert captured.err.count(marker) == 1, (
        f"expected exactly one degraded-mode notification on stderr, stderr was: {captured.err!r}"
    )

    # Fallback StreamHandler must still be installed so subsequent FT records reach a sink.
    ft = logging.getLogger("flextensor")
    assert len(ft.handlers) == 1
    assert getattr(ft.handlers[0], _HELPER_HANDLER_MARKER, False) is True


def test_degraded_mode_stderr_notice_emitted_once_per_process(
    stub_vllm_logger, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The degraded-mode stderr notice must not re-emit across repeated install calls.

    The module-level safe-install runs from ``worker.py``, ``loader.py``, and
    ``snapshot.py`` can all re-enter the degraded branch in a single worker
    process because ``_BRIDGE_MARKER`` is deliberately not stamped in degraded
    mode (so a later call can upgrade to real vLLM handlers). Without a
    separate "already notified" guard, the operator sees the same stderr line
    2-3 times per process.
    """
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()
    bridge.install_flextensor_logging_bridge()
    bridge.install_flextensor_logging_bridge()

    captured = capsys.readouterr()
    assert captured.err.count("no ancestor with handlers") == 1, (
        f"expected one stderr notice across three installs, got: {captured.err!r}"
    )


def test_degraded_mode_stderr_notice_survives_degraded_normal_degraded_cycle(
    stub_vllm_logger, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Marker must not be cleared by an intermediate successful bridge install.

    Pins the separation between ``_BRIDGE_MARKER`` (bridge installed) and
    ``_DEGRADED_NOTIFIED_MARKER`` (stderr notice already emitted). If a future
    refactor reuses one for the other, the third call (degraded again) would
    re-emit the notice.
    """
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()  # 1st: degraded, emits

    # Add real vLLM handlers and clear the stored degraded-mode fallback, then
    # reset _BRIDGE_MARKER so the bridge walks the handler chain again.
    real_handler = logging.Handler()
    logging.getLogger("vllm.flextensor").addHandler(real_handler)
    bridge.install_flextensor_logging_bridge()  # 2nd: upgrades to real handlers

    # Tear down the vLLM handlers and the bridge marker, then force degraded again.
    logging.getLogger("vllm.flextensor").handlers[:] = []
    ft = logging.getLogger("flextensor")
    delattr(ft, _BRIDGE_MARKER)
    ft.handlers[:] = []
    bridge.install_flextensor_logging_bridge()  # 3rd: degraded again, must NOT re-emit

    captured = capsys.readouterr()
    assert captured.err.count("no ancestor with handlers") == 1, (
        f"expected one stderr notice across degraded→normal→degraded, got: {captured.err!r}"
    )


def test_degraded_mode_writes_to_stderr_directly(
    stub_vllm_logger, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Degraded-mode notification must hit stderr directly, not only through the fallback handler.

    Belt-and-suspenders for issue 139: the ``logger.warning(...)`` emitted in
    degraded mode currently relies on the fallback StreamHandler that was
    just installed on ``flextensor``. A direct ``sys.stderr`` write guarantees
    the operator sees the signal even if the fallback install silently
    misbehaves or the propagation chain is disrupted.
    """
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    # Detach the fallback handler's potential sink BEFORE the bridge runs by
    # pre-populating the logger with a helper-marker-stamped handler that
    # drops records. The bridge's _install_fallback_handler is idempotent on
    # the marker, so no new handler is added — any stderr output must come
    # from a direct write, not from the StreamHandler.
    ft = logging.getLogger("flextensor")
    sentinel_handler = logging.NullHandler()
    setattr(sentinel_handler, _HELPER_HANDLER_MARKER, True)
    ft.addHandler(sentinel_handler)

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()

    # Premise: the sentinel suppressed a real fallback StreamHandler install.
    # If a future refactor drops the marker guard, `len(ft.handlers) > 1`
    # would silently turn this test into a tautology (the formatted record
    # would reach stderr via the StreamHandler anyway).
    assert len(ft.handlers) == 1, (
        "sentinel marker-guard did not suppress fallback install; test no longer isolates the direct write"
    )

    captured = capsys.readouterr()
    assert "FlexTensor" in captured.err, f"degraded-mode direct stderr signal missing; stderr was: {captured.err!r}"


def test_bridge_degraded_recovers_when_vllm_handlers_appear(stub_vllm_logger, monkeypatch: pytest.MonkeyPatch) -> None:
    """Degraded → normal: once vLLM handlers appear, re-invoking the bridge evicts the fallback."""
    stub_vllm_logger(n_handlers=0)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    bridge = _fresh_bridge_module()
    bridge.install_flextensor_logging_bridge()

    ft = logging.getLogger("flextensor")
    assert getattr(ft, _BRIDGE_MARKER, None) is None
    assert len(ft.handlers) == 1
    assert getattr(ft.handlers[0], _HELPER_HANDLER_MARKER, False)

    real_handler = logging.Handler()
    logging.getLogger("vllm.flextensor").addHandler(real_handler)

    bridge.install_flextensor_logging_bridge()

    assert real_handler in ft.handlers
    assert all(not getattr(h, _HELPER_HANDLER_MARKER, False) for h in ft.handlers)
    assert ft.propagate is False
    assert getattr(ft, _BRIDGE_MARKER, False) is True
