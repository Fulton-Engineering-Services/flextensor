# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP lifecycle for the LTX outpaint replica leader."""

from __future__ import annotations

import contextlib
import json
import logging
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Lock, Thread
from typing import Any, NoReturn, Protocol

from context_parallel import DistributedRequestError, ReplicaFatalError, is_replica_fatal_error

LOGGER = logging.getLogger(__name__)


class GenerationService(Protocol):
    request_id: int

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ReplicaRuntime(Protocol):
    poisoned: bool
    poison_reason: str | None

    def mark_poisoned(self, reason: str) -> None: ...

    def terminate(self, reason: str | None = None, *, exit_code: int = 1) -> NoReturn: ...


class ShutdownCallback(Protocol):
    def __call__(self) -> None: ...


class FollowerShutdownRuntime(Protocol):
    control_timeout_seconds: int
    poisoned: bool

    def broadcast_object(self, value: Any = None, *, timeout_seconds: float | None = None) -> Any: ...

    def terminate(self, reason: str | None = None, *, exit_code: int = 1) -> NoReturn: ...


def _safe_error_text(error: BaseException) -> str:
    with contextlib.suppress(Exception):
        return str(error)
    return type(error).__name__


class ShutdownSignalHandlers:
    """Latch SIGINT/SIGTERM and dispatch shutdown outside signal context."""

    def __init__(self, request_shutdown: ShutdownCallback) -> None:
        self.request_shutdown = request_shutdown
        self.received_signal: int | None = None
        self._previous_handlers: dict[signal.Signals, Any] = {}
        self._dispatcher_stop = Event()
        self._dispatcher = Thread(
            target=self._dispatch_shutdown,
            name="ltx-replica-signal-dispatcher",
            daemon=True,
        )

    def __enter__(self) -> ShutdownSignalHandlers:
        try:
            for shutdown_signal in (signal.SIGTERM, signal.SIGINT):
                self._previous_handlers[shutdown_signal] = signal.getsignal(shutdown_signal)
                signal.signal(shutdown_signal, self._handle)
            self._dispatcher.start()
        except Exception:
            self._restore()
            self._dispatcher_stop.set()
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        self._restore()
        self._dispatcher_stop.set()
        self._dispatcher.join(timeout=1)

    def _handle(self, signum: int, _frame: Any) -> None:
        # Python signal handlers can be re-entered between bytecodes. Keep this
        # path to a single idempotent assignment: Event.set(), Thread.start(),
        # logging, and distributed calls can all acquire non-reentrant locks.
        if self.received_signal is None:
            self.received_signal = signum

    def _dispatch_shutdown(self) -> None:
        while not self._dispatcher_stop.wait(timeout=0.05):
            if self.received_signal is not None:
                self.request_shutdown()
                return

    def _restore(self) -> None:
        for shutdown_signal, previous_handler in self._previous_handlers.items():
            signal.signal(shutdown_signal, previous_handler)
        self._previous_handlers.clear()


def should_broadcast_follower_shutdown(
    *,
    http_loop_stopped: bool,
    runtime_poisoned: bool,
    request_in_flight: bool,
) -> bool:
    """Return whether a final follower command preserves collective ordering."""
    return http_loop_stopped and not runtime_poisoned and not request_in_flight


def broadcast_follower_shutdown(
    runtime: FollowerShutdownRuntime,
    *,
    logger: logging.Logger = LOGGER,
    timeout_seconds: float | None = None,
) -> None:
    """Send the final command, hard-exiting if a missing follower cannot receive it."""
    timeout = runtime.control_timeout_seconds if timeout_seconds is None else timeout_seconds
    logger.debug("Broadcasting follower shutdown with a %gs control-plane timeout", timeout)
    runtime.broadcast_object({"operation": "shutdown"}, timeout_seconds=timeout)


def finalize_follower_shutdown(
    runtime: FollowerShutdownRuntime,
    *,
    http_loop_stopped: bool,
    request_in_flight: bool,
    logger: logging.Logger = LOGGER,
) -> None:
    """Notify live followers only after local request collectives have stopped."""
    if should_broadcast_follower_shutdown(
        http_loop_stopped=http_loop_stopped,
        runtime_poisoned=runtime.poisoned,
        request_in_flight=request_in_flight,
    ):
        broadcast_follower_shutdown(runtime, logger=logger)
        return
    runtime.terminate(
        "Follower shutdown broadcast was unsafe because the HTTP loop or request did not stop cleanly",
        exit_code=1,
    )


class ReplicaRequestServer:
    """Serve leader HTTP traffic and stop permanently after a fatal rank error."""

    def __init__(
        self,
        host: str,
        port: int,
        service: GenerationService,
        runtime: ReplicaRuntime,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.service = service
        self.runtime = runtime
        self.logger = logger
        self._shutdown_started = Event()
        self._shutdown_lock = Lock()
        self._request_in_flight = Event()
        self._server = HTTPServer((host, port), self._build_handler())

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def request_in_flight(self) -> bool:
        return self._request_in_flight.is_set()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:  # noqa: C901 - protocol routing stays explicit.
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, result: dict[str, Any]) -> None:
                body = json.dumps(result, indent=2).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler requires this method name.
                if self.path != "/healthz":
                    self.send_error(404, "Not Found")
                    return

                healthy = not owner.runtime.poisoned
                result: dict[str, Any] = {"ok": healthy}
                if not healthy:
                    result["reason"] = owner.runtime.poison_reason or "Replica is unavailable"
                self._send_json(200 if healthy else 503, result)

            def do_POST(self) -> None:  # noqa: C901, N802 - explicit BaseHTTPRequestHandler routing.
                if self.path != "/":
                    self.send_error(404, "Not Found")
                    return

                fatal_error: BaseException | None = None
                try:
                    if owner.runtime.poisoned:
                        raise ReplicaFatalError(
                            f"Replica is already poisoned: {owner.runtime.poison_reason or 'unknown reason'}"
                        )
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) if length > 0 else b"{}")
                    if not isinstance(payload, dict):
                        raise ValueError(f"JSON request body must be an object; got {type(payload).__name__}.")
                    result = owner.generate_request(payload)
                    if owner.runtime.poisoned:
                        raise ReplicaFatalError(
                            f"Request returned after poisoning the replica: "
                            f"{owner.runtime.poison_reason or 'unknown reason'}"
                        )
                    status = 200
                except (json.JSONDecodeError, ValueError) as exc:
                    if owner.should_terminate(exc):
                        fatal_error = exc
                        result, status = owner.record_fatal_error(exc)
                    else:
                        result = {"ok": False, "error": f"Invalid request: {exc}"}
                        status = 400
                except DistributedRequestError as exc:
                    if owner.should_terminate(exc):
                        fatal_error = exc
                        result, status = owner.record_fatal_error(exc)
                    else:
                        # A completed outcome exchange makes this request failure safe
                        # to report while the replica continues serving.
                        owner.logger.error("%s", exc)
                        result = {"ok": False, "error": _safe_error_text(exc)}
                        status = 500
                except ReplicaFatalError as exc:
                    fatal_error = exc
                    result, status = owner.record_fatal_error(exc)
                except Exception as exc:
                    if owner.should_terminate(exc):
                        fatal_error = exc
                        result, status = owner.record_fatal_error(exc)
                    else:
                        owner.logger.exception("Recoverable request %s failed", owner.service.request_id)
                        result = {"ok": False, "error": _safe_error_text(exc)}
                        status = 500

                try:
                    self._send_json(status, result)
                finally:
                    if fatal_error is not None:
                        with contextlib.suppress(Exception):
                            self.wfile.flush()
                        owner.request_shutdown()

        return Handler

    def generate_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._request_in_flight.set()
        try:
            return self.service.generate(payload)
        finally:
            self._request_in_flight.clear()

    def should_terminate(self, error: BaseException) -> bool:
        return self.runtime.poisoned or is_replica_fatal_error(error)

    def record_fatal_error(self, error: BaseException) -> tuple[dict[str, Any], int]:
        request_id = self.service.request_id
        reason = f"Fatal request {request_id}: {type(error).__name__}: {_safe_error_text(error)}"
        if not self.runtime.poisoned:
            self.runtime.mark_poisoned(reason)
        self.logger.error(
            "Replica became unhealthy during request %s; shutting down",
            request_id,
            exc_info=(type(error), error, error.__traceback__),
        )
        return {"ok": False, "error": "Replica is unavailable", "request_id": request_id}, 503

    def request_shutdown(self) -> None:
        """Ask ``serve_forever`` to stop without deadlocking its handler thread."""
        with self._shutdown_lock:
            if self._shutdown_started.is_set():
                return
            self._shutdown_started.set()
        shutdown_thread = Thread(
            target=self._shutdown_server,
            name="ltx-replica-http-shutdown",
            daemon=True,
        )
        try:
            shutdown_thread.start()
        except Exception as exc:
            self.logger.critical("Could not start replica shutdown thread", exc_info=True)
            self.runtime.terminate(f"HTTP shutdown failed: {_safe_error_text(exc)}", exit_code=1)

    def _shutdown_server(self) -> None:
        try:
            self._server.shutdown()
        except Exception as exc:
            self.logger.critical("Replica HTTP shutdown failed", exc_info=True)
            self.runtime.terminate(f"HTTP shutdown failed: {_safe_error_text(exc)}", exit_code=1)

    def serve_forever(self) -> None:
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
        if self.runtime.poisoned:
            self.runtime.terminate(self.runtime.poison_reason or "Replica became unhealthy", exit_code=1)


class ReplicaShutdownCoordinator:
    """Carry an early shutdown request forward until the HTTP server exists."""

    def __init__(self) -> None:
        self._requested = Event()
        self._server: ReplicaRequestServer | None = None

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def request_shutdown(self) -> None:
        self._requested.set()
        server = self._server
        if server is not None:
            server.request_shutdown()

    def attach_server(self, server: ReplicaRequestServer) -> None:
        if self._server is not None:
            raise RuntimeError("A replica HTTP server is already attached")
        self._server = server
        if self._requested.is_set():
            server.request_shutdown()
