# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM logging bridge -- copies vLLM's handlers onto the ``flextensor`` logger."""

from __future__ import annotations

import logging
import sys
import traceback

from flextensor._logging import _BRIDGE_MARKER, _DEGRADED_NOTIFIED_MARKER, _HELPER_HANDLER_MARKER

_FALLBACK_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_STDERR_PREFIX = "FlexTensor: "


def _install_fallback_handler(ft: logging.Logger) -> None:
    """Attach a helper-marked StreamHandler on ``ft`` if one isn't already present.

    Used by the degraded-mode path to ensure FlexTensor diagnostic records still
    reach a sink when no vLLM ancestor carries handlers.
    """
    if any(getattr(h, _HELPER_HANDLER_MARKER, False) for h in ft.handlers):
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_FALLBACK_FORMAT))
    setattr(handler, _HELPER_HANDLER_MARKER, True)
    ft.addHandler(handler)
    if ft.level == logging.NOTSET or ft.level > logging.INFO:
        ft.setLevel(logging.INFO)


def install_flextensor_logging_bridge() -> None:
    """Route ``flextensor.*`` logs through vLLM's configured handlers.

    Idempotent. If a prior diagnostics-helper handler is present and vLLM
    handlers are found, the helper is evicted. Degraded mode: if no
    ``vllm.flextensor`` ancestor has handlers, installs a helper-marked
    fallback ``StreamHandler`` on ``flextensor`` and does NOT stamp
    ``_BRIDGE_MARKER``, so a subsequent call can upgrade to real sinks.

    May raise on a broken vLLM install (missing ``vllm.logger``, partial
    import, etc.). Module-level callers should use
    :func:`safely_install_flextensor_logging_bridge`.
    """
    ft = logging.getLogger("flextensor")
    if getattr(ft, _BRIDGE_MARKER, False):
        return

    from vllm.logger import init_logger

    source = init_logger("vllm.flextensor")

    # Walk up to find the ancestor where vLLM installed handlers.
    cursor: logging.Logger | None = source
    handlers: list[logging.Handler] = []
    while cursor is not None:
        if cursor.handlers:
            handlers = cursor.handlers
            break
        cursor = cursor.parent

    if not handlers:
        _install_fallback_handler(ft)
        # Bypass the logger — it would propagate through `flextensor` and
        # duplicate via the StreamHandler just installed above.
        if not getattr(ft, _DEGRADED_NOTIFIED_MARKER, False):
            sys.stderr.write(
                f"{_STDERR_PREFIX}vLLM logging bridge found no ancestor with handlers; installed a "
                f"fallback StreamHandler so FlexTensor diagnostic tables remain visible.\n"
            )
            setattr(ft, _DEGRADED_NOTIFIED_MARKER, True)
        return

    # Evict any helper/fallback handler; vLLM's handlers are taking over.
    ft.handlers[:] = [h for h in ft.handlers if not getattr(h, _HELPER_HANDLER_MARKER, False)]

    for h in handlers:
        if h not in ft.handlers:
            ft.addHandler(h)
    ft.setLevel(source.getEffectiveLevel())
    ft.propagate = False
    setattr(ft, _BRIDGE_MARKER, True)


def safely_install_flextensor_logging_bridge() -> None:
    """Call :func:`install_flextensor_logging_bridge`, swallowing any exception.

    For module-level use in ``loader.py`` / ``worker.py`` / ``snapshot.py`` —
    diagnostic logging must never block class registration. Failures are
    written to ``sys.stderr`` and logged.
    """
    try:
        install_flextensor_logging_bridge()
    except Exception:
        # stderr is the only sink guaranteed when the stdlib handler chain
        # (which this bridge was meant to populate) is unreachable.
        msg = "FlexTensor logging bridge install failed; diagnostic tables may be invisible."
        sys.stderr.write(f"{_STDERR_PREFIX}{msg}\n")
        traceback.print_exc(file=sys.stderr)
        logging.getLogger(__name__).exception("%s Continuing without the bridge.", msg)


__all__ = [
    "install_flextensor_logging_bridge",
    "safely_install_flextensor_logging_bridge",
]
