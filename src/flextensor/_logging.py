# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlexTensor logging utilities.

Defines the ``flextensor.diagnostics`` logger and sentinel markers that
coordinate with optional external handler bridges (e.g. the one in
``flextensor.contrib.vllm``), so the diagnostic tables remain visible
regardless of the host application's root/package log level.
"""

from __future__ import annotations

import logging

DIAGNOSTICS_LOGGER_NAME = "flextensor.diagnostics"

_BRIDGE_MARKER = "_flextensor_bridge_installed"
_DIAGNOSTICS_MARKER = "_flextensor_diagnostics_handler_installed"
_HELPER_HANDLER_MARKER = "_flextensor_helper_installed"
# Separate from _BRIDGE_MARKER: degraded mode deliberately does not stamp
# _BRIDGE_MARKER so a later call can upgrade to real vLLM handlers.
_DEGRADED_NOTIFIED_MARKER = "_flextensor_degraded_notified"


def get_diagnostics_logger() -> logging.Logger:
    """Return the dedicated logger for FlexTensor diagnostic tables."""
    return logging.getLogger(DIAGNOSTICS_LOGGER_NAME)


def ensure_diagnostics_visible() -> None:
    """Configure ``flextensor.diagnostics`` so its INFO records are visible.

    Forces the diagnostics logger's level to ``INFO`` so its records aren't
    dropped if an ancestor has been raised above INFO. When no external
    bridge has marked ``flextensor`` and it has no handlers, attaches a
    sentinel-marked fallback ``StreamHandler`` to the ``flextensor`` logger
    (not ``flextensor.diagnostics``) that a later bridge call can evict.
    Idempotent.
    """
    diag = get_diagnostics_logger()
    if getattr(diag, _DIAGNOSTICS_MARKER, False):
        return
    diag.setLevel(logging.INFO)
    diag.propagate = True

    ft = logging.getLogger("flextensor")
    if not getattr(ft, _BRIDGE_MARKER, False) and not ft.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        setattr(handler, _HELPER_HANDLER_MARKER, True)
        ft.addHandler(handler)

    setattr(diag, _DIAGNOSTICS_MARKER, True)


__all__ = [
    "DIAGNOSTICS_LOGGER_NAME",
    "ensure_diagnostics_visible",
    "get_diagnostics_logger",
]
