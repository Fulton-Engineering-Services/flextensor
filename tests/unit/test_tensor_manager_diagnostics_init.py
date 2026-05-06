# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests that TensorManager.__init__ wires the diagnostics visibility helper."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import torch

from flextensor._logging import _DIAGNOSTICS_MARKER, DIAGNOSTICS_LOGGER_NAME
from flextensor.strategy import GreedyStrategy
from flextensor.tensor_manager import TensorManager


@pytest.fixture(autouse=True)
def _reset_diag_marker() -> None:
    diag = logging.getLogger(DIAGNOSTICS_LOGGER_NAME)
    had_marker = getattr(diag, _DIAGNOSTICS_MARKER, None)
    level = diag.level
    handlers = list(diag.handlers)
    if hasattr(diag, _DIAGNOSTICS_MARKER):
        delattr(diag, _DIAGNOSTICS_MARKER)
    diag.setLevel(logging.NOTSET)
    diag.handlers[:] = []

    yield

    diag.handlers[:] = handlers
    diag.setLevel(level)
    if had_marker is not None:
        setattr(diag, _DIAGNOSTICS_MARKER, had_marker)


@pytest.fixture(autouse=True)
def _fake_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend CUDA is available so ``TensorManager(pinned_memory=True)``
    construction doesn't raise on CPU-only CI hosts. These tests only
    exercise the diagnostics-logging wiring; they don't need real CUDA.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_tensor_manager_calls_ensure_diagnostics_visible_when_enabled() -> None:
    device = torch.device("cpu")
    strategy = GreedyStrategy()

    with patch("flextensor.tensor_manager.ensure_diagnostics_visible") as mock_helper:
        TensorManager(device_gpu=device, tensor_manager_load_strategy=strategy, enable_diagnostics=True)

    mock_helper.assert_called_once()


def test_tensor_manager_does_not_configure_logging_when_diagnostics_disabled() -> None:
    device = torch.device("cpu")
    strategy = GreedyStrategy()

    with patch("flextensor.tensor_manager.ensure_diagnostics_visible") as mock_helper:
        TensorManager(device_gpu=device, tensor_manager_load_strategy=strategy, enable_diagnostics=False)

    mock_helper.assert_not_called()
