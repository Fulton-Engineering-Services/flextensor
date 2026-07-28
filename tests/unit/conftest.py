# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for unit tests."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from flextensor._logging import (
    _BRIDGE_MARKER,
    _DEGRADED_NOTIFIED_MARKER,
    _DIAGNOSTICS_MARKER,
    DIAGNOSTICS_LOGGER_NAME,
)

_LOGGER_MARKERS = (_BRIDGE_MARKER, _DEGRADED_NOTIFIED_MARKER, _DIAGNOSTICS_MARKER)


@pytest.fixture(autouse=True)
def restore_logging_state():
    """Prevent logging mutations from leaking between unit tests."""
    logger_names = ("", "flextensor", DIAGNOSTICS_LOGGER_NAME, "vllm", "vllm.flextensor")
    snapshots = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        snapshots[name] = {
            "handlers": list(logger.handlers),
            "level": logger.level,
            "propagate": logger.propagate,
            "disabled": logger.disabled,
            "markers": {marker: getattr(logger, marker, None) for marker in _LOGGER_MARKERS},
        }

    yield

    for name, snapshot in snapshots.items():
        logger = logging.getLogger(name)
        logger.handlers[:] = snapshot["handlers"]
        logger.setLevel(snapshot["level"])
        logger.propagate = snapshot["propagate"]
        logger.disabled = snapshot["disabled"]
        for marker, value in snapshot["markers"].items():
            if value is None:
                if hasattr(logger, marker):
                    delattr(logger, marker)
            else:
                setattr(logger, marker, value)


@pytest.fixture(autouse=True)
def mock_cuda_device_properties():
    """Mock torch.cuda.get_device_properties to avoid requiring a real GPU.

    Tests that need to assert specific get_device_properties behaviour (e.g.
    TestMaxGpuMemResolution) override this fixture locally via their own patch context.
    """
    fake_props = MagicMock()
    fake_props.total_memory = 40 * 1024**3  # 40 GB
    fake_props.major = 8  # Ampere compute capability — prevents DeferredCudaCallError from
    fake_props.minor = 0  # _check_capability comparing (props.major, props.minor) < (3, 5)
    fake_props.name = "Mock GPU"
    with patch("flextensor.utils.torch.cuda.get_device_properties", return_value=fake_props):
        yield
