# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pre-transition memory warning in vLLM worker.

Tests the low-memory warning that fires before inference transition.
vLLM is not required — we test the warning logic in isolation.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# We test the warning logic by importing the constant and simulating the check
# rather than instantiating the full FlexTensorOffloadWorker (which requires vLLM).
# The worker inserts this check inline, so we replicate the exact logic here.

_LOW_MEMORY_THRESHOLD_GIB = 2.0


def _check_pre_transition_memory(logger: logging.Logger, gib_bytes: int) -> None:
    """Replicate the exact pre-transition memory check from the worker.

    This mirrors the code inserted into warmup_and_profile_model() so the test
    validates the logic without requiring a full vLLM installation.
    """
    import psutil

    try:
        vm = psutil.virtual_memory()
        free_gib = vm.available / gib_bytes
        if free_gib < _LOW_MEMORY_THRESHOLD_GIB:
            logger.warning(
                "FlexTensor: Low host memory (%.1f GiB free) — inference transition may OOM",
                free_gib,
            )
    except Exception:  # noqa: S110
        pass


class TestPreTransitionMemoryWarning:
    """Tests for pre-transition memory warning logic."""

    def test_warns_when_memory_low(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify warning fires when available memory is below threshold."""
        logger = logging.getLogger("test_worker")
        gib_bytes = 1 << 30
        low_mem = SimpleNamespace(available=int(1.5 * gib_bytes))  # 1.5 GiB — below 2.0 threshold

        with (
            patch("psutil.virtual_memory", return_value=low_mem),
            caplog.at_level(logging.WARNING, logger="test_worker"),
        ):
            _check_pre_transition_memory(logger, gib_bytes)

        assert "Low host memory" in caplog.text
        assert "1.5 GiB free" in caplog.text

    def test_no_warning_when_memory_ok(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify no warning when available memory is above threshold."""
        logger = logging.getLogger("test_worker")
        gib_bytes = 1 << 30
        ok_mem = SimpleNamespace(available=int(16.0 * gib_bytes))  # 16 GiB — well above threshold

        with (
            patch("psutil.virtual_memory", return_value=ok_mem),
            caplog.at_level(logging.WARNING, logger="test_worker"),
        ):
            _check_pre_transition_memory(logger, gib_bytes)

        assert "Low host memory" not in caplog.text

    def test_psutil_failure_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify psutil failures are silently swallowed (best-effort check)."""
        logger = logging.getLogger("test_worker")
        gib_bytes = 1 << 30

        with (
            patch("psutil.virtual_memory", side_effect=OSError("psutil broken")),
            caplog.at_level(logging.WARNING, logger="test_worker"),
        ):
            _check_pre_transition_memory(logger, gib_bytes)

        assert "Low host memory" not in caplog.text
        assert "psutil broken" not in caplog.text
