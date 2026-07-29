# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trap exit path must hand GPU tensors back even when timing calls raise.

``__exit__`` records a CUDA event and synchronizes on it. When the wrapped
forward has already faulted the CUDA context, ``synchronize()`` raises — and if
``tensor_layer_loader.exit()`` sits inside that same ``try``, it never runs. The
label's GPU copies stay in ``cpu_to_gpu_map``, ``_active_counts`` keeps a
nonzero entry, and ``param.data`` is left pointing at storage the loader still
believes it owns. ``WarmupTrapDirect`` already used the nested-``finally``
pattern these tests pin for ``Trap`` and ``TrapDirect``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from flextensor.trap_tensor_mode import TrapDirect


class _BoomError(RuntimeError):
    """Stand-in for a CUDA fault surfaced by ``end_event.synchronize()``."""


def _tensor_manager(*, sync_raises: bool) -> MagicMock:
    tm = MagicMock()
    tm.trap_start_event = MagicMock()
    tm.trap_end_event = MagicMock()
    if sync_raises:
        tm.trap_end_event.synchronize.side_effect = _BoomError("CUDA error: unspecified launch failure")
    tm.trap_nesting_guard = MagicMock()
    tm.is_current_trap_tainted.return_value = False
    return tm


class TestTrapDirectExitReleasesLoader:
    def test_loader_exit_runs_when_synchronize_raises(self) -> None:
        tm = _tensor_manager(sync_raises=True)
        trap = TrapDirect(tm, "layers.0", device_gpu="cuda:0")

        with pytest.raises(_BoomError):
            trap.__exit__(None, None, None)

        tm.tensor_layer_loader.exit.assert_called_once_with("layers.0")

    def test_nesting_guard_still_released_when_synchronize_raises(self) -> None:
        tm = _tensor_manager(sync_raises=True)
        trap = TrapDirect(tm, "layers.0", device_gpu="cuda:0")

        with pytest.raises(_BoomError):
            trap.__exit__(None, None, None)

        tm.trap_nesting_guard.release.assert_called_once()

    def test_original_error_is_not_masked_by_the_release(self) -> None:
        """The CUDA fault must still surface; the cleanup must not swallow it."""
        tm = _tensor_manager(sync_raises=True)
        trap = TrapDirect(tm, "layers.0", device_gpu="cuda:0")

        with pytest.raises(_BoomError, match="unspecified launch failure"):
            trap.__exit__(None, None, None)

    def test_happy_path_still_releases_exactly_once(self) -> None:
        tm = _tensor_manager(sync_raises=False)
        trap = TrapDirect(tm, "layers.0", device_gpu="cuda:0")

        trap.__exit__(None, None, None)

        tm.tensor_layer_loader.exit.assert_called_once_with("layers.0")
        tm.reset_current_trap_taint.assert_called_once()
