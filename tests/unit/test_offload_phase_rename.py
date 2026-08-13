# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests verifying the v0.4.0 OffloadPhase surface."""

import pytest

import flextensor.offload_manager as _om_module
from flextensor.offload_manager import OffloadPhase


def test_offload_phase_has_only_canonical_members():
    assert list(OffloadPhase) == [
        OffloadPhase.NOT_INITIALIZED,
        OffloadPhase.DISCOVERY,
        OffloadPhase.PROFILING,
        OffloadPhase.INFERENCE,
    ]


def test_expired_phase_aliases_are_absent():
    with pytest.raises(AttributeError):
        _ = _om_module.OffloadState
    with pytest.raises(AttributeError):
        _ = OffloadPhase.WARMUP
    with pytest.raises(AttributeError):
        _ = OffloadPhase.PROFILE
