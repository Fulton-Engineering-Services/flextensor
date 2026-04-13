# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests verifying OffloadPhase rename and deprecated OffloadState aliases."""

import warnings

import pytest

import flextensor.offload_manager as _om_module
from flextensor.offload_manager import OffloadPhase


def test_offload_phase_new_names():
    assert OffloadPhase.NOT_INITIALIZED.value == "not_initialized"
    assert OffloadPhase.DISCOVERY.value == "discovery"
    assert OffloadPhase.PROFILING.value == "profiling"
    assert OffloadPhase.INFERENCE.value == "inference"


def test_offload_state_class_alias_warns():
    with pytest.warns(DeprecationWarning, match="OffloadState.*deprecated.*OffloadPhase"):
        _om_module.__getattr__("OffloadState")


def test_offload_state_class_alias_is_offload_phase():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = _om_module.__getattr__("OffloadState")
    assert result is OffloadPhase


def test_warmup_dot_access_warns():
    with pytest.warns(DeprecationWarning, match="OffloadPhase.WARMUP.*deprecated.*DISCOVERY"):
        _ = OffloadPhase.WARMUP  # noqa: B018


def test_profile_dot_access_warns():
    with pytest.warns(DeprecationWarning, match="OffloadPhase.PROFILE.*deprecated.*PROFILING"):
        _ = OffloadPhase.PROFILE  # noqa: B018


def test_warmup_alias_is_discovery():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert OffloadPhase.WARMUP is OffloadPhase.DISCOVERY


def test_profile_alias_is_profiling():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert OffloadPhase.PROFILE is OffloadPhase.PROFILING


def test_warmup_bracket_access_warns():
    with pytest.warns(DeprecationWarning, match="OffloadPhase.WARMUP.*deprecated.*DISCOVERY"):
        result = OffloadPhase["WARMUP"]
    assert result is OffloadPhase.DISCOVERY


def test_profile_bracket_access_warns():
    with pytest.warns(DeprecationWarning, match="OffloadPhase.PROFILE.*deprecated.*PROFILING"):
        result = OffloadPhase["PROFILE"]
    assert result is OffloadPhase.PROFILING


def test_bracket_access_valid_member():
    assert OffloadPhase["DISCOVERY"] is OffloadPhase.DISCOVERY
    assert OffloadPhase["PROFILING"] is OffloadPhase.PROFILING


def test_bracket_access_invalid_member_raises_key_error():
    with pytest.raises(KeyError):
        _ = OffloadPhase["NONEXISTENT"]


def test_offload_state_warmup_warns():
    """OffloadState.WARMUP (through the class alias) emits deprecation warning."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        OffloadState = _om_module.__getattr__("OffloadState")  # noqa: N806

    with pytest.warns(DeprecationWarning, match="WARMUP.*deprecated"):
        result = OffloadState.WARMUP  # noqa: B018
    assert result is OffloadPhase.DISCOVERY


def test_list_offload_phase_has_exactly_four_members():
    """Deprecated aliases must not leak into iteration."""
    members = list(OffloadPhase)
    assert len(members) == 4
    assert set(members) == {
        OffloadPhase.NOT_INITIALIZED,
        OffloadPhase.DISCOVERY,
        OffloadPhase.PROFILING,
        OffloadPhase.INFERENCE,
    }
