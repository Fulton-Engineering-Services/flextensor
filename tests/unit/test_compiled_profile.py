# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for view-mode compiled profile (auto path / no replan tail)."""

from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from flextensor.compile.lifecycle import PROFILE_COMPILE_WARMUP_FORWARDS
from flextensor.compile.warmup_tail import CompiledOffloadTailState
from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager


@pytest.fixture
def manager() -> OffloadManager:
    om = OffloadManager("test_compiled_profile")
    om.set_config(
        OffloadConfig(
            enabled=True,
            profiling_iters=10,
            profile_mode="view",
        )
    )
    co = om._compiled
    co.compile_fn = lambda m: m
    co.active = True
    co.replan_active = False
    co.profile_active = True
    om._tensor_manager = MagicMock()
    om._tensor_manager.trap_start_event = MagicMock()
    om._tensor_manager.trap_end_event = MagicMock()
    return om


def test_compiled_profile_activation_auto():
    om = OffloadManager("test_activation")
    config = OffloadConfig(profile_mode="view")
    om._compiled.resolve_activation(config, compile_fn=lambda m: m)
    assert om.compiled_profile_active is True
    assert om.compiled_replan_active is False


def test_compile_fn_non_view_skips_view_profile_and_marks_replan():
    om = OffloadManager("test_activation_getter")
    config = OffloadConfig(profile_mode="getter")
    om._compiled.resolve_activation(config, compile_fn=lambda m: m)
    assert om.compiled_profile_active is False
    assert om.compiled_replan_active is True
    assert om.compiled_offload_active is True


def test_external_compiled_offload_arms_replan():
    om = OffloadManager("test_external_replan")
    config = OffloadConfig(external_compile=True)
    om._compiled.resolve_activation(config, compile_fn=None)
    assert om.compiled_offload_active is True
    assert om.compiled_replan_active is True
    assert om.compiled_profile_active is False


def test_eager_profiling_budget_uses_full_iters_when_compiled_profile():
    # ``skip_discovery=False`` so ``discovery_iters`` is included in the
    # ``iters_before_inference`` arithmetic this test pins. Under the
    # ``skip_discovery=True`` default the discovery component drops to
    # zero — the skip variants of ``iters_before_inference`` are covered
    # in ``tests/unit/test_offload_manager_phase.py``.
    om = OffloadManager("test_eager_budget")
    om.config = OffloadConfig(enabled=True, profiling_iters=12, skip_discovery=False)
    co = om._compiled
    co.active = True
    co.replan_active = False
    co.profile_active = True
    assert om._eager_profiling_iters() == 12
    assert co.extra_iters_before_inference() == PROFILE_COMPILE_WARMUP_FORWARDS
    assert om.iters_before_inference == (om.config.discovery_iters + 12 + PROFILE_COMPILE_WARMUP_FORWARDS)


def test_should_record_profile_compile_duration_warmup_slots(manager: OffloadManager):
    co = manager._compiled
    co.profile_compile_warm_remaining = PROFILE_COMPILE_WARMUP_FORWARDS
    for _ in range(PROFILE_COMPILE_WARMUP_FORWARDS):
        # Multiple units in one model forward must share the same warm decision.
        assert co.should_record_profile_compile_duration() is False
        assert co.should_record_profile_compile_duration() is False
        co.advance_profile_compile_warmup()
    assert co.should_record_profile_compile_duration() is True
    assert co.should_record_profile_compile_duration() is True


def test_profile_compile_warmup_counts_model_forwards_not_units(manager: OffloadManager):
    """Two units across several model forwards must not burn warmup per unit call."""
    co = manager._compiled
    co.profile_compile_warm_remaining = 2

    # Model forward 1: both units still warming.
    assert co.should_record_profile_compile_duration() is False
    assert co.should_record_profile_compile_duration() is False
    co.advance_profile_compile_warmup()
    assert co.profile_compile_warm_remaining == 1

    # Model forward 2: still warming for every unit.
    assert co.should_record_profile_compile_duration() is False
    assert co.should_record_profile_compile_duration() is False
    co.advance_profile_compile_warmup()
    assert co.profile_compile_warm_remaining == 0

    # Model forward 3: both units record.
    assert co.should_record_profile_compile_duration() is True
    assert co.should_record_profile_compile_duration() is True


def test_profile_compile_warmup_does_not_consume_profiling_iters_budget(manager: OffloadManager):
    """With profiling_iters=3, three warmup forwards leave the full measure window."""
    from flextensor.offload_manager import OffloadPhase

    manager.set_config(
        OffloadConfig(
            enabled=True,
            profiling_iters=3,
            profile_mode="view",
        )
    )
    co = manager._compiled
    co.profile_compile_warm_remaining = PROFILE_COMPILE_WARMUP_FORWARDS
    manager._current_phase = OffloadPhase.PROFILING
    manager._iteration_count = 0
    manager._tensor_manager.is_profiling_suspended.return_value = False

    with patch.object(manager, "_transition_to_inference") as mock_transition:
        # Warmup: counter frozen, recording suppressed.
        for _ in range(PROFILE_COMPILE_WARMUP_FORWARDS):
            assert co.should_record_profile_compile_duration() is False
            manager.update_state()
            assert manager._iteration_count == 0
            mock_transition.assert_not_called()

        assert co.profile_compile_warm_remaining == 0
        assert co.should_record_profile_compile_duration() is True

        # Measured window: full profiling_iters samples before INFERENCE.
        for expected in range(1, 3):
            manager.update_state()
            assert manager._iteration_count == expected
            mock_transition.assert_not_called()

        manager.update_state()
        assert manager._iteration_count == 3
        mock_transition.assert_called_once()


def test_setup_compiled_profile_phase_installs_forwards_and_inner_compile(manager: OffloadManager):
    block = nn.Linear(4, 4)
    block._ft_original_forward_func = type(block).forward  # noqa: SLF001
    block._ft_offload_name = "block0"  # noqa: SLF001
    model = nn.Sequential(block)
    manager._model = model
    co = manager._compiled

    with (
        patch.object(co, "install_profile_compiled_forwards") as mock_install,
        patch.object(co, "apply_profile_compile_fn") as mock_apply,
    ):
        co.on_enter_profile()

    mock_install.assert_called_once_with(model)
    mock_apply.assert_called_once_with(model)
    assert co.profile_compile_warm_remaining == PROFILE_COMPILE_WARMUP_FORWARDS


def test_setup_compiled_inference_no_replan_skips_tail_arm(manager: OffloadManager):
    manager._model = nn.Linear(1, 1)
    co = manager._compiled
    with (
        patch.object(co, "require_compiled_loader") as mock_loader,
        patch.object(co, "apply_compile_fn") as mock_apply,
    ):
        co.setup_inference_no_replan()

    mock_loader.assert_called_once()
    mock_apply.assert_called_once()
    assert co.tail_state == CompiledOffloadTailState.DONE


def test_profile_compiled_inner_no_module_cycle_and_cleared_on_release(manager: OffloadManager):
    """Identity compile_fn must not register owner↔carrier cycles; release clears the attr."""
    block = nn.Linear(4, 4)
    block.__dict__["_ft_original_forward_func"] = type(block).forward
    block.__dict__["_ft_offload_name"] = "block0"
    model = nn.Sequential(block)
    manager._model = model
    co = manager._compiled
    co.compile_fn = lambda m: m

    co.apply_profile_compile_fn(model)

    inner = block.__dict__.get("_ft_profile_compiled_inner")
    assert inner is not None
    assert "_owner" not in inner._modules
    assert "_ft_profile_compiled_inner" not in block._modules
    # Would hang / recurse forever if owner -> compiled_inner -> owner were registered.
    _ = model.state_dict()

    manager.release()
    assert "_ft_profile_compiled_inner" not in block.__dict__
