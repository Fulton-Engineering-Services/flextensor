# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``TensorManager.replan_from_compiled_durations``.

The re-plan rewrites per-layer compute durations with compiled timings,
recomputes the strategy via the shared build helper, repoints the model, and
releases the previous (non-destructive) loader. These tests exercise that
orchestration without a GPU or a real model by stubbing the build/repoint steps.
"""

from unittest.mock import MagicMock

import pytest
import torch

from flextensor.collectors import LayerStatistics
from flextensor.tensor_manager import TensorManager


def _make_manager(durations: dict[str, float]) -> tuple[TensorManager, MagicMock, MagicMock]:
    tm = TensorManager.__new__(TensorManager)
    tm.stats = [LayerStatistics(label=label, tensors=[], duration=7.0) for label in durations]
    old_loader = MagicMock()
    tm.tensor_layer_loader = old_loader
    # A non-empty snapshot is required by the re-plan guard (it refuses to
    # rebuild without the pre-repoint originals). Empty tensors_map means the
    # restore loop is a no-op, which is fine for the orchestration-only tests.
    tm.tensors_map = {}
    tm._replan_source_data = {0: torch.empty(0)}
    # Stub the heavy build/repoint steps; the re-plan logic is what we test.
    build = MagicMock()

    def _fake_build(*, release_tensor_memory: bool = True) -> None:
        # Simulate a rebuilt (different) loader so the old one is released.
        tm.tensor_layer_loader = MagicMock(name="rebuilt_loader")
        build(release_tensor_memory=release_tensor_memory)

    tm._compute_strategy_and_build_loader = _fake_build  # type: ignore[method-assign]
    tm.prepare_final_model = MagicMock()  # type: ignore[method-assign]
    return tm, build, old_loader


def test_replan_rewrites_durations_and_rebuilds_destructively() -> None:
    tm, build, old_loader = _make_manager({"model.layers.0": 3.9, "model.layers.1": 4.0})
    model = object()

    assert tm.replan_from_compiled_durations({"model.layers.0": 3.9, "model.layers.1": 4.0}, model) is True

    durations = {stat.label: stat.duration for stat in tm.stats}
    assert durations == {"model.layers.0": 3.9, "model.layers.1": 4.0}
    build.assert_called_once_with(release_tensor_memory=True)
    tm.prepare_final_model.assert_called_once_with(model)
    old_loader.shutdown.assert_called_once()
    # Model reference is released again after the re-plan.
    assert tm.model is None


def test_replan_frees_old_loader_before_rebuild() -> None:
    """The previous loader's GPU+CPU blocks must be released *before* the rebuild.

    This is the peak-memory guard: freeing the old blocks first lets the caching
    allocator reuse the segments for the rebuilt loader, keeping peak GPU memory
    at ~1x. The test records call order and asserts the old loader's
    ``shutdown`` and ``release_gpu_blocks`` both run before the build.
    """
    tm, _build, old_loader = _make_manager({"model.layers.0": 3.9})
    events: list[str] = []
    old_loader.shutdown.side_effect = lambda: events.append("cpu_release")
    old_loader.allocation_controller.release_gpu_blocks.side_effect = lambda: events.append("gpu_release")

    def _tracking_build(*, release_tensor_memory: bool = True) -> None:
        events.append("build")
        tm.tensor_layer_loader = MagicMock(name="rebuilt_loader")

    tm._compute_strategy_and_build_loader = _tracking_build  # type: ignore[method-assign]

    assert tm.replan_from_compiled_durations({"model.layers.0": 3.9}, object()) is True

    assert events.index("cpu_release") < events.index("build")
    assert events.index("gpu_release") < events.index("build")
    old_loader.allocation_controller.release_gpu_blocks.assert_called_once()


def test_replan_preserves_unmatched_layer_durations() -> None:
    tm, _build, _old = _make_manager({"model.layers.0": 7.0, "model.layers.1": 7.0})

    # Only one label has a compiled timing; the other keeps its eager duration.
    assert tm.replan_from_compiled_durations({"model.layers.0": 3.5}, object()) is True

    durations = {stat.label: stat.duration for stat in tm.stats}
    assert durations == {"model.layers.0": 3.5, "model.layers.1": 7.0}


def test_replan_noops_without_durations() -> None:
    tm, build, _old = _make_manager({"model.layers.0": 7.0})

    assert tm.replan_from_compiled_durations({}, object()) is False
    build.assert_not_called()


def test_replan_noops_when_no_label_matches() -> None:
    tm, build, _old = _make_manager({"model.layers.0": 7.0})

    assert tm.replan_from_compiled_durations({"other.layer": 3.0}, object()) is False
    build.assert_not_called()


def test_replan_refuses_without_original_weight_snapshot() -> None:
    """No pre-repoint snapshot => rebuild would copy stale views => must refuse."""
    tm, build, _old = _make_manager({"model.layers.0": 7.0})
    tm._replan_source_data = {}  # arming was missed before the first build

    assert tm.replan_from_compiled_durations({"model.layers.0": 3.5}, object()) is False
    build.assert_not_called()


def test_replan_restores_original_weights_before_rebuild() -> None:
    """The rebuild must read the original CPU weights, not the stale GPU views.

    Reproduces the corruption fix: the first ``prepare_final_model`` repoints
    ``param.data`` onto loader-1's rolling views (here a ``stale`` tensor). The
    re-plan must restore ``param.data`` to the captured original *before* the
    destructive rebuild copies it, otherwise the rebuilt blocks get stale bytes.
    """
    tm = TensorManager.__new__(TensorManager)
    tm.stats = [LayerStatistics(label="model.layers.0", tensors=[], duration=7.0)]
    tm.tensor_layer_loader = MagicMock()

    original = torch.full((4,), 3.0)
    stale = torch.full((4,), 99.0)  # stands in for loader-1's rolling GPU view
    param = torch.nn.Parameter(original.clone(), requires_grad=False)
    tid = id(param)
    tm.tensors_map = {tid: param}
    # Snapshot captured right after the non-destructive build, before repoint.
    tm._replan_source_data = {tid: param.data}
    # Simulate the first prepare_final_model repointing onto the stale view.
    param.data = stale

    seen_at_build: dict[int, torch.Tensor] = {}

    def _fake_build(*, release_tensor_memory: bool = True) -> None:
        # Capture what the rebuild would copy from for each managed param.
        for t, p in tm.tensors_map.items():
            seen_at_build[t] = p.data.clone()
        tm.tensor_layer_loader = MagicMock(name="rebuilt_loader")

    tm._compute_strategy_and_build_loader = _fake_build  # type: ignore[method-assign]
    tm.prepare_final_model = MagicMock()  # type: ignore[method-assign]

    assert tm.replan_from_compiled_durations({"model.layers.0": 3.9}, object()) is True

    # The rebuild saw the original weights, not the stale rolling view.
    assert torch.equal(seen_at_build[tid], original)
    assert not torch.equal(seen_at_build[tid], stale)
    # Snapshot is cleared afterwards so its host memory can be reclaimed.
    assert tm._replan_source_data == {}


def test_replan_raises_after_loader_release_when_build_fails() -> None:
    tm, _build, old_loader = _make_manager({"model.layers.0": 3.9})
    snapshot = dict(tm._replan_source_data)
    original_stats = list(tm.stats)

    def _failing_build(*, release_tensor_memory: bool = True) -> None:
        raise RuntimeError("OOM during rebuild")

    tm._compute_strategy_and_build_loader = _failing_build  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Inference is unsafe"):
        tm.replan_from_compiled_durations({"model.layers.0": 3.9}, object())

    old_loader.shutdown.assert_called_once()
    assert tm._replan_source_data == snapshot
    assert [stat.duration for stat in tm.stats] == [stat.duration for stat in original_stats]


def _make_load_mode_manager(*, armed: bool) -> tuple[TensorManager, MagicMock]:
    """Minimal TM for ``prepare_infer_load_mode`` release/snapshot contracts."""
    tm = TensorManager.__new__(TensorManager)
    tm.enable_diagnostics = False
    tm._first_loader_non_destructive = armed
    tm._replan_source_data = {}
    weight = torch.nn.Parameter(torch.full((2,), 1.5), requires_grad=False)
    tm.tensors_map = {id(weight): weight}
    state = MagicMock()
    state.load_strategy = object()
    state.stats = []
    state.to_loader_input_data.return_value = object()
    tm.tensor_manager_state = state
    create = MagicMock()
    tm._create_loader = create  # type: ignore[method-assign]
    return tm, create


def test_prepare_infer_load_mode_preserves_sources_when_replan_armed() -> None:
    """Saved-profile / SHM restore must snapshot weights when replan is armed."""
    tm, create = _make_load_mode_manager(armed=True)
    weight = next(iter(tm.tensors_map.values()))

    tm.prepare_infer_load_mode()

    create.assert_called_once_with(
        tm.tensor_manager_state.to_loader_input_data.return_value,
        prepare_state=False,
        release_tensor_memory=False,
    )
    assert set(tm._replan_source_data) == {id(weight)}
    assert torch.equal(tm._replan_source_data[id(weight)], weight.data)
    # Arming is one-shot after the first inference loader build.
    assert tm._first_loader_non_destructive is False


def test_prepare_infer_load_mode_releases_sources_when_replan_not_armed() -> None:
    tm, create = _make_load_mode_manager(armed=False)

    tm.prepare_infer_load_mode()

    create.assert_called_once_with(
        tm.tensor_manager_state.to_loader_input_data.return_value,
        prepare_state=False,
        release_tensor_memory=True,
    )
    assert tm._replan_source_data == {}


def test_clear_replan_state_drops_arm_and_snapshot() -> None:
    tm = TensorManager.__new__(TensorManager)
    tm._first_loader_non_destructive = True
    tm._replan_source_data = {0: torch.empty(0)}

    tm.clear_replan_state()

    assert tm._first_loader_non_destructive is False
    assert tm._replan_source_data == {}
