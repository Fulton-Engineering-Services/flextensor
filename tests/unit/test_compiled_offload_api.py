# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``offload(compile_fn=...)`` surface.

Covers the CPU-only pieces of the compiled-offload path: which units
``compile_fn`` is applied to (one compiled graph per offloaded unit, derived from
the patched modules; resident modules left eager; nested patched units skipped in
favour of their outermost patched ancestor), in-place substitution, and the
warm -> measure -> re-plan tail state machine used by
:meth:`~flextensor.OffloadManager.request_strategy_replan` (external compile /
direct-profile paths — not the default ``compile_fn`` + view-profile path).
The full GPU compile path is exercised by the diffusers benchmark, not here.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from torch import nn

from flextensor.compile.lifecycle import COMPILED_WARMUP_FORWARDS
from flextensor.compile.module_swap import resolve_compile_targets
from flextensor.compile.warmup_tail import CompiledOffloadTailState
from flextensor.compiled_offload import bump_dynamo_limits_for_compiled_offload
from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager


class _Block(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):  # noqa: D102
        return self.lin(x)


def _model_with(attr: str, n: int = 3) -> nn.Module:
    """Build a model whose repeated-block container is named ``attr``."""

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            setattr(self, attr, nn.ModuleList([_Block() for _ in range(n)]))
            self.head = nn.Linear(4, 4)

    return _M()


@pytest.fixture
def manager() -> Iterator[OffloadManager]:
    mgr = OffloadManager(f"test-compiled-api-{uuid4()}")
    try:
        yield mgr
    finally:
        mgr.release()


# -- target resolution -----------------------------------------------------
#
# One compiled graph per offloaded unit: targets are derived straight from
# ``manager._patched_modules`` (the modules FlexTensor offloaded), not from any
# block-container heuristic or ``compile_targets`` glob. That makes each graph read
# exactly one rolling slot -> slot-alias safe by construction.


@pytest.mark.parametrize("attr", ["blocks", "transformer_blocks", "layers"])
def test_one_graph_per_offloaded_unit(manager: OffloadManager, attr: str) -> None:
    model = _model_with(attr, n=3)
    manager._model = model  # noqa: SLF001
    manager._patched_modules = list(getattr(model, attr))  # noqa: SLF001
    targets = resolve_compile_targets(manager._model, manager._patched_modules)  # noqa: SLF001
    assert len(targets) == 3
    assert all(isinstance(module, _Block) for _setter, module in targets)


def test_targets_offloaded_units_across_containers(manager: OffloadManager) -> None:
    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block() for _ in range(2)])
            self.layers = nn.ModuleList([_Block() for _ in range(5)])

    model = _M()
    manager._model = model  # noqa: SLF001
    # Every offloaded unit gets its own graph regardless of which container it is
    # in -- there is no "pick one container" heuristic any more.
    manager._patched_modules = [*model.blocks, *model.layers]  # noqa: SLF001
    assert len(resolve_compile_targets(manager._model, manager._patched_modules)) == 7  # noqa: SLF001


def test_resident_modules_are_not_compiled(manager: OffloadManager) -> None:
    model = _model_with("blocks", n=3)
    manager._model = model  # noqa: SLF001
    # Only blocks.0 and blocks.2 are offloaded; blocks.1 and head stay resident.
    manager._patched_modules = [model.blocks[0], model.blocks[2]]  # noqa: SLF001
    targets = resolve_compile_targets(manager._model, manager._patched_modules)  # noqa: SLF001
    compiled = {module for _setter, module in targets}
    assert compiled == {model.blocks[0], model.blocks[2]}
    assert model.blocks[1] not in compiled
    assert model.head not in compiled


def test_nested_patched_unit_skipped_for_outermost(manager: OffloadManager) -> None:
    # An offloaded unit that lives *inside* another offloaded unit must not get its
    # own graph: the outermost patched ancestor is the graph, and compiling both
    # would double-compile and put two slots in the ancestor's graph.
    model = _model_with("blocks", n=2)
    outer = model.blocks[0]
    outer._ft_original_forward_func = outer.forward  # noqa: SLF001 - mark as patched
    manager._model = model  # noqa: SLF001
    # Both the outer block and its inner ``lin`` are in the patched set.
    manager._patched_modules = [outer, outer.lin, model.blocks[1]]  # noqa: SLF001
    compiled = {module for _setter, module in resolve_compile_targets(manager._model, manager._patched_modules)}  # noqa: SLF001
    assert outer in compiled
    assert outer.lin not in compiled  # skipped: patched ancestor ``outer``
    assert model.blocks[1] in compiled


def test_no_targets_when_nothing_offloaded(manager: OffloadManager) -> None:
    manager._model = _model_with("blocks", n=3)  # noqa: SLF001
    manager._patched_modules = []  # noqa: SLF001
    assert resolve_compile_targets(manager._model, manager._patched_modules) == []  # noqa: SLF001


# -- substitution ----------------------------------------------------------


def test_setter_replaces_modulelist_child(manager: OffloadManager) -> None:
    model = _model_with("blocks", n=3)
    manager._model = model  # noqa: SLF001
    manager._patched_modules = list(model.blocks)  # noqa: SLF001
    setter, original = resolve_compile_targets(manager._model, manager._patched_modules)[1]  # noqa: SLF001
    sentinel = nn.Identity()
    setter(sentinel)
    assert model.blocks[1] is sentinel
    setter(original)
    assert model.blocks[1] is original


def test_apply_compile_fn_wraps_every_unit_and_records_undo(manager: OffloadManager) -> None:
    model = _model_with("blocks", n=3)
    manager._model = model  # noqa: SLF001
    manager._patched_modules = list(model.blocks)  # noqa: SLF001

    seen: list[nn.Module] = []

    def compile_fn(unit: nn.Module) -> nn.Module:
        seen.append(unit)
        return nn.Identity()

    manager._compiled.compile_fn = compile_fn
    manager._compiled.apply_compile_fn()

    assert len(seen) == 3
    assert all(isinstance(model.blocks[i], nn.Identity) for i in range(3))
    assert len(manager._compiled.substitutions) == 3

    manager._compiled.teardown()
    # Originals restored, bookkeeping cleared.
    assert all(isinstance(model.blocks[i], _Block) for i in range(3))
    assert manager._compiled.substitutions == []


def test_apply_compile_fn_skips_unit_when_callable_raises(manager: OffloadManager) -> None:
    model = _model_with("blocks", n=3)
    manager._model = model  # noqa: SLF001
    manager._patched_modules = list(model.blocks)  # noqa: SLF001

    def compile_fn(unit: nn.Module) -> nn.Module:
        raise RuntimeError("boom")

    manager._compiled.compile_fn = compile_fn
    manager._compiled.apply_compile_fn()
    # No substitution recorded; units untouched.
    assert manager._compiled.substitutions == []
    assert all(isinstance(model.blocks[i], _Block) for i in range(3))


# -- dynamo recompile-limit bump -------------------------------------------
#
# Every offloaded unit shares one patched forward code object specialized per unit
# on the closed-over offload-unit name, so a stack of >8 units would blow Dynamo's default recompile
# limit (a hard error under fullgraph=True). Compiled offload raises the limits to
# fit the unit count automatically -- no caller ceremony.


def test_bump_dynamo_limits_raises_to_fit_units(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._dynamo as dynamo

    monkeypatch.setattr(dynamo.config, "recompile_limit", 8, raising=False)
    monkeypatch.setattr(dynamo.config, "cache_size_limit", 8, raising=False)

    bump_dynamo_limits_for_compiled_offload(40)

    # needed = 40*2 + 16 = 96, comfortably above the 40 units and the default 8.
    assert dynamo.config.recompile_limit >= 40
    assert dynamo.config.cache_size_limit >= 40


def test_bump_dynamo_limits_never_lowers(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._dynamo as dynamo

    monkeypatch.setattr(dynamo.config, "recompile_limit", 10_000, raising=False)
    bump_dynamo_limits_for_compiled_offload(3)
    # A tiny stack must not shrink an already-generous limit.
    assert dynamo.config.recompile_limit == 10_000


def test_apply_compile_fn_bumps_dynamo_limit(manager: OffloadManager, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch._dynamo as dynamo

    monkeypatch.setattr(dynamo.config, "recompile_limit", 8, raising=False)
    model = _model_with("blocks", n=12)  # > default limit of 8
    manager._model = model  # noqa: SLF001
    manager._patched_modules = list(model.blocks)  # noqa: SLF001
    manager._compiled.compile_fn = lambda m: m

    manager._compiled.apply_compile_fn()

    assert dynamo.config.recompile_limit >= 12


# -- passive tail state machine --------------------------------------------


def test_tail_counts_warm_then_measure_then_replan(manager: OffloadManager, monkeypatch: pytest.MonkeyPatch) -> None:
    import flextensor.custom_ops as custom_ops

    enabled = {"n": 0}
    monkeypatch.setattr(
        custom_ops, "enable_compiled_profiling", lambda *_args, **_kw: enabled.__setitem__("n", enabled["n"] + 1)
    )

    replan_calls = {"n": 0}
    monkeypatch.setattr(
        manager._compiled,
        "finish_replan",
        lambda: replan_calls.__setitem__("n", replan_calls["n"] + 1),
    )

    co = manager._compiled
    co.tail_state = CompiledOffloadTailState.WARMING
    co.warm_seen = 0
    co.measure_seen = 0

    measure_forwards = co.measure_forwards()

    states: list[CompiledOffloadTailState] = []
    for _ in range(COMPILED_WARMUP_FORWARDS + measure_forwards + 2):
        co.on_forward()
        states.append(co.tail_state)

    # Exactly one profiling enable (warm->measure) and one re-plan (measure->done).
    assert enabled["n"] == 1
    assert replan_calls["n"] == 1
    assert states[-1] == CompiledOffloadTailState.DONE
    # Warm window has the expected length before flipping to measuring.
    assert states[: COMPILED_WARMUP_FORWARDS - 1] == [CompiledOffloadTailState.WARMING] * (COMPILED_WARMUP_FORWARDS - 1)


def test_tail_idle_and_done_are_noops(manager: OffloadManager) -> None:
    co = manager._compiled
    for terminal in (CompiledOffloadTailState.IDLE, CompiledOffloadTailState.DONE):
        co.tail_state = terminal
        co.on_forward()
        assert co.tail_state == terminal


def test_tail_unknown_state_raises(manager: OffloadManager) -> None:
    # Bypass the typed setter so we can inject a corrupt internal state.
    manager._compiled._tail.state = "warmimg"  # type: ignore[assignment]  # noqa: SLF001
    with pytest.raises(RuntimeError, match="unexpected tail state"):
        manager._compiled.on_forward()


def test_tail_failure_raises_and_clears_active_loader(manager: OffloadManager, monkeypatch: pytest.MonkeyPatch) -> None:
    import flextensor.custom_ops as custom_ops

    cleared = {"n": 0}
    monkeypatch.setattr(
        custom_ops, "clear_active_loader", lambda *_args, **_kw: cleared.__setitem__("n", cleared["n"] + 1)
    )

    def _boom() -> None:
        raise RuntimeError("FlexTensor compiled-offload: replan failed")

    monkeypatch.setattr(manager._compiled, "finish_replan", _boom)
    co = manager._compiled
    co.tail_state = CompiledOffloadTailState.MEASURING
    co.measure_seen = co.measure_forwards() - 1
    with pytest.raises(RuntimeError, match="replan failed"):
        co.on_forward()
    assert co.tail_state == CompiledOffloadTailState.FAILED
    assert cleared["n"] == 1


def test_tail_failure_subsequent_calls_also_raise(manager: OffloadManager, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise RuntimeError("FlexTensor compiled-offload: replan failed")

    monkeypatch.setattr(manager._compiled, "finish_replan", _boom)
    co = manager._compiled
    co.tail_state = CompiledOffloadTailState.MEASURING
    co.measure_seen = co.measure_forwards() - 1

    with pytest.raises(RuntimeError, match="replan failed"):
        co.on_forward()

    with pytest.raises(RuntimeError, match="re-plan previously failed"):
        co.on_forward()
    assert co.tail_state == CompiledOffloadTailState.FAILED


# -- external compile (external_compile=True, no compile_fn) ---------------


def test_setup_external_compiled_offload_installs_loader(
    manager: OffloadManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    co = manager._compiled
    co.active = True
    co.compile_fn = None
    calls: list[str] = []

    def _record() -> None:
        calls.append("loader")

    monkeypatch.setattr(co, "require_compiled_loader", _record)
    co.setup_external_compiled_offload()
    assert calls == ["loader"]


def test_setup_external_compiled_offload_skipped_when_compile_fn_set(
    manager: OffloadManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    co = manager._compiled
    co.active = True
    co.compile_fn = lambda m: m
    monkeypatch.setattr(
        co,
        "require_compiled_loader",
        lambda: (_ for _ in ()).throw(AssertionError("must not install loader")),
    )
    co.setup_external_compiled_offload()


class _LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):  # noqa: D102
        return self.linear(x)


@patch("flextensor.offload_manager.OffloadManager._transition_to_warmup")
@patch("flextensor.tensor_manager.TensorManager")
@patch("flextensor.strategy.KnapsackStrategy")
class TestFirstLoaderNonDestructiveArming:
    def test_does_not_arm_with_compile_fn_view_profile(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        _mock_transition,
    ) -> None:
        """Default compile_fn + view: no replan → destructive first loader."""
        mock_tm = MagicMock()
        mock_tm._first_loader_non_destructive = False
        mock_tensor_manager_cls.return_value = mock_tm

        om = OffloadManager("test_arming_compile_fn_view")
        config = OffloadConfig(enabled=True, include_patterns=["linear"], profile_mode="view")
        om.offload(_LinearModel(), config=config, compile_fn=lambda m: m)

        assert mock_tm._first_loader_non_destructive is False

    def test_arms_with_compile_fn_non_view_profile(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        _mock_transition,
    ) -> None:
        """compile_fn + non-view: replan intended → preserve source weights."""
        mock_tm = MagicMock()
        mock_tm._first_loader_non_destructive = False
        mock_tm.arm_non_destructive_first_loader.side_effect = lambda: setattr(
            mock_tm, "_first_loader_non_destructive", True
        )
        mock_tensor_manager_cls.return_value = mock_tm

        om = OffloadManager("test_arming_compile_fn_getter")
        config = OffloadConfig(enabled=True, include_patterns=["linear"], profile_mode="getter")
        om.offload(_LinearModel(), config=config, compile_fn=lambda m: m)

        assert mock_tm._first_loader_non_destructive is True

    def test_arms_with_compiled_offload_flag(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        _mock_transition,
    ) -> None:
        mock_tm = MagicMock()
        mock_tm._first_loader_non_destructive = False
        mock_tm.arm_non_destructive_first_loader.side_effect = lambda: setattr(
            mock_tm, "_first_loader_non_destructive", True
        )
        mock_tensor_manager_cls.return_value = mock_tm

        om = OffloadManager("test_arming_external_compile")
        config = OffloadConfig(
            enabled=True,
            external_compile=True,
            include_patterns=["linear"],
        )
        om.offload(_LinearModel(), config=config)

        assert mock_tm._first_loader_non_destructive is True
