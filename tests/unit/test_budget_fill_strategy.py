# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from itertools import pairwise

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy import BudgetFillGreedyStrategy
from flextensor.strategy.assignment import OptimizedRoundRobinAssignment, StrictRoundRobinAssignment
from flextensor.strategy.evaluation import evaluate_strategy_result
from flextensor.strategy.protocol import StrategyComputeError

_MB = 1024 * 1024


def _tensor(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=tensor_id,
        name=f"t{tensor_id}",
        size_bytes=size_bytes,
        load_time_ms=1.0,
    )


def _layers(sizes: list[int]) -> list[LayerStatistics]:
    return [
        LayerStatistics(
            label=f"l{index}",
            tensors=[_tensor(index + 1, size)],
            duration=1.0,
        )
        for index, size in enumerate(sizes)
    ]


def test_requires_budget() -> None:
    strategy = BudgetFillGreedyStrategy(n_blocks=2)
    with pytest.raises(StrategyComputeError, match="max_gpu_mem_bytes"):
        strategy.compute(_layers([_MB, _MB, _MB, _MB]), memory_stats={_MB: 1.0})


def test_requires_two_layers() -> None:
    strategy = BudgetFillGreedyStrategy(n_blocks=2)
    with pytest.raises(StrategyComputeError, match="two layers"):
        strategy.compute(
            [LayerStatistics(label="only", tensors=[_tensor(1, _MB)], duration=1.0)],
            max_gpu_mem_bytes=10 * _MB,
        )


def test_no_offload_when_model_fits() -> None:
    layers = _layers([_MB, _MB, _MB, _MB])
    result = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=10 * _MB)
    assert result.strategy_map == {}
    offloaded = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    assert offloaded == 0


def test_fills_budget_under_limit() -> None:
    # 16 MiB model; with 2 blocks peak can reach ~6 MiB after offloading layers 1..n-1.
    layers = _layers([2 * _MB] * 8)
    memory_stats = {2 * _MB: 0.5}
    budget = 6 * _MB
    result = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, memory_stats, budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    offloaded = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    assert offloaded > 0
    assert score.peak_memory_bytes <= budget
    assert score.is_valid
    # First-layer tensor stays resident (never appears in strategy_map values).
    first_id = layers[0].tensors[0].tensor_id
    selected = {t.tensor_id for ts in result.strategy_map.values() for t in ts}
    assert first_id not in selected


def test_minimizes_offload_under_budget() -> None:
    """Offload only as much as needed so peak stays under the budget."""
    layers = _layers([_MB] + [4 * _MB] * 4)
    budget = 10 * _MB
    result = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    # Stay near the budget rather than leaving a large unused gap.
    assert budget - score.peak_memory_bytes < 4 * _MB


def test_warns_when_budget_unreachable() -> None:
    """Best-effort plan is returned (for comparison) when the budget is unreachable."""
    layers = _layers([20 * _MB, _MB, _MB, _MB])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=5 * _MB)
    assert any("could not meet" in str(item.message) for item in caught)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=5 * _MB)
    assert score.peak_memory_bytes > 5 * _MB
    assert not score.is_valid


def test_within_layer_prefers_fit_then_largest() -> None:
    """Fit under ``duration * scale`` ranks before size; among fits, larger first."""
    from flextensor.strategy.budget_fill import _rank_candidates

    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, _MB)], duration=1.0),
        LayerStatistics(
            label="l1",
            tensors=[
                # 8 MiB misses the 1.0 ms window; 2/4 MiB fit.
                TensorStatistics(tensor_id=2, name="small", size_bytes=2 * _MB, load_time_ms=0.5),
                TensorStatistics(tensor_id=3, name="huge", size_bytes=8 * _MB, load_time_ms=5.0),
                TensorStatistics(tensor_id=4, name="mid", size_bytes=4 * _MB, load_time_ms=0.8),
            ],
            duration=1.0,
        ),
    ]
    ranked = _rank_candidates(layers, threshold_mb=0.1, scale=1.0, interpolator=None)
    assert [c.tensor.tensor_id for c in ranked] == [4, 2, 3]


def test_scale_changes_fit_ranking() -> None:
    """Larger scale widens the window so a previously non-fitting tensor ranks first."""
    from flextensor.strategy.budget_fill import _rank_candidates

    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, _MB)], duration=1.0),
        LayerStatistics(
            label="l1",
            tensors=[
                TensorStatistics(tensor_id=2, name="small_fit", size_bytes=2 * _MB, load_time_ms=0.5),
                TensorStatistics(tensor_id=3, name="large_slow", size_bytes=4 * _MB, load_time_ms=2.0),
            ],
            duration=1.0,
        ),
    ]
    tight = _rank_candidates(layers, threshold_mb=0.1, scale=1.0, interpolator=None)
    assert [c.tensor.tensor_id for c in tight] == [2, 3]
    assert {c.tensor.tensor_id for c in tight[:1]} == {2}

    wide = _rank_candidates(layers, threshold_mb=0.1, scale=3.0, interpolator=None)
    # Both fit; larger (id 3) ranks first.
    assert [c.tensor.tensor_id for c in wide] == [3, 2]
    assert {c.tensor.tensor_id for c in wide[:1]} == {3}


def test_prefers_fit_tensors_when_selecting_offload() -> None:
    """When offload is needed, fit-ranked tensors are chosen before slow ones."""
    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, 2 * _MB)], duration=1.0),
        LayerStatistics(
            label="l1",
            tensors=[
                TensorStatistics(tensor_id=2, name="fit", size_bytes=2 * _MB, load_time_ms=0.5),
                TensorStatistics(tensor_id=3, name="slow", size_bytes=2 * _MB, load_time_ms=5.0),
            ],
            duration=1.0,
        ),
        LayerStatistics(label="l2", tensors=[_tensor(4, 2 * _MB)], duration=1.0),
        LayerStatistics(label="l3", tensors=[_tensor(5, 2 * _MB)], duration=1.0),
    ]
    # Total 10 MiB; empty peak 10. Budget 8 needs a ranked prefix that excludes the slow tensor.
    budget = 8 * _MB
    result = BudgetFillGreedyStrategy(n_blocks=2, threshold_mb=0.1).compute(layers, max_gpu_mem_bytes=budget)
    selected = {t.tensor_id for ts in result.strategy_map.values() for t in ts}
    assert 2 in selected
    assert 3 not in selected
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget


def test_optimized_assignment_can_use_fewer_blocks() -> None:
    """OptimizedRoundRobin may pick fewer than n_blocks when that lowers peak."""
    layers = _layers([2 * _MB] * 8)
    budget = 6 * _MB
    result = BudgetFillGreedyStrategy(
        n_blocks=4,
        assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=2, max_blocks=4),
    ).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert result.block_data is not None
    assert 2 <= len(result.block_data.block_sizes) <= 4


def test_rejects_n_blocks_one() -> None:
    """Single-block pipelines are unsupported; use 0 or >= 2."""
    with pytest.raises(ValueError, match="n_blocks"):
        BudgetFillGreedyStrategy(n_blocks=1)


def test_defaults_to_strict_assignment() -> None:
    strategy = BudgetFillGreedyStrategy(n_blocks=2)
    assert isinstance(strategy.assignment_strategy, StrictRoundRobinAssignment)


def test_two_layers_cannot_beat_model_size_peak() -> None:
    """One transfer slot: block size equals offload, so peak stays at model size."""
    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, 4 * _MB)], duration=1.0),
        LayerStatistics(label="l1", tensors=[_tensor(2, 4 * _MB)], duration=1.0),
    ]
    budget = 6 * _MB
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    assert any("could not meet" in str(item.message) for item in caught)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes == 8 * _MB
    assert not score.is_valid


def test_partial_offload_can_beat_full_under_strict_rr() -> None:
    """Full offload is not an infeasibility proof: dropping a transfer can lower peak.

    Transfer sizes [100, 1, 1, 90] MiB under Strict@2: full plan peaks at 191 MiB,
    but skipping one 1 MiB slot yields [100, 1, 90] at exactly 103 MiB.
    """
    from flextensor.strategy import BudgetFillLayerDEStrategy, BudgetFillStrategy, BudgetFillTensorDEStrategy

    sizes = [1 * _MB, 100 * _MB, 1 * _MB, 1 * _MB, 90 * _MB]
    layers = [
        LayerStatistics(label=f"l{index}", tensors=[_tensor(index + 1, size)], duration=100.0)
        for index, size in enumerate(sizes)
    ]
    budget = 103 * _MB
    assign = StrictRoundRobinAssignment()

    for strategy in (
        BudgetFillGreedyStrategy(n_blocks=2, threshold_mb=0.0, assignment_strategy=assign),
        BudgetFillStrategy(
            n_blocks=2,
            threshold_mb=0.0,
            assignment_strategy=assign,
            enable_layer_de=True,
            enable_tensor_de=False,
        ),
        BudgetFillLayerDEStrategy(n_blocks=2, threshold_mb=0.0, assignment_strategy=assign, seed=0),
        BudgetFillTensorDEStrategy(n_blocks=2, threshold_mb=0.0, assignment_strategy=assign, seed=0),
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strategy.compute(layers, max_gpu_mem_bytes=budget)
        assert not any("could not meet" in str(item.message) for item in caught), type(strategy).__name__
        score = evaluate_strategy_result(result, layers, type(strategy).__name__, max_gpu_mem_bytes=budget)
        assert score.peak_memory_bytes <= budget, (
            type(strategy).__name__,
            score.peak_memory_bytes / _MB,
        )
        assert score.is_valid


def test_greedy_large_prefix_rescan_uses_configured_assignment() -> None:
    """With >64 candidates, expand around Strict@2 when real assignment disagrees.

    One 1 MiB resident + 65 one-MiB transfers, Optimized over 3-4 blocks, 10 MiB
    budget: Strict@2 picks an over-budget prefix (11 MiB peak); a nearby
    configured-assignment prefix (60) is feasible at 9 MiB.
    """
    layers = _layers([_MB] * 66)  # resident + 65 transfers
    budget = 10 * _MB
    assign = OptimizedRoundRobinAssignment(min_blocks=3, max_blocks=4)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = BudgetFillGreedyStrategy(
            n_blocks=4,
            threshold_mb=0.0,
            assignment_strategy=assign,
        ).compute(layers, max_gpu_mem_bytes=budget)

    assert not any("could not meet" in str(item.message) for item in caught)
    score = evaluate_strategy_result(result, layers, "greedy-large", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert score.is_valid
    assert score.peak_memory_bytes == 9 * _MB
    offloaded = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    assert offloaded == 60 * _MB


def test_strict_neighbors_differ() -> None:
    """Strict RR with 2 blocks assigns alternating blocks to consecutive transfers."""
    layers = _layers([2 * _MB] * 6)
    budget = 8 * _MB
    result = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    assert result.block_data is not None
    labels = [layer.label for layer in layers if layer.label in result.strategy_map]
    blocks = [result.block_data.label_to_block_id[label] for label in labels]
    for left, right in pairwise(blocks):
        assert left != right


def test_tensor_de_meets_budget_and_minimizes_offload() -> None:
    from flextensor.strategy import BudgetFillTensorDEStrategy

    layers = _layers([2 * _MB] * 8)
    budget = 6 * _MB
    result = BudgetFillTensorDEStrategy(
        n_blocks=2,
        pop_size=12,
        epoch=20,
        max_early_stop=8,
        seed=0,
    ).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget-tensor-de", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert score.is_valid
    offloaded = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    assert offloaded > 0


def test_tensor_de_requires_budget() -> None:
    from flextensor.strategy import BudgetFillTensorDEStrategy

    with pytest.raises(StrategyComputeError, match="max_gpu_mem_bytes"):
        BudgetFillTensorDEStrategy(n_blocks=2).compute(_layers([_MB, _MB, _MB]), memory_stats={_MB: 1.0})


def test_tensor_de_no_worse_offload_than_greedy_when_both_feasible() -> None:
    """DE is seeded by greedy BudgetFill; result should not offload more when both meet budget."""
    from flextensor.strategy import BudgetFillTensorDEStrategy

    layers = _layers([2 * _MB] * 8)
    budget = 8 * _MB
    greedy = BudgetFillGreedyStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    tensor_de_result = BudgetFillTensorDEStrategy(
        n_blocks=2,
        pop_size=12,
        epoch=15,
        max_early_stop=6,
        seed=1,
    ).compute(layers, max_gpu_mem_bytes=budget)
    greedy_off = sum(t.size_bytes for ts in greedy.strategy_map.values() for t in ts)
    scipy_off = sum(t.size_bytes for ts in tensor_de_result.strategy_map.values() for t in ts)
    score = evaluate_strategy_result(tensor_de_result, layers, "budget-tensor-de", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert scipy_off <= greedy_off


def test_layer_de_meets_budget() -> None:
    from flextensor.strategy import BudgetFillLayerDEStrategy

    layers = _layers([2 * _MB] * 8)
    budget = 6 * _MB
    result = BudgetFillLayerDEStrategy(
        n_blocks=2,
        pop_size=12,
        epoch=25,
        max_early_stop=10,
        seed=0,
    ).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget-layer-de", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert score.is_valid


def test_soft_objective_is_lexicographic_offload_then_fit() -> None:
    """One extra offloaded byte beats any amount of non-fitting offload."""
    from flextensor.strategy.budget_fill import _soft_objective

    total = 100 * _MB
    # Review example: 10 MiB all nonfit must lose to 12 MiB all fit under the
    # old weighted formula — and must win under lexicographic offload-first.
    ten_nonfit = _soft_objective(10 * _MB, 10 * _MB, total)
    twelve_fit = _soft_objective(12 * _MB, 0.0, total)
    assert ten_nonfit < twelve_fit

    # Equal offload: less nonfit wins.
    assert _soft_objective(10 * _MB, 0.0, total) < _soft_objective(10 * _MB, 1 * _MB, total)

    # One byte more offload loses even if the smaller plan is all nonfit.
    assert _soft_objective(10 * _MB, 10 * _MB, total) < _soft_objective(10 * _MB + 1, 0.0, total)


def test_auto_de_params_scales_down_for_small_problems() -> None:
    from flextensor.strategy.budget_fill import _auto_de_params

    pop, epoch, stall = _auto_de_params(6, pop_size=20, epoch=60, max_early_stop=20, binary=False)
    assert pop <= 10
    assert epoch <= 15
    assert stall <= 8

    pop_big, epoch_big, _ = _auto_de_params(80_000, pop_size=30, epoch=80, max_early_stop=25, binary=True)
    assert pop_big <= 12
    assert epoch_big <= 20


def test_tensor_de_enumerates_tiny_non_prefix_subset() -> None:
    """With ≤4 candidates, exhaustively search masks instead of returning greedy.

    Resident 1 MiB + transfers [1, 1, 1, 2] MiB under Strict@2 @ 5 MiB: greedy
    ranked-prefix offloads 4 MiB; the three 1 MiB transfers meet the same peak
    while offloading only 3 MiB.
    """
    from flextensor.strategy import BudgetFillGreedyStrategy, BudgetFillTensorDEStrategy

    layers = _layers([_MB, _MB, _MB, _MB, 2 * _MB])
    budget = 5 * _MB
    assign = StrictRoundRobinAssignment()

    greedy = BudgetFillGreedyStrategy(n_blocks=2, threshold_mb=0.0, assignment_strategy=assign).compute(
        layers, max_gpu_mem_bytes=budget
    )
    tensor = BudgetFillTensorDEStrategy(n_blocks=2, threshold_mb=0.0, assignment_strategy=assign, seed=0).compute(
        layers, max_gpu_mem_bytes=budget
    )

    greedy_off = sum(t.size_bytes for ts in greedy.strategy_map.values() for t in ts)
    tensor_off = sum(t.size_bytes for ts in tensor.strategy_map.values() for t in ts)
    greedy_score = evaluate_strategy_result(greedy, layers, "greedy", max_gpu_mem_bytes=budget)
    tensor_score = evaluate_strategy_result(tensor, layers, "tensor", max_gpu_mem_bytes=budget)

    assert greedy_score.peak_memory_bytes <= budget
    assert tensor_score.peak_memory_bytes <= budget
    assert greedy_off == 4 * _MB
    assert tensor_off == 3 * _MB
    assert tensor_off < greedy_off
    tensor_ids = {t.tensor_id for ts in tensor.strategy_map.values() for t in ts}
    assert tensor_ids == {2, 3, 4}  # three 1 MiB transfers; not the 2 MiB


def test_tensor_de_can_beat_layer_de_on_non_prefix_subset() -> None:
    """Tensor DE can take a mid-ranked tensor without its within-layer prefix.

    L1 ranked fit→size as [5, 5, 6]; Layer DE cannot take only the 6, but Tensor
    DE can — meeting the budget with less total offload.
    """
    from flextensor.strategy import BudgetFillLayerDEStrategy, BudgetFillTensorDEStrategy

    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, 50 * _MB)], duration=1.0),
        LayerStatistics(
            label="l1",
            tensors=[
                TensorStatistics(tensor_id=2, name="fit_a", size_bytes=5 * _MB, load_time_ms=0.4),
                TensorStatistics(tensor_id=3, name="fit_b", size_bytes=5 * _MB, load_time_ms=0.4),
                TensorStatistics(tensor_id=4, name="slow_6", size_bytes=6 * _MB, load_time_ms=9.0),
            ],
            duration=1.0,
        ),
        LayerStatistics(label="l2", tensors=[_tensor(5, 10 * _MB)], duration=10.0),
        LayerStatistics(label="l3", tensors=[_tensor(6, 10 * _MB)], duration=10.0),
        LayerStatistics(label="l4", tensors=[_tensor(7, 10 * _MB)], duration=10.0),
    ]
    budget = 90 * _MB
    assign = StrictRoundRobinAssignment()
    common = {"n_blocks": 2, "threshold_mb": 0.0, "assignment_strategy": assign, "seed": 42}

    layer = BudgetFillLayerDEStrategy(**common, pop_size=24, epoch=60).compute(layers, max_gpu_mem_bytes=budget)
    tensor = BudgetFillTensorDEStrategy(**common, pop_size=24, epoch=60).compute(layers, max_gpu_mem_bytes=budget)
    layer_off = sum(t.size_bytes for ts in layer.strategy_map.values() for t in ts)
    tensor_off = sum(t.size_bytes for ts in tensor.strategy_map.values() for t in ts)
    layer_score = evaluate_strategy_result(layer, layers, "layer", max_gpu_mem_bytes=budget)
    tensor_score = evaluate_strategy_result(tensor, layers, "tensor", max_gpu_mem_bytes=budget)

    assert layer_score.peak_memory_bytes <= budget
    assert tensor_score.peak_memory_bytes <= budget
    assert tensor_off < layer_off
    tensor_ids = {t.tensor_id for ts in tensor.strategy_map.values() for t in ts}
    assert 4 in tensor_ids  # the non-prefix 6 MiB tensor
    assert not ({2, 3} <= tensor_ids)  # without both fit-ranked neighbors


def test_facade_defaults_enable_both_de_solvers() -> None:
    from flextensor.strategy import BudgetFillStrategy

    strategy = BudgetFillStrategy(n_blocks=2)
    assert strategy.enable_layer_de is True
    assert strategy.enable_tensor_de is True


def test_facade_does_not_skip_tensor_de_when_peak_pinches() -> None:
    """Pinched peak ≠ min offload: non-prefix subset can offload less at same peak.

    Resident 1 MiB + transfers [1, 1, 1, 1, 2] MiB under Strict@2 @ 6 MiB: greedy
    ranked-prefix offloads 4 MiB while pinching the budget; the three 1 MiB
    transfers also peak at 6 MiB with only 3 MiB offload.
    """
    from flextensor.strategy import BudgetFillGreedyStrategy, BudgetFillStrategy

    layers = _layers([_MB, _MB, _MB, _MB, _MB, 2 * _MB])
    budget = 6 * _MB
    assign = StrictRoundRobinAssignment()

    greedy = BudgetFillGreedyStrategy(n_blocks=2, threshold_mb=0.0, assignment_strategy=assign).compute(
        layers, max_gpu_mem_bytes=budget
    )
    facade = BudgetFillStrategy(
        n_blocks=2,
        threshold_mb=0.0,
        assignment_strategy=assign,
        enable_layer_de=True,
        enable_tensor_de=True,
        seed=0,
    )
    result = facade.compute(layers, max_gpu_mem_bytes=budget)

    greedy_off = sum(t.size_bytes for ts in greedy.strategy_map.values() for t in ts)
    facade_off = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    greedy_score = evaluate_strategy_result(greedy, layers, "greedy", max_gpu_mem_bytes=budget)
    facade_score = evaluate_strategy_result(result, layers, "facade", max_gpu_mem_bytes=budget)

    assert greedy_score.peak_memory_bytes <= budget
    assert facade_score.peak_memory_bytes <= budget
    assert greedy_score.peak_memory_bytes == budget  # pinched
    assert greedy_off == 4 * _MB
    assert facade_off == 3 * _MB
    assert facade_off < greedy_off
    assert facade.selected_solver_name == "BudgetFillTensorDE"


def test_facade_meets_budget_like_greedy() -> None:
    from flextensor.strategy import BudgetFillStrategy

    layers = _layers([2 * _MB] * 8)
    budget = 6 * _MB
    facade = BudgetFillStrategy(n_blocks=2, enable_layer_de=True, enable_tensor_de=True)
    result = facade.compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget-fill", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert score.is_valid
    assert facade.selected_solver_name in {"BudgetFillGreedy", "BudgetFillLayerDE", "BudgetFillTensorDE"}
