# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy import BudgetFillStrategy
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
    strategy = BudgetFillStrategy(n_blocks=2)
    with pytest.raises(StrategyComputeError, match="max_gpu_mem_bytes"):
        strategy.compute(_layers([_MB, _MB, _MB, _MB]), memory_stats={_MB: 1.0})


def test_requires_two_layers() -> None:
    strategy = BudgetFillStrategy(n_blocks=2)
    with pytest.raises(StrategyComputeError, match="two layers"):
        strategy.compute(
            [LayerStatistics(label="only", tensors=[_tensor(1, _MB)], duration=1.0)],
            max_gpu_mem_bytes=10 * _MB,
        )


def test_no_offload_when_model_fits() -> None:
    layers = _layers([_MB, _MB, _MB, _MB])
    result = BudgetFillStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=10 * _MB)
    assert result.strategy_map == {}
    offloaded = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    assert offloaded == 0


def test_fills_budget_under_limit() -> None:
    # 16 MiB model; with 2 blocks peak can reach ~6 MiB after offloading layers 1..n-1.
    layers = _layers([2 * _MB] * 8)
    memory_stats = {2 * _MB: 0.5}
    budget = 6 * _MB
    result = BudgetFillStrategy(n_blocks=2).compute(layers, memory_stats, budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    offloaded = sum(t.size_bytes for ts in result.strategy_map.values() for t in ts)
    assert offloaded > 0
    assert score.peak_memory_bytes <= budget
    assert score.is_valid
    # First-layer tensor stays resident (never appears in strategy_map values).
    first_id = layers[0].tensors[0].tensor_id
    selected = {t.tensor_id for ts in result.strategy_map.values() for t in ts}
    assert first_id not in selected


def test_spreads_offload_across_layers() -> None:
    """Partial budget fill should touch every pipelinable layer, not a few fat ones."""
    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, 2 * _MB)], duration=1.0),
        LayerStatistics(
            label="l1",
            tensors=[_tensor(2, 3 * _MB), _tensor(3, 3 * _MB)],
            duration=1.0,
        ),
        LayerStatistics(
            label="l2",
            tensors=[_tensor(4, 3 * _MB), _tensor(5, 3 * _MB)],
            duration=1.0,
        ),
        LayerStatistics(
            label="l3",
            tensors=[_tensor(6, 3 * _MB), _tensor(7, 3 * _MB)],
            duration=1.0,
        ),
        LayerStatistics(
            label="l4",
            tensors=[_tensor(8, 3 * _MB), _tensor(9, 3 * _MB)],
            duration=1.0,
        ),
    ]
    budget = 14 * _MB
    result = BudgetFillStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget

    selected = {t.tensor_id for ts in result.strategy_map.values() for t in ts}
    for layer in layers[1:]:
        assert any(t.tensor_id in selected for t in layer.tensors), layer.label

    assert len(result.strategy_map) == 4


def test_stops_near_budget_without_over_offloading() -> None:
    layers = _layers([_MB] + [4 * _MB] * 4)
    budget = 10 * _MB
    result = BudgetFillStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert budget - score.peak_memory_bytes < 4 * _MB


def test_warns_when_budget_unreachable() -> None:
    layers = _layers([20 * _MB, _MB, _MB, _MB])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = BudgetFillStrategy(n_blocks=2).compute(layers, max_gpu_mem_bytes=5 * _MB)
    assert any("could not meet" in str(item.message) for item in caught)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=5 * _MB)
    assert score.peak_memory_bytes > 5 * _MB


def test_within_layer_prefers_fit_then_largest() -> None:
    """Fit under ``duration * scale`` ranks before size; among fits, larger first."""
    from flextensor.strategy.budget_fill import _eligible_by_layer

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
    by_layer = _eligible_by_layer(layers, threshold_mb=0.1, scale=1.0, interpolator=None)
    assert len(by_layer) == 1
    assert [t.tensor_id for t in by_layer[0].tensors] == [4, 2, 3]


def test_scale_changes_fit_ranking() -> None:
    """Larger scale widens the window so a previously non-fitting tensor ranks first."""
    from flextensor.strategy.budget_fill import _eligible_by_layer, _select_ids_for_cap

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
    tight = _eligible_by_layer(layers, threshold_mb=0.1, scale=1.0, interpolator=None)
    assert [t.tensor_id for t in tight[0].tensors] == [2, 3]
    assert _select_ids_for_cap(tight, 2 * _MB) == {2}

    wide = _eligible_by_layer(layers, threshold_mb=0.1, scale=3.0, interpolator=None)
    # Both fit; larger (id 3) ranks first.
    assert [t.tensor_id for t in wide[0].tensors] == [3, 2]
    assert _select_ids_for_cap(wide, 4 * _MB) == {3}


def test_searches_min_to_max_blocks_and_can_pick_two() -> None:
    """With n_blocks=4, BudgetFill may select 2 blocks when that lowers peak."""
    layers = _layers([2 * _MB] * 8)
    budget = 6 * _MB
    result = BudgetFillStrategy(n_blocks=4, min_blocks=2).compute(layers, max_gpu_mem_bytes=budget)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert result.block_data is not None
    assert len(result.block_data.block_sizes) == 2


def test_rejects_invalid_min_blocks() -> None:
    with pytest.raises(ValueError, match="min_blocks"):
        BudgetFillStrategy(n_blocks=4, min_blocks=1)
    with pytest.raises(ValueError, match="min_blocks"):
        BudgetFillStrategy(n_blocks=2, min_blocks=3)


def test_rejects_n_blocks_one() -> None:
    """Single-block pipelines are unsupported; use 0 or >= 2."""
    with pytest.raises(ValueError, match="n_blocks"):
        BudgetFillStrategy(n_blocks=1)


def test_size_aware_blocks_meet_budget_label_balance_misses() -> None:
    """Regression: feasibility must use size-aware block packing.

    Transfer sizes [2,5,2,1,5] MiB, 1 MiB resident first layer, 3 blocks, 9 MiB
    budget. Count-balanced blocks sum to 9 MiB (peak 10); size-aware can sum to
    8 MiB and meet the budget.
    """
    layers = [
        LayerStatistics(label="l0", tensors=[_tensor(1, _MB)], duration=1.0),
        LayerStatistics(label="l1", tensors=[_tensor(2, 2 * _MB)], duration=1.0),
        LayerStatistics(label="l2", tensors=[_tensor(3, 5 * _MB)], duration=1.0),
        LayerStatistics(label="l3", tensors=[_tensor(4, 2 * _MB)], duration=1.0),
        LayerStatistics(label="l4", tensors=[_tensor(5, 1 * _MB)], duration=1.0),
        LayerStatistics(label="l5", tensors=[_tensor(6, 5 * _MB)], duration=1.0),
    ]
    budget = 9 * _MB
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = BudgetFillStrategy(n_blocks=3, min_blocks=3).compute(
            layers,
            max_gpu_mem_bytes=budget,
        )
    assert not any("could not meet" in str(item.message) for item in caught)
    score = evaluate_strategy_result(result, layers, "budget", max_gpu_mem_bytes=budget)
    assert score.peak_memory_bytes <= budget
    assert score.is_valid
    assert result.block_data is not None
    block_total = sum(result.block_data.block_sizes.values())
    assert block_total <= 8 * _MB
