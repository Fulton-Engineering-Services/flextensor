# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import logging
import types

import pytest
import torch

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.config import OffloadConfig
from flextensor.model_state_capture import LiveStorageKey
from flextensor.strategy import StrategyResult

from ._v2_test_utils import (
    InvalidBlockIdStrategy,
    MalformedResultStrategy,
    RecordingStateStrategy,
    _assert_value_only,
    _scan_loaded_model,
    _set_cuda_snapshot,
    _state_offloader,
)


def test_resolve_cuda_device_indexes_bare_cuda(bootstrap_module, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert bootstrap_module.state_builder.resolve_cuda_device("cuda") == torch.device("cuda:3")


@pytest.mark.parametrize("device", [2, "cuda:2", torch.device("cuda:2")])
def test_resolve_cuda_device_preserves_explicit_index(bootstrap_module, device) -> None:
    assert bootstrap_module.state_builder.resolve_cuda_device(device) == torch.device("cuda:2")


def test_resolve_cuda_device_rejects_cpu(bootstrap_module) -> None:
    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"CUDA.*cpu"):
        bootstrap_module.state_builder.resolve_cuda_device("cpu")


def _profile_with_timings(state):
    return dataclasses.replace(
        state,
        stats=[
            layer.model_copy(
                update={
                    "duration": float(index + 4),
                    "tensors": [
                        tensor.model_copy(update={"load_time_ms": float(index + 1) / 10}) for tensor in layer.tensors
                    ],
                }
            )
            for index, layer in enumerate(state.stats)
        ],
    )


def test_build_state_merges_only_profile_compute_durations_onto_fresh_tensor_ids(
    bootstrap_module,
    monkeypatch,
) -> None:
    current_memory_stats = {4: 0.1, 16: 0.4}
    monkeypatch.setattr(
        bootstrap_module,
        "benchmark_memory_transfers",
        lambda _stats, _device: current_memory_stats,
    )
    old_offloader, old_model = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)
    old_state = old_offloader.build_state(
        old_model,
        OffloadConfig(
            load_strategy=RecordingStateStrategy("strategy"),
            transfer_mode="strategy",
            pinned_memory=False,
        ),
        "cuda:0",
    )
    profile = _profile_with_timings(old_state)
    profile = dataclasses.replace(
        profile,
        stats=[
            layer.model_copy(
                update={
                    "tensors": [tensor.model_copy(update={"load_time_ms": float("nan")}) for tensor in layer.tensors]
                }
            )
            for layer in profile.stats
        ],
    )

    new_offloader, new_model = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy("strategy")
    new_state = new_offloader.build_state(
        new_model,
        OffloadConfig(load_strategy=strategy, transfer_mode="strategy", pinned_memory=False),
        "cuda:0",
        profile=profile,
    )

    planned_stats = strategy.calls[0][0]
    assert [layer.duration for layer in planned_stats] == [4.0, 5.0]
    assert [tensor.load_time_ms for layer in planned_stats for tensor in layer.tensors] == pytest.approx([0.2, 0.3])
    assert [tensor.load_time_ms for layer in new_state.stats for tensor in layer.tensors] == pytest.approx([0.2, 0.3])
    old_ids = set(profile.tensor_id_to_name_map)
    assert set(new_state.tensor_id_to_name_map).isdisjoint(old_ids)
    assert all(
        tensor.tensor_id in new_state.tensor_id_to_name_map
        for tensors in new_state.load_strategy.values()
        for tensor in tensors
    )
    new_state.validate_internal()


def test_build_state_rejects_incompatible_profile_and_uses_conservative_stats(
    bootstrap_module,
    monkeypatch,
    caplog,
) -> None:
    old_offloader, old_model = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)
    old_state = old_offloader.build_state(
        old_model,
        OffloadConfig(
            load_strategy=RecordingStateStrategy("strategy"),
            transfer_mode="strategy",
            pinned_memory=False,
        ),
        "cuda:0",
    )
    profile = _profile_with_timings(old_state)
    profile = dataclasses.replace(
        profile,
        stats=[profile.stats[0].model_copy(update={"label": "stale.layer"}), *profile.stats[1:]],
    )

    new_offloader, new_model = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy("strategy")
    with caplog.at_level(logging.WARNING, logger=bootstrap_module.LOGGER.name):
        new_offloader.build_state(
            new_model,
            OffloadConfig(load_strategy=strategy, transfer_mode="strategy", pinned_memory=False),
            "cuda:0",
            profile=profile,
        )

    assert [layer.duration for layer in strategy.calls[0][0]] == [1.0, 1.0]
    assert "saved profile is incompatible" in caplog.text


@pytest.mark.parametrize(
    "invalid_timing",
    [float("nan"), float("inf"), 0.0, -1.0],
    ids=["nan", "inf", "zero", "negative"],
)
def test_build_state_rejects_invalid_profile_compute_timings_and_uses_conservative_stats(
    bootstrap_module,
    monkeypatch,
    caplog,
    invalid_timing,
) -> None:
    old_offloader, old_model = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)
    old_state = old_offloader.build_state(
        old_model,
        OffloadConfig(
            load_strategy=RecordingStateStrategy("strategy"),
            transfer_mode="strategy",
            pinned_memory=False,
        ),
        "cuda:0",
    )
    profile = _profile_with_timings(old_state)
    layer = profile.stats[0]
    layer = layer.model_copy(update={"duration": invalid_timing})
    profile = dataclasses.replace(profile, stats=[layer, *profile.stats[1:]])

    new_offloader, new_model = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy("strategy")
    with caplog.at_level(logging.WARNING, logger=bootstrap_module.LOGGER.name):
        new_offloader.build_state(
            new_model,
            OffloadConfig(load_strategy=strategy, transfer_mode="strategy", pinned_memory=False),
            "cuda:0",
            profile=profile,
        )

    planned_stats = strategy.calls[0][0]
    assert [layer.duration for layer in planned_stats] == [1.0, 1.0]
    assert [tensor.load_time_ms for layer in planned_stats for tensor in layer.tensors] == pytest.approx([0.1, 0.1])
    assert "saved profile is incompatible" in caplog.text


def test_build_state_requires_successful_bootstrap_post_init(bootstrap_module, monkeypatch) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch, complete=False)
    strategy = RecordingStateStrategy("strategy")

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="post_init"):
        offloader.build_state(
            transformed,
            OffloadConfig(load_strategy=strategy, pinned_memory=False),
            "cuda:0",
        )

    assert strategy.calls == []


def test_build_state_rejects_second_build(bootstrap_module, monkeypatch) -> None:
    offloader, root = _state_offloader(bootstrap_module, monkeypatch)
    config = OffloadConfig(
        load_strategy=RecordingStateStrategy("strategy"),
        transfer_mode="strategy",
        pinned_memory=False,
    )
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)
    offloader.build_state(root, config, "cuda:0")

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="state already built"):
        offloader.build_state(root, config, "cuda:0")


@pytest.mark.parametrize(
    "loader_type",
    ["strategy", "allocation_block_transfer", "raw_block_transfer"],
)
def test_build_state_supports_all_transfer_modes(
    bootstrap_module,
    monkeypatch,
    loader_type,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy(loader_type)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    state = offloader.build_state(
        transformed,
        OffloadConfig(
            load_strategy=strategy,
            transfer_mode=loader_type,
            pinned_memory=False,
        ),
        "cuda:0",
    )

    assert len(strategy.calls) == 1
    assert state.loader_type == loader_type
    assert state.tensor_id_to_name_map[id(transformed.second.weight)] == "second.weight"
    assert state.load_strategy["first"][0].tensor_id == id(transformed.second.weight)
    state.validate_internal()
    _assert_value_only(state)
    if loader_type == "strategy":
        assert state.release_strategy
        assert state.allocation_ordered == {}
    else:
        assert state.release_strategy == {}
        assert state.allocation_ordered == {0: ["first"]}
        assert state.view_tensors_ids == [id(transformed.second.weight)]


def test_build_state_applies_supplied_memory_stats_before_strategy(bootstrap_module, monkeypatch) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    scan = _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda:0")
    strategy = RecordingStateStrategy("strategy")
    memory_stats = {4: 0.1, 16: 0.4}
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    state = bootstrap_module.state_builder.build_conservative_state(
        scan,
        OffloadConfig(load_strategy=strategy, transfer_mode="strategy", pinned_memory=False),
        torch.device("cuda:0"),
        memory_stats=memory_stats,
    )

    assert strategy.calls[0][1] is memory_stats
    assert [tensor.load_time_ms for layer in strategy.calls[0][0] for tensor in layer.tensors] == pytest.approx([
        0.2,
        0.3,
    ])
    assert [tensor.load_time_ms for layer in state.stats for tensor in layer.tensors] == pytest.approx([0.2, 0.3])
    assert state.load_strategy["first"][0].load_time_ms == pytest.approx(0.3)


@pytest.mark.parametrize(
    "memory_stats",
    [{}, {0: 0.1}, {4: 0.0}, {4: float("nan")}],
    ids=["empty", "non-positive-size", "non-positive-duration", "non-finite-duration"],
)
def test_build_state_rejects_invalid_memory_stats_before_strategy(
    bootstrap_module,
    monkeypatch,
    memory_stats,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    scan = _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda:0")
    strategy = RecordingStateStrategy("strategy")
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="invalid memory transfer benchmark"):
        bootstrap_module.state_builder.build_conservative_state(
            scan,
            OffloadConfig(load_strategy=strategy, transfer_mode="strategy", pinned_memory=False),
            torch.device("cuda:0"),
            memory_stats=memory_stats,
        )

    assert strategy.calls == []


def test_compute_strategy_result_returns_canonical_result_and_selected_ids(bootstrap_module) -> None:
    selected = TensorStatistics(
        tensor_id=2,
        name="second.weight",
        size_bytes=16,
        load_time_ms=1.0,
    )
    layer_stats = [
        LayerStatistics(label="first", tensors=[], duration=1.0),
        LayerStatistics(label="second", tensors=[selected], duration=1.0),
    ]
    strategy = RecordingStateStrategy("strategy")

    result, selected_ids = bootstrap_module.state_builder._compute_strategy_result(
        strategy,
        layer_stats,
        {16: 1.0},
        64,
        loader_type="strategy",
        canonical_by_id={2: selected},
    )

    assert selected_ids == {2}
    assert result.strategy_map == {"first": [selected]}
    assert result.strategy_map["first"][0] is selected


def test_build_state_rejects_a_fourth_transfer_mode(bootstrap_module, monkeypatch) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="unsupported"):
        offloader.build_state(
            transformed,
            OffloadConfig(load_strategy=RecordingStateStrategy("future"), transfer_mode="future"),
            "cuda:0",
        )


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("unknown_label", "unknown labels"),
        ("unknown_id", "absent or ineligible"),
        ("conflicting_metadata", "metadata"),
    ],
)
def test_build_state_rejects_unknown_or_conflicting_strategy_results(
    bootstrap_module,
    monkeypatch,
    failure,
    match,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=match):
        offloader.build_state(
            transformed,
            OffloadConfig(load_strategy=RecordingStateStrategy("strategy", failure=failure)),
            "cuda:0",
        )


def test_build_state_rejects_malformed_strategy_result(bootstrap_module, monkeypatch) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="malformed result"):
        offloader.build_state(
            transformed,
            OffloadConfig(load_strategy=MalformedResultStrategy()),
            "cuda:0",
        )


@pytest.mark.parametrize(
    ("failure", "match"),
    [
        ("missing_block_data", "block_data"),
        ("wrong_block_types", "malformed block_data"),
        ("label_size_mismatch", "label_to_size_map"),
        ("undersized_block", "block_sizes"),
        ("missing_allocation", "allocation_ordered"),
        ("duplicate_allocation", "allocation_ordered"),
        ("inconsistent_block_id", "label_to_block_id"),
        ("transfer_mapping_mismatch", "transfer_to_compute_map"),
    ],
)
def test_build_state_validates_all_block_maps(
    bootstrap_module,
    monkeypatch,
    failure,
    match,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=match):
        offloader.build_state(
            transformed,
            OffloadConfig(
                load_strategy=RecordingStateStrategy("raw_block_transfer", failure=failure),
                transfer_mode="raw_block_transfer",
            ),
            "cuda:0",
        )


@pytest.mark.parametrize("invalid_block_id", [-1, False])
@pytest.mark.parametrize("surface", ["allocation_ordered", "label_to_block_id", "block_sizes"])
def test_build_state_rejects_non_exact_non_negative_block_ids(
    bootstrap_module,
    monkeypatch,
    surface,
    invalid_block_id,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=rf"malformed block_data.*{surface}"):
        offloader.build_state(
            transformed,
            OffloadConfig(
                load_strategy=InvalidBlockIdStrategy(surface, invalid_block_id),
                transfer_mode="raw_block_transfer",
            ),
            "cuda:0",
        )


@pytest.mark.parametrize(
    ("max_gpu_mem_fraction", "expected_strategy_budget"),
    [(0.1, 80), (0.7, 680)],
)
def test_build_state_caps_reclaimed_gpu_budget_and_deduplicates_constants(
    bootstrap_module,
    monkeypatch,
    caplog,
    max_gpu_mem_fraction,
    expected_strategy_budget,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy("strategy")
    stats = [
        LayerStatistics(
            label=label,
            tensors=[
                TensorStatistics(
                    tensor_id=tensor_id,
                    name=f"{label}.weight",
                    size_bytes=size,
                    load_time_ms=1.0,
                )
            ],
            duration=1.0,
        )
        for label, tensor_id, size in (("first", 1, 30), ("second", 2, 40))
    ]
    monkeypatch.setattr(
        bootstrap_module.model_scan,
        "scan_loaded_model",
        lambda *_args, **_kwargs: bootstrap_module.model_scan.LoadedModelScan(
            layer_stats=stats,
            name_by_tensor_id={1: "first.weight", 2: "second.weight", 3: "constant", 4: "constant_alias"},
            storage_by_tensor_id={
                1: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cuda:0"), storage_impl_id=10), 30, False
                ),
                2: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cpu"), storage_impl_id=20), 40, False
                ),
                3: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cuda:0"), storage_impl_id=30), 20, False
                ),
                4: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cuda:0"), storage_impl_id=30), 20, False
                ),
            },
            gpu_constant_ids=frozenset({3, 4}),
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=800, total=1000)
    caplog.set_level("INFO", logger="vllm.flextensor.v2.state_builder")

    state = offloader.build_state(
        transformed,
        OffloadConfig(
            load_strategy=strategy,
            transfer_mode="strategy",
            max_gpu_mem_fraction=max_gpu_mem_fraction,
        ),
        "cuda",
    )

    assert strategy.calls[0][2] == expected_strategy_budget
    assert (
        "FlexTensor v2 GPU budget resolved: "
        f"whole_model_budget_bytes={expected_strategy_budget + 20} managed_gpu_resident_bytes=50"
    ) in caplog.messages
    state.validate_internal()


def test_build_state_charges_only_new_unique_host_storage(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy("strategy")
    stats = [
        LayerStatistics(
            label=label,
            tensors=[
                TensorStatistics(
                    tensor_id=tensor_id,
                    name=f"{label}.weight",
                    size_bytes=size,
                    load_time_ms=1.0,
                )
            ],
            duration=1.0,
        )
        for label, tensor_id, size in (("first", 1, 30), ("second", 2, 30), ("third", 3, 40))
    ]
    strategy.compute = lambda layer_stats, memory_stats=None, max_gpu_mem_bytes=None: (
        strategy.calls.append((layer_stats, memory_stats, max_gpu_mem_bytes))
        or StrategyResult(strategy_map={"first": [layer.tensors[0] for layer in layer_stats]})
    )
    monkeypatch.setattr(
        bootstrap_module.model_scan,
        "scan_loaded_model",
        lambda *_args, **_kwargs: bootstrap_module.model_scan.LoadedModelScan(
            layer_stats=stats,
            name_by_tensor_id={1: "first.weight", 2: "second.weight", 3: "third.weight"},
            storage_by_tensor_id={
                1: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cuda:0"), storage_impl_id=10), 30, False
                ),
                2: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cuda:0"), storage_impl_id=10), 30, False
                ),
                3: bootstrap_module.model_scan.LiveStorageInfo(
                    LiveStorageKey(device=torch.device("cpu"), storage_impl_id=20), 40, False
                ),
            },
            gpu_constant_ids=frozenset(),
        ),
    )
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)
    monkeypatch.setattr(
        bootstrap_module.state_builder,
        "psutil",
        types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=29)),
    )

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"required=30.*available=29"):
        offloader.build_state(
            transformed,
            OffloadConfig(load_strategy=strategy, transfer_mode="strategy"),
            "cuda:0",
        )
