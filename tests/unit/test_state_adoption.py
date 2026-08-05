# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for read-only planning of saved-state adoption."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import flextensor.state_transition as state_adoption_module
from flextensor import state_handler as state_handler_module
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.model_state_capture import capture_model_state
from flextensor.state_adoption import target_from_profile
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler
from flextensor.state_transition import (
    MemoryCapacity,
    StateTransitionPlan,
    StorageMigration,
    plan_state_transition,
)


def _plan_profile_transition(
    settings: object,
    model: torch.nn.Module,
    state: TensorManagerState,
    *,
    host_available_bytes: int,
    host_reserve_bytes: int,
    gpu_available_bytes: int,
    gpu_reserve_bytes: int,
) -> StateTransitionPlan:
    """Exercise the old scenarios through the new explicit adapter chain."""
    configured_loader = getattr(settings, "loader_type", state.loader_type)
    if configured_loader != state.loader_type:
        raise ValueError(
            f"Saved state uses loader_type='{state.loader_type}' but runtime uses '{configured_loader}'. "
            "Re-profile or use a matching transfer mode."
        )
    current = capture_model_state(model)
    configured_device = getattr(settings, "device_gpu", None)
    cuda_devices = {storage.device for storage in current.storages if storage.device.startswith("cuda:")}
    if configured_device is not None:
        target_device = str(configured_device)
    elif len(cuda_devices) == 1:
        target_device = next(iter(cuda_devices))
    elif len(cuda_devices) > 1:
        raise ValueError(f"State adoption supports one CUDA device, got {sorted(cuda_devices)}")
    else:
        target_device = "cuda:0"
    pinned_memory = bool(getattr(settings, "pinned_memory", False))
    registry = getattr(getattr(settings, "host_pinner", None), "registry", None)
    pinning = "none" if not pinned_memory else "in_place" if registry is not None else "copy"
    target, transition = target_from_profile(
        current,
        state,
        target_device=target_device,
        pinning=pinning,
        use_shm=bool(getattr(settings, "use_shm", False)),
    )
    return plan_state_transition(
        current,
        target,
        transition=transition,
        capacity=MemoryCapacity(
            host_bytes=max(0, host_available_bytes - host_reserve_bytes),
            gpu_bytes=max(0, gpu_available_bytes - gpu_reserve_bytes),
        ),
    )


def _stat(name: str, tensor: torch.Tensor) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=id(tensor),
        name=name,
        size_bytes=tensor.numel() * tensor.element_size(),
        load_time_ms=0.1,
    )


def _state_for(
    model: torch.nn.Module,
    *,
    load_names: tuple[str, ...],
    view_names: tuple[str, ...] = (),
) -> TensorManagerState:
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    inventory = tuple(tensors)
    load_stats = [_stat(name, tensors[name]) for name in load_names]
    managed_names = {*load_names, *view_names}
    gpu_names = [name for name in inventory if name not in managed_names]
    return TensorManagerState(
        loader_type="strategy",
        tensor_id_to_name_map={id(tensor): name for name, tensor in tensors.items()},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"layer": load_stats},
        release_strategy={"layer": load_stats},
        label_to_block_id={},
        stats=[LayerStatistics(label="layer", tensors=load_stats, duration=1.0)],
        transfer_to_compute_map={},
        view_tensors_ids=[id(tensors[name]) for name in view_names],
        view_tensors_names=list(view_names),
        gpu_tensors_names=gpu_names,
        shm_block_name_map=None,
    )


def _snapshot(model: torch.nn.Module) -> dict[str, tuple[int, torch.device, tuple[str, int | None, int, int]]]:
    tensors = list(model.named_parameters(remove_duplicate=False))
    tensors.extend(model.named_buffers(remove_duplicate=False))
    return {
        name: (
            id(tensor),
            tensor.device,
            (
                tensor.device.type,
                tensor.device.index,
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
            ),
        )
        for name, tensor in tensors
    }


def _single_storage_migration_plan(model: torch.nn.Module, names: tuple[str, ...]) -> StateTransitionPlan:
    storage = capture_model_state(model).storages[0]
    return StateTransitionPlan(
        migrations=(StorageMigration(storage.id, names, storage.device, "cuda:0", storage.nbytes),),
        pinning_groups=(),
        peak_host_bytes=0,
        peak_gpu_bytes=storage.nbytes,
    )


def test_plan_state_adoption_returns_read_only_cpu_plan() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    state = _state_for(model, load_names=("weight",))

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.migrations == ()
    assert plan.peak_host_bytes == 0
    assert plan.peak_gpu_bytes == model.weight.untyped_storage().nbytes()


def test_prepare_state_adds_registered_buffers_without_mutating_parameter_mapping() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    model.register_buffer("constant", torch.ones(3))
    parameter_mapping = {id(model.weight): "weight"}
    tensor_manager = SimpleNamespace(
        loader_type="strategy",
        model=model,
        tensor_id_to_name_map=parameter_mapping.copy(),
    )
    weight_stat = _stat("", model.weight)

    state = TensorManagerStateHandler(tensor_manager).prepare_state(
        loader_type="strategy",
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"layer": [weight_stat]},
        release_strategy={"layer": [weight_stat]},
        label_to_block_id={},
        stats=[LayerStatistics(label="layer", tensors=[weight_stat], duration=1.0)],
        transfer_to_compute_map={},
        shm_block_name_map=None,
    )

    assert tensor_manager.tensor_id_to_name_map == parameter_mapping
    assert state.tensor_id_to_name_map == {
        id(model.weight): "weight",
        id(model.constant): "constant",
    }
    assert set(state.gpu_tensors_names) == {"constant"}


def test_plan_canonicalizes_same_parameter_reachable_by_two_names() -> None:
    shared = torch.nn.Parameter(torch.arange(4.0), requires_grad=False)
    model = torch.nn.Module()
    model.left = torch.nn.Module()
    model.right = torch.nn.Module()
    model.left.register_parameter("weight", shared)
    model.right.register_parameter("weight", shared)
    state = _state_for(model, load_names=("left.weight",))
    state.tensor_id_to_name_map = {id(shared): "left.weight"}
    state.gpu_tensors_names = []

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.migrations == ()


def test_target_from_profile_ignores_unnamed_tensor_order() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    model.workspace = torch.arange(2.0)
    state = _state_for(model, load_names=("weight",))
    current = capture_model_state(model)
    current = type(current)(
        tensors=tuple(sorted(current.tensors, key=lambda tensor: bool(tensor.names))),
        storages=current.storages,
    )

    target, _transition = target_from_profile(
        current,
        state,
        target_device="cuda:0",
        pinning="none",
        use_shm=False,
    )

    storage_by_id = {storage.id: storage for storage in target.storages}
    weight_storage_id = next(tensor.storage_id for tensor in current.tensors if tensor.names == ("weight",))
    workspace_storage_id = next(tensor.storage_id for tensor in current.tensors if not tensor.names)
    assert storage_by_id[weight_storage_id].device == "cpu"
    assert storage_by_id[workspace_storage_id].device == "cuda:0"


def test_plan_accepts_logical_statistics_for_view_with_larger_backing_storage() -> None:
    model = torch.nn.Module()
    backing = torch.arange(8.0)
    model.register_parameter("view", torch.nn.Parameter(backing[2:6], requires_grad=False))
    state = _state_for(model, load_names=("view",))

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert state.load_strategy["layer"][0].size_bytes == 4 * backing.element_size()
    assert model.view.untyped_storage().nbytes() == 8 * backing.element_size()
    assert plan.peak_gpu_bytes == state.load_strategy["layer"][0].size_bytes


def _two_managed_tensor_state() -> tuple[torch.nn.Module, TensorManagerState, TensorStatistics, TensorStatistics]:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    model.register_parameter("second", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    state = _state_for(model, load_names=("first", "second"))
    return model, state, _stat("first", model.first), _stat("second", model.second)


def test_strategy_plan_rejects_overlapping_load_residency_before_mutation() -> None:
    model, state, first_stat, second_stat = _two_managed_tensor_state()
    state.load_strategy = {"load_first": [first_stat], "load_second": [second_stat]}
    state.release_strategy = {"release_first": [first_stat], "release_second": [second_stat]}
    state.stats = [
        LayerStatistics(label="load_first", tensors=[first_stat], duration=1.0),
        LayerStatistics(label="load_second", tensors=[second_stat], duration=1.0),
        LayerStatistics(label="release_first", tensors=[], duration=1.0),
        LayerStatistics(label="release_second", tensors=[], duration=1.0),
    ]
    gpu_capacity = first_stat.size_bytes + second_stat.size_bytes // 2
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(RuntimeError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=gpu_capacity,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert "insufficient capacity" in message
    assert "gpu" in message
    assert "required=" in message
    assert f"available={gpu_capacity}" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_strategy_plan_applies_scheduled_release_for_capacity() -> None:
    model, state, first_stat, second_stat = _two_managed_tensor_state()
    state.load_strategy = {"first": [first_stat], "second": [second_stat]}
    state.release_strategy = {"first": [first_stat], "second": [second_stat]}
    state.stats = [
        LayerStatistics(label="first", tensors=[first_stat], duration=1.0),
        LayerStatistics(label="second", tensors=[second_stat], duration=1.0),
    ]
    gpu_capacity = max(first_stat.size_bytes, second_stat.size_bytes)
    before = _snapshot(model)
    state_before = state.to_dict()

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=gpu_capacity,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_gpu_bytes == gpu_capacity
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


@pytest.mark.parametrize("resident_kind", ["preload", "rescue"])
def test_strategy_plan_counts_constructor_resident_tensor_before_other_load(
    resident_kind: str,
) -> None:
    model, state, resident_stat, loaded_stat = _two_managed_tensor_state()
    state.load_strategy = {"unused_resident_load": [resident_stat], "load_other": [loaded_stat]}
    state.release_strategy = {"release_other": [loaded_stat], "release_resident": [resident_stat]}
    state.stats = [
        *(
            [LayerStatistics(label="first_use", tensors=[resident_stat], duration=1.0)]
            if resident_kind == "preload"
            else []
        ),
        LayerStatistics(label="load_other", tensors=[loaded_stat], duration=1.0),
        LayerStatistics(label="release_other", tensors=[], duration=1.0),
        LayerStatistics(label="release_resident", tensors=[], duration=1.0),
    ]
    gpu_capacity = resident_stat.size_bytes + loaded_stat.size_bytes // 2
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(RuntimeError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=gpu_capacity,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert "insufficient capacity" in message
    assert "gpu" in message
    assert "required=" in message
    assert f"available={gpu_capacity}" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def _cpu_model_and_state() -> tuple[torch.nn.Module, TensorManagerState]:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    return model, _state_for(model, load_names=("weight",))


def test_plan_rejects_loader_type_mismatch_before_mutation() -> None:
    model, state = _cpu_model_and_state()
    tensor_manager = SimpleNamespace(loader_type="allocation_block_transfer")
    before = _snapshot(model)

    with pytest.raises(ValueError) as exc_info:
        _plan_profile_transition(
            tensor_manager,
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value)
    assert "strategy" in message
    assert "allocation_block_transfer" in message
    assert "re-profile" in message or "matching transfer mode" in message
    assert _snapshot(model) == before


@pytest.mark.parametrize("declared_device", [torch.device("cpu"), torch.device("cuda")])
def test_plan_rejects_declared_device_gpu_that_is_not_explicit_cuda(declared_device: torch.device) -> None:
    model, state = _cpu_model_and_state()
    tensor_manager = SimpleNamespace(loader_type="strategy", device_gpu=declared_device)
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(ValueError) as exc_info:
        _plan_profile_transition(
            tensor_manager,
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert "target_device" in message
    assert "explicit" in message
    assert "cuda" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


class _SelectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.managed = torch.nn.Linear(4, 4, bias=False)
        self.sibling = torch.nn.Linear(4, 4, bias=False)


def _selection_state(
    *, load_names: tuple[str, ...] = ("managed.weight",)
) -> tuple[_SelectionModel, TensorManagerState]:
    model = _SelectionModel()
    return model, _state_for(model, load_names=load_names)


def test_saved_profile_is_authoritative_over_runtime_selection_config() -> None:
    model, state = _selection_state()
    managed_stats = state.load_strategy.pop("layer")
    state.load_strategy = {"manual.trap": managed_stats}
    state.release_strategy = {"manual.trap": managed_stats}
    state.stats = [LayerStatistics(label="manual.trap", tensors=managed_stats, duration=1.0)]
    tensor_manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        include_patterns=[],
        exclude_patterns=["managed"],
    )
    before = _snapshot(model)
    state_before = state.to_dict()

    plan = _plan_profile_transition(
        tensor_manager,
        model,
        state,
        host_available_bytes=4096,
        host_reserve_bytes=0,
        gpu_available_bytes=4096,
        gpu_reserve_bytes=0,
    )

    assert plan.migrations[0].names == ("sibling.weight",)
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_plan_does_not_require_loaded_tensor_prefix_to_match_selected_trap() -> None:
    model, state = _selection_state(load_names=("sibling.weight",))
    sibling_stats = state.load_strategy.pop("layer")
    state.load_strategy = {"managed": sibling_stats}
    state.release_strategy = {"managed": sibling_stats}
    state.stats = [LayerStatistics(label="managed", tensors=sibling_stats, duration=1.0)]
    tensor_manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        include_patterns=["*"],
        exclude_patterns=[],
    )
    before = _snapshot(model)
    state_before = state.to_dict()

    plan = _plan_profile_transition(
        tensor_manager,
        model,
        state,
        host_available_bytes=4096,
        host_reserve_bytes=0,
        gpu_available_bytes=4096,
        gpu_reserve_bytes=0,
    )

    assert plan.migrations[0].names == ("managed.weight",)
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_plan_rejects_partial_nonempty_strategy_release_map() -> None:
    model, state = _selection_state(load_names=("managed.weight", "sibling.weight"))
    released_stat = next(stat for stat in state.release_strategy["layer"] if stat.name == "managed.weight")
    state.release_strategy = {"layer": [released_stat]}
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(ValueError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=4096,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert "release_strategy" in message
    assert "sibling.weight" in message
    assert "re-profile" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_plan_preserves_empty_strategy_release_fallback() -> None:
    model, state = _selection_state(load_names=("managed.weight", "sibling.weight"))
    state.release_strategy = {}
    before = _snapshot(model)
    state_before = state.to_dict()

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=4096,
        host_reserve_bytes=0,
        gpu_available_bytes=4096,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_gpu_bytes == sum(stat.size_bytes for stat in state.load_strategy["layer"])
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def _cpu_block_state(loader_type: str) -> tuple[torch.nn.Module, TensorManagerState, int]:
    model, state = _cpu_model_and_state()
    logical_bytes = state.load_strategy["layer"][0].size_bytes
    state.loader_type = loader_type
    state.release_strategy = {}
    state.view_tensors_ids = [id(model.weight)]
    state.view_tensors_names = ["weight"]
    state.allocation_ordered = {0: ["layer"]}
    state.label_to_size_map = {"layer": logical_bytes}
    state.block_sizes = {0: logical_bytes}
    state.label_to_block_id = {"layer": 0}
    state.transfer_to_compute_map = {"layer": "layer"}
    return model, state, logical_bytes


def test_block_plan_counts_unregistered_cpu_tensor_finalization() -> None:
    model, state, _logical_bytes = _cpu_block_state("allocation_block_transfer")
    tensor_manager = SimpleNamespace(
        loader_type="allocation_block_transfer",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=False,
        use_shm=False,
    )
    baseline = _plan_profile_transition(
        tensor_manager,
        model,
        state,
        host_available_bytes=4096,
        host_reserve_bytes=0,
        gpu_available_bytes=4096,
        gpu_reserve_bytes=0,
    )
    model.workspace = torch.empty(8, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="gpu"):
        _plan_profile_transition(
            tensor_manager,
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=baseline.peak_gpu_bytes + model.workspace.nbytes - 1,
            gpu_reserve_bytes=0,
        )


@pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
def test_block_plan_rejects_view_tensor_names_not_owned_by_load_strategy(loader_type: str) -> None:
    model, state = _selection_state()
    load_bytes = state.load_strategy["layer"][0].size_bytes
    state.loader_type = loader_type
    state.release_strategy = {}
    state.view_tensors_ids = [id(model.managed.weight), id(model.sibling.weight)]
    state.view_tensors_names = ["managed.weight", "sibling.weight"]
    state.gpu_tensors_names = []
    state.allocation_ordered = {0: ["layer"]}
    state.label_to_size_map = {"layer": load_bytes} if loader_type == "raw_block_transfer" else {}
    state.block_sizes = {0: load_bytes}
    state.label_to_block_id = {"layer": 0}
    state.transfer_to_compute_map = {"layer": "layer"}
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(ValueError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=4096,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert "view_tensors_names" in message
    assert "load_strategy" in message
    assert "sibling.weight" in message
    assert "re-profile" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


@pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
@pytest.mark.parametrize(
    ("mismatch", "expected_field"),
    [
        ("missing-allocation", "allocation_ordered"),
        ("duplicate-allocation", "allocation_ordered"),
        ("block-id-disagreement", "label_to_block_id"),
        ("label-size", "label_to_size_map"),
        ("block-size", "block_sizes"),
        ("unknown-label", "unknown"),
    ],
)
def test_plan_rejects_inconsistent_block_loader_maps(
    loader_type: str,
    mismatch: str,
    expected_field: str,
) -> None:
    model, state, logical_bytes = _cpu_block_state(loader_type)
    if mismatch == "missing-allocation":
        state.allocation_ordered = {}
    elif mismatch == "duplicate-allocation":
        state.allocation_ordered = {0: ["layer"], 1: ["layer"]}
        state.block_sizes[1] = logical_bytes
    elif mismatch == "block-id-disagreement":
        state.label_to_block_id["layer"] = 1
    elif mismatch == "label-size":
        state.label_to_size_map["layer"] = logical_bytes - 1
    elif mismatch == "block-size":
        state.block_sizes[0] = logical_bytes - 1
    else:
        state.allocation_ordered[0].append("unknown")
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(ValueError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=4096,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert expected_field in message
    assert "re-profile" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


@pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
def test_block_plan_rejects_tensor_owned_by_multiple_load_labels(loader_type: str) -> None:
    model, state, logical_bytes = _cpu_block_state(loader_type)
    weight_stat = state.load_strategy["layer"][0]
    state.load_strategy = {"first": [weight_stat], "second": [weight_stat]}
    state.stats = [
        LayerStatistics(label="first", tensors=[weight_stat], duration=1.0),
        LayerStatistics(label="second", tensors=[weight_stat], duration=1.0),
    ]
    state.allocation_ordered = {0: ["first", "second"]}
    state.label_to_size_map = (
        {"first": logical_bytes, "second": logical_bytes} if loader_type == "raw_block_transfer" else {}
    )
    state.block_sizes = {0: logical_bytes}
    state.label_to_block_id = {"first": 0, "second": 0}
    state.transfer_to_compute_map = {"first": "first", "second": "second"}
    before = _snapshot(model)

    with pytest.raises(ValueError, match=r"multiple.*load labels") as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=4096,
            gpu_reserve_bytes=0,
        )

    assert "weight" in str(exc_info.value)
    assert _snapshot(model) == before


def test_allocation_block_plan_counts_128_byte_packing_before_mutation() -> None:
    model, state, logical_bytes = _cpu_block_state("allocation_block_transfer")
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(RuntimeError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=127,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert logical_bytes == 16
    assert "gpu" in message
    assert "required=128" in message
    assert "available=127" in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_raw_block_plan_keeps_logical_block_capacity() -> None:
    model, state, logical_bytes = _cpu_block_state("raw_block_transfer")
    before = _snapshot(model)
    state_before = state.to_dict()

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=4096,
        host_reserve_bytes=0,
        gpu_available_bytes=logical_bytes,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_gpu_bytes == logical_bytes
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_allocation_block_plan_allows_empty_shm_map_when_shm_is_disabled() -> None:
    model, state, _logical_bytes = _cpu_block_state("allocation_block_transfer")
    state.shm_block_name_map = {}
    tensor_manager = SimpleNamespace(loader_type="allocation_block_transfer", use_shm=False)
    before = _snapshot(model)
    state_before = state.to_dict()

    plan = _plan_profile_transition(
        tensor_manager,
        model,
        state,
        host_available_bytes=4096,
        host_reserve_bytes=0,
        gpu_available_bytes=4096,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_host_bytes == 128
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_allocation_block_plan_rejects_empty_shm_map_when_shm_is_enabled() -> None:
    model, state, _logical_bytes = _cpu_block_state("allocation_block_transfer")
    state.shm_block_name_map = {}
    tensor_manager = SimpleNamespace(loader_type="allocation_block_transfer", use_shm=True)
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(ValueError, match="shm_block_name_map"):
        _plan_profile_transition(
            tensor_manager,
            model,
            state,
            host_available_bytes=4096,
            host_reserve_bytes=0,
            gpu_available_bytes=4096,
            gpu_reserve_bytes=0,
        )

    assert _snapshot(model) == before
    assert state.to_dict() == state_before


@pytest.mark.parametrize("mismatch", ["missing", "unexpected"])
def test_plan_rejects_saved_live_inventory_mismatch_before_mutation(mismatch: str) -> None:
    model, state = _cpu_model_and_state()
    if mismatch == "missing":
        state.tensor_id_to_name_map[-1] = "missing.weight"
        expected_name = "missing.weight"
    else:
        model.register_buffer("unexpected", torch.ones(1))
        expected_name = "unexpected"
    before = _snapshot(model)

    with pytest.raises(ValueError, match="inventory") as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    assert expected_name in str(exc_info.value)
    assert _snapshot(model) == before


def test_plan_rejects_duplicate_saved_inventory_names() -> None:
    model, state = _cpu_model_and_state()
    state.tensor_id_to_name_map[-1] = "weight"

    with pytest.raises(ValueError, match=r"inventory names.*unique"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


@pytest.mark.parametrize("location", ["load", "release", "stats"])
def test_plan_rejects_tensor_statistics_id_name_mismatch(location: str) -> None:
    model, state = _cpu_model_and_state()
    invalid_stat = state.load_strategy["layer"][0].model_copy(update={"tensor_id": -1})
    if location == "load":
        state.load_strategy["layer"] = [invalid_stat]
    elif location == "release":
        state.release_strategy["layer"] = [invalid_stat]
    else:
        state.stats[0] = state.stats[0].model_copy(update={"tensors": [invalid_stat]})

    with pytest.raises(ValueError, match=r"TensorStatistics.*tensor_id.*name"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


@pytest.mark.parametrize("mismatch", ["length", "duplicate-id", "duplicate-name", "wrong-pair"])
def test_plan_rejects_inconsistent_saved_view_id_name_pairs(mismatch: str) -> None:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    model.register_parameter("second", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    state = _state_for(model, load_names=(), view_names=("first", "second"))
    first_id, second_id = state.view_tensors_ids
    if mismatch == "length":
        state.view_tensors_ids.pop()
    elif mismatch == "duplicate-id":
        state.view_tensors_ids = [first_id, first_id]
    elif mismatch == "duplicate-name":
        state.view_tensors_names = ["first", "first"]
    else:
        state.view_tensors_ids = [second_id, first_id]

    with pytest.raises(ValueError, match=r"view_tensors_ids.*view_tensors_names"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


def test_plan_uses_saved_ids_only_for_internal_consistency() -> None:
    model, state = _cpu_model_and_state()
    stale_id = -123
    state.tensor_id_to_name_map = {stale_id: "weight"}
    for strategy in (state.load_strategy, state.release_strategy):
        strategy["layer"] = [strategy["layer"][0].model_copy(update={"tensor_id": stale_id})]
    stale_stat = state.stats[0].tensors[0].model_copy(update={"tensor_id": stale_id})
    state.stats[0] = state.stats[0].model_copy(update={"tensors": [stale_stat]})

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.migrations == ()


@pytest.mark.parametrize("reference", ["load", "release", "stats", "view", "gpu"])
def test_plan_rejects_name_references_outside_saved_inventory(reference: str) -> None:
    model, state = _cpu_model_and_state()
    invalid_stat = TensorStatistics(tensor_id=-1, name="invalid.weight", size_bytes=16, load_time_ms=0.1)
    if reference == "load":
        state.load_strategy["layer"].append(invalid_stat)
    elif reference == "release":
        state.release_strategy["layer"].append(invalid_stat)
    elif reference == "stats":
        state.stats[0].tensors.append(invalid_stat)
    elif reference == "view":
        state.view_tensors_ids.append(-1)
        state.view_tensors_names.append("invalid.weight")
    else:
        state.gpu_tensors_names.append("invalid.weight")

    with pytest.raises(ValueError, match=r"invalid\.weight"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


def test_plan_rejects_explicit_gpu_inventory_mismatch() -> None:
    model, state = _cpu_model_and_state()
    model.register_buffer("constant", torch.ones(1))
    state = _state_for(model, load_names=("weight",))
    state.gpu_tensors_names.clear()

    with pytest.raises(ValueError, match="gpu_tensors_names"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


def test_plan_rejects_managed_buffer() -> None:
    model = torch.nn.Module()
    model.register_buffer("managed_buffer", torch.ones(1))
    state = _state_for(model, load_names=("managed_buffer",))

    with pytest.raises(ValueError, match="buffer"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


def test_plan_rejects_meta_tensor() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.empty(4, device="meta"), requires_grad=False))
    state = _state_for(model, load_names=("weight",))

    with pytest.raises(ValueError, match="meta"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


@pytest.mark.parametrize("operation", ["migration", "pinning"])
def test_execute_rejects_quantized_storage_before_rebinding(monkeypatch, operation: str) -> None:
    model = torch.nn.Module()
    model.register_buffer("workspace", torch.quantize_per_tensor(torch.arange(4.0), 0.1, 2, torch.qint8))
    migration_plan = _single_storage_migration_plan(model, ("workspace",))
    plan = StateTransitionPlan(
        migrations=migration_plan.migrations if operation == "migration" else (),
        pinning_groups=(("workspace",),) if operation == "pinning" else (),
        peak_host_bytes=0,
        peak_gpu_bytes=migration_plan.peak_gpu_bytes,
    )
    expected = model.workspace.dequantize().clone()
    manager = SimpleNamespace(host_pinner=SimpleNamespace(pin=lambda tensor: tensor.clone()))
    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", lambda source, _device: source.clone())
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    with pytest.raises(ValueError, match="quantized tensor"):
        TensorManagerStateHandler(manager).execute_state_adoption(model, plan)

    torch.testing.assert_close(model.workspace.dequantize(), expected)


def test_plan_rejects_unsupported_live_layout_before_storage_inspection() -> None:
    model = torch.nn.Module()
    sparse = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2))
    model.register_parameter("weight", torch.nn.Parameter(sparse, requires_grad=False))
    state = _state_for(model, load_names=("weight",))

    with pytest.raises(ValueError, match=r"layout.*torch.strided"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


def test_plan_rejects_serialized_storage_size_drift_before_mutation() -> None:
    model, state = _cpu_model_and_state()
    before = _snapshot(model)
    stat = state.load_strategy["layer"][0]
    state.load_strategy["layer"][0] = stat.model_copy(update={"size_bytes": stat.size_bytes + 1})

    with pytest.raises(ValueError, match="size_bytes"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    assert _snapshot(model) == before


class _AliasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        storage = torch.arange(8, dtype=torch.float32)
        self.register_parameter("first", torch.nn.Parameter(storage[:4], requires_grad=False))
        self.register_parameter("second", torch.nn.Parameter(storage[2:6], requires_grad=False))


def test_plan_rejects_alias_group_split_between_destinations() -> None:
    model = _AliasModel()
    state = _state_for(model, load_names=("first",))

    with pytest.raises(ValueError, match="alias"):
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )


def test_plan_keeps_distinct_empty_storages_in_separate_groups() -> None:
    model = torch.nn.Module()
    model.register_parameter("managed", torch.nn.Parameter(torch.empty(0), requires_grad=False))
    model.register_buffer("constant", torch.empty(0))
    state = _state_for(model, load_names=("managed",))
    manager = SimpleNamespace(loader_type="strategy", device_gpu=torch.device("cuda:0"))

    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert [(migration.names, migration.nbytes) for migration in plan.migrations] == [(("constant",), 0)]


class _TaggedParameter(torch.nn.Parameter):
    def __new__(cls, data: torch.Tensor, *, tag: str) -> "_TaggedParameter":
        parameter = super().__new__(cls, data, requires_grad=False)
        parameter.tag = tag
        return parameter


class _SplitDeviceModel(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.promote = torch.nn.Linear(4, 1, bias=False, device="cpu")

        self.demote = torch.nn.Module()
        demote_storage = torch.arange(8, dtype=torch.float32, device=device)
        self.demote.register_parameter(
            "weight",
            _TaggedParameter(demote_storage[:4].view(2, 2), tag="demote"),
        )
        self.demote.register_parameter(
            "weight_view",
            torch.nn.Parameter(demote_storage[2:6].view(2, 2), requires_grad=False),
        )
        self.demote.weight.scale = self.demote.weight_view

        self.permanent = torch.nn.Linear(3, 1, bias=False, device=device)
        self.register_buffer("constant", torch.arange(5, dtype=torch.float32, device=device))


def _split_model_and_state() -> tuple[_SplitDeviceModel, TensorManagerState, torch.device]:
    device = torch.device("cuda", torch.cuda.current_device())
    model = _SplitDeviceModel(device)
    state = _state_for(
        model,
        load_names=("demote.weight",),
        view_names=("demote.weight_view",),
    )
    return model, state, device


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for split-device adoption")


def test_capacity_order_search_recovers_from_largest_first_dead_end() -> None:
    search = getattr(state_adoption_module, "_find_migration_order", None)
    assert search is not None
    migrations = (
        StorageMigration("promote", ("promote",), "cpu", "cuda:0", 4),
        StorageMigration("demote_a", ("demote_a",), "cuda:0", "cpu", 2),
        StorageMigration("demote_b", ("demote_b",), "cuda:0", "cpu", 2),
        StorageMigration("demote_c", ("demote_c",), "cuda:0", "cpu", 3),
    )

    ordered = search(migrations, host_capacity=4, gpu_capacity=0, extra_gpu_bytes=0)

    assert ordered is not None
    assert [migration.nbytes for migration in ordered] == [2, 2, 4, 3]


@requires_cuda
def test_plan_rejects_configured_gpu_mismatch_with_live_final_gpu_before_mutation() -> None:
    live_device = torch.device("cuda", torch.cuda.current_device())
    configured_device = torch.device("cuda", (live_device.index or 0) + 1)
    model = torch.nn.Module()
    model.register_parameter(
        "weight",
        torch.nn.Parameter(torch.arange(4.0, device=live_device), requires_grad=False),
    )
    state = _state_for(model, load_names=())
    tensor_manager = SimpleNamespace(loader_type="strategy", device_gpu=configured_device)
    before = _snapshot(model)
    state_before = state.to_dict()

    with pytest.raises(ValueError) as exc_info:
        _plan_profile_transition(
            tensor_manager,
            model,
            state,
            host_available_bytes=1024,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    message = str(exc_info.value).lower()
    assert "cross-device cuda migration is not modeled by this planner" in message
    assert str(live_device) in message
    assert str(configured_device) in message
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


@requires_cuda
def test_plan_contains_both_migration_directions_without_mutating_model() -> None:
    model, state, device = _split_model_and_state()
    before = _snapshot(model)
    state_before = state.to_dict()
    demote_bytes = model.demote.weight.untyped_storage().nbytes()
    promote_bytes = model.promote.weight.untyped_storage().nbytes()

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=1024**3,
        host_reserve_bytes=0,
        gpu_available_bytes=1024**3,
        gpu_reserve_bytes=0,
    )

    migrations = {
        (migration.names, migration.source_device, migration.destination_device, migration.nbytes)
        for migration in plan.migrations
    }
    assert (
        ("demote.weight", "demote.weight_view"),
        str(device),
        "cpu",
        demote_bytes,
    ) in migrations
    assert (("promote.weight",), "cpu", str(device), promote_bytes) in migrations
    assert plan.peak_host_bytes == promote_bytes
    assert plan.peak_gpu_bytes == promote_bytes
    assert _snapshot(model) == before
    assert state.to_dict() == state_before


def test_block_loader_gpu_bytes_sum_all_blocks() -> None:
    model = torch.nn.Module()
    model.demote = torch.nn.Module()
    model.demote.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    model.demote.register_parameter("weight_view", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    state = _state_for(
        model,
        load_names=("demote.weight",),
        view_names=("demote.weight_view",),
    )
    state.loader_type = "allocation_block_transfer"
    tensors = dict(model.named_parameters())
    weight_stat = _stat("demote.weight", tensors["demote.weight"])
    view_stat = _stat("demote.weight_view", tensors["demote.weight_view"])
    state.load_strategy = {"weight": [weight_stat], "view": [view_stat]}
    state.release_strategy = {}
    state.view_tensors_ids = [id(tensors["demote.weight"]), id(tensors["demote.weight_view"])]
    state.view_tensors_names = ["demote.weight", "demote.weight_view"]
    state.stats = [
        LayerStatistics(label="weight", tensors=[weight_stat], duration=1.0),
        LayerStatistics(label="view", tensors=[view_stat], duration=1.0),
    ]
    state.allocation_ordered = {0: ["weight"], 1: ["view"]}
    state.label_to_size_map = {}
    state.block_sizes = {0: weight_stat.size_bytes, 1: view_stat.size_bytes}
    state.label_to_block_id = {"weight": 0, "view": 1}
    state.transfer_to_compute_map = {"weight": "weight", "view": "view"}

    _target, transition = target_from_profile(
        capture_model_state(model),
        state,
        target_device="cuda:0",
        pinning="none",
        use_shm=False,
    )

    assert transition.extra_gpu_bytes == 256


@requires_cuda
@pytest.mark.parametrize(
    ("constrained_device", "expected_first_source"),
    [("gpu", "cuda"), ("host", "cpu")],
)
def test_plan_orders_migrations_to_free_constrained_side(
    constrained_device: str,
    expected_first_source: str,
) -> None:
    model, state, _device = _split_model_and_state()
    reserve_bytes = 8
    final_growth_bytes = model.promote.weight.untyped_storage().nbytes()
    host_available_bytes = reserve_bytes + final_growth_bytes if constrained_device == "host" else 1024**3
    gpu_available_bytes = reserve_bytes + final_growth_bytes if constrained_device == "gpu" else 1024**3

    plan = _plan_profile_transition(
        object(),
        model,
        state,
        host_available_bytes=host_available_bytes,
        host_reserve_bytes=reserve_bytes if constrained_device == "host" else 0,
        gpu_available_bytes=gpu_available_bytes,
        gpu_reserve_bytes=reserve_bytes if constrained_device == "gpu" else 0,
    )

    assert plan.migrations[0].source_device.split(":")[0] == expected_first_source


@requires_cuda
@pytest.mark.parametrize("constrained_device", ["host", "gpu"])
def test_plan_rejects_capacity_before_mutation(constrained_device: str) -> None:
    model, state, _device = _split_model_and_state()
    before = _snapshot(model)
    reserve_bytes = 8
    host_available_bytes = reserve_bytes if constrained_device == "host" else 1024**3
    gpu_available_bytes = reserve_bytes if constrained_device == "gpu" else 1024**3

    with pytest.raises(RuntimeError) as exc_info:
        _plan_profile_transition(
            object(),
            model,
            state,
            host_available_bytes=host_available_bytes,
            host_reserve_bytes=reserve_bytes if constrained_device == "host" else 0,
            gpu_available_bytes=gpu_available_bytes,
            gpu_reserve_bytes=reserve_bytes if constrained_device == "gpu" else 0,
        )

    message = str(exc_info.value)
    assert constrained_device in message.lower()
    assert "required" in message.lower()
    assert "available=0" in message
    assert _snapshot(model) == before


class _CopyingHostPinner:
    registry = None

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(tensor)
        return tensor.clone()


class _NonRetainingCopyingHostPinner:
    registry = None

    def __init__(self) -> None:
        self.calls = 0

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return tensor.clone()

    def is_pinned(self, _tensor: torch.Tensor) -> bool:
        return self.calls > 0


def _pinning_manager(*, pinned_memory: bool, registry: object | None = None) -> SimpleNamespace:
    pinner = _CopyingHostPinner()
    pinner.registry = registry
    return SimpleNamespace(loader_type="strategy", pinned_memory=pinned_memory, host_pinner=pinner)


def test_strategy_plan_marks_stationary_cpu_home_for_pinning_without_mutation() -> None:
    model, state = _cpu_model_and_state()
    manager = _pinning_manager(pinned_memory=True)
    before = _snapshot(model)

    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=model.weight.untyped_storage().nbytes(),
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.pinning_groups == (("weight",),)
    assert manager.host_pinner.calls == []
    assert _snapshot(model) == before


def test_strategy_plan_skips_pinning_when_disabled_without_mutation() -> None:
    model, state = _cpu_model_and_state()
    manager = _pinning_manager(pinned_memory=False)
    before = _snapshot(model)

    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=0,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.pinning_groups == ()
    assert manager.host_pinner.calls == []
    assert _snapshot(model) == before


def test_strategy_torch_pinning_capacity_fails_before_stationary_cpu_mutation() -> None:
    model, state = _cpu_model_and_state()
    manager = _pinning_manager(pinned_memory=True)
    before = _snapshot(model)
    required = model.weight.untyped_storage().nbytes()

    with pytest.raises(RuntimeError, match="Insufficient capacity"):
        _plan_profile_transition(
            manager,
            model,
            state,
            host_available_bytes=required - 1,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    assert _snapshot(model) == before
    assert manager.host_pinner.calls == []


@pytest.mark.parametrize(
    ("loader_type", "expected_peak"),
    [("allocation_block_transfer", 128), ("raw_block_transfer", 16)],
)
def test_block_plan_counts_exact_host_block_construction(loader_type: str, expected_peak: int) -> None:
    model, state, _logical_bytes = _cpu_block_state(loader_type)
    manager = SimpleNamespace(loader_type=loader_type, pinned_memory=False, use_shm=False)

    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=expected_peak,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_host_bytes == expected_peak


@pytest.mark.parametrize(("registry", "expected_peak"), [(None, 256), (object(), 128)])
def test_allocation_block_plan_counts_actual_pinning_construction_peak(
    registry: object | None,
    expected_peak: int,
) -> None:
    model, state, _logical_bytes = _cpu_block_state("allocation_block_transfer")
    manager = _pinning_manager(pinned_memory=True, registry=registry)
    manager.use_shm = False
    manager.loader_type = "allocation_block_transfer"

    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=expected_peak,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_host_bytes == expected_peak


def test_allocation_block_shm_follower_does_not_charge_existing_host_block() -> None:
    model, state, _logical_bytes = _cpu_block_state("allocation_block_transfer")
    state.shm_block_name_map = {"layer": "existing"}
    manager = SimpleNamespace(loader_type="allocation_block_transfer", pinned_memory=True, use_shm=True)

    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=0,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    assert plan.peak_host_bytes == 0


def test_block_plan_rejects_zero_host_headroom_before_mutation() -> None:
    model, state, _logical_bytes = _cpu_block_state("allocation_block_transfer")
    manager = SimpleNamespace(loader_type="allocation_block_transfer", pinned_memory=False, use_shm=False)
    before = _snapshot(model)

    with pytest.raises(RuntimeError, match="Insufficient capacity"):
        _plan_profile_transition(
            manager,
            model,
            state,
            host_available_bytes=0,
            host_reserve_bytes=0,
            gpu_available_bytes=1024,
            gpu_reserve_bytes=0,
        )

    assert _snapshot(model) == before


def _storage_impl_key(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return tensor.device.type, tensor.device.index, storage._cdata, storage.nbytes()  # noqa: SLF001


def test_execute_resolves_unnamed_storage_after_reachability_reorder(monkeypatch) -> None:
    model = torch.nn.Module()
    planned = torch.arange(4.0)
    other = torch.arange(4.0) + 10
    model.workspaces = [planned, other]
    captured = capture_model_state(model)
    plan = StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id=captured.storages[0].id,
                names=(),
                source_device="cpu",
                destination_device="cuda:0",
                nbytes=planned.untyped_storage().nbytes(),
            ),
        ),
        pinning_groups=(),
        peak_host_bytes=0,
        peak_gpu_bytes=planned.untyped_storage().nbytes(),
    )
    planned_storage = _storage_impl_key(planned)
    other_storage = _storage_impl_key(other)
    model.workspaces.reverse()

    monkeypatch.setattr(
        state_handler_module,
        "_copy_storage_to_device",
        lambda source, _destination: source.clone(),
    )
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert _storage_impl_key(planned) != planned_storage
    assert _storage_impl_key(other) == other_storage


@pytest.mark.parametrize("view_kind", ["conjugate", "negative"])
def test_execute_preserves_lazy_view_metadata(monkeypatch, view_kind: str) -> None:
    model = torch.nn.Module()
    model.base = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    model.view = model.base.T.conj() if view_kind == "conjugate" else model.base.conj().imag
    captured = capture_model_state(model)
    plan = StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id=captured.storages[0].id,
                names=(),
                source_device="cpu",
                destination_device="cuda:0",
                nbytes=model.base.untyped_storage().nbytes(),
            ),
        ),
        pinning_groups=(),
        peak_host_bytes=0,
        peak_gpu_bytes=model.base.untyped_storage().nbytes(),
    )
    view_id = id(model.view)
    expected = model.view.clone()
    expected_shape = model.view.shape
    expected_stride = model.view.stride()
    expected_offset = model.view.storage_offset()

    monkeypatch.setattr(
        state_handler_module,
        "_copy_storage_to_device",
        lambda source, _destination: source.clone(),
    )
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert id(model.view) == view_id
    assert model.view.is_conj() if view_kind == "conjugate" else model.view.is_neg()
    torch.testing.assert_close(model.view, expected)
    assert model.view.shape == expected_shape
    assert model.view.stride() == expected_stride
    assert model.view.storage_offset() == expected_offset
    assert _storage_impl_key(model.base) == _storage_impl_key(model.view)


@pytest.mark.skipif(
    not callable(getattr(torch.Tensor, "refine_names", None)), reason="PyTorch named tensors are unavailable"
)
def test_execute_preserves_named_tensor_dimensions(monkeypatch) -> None:
    model = torch.nn.Module()
    model.workspace = torch.arange(6.0).reshape(2, 3).refine_names("row", "column")
    plan = _single_storage_migration_plan(model, ())
    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", lambda source, _device: source.clone())
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert model.workspace.names == ("row", "column")


def test_execute_rebinds_when_named_tensor_api_is_unavailable(monkeypatch) -> None:
    model = torch.nn.Module()
    model.workspace = torch.arange(6.0).reshape(2, 3)
    plan = _single_storage_migration_plan(model, ())
    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", lambda source, _device: source.clone())
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)
    monkeypatch.setattr(torch.Tensor, "has_names", None, raising=False)
    monkeypatch.setattr(torch.Tensor, "names", None, raising=False)
    monkeypatch.setattr(torch.Tensor, "rename_", None, raising=False)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert model.workspace.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.mark.parametrize(
    "invalid_state",
    [
        "loader",
        "compiling",
        "capturing",
        "compiled_wrapper",
        "live_wrapper",
        "compiled_in_place",
        "compiled_forward",
    ],
)
def test_execute_rejects_invalid_lifecycle_state_before_mutation(monkeypatch, invalid_state: str) -> None:
    manager = SimpleNamespace(tensor_layer_loader=None)
    model = torch.nn.Linear(2, 2)
    model_to_execute = model
    if invalid_state == "loader":
        manager.tensor_layer_loader = object()
    elif invalid_state == "compiling":
        monkeypatch.setattr(state_handler_module, "is_compiling", lambda: True, raising=False)
    elif invalid_state == "capturing":
        monkeypatch.setattr(state_handler_module.torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(state_handler_module.torch.cuda, "is_current_stream_capturing", lambda: True)
    elif invalid_state == "compiled_wrapper":
        model_to_execute = torch.compile(model, backend="eager")
    elif invalid_state == "live_wrapper":
        _live_wrapper = torch.compile(model, backend="eager")
    elif invalid_state == "compiled_in_place":
        model.compile(backend="eager")
    else:
        model.forward = torch.compile(model.forward, backend="eager")

    plan = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
    handler = TensorManagerStateHandler(manager)

    with (
        patch.object(handler, "_resolve_state_adoption_plan") as resolve_plan,
        pytest.raises(RuntimeError, match="State adoption must run before"),
    ):
        handler.execute_state_adoption(model_to_execute, plan)
    resolve_plan.assert_not_called()


def test_execute_stages_cross_device_replacements_on_destination(monkeypatch) -> None:
    model = torch.nn.Module()
    model.workspace = torch.arange(4.0)
    captured = capture_model_state(model)
    plan = StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id=captured.storages[0].id,
                names=(),
                source_device="cpu",
                destination_device="cuda:0",
                nbytes=model.workspace.untyped_storage().nbytes(),
            ),
        ),
        pinning_groups=(),
        peak_host_bytes=0,
        peak_gpu_bytes=model.workspace.untyped_storage().nbytes(),
    )
    replacements: list[torch.Tensor] = []

    monkeypatch.setattr(
        state_handler_module,
        "_copy_storage_to_device",
        lambda source, _destination: torch.empty(source.numel(), dtype=source.dtype, device="meta"),
    )
    monkeypatch.setattr(
        state_handler_module,
        "_set_tensor_data",
        lambda _tensor, replacement: replacements.append(replacement),
    )
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert [replacement.device.type for replacement in replacements] == ["meta"]


def test_execute_migrates_unregistered_workspace_storage(monkeypatch) -> None:
    model = torch.nn.Module()
    model.workspace = torch.arange(8.0)
    list_view = model.workspace[1:4]
    tuple_view = model.workspace[4:7]
    model.views = [list_view, (tuple_view,)]
    state = _state_for(model, load_names=())
    manager = SimpleNamespace(loader_type="strategy", pinned_memory=False)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    tensors = (model.workspace, list_view, tuple_view)
    tensor_ids = tuple(map(id, tensors))
    expected = tuple(tensor.clone() for tensor in tensors)
    destinations: list[str] = []

    def copy_to_requested_device(source: torch.Tensor, destination: str) -> torch.Tensor:
        destinations.append(destination)
        return source.clone()

    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", copy_to_requested_device)
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert plan.migrations[0].names == ()
    assert destinations == ["cuda:0"]
    migrated = (model.workspace, model.views[0], model.views[1][0])
    assert tuple(map(id, migrated)) == tensor_ids
    assert len({_storage_impl_key(tensor) for tensor in migrated}) == 1
    for tensor, value in zip(migrated, expected, strict=True):
        torch.testing.assert_close(tensor, value)


def test_execute_walks_reachable_tensors_once_during_preflight(monkeypatch) -> None:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.arange(2.0), requires_grad=False))
    model.register_parameter("second", torch.nn.Parameter(torch.arange(3.0), requires_grad=False))
    state = _state_for(model, load_names=("first", "second"))
    manager = _pinning_manager(pinned_memory=True)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    original_compute = state_handler_module.compute_reachable_tensor_map
    walks = 0

    def count_walks(current_model: torch.nn.Module) -> dict[int, torch.Tensor]:
        nonlocal walks
        walks += 1
        return original_compute(current_model)

    monkeypatch.setattr(state_handler_module, "compute_reachable_tensor_map", count_walks)

    TensorManagerStateHandler(manager).execute_state_adoption(model, plan)

    assert walks == 1


def test_execute_pins_stationary_alias_group_without_replacing_tensor_objects() -> None:
    model = _AliasModel()
    state = _state_for(model, load_names=("first",), view_names=("second",))
    manager = _pinning_manager(pinned_memory=True)
    handler = TensorManagerStateHandler(manager)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=2 * model.first.untyped_storage().nbytes(),
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    tensors_before = {
        name: (
            id(tensor),
            tensor.detach().clone(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
        )
        for name, tensor in model.named_parameters()
    }
    old_storage_key = _storage_impl_key(model.first)

    handler.execute_state_adoption(model, plan)

    assert len(manager.host_pinner.calls) == 1
    assert _storage_impl_key(model.first) == _storage_impl_key(model.second)
    assert _storage_impl_key(model.first) != old_storage_key
    for name, tensor in model.named_parameters():
        object_id, value, shape, stride, storage_offset = tensors_before[name]
        assert id(tensor) == object_id
        torch.testing.assert_close(tensor, value)
        assert tensor.shape == shape
        assert tensor.stride() == stride
        assert tensor.storage_offset() == storage_offset


def test_execute_keeps_distinct_empty_storage_groups_distinct_when_pinning() -> None:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.empty(0), requires_grad=False))
    model.register_parameter("second", torch.nn.Parameter(torch.empty(0), requires_grad=False))
    state = _state_for(model, load_names=("first", "second"))
    manager = _pinning_manager(pinned_memory=True)
    handler = TensorManagerStateHandler(manager)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    ids_before = {name: id(tensor) for name, tensor in model.named_parameters()}

    handler.execute_state_adoption(model, plan)

    assert len(manager.host_pinner.calls) == 2
    assert _storage_impl_key(model.first) != _storage_impl_key(model.second)
    assert {name: id(tensor) for name, tensor in model.named_parameters()} == ids_before


def test_execute_and_restore_preserve_unnamed_nested_view_alias() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(8.0), requires_grad=False))
    nested_view = model.weight[1:5]
    model.weight.views = {"nested": {"slice": nested_view}}
    state = _state_for(model, load_names=("weight",))
    pinner = _NonRetainingCopyingHostPinner()
    manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=True,
        host_pinner=pinner,
        use_shm=False,
        tensors_map={},
        traced_tensors=set(),
        tensor_manager_state=None,
    )
    manager.set_model = lambda restored_model: setattr(manager, "model", restored_model)
    handler = TensorManagerStateHandler(manager)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=2 * model.weight.untyped_storage().nbytes(),
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    old_storage = weakref.ref(model.weight.untyped_storage())
    nested_id = id(nested_view)
    expected = nested_view.clone()

    handler.execute_state_adoption(model, plan)
    gc.collect()

    assert pinner.calls == 1
    assert old_storage() is None
    assert id(model.weight.views["nested"]["slice"]) == nested_id
    torch.testing.assert_close(model.weight.views["nested"]["slice"], expected)
    assert _storage_impl_key(model.weight) == _storage_impl_key(model.weight.views["nested"]["slice"])

    handler.restore_state(model, state, preprocess_model_state=False)

    assert _storage_impl_key(model.weight) == _storage_impl_key(model.weight.views["nested"]["slice"])


def test_execute_rebinds_hidden_view_base_and_releases_source_storage(monkeypatch) -> None:
    model = torch.nn.Module()
    backing = torch.arange(8.0)
    model.view = backing[1:5]
    captured = capture_model_state(model)
    plan = StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id=captured.storages[0].id,
                names=(),
                source_device="cpu",
                destination_device="cuda:0",
                nbytes=backing.untyped_storage().nbytes(),
            ),
        ),
        pinning_groups=(),
        peak_host_bytes=0,
        peak_gpu_bytes=backing.untyped_storage().nbytes(),
    )
    old_storage = weakref.ref(backing.untyped_storage())
    expected = model.view.clone()
    del backing

    monkeypatch.setattr(
        state_handler_module,
        "_copy_storage_to_device",
        lambda source, _destination: source.clone(),
    )
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)
    gc.collect()

    assert old_storage() is None
    assert _storage_impl_key(model.view) == _storage_impl_key(model.view._base)
    torch.testing.assert_close(model.view, expected)


class _RejectDataAssignmentParameter(torch.nn.Parameter):
    def __new__(cls, data: torch.Tensor) -> "_RejectDataAssignmentParameter":
        return super().__new__(cls, data, requires_grad=False)

    @property
    def data(self) -> torch.Tensor:
        return torch.Tensor.data.__get__(self, type(self))

    @data.setter
    def data(self, _value: torch.Tensor) -> None:
        raise RuntimeError("custom parameter rejects .data assignment")


def test_execute_rebinds_parameter_subclass_without_using_custom_data_setter() -> None:
    model = torch.nn.Module()
    backing = torch.arange(8.0)
    model.register_parameter("first", torch.nn.Parameter(backing[:4], requires_grad=False))
    model.register_parameter("second", _RejectDataAssignmentParameter(backing[2:6]))
    state = _state_for(model, load_names=("first",), view_names=("second",))
    pinner = _NonRetainingCopyingHostPinner()
    manager = SimpleNamespace(loader_type="strategy", pinned_memory=True, host_pinner=pinner)
    handler = TensorManagerStateHandler(manager)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=2 * model.first.untyped_storage().nbytes(),
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    ids_before = {name: id(tensor) for name, tensor in model.named_parameters()}
    values_before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    handler.execute_state_adoption(model, plan)

    assert isinstance(model.second, _RejectDataAssignmentParameter)
    assert {name: id(tensor) for name, tensor in model.named_parameters()} == ids_before
    assert _storage_impl_key(model.first) == _storage_impl_key(model.second)
    for name, tensor in model.named_parameters():
        torch.testing.assert_close(tensor, values_before[name])


def test_execute_rolls_back_current_alias_group_when_rebind_fails(monkeypatch) -> None:
    model = _AliasModel()
    state = _state_for(model, load_names=("first",), view_names=("second",))
    pinner = _NonRetainingCopyingHostPinner()
    manager = SimpleNamespace(loader_type="strategy", pinned_memory=True, host_pinner=pinner)
    handler = TensorManagerStateHandler(manager)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=2 * model.first.untyped_storage().nbytes(),
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )
    original_set_data = state_handler_module._set_tensor_data  # noqa: SLF001
    set_data_calls = 0
    source_key = _storage_impl_key(model.first)
    values_before = {name: tensor.detach().clone() for name, tensor in model.named_parameters()}

    def fail_second_rebind(tensor: torch.Tensor, replacement: torch.Tensor) -> None:
        nonlocal set_data_calls
        set_data_calls += 1
        if set_data_calls == 2:
            raise RuntimeError("injected alias rebind failure")
        original_set_data(tensor, replacement)

    monkeypatch.setattr(state_handler_module, "_set_tensor_data", fail_second_rebind)

    with pytest.raises(RuntimeError, match="injected alias rebind failure"):
        handler.execute_state_adoption(model, plan)

    assert set_data_calls == 3
    assert _storage_impl_key(model.first) == source_key
    assert _storage_impl_key(model.second) == source_key
    for name, tensor in model.named_parameters():
        torch.testing.assert_close(tensor, values_before[name])


@pytest.mark.skipif(
    not callable(getattr(torch.Tensor, "refine_names", None)), reason="PyTorch named tensors are unavailable"
)
def test_execute_rolls_back_named_tensor_when_name_restore_fails(monkeypatch) -> None:
    model = torch.nn.Module()
    model.workspace = torch.arange(6.0).reshape(2, 3).refine_names("row", "column")
    plan = _single_storage_migration_plan(model, ())
    source_key = _storage_impl_key(model.workspace)
    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", lambda source, _device: source.clone())
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    def fail_name_restore(_tensor: torch.Tensor, *_names: str | None) -> None:
        raise RuntimeError("injected name restoration failure")

    monkeypatch.setattr(torch.Tensor, "rename_", fail_name_restore)

    with pytest.raises(RuntimeError, match="injected name restoration failure"):
        TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert _storage_impl_key(model.workspace) == source_key
    assert model.workspace.names == ("row", "column")


def test_execute_rejects_stale_plan_before_any_pinning_or_migration() -> None:
    model, _state = _cpu_model_and_state()
    manager = _pinning_manager(pinned_memory=True)
    before = _snapshot(model)
    stale_plan = StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id="storage:stale",
                names=("weight",),
                source_device="cpu",
                destination_device="cuda:0",
                nbytes=model.weight.untyped_storage().nbytes() + 1,
            ),
        ),
        peak_host_bytes=0,
        peak_gpu_bytes=0,
        pinning_groups=(("weight",),),
    )

    with pytest.raises(ValueError, match=r"plan.*weight.*nbytes|nbytes.*weight"):
        TensorManagerStateHandler(manager).execute_state_adoption(model, stale_plan)

    assert manager.host_pinner.calls == []
    assert _snapshot(model) == before


def test_execute_rejects_same_sized_replaced_named_storage(monkeypatch) -> None:
    model, _state = _cpu_model_and_state()
    plan = _single_storage_migration_plan(model, ("weight",))
    model.weight.data = model.weight.data.clone()
    replacement_key = _storage_impl_key(model.weight)
    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", lambda source, _device: source.clone())
    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", lambda _device=None: None)

    with pytest.raises(ValueError, match="storage"):
        TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert _storage_impl_key(model.weight) == replacement_key


@pytest.mark.parametrize(
    ("names", "destination_device", "error"),
    [
        (("missing",), "cuda:0", "missing tensor"),
        (("weight",), "meta", "unsupported devices"),
    ],
)
def test_execute_rejects_invalid_migration_before_copy(
    monkeypatch,
    names: tuple[str, ...],
    destination_device: str,
    error: str,
) -> None:
    model, _state = _cpu_model_and_state()
    captured = capture_model_state(model)
    plan = StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id=captured.storages[0].id,
                names=names,
                source_device="cpu",
                destination_device=destination_device,
                nbytes=model.weight.untyped_storage().nbytes(),
            ),
        ),
        pinning_groups=(),
        peak_host_bytes=0,
        peak_gpu_bytes=0,
    )
    monkeypatch.setattr(
        state_handler_module,
        "_copy_storage_to_device",
        lambda *_args: pytest.fail("invalid plan reached storage copy"),
    )

    with pytest.raises(ValueError, match=error):
        TensorManagerStateHandler(object()).execute_state_adoption(model, plan)


@pytest.mark.parametrize("duplicate_group", ["migration", "pinning"])
def test_execute_rejects_duplicate_names_inside_a_planned_group(duplicate_group: str) -> None:
    model, _state = _cpu_model_and_state()
    manager = _pinning_manager(pinned_memory=True)
    names = ("weight", "weight")
    migrations = (
        (
            StorageMigration(
                storage_id="storage:duplicate",
                names=names,
                source_device="cpu",
                destination_device="cuda:0",
                nbytes=model.weight.untyped_storage().nbytes(),
            ),
        )
        if duplicate_group == "migration"
        else ()
    )
    plan = StateTransitionPlan(
        migrations=migrations,
        peak_host_bytes=0,
        peak_gpu_bytes=0,
        pinning_groups=(names,) if duplicate_group == "pinning" else (),
    )
    before = _snapshot(model)

    with pytest.raises(ValueError, match=r"duplicate.*weight|weight.*duplicate"):
        TensorManagerStateHandler(manager).execute_state_adoption(model, plan)

    assert manager.host_pinner.calls == []
    assert _snapshot(model) == before


class _FailSecondHostPinner(_CopyingHostPinner):
    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(self.calls) == 1:
            raise RuntimeError("injected second pin failure")
        return super().pin(tensor)


def test_execute_reports_completed_and_current_groups_after_partial_pinning_failure() -> None:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.arange(2.0), requires_grad=False))
    model.register_parameter("second", torch.nn.Parameter(torch.arange(3.0), requires_grad=False))
    state = _state_for(model, load_names=("first", "second"))
    pinner = _FailSecondHostPinner()
    manager = SimpleNamespace(loader_type="strategy", pinned_memory=True, host_pinner=pinner)
    handler = TensorManagerStateHandler(manager)
    plan = _plan_profile_transition(
        manager,
        model,
        state,
        host_available_bytes=1024,
        host_reserve_bytes=0,
        gpu_available_bytes=1024,
        gpu_reserve_bytes=0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        handler.execute_state_adoption(model, plan)

    message = str(exc_info.value)
    assert "partial" in message.lower()
    assert "completed" in message.lower()
    assert "current" in message.lower()
    assert "No rollback of previously completed groups was attempted" in message
    assert "first" in message
    assert "second" in message


def test_restore_adopted_state_skips_preprocess_but_rebinds_manager_state() -> None:
    model, state = _cpu_model_and_state()
    manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=False,
        use_shm=False,
        use_trace_tensor=False,
        tensors_map={},
        traced_tensors=set(),
        tensor_manager_state=None,
        set_model=lambda restored_model: setattr(manager, "model", restored_model),
    )

    with patch("flextensor.state_handler.preprocess_model") as preprocess_model:
        result = TensorManagerStateHandler(manager).restore_state(
            model,
            state,
            preprocess_model_state=False,
        )

    preprocess_model.assert_not_called()
    assert set(manager.tensors_map) == {id(model.weight)}
    assert manager.traced_tensors == {id(model.weight)}
    assert manager.tensor_manager_state is not state
    assert manager.model is model
    assert not result


def test_restore_adopted_state_rejects_unexecuted_gpu_placement() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    state = _state_for(model, load_names=())
    manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=False,
        use_shm=False,
        use_trace_tensor=False,
        tensors_map={},
        traced_tensors=set(),
        tensor_manager_state=None,
        set_model=lambda restored_model: setattr(manager, "model", restored_model),
    )

    with pytest.raises(ValueError, match="execute_state_adoption"):
        TensorManagerStateHandler(manager).restore_state(model, state, preprocess_model_state=False)


def test_restore_adopted_state_rejects_unexecuted_host_pinning() -> None:
    model, state = _cpu_model_and_state()
    manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=True,
        host_pinner=SimpleNamespace(registry=None, is_pinned=lambda _tensor: False),
        use_shm=False,
        use_trace_tensor=False,
        tensors_map={},
        traced_tensors=set(),
        tensor_manager_state=None,
        set_model=lambda restored_model: setattr(manager, "model", restored_model),
    )

    with pytest.raises(ValueError, match="execute_state_adoption"):
        TensorManagerStateHandler(manager).restore_state(model, state, preprocess_model_state=False)


def test_restore_adopted_state_accepts_in_place_registered_storage() -> None:
    model, state = _cpu_model_and_state()
    manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=True,
        host_pinner=SimpleNamespace(registry=object(), is_pinned=lambda _tensor: True),
        use_shm=False,
        use_trace_tensor=False,
        tensors_map={},
        traced_tensors=set(),
        tensor_manager_state=None,
        set_model=lambda restored_model: setattr(manager, "model", restored_model),
    )

    result = TensorManagerStateHandler(manager).restore_state(model, state, preprocess_model_state=False)

    assert not result
    assert manager.model is model


def test_restore_adopted_state_still_disables_gradients() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0)))
    state = _state_for(model, load_names=("weight",))
    manager = SimpleNamespace(
        loader_type="strategy",
        device_gpu=torch.device("cuda:0"),
        pinned_memory=False,
        use_shm=False,
        use_trace_tensor=False,
        tensors_map={},
        traced_tensors=set(),
        tensor_manager_state=None,
        set_model=lambda restored_model: setattr(manager, "model", restored_model),
    )

    TensorManagerStateHandler(manager).restore_state(model, state, preprocess_model_state=False)

    assert model.weight.requires_grad is False


def _split_execution_plan(model: _SplitDeviceModel, device: torch.device) -> StateTransitionPlan:
    captured = capture_model_state(model)
    storage_ids = {name: tensor.storage_id for tensor in captured.tensors for name in tensor.names}
    return StateTransitionPlan(
        migrations=(
            StorageMigration(
                storage_id=storage_ids["promote.weight"],
                names=("promote.weight",),
                source_device="cpu",
                destination_device=str(device),
                nbytes=model.promote.weight.untyped_storage().nbytes(),
            ),
            StorageMigration(
                storage_id=storage_ids["demote.weight"],
                names=("demote.weight", "demote.weight_view"),
                source_device=str(device),
                destination_device="cpu",
                nbytes=model.demote.weight.untyped_storage().nbytes(),
            ),
        ),
        pinning_groups=(),
        peak_host_bytes=model.promote.weight.untyped_storage().nbytes(),
        peak_gpu_bytes=model.promote.weight.untyped_storage().nbytes(),
    )


@requires_cuda
def test_execute_migrates_storage_groups_and_preserves_views_and_tensor_identity(monkeypatch) -> None:
    model, _state, device = _split_model_and_state()
    inner_view = torch.empty(0, dtype=model.demote.weight.dtype, device=device)
    inner_view.set_(model.demote.weight.untyped_storage(), 1, (2, 2), (2, 1))
    model.demote.weight.views = {"nested": {"slice": inner_view}}
    base_plan = _split_execution_plan(model, device)
    plan = StateTransitionPlan(
        migrations=tuple(reversed(base_plan.migrations)),
        pinning_groups=(),
        peak_host_bytes=base_plan.peak_host_bytes,
        peak_gpu_bytes=base_plan.peak_gpu_bytes,
    )
    tensors = dict(model.named_parameters(remove_duplicate=False))
    expected = {
        name: (
            id(tensor),
            tensor.detach().cpu().clone(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
        )
        for name, tensor in tensors.items()
    }
    expected_inner = (
        id(inner_view),
        inner_view.cpu().clone(),
        inner_view.shape,
        inner_view.stride(),
        inner_view.storage_offset(),
    )
    source_storages = [
        weakref.ref(dict(model.named_parameters(remove_duplicate=False))[migration.names[0]].untyped_storage())
        for migration in plan.migrations
    ]
    original_copy = state_handler_module._copy_storage_to_device
    copy_calls = 0

    def record_copy(source_bytes: torch.Tensor, destination_device: str) -> torch.Tensor:
        nonlocal copy_calls
        if copy_calls:
            gc.collect()
            assert source_storages[copy_calls - 1]() is None
        copy_calls += 1
        return original_copy(source_bytes, destination_device)

    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", record_copy)

    TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    assert copy_calls == len(plan.migrations)
    tensors = dict(model.named_parameters(remove_duplicate=False))
    destinations = {name: migration.destination_device for migration in plan.migrations for name in migration.names}
    for name, (object_id, value, shape, stride, storage_offset) in expected.items():
        tensor = tensors[name]
        assert id(tensor) == object_id
        assert torch.equal(tensor.detach().cpu(), value)
        assert tensor.shape == shape
        assert tensor.stride() == stride
        assert tensor.storage_offset() == storage_offset
        if name in destinations:
            assert str(tensor.device) == destinations[name]
    migrated_inner = model.demote.weight.views["nested"]["slice"]
    assert id(migrated_inner) == expected_inner[0]
    assert torch.equal(migrated_inner, expected_inner[1])
    assert migrated_inner.shape == expected_inner[2]
    assert migrated_inner.stride() == expected_inner[3]
    assert migrated_inner.storage_offset() == expected_inner[4]
    assert _storage_impl_key(model.demote.weight) == _storage_impl_key(model.demote.weight_view)
    assert _storage_impl_key(model.demote.weight) == _storage_impl_key(migrated_inner)
    assert isinstance(model.demote.weight, _TaggedParameter)
    assert model.demote.weight.tag == "demote"
    assert model.demote.weight.scale is model.demote.weight_view


@requires_cuda
def test_execute_reports_rebound_group_as_completed_when_synchronization_fails(monkeypatch) -> None:
    model, _state, device = _split_model_and_state()
    plan = _split_execution_plan(model, device)
    first = plan.migrations[0]
    synchronize_calls = 0

    def fail_synchronization(_device=None) -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1
        raise RuntimeError(f"injected synchronize failure {synchronize_calls}")

    monkeypatch.setattr(state_handler_module.torch.cuda, "synchronize", fail_synchronization)

    with pytest.raises(RuntimeError) as exc_info:
        TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    message = str(exc_info.value)
    assert "partial" in message.lower()
    assert "No rollback of previously completed groups was attempted" in message
    assert str(first.names) in message
    assert "completed" in message.lower()
    assert "synchron" in message.lower()
    assert synchronize_calls == 2
    assert str(model.promote.weight.device) == first.destination_device


@requires_cuda
def test_execute_reports_completed_and_current_migrations_when_second_copy_fails(monkeypatch) -> None:
    model, _state, device = _split_model_and_state()
    plan = _split_execution_plan(model, device)
    first, second = plan.migrations
    original_copy = state_handler_module._copy_storage_to_device
    copy_calls = 0

    def fail_second_copy(source_bytes: torch.Tensor, destination_device: str) -> torch.Tensor:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise RuntimeError("injected second migration failure")
        return original_copy(source_bytes, destination_device)

    monkeypatch.setattr(state_handler_module, "_copy_storage_to_device", fail_second_copy)

    with pytest.raises(RuntimeError) as exc_info:
        TensorManagerStateHandler(object()).execute_state_adoption(model, plan)

    message = str(exc_info.value)
    assert f"completed migration groups={[first.names]}" in message
    assert f"current migration group={second.names}" in message
    assert "No rollback of previously completed groups was attempted" in message
    assert str(model.promote.weight.device) == first.destination_device
    assert str(model.demote.weight.device) == second.source_device


@requires_cuda
def test_execute_pins_a_group_after_migrating_it_to_cpu() -> None:
    model, _state, device = _split_model_and_state()
    base_plan = _split_execution_plan(model, device)
    demotion = base_plan.migrations[1]
    plan = StateTransitionPlan(
        migrations=base_plan.migrations,
        peak_host_bytes=base_plan.peak_host_bytes,
        peak_gpu_bytes=base_plan.peak_gpu_bytes,
        pinning_groups=(demotion.names,),
    )
    manager = _pinning_manager(pinned_memory=True)

    TensorManagerStateHandler(manager).execute_state_adoption(model, plan)

    assert len(manager.host_pinner.calls) == 1
    assert model.demote.weight.device.type == "cpu"
    assert _storage_impl_key(model.demote.weight) == _storage_impl_key(model.demote.weight_view)
