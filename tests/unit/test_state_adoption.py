# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for read-only planning of saved-state adoption."""

from types import SimpleNamespace

import pytest
import torch

import flextensor.state_transition as state_adoption_module
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
