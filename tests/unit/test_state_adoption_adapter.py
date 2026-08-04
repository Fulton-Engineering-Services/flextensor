# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.model_state_capture import capture_model_state
from flextensor.state_adoption import target_from_profile
from flextensor.state_handler import TensorManagerState
from flextensor.state_transition import MemoryCapacity, StateTransitionPlan, TransitionSpec


def _stat(name: str, tensor: torch.Tensor) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=id(tensor),
        name=name,
        size_bytes=tensor.numel() * tensor.element_size(),
        load_time_ms=0.1,
    )


def _profile(model: torch.nn.Module, *, loader_type: str = "strategy") -> TensorManagerState:
    tensors = dict(model.named_parameters())
    tensors.update(model.named_buffers())
    stat = _stat("weight", tensors["weight"])
    is_block = loader_type != "strategy"
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map={id(tensor): name for name, tensor in tensors.items()},
        allocation_ordered={0: ["layer"]} if is_block else {},
        label_to_size_map={"layer": stat.size_bytes} if loader_type == "raw_block_transfer" else {},
        block_sizes={0: stat.size_bytes} if is_block else {},
        load_strategy={"layer": [stat]},
        release_strategy={"layer": [stat]},
        label_to_block_id={"layer": 0} if is_block else {},
        stats=[LayerStatistics(label="layer", tensors=[stat], duration=1.0)],
        transfer_to_compute_map={"layer": "layer"} if is_block else {},
        view_tensors_ids=[id(tensors["weight"])] if is_block else [],
        view_tensors_names=["weight"] if is_block else [],
        gpu_tensors_names=[name for name in tensors if name != "weight"],
        shm_block_name_map=None,
    )


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    model.register_buffer("constant", torch.ones(2))
    return model


def test_strategy_profile_builds_authoritative_target_without_runtime_manager() -> None:
    model = _model()
    current = capture_model_state(model)

    target, transition = target_from_profile(
        current,
        _profile(model),
        target_device="cuda:0",
        pinning="copy",
        use_shm=False,
    )

    tensor_by_name = {name: tensor for tensor in target.tensors for name in tensor.names}
    storage_by_id = {storage.id: storage for storage in target.storages}
    weight_storage = storage_by_id[tensor_by_name["weight"].storage_id]
    constant_storage = storage_by_id[tensor_by_name["constant"].storage_id]
    assert (weight_storage.device, weight_storage.pinned) == ("cpu", True)
    assert (constant_storage.device, constant_storage.pinned) == ("cuda:0", False)
    assert transition.extra_gpu_bytes == model.weight.numel() * model.weight.element_size()
    assert transition.pinning_copy_storage_ids == (weight_storage.id,)
    assert transition.host_allocations == ()


def test_profile_rejects_current_cuda_device_different_from_target() -> None:
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.arange(4.0), requires_grad=False))
    current = capture_model_state(model)
    current = replace(current, storages=(replace(current.storages[0], device="cuda:1"),))

    with pytest.raises(ValueError, match="Cross-device CUDA migration"):
        target_from_profile(
            current,
            _profile(model),
            target_device="cuda:0",
            pinning="none",
            use_shm=False,
        )


def test_allocation_block_profile_accounts_for_alignment_and_releases_homes() -> None:
    model = _model()
    current = capture_model_state(model)

    target, transition = target_from_profile(
        current,
        _profile(model, loader_type="allocation_block_transfer"),
        target_device="cuda:0",
        pinning="copy",
        use_shm=False,
    )

    weight = next(tensor for tensor in target.tensors if "weight" in tensor.names)
    assert transition.extra_gpu_bytes == 128
    assert [(allocation.nbytes, allocation.temporary_copy_bytes) for allocation in transition.host_allocations] == [
        (128, 128)
    ]
    assert transition.release_host_storage_ids == (weight.storage_id,)


def test_block_profile_rejects_distinct_tensor_views_sharing_storage() -> None:
    backing = torch.arange(8.0)
    model = torch.nn.Module()
    model.register_parameter("left", torch.nn.Parameter(backing[:4], requires_grad=False))
    model.register_parameter("right", torch.nn.Parameter(backing[2:6], requires_grad=False))
    tensors = dict(model.named_parameters())
    stats = [_stat(name, tensor) for name, tensor in tensors.items()]
    profile = TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={id(tensor): name for name, tensor in tensors.items()},
        allocation_ordered={0: ["layer"]},
        label_to_size_map={},
        block_sizes={0: sum(stat.size_bytes for stat in stats)},
        load_strategy={"layer": stats},
        release_strategy={"layer": stats},
        label_to_block_id={"layer": 0},
        stats=[LayerStatistics(label="layer", tensors=stats, duration=1.0)],
        transfer_to_compute_map={"layer": "layer"},
        view_tensors_ids=[id(tensor) for tensor in tensors.values()],
        view_tensors_names=list(tensors),
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )

    with pytest.raises(ValueError, match="distinct tensor views sharing storage"):
        target_from_profile(
            capture_model_state(model),
            profile,
            target_device="cuda:0",
            pinning="copy",
            use_shm=False,
        )


def test_shm_follower_does_not_allocate_local_host_block() -> None:
    model = _model()
    profile = _profile(model, loader_type="allocation_block_transfer")
    profile.shm_block_name_map = {"layer": "block"}

    _target, transition = target_from_profile(
        capture_model_state(model),
        profile,
        target_device="cuda:0",
        pinning="in_place",
        use_shm=True,
    )

    assert transition.host_allocations == ()


@pytest.mark.parametrize(
    ("reserves", "expected_capacity"),
    [
        ({}, MemoryCapacity(host_bytes=8 * 1024**3, gpu_bytes=24 * 1024**3)),
        (
            {"host_reserve_bytes": 1 * 1024**3, "gpu_reserve_bytes": 2 * 1024**3},
            MemoryCapacity(host_bytes=11 * 1024**3, gpu_bytes=23 * 1024**3),
        ),
    ],
)
def test_tensor_manager_is_only_a_runtime_convenience_wrapper(
    reserves: dict[str, int],
    expected_capacity: MemoryCapacity,
) -> None:
    from flextensor.tensor_manager import TensorManager

    gib = 1024**3
    model = _model()
    profile = _profile(model)
    current = capture_model_state(model)
    target = current
    transition = TransitionSpec()
    expected = StateTransitionPlan(migrations=(), pinning_groups=(), peak_host_bytes=0, peak_gpu_bytes=0)
    manager = TensorManager.__new__(TensorManager)
    manager.loader_type = "strategy"
    manager.device_gpu = torch.device("cuda:2")
    manager.pinned_memory = True
    manager.host_pinner = type("Pinner", (), {"registry": object()})()
    manager.use_shm = False

    with (
        patch("flextensor.tensor_manager.capture_model_state", return_value=current) as capture,
        patch("flextensor.tensor_manager.target_from_profile", return_value=(target, transition)) as adapt,
        patch("flextensor.tensor_manager.plan_state_transition", return_value=expected) as plan,
        patch(
            "flextensor.tensor_manager.psutil.virtual_memory",
            return_value=type("VM", (), {"available": 12 * gib})(),
        ),
        patch("torch.cuda.mem_get_info", return_value=(20 * gib, 80 * gib)),
        patch("torch.cuda.memory_reserved", return_value=7 * gib),
        patch("torch.cuda.memory_allocated", return_value=2 * gib),
    ):
        result = manager.plan_state_adoption(model, profile, **reserves)

    capture.assert_called_once_with(model)
    adapt.assert_called_once_with(
        current,
        profile,
        target_device="cuda:2",
        pinning="in_place",
        use_shm=False,
    )
    plan.assert_called_once_with(
        current,
        target,
        transition=transition,
        capacity=expected_capacity,
    )
    assert result is expected


@pytest.mark.parametrize(
    ("reserve_name", "reserve_value"),
    [
        ("host_reserve_bytes", -1),
        ("host_reserve_bytes", False),
        ("gpu_reserve_bytes", -1),
        ("gpu_reserve_bytes", True),
    ],
)
def test_tensor_manager_rejects_invalid_capacity_reserve(reserve_name: str, reserve_value: object) -> None:
    from flextensor.tensor_manager import TensorManager

    model = _model()
    profile = _profile(model)
    manager = TensorManager.__new__(TensorManager)
    manager.loader_type = "strategy"

    with pytest.raises(ValueError, match=f"{reserve_name} must be a non-negative integer"):
        manager.plan_state_adoption(model, profile, **{reserve_name: reserve_value})


def test_tensor_manager_rejects_profile_for_another_loader() -> None:
    from flextensor.tensor_manager import TensorManager

    model = _model()
    profile = _profile(model)
    manager = TensorManager.__new__(TensorManager)
    manager.loader_type = "allocation_block_transfer"

    with (
        patch("flextensor.tensor_manager.capture_model_state") as capture,
        pytest.raises(ValueError, match="matching transfer mode"),
    ):
        manager.plan_state_adoption(model, profile)

    capture.assert_not_called()
