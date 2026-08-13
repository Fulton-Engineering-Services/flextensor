# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import FrozenInstanceError

import pytest
import torch

from flextensor import model_state_capture
from flextensor.model_state_capture import LiveStorageKey, capture_model_state


def test_live_storage_key_is_nominal_frozen_and_hashable() -> None:
    key = LiveStorageKey(device=torch.device("cpu"), storage_impl_id=17)

    assert key == LiveStorageKey(device=torch.device("cpu"), storage_impl_id=17)
    assert key != (torch.device("cpu"), 17)
    assert {key: "storage"}[LiveStorageKey(device=torch.device("cpu"), storage_impl_id=17)] == "storage"
    with pytest.raises(FrozenInstanceError):
        key.device = torch.device("cuda:0")  # type: ignore[misc]
    assert not hasattr(key, "__dict__")


def test_inspect_tensor_storage_returns_named_storage_details() -> None:
    tensor = torch.ones(3)

    inspection = model_state_capture.inspect_tensor_storage("tensor", tensor)

    assert inspection.key == LiveStorageKey(
        device=torch.device("cpu"),
        storage_impl_id=tensor.untyped_storage()._cdata,
    )
    assert not hasattr(inspection, "__dict__")
    assert inspection.key.device == tensor.device
    assert inspection.key.storage_impl_id == tensor.untyped_storage()._cdata
    assert inspection.nbytes == tensor.untyped_storage().nbytes()
    assert not inspection.pinned


def test_capture_preserves_names_views_buffers_and_reachable_tensors() -> None:
    backing = torch.arange(8.0)
    shared = torch.nn.Parameter(backing[:4], requires_grad=False)
    view = torch.nn.Parameter(backing[2:6], requires_grad=False)
    model = torch.nn.Module()
    model.left = torch.nn.Module()
    model.right = torch.nn.Module()
    model.left.register_parameter("weight", shared)
    model.right.register_parameter("weight", shared)
    model.register_parameter("view", view)
    model.register_buffer("constant", torch.ones(3))
    model.extra = torch.arange(2.0)

    state = capture_model_state(model)

    by_names = {tensor.names: tensor for tensor in state.tensors}
    shared_state = by_names["left.weight", "right.weight"]
    view_state = by_names["view",]
    assert shared_state.id != view_state.id
    assert shared_state.storage_id == view_state.storage_id
    assert shared_state.logical_bytes == shared.numel() * shared.element_size()
    storage = next(storage for storage in state.storages if storage.id == shared_state.storage_id)
    assert storage.nbytes == backing.untyped_storage().nbytes()
    assert by_names["constant",].kind == "buffer"
    assert any(tensor.names == () and tensor.kind == "tensor" for tensor in state.tensors)
    json.dumps(state.to_dict())


def test_capture_keeps_distinct_empty_storages_separate() -> None:
    model = torch.nn.Module()
    model.register_parameter("first", torch.nn.Parameter(torch.empty(0), requires_grad=False))
    model.register_parameter("second", torch.nn.Parameter(torch.empty(0), requires_grad=False))

    state = capture_model_state(model)

    storage_ids = {tensor.storage_id for tensor in state.tensors}
    assert len(storage_ids) == 2
    assert all(storage.nbytes == 0 for storage in state.storages)


def test_capture_treats_parameter_buffer_alias_as_buffer() -> None:
    model = torch.nn.Module()
    shared = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    model.register_parameter("weight", shared)
    model.register_buffer("constant", shared)

    state = capture_model_state(model)

    assert len(state.tensors) == 1
    assert state.tensors[0].names == ("constant", "weight")
    assert state.tensors[0].kind == "buffer"


@pytest.mark.parametrize(
    "tensor",
    [
        torch.empty(2, device="meta"),
        torch.sparse_coo_tensor(indices=[[0]], values=[1.0], size=(2,)),
    ],
    ids=["meta", "sparse"],
)
def test_capture_rejects_uninspectable_registered_tensor(tensor: torch.Tensor) -> None:
    model = torch.nn.Module()
    model.register_buffer("value", tensor)

    with pytest.raises(ValueError, match="value"):
        capture_model_state(model)
