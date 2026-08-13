# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct-mode discovery coverage for non-torch custom-kernel call sites."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

import flextensor as ft
from flextensor import loaders
from flextensor.collectors import IterativeLayerStatistics
from flextensor.helpers import TrapNestingGuard
from flextensor.loaders import TensorLayerLoader, WarmupDirectTensorLoader, _RawTensorDataBinder
from flextensor.tensor_manager import TensorManager, extend_nn_module
from flextensor.trap_tensor_mode import WarmupTrap, WarmupTrapDirect


class _PlainKernelLayer(nn.Module):
    """Layer that passes a raw parameter to a Python call site.

    This simulates custom launchers such as Triton kernels: the tensor is read
    from ``self.weight`` and passed along without going through a torch op that
    ``TorchFunctionMode`` could rewrite.
    """

    def __init__(self, seen: list[torch.Tensor]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.seen = seen

    def forward(self) -> torch.Tensor:
        weight = self.weight
        self.seen.append(weight)
        return weight


class _TensorManagerStub(TensorManager):
    def __init__(self, loader: Any, traced_ids: set[int]) -> None:
        self.tensor_layer_loader = loader
        self.device_gpu = loader.device_gpu
        self.traced_ids = traced_ids
        self.trap_nesting_guard = TrapNestingGuard()
        self.module_tracker = None
        self.recorded: list[tuple[str, set[int]]] = []

    def is_traced_by_id(self, tensor_id: int) -> bool:
        return tensor_id in self.traced_ids

    def is_traced(self, tensor: torch.Tensor) -> bool:
        return id(tensor) in self.traced_ids

    def record_tensors(self, label: str, tensor_ids: list[int] | set[int]) -> None:
        self.recorded.append((label, set(tensor_ids)))


class _AccessReportingLoader:
    device_gpu = torch.device("cpu")

    def __init__(self, accessed_ids: set[int]) -> None:
        self.accessed_ids = accessed_ids

    def enter(self, label: str) -> None:
        pass

    def exit(self, label: str) -> None:
        pass

    def get_label_tensor_ids(self, label: str) -> set[int]:
        return set()

    def get_accessed_tensor_ids(self, label: str) -> set[int]:
        return set(self.accessed_ids)


class _CrossLayerReader(nn.Module):
    def __init__(self, peer: nn.Module) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.peer = peer

    def forward(self) -> torch.Tensor:
        return cast("torch.Tensor", self.peer.weight)


def test_warmup_direct_trap_materializes_raw_parameter_access() -> None:
    seen: list[torch.Tensor] = []
    layer = _PlainKernelLayer(seen)
    weight_id = id(layer.weight)

    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"layer": {weight_id}},
        tensors_map={weight_id: layer.weight},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, {weight_id})
    original_data_ptr = layer.weight.data_ptr()

    assert layer() is layer.weight
    seen.clear()

    with WarmupTrapDirect(tensor_manager, "layer", torch.device("cpu")):
        materialized = layer()
        active_data_ptr = materialized.data_ptr()
        assert active_data_ptr != original_data_ptr

    assert seen == [materialized]
    assert materialized is layer.weight
    assert materialized.device.type == "cpu"
    assert materialized.data_ptr() == original_data_ptr
    assert tensor_manager.recorded == [("layer", {weight_id})]
    assert loader.cpu_to_gpu_map == {}


def test_warmup_direct_trap_restores_raw_parameter_on_exception() -> None:
    seen: list[torch.Tensor] = []
    layer = _PlainKernelLayer(seen)
    weight_id = id(layer.weight)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"layer": {weight_id}},
        tensors_map={weight_id: layer.weight},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, {weight_id})
    original_data_ptr = layer.weight.data_ptr()

    with pytest.raises(RuntimeError, match="boom"), WarmupTrapDirect(tensor_manager, "layer", torch.device("cpu")):
        materialized = layer()
        assert materialized.data_ptr() != original_data_ptr
        raise RuntimeError("boom")

    assert layer.weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_warmup_direct_trap_rolls_back_partial_enter_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _PlainKernelLayer([])
    second = _PlainKernelLayer([])
    tensor_ids = {id(first.weight), id(second.weight)}
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"layer": tensor_ids},
        tensors_map={id(first.weight): first.weight, id(second.weight): second.weight},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, tensor_ids)
    original_ptrs = {
        id(first.weight): first.weight.data_ptr(),
        id(second.weight): second.weight.data_ptr(),
    }

    original_bind = loader._data_binder.bind
    bind_calls = 0

    def fail_second_bind(tensor_id: int, active_tensor: torch.Tensor | None) -> None:
        nonlocal bind_calls
        bind_calls += 1
        if bind_calls == 2:
            raise RuntimeError("second bind failed")
        original_bind(tensor_id, active_tensor)

    monkeypatch.setattr(loader._data_binder, "bind", fail_second_bind)

    with (
        pytest.raises(RuntimeError, match="second bind failed"),
        WarmupTrapDirect(
            tensor_manager,
            "layer",
            torch.device("cpu"),
        ),
    ):
        pass

    assert bind_calls == 2
    assert first.weight.data_ptr() == original_ptrs[id(first.weight)]
    assert second.weight.data_ptr() == original_ptrs[id(second.weight)]
    assert loader.cpu_to_gpu_map == {}
    assert loader._active_counts == {}


def test_warmup_direct_trap_rejects_nested_traps_without_disturbing_outer_binding() -> None:
    layer = _PlainKernelLayer([])
    weight_id = id(layer.weight)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"outer": {weight_id}, "inner": {weight_id}},
        tensors_map={weight_id: layer.weight},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, {weight_id})
    original_data_ptr = layer.weight.data_ptr()

    with WarmupTrapDirect(tensor_manager, "outer", torch.device("cpu")):
        outer_data_ptr = layer.weight.data_ptr()
        assert outer_data_ptr != original_data_ptr

        with (
            pytest.raises(RuntimeError, match="Nested traps are not supported"),
            WarmupTrapDirect(
                tensor_manager,
                "inner",
                torch.device("cpu"),
            ),
        ):
            pass

        assert layer.weight.data_ptr() == outer_data_ptr

    assert layer.weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_warmup_direct_trap_rejects_cross_type_nesting() -> None:
    layer = _PlainKernelLayer([])
    weight_id = id(layer.weight)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"direct": {weight_id}},
        tensors_map={weight_id: layer.weight},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, {weight_id})
    original_data_ptr = layer.weight.data_ptr()

    with WarmupTrapDirect(tensor_manager, "direct", torch.device("cpu")):
        outer_data_ptr = layer.weight.data_ptr()
        assert outer_data_ptr != original_data_ptr

        with (
            pytest.raises(RuntimeError, match="Nested traps are not supported"),
            WarmupTrap(tensor_manager, "warmup", torch.device("cpu")),
        ):
            pass

        assert layer.weight.data_ptr() == outer_data_ptr

    assert layer.weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_warmup_direct_loader_refcounts_overlapping_labels() -> None:
    seen: list[torch.Tensor] = []
    layer = _PlainKernelLayer(seen)
    weight_id = id(layer.weight)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"outer": {weight_id}, "inner": {weight_id}},
        tensors_map={weight_id: layer.weight},
        device_gpu=torch.device("cpu"),
    )
    original_data_ptr = layer.weight.data_ptr()

    loader.enter("outer")
    outer_data_ptr = layer.weight.data_ptr()
    assert outer_data_ptr != original_data_ptr

    loader.enter("inner")
    assert layer.weight.data_ptr() == outer_data_ptr

    loader.exit("inner")
    assert layer.weight.data_ptr() == outer_data_ptr
    assert loader.get(weight_id) is not None

    loader.exit("outer")
    assert layer.weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_raw_tensor_binder_rejects_stride_mismatch() -> None:
    weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3).t())
    weight_id = id(weight)
    binder = _RawTensorDataBinder({weight_id: weight})
    contiguous_active = weight.detach().clone().contiguous()

    with pytest.raises(RuntimeError, match=r"stride.*does not match"):
        binder.bind(weight_id, contiguous_active)

    assert weight.stride() == (1, 3)


def test_warmup_direct_loader_rejects_distinct_parameters_sharing_storage() -> None:
    base = torch.ones(4)
    weight_a = nn.Parameter(base[:2])
    weight_b = nn.Parameter(base[:2])
    original_data_ptr = weight_a.data_ptr()
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"layer": {id(weight_a)}},
        tensors_map={id(weight_a): weight_a, id(weight_b): weight_b},
        device_gpu=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="share storage"):
        loader.enter("layer")

    assert weight_a.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_raw_tensor_binder_allows_disjoint_views_of_packed_storage() -> None:
    packed = torch.arange(6, dtype=torch.float32)
    weight_a = nn.Parameter(packed[:2], requires_grad=False)
    weight_b = nn.Parameter(packed[2:], requires_grad=False)
    original_a = weight_a.detach().clone()
    original_b = weight_b.detach().clone()
    active_a = torch.full_like(weight_a, 9)
    binder = _RawTensorDataBinder({id(weight_a): weight_a, id(weight_b): weight_b})

    binder.bind(id(weight_a), active_a)

    torch.testing.assert_close(weight_a, active_a)
    torch.testing.assert_close(weight_b, original_b)
    binder.restore_all()
    torch.testing.assert_close(weight_a, original_a)
    torch.testing.assert_close(weight_b, original_b)


def test_warmup_direct_loader_preserves_requires_grad_grad_and_stride() -> None:
    weight = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3).t(), requires_grad=True)
    weight.grad = torch.ones_like(weight)
    weight_id = id(weight)
    original_data_ptr = weight.data_ptr()
    original_stride = weight.stride()
    original_grad = weight.grad
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"layer": {weight_id}},
        tensors_map={weight_id: weight},
        device_gpu=torch.device("cpu"),
    )

    loader.enter("layer")
    try:
        assert weight.requires_grad is True
        assert weight.grad is original_grad
        assert weight.stride() == original_stride
        assert weight.data_ptr() != original_data_ptr
    finally:
        loader.exit("layer")

    assert weight.requires_grad is True
    assert weight.grad is original_grad
    assert weight.stride() == original_stride
    assert weight.data_ptr() == original_data_ptr


def test_warmup_direct_loader_supports_tied_parameter_object() -> None:
    shared_weight = nn.Parameter(torch.ones(2))
    weight_id = id(shared_weight)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"first": {weight_id}, "second": {weight_id}},
        tensors_map={weight_id: shared_weight},
        device_gpu=torch.device("cpu"),
    )
    original_data_ptr = shared_weight.data_ptr()

    loader.enter("first")
    first_data_ptr = shared_weight.data_ptr()
    try:
        loader.enter("second")
        try:
            assert shared_weight.data_ptr() == first_data_ptr
        finally:
            loader.exit("second")
        assert shared_weight.data_ptr() == first_data_ptr
    finally:
        loader.exit("first")

    assert shared_weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_warmup_direct_loader_skips_raw_data_binding_while_compiling(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[torch.Tensor] = []
    layer = _PlainKernelLayer(seen)
    weight_id = id(layer.weight)
    original_data_ptr = layer.weight.data_ptr()

    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"layer": {weight_id}},
        tensors_map={weight_id: layer.weight},
        device_gpu=torch.device("cpu"),
    )

    monkeypatch.setattr(loaders, "_is_compiling", lambda: True)

    loader.enter("layer")
    try:
        active_tensor = loader.get(weight_id)
        assert active_tensor is not None
        assert active_tensor.data_ptr() != original_data_ptr
        assert layer.weight.data_ptr() == original_data_ptr
    finally:
        loader.exit("layer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
def test_tensor_layer_loader_materializes_raw_parameter_access_on_cuda() -> None:
    seen: list[torch.Tensor] = []
    layer = _PlainKernelLayer(seen)
    weight_id = id(layer.weight)
    original_data_ptr = layer.weight.data_ptr()

    loader = TensorLayerLoader(
        [IterativeLayerStatistics(label="layer", tensor_ids={weight_id}, duration=0.1)],
        {weight_id: layer.weight},
        torch.device("cuda"),
    )

    loader.enter("layer")
    try:
        materialized = layer()
        assert materialized is layer.weight
        assert materialized.device.type == "cuda"
        assert materialized.data_ptr() != original_data_ptr
    finally:
        loader.exit("layer")

    assert layer.weight.device.type == "cpu"
    assert layer.weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
def test_tensor_layer_loader_materializes_cross_layer_parameter_access_on_cuda() -> None:
    peer = _PlainKernelLayer([])
    peer_weight_id = id(peer.weight)
    original_data_ptr = peer.weight.data_ptr()

    loader = TensorLayerLoader(
        [IterativeLayerStatistics(label="reader", tensor_ids={peer_weight_id}, duration=0.1)],
        {peer_weight_id: peer.weight},
        torch.device("cuda"),
    )
    tensor_manager = _TensorManagerStub(loader, {peer_weight_id})
    reader = _CrossLayerReader(extend_nn_module(peer, tensor_manager))

    loader.enter("reader")
    try:
        materialized = reader()
        assert materialized.device.type == "cuda"
        assert materialized.data_ptr() != original_data_ptr
    finally:
        loader.exit("reader")

    assert peer.weight.data_ptr() == original_data_ptr
    assert loader.cpu_to_gpu_map == {}


def test_tensor_layer_loader_release_memory_releases_materialized_tensors() -> None:
    released: list[torch.Tensor] = []
    layer = _PlainKernelLayer([])
    weight_id = id(layer.weight)
    loader = TensorLayerLoader(
        [IterativeLayerStatistics(label="layer", tensor_ids={weight_id}, duration=0.1)],
        {weight_id: layer.weight},
        torch.device("cpu"),
        delete_tensor_func=released.append,
    )
    materialized = torch.ones(2)
    loader.cpu_to_gpu_map[weight_id] = materialized
    loader.model_ids = {weight_id}

    loader.release_memory()

    assert len(released) == 1
    assert released[0] is materialized
    assert loader.cpu_to_gpu_map == {}


def test_warmup_direct_trap_records_torch_dispatched_tensors() -> None:
    tensor = torch.ones(2)
    tensor_id = id(tensor)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"logits_processor": set()},
        tensors_map={tensor_id: tensor},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, {tensor_id})

    with WarmupTrapDirect(tensor_manager, "logits_processor", torch.device("cpu")):
        assert torch.equal(tensor + 1, torch.full_like(tensor, 2))

    assert tensor_manager.recorded == [("logits_processor", {tensor_id})]


def test_warmup_direct_trap_records_loader_reported_accessed_tensors() -> None:
    tensor_id = id(torch.ones(1))
    loader = _AccessReportingLoader({tensor_id})
    tensor_manager = _TensorManagerStub(loader, set())

    with WarmupTrapDirect(tensor_manager, "reader", torch.device("cpu")):
        pass

    assert tensor_manager.recorded == [("reader", {tensor_id})]


def test_warmup_direct_loader_tracks_cross_layer_getter_access() -> None:
    peer = _PlainKernelLayer([])
    peer_weight_id = id(peer.weight)
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"reader": set(), "peer": {peer_weight_id}},
        tensors_map={peer_weight_id: peer.weight},
        device_gpu=torch.device("cpu"),
    )

    loader.enter("reader")
    try:
        assert loader.get(peer_weight_id) is not None
        assert loader.get_accessed_tensor_ids("reader") == {peer_weight_id}
    finally:
        loader.exit("reader")


def test_warmup_direct_trap_records_cross_layer_parameter_access() -> None:
    peer = _PlainKernelLayer([])
    peer_weight_id = id(peer.weight)
    original_data_ptr = peer.weight.data_ptr()
    loader = WarmupDirectTensorLoader(
        label_to_tensor_ids={"reader": set(), "peer": {peer_weight_id}},
        tensors_map={peer_weight_id: peer.weight},
        device_gpu=torch.device("cpu"),
    )
    tensor_manager = _TensorManagerStub(loader, {peer_weight_id})
    reader = _CrossLayerReader(extend_nn_module(peer, tensor_manager))

    with WarmupTrapDirect(tensor_manager, "reader", torch.device("cpu")):
        materialized = reader()
        assert materialized.data_ptr() != original_data_ptr

    assert peer.weight.data_ptr() == original_data_ptr
    assert tensor_manager.recorded == [("reader", {peer_weight_id})]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
@pytest.mark.parametrize("skip_discovery", [False, True])
def test_offload_lifecycle_preserves_raw_parameter_access_on_cuda(skip_discovery: bool) -> None:
    """End-to-end raw-parameter access under both discovery paths.

    ``skip_discovery=False`` exercises the ``WarmupTrapDirect`` discovery
    iteration that materializes raw parameter storage during warmup;
    ``skip_discovery=True`` skips discovery entirely and
    relies on the static layer-stats seed. Both paths must land in
    inference with parameters visible on CUDA to the raw ``forward`` reads.
    """

    class RawLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weight = self.weight
            assert weight.device.type == "cuda"
            return x + weight

    class RawModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer1 = RawLayer()
            self.layer2 = RawLayer()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return cast("torch.Tensor", self.layer2(self.layer1(x)))

    model = RawModel().eval()
    original_layer1 = model.layer1
    config = ft.OffloadConfig(
        enabled=True,
        discovery_iters=1,
        profiling_iters=1,
        include_patterns=["layer1", "layer2"],
        pinned_memory=False,
        max_gpu_mem_fraction=0.5,
        skip_discovery=skip_discovery,
    )

    proxy = ft.offload(model, config)
    assert model.layer1 is original_layer1

    try:
        x = torch.ones(1, device="cuda")
        for _ in range(3):
            y = proxy(x)
            torch.cuda.synchronize()
            assert y.device.type == "cuda"
            assert torch.allclose(y, torch.full_like(y, 3.0))
    finally:
        ft.release()
