# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import types

import torch
from torch import nn

from flextensor.offload_manager import OffloadManager
from flextensor.strategy import BlockStrategyData, StrategyResult


class _OnlineQuantMethod:
    uses_meta_device = True

    def process_weights_after_loading(self, module: nn.Module):
        if error := getattr(self, "error", None):
            raise error

        old_weight = module.weight
        module.weight = nn.Parameter(torch.ones_like(old_weight, device="cpu"))
        module.register_parameter(
            "weight_scale",
            nn.Parameter(torch.ones(1, dtype=old_weight.dtype)),
        )
        return getattr(self, "result", None)


class SelectTensorStrategy:
    def __init__(self, selected_ids: set[int]) -> None:
        self.selected_ids = selected_ids
        self.calls = []

    def compute(self, layer_stats, memory_stats=None, max_gpu_mem_bytes=None):
        self.calls.append((layer_stats, memory_stats, max_gpu_mem_bytes))
        return StrategyResult(
            strategy_map={
                layer.label: [tensor for tensor in layer.tensors if tensor.tensor_id in self.selected_ids]
                for layer in layer_stats
            }
        )


class MalformedResultStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def compute(self, layer_stats, memory_stats=None, max_gpu_mem_bytes=None):
        self.calls += 1
        return None


class _RecordingOffloadManager(OffloadManager):
    def __init__(self) -> None:
        super().__init__("vllm-bootstrap-runtime")
        self.runtime_calls: list[str] = []

    def sync_prev_onload(self) -> None:
        self.runtime_calls.append("sync_prev_onload")

    def join_after_forward(self) -> None:
        self.runtime_calls.append("join_after_forward")


class RecordingStateStrategy:
    def __init__(self, loader_type: str, *, failure: str | None = None) -> None:
        self.loader_type = loader_type
        self.failure = failure
        self.calls = []

    # ruff: ignore[noqa-comments] - compatibility with the pre-commit Ruff version.
    def compute(  # noqa: C901 - malformed fixtures exercise validation branches.
        self,
        layer_stats,
        memory_stats=None,
        max_gpu_mem_bytes=None,
    ):
        self.calls.append((layer_stats, memory_stats, max_gpu_mem_bytes))
        transfer_label = layer_stats[0].label
        selected = layer_stats[1].tensors[0].model_copy()
        if self.failure == "unknown_label":
            transfer_label = "unknown"
        elif self.failure == "unknown_id":
            selected = selected.model_copy(update={"tensor_id": -1})
        elif self.failure == "conflicting_metadata":
            selected = selected.model_copy(update={"size_bytes": selected.size_bytes + 1})

        block_data = None
        if self.loader_type != "strategy" and self.failure != "missing_block_data":
            block_data = BlockStrategyData(
                label_to_size_map={transfer_label: selected.size_bytes},
                allocation_ordered={0: [transfer_label]},
                block_sizes=[selected.size_bytes],
                label_to_block_id={transfer_label: 0},
                transfer_to_compute_map={transfer_label: layer_stats[1].label},
            )
            if self.failure == "wrong_block_types":
                block_data.allocation_ordered = []
            elif self.failure == "label_size_mismatch":
                block_data.label_to_size_map[transfer_label] -= 1
            elif self.failure == "undersized_block":
                block_data.block_sizes[0] -= 1
            elif self.failure == "missing_allocation":
                block_data.allocation_ordered = {}
            elif self.failure == "duplicate_allocation":
                block_data.allocation_ordered = {0: [transfer_label], 1: [transfer_label]}
            elif self.failure == "inconsistent_block_id":
                block_data.label_to_block_id[transfer_label] = 1
            elif self.failure == "transfer_mapping_mismatch":
                block_data.transfer_to_compute_map[transfer_label] = layer_stats[0].label
        return StrategyResult(strategy_map={transfer_label: [selected]}, block_data=block_data)


class InvalidBlockIdStrategy(RecordingStateStrategy):
    def __init__(self, surface: str, invalid_block_id: object) -> None:
        super().__init__("raw_block_transfer")
        self.surface = surface
        self.invalid_block_id = invalid_block_id

    def compute(self, layer_stats, memory_stats=None, max_gpu_mem_bytes=None):
        result = super().compute(layer_stats, memory_stats, max_gpu_mem_bytes)
        block_data = result.block_data
        assert block_data is not None
        transfer_label = layer_stats[0].label
        if self.surface == "allocation_ordered":
            block_data.allocation_ordered = {self.invalid_block_id: [transfer_label]}
        elif self.surface == "label_to_block_id":
            block_data.label_to_block_id[transfer_label] = self.invalid_block_id
        else:
            block_data.block_sizes = {self.invalid_block_id: layer_stats[1].tensors[0].size_bytes}
        return result


class _InventoryRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_only = nn.Parameter(torch.empty(5))

        self.first = nn.Module()
        self.first.weight = nn.Parameter(torch.empty(2, 3))
        self.first.weight_alias = self.first.weight
        self.first.dense = nn.Parameter(torch.empty(4))
        indices = torch.tensor([[0], [1]])
        values = torch.tensor([1.0])
        self.first.sparse = nn.Parameter(torch.sparse_coo_tensor(indices, values, (2, 2)))
        self.first.register_buffer("cache", torch.empty(2))

        self.second = nn.Module()
        self.second.shared = self.first.weight
        self.second.dense = nn.Parameter(torch.empty(3))

        self.quantized = nn.Module()
        self.quantized.projection = nn.Linear(2, 2, bias=False)
        self.quantized.quantizer = nn.Module()
        self.quantized.quantizer.quant_method = _OnlineQuantMethod()
        self.quantized_alias = self.quantized


def _module_with_tensor(
    name: str,
    *,
    kind: str = "parameter",
    shape: tuple[int, ...] = (2, 2),
    dtype: torch.dtype = torch.float32,
    transposed: bool = False,
) -> nn.Module:
    module = nn.Module()
    tensor = torch.empty(tuple(reversed(shape)), dtype=dtype).t() if transposed else torch.empty(shape, dtype=dtype)
    if kind == "parameter":
        setattr(module, name, nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)
    return module


def _online_quant_unit(*, auxiliary_size: int = 0) -> nn.Module:
    unit = nn.Module()
    unit.first = _module_with_tensor("weight", shape=(3,), dtype=torch.bfloat16)
    unit.first.quant_method = _OnlineQuantMethod()
    if auxiliary_size:
        unit.first.bias = nn.Parameter(torch.empty(auxiliary_size, dtype=torch.bfloat16))
    unit.second = _module_with_tensor("weight", shape=(5,), dtype=torch.bfloat16)
    unit.second.quant_method = _OnlineQuantMethod()
    unit.register_buffer("cache", torch.empty(7, dtype=torch.bfloat16))
    return unit


def _meta_online_quant_unit() -> nn.Module:
    with torch.device("meta"):
        unit = _online_quant_unit()
    unit.cache = torch.empty(7, dtype=torch.bfloat16)
    return unit


def _set_available_memory(monkeypatch, bootstrap_module, *, gpu: int, host: int) -> None:
    memory = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=host))
    monkeypatch.setattr(bootstrap_module, "psutil", memory, raising=False)
    monkeypatch.setattr(bootstrap_module.state_builder, "psutil", memory)


def _state_root() -> nn.Module:
    root = nn.Module()
    root.root_only = nn.Parameter(torch.empty(1))
    root.first = _module_with_tensor("weight", shape=(2,))
    root.first.weight_alias = root.first.weight
    root.first.register_buffer("cache", torch.empty(1))
    root.first_alias = root.first
    root.second = _module_with_tensor("weight", shape=(3,))
    shared = nn.Parameter(torch.empty(1))
    root.first.cross_unit = shared
    root.second.cross_unit = shared
    backing = torch.empty(8)
    root.first.left_view = nn.Parameter(backing[:4])
    root.first.right_view = nn.Parameter(backing[4:])
    return root


def _state_offloader(bootstrap_module, monkeypatch, *, complete: bool = True):
    root = _state_root()
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([root.first, root.second]))
    _set_available_memory(monkeypatch, bootstrap_module, gpu=1 << 20, host=1 << 20)
    if complete:
        offloader.post_init()
    return offloader, root


def _scan_loaded_model(bootstrap_module, offloader, model, device):
    resolved_device = bootstrap_module.state_builder.resolve_cuda_device(device)
    return bootstrap_module.model_scan.scan_loaded_model(
        model,
        tuple(offloader._live_units),
        resolved_device,
    )


def _set_cuda_snapshot(monkeypatch, bootstrap_module, *, available: int, total: int) -> None:
    monkeypatch.setattr(
        bootstrap_module.state_builder.CUDAMemorySnapshot,
        "capture",
        classmethod(
            lambda cls, _device: cls(
                free_bytes=available,
                total_bytes=total,
                reserved_bytes=0,
                allocated_bytes=0,
            )
        ),
    )


def _assert_value_only(value: object) -> None:
    assert not isinstance(value, (nn.Module, torch.Tensor))
    if hasattr(value, "model_dump"):
        _assert_value_only(value.model_dump())
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _assert_value_only(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_value_only(key)
            _assert_value_only(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _assert_value_only(item)
    else:
        assert value is None or isinstance(value, (bool, int, float, str, torch.device, torch.dtype, torch.layout))
