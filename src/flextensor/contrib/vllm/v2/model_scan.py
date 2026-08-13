# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect a transformed vLLM model and classify tensors for offloading or residency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch
from torch import nn

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.contrib.vllm._patterns import VLLM_DEFAULT_EXCLUDE_PATTERNS
from flextensor.contrib.vllm.v2.errors import VllmFlexTensorV2Error
from flextensor.model_state_capture import LiveStorageInfo, LiveStorageKey, inspect_tensor_storage
from flextensor.tensor_discovery import get_non_offloaded_tensor_ids
from flextensor.tensor_processors import compute_reachable_tensor_map

TensorKind: TypeAlias = Literal["parameter", "buffer"]
LiveUnit: TypeAlias = tuple[int, int, nn.Module]


@dataclass(slots=True)
class CapturedTensor:
    tensor: torch.Tensor
    qualified_names: list[str]
    kind: TensorKind


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    label: str
    qualified_modules: tuple[tuple[str, nn.Module], ...]


@dataclass(frozen=True, slots=True)
class LoadedModelScan:
    layer_stats: list[LayerStatistics]
    name_by_tensor_id: dict[int, str]
    storage_by_tensor_id: dict[int, LiveStorageInfo]
    gpu_constant_ids: frozenset[int]


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _resolve_units(root: nn.Module, live_units: tuple[LiveUnit, ...]) -> list[ResolvedUnit]:
    if not live_units:
        raise VllmFlexTensorV2Error("cannot scan model before observing at least one wrapped unit")

    qualified_modules_by_id: dict[int, list[tuple[str, nn.Module]]] = {}
    for path, module in root.named_modules(remove_duplicate=False):
        qualified_modules_by_id.setdefault(id(module), []).append((path, module))

    labels: set[str] = set()
    resolved_units = []
    for wrap_call_index, module_index, module in live_units:
        qualified_modules = tuple(qualified_modules_by_id.get(id(module), ()))
        if not qualified_modules:
            raise VllmFlexTensorV2Error(
                "wrapped module is not reachable from the supplied model root: "
                f"coordinate={(wrap_call_index, module_index)} module_type={_type_name(module)} "
                f"root_type={_type_name(root)}; the unit may belong to a separately loaded model or drafter, "
                "the wrong root may have been supplied, or the module may be held outside the registered "
                "nn.Module tree"
            )
        label = qualified_modules[0][0]
        if not label:
            raise VllmFlexTensorV2Error(
                f"transformed runtime label is empty: coordinate={(wrap_call_index, module_index)}"
            )
        if label in labels:
            raise VllmFlexTensorV2Error(f"transformed runtime label is duplicate: label={label!r}")
        labels.add(label)
        resolved_units.append(ResolvedUnit(label=label, qualified_modules=qualified_modules))
    return resolved_units


def _capture_tensors(root: nn.Module) -> dict[int, CapturedTensor]:
    captured: dict[int, CapturedTensor] = {}
    for kind, named_tensors in (
        ("parameter", root.named_parameters(remove_duplicate=False)),
        ("buffer", root.named_buffers(remove_duplicate=False)),
    ):
        tensor_kind = cast("TensorKind", kind)
        for name, tensor in named_tensors:
            tensor_id = id(tensor)
            entry = captured.get(tensor_id)
            if entry is None:
                captured[tensor_id] = CapturedTensor(tensor=tensor, qualified_names=[name], kind=tensor_kind)
            else:
                entry.qualified_names.append(name)
                if kind == "buffer":
                    entry.kind = tensor_kind
    return captured


def _inspect_storages(
    root: nn.Module,
    captured: dict[int, CapturedTensor],
    normalized_device: torch.device,
) -> tuple[dict[int, LiveStorageInfo], dict[LiveStorageKey, set[int]]]:
    storage_by_tensor_id: dict[int, LiveStorageInfo] = {}
    tensor_ids_by_storage_key: dict[LiveStorageKey, set[int]] = {}
    for tensor_id, captured_tensor in captured.items():
        tensor = captured_tensor.tensor
        qualified_names = tuple(captured_tensor.qualified_names)
        try:
            inspection = inspect_tensor_storage(str(qualified_names), tensor)
        except ValueError as exc:
            raise VllmFlexTensorV2Error(
                f"cannot inspect transformed tensor storage: qualified_names={qualified_names}: {exc}"
            ) from exc
        if tensor.device.type != "cpu" and tensor.device != normalized_device:
            raise VllmFlexTensorV2Error(
                "transformed tensor is on an unsupported device: "
                f"qualified_names={qualified_names} device={tensor.device} target={normalized_device}"
            )
        storage_by_tensor_id[tensor_id] = inspection
        tensor_ids_by_storage_key.setdefault(inspection.key, set()).add(tensor_id)

    for tensor_id, tensor in compute_reachable_tensor_map(root).items():
        if tensor_id in captured:
            continue
        try:
            inspection = inspect_tensor_storage("<reachable tensor>", tensor)
        except ValueError as exc:
            raise VllmFlexTensorV2Error(
                f"cannot inspect reachable transformed tensor storage: tensor_id={tensor_id}: {exc}"
            ) from exc
        tensor_ids_by_storage_key.setdefault(inspection.key, set()).add(tensor_id)
    return storage_by_tensor_id, tensor_ids_by_storage_key


def _classify_tensors(
    resolved_units: list[ResolvedUnit],
    captured: dict[int, CapturedTensor],
    storage_by_tensor_id: dict[int, LiveStorageInfo],
    tensor_ids_by_storage_key: dict[LiveStorageKey, set[int]],
    known_resident_ids: set[int],
) -> tuple[list[LayerStatistics], frozenset[int]]:
    unit_indexes_by_tensor_id: dict[int, set[int]] = {}
    qualified_names_by_unit_and_tensor_id: dict[tuple[int, int], set[str]] = {}
    for unit_index, unit in enumerate(resolved_units):
        for path, module in unit.qualified_modules:
            for local_name, parameter in module.named_parameters(remove_duplicate=False):
                qualified_name = f"{path}.{local_name}" if path else local_name
                tensor_id = id(parameter)
                unit_indexes_by_tensor_id.setdefault(tensor_id, set()).add(unit_index)
                qualified_names_by_unit_and_tensor_id.setdefault((unit_index, tensor_id), set()).add(qualified_name)

    tensors_by_unit: list[list[TensorStatistics]] = [[] for _unit in resolved_units]
    gpu_constant_ids: set[int] = set()
    for tensor_id, captured_tensor in captured.items():
        unit_indexes = unit_indexes_by_tensor_id.get(tensor_id, set())
        storage_key = storage_by_tensor_id[tensor_id].key
        # Eligible tensors are parameters in exactly one unit, with exclusive storage and every alias in that unit.
        eligible = (
            captured_tensor.kind == "parameter"
            and tensor_id not in known_resident_ids
            and len(unit_indexes) == 1
            and len(tensor_ids_by_storage_key[storage_key]) == 1
        )
        if eligible:
            unit_index = next(iter(unit_indexes))
            eligible = set(captured_tensor.qualified_names).issubset(
                qualified_names_by_unit_and_tensor_id[unit_index, tensor_id]
            )
        if not eligible:
            gpu_constant_ids.add(tensor_id)
            continue
        tensors_by_unit[unit_index].append(
            TensorStatistics(
                tensor_id=tensor_id,
                name=captured_tensor.qualified_names[0],
                size_bytes=captured_tensor.tensor.numel() * captured_tensor.tensor.element_size(),
                load_time_ms=0.0,
            )
        )

    layer_stats = [
        LayerStatistics(label=unit.label, tensors=tensors, duration=1.0)
        for unit, tensors in zip(resolved_units, tensors_by_unit, strict=True)
        if tensors
    ]
    if not layer_stats:
        raise VllmFlexTensorV2Error(
            "no runtime-observed offloadable tensors remain after applying include/exclude patterns and "
            "tensor safety checks; select parameters owned by vLLM-wrapped units"
        )
    return layer_stats, frozenset(gpu_constant_ids)


def scan_loaded_model(
    root: nn.Module,
    live_units: tuple[LiveUnit, ...],
    device_gpu: torch.device,
    *,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> LoadedModelScan:
    """Return transient stats, names, storage values, and final-GPU constant IDs."""
    effective_include_patterns = ["*"] if include_patterns is None else include_patterns
    effective_exclude_patterns = VLLM_DEFAULT_EXCLUDE_PATTERNS if exclude_patterns is None else exclude_patterns
    resolved_units = _resolve_units(root, live_units)
    captured = _capture_tensors(root)
    known_resident_ids = get_non_offloaded_tensor_ids(
        root,
        {
            tensor_id: captured_tensor.tensor
            for tensor_id, captured_tensor in captured.items()
            if captured_tensor.kind == "parameter"
        },
        include_patterns=effective_include_patterns,
        exclude_patterns=effective_exclude_patterns,
    )
    storage_by_tensor_id, tensor_ids_by_storage_key = _inspect_storages(root, captured, device_gpu)
    layer_stats, gpu_constant_ids = _classify_tensors(
        resolved_units,
        captured,
        storage_by_tensor_id,
        tensor_ids_by_storage_key,
        known_resident_ids,
    )
    return LoadedModelScan(
        layer_stats=layer_stats,
        # PyTorch's first registered alias is the canonical serialized name, matching deduplicated traversal.
        name_by_tensor_id={
            tensor_id: captured_tensor.qualified_names[0] for tensor_id, captured_tensor in captured.items()
        },
        storage_by_tensor_id=storage_by_tensor_id,
        gpu_constant_ids=gpu_constant_ids,
    )
