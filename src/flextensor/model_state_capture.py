# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch adapter for serializable model placement capture."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flextensor.state_transition import ModelPlacementState, StorageState, TensorState
from flextensor.tensor_processors import compute_reachable_tensor_map


@dataclass(frozen=True, slots=True)
class LiveStorageKey:
    device: torch.device
    storage_impl_id: int


@dataclass(frozen=True, slots=True)
class LiveStorageInfo:
    key: LiveStorageKey
    nbytes: int
    pinned: bool


def inspect_tensor_storage(name: str, tensor: torch.Tensor) -> LiveStorageInfo:
    """Return the tensor's storage key, allocation bytes, and pinning state."""
    if tensor.is_meta:
        raise ValueError(f"Cannot capture meta tensor {name}")
    if tensor.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Cannot capture tensor {name} on unsupported device {tensor.device}")
    if tensor.layout != torch.strided:
        raise ValueError(
            f"Cannot capture tensor {name} with unsupported layout {tensor.layout}; only torch.strided is supported"
        )
    try:
        storage = tensor.untyped_storage()
    except (NotImplementedError, RuntimeError) as error:
        raise ValueError(f"Cannot inspect backing storage for tensor {name}") from error
    # Storage identity stays shared across views even when their data pointers start at different offsets.
    return LiveStorageInfo(
        key=LiveStorageKey(device=tensor.device, storage_impl_id=storage._cdata),  # noqa: SLF001
        nbytes=storage.nbytes(),
        pinned=tensor.device.type == "cpu" and tensor.is_pinned(),
    )


def _storage_id_from_key(storage_key: LiveStorageKey) -> str:
    """Return an opaque process-local ID for one inspected storage."""
    return f"storage:{storage_key.device}:{storage_key.storage_impl_id}"


def capture_model_state(model: torch.nn.Module) -> ModelPlacementState:
    """Capture tensor identity, aliases, storage size, and placement without mutation."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)}")

    # Python tensor identity -> (object, every registered path, classification).
    captured: dict[int, tuple[torch.Tensor, list[str], str]] = {}
    for kind, named_tensors in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in named_tensors:
            entry = captured.get(id(tensor))
            if entry is None:
                captured[id(tensor)] = (tensor, [name], kind)
            else:
                entry[1].append(name)
                if kind == "buffer":
                    # Buffers must remain final-GPU, so that classification wins for dual registration.
                    captured[id(tensor)] = (tensor, entry[1], kind)

    # Unregistered tensor attributes can still alias model storage and must participate in the plan.
    for tensor_id, tensor in compute_reachable_tensor_map(model).items():
        captured.setdefault(tensor_id, (tensor, [], "tensor"))

    storage_keys: set[LiveStorageKey] = set()
    storages: list[StorageState] = []
    tensors: list[TensorState] = []
    for index, (tensor, names, kind) in enumerate(captured.values()):
        display_name = names[0] if names else f"<reachable:{index}>"
        inspection = inspect_tensor_storage(display_name, tensor)
        storage_id = _storage_id_from_key(inspection.key)
        if inspection.key not in storage_keys:
            storage_keys.add(inspection.key)
            storages.append(
                StorageState(
                    id=storage_id,
                    device=str(inspection.key.device),
                    nbytes=inspection.nbytes,
                    pinned=inspection.pinned,
                )
            )
        tensors.append(
            TensorState(
                id=f"tensor:{index}",
                names=tuple(sorted(names)),
                storage_id=storage_id,
                logical_bytes=tensor.numel() * tensor.element_size(),
                kind=kind,
            )
        )

    return ModelPlacementState(tensors=tuple(tensors), storages=tuple(storages))
