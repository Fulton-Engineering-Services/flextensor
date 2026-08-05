# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch adapter for serializable model placement capture."""

from __future__ import annotations

import torch

from flextensor.state_transition import ModelPlacementState, StorageState, TensorState
from flextensor.tensor_processors import compute_reachable_tensor_map


def _inspect_storage(name: str, tensor: torch.Tensor) -> tuple[tuple[str, int], int, bool]:
    """Return ``((device, storage identity), allocation bytes, pinned)``."""
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
    return (str(tensor.device), storage._cdata), storage.nbytes(), tensor.device.type == "cpu" and tensor.is_pinned()  # noqa: SLF001


def _storage_id_from_key(storage_key: tuple[str, int]) -> str:
    """Return an opaque process-local ID for one inspected storage."""
    device, storage_impl = storage_key
    return f"storage:{device}:{storage_impl}"


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

    storage_keys: set[tuple[str, int]] = set()
    storages: list[StorageState] = []
    tensors: list[TensorState] = []
    for index, (tensor, names, kind) in enumerate(captured.values()):
        display_name = names[0] if names else f"<reachable:{index}>"
        storage_key, nbytes, pinned = _inspect_storage(display_name, tensor)
        storage_id = _storage_id_from_key(storage_key)
        if storage_key not in storage_keys:
            storage_keys.add(storage_key)
            storages.append(
                StorageState(
                    id=storage_id,
                    device=storage_key[0],
                    nbytes=nbytes,
                    pinned=pinned,
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
