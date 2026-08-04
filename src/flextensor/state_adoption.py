# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlexTensor saved-profile adapter for neutral state-transition planning."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from flextensor.collectors import TensorStatistics  # noqa: TC001 - beartype needs the runtime symbol
from flextensor.state_handler import TensorManagerState  # noqa: TC001 - beartype needs the runtime symbol
from flextensor.state_transition import (
    HostAllocation,
    ModelPlacementState,
    StorageState,
    TransitionSpec,
)
from flextensor.utils import _compute_packed_byte_layout

PinningMode = Literal["none", "copy", "in_place"]


def _serialized_statistics(profile: TensorManagerState) -> list[TensorStatistics]:
    statistics = [stat for stats in profile.load_strategy.values() for stat in stats]
    statistics.extend(stat for stats in profile.release_strategy.values() for stat in stats)
    statistics.extend(stat for layer in profile.stats for stat in layer.tensors)
    return statistics


def _current_tensor_inventory(
    current: ModelPlacementState,
    profile: TensorManagerState,
) -> tuple[dict[str, int], set[str], set[str]]:
    saved_names = set(profile.tensor_id_to_name_map.values())
    tensor_by_name = {name: tensor for tensor in current.tensors for name in tensor.names}
    missing_names = sorted(saved_names - tensor_by_name.keys())
    unexpected_groups = sorted(
        tensor.names for tensor in current.tensors if tensor.names and saved_names.isdisjoint(tensor.names)
    )
    if missing_names or unexpected_groups:
        raise ValueError(
            f"State inventory mismatch: missing_from_model={missing_names}, unexpected_in_model={unexpected_groups}"
        )

    serialized_stats = _serialized_statistics(profile)
    logical_by_name = {name: tensor_by_name[name].logical_bytes for name in saved_names}
    for stat in serialized_stats:
        if stat.size_bytes != logical_by_name[stat.name]:
            raise ValueError(
                f"TensorStatistics.size_bytes drift for {stat.name}: "
                f"saved={stat.size_bytes}, current_logical={logical_by_name[stat.name]}"
            )

    managed_names = {stat.name for stats in profile.load_strategy.values() for stat in stats}
    managed_names.update(profile.view_tensors_names)
    managed_buffers = sorted(name for name in managed_names if tensor_by_name[name].kind == "buffer")
    if managed_buffers:
        raise ValueError(f"Registered buffer(s) must remain final-GPU, not managed: {managed_buffers}")

    return logical_by_name, saved_names, managed_names


def _strategy_gpu_bytes(
    profile: TensorManagerState,
    logical_by_name: dict[str, int],
    managed_names: set[str],
) -> int:
    # Tensors absent from the transfer schedule at first use are already resident on GPU.
    scheduled_names: set[str] = set()
    layer_tensor_names: set[str] = set()
    preload_names: set[str] = set()
    for layer in profile.stats:
        scheduled_names.update(stat.name for stat in profile.load_strategy.get(layer.label, ()))
        current_layer_names = {stat.name for stat in layer.tensors}
        layer_tensor_names.update(current_layer_names)
        preload_names.update(current_layer_names - scheduled_names)

    resident_names = (preload_names | (managed_names - layer_tensor_names)) & managed_names
    current_bytes = sum(logical_by_name[name] for name in resident_names)
    peak_bytes = current_bytes
    for layer in profile.stats:
        for stat in profile.load_strategy.get(layer.label, ()):
            if stat.name not in resident_names:
                resident_names.add(stat.name)
                current_bytes += logical_by_name[stat.name]
        peak_bytes = max(peak_bytes, current_bytes)
        for stat in profile.release_strategy.get(layer.label, ()):
            if stat.name in resident_names:
                resident_names.remove(stat.name)
                current_bytes -= logical_by_name[stat.name]
    return peak_bytes


def _aligned_label_bytes(
    load_stats: dict[str, list[TensorStatistics]],
    logical_by_name: dict[str, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, stats in load_stats.items():
        names = dict.fromkeys(stat.name for stat in stats)
        _, result[label] = _compute_packed_byte_layout(logical_by_name[name] for name in names)
    return result


def _block_requirements(
    profile: TensorManagerState,
    logical_by_name: dict[str, int],
    *,
    use_shm: bool,
) -> tuple[int, dict[str, int]]:
    load_stats = {label: stats for label, stats in profile.load_strategy.items() if stats}
    if use_shm and profile.shm_block_name_map is not None and set(profile.shm_block_name_map) != set(load_stats):
        raise ValueError(
            "Saved state mismatch: shm_block_name_map must cover exactly the nonempty load labels. "
            "Re-profile the model."
        )
    if profile.loader_type == "raw_block_transfer":
        return sum(profile.block_sizes.values()), dict(profile.label_to_size_map)
    label_bytes = _aligned_label_bytes(load_stats, logical_by_name)
    gpu_bytes = sum(max(label_bytes[label] for label in labels) for labels in profile.allocation_ordered.values())
    return gpu_bytes, label_bytes


def _target_storages(
    current: ModelPlacementState,
    saved_names: set[str],
    managed_names: set[str],
    *,
    target_device: str,
    pinning: PinningMode,
    strategy_loader: bool,
) -> tuple[tuple[StorageState, ...], set[str]]:
    destinations: dict[str, set[str]] = {storage.id: set() for storage in current.storages}
    for tensor in current.tensors:
        profile_names = saved_names.intersection(tensor.names)
        destination = "cpu" if profile_names & managed_names else target_device
        destinations[tensor.storage_id].add(destination)

    contradictory = sorted(storage_id for storage_id, devices in destinations.items() if len(devices) != 1)
    if contradictory:
        raise ValueError(f"Contradictory alias destinations for shared storage: {contradictory}")

    managed_storage_ids = {storage_id for storage_id, devices in destinations.items() if next(iter(devices)) == "cpu"}
    target_storages: list[StorageState] = []
    for storage in current.storages:
        device = next(iter(destinations[storage.id]))
        pinned = False
        if device == "cpu":
            pinned = storage.pinned or (strategy_loader and pinning != "none")
        target_storages.append(replace(storage, device=device, pinned=pinned))
    return tuple(target_storages), managed_storage_ids


def target_from_profile(
    current: ModelPlacementState,
    profile: TensorManagerState,
    *,
    target_device: str,
    pinning: PinningMode,
    use_shm: bool,
) -> tuple[ModelPlacementState, TransitionSpec]:
    """Translate a saved FlexTensor profile into neutral target and transition data."""
    profile.validate_internal()
    if not (
        isinstance(target_device, str)
        and target_device.startswith("cuda:")
        and target_device.removeprefix("cuda:").isdigit()
    ):
        raise ValueError(f"target_device must be an explicit CUDA device, got {target_device!r}")
    current_cuda_devices = {storage.device for storage in current.storages if storage.device.startswith("cuda:")}
    if current_cuda_devices - {target_device}:
        raise ValueError(
            f"Cross-device CUDA migration is not modeled by this planner: "
            f"{sorted(current_cuda_devices)} -> {target_device}"
        )
    if pinning not in {"none", "copy", "in_place"}:
        raise ValueError(f"Unknown pinning mode {pinning!r}")

    logical_by_name, saved_names, managed_names = _current_tensor_inventory(current, profile)
    strategy_loader = profile.loader_type == "strategy"
    target_storages, managed_storage_ids = _target_storages(
        current,
        saved_names,
        managed_names,
        target_device=target_device,
        pinning=pinning,
        strategy_loader=strategy_loader,
    )
    target = ModelPlacementState(tensors=current.tensors, storages=target_storages)

    if strategy_loader:
        extra_gpu_bytes = _strategy_gpu_bytes(profile, logical_by_name, managed_names)
        host_allocations: tuple[HostAllocation, ...] = ()
        release_ids: tuple[str, ...] = ()
    elif profile.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
        tensors_by_managed_storage: dict[str, list[tuple[str, ...]]] = {}
        for tensor in current.tensors:
            if tensor.storage_id in managed_storage_ids:
                tensors_by_managed_storage.setdefault(tensor.storage_id, []).append(tensor.names)
        shared_storage_views = {
            storage_id: names for storage_id, names in tensors_by_managed_storage.items() if len(names) > 1
        }
        if shared_storage_views:
            # Independent block slots would silently detach views that share one backing allocation.
            raise ValueError(
                f"Block loaders cannot preserve distinct tensor views sharing storage: {shared_storage_views}"
            )
        extra_gpu_bytes, label_bytes = _block_requirements(profile, logical_by_name, use_shm=use_shm)
        if profile.loader_type == "allocation_block_transfer":
            labels = [label for labels in profile.allocation_ordered.values() for label in labels]
            shm_follower = use_shm and profile.shm_block_name_map is not None
        else:
            labels = [label for label, stats in profile.load_strategy.items() if stats]
            shm_follower = False
        copy_blocks = pinning == "copy" and not (profile.loader_type == "allocation_block_transfer" and use_shm)
        # Packed host blocks coexist with standalone tensor homes until loader construction releases them.
        host_allocations = (
            ()
            if shm_follower
            else tuple(
                HostAllocation(
                    nbytes=label_bytes[label],
                    temporary_copy_bytes=label_bytes[label] if copy_blocks else 0,
                )
                for label in labels
            )
        )
        release_ids = tuple(storage.id for storage in current.storages if storage.id in managed_storage_ids)
    else:
        raise ValueError(f"Unknown loader type: {profile.loader_type}")

    target_storage_by_id = {storage.id: storage for storage in target.storages}
    pinning_copy_ids = tuple(
        storage.id
        for storage in current.storages
        if pinning == "copy" and not storage.pinned and target_storage_by_id[storage.id].pinned
    )
    return target, TransitionSpec(
        extra_gpu_bytes=extra_gpu_bytes,
        pinning_copy_storage_ids=pinning_copy_ids,
        host_allocations=host_allocations,
        release_host_storage_ids=release_ids,
    )
