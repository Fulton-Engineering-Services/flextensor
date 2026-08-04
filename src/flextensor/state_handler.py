# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State handler for TensorManager state serialization and restoration."""

import copy
import json
import logging
import pathlib
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.model_state_capture import capture_model_state
from flextensor.tensor_processors import preprocess_model
from flextensor.utils import atomic_write_json

logger = logging.getLogger(__name__)


@dataclass
class StateValidationResult:
    """
    Result of validating state compatibility between a saved state and current model.

    Follows the pattern of torch.nn.Module.load_state_dict() return value.
    """

    missing_keys: list[str] = field(default_factory=list)
    """Tensor names in the saved state that are not present in the current model."""

    unexpected_keys: list[str] = field(default_factory=list)
    """Tensor names in the current model that are not present in the saved state."""

    def __bool__(self) -> bool:
        """Return True if there are any missing or unexpected keys."""
        return bool(self.missing_keys or self.unexpected_keys)


@dataclass
class LoaderInputData:
    """Data needed to create a tensor loader.

    This dataclass provides a unified interface for loader creation, regardless of whether
    the data comes from a freshly computed strategy or a restored saved state.

    Use cases:
    - Fresh computation: Create from BlockStrategyData in prepare_infer_mode()
    - State restoration: Create from TensorManagerState.to_loader_input_data()
    """

    allocation_ordered: dict[int, list[str]] = field(default_factory=dict)
    label_to_block_id: dict[str, int] = field(default_factory=dict)
    transfer_to_compute_map: dict[str, str] = field(default_factory=dict)
    label_to_size_map: dict[str, int] = field(default_factory=dict)
    block_sizes: dict[int, int] = field(default_factory=dict)
    shm_block_name_map: dict[str, str] | None = None
    release_strategy: dict[str, list[TensorStatistics]] = field(default_factory=dict)


@dataclass
class TensorManagerState:
    """State container for TensorManager that can be serialized/deserialized.

    Note:
        ``pinned_memory_mode`` is host-side policy, not part of the saved
        plan — restored managers take it from their constructor argument.
    """

    # Schema version for serialization. Bump when the persisted structure changes.
    # See CHANGELOG.md for version history.
    SCHEMA_VERSION: ClassVar[int] = 3

    loader_type: str
    tensor_id_to_name_map: dict[int, str]
    allocation_ordered: dict[int, list[str]]
    label_to_size_map: dict[str, int]
    block_sizes: dict[int, int]
    load_strategy: dict[str, list[TensorStatistics]]
    release_strategy: dict[str, list[TensorStatistics]]
    label_to_block_id: dict[str, int]
    stats: list[LayerStatistics]
    transfer_to_compute_map: dict[str, str]

    view_tensors_ids: list[int]
    view_tensors_names: list[str]
    gpu_tensors_names: list[str]
    shm_block_name_map: dict[str, str] | None

    def validate_internal(self) -> None:  # noqa: C901
        """Validate consistency that does not depend on a live model or runtime."""

        def mismatch(detail: str) -> ValueError:
            return ValueError(f"Saved state mismatch: {detail}. Re-profile the model.")

        if self.loader_type not in {"strategy", "allocation_block_transfer", "raw_block_transfer"}:
            raise ValueError(f"Unknown loader type: {self.loader_type}")
        inventory_names = list(self.tensor_id_to_name_map.values())
        duplicate_names = sorted(name for name, count in Counter(inventory_names).items() if count > 1)
        if duplicate_names:
            raise mismatch(f"tensor_id_to_name_map inventory names must be unique; duplicates={duplicate_names}")

        statistics = [stat for stats in self.load_strategy.values() for stat in stats]
        statistics.extend(stat for stats in self.release_strategy.values() for stat in stats)
        statistics.extend(stat for layer in self.stats for stat in layer.tensors)
        for stat in statistics:
            inventory_name = self.tensor_id_to_name_map.get(stat.tensor_id)
            if inventory_name != stat.name:
                raise mismatch(
                    "TensorStatistics tensor_id/name pair disagrees with tensor_id_to_name_map: "
                    f"tensor_id={stat.tensor_id}, name={stat.name!r}, inventory_name={inventory_name!r}"
                )

        if len(self.view_tensors_ids) != len(self.view_tensors_names):
            raise mismatch("view_tensors_ids and view_tensors_names must have equal lengths")
        if len(set(self.view_tensors_ids)) != len(self.view_tensors_ids) or len(set(self.view_tensors_names)) != len(
            self.view_tensors_names
        ):
            raise mismatch("view_tensors_ids and view_tensors_names must each be unique")
        for tensor_id, name in zip(self.view_tensors_ids, self.view_tensors_names, strict=True):
            if self.tensor_id_to_name_map.get(tensor_id) != name:
                raise mismatch(
                    "view_tensors_ids/view_tensors_names pair disagrees with tensor_id_to_name_map: "
                    f"tensor_id={tensor_id}, name={name!r}"
                )

        saved_names = set(inventory_names)
        referenced_names = {stat.name for stat in statistics}
        referenced_names.update(self.view_tensors_names)
        referenced_names.update(self.gpu_tensors_names)
        invalid_references = sorted(referenced_names - saved_names)
        if invalid_references:
            raise mismatch(f"state references names outside saved inventory: {invalid_references}")

        managed_names = {stat.name for stats in self.load_strategy.values() for stat in stats}
        managed_names.update(self.view_tensors_names)
        final_gpu_names = saved_names - managed_names
        explicit_gpu_names = set(self.gpu_tensors_names)
        if explicit_gpu_names != final_gpu_names:
            raise mismatch(
                "gpu_tensors_names does not match computed final-GPU inventory: "
                f"missing={sorted(final_gpu_names - explicit_gpu_names)}, "
                f"unexpected={sorted(explicit_gpu_names - final_gpu_names)}"
            )

        if self.loader_type == "strategy":
            if self.release_strategy:
                loaded_names = {stat.name for stats in self.load_strategy.values() for stat in stats}
                released_names = {stat.name for stats in self.release_strategy.values() for stat in stats}
                missing_release_names = sorted(loaded_names - released_names)
                if missing_release_names:
                    raise mismatch(f"release_strategy has no release mapping for {missing_release_names}")
            return

        load_stats = {label: stats for label, stats in self.load_strategy.items() if stats}
        labels_by_name: dict[str, set[str]] = {}
        for label, stats in load_stats.items():
            for name in {stat.name for stat in stats}:
                labels_by_name.setdefault(name, set()).add(label)
        multiply_owned_names = {name: sorted(labels) for name, labels in labels_by_name.items() if len(labels) > 1}
        if multiply_owned_names:
            raise mismatch(f"tensor names cannot belong to multiple nonempty block load labels: {multiply_owned_names}")

        loaded_names = set(labels_by_name)
        view_names = set(self.view_tensors_names)
        if view_names != loaded_names:
            raise mismatch(
                "view_tensors_names must match names owned by block load_strategy: "
                f"missing={sorted(loaded_names - view_names)}, unexpected={sorted(view_names - loaded_names)}"
            )

        load_labels = set(load_stats)
        allocated_labels = [label for labels in self.allocation_ordered.values() for label in labels]
        allocation_counts = Counter(allocated_labels)
        missing_labels = sorted(load_labels - allocation_counts.keys())
        duplicate_labels = sorted(label for label, count in allocation_counts.items() if count != 1)
        unknown_labels = sorted(allocation_counts.keys() - load_labels)
        empty_blocks = sorted(block_id for block_id, labels in self.allocation_ordered.items() if not labels)
        if missing_labels or duplicate_labels or unknown_labels or empty_blocks:
            raise mismatch(
                "allocation_ordered must contain every nonempty load label exactly once: "
                f"missing={missing_labels}, duplicate={duplicate_labels}, unknown={unknown_labels}, "
                f"empty_blocks={empty_blocks}"
            )

        expected_label_to_block_id = {
            label: block_id for block_id, labels in self.allocation_ordered.items() for label in labels
        }
        if self.label_to_block_id != expected_label_to_block_id:
            raise mismatch("label_to_block_id disagrees with allocation_ordered")

        logical_label_sizes = {label: sum(stat.size_bytes for stat in stats) for label, stats in load_stats.items()}
        expected_label_sizes = (
            logical_label_sizes if self.loader_type == "raw_block_transfer" or self.label_to_size_map else {}
        )
        if self.label_to_size_map != expected_label_sizes:
            raise mismatch("label_to_size_map does not match logical load-strategy sizes")

        logical_block_sizes = {
            block_id: max(logical_label_sizes[label] for label in labels)
            for block_id, labels in self.allocation_ordered.items()
        }
        if self.block_sizes != logical_block_sizes:
            raise mismatch("block_sizes do not match logical allocation_ordered capacities")

        stats_labels = {layer.label for layer in self.stats}
        unknown_compute_labels = sorted(set(self.transfer_to_compute_map.values()) - stats_labels)
        if set(self.transfer_to_compute_map) != load_labels or unknown_compute_labels:
            raise mismatch("transfer_to_compute_map must cover the load labels and reference known stats labels")

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary, handling Pydantic models."""
        # Convert tensor_id_to_name_map keys to strings (JSON doesn't support int keys)
        tensor_id_to_name_map_str = {str(k): v for k, v in self.tensor_id_to_name_map.items()}

        # Convert allocation_ordered keys to strings
        allocation_ordered_str = {str(k): v for k, v in self.allocation_ordered.items()}

        # Convert label_to_size_map keys to strings
        label_to_size_map_str = {str(k): v for k, v in self.label_to_size_map.items()}

        # Convert block_sizes keys to strings
        block_sizes_str = {str(k): v for k, v in self.block_sizes.items()}

        # Convert load_strategy: dict[str, list[TensorStatistics]] to serializable format
        load_strategy_dict = {}
        for label, tensor_stats_list in self.load_strategy.items():
            load_strategy_dict[label] = [ts.model_dump() for ts in tensor_stats_list]

        # Convert release_strategy: dict[str, list[TensorStatistics]] to serializable format
        release_strategy_dict = {}
        for label, tensor_stats_list in self.release_strategy.items():
            release_strategy_dict[label] = [ts.model_dump() for ts in tensor_stats_list]

        # Convert stats: list[LayerStatistics] to serializable format
        stats_list = [ls.model_dump() for ls in self.stats]

        return {
            "version": TensorManagerState.SCHEMA_VERSION,
            "loader_type": self.loader_type,
            "tensor_id_to_name_map": tensor_id_to_name_map_str,
            "allocation_ordered": allocation_ordered_str,
            "label_to_size_map": label_to_size_map_str,
            "block_sizes": block_sizes_str,
            "load_strategy": load_strategy_dict,
            "release_strategy": release_strategy_dict,
            "label_to_block_id": self.label_to_block_id,
            "stats": stats_list,
            "transfer_to_compute_map": self.transfer_to_compute_map,
            "view_tensors_ids": self.view_tensors_ids,
            "view_tensors_names": self.view_tensors_names,
            "gpu_tensors_names": self.gpu_tensors_names,
            "shm_block_name_map": self.shm_block_name_map,
        }

    @staticmethod
    def _validate_schema_version(data: dict[str, Any]) -> None:
        """Ensure the state dict's schema version is supported. Raises ValueError if not."""
        version = data.get("version", 1)
        if version > TensorManagerState.SCHEMA_VERSION:
            raise ValueError(
                f"State file schema version {version} is newer than supported "
                f"version {TensorManagerState.SCHEMA_VERSION}. Upgrade flextensor to load this file."
            )
        if version < TensorManagerState.SCHEMA_VERSION:
            raise ValueError(
                f"State file schema version {version} is outdated (current: "
                f"{TensorManagerState.SCHEMA_VERSION}). Re-run profiling to generate a new profile."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorManagerState":
        """Create an object from a dictionary, reconstructing Pydantic models."""
        cls._validate_schema_version(data)
        # Convert tensor_id_to_name_map keys back to integers
        tensor_id_to_name_map = {int(k): v for k, v in data["tensor_id_to_name_map"].items()}

        # Convert allocation_ordered keys back to integers
        allocation_ordered = {int(k): v for k, v in data["allocation_ordered"].items()}

        label_to_size_map = {str(k): v for k, v in data["label_to_size_map"].items()}

        block_sizes = {int(k): v for k, v in data["block_sizes"].items()}

        # Reconstruct load_strategy: dict[str, list[TensorStatistics]]
        load_strategy = {}
        for label, tensor_stats_list in data["load_strategy"].items():
            load_strategy[label] = [TensorStatistics(**ts) for ts in tensor_stats_list]

        # Reconstruct release_strategy: dict[str, list[TensorStatistics]]
        release_strategy = {}
        for label, tensor_stats_list in data["release_strategy"].items():
            release_strategy[label] = [TensorStatistics(**ts) for ts in tensor_stats_list]

        # Reconstruct stats: list[LayerStatistics]
        stats = [LayerStatistics(**ls) for ls in data["stats"]]

        state = cls(
            loader_type=data["loader_type"],
            tensor_id_to_name_map=tensor_id_to_name_map,
            allocation_ordered=allocation_ordered,
            label_to_size_map=label_to_size_map,
            block_sizes=block_sizes,
            load_strategy=load_strategy,
            release_strategy=release_strategy,
            label_to_block_id=data["label_to_block_id"],
            stats=stats,
            transfer_to_compute_map=data["transfer_to_compute_map"],
            view_tensors_ids=data["view_tensors_ids"],
            view_tensors_names=data["view_tensors_names"],
            gpu_tensors_names=data["gpu_tensors_names"],
            shm_block_name_map=data["shm_block_name_map"],
        )
        state.validate_internal()
        return state

    def to_loader_input_data(self) -> LoaderInputData:
        """Create LoaderInputData from this state for loader creation.

        This method extracts the loader configuration data from the saved state
        into a LoaderInputData object that can be used by TensorManager's
        _setup_*_loader methods.

        Returns:
            LoaderInputData containing all necessary loader configuration.
        """
        return LoaderInputData(
            allocation_ordered=self.allocation_ordered,
            label_to_block_id=self.label_to_block_id,
            transfer_to_compute_map=self.transfer_to_compute_map,
            label_to_size_map=self.label_to_size_map,
            block_sizes=self.block_sizes,
            shm_block_name_map=self.shm_block_name_map,
            release_strategy=self.release_strategy,
        )


class TensorManagerStateHandler:
    """
    Handler class for TensorManager state operations.

    This class encapsulates all state-related functionality including:
    - Creating state from a TensorManager instance
    - Restoring state to a TensorManager instance
    - Saving/loading state to/from files
    """

    def __init__(self, tensor_manager: Any) -> None:
        """
        Initialize the state handler with a TensorManager instance.

        Args:
            tensor_manager: The TensorManager instance to handle state for.
        """
        self.tensor_manager = tensor_manager

    def prepare_state(  # noqa: C901
        self,
        loader_type: str,
        allocation_ordered: dict[int, list[str]],
        label_to_size_map: dict[str, int],
        block_sizes: dict[int, int],
        load_strategy: dict[str, list[TensorStatistics]],
        release_strategy: dict[str, list[TensorStatistics]],
        label_to_block_id: dict[str, int],
        stats: list[LayerStatistics],
        transfer_to_compute_map: dict[str, str],
        shm_block_name_map: dict[str, str] | None,
    ) -> TensorManagerState:
        """
        Prepare a TensorManagerState object from the current tensor manager state.

        This method updates tensor statistics with tensor names and creates
        a serializable state object.

        Args:
            loader_type: Type of tensor loader being used.
            allocation_ordered: Mapping of block IDs to ordered tensor names.
            label_to_size_map: Mapping of labels to their sizes.
            block_sizes: Mapping of block IDs to their sizes.
            load_strategy: Strategy for loading tensors per label.
            release_strategy: Strategy for releasing tensors per label.
            label_to_block_id: Mapping of labels to block IDs.
            stats: List of layer statistics.
            transfer_to_compute_map: Mapping of transfer labels to compute labels.
            shm_block_name_map: Optional mapping for shared memory block names.

        Returns:
            TensorManagerState object containing all state information.
        """
        tm = self.tensor_manager
        tensor_id_to_name_map = tm.tensor_id_to_name_map.copy()
        if isinstance(getattr(tm, "model", None), torch.nn.Module):
            for name, buffer in tm.model.named_buffers():
                tensor_id_to_name_map.setdefault(id(buffer), name)

        tensor_id_to_view_map = {}
        if tm.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
            block_controller = tm.tensor_layer_loader.allocation_controller
            tensor_id_to_view_map = block_controller.get_tensor_id_to_view_mapping()

        view_tensors_ids = []
        view_tensors_names = []
        for tensor_id, _view in tensor_id_to_view_map.items():
            view_tensors_ids.append(tensor_id)
            tensor_name = tensor_id_to_name_map[tensor_id]
            view_tensors_names.append(tensor_name)

        # add tensor names
        updated_load_strategy = {}
        for label, s_stats in load_strategy.items():
            updated_stats = []
            for stat in s_stats:
                tensor_name = tensor_id_to_name_map[stat.tensor_id]
                new_stat = TensorStatistics(
                    tensor_id=stat.tensor_id,
                    name=tensor_name,
                    size_bytes=stat.size_bytes,
                    load_time_ms=stat.load_time_ms,
                )
                updated_stats.append(new_stat)
            updated_load_strategy[label] = updated_stats

        updated_release_strategy = {}
        for label, s_stats in release_strategy.items():
            updated_stats = []
            for stat in s_stats:
                tensor_name = tensor_id_to_name_map[stat.tensor_id]
                new_stat = TensorStatistics(
                    tensor_id=stat.tensor_id,
                    name=tensor_name,
                    size_bytes=stat.size_bytes,
                    load_time_ms=stat.load_time_ms,
                )
                updated_stats.append(new_stat)
            updated_release_strategy[label] = updated_stats

        updated_layer_stats = []
        for layer_stats in stats:
            updated_stats = []
            for stat in layer_stats.tensors:
                tensor_name = tensor_id_to_name_map[stat.tensor_id]
                new_stat = TensorStatistics(
                    tensor_id=stat.tensor_id,
                    name=tensor_name,
                    size_bytes=stat.size_bytes,
                    load_time_ms=stat.load_time_ms,
                )
                updated_stats.append(new_stat)
            updated_layer_stats.append(
                LayerStatistics(label=layer_stats.label, tensors=updated_stats, duration=layer_stats.duration)
            )

        strategy_tensors_names = set()
        for _label, s_stats in updated_load_strategy.items():
            for stat in s_stats:
                strategy_tensors_names.add(stat.name)
        offload_tensors_names = strategy_tensors_names.union(view_tensors_names)
        gpu_tensors_names = list(set(tensor_id_to_name_map.values()).difference(offload_tensors_names))

        tensor_manager_state = TensorManagerState(
            loader_type=loader_type,
            tensor_id_to_name_map=tensor_id_to_name_map,
            allocation_ordered=allocation_ordered,
            label_to_size_map=label_to_size_map,
            block_sizes=block_sizes,
            load_strategy=updated_load_strategy,
            release_strategy=updated_release_strategy,
            label_to_block_id=label_to_block_id,
            stats=updated_layer_stats,
            transfer_to_compute_map=transfer_to_compute_map,
            view_tensors_ids=view_tensors_ids,
            view_tensors_names=view_tensors_names,
            gpu_tensors_names=gpu_tensors_names,
            shm_block_name_map=shm_block_name_map,
        )
        return tensor_manager_state

    @staticmethod
    def _extract_state_tensor_names(state: TensorManagerState) -> set[str]:
        """
        Extract all tensor names referenced in the state.

        Args:
            state: The TensorManagerState to extract names from.

        Returns:
            Set of all tensor names referenced in load_strategy, release_strategy, and stats.
        """
        names: set[str] = set()
        for statistics in state.load_strategy.values():
            names.update(stat.name for stat in statistics)
        for statistics in state.release_strategy.values():
            names.update(stat.name for stat in statistics)
        for layer in state.stats:
            names.update(stat.name for stat in layer.tensors)
        names.update(state.view_tensors_names)
        names.update(state.gpu_tensors_names)
        return names

    @staticmethod
    def _get_model_tensor_names(model: torch.nn.Module | dict) -> set[str]:
        """
        Get all tensor names from a model.

        Args:
            model: The model (nn.Module or dict) to get tensor names from.

        Returns:
            Set of all tensor names in the model.
        """
        if isinstance(model, torch.nn.Module):
            return {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}
        elif isinstance(model, dict):
            return set(model.keys())
        else:
            raise TypeError(f"model must be nn.Module or dict, got {type(model)}")

    def validate_state_compatibility(
        self,
        model: torch.nn.Module | dict,
        state: TensorManagerState,
    ) -> StateValidationResult:
        """
        Validate compatibility between a saved state and the current model.

        This method checks whether all tensor names referenced in the saved state
        exist in the current model, and identifies any tensors in the model that
        are not covered by the saved state.

        Args:
            model: The model (nn.Module or dict) to validate against.
            state: The TensorManagerState to validate.

        Returns:
            StateValidationResult containing missing_keys (tensors in state but not model)
            and unexpected_keys (tensors in model but not state).

        Example:
            >>> handler = TensorManagerStateHandler(tm)
            >>> result = handler.validate_state_compatibility(model, state)
            >>> if result.missing_keys:
            ...     print(f"Missing tensors: {result.missing_keys[:5]}")
        """
        state_tensor_names = self._extract_state_tensor_names(state)
        model_tensor_names = self._get_model_tensor_names(model)

        missing_keys = sorted(state_tensor_names - model_tensor_names)
        unexpected_keys = sorted(model_tensor_names - state_tensor_names)

        return StateValidationResult(
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )

    @staticmethod
    def _check_validation_result(validation_result: StateValidationResult, *, strict: bool) -> None:
        """
        Check validation result and raise or log appropriately.

        Args:
            validation_result: The result from validate_state_compatibility.
            strict: If True, raises ValueError on missing keys. If False, logs warning.

        Raises:
            ValueError: If strict=True and there are missing keys.
        """
        if validation_result.missing_keys:
            sample_missing = validation_result.missing_keys[:5]
            suffix = "..." if len(validation_result.missing_keys) > 5 else ""
            msg = (
                f"Cannot restore state: {len(validation_result.missing_keys)} tensor(s) "
                f"from saved profile not found in current model. "
                f"Examples: {sample_missing}{suffix}"
            )
            if strict:
                raise ValueError(f"{msg} Use strict=False to skip missing tensors.")
            else:
                logger.warning("%s Skipping these tensors.", msg)

        if validation_result.unexpected_keys:
            sample_unexpected = validation_result.unexpected_keys[:5]
            suffix = "..." if len(validation_result.unexpected_keys) > 5 else ""
            logger.info(
                "%d tensor(s) in model not covered by saved profile. Examples: %s%s",
                len(validation_result.unexpected_keys),
                sample_unexpected,
                suffix,
            )

    def restore_state(  # noqa: C901
        self, model: torch.nn.Module | dict, state: TensorManagerState, *, strict: bool = True
    ) -> StateValidationResult:
        """
        Restore a TensorManager state from a TensorManagerState object.

        This method updates the tensor IDs in the state to match the current
        model's tensor IDs, then applies the state to the tensor manager.

        Args:
            model: The model (nn.Module or dict) to restore state for.
            state: The TensorManagerState to restore.
            strict: If True (default), raises ValueError if any tensor names in the
                saved state are not found in the current model. If False, missing
                tensors are skipped with a warning.

        Raises:
            ValueError: If strict=True and the saved state contains tensor names
                not present in the current model.
            ValueError: If the state's loader_type doesn't match the tensor manager's
                loader_type (incompatible mappings between strategy vs block-transfer).

        Returns:
            StateValidationResult: Contains missing_keys and unexpected_keys lists
            for inspection. In strict mode, missing_keys will always be empty
            (otherwise an exception would have been raised).
        """
        tm = self.tensor_manager

        # Validate loader_type compatibility early to fail fast with a clear error
        state_loader_type = state.loader_type
        if state_loader_type != tm.loader_type:
            msg = (
                f"Saved profile uses loader_type='{state_loader_type}' but TensorManager is configured "
                f"with loader_type='{tm.loader_type}'. These loader types use incompatible tensor "
                f"mappings and cannot be mixed. Either re-profile the model with loader_type='{tm.loader_type}' "
                f"or create TensorManager with loader_type='{state_loader_type}'."
            )
            raise ValueError(msg)

        state.validate_internal()
        load_labels = {label for label, stats in state.load_strategy.items() if stats}
        if (
            state.loader_type != "strategy"
            and tm.use_shm
            and state.shm_block_name_map is not None
            and set(state.shm_block_name_map) != load_labels
        ):
            raise ValueError(
                "Saved state mismatch: shm_block_name_map must cover exactly the nonempty load labels. "
                "Re-profile the model."
            )

        validation_result = self.validate_state_compatibility(model, state)
        self._check_validation_result(validation_result, strict=strict)
        if isinstance(model, torch.nn.Module):
            live_tensors: dict[str, torch.Tensor] = dict(model.named_parameters(remove_duplicate=False))
            live_tensors.update(model.named_buffers(remove_duplicate=False))
            buffer_names = {name for name, _ in model.named_buffers(remove_duplicate=False)}
            current = capture_model_state(model)
            tensors_by_storage: dict[object, dict[object, list[str]]] = {storage.id: {} for storage in current.storages}
            for captured_tensor in current.tensors:
                tensors_by_storage[captured_tensor.storage_id][captured_tensor.id] = list(captured_tensor.names)
        else:
            live_tensors = model
            buffer_names = set()
            tensors_by_storage = {}
            for name, live_tensor in live_tensors.items():
                if isinstance(live_tensor, torch.Tensor):
                    tensors_by_storage.setdefault(live_tensor.untyped_storage()._cdata, {}).setdefault(  # noqa: SLF001
                        id(live_tensor), []
                    ).append(name)
        managed_names = {stat.name for stats in state.load_strategy.values() for stat in stats}
        managed_names.update(state.view_tensors_names)
        managed_buffers = sorted(managed_names & buffer_names)
        if managed_buffers:
            raise ValueError(f"Registered buffer(s) must remain final-GPU, not managed: {managed_buffers}")
        contradictory_storages = {
            storage_id: sorted(name for names in tensors.values() for name in names)
            for storage_id, tensors in tensors_by_storage.items()
            if len({"cpu" if managed_names.intersection(names) else "gpu" for names in tensors.values()}) > 1
        }
        if contradictory_storages:
            raise ValueError(f"Contradictory alias destinations for shared storage: {contradictory_storages}")
        if state.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
            shared_storage_views = {
                storage_id: sorted(name for names in tensors.values() for name in names)
                for storage_id, tensors in tensors_by_storage.items()
                if sum(bool(managed_names.intersection(names)) for names in tensors.values()) > 1
            }
            if shared_storage_views:
                raise ValueError(
                    f"Block loaders cannot preserve distinct tensor views sharing storage: {shared_storage_views}"
                )
        statistics = [stat for stats in state.load_strategy.values() for stat in stats]
        statistics.extend(stat for stats in state.release_strategy.values() for stat in stats)
        statistics.extend(stat for layer in state.stats for stat in layer.tensors)
        for stat in statistics:
            matching_tensor = live_tensors.get(stat.name)
            if matching_tensor is not None:
                logical_bytes = matching_tensor.numel() * matching_tensor.element_size()
                if stat.size_bytes != logical_bytes:
                    raise ValueError(
                        f"TensorStatistics.size_bytes drift for {stat.name}: "
                        f"saved={stat.size_bytes}, current_logical={logical_bytes}"
                    )

        # Deep-copy the state so we don't mutate the caller's object (it may be reused).
        state = copy.deepcopy(state)

        # Put model in the same state as the discovery path: disable grad, params/buffers on GPU,
        # pinned memory for strategy loader, and traced_tensors populated for inference traps.
        if not tm.use_trace_tensor:  # TODO: add support for trace tensor
            preprocess_model(
                model,
                tm,
                tm.device_gpu,
                pin_memory=tm.should_pin_in_preprocess(),
                host_pinner=tm.host_pinner,
                move_top_level_buffers_to_gpu=tm.move_top_level_buffers_to_gpu,
            )

        tensor_name_to_id_map = {}
        if isinstance(model, torch.nn.Module):
            for name, tensor in model.named_parameters():
                tensor_name_to_id_map[name] = id(tensor)
        elif isinstance(model, dict):
            for name, tensor in model.items():
                tensor_name_to_id_map[name] = id(tensor)

        # Set of valid tensor names (present in current model)
        valid_tensor_names = set(tensor_name_to_id_map.keys())

        # add correct tensor ids
        updated_load_strategy = {}
        for label, s_stats in state.load_strategy.items():
            updated_stats = []
            for stat in s_stats:
                tensor_name = stat.name
                if tensor_name not in valid_tensor_names:
                    continue  # Skip missing tensors (only reachable when strict=False)
                tensor_id = tensor_name_to_id_map[tensor_name]
                new_stat = TensorStatistics(
                    tensor_id=tensor_id, name=tensor_name, size_bytes=stat.size_bytes, load_time_ms=stat.load_time_ms
                )
                updated_stats.append(new_stat)
            updated_load_strategy[label] = updated_stats
        state.load_strategy = updated_load_strategy

        updated_release_strategy = {}
        for label, s_stats in state.release_strategy.items():
            updated_stats = []
            for stat in s_stats:
                tensor_name = stat.name
                if tensor_name not in valid_tensor_names:
                    continue  # Skip missing tensors (only reachable when strict=False)
                tensor_id = tensor_name_to_id_map[tensor_name]
                new_stat = TensorStatistics(
                    tensor_id=tensor_id, name=tensor_name, size_bytes=stat.size_bytes, load_time_ms=stat.load_time_ms
                )
                updated_stats.append(new_stat)

            updated_release_strategy[label] = updated_stats
        state.release_strategy = updated_release_strategy

        updated_layer_stats = []
        for layer_stats in state.stats:
            updated_stats = []
            for stat in layer_stats.tensors:
                tensor_name = stat.name
                if tensor_name not in valid_tensor_names:
                    continue  # Skip missing tensors (only reachable when strict=False)
                tensor_id = tensor_name_to_id_map[tensor_name]
                new_stat = TensorStatistics(
                    tensor_id=tensor_id, name=tensor_name, size_bytes=stat.size_bytes, load_time_ms=stat.load_time_ms
                )
                updated_stats.append(new_stat)
            updated_layer_stats.append(
                LayerStatistics(label=layer_stats.label, tensors=updated_stats, duration=layer_stats.duration)
            )
        state.stats = updated_layer_stats

        offload_keys = set()
        for _label, s_stats in state.load_strategy.items():
            for stat in s_stats:
                offload_keys.add(stat.name)

        # Filter view_tensors_names to only include valid tensors and recompute view_tensors_ids
        valid_view_tensors = [name for name in state.view_tensors_names if name in valid_tensor_names]
        state.view_tensors_names = valid_view_tensors
        state.view_tensors_ids = [tensor_name_to_id_map[name] for name in valid_view_tensors]

        for name in state.view_tensors_names:
            offload_keys.add(name)

        # Restore tensors_map to only the tensors in the profile (load_strategy + view_tensors).
        tensors_map = {}
        if isinstance(model, torch.nn.Module):
            for name, tensor in model.named_parameters():
                if name in offload_keys:
                    tensors_map[id(tensor)] = tensor
        elif isinstance(model, dict):
            for name, tensor in model.items():
                if name in offload_keys:
                    tensors_map[id(tensor)] = tensor
        tm.tensors_map = tensors_map
        # Align traced_tensors with the tensors we actually manage (traps use this for load decisions).
        tm.traced_tensors = set(tm.tensors_map.keys())

        tm.tensor_manager_state = state
        # Mark the state as externally restored so ``initialize_warmup`` takes
        # the short-circuit. The flag — not ``tensor_manager_state`` itself —
        # is the signal, because a normal cycle reaching INFERENCE also stores
        # state and must not be mistaken for a restored profile.
        tm._state_restored_from_profile = True  # noqa: SLF001
        tm.set_model(model)

        return validation_result

    @staticmethod
    def save_to_file(file_path: str | pathlib.Path, state: TensorManagerState) -> None:
        """
        Save a TensorManagerState to a JSON file.

        Uses atomic write to prevent file corruption during crashes.

        Args:
            file_path: Path to the file to save the state to.
            state: The TensorManagerState to save.
        """
        state_dict = state.to_dict()
        atomic_write_json(file_path, state_dict)

    @staticmethod
    def load_from_file(file_path: str | pathlib.Path) -> TensorManagerState:
        """
        Load a TensorManagerState from a JSON file.

        Args:
            file_path: Path to the file to load the state from.

        Returns:
            The loaded TensorManagerState object.
        """
        with pathlib.Path(file_path).open() as f:
            loaded_state = json.load(f)

        return TensorManagerState.from_dict(loaded_state)

    @staticmethod
    def save_state_to_bytes(state: TensorManagerState) -> bytes:
        """Serialize a TensorManagerState to bytes with length prefix.

        Format: 4-byte big-endian uint32 length + JSON payload.

        Args:
            state: The TensorManagerState to serialize.

        Returns:
            Bytes containing length-prefixed JSON.
        """
        state_dict = state.to_dict()
        payload = json.dumps(state_dict, separators=(",", ":")).encode("utf-8")
        length_prefix = struct.pack("!I", len(payload))
        return length_prefix + payload

    @staticmethod
    def load_state_from_bytes(data: bytes | bytearray | memoryview) -> TensorManagerState:
        """Deserialize a TensorManagerState from length-prefixed bytes.

        Args:
            data: Bytes containing length-prefixed JSON (as produced by save_state_to_bytes).

        Returns:
            The deserialized TensorManagerState.

        Raises:
            ValueError: If the buffer is too short or length prefix is invalid.
        """
        if len(data) < 4:
            raise ValueError(f"Buffer too short: expected at least 4 bytes for length prefix, got {len(data)}")
        length = struct.unpack("!I", bytes(data[:4]))[0]
        if len(data) < 4 + length:
            raise ValueError(
                f"Buffer too short: length prefix indicates {length} bytes, "
                f"but only {len(data) - 4} bytes available after prefix"
            )
        payload = bytes(data[4 : 4 + length])
        state_dict = json.loads(payload)
        return TensorManagerState.from_dict(state_dict)
