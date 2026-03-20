# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import types
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from flextensor.benchmark_tensor_mode import BenchmarkReplace, TensorBenchmarkMode
from flextensor.collectors import (
    IterativeLayerStatisticsCollector,
    IterativeLayerStatisticsFilter,
    LayerStatistics,
    TensorStatistics,
)
from flextensor.helpers import TrapNestingGuard
from flextensor.instrumentation import instrumentable
from flextensor.layer_statistics_analyzer import LayerStatisticsAnalyzer
from flextensor.loaders import (
    AllocationBlockController,
    PreallocatedBatchTransferTensorLoader,
    PreallocatedBatchTransferTensorLoaderReordered,
    RawBlockController,
    TensorLayerLoader,
    TensorStrategyLoader,
)
from flextensor.memory_transfer_benchmark import (
    benchmark_memory_transfers,
    extract_memory_transfers_from_layer_stats,
    format_memory_transfer_table,
)
from flextensor.state_handler import (
    LoaderInputData,
    TensorManagerStateHandler,
)
from flextensor.strategy import (
    BlockStrategyData,
    GreedyStrategy,
    KnapsackStrategy,
    NthLayerStrategy,
)
from flextensor.strategy.utils import compute_label_to_size_map, log_block_table, strategy_has_transfer_gaps
from flextensor.strategy_operations import (
    create_allocation_ordered,
    rearrange_transfers,
    remap_strategy,
    remove_layers_compound,
)
from flextensor.tensor import TraceTensor
from flextensor.tensor_discovery import (
    ModuleTracker,
    discover_untraced_tensors_for_layers,
    has_offload_modules,
)
from flextensor.tensor_processors import (
    BenchmarkTensorProcessor,
    MoveUnmappedTensorsToGPUProcessor,
    TensorReplacementProcessor,
    create_model_with_shared_tensors,
    preprocess_model,
)
from flextensor.trap_tensor_mode import Trap, TrapDirect, TrapInfer, TrapInferDirect, WarmupTrap
from flextensor.types import GPUMemoryUsage

logger = logging.getLogger(__name__)

_GiB = 1 << 30
_MIN_GPU_BUDGET_BYTES = 256 * 1024**2  # 256 MiB — floor for strategy budget

# =============================================================================
# Helper Functions
# =============================================================================


def _compute_duration(layer_stats: list[LayerStatistics]) -> float:
    """Calculate total duration across all layers."""
    duration_ms = 0.0
    for layer in layer_stats:
        duration_ms += layer.duration
    return duration_ms


def _compute_resource_release_strategy(
    strategy: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
) -> dict[str, list[TensorStatistics]]:
    """Compute which tensors to release at each layer.

    Args:
        strategy: Strategy mapping layer labels to tensors to offload.
        layer_stats: Layer statistics.

    Returns:
        Release strategy mapping layer labels to tensors to release.
    """
    release_strategy: dict[str, list[TensorStatistics]] = {}
    offload_tensor_ids: set[int] = set()
    for _layer, tensor_info_list in strategy.items():
        for tensor_info in tensor_info_list:
            offload_tensor_ids.add(tensor_info.tensor_id)

    for stat in layer_stats:
        tensor_info_list = []
        for tensor_info in stat.tensors:
            if tensor_info.tensor_id in offload_tensor_ids:
                tensor_info_list.append(tensor_info)
        if len(tensor_info_list) > 0:
            release_strategy[stat.label] = tensor_info_list

    return release_strategy


class ModelDict:
    def __init__(self, tensor_manager=None, model=None):
        self.key_to_id_map = {}
        for key, tensor in model.items():
            tensor_id = id(tensor)
            self.key_to_id_map[key] = tensor_id

        self.tensor_manager = tensor_manager
        self.model = model

    def __getitem__(self, key):
        tensor_id = self.key_to_id_map[key]

        tensor = self.model[key]

        if self.tensor_manager.is_traced_by_id(tensor_id):
            new_tensor = self.tensor_manager.tensor_layer_loader.get(tensor_id)
            if new_tensor is not None:
                tensor = new_tensor
        return tensor

    def items(self):
        return self.model.items()


def _rebind_bound_methods(module_copy, original_module):
    """Rebind any bound methods in __dict__ that are still bound to the original module."""
    for name, value in list(module_copy.__dict__.items()):
        if isinstance(value, types.MethodType) and value.__self__ is original_module:
            rebound_method = types.MethodType(value.__func__, module_copy)
            module_copy.__dict__[name] = rebound_method


def _apply_missing_field(tensor_ref, new_tensor, missing_field, tensor_manager):
    """Apply a missing field from tensor_ref to new_tensor, handling device placement."""
    field = getattr(tensor_ref, missing_field)
    if isinstance(field, torch.Tensor):
        field_id = id(field)
        new_tensor_field = tensor_manager.tensor_layer_loader.get(field_id)
        if new_tensor_field is None:
            # Inner field not loaded yet, move it to the same device as parent tensor
            new_tensor_field = field.to(device=new_tensor.device) if field.device != new_tensor.device else field
        setattr(new_tensor, missing_field, new_tensor_field)
    else:
        setattr(new_tensor, missing_field, field)


def _make_tensor_getter(tensor_ref, tensor_manager, missing_fields):
    """Create a property getter for a tensor that handles offloading."""
    tensor_id = id(tensor_ref)

    def getter(_self):
        tensor = tensor_ref
        if tensor_manager.is_traced_by_id(tensor_id):
            new_tensor = tensor_manager.tensor_layer_loader.get(tensor_id)
            if new_tensor is not None:
                tensor = new_tensor
                for missing_field in missing_fields:
                    _apply_missing_field(tensor_ref, new_tensor, missing_field, tensor_manager)
        return tensor

    return getter


def extend_nn_module(module, tensor_manager):
    """
    Extend an nn.Module with property getters and setters for all parameters and tensors.
    Similar to extend_with_temperature_properties but for PyTorch modules.
    """
    # Create a copy of the module to avoid modifying the original
    module_copy = copy.copy(module)
    _rebind_bound_methods(module_copy, module)

    # Create a new class dynamically for this module
    module_class = type(module_copy.__class__.__name__, (module_copy.__class__,), {})

    # Get all parameters and buffers
    all_params = dict(module_copy.named_parameters(recurse=False))
    all_buffers = dict(module_copy.named_buffers(recurse=False))
    all_tensors = {**all_params, **all_buffers}

    # Create property getters and setters for each tensor
    for name, tensor in all_tensors.items():
        # Compute missing fields (custom attributes not in base torch.Tensor)
        ref_tensor_fields = set(dir(tensor))
        new_tensor_fields = set(dir(torch.Tensor))
        missing_fields = [f for f in (ref_tensor_fields - new_tensor_fields) if not f.startswith("_")]

        getter_func = _make_tensor_getter(tensor, tensor_manager, missing_fields)
        setattr(module_class, name, property(getter_func))

    # Change the module's class
    module_copy.__class__ = module_class

    return module_copy


def prepare_model(model, tensor_manager):
    """
    Recursively prepare a model by extending all its modules with tensor management capabilities.
    This function traverses the model hierarchy and applies extend_nn_module to each module.
    """

    def _prepare_module(module):
        # Extend the current module
        extended_module = extend_nn_module(module, tensor_manager)
        # Recursively process all child modules
        for name, child_module in extended_module.named_children():
            if isinstance(child_module, torch.nn.Module):
                # Replace the child module with its extended version
                setattr(extended_module, name, _prepare_module(child_module))

        return extended_module

    return _prepare_module(model)


def compute_layer_statistics(iterative_layer_statistics, tensor_statistics_map) -> list[LayerStatistics]:
    layers_stats = []
    for iterative_layer_stat in iterative_layer_statistics:
        tensors = []
        for tensor_id in iterative_layer_stat.tensor_ids:
            if tensor_id not in tensor_statistics_map:
                continue
            size_bytes = tensor_statistics_map[tensor_id].size_bytes
            load_time_ms = tensor_statistics_map[tensor_id].load_time_ms
            tensors.append(
                TensorStatistics(tensor_id=tensor_id, name="", size_bytes=size_bytes, load_time_ms=load_time_ms),
            )

        layers_statistics = LayerStatistics(
            label=iterative_layer_stat.label,
            tensors=tensors,
            duration=iterative_layer_stat.duration,
        )
        layers_stats.append(layers_statistics)
    return layers_stats


@instrumentable
class TensorManager:
    def __init__(
        self,
        device_gpu,
        pinned_memory,
        tensor_manager_load_strategy,
        release_tensors=True,
        benchmark_cls: type[TensorBenchmarkMode] = BenchmarkReplace,
        direct_mode=True,
        use_trace_tensor=False,
        loader_type="allocation_block_transfer",
        remove_layers_operations=None,
        rearrange_transfers=False,
        min_compute_transfer_gap=1,
        blocks=4,
        move_top_level_buffers_to_gpu=True,
        use_shm=False,
        enable_untraced_tensor_discovery=True,
        enable_module_tracker=True,
        enable_diagnostics: bool = False,
        max_gpu_mem_fraction: float | None = None,
    ):
        """
        Initialize TensorManager with configurable tensor loading strategy.

        Args:
            device_gpu: GPU device to use
            pinned_memory: Whether to use pinned memory
            tensor_manager_load_strategy: Strategy for loading tensors
            release_tensors: Whether to release tensors after use
            benchmark_cls: Benchmark class to use
            direct_mode: Whether to enable direct mode
            use_trace_tensor: Whether to use trace tensor functionality
            loader_type: Type of tensor loader to use. Options:
                - "strategy": Uses TensorStrategyLoader
                - "raw_block_transfer": Uses PreallocatedBatchTransferTensorLoader with RawBlockController
                - "allocation_block_transfer": Uses PreallocatedBatchTransferTensorLoader with AllocationBlockController
            remove_layers_operations: List of operations to remove layers from the strategy map
            rearrange_transfers: Whether to enable transfer rearrangement optimization (default: True)
            min_compute_transfer_gap: Minimum gap between compute and transfer layers for optimization (default: 1)
            blocks: Number of blocks to use for the block transfer loaders (default: 4)
            move_top_level_buffers_to_gpu: Whether to move top-level model buffers to GPU
                during warmup initialization (default: True)
            enable_untraced_tensor_discovery: Whether to discover untraced tensors (e.g., FP8 weights
                passed to Triton kernels) using discovery strategies (default: True)
            enable_module_tracker: Whether to enable ModuleTracker for manual traps when forward
                patching is not used (default: True)
            enable_diagnostics: Whether to log diagnostic information (layer duration statistics,
                block assignment table) after strategy computation (default: False)
            max_gpu_mem_fraction: Fraction of total GPU memory to use as budget, in (0.0, 1.0].
                Resolved to bytes at compute time via :meth:`_resolve_gpu_budget`. If None,
                no memory constraint is applied (latency mode). (default: None)
        """
        if remove_layers_operations is None:
            remove_layers_operations = []
        self.device_gpu = device_gpu
        self.tensor_statistics_map = {}
        self.tensors_map = {}
        self.trap_class = None
        self.tensor_layer_loader = None
        self.layer_statistics_collector = None
        self.tensor_manager_load_strategy = tensor_manager_load_strategy
        self.release_tensors = release_tensors
        self.pinned_memory = pinned_memory
        self._benchmark_cls = benchmark_cls
        self.direct_enabled = direct_mode
        self.traced_tensors = set()
        self.transfer_stream_priority = 1
        self.blocks = blocks
        self.move_top_level_buffers_to_gpu = move_top_level_buffers_to_gpu
        self.enable_untraced_tensor_discovery = enable_untraced_tensor_discovery
        self.enable_module_tracker = enable_module_tracker
        self.enable_diagnostics = enable_diagnostics
        self._max_gpu_mem_fraction = max_gpu_mem_fraction

        self.traps_duration_ms = 0
        self.model_ids = set()
        self.block_data: BlockStrategyData | None = None

        self.loader_type = loader_type
        self.remove_layers_operations = remove_layers_operations
        self.rearrange_transfers = rearrange_transfers
        self.min_compute_transfer_gap = min_compute_transfer_gap
        self.use_trace_tensor = use_trace_tensor
        if use_trace_tensor:
            self.is_traced = self.is_traced_trace_tensor

        if self.loader_type in ["allocation_block_transfer", "raw_block_transfer"] and not direct_mode:
            msg = "Direct mode is required for allocation_block_transfer and raw_block_transfer"
            raise Exception(msg)
        self.model = None
        self.tensor_id_to_name_map = {}
        self.tensor_manager_state = None
        self.use_shm = use_shm
        self.shm_namespace: str | None = None
        self._unmapped_tensors_bytes = 0
        self.module_tracker: ModuleTracker | None = None
        self.memory_transfer_stats: dict[int, float] | None = None
        self.trap_start_event = torch.cuda.Event(enable_timing=True)
        self.trap_end_event = torch.cuda.Event(enable_timing=True)
        self.trap_nesting_guard = TrapNestingGuard()

    def build_parameters_mapping(self, model):
        self.tensor_id_to_name_map = {}
        if isinstance(model, torch.nn.Module):
            for name, tensor in model.named_parameters():
                self.tensor_id_to_name_map[id(tensor)] = name
        elif isinstance(model, dict):
            for name, tensor in model.items():
                self.tensor_id_to_name_map[id(tensor)] = name

    def set_model(self, model):
        self.model = model
        self.build_parameters_mapping(self.model)

    def initialize_warmup(self):
        if self.tensor_manager_state is not None:
            self.prepare_infer_load_mode()
            final_model = self.prepare_final_model(self.model)  # patch model
            self.model = final_model
            return self.model

        if not self.use_trace_tensor:  # TODO: add support for trace tensor
            pin_memory = self.pinned_memory
            if self.loader_type in ["allocation_block_transfer", "raw_block_transfer"]:
                pin_memory = False
            preprocess_model(
                self.model,
                self,
                self.device_gpu,
                pin_memory=pin_memory,
                move_top_level_buffers_to_gpu=self.move_top_level_buffers_to_gpu,
            )
        self.prepare_model_ids(self.model)  # TODO: Move to preprocess_model after remove benchmark context

        self.prepare_warmup_mode()
        return self.model

    def initialize_profile(self):
        if self.tensor_manager_state is not None:
            return self.model
        profile_model = self.model
        if self.direct_enabled:
            profile_model = self.prepare_profile_direct_mode_model(self.model)
            self.prepare_profile_direct_mode()
        else:
            self.prepare_profile_mode()
        return profile_model

    def initialize_inference(self):
        if self.tensor_manager_state is not None:
            final_model = self.model
            self.model = None
        else:
            self.prepare_infer_mode()
            final_model = self.prepare_final_model(self.model)  # patch model
            self.model = None
        return final_model

    # TODO: Fix me, remove!
    def prepare_model_ids(self, model):
        tensors_ids = set()
        iterator = None
        if isinstance(model, torch.nn.Module):
            iterator = model.named_parameters()
        elif isinstance(model, dict):
            iterator = model.items()

        for _name, tensor in iterator:
            tensors_ids.add(id(tensor))
            self.set_model_ids(tensors_ids)

    def set_model_ids(self, tensors_ids):
        self.model_ids = tensors_ids

    def prepare_warmup_mode(self):
        self.layer_statistics_collector = IterativeLayerStatisticsCollector()
        self.trap_class = WarmupTrap
        self.tensor_layer_loader = None
        # Initialize module tracker only for manual traps (no forward patching).
        # With forward patching, we can discover tensors directly via named_parameters().
        if (
            self.enable_module_tracker
            and self.model is not None
            and isinstance(self.model, torch.nn.Module)
            and not has_offload_modules(self.model)
        ):
            self.module_tracker = ModuleTracker()
            self.module_tracker.register(self.model)

    def prepare_profile_direct_mode_model(self, model):
        profile_model = model
        if self.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
            profile_model = copy.copy(model) if isinstance(model, dict) else create_model_with_shared_tensors(model)
            # TODO: Consider using prepare_view_model() instead of prepare_model() for block transfer loaders
            # to get better statistics during profiling
            profile_model = self.prepare_model(profile_model)
        elif self.direct_enabled:
            profile_model = self.prepare_model(model)
        return profile_model

    def prepare_profile_direct_mode(self):
        self.layer_stats = self.layer_statistics_collector.get_layer_stats()
        self.layer_stats = IterativeLayerStatisticsFilter().filter_by_tensor_ids(
            self.layer_stats,
            set(self.tensors_map.keys()),
        )
        # Add untraced tensors (e.g., fp8 weights passed to Triton) to layer stats for profiling
        if self.enable_untraced_tensor_discovery:
            self.layer_stats = discover_untraced_tensors_for_layers(
                self.layer_stats,
                self.tensors_map,
                self.model,
                self.tensor_id_to_name_map,
                module_tracker=self.module_tracker,
            )

        # Remove module tracker hooks after warmup - no longer needed
        if self.module_tracker is not None:
            self.module_tracker.unregister()
            self.module_tracker = None

        # TODO: When profile direct mode switches to using view models (prepare_view_model),
        # this TensorLayerLoader may need to be replaced with a view-compatible loader
        self.tensor_layer_loader = TensorLayerLoader(self.layer_stats, self.tensors_map, self.device_gpu)
        self.tensor_layer_loader.set_model_ids(self.model_ids)  # TODO: Fix me, remove

        self.trap_class = TrapDirect
        self.layer_statistics_collector.clear_duration_measurements()

    def prepare_profile_mode(self):
        self.layer_stats = self.layer_statistics_collector.get_layer_stats()
        self.layer_stats = IterativeLayerStatisticsFilter().filter_by_tensor_ids(
            self.layer_stats,
            set(self.tensors_map.keys()),
        )
        # Add untraced tensors (e.g., fp8 weights passed to Triton) to layer stats for profiling
        if self.enable_untraced_tensor_discovery:
            self.layer_stats = discover_untraced_tensors_for_layers(
                self.layer_stats,
                self.tensors_map,
                self.model,
                self.tensor_id_to_name_map,
                module_tracker=self.module_tracker,
            )

        # Remove module tracker hooks after warmup - no longer needed
        if self.module_tracker is not None:
            self.module_tracker.unregister()
            self.module_tracker = None

        self.tensor_layer_loader = TensorLayerLoader(self.layer_stats, self.tensors_map, self.device_gpu)
        self.tensor_layer_loader.set_model_ids(self.model_ids)  # TODO: Fix me, remove
        self.trap_class = Trap
        self.layer_statistics_collector.clear_duration_measurements()

    def _select_infer_trap_class(self) -> type:
        """Select appropriate trap class for inference mode based on direct_enabled setting."""
        return TrapInferDirect if self.direct_enabled else TrapInfer

    def _create_loader(self, data: LoaderInputData, *, prepare_state: bool = True) -> None:
        """Create tensor layer loader from input data.

        This is the unified loader creation method that works with both fresh computation
        and restored state data.

        Args:
            data: LoaderInputData containing all necessary loader configuration.
            prepare_state: Whether to prepare and store the tensor manager state after
                creating the loader. Set to False when loading from saved state
                (state is already prepared).
        """
        self.trap_class = self._select_infer_trap_class()

        if self.loader_type == "strategy":
            self._setup_strategy_loader(data, prepare_state=prepare_state)
        elif self.loader_type == "raw_block_transfer":
            self._setup_raw_block_loader(data, prepare_state=prepare_state)
        elif self.loader_type == "allocation_block_transfer":
            self._setup_allocation_block_loader(data, prepare_state=prepare_state)
        else:
            msg = f"Unknown loader type: {self.loader_type}"
            raise ValueError(msg)

    def prepare_infer_load_mode(self):
        """Prepare inference mode from saved state.

        This method is called when tensor_manager_state has been loaded from a saved profile.
        It extracts LoaderInputData from the state and creates the appropriate loader.
        """
        self.load_strategy = self.tensor_manager_state.load_strategy
        self.stats = self.tensor_manager_state.stats

        loader_data = self.tensor_manager_state.to_loader_input_data()
        self._create_loader(loader_data, prepare_state=False)

    def _setup_strategy_loader(self, data: LoaderInputData, *, prepare_state: bool = True) -> None:
        """Setup TensorStrategyLoader for 'strategy' loader type.

        Args:
            data: LoaderInputData containing loader configuration.
            prepare_state: Whether to prepare and store state after setup
                (False when loading from saved state).
        """
        # Use release_strategy from data if available (from state), otherwise compute it
        release_strategy = data.release_strategy
        if not release_strategy:
            release_strategy = _compute_resource_release_strategy(self.load_strategy, self.stats)

        self.tensor_layer_loader = TensorStrategyLoader(
            self.stats,
            self.load_strategy,
            release_strategy,
            self.tensors_map,
            self.device_gpu,
            self.release_tensors,
            self.transfer_stream_priority,
        )

        if prepare_state:
            self.tensor_manager_state = self.prepare_tensor_manager_state(
                self.loader_type,
                data.allocation_ordered,
                data.label_to_size_map,
                data.block_sizes,
                self.load_strategy,
                release_strategy,
                data.label_to_block_id,
                self.stats,
                data.transfer_to_compute_map,
                shm_block_name_map=None,
            )

    def _setup_raw_block_loader(self, data: LoaderInputData, *, prepare_state: bool = True) -> None:
        """Setup RawBlockController loader for 'raw_block_transfer' loader type.

        Args:
            data: LoaderInputData containing loader configuration.
            prepare_state: Whether to prepare and store state after setup
                (False when loading from saved state).
        """
        label_to_size_map = data.label_to_size_map
        allocation_ordered = data.allocation_ordered
        block_sizes = data.block_sizes
        label_to_block_id = data.label_to_block_id
        transfer_to_compute_map = data.transfer_to_compute_map

        # Apply transfer rearrangement only during fresh computation (prepare_state=True)
        # When loading from state, rearrangements were already applied
        tensor_loader_class = PreallocatedBatchTransferTensorLoader
        if self.rearrange_transfers:
            tensor_loader_class = PreallocatedBatchTransferTensorLoaderReordered
            if prepare_state:
                transfer_to_compute_map, label_to_block_id, remapped_layers = rearrange_transfers(
                    transfer_to_compute_map,
                    label_to_block_id,
                    self.stats,
                    self.min_compute_transfer_gap,
                )
                allocation_ordered = create_allocation_ordered(label_to_block_id, self.stats)
                self.load_strategy = remap_strategy(self.load_strategy, remapped_layers)
                label_to_size_map = compute_label_to_size_map(self.stats, self.load_strategy)

        block_controller = RawBlockController(
            label_to_size_map,
            block_sizes,
            self.device_gpu,
            self.tensors_map,
            self.load_strategy,
            label_to_block_id,
        )

        self.tensor_layer_loader = tensor_loader_class(
            self.stats,
            self.device_gpu,
            label_to_block_id,
            transfer_to_compute_map,
            stream_priority=self.transfer_stream_priority,
            allocation_controller=block_controller,
        )

        if prepare_state:
            self.tensor_manager_state = self.prepare_tensor_manager_state(
                self.loader_type,
                allocation_ordered,
                label_to_size_map,
                block_sizes,
                self.load_strategy,
                {},  # release_strategy is empty for block loaders
                label_to_block_id,
                self.stats,
                transfer_to_compute_map,
                shm_block_name_map=None,
            )

    def _make_shm_block_name_fn(self) -> Callable[[int], str] | None:
        """Build a block-naming function from the SHM namespace, if available."""
        if self.shm_namespace is not None:
            from flextensor.shm.namespace import weight_block_name

            ns = self.shm_namespace

            def _name_fn(index: int) -> str:
                return weight_block_name(ns, index)

            return _name_fn

        if self.use_shm:
            logger.warning("SHM enabled but shm_namespace is None — using PID-based block names (not shareable)")
        return None

    def _setup_allocation_block_loader(self, data: LoaderInputData, *, prepare_state: bool = True) -> None:
        """Setup AllocationBlockController loader for 'allocation_block_transfer' loader type.

        Args:
            data: LoaderInputData containing loader configuration.
            prepare_state: Whether to prepare and store state after setup
                (False when loading from saved state).
        """
        allocation_ordered = data.allocation_ordered
        label_to_block_id = data.label_to_block_id
        transfer_to_compute_map = data.transfer_to_compute_map

        # Apply transfer rearrangement only during fresh computation (prepare_state=True)
        # When loading from state, rearrangements were already applied
        tensor_loader_class = PreallocatedBatchTransferTensorLoader
        if self.rearrange_transfers:
            tensor_loader_class = PreallocatedBatchTransferTensorLoaderReordered
            if prepare_state:
                transfer_to_compute_map, label_to_block_id, remapped_layers = rearrange_transfers(
                    transfer_to_compute_map,
                    label_to_block_id,
                    self.stats,
                    self.min_compute_transfer_gap,
                )
                allocation_ordered = create_allocation_ordered(label_to_block_id, self.stats)
                self.load_strategy = remap_strategy(self.load_strategy, remapped_layers)

        block_name_fn = self._make_shm_block_name_fn()

        block_controller = AllocationBlockController(
            allocation_ordered,
            self.device_gpu,
            self.tensors_map,
            self.load_strategy,
            label_to_block_id,
            use_shm=self.use_shm,
            shm_block_name_map=data.shm_block_name_map,
            block_name_fn=block_name_fn,
        )

        self.tensor_layer_loader = tensor_loader_class(
            self.stats,
            self.device_gpu,
            label_to_block_id,
            transfer_to_compute_map,
            stream_priority=self.transfer_stream_priority,
            allocation_controller=block_controller,
        )

        if prepare_state:
            self.tensor_manager_state = self.prepare_tensor_manager_state(
                self.loader_type,
                allocation_ordered,
                {},  # label_to_size_map is empty for allocation block loader
                data.block_sizes,
                self.load_strategy,
                {},  # release_strategy is empty for block loaders
                label_to_block_id,
                self.stats,
                transfer_to_compute_map,
                shm_block_name_map=block_controller.shm_block_name_map,
            )

    def _benchmark_tensor_statistics(self) -> dict[int, TensorStatistics]:
        """
        Benchmark tensor transfer times for all tracked tensors.

        Returns:
            Dictionary mapping tensor_id to TensorStatistics.
        """
        benchmark_tensors = BenchmarkTensorProcessor(device_gpu=self.device_gpu)
        for _tensor_id, tensor in self.tensors_map.items():
            benchmark_tensors.process(tensor)
        return benchmark_tensors.get_results()["tensor_statistics_map"]

    def _get_memory_transfer_stats(self) -> dict[int, float]:
        """
        Get memory transfer statistics for the strategy.

        Extracts from layer stats for KnapsackStrategy, GreedyStrategy, NthLayerStrategy.
        Uses live GPU benchmarking for all other strategies.

        Returns:
            Dictionary mapping tensor size (bytes) to transfer time (ms).
        """
        if isinstance(self.tensor_manager_load_strategy, (KnapsackStrategy, GreedyStrategy, NthLayerStrategy)):
            return extract_memory_transfers_from_layer_stats(self.stats)
        return benchmark_memory_transfers(self.stats, self.device_gpu)

    def get_memory_transfer_stats(self) -> dict[int, float] | None:
        """Get the memory transfer statistics computed during profiling.

        Returns:
            Dictionary mapping tensor size (bytes) to transfer time (ms),
            or None if profiling has not yet completed.
        """
        return self.memory_transfer_stats

    def _resolve_gpu_budget(self) -> int | None:
        """Resolve GPU memory budget from fraction and current device state.

        Caps the fractional budget by actual available GPU memory to prevent OOM
        when CUDA context, KV cache, or framework buffers have already consumed memory.

        Available memory is computed as ``free_cuda + (reserved - allocated)``.
        The ``reserved - allocated`` term accounts for PyTorch allocator cache that
        CUDA reports as used but is actually reusable (reserved >= allocated is a
        PyTorch allocator invariant).

        Returns:
            Budget in bytes (capped by available memory), or None for latency mode.

        Raises:
            RuntimeError: If available GPU memory < 256 MiB.
        """
        if self._max_gpu_mem_fraction is None:
            return None

        free_cuda, total = torch.cuda.mem_get_info(self.device_gpu)
        budget = int(total * self._max_gpu_mem_fraction)

        reserved = torch.cuda.memory_reserved(self.device_gpu)
        allocated = torch.cuda.memory_allocated(self.device_gpu)
        available = free_cuda + (reserved - allocated)

        if available < _MIN_GPU_BUDGET_BYTES:
            raise RuntimeError(
                f"Insufficient free GPU memory: {available / _GiB:.2f} GiB available "
                f"(free={free_cuda / _GiB:.2f}, reserved={reserved / _GiB:.2f}, "
                f"allocated={allocated / _GiB:.2f}), "
                f"minimum required: {_MIN_GPU_BUDGET_BYTES / _GiB:.2f} GiB"
            )

        if budget > available:
            logger.warning(
                "Capping GPU memory budget from %.2f GiB to %.2f GiB "
                "(available: %.2f GiB free_cuda + %.2f GiB allocator cache)",
                budget / _GiB,
                available / _GiB,
                free_cuda / _GiB,
                (reserved - allocated) / _GiB,
            )
            budget = available

        return budget

    def prepare_infer_mode(self):
        """Prepare inference mode from fresh profiling data.

        This method is called after profiling to compute the load strategy and create
        the appropriate loader for inference.
        """
        # Analyze layer statistics and check measurement consistency
        layer_statistics_analyzer = LayerStatisticsAnalyzer(self.layer_statistics_collector)
        is_consistent = layer_statistics_analyzer.check_measurement_consistency()

        if self.enable_diagnostics and is_consistent:
            logger.info("Layer duration statistics:\n%s", layer_statistics_analyzer.format_statistics_table())

        self.layer_stats = self.layer_statistics_collector.get_layer_stats()
        # Remove tensors that are not parameters!
        self.build_parameters_mapping(self.model)
        tensors_map_keys = set(self.tensors_map.keys())
        tensors_names_keys = set(self.tensor_id_to_name_map.keys())
        difference = tensors_map_keys.difference(tensors_names_keys)
        self.layer_stats = IterativeLayerStatisticsFilter().filter_excluding_tensor_ids(
            self.layer_stats,
            difference,
        )

        # Remove duplicates
        self.layer_stats = IterativeLayerStatisticsFilter().filter_by_tensor_ids(
            self.layer_stats,
            set(self.tensors_map.keys()),
        )

        self.tensor_statistics_map = self._benchmark_tensor_statistics()
        self.stats = compute_layer_statistics(self.layer_stats, self.tensor_statistics_map)
        memory_stats = self._get_memory_transfer_stats()
        self.memory_transfer_stats = memory_stats

        if self.enable_diagnostics and memory_stats:
            logger.info("Memory transfer statistics:\n%s", format_memory_transfer_table(memory_stats))

        max_gpu_mem_bytes = self._resolve_gpu_budget()
        result = self.tensor_manager_load_strategy.compute(self.stats, memory_stats, max_gpu_mem_bytes)
        load_strategy = remove_layers_compound(result.strategy_map, self.stats, self.remove_layers_operations)
        self.traps_duration_ms = _compute_duration(self.stats)
        self.load_strategy = load_strategy
        self.block_data = result.block_data

        # Auto-enable rearrange_transfers when there are permanent gap layers
        # (layers with zero tensors).  The reordered loader can shift transfers
        # into those gap slots for better amortisation.
        if not self.rearrange_transfers:
            has_permanent_gaps = any(len(layer.tensors) == 0 and i > 0 for i, layer in enumerate(self.stats))
            if has_permanent_gaps and strategy_has_transfer_gaps(load_strategy, self.stats):
                self.rearrange_transfers = True
                logger.info("Auto-enabled rearrange_transfers: detected permanent gap layers")

        if self.enable_diagnostics:
            strategy_name = type(self.tensor_manager_load_strategy).__name__
            log_block_table(self.stats, load_strategy, self.block_data, strategy_name)

        # Create LoaderInputData from fresh computation
        if self.loader_type == "strategy":
            loader_data = LoaderInputData()  # Strategy loader uses minimal data
        else:
            if self.block_data is None:
                raise ValueError("Block data not available. Strategy must compute block_data for block loaders.")
            loader_data = LoaderInputData(
                allocation_ordered=self.block_data.allocation_ordered,
                label_to_block_id=self.block_data.label_to_block_id,
                transfer_to_compute_map=self.block_data.transfer_to_compute_map,
                label_to_size_map=self.block_data.label_to_size_map,
                block_sizes=self.block_data.block_sizes,
            )

        self._create_loader(loader_data, prepare_state=True)

    def prepare_tensor_manager_state(
        self,
        loader_type,
        allocation_ordered,
        label_to_size_map,
        block_sizes,
        load_strategy,
        release_strategy,
        label_to_block_id,
        stats,
        transfer_to_compute_map,
        shm_block_name_map,
    ):
        """Delegate to TensorManagerStateHandler for state preparation."""
        state_handler = TensorManagerStateHandler(self)
        return state_handler.prepare_state(
            loader_type=loader_type,
            allocation_ordered=allocation_ordered,
            label_to_size_map=label_to_size_map,
            block_sizes=block_sizes,
            load_strategy=load_strategy,
            release_strategy=release_strategy,
            label_to_block_id=label_to_block_id,
            stats=stats,
            transfer_to_compute_map=transfer_to_compute_map,
            shm_block_name_map=shm_block_name_map,
        )

    def restore_state(self, model, state):
        """Delegate to TensorManagerStateHandler for state restoration."""
        state_handler = TensorManagerStateHandler(self)
        state_handler.restore_state(model, state)

    def _save_state_to_file(self, file_path):
        """Internal: Save state to a specific file path."""
        if self.tensor_manager_state is None:
            raise RuntimeError(
                "Cannot save profile: no state available. "
                "Profile is only available after profiling completes and inference mode is initialized."
            )
        TensorManagerStateHandler.save_to_file(file_path, self.tensor_manager_state)

    def _load_state_from_file(self, file_path):
        """Internal: Load state from a specific file path."""
        return TensorManagerStateHandler.load_from_file(file_path)

    def save_profile(self, profile_directory: str) -> None:
        """Save current profile to directory.

        Args:
            profile_directory: Directory to save profile to
        """
        profile_path = Path(profile_directory)
        profile_path.mkdir(parents=True, exist_ok=True)
        profile_file = profile_path / "profile.json"
        self._save_state_to_file(profile_file)

    def load_state(self, profile_directory: str):
        """Load profile state from directory without restoring.

        Use this when you need access to the state object before restoring,
        e.g., to read state.gpu_tensors_names or state.view_tensors_names.

        Args:
            profile_directory: Directory containing the profile

        Returns:
            TensorManagerState: The loaded state object
        """
        profile_file = Path(profile_directory) / "profile.json"
        return self._load_state_from_file(profile_file)

    def load_profile(self, profile_directory: str, model) -> None:
        """Load profile from directory and restore state to model.

        This method loads a saved TensorManagerState and restores it to the manager,
        but does NOT configure loaders or finalize the manager for inference. It only
        sets the internal tensor_manager_state and associates the model.

        After calling this method, you MUST call the following methods in order:

        1. initialize_warmup() - Configures tensor layer loaders via prepare_infer_load_mode()
           and patches the model with inference traps via prepare_final_model().

        2. initialize_profile() - Prepares the model for profiling mode. When a saved state
           exists, this is a no-op and returns the model as-is. However, if there are changes
           that require re-profiling, this method handles the profile mode setup.

        3. initialize_inference() - Finalizes the manager for inference mode and releases
           the internal model reference (sets self.model = None) to free memory.

        Example:
            >>> tm = TensorManager(device_gpu="cuda:0", ...)
            >>> tm.load_profile("/path/to/profile", model)
            >>> model = tm.initialize_warmup()  # Required: configures loaders and patches model
            >>> model = tm.initialize_profile()  # Prepares profiling (no-op if state exists)
            >>> model = tm.initialize_inference()  # Releases model reference
            >>> # Now the model is ready for inference

        Args:
            profile_directory: Directory containing the profile.
            model: Model to restore state to. The model should match the architecture
                used when the profile was originally saved.

        Note:
            Future versions may add an optional parameter to automatically run
            finalization (initialize_warmup, initialize_profile, initialize_inference)
            after loading the profile.
        """
        state = self.load_state(profile_directory)
        self.restore_state(model, state)

    def trap(self, name):
        return self.trap_class(self, name, self.device_gpu)

    def release_memory(self):
        if self.tensor_layer_loader is not None:
            self.tensor_layer_loader.release_memory()

    def shutdown(self):
        if self.tensor_layer_loader is not None:
            self.tensor_layer_loader.shutdown()

    def get_gpu_memory_usage(self) -> GPUMemoryUsage:
        """Get GPU memory usage by FlexTensor in inference mode.

        Returns the memory used by GPU transfer blocks and unmapped tensors
        that were moved to GPU. This method should be called after the manager
        has transitioned to inference mode (after warmup and profile phases).

        Returns:
            GPUMemoryUsage: Memory breakdown with per-component bytes and MB values.
                See `GPUMemoryUsage` for field details.

        Raises:
            RuntimeError: If called before inference mode is initialized
                (tensor_layer_loader is None)

        Examples:
            >>> tensor_manager.initialize_inference()
            >>> usage = tensor_manager.get_gpu_memory_usage()
            >>> print(f"GPU blocks: {usage.blocks_mb:.2f} MB")
            >>> print(f"Unmapped tensors: {usage.unmapped_tensors_mb:.2f} MB")
            >>> print(f"Total: {usage.total_mb:.2f} MB")
        """
        if self.tensor_layer_loader is None:
            msg = "Cannot get GPU memory usage before inference mode is initialized"
            raise RuntimeError(msg)

        blocks_bytes = self.tensor_layer_loader.get_gpu_memory_bytes()
        return GPUMemoryUsage(
            blocks_bytes=blocks_bytes,
            unmapped_tensors_bytes=self._unmapped_tensors_bytes,
            total_bytes=blocks_bytes + self._unmapped_tensors_bytes,
        )

    def is_traced_trace_tensor(self, tensor):
        return isinstance(tensor, TraceTensor)

    def is_traced(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return False
        tensor_id = id(tensor)
        return tensor_id in self.traced_tensors

    def is_traced_by_id(self, tensor_id):
        return tensor_id in self.traced_tensors

    # Return type is `Any` instead of `Iterator[TensorBenchmarkMode]` or `Generator[...]` due to
    # beartype compatibility issues across Python versions. The `@contextmanager` decorator returns
    # a `_GeneratorContextManager` which beartype fails to validate against `Iterator`/`Generator`
    # on Python 3.10, while `AbstractContextManager` fails on Python 3.11+.
    @contextmanager
    def benchmark_context(self, iterations: int = 10) -> Any:
        """
        Create a benchmarking context that automatically integrates results into TensorManager.

        This method provides a context manager that creates a benchmark instance using the
        configured benchmark_cls, and automatically copies the benchmark results to the
        TensorManager's stats attributes upon context exit.

        Args:
            iterations: Number of benchmark iterations to perform

        Returns:
            Context manager that yields the benchmark instance (TensorBenchmarkMode).

        Examples:
            >>> with tensor_manager.benchmark_context() as benchmark:
            ...     model = {name: layer.to(torch.bfloat16) for name, layer in model.items()}
            >>> # Stats are automatically integrated into tensor_manager
        """
        benchmark_instance = self._benchmark_cls(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            iterations=iterations,
        )

        with benchmark_instance as benchmark:
            try:
                yield benchmark
            finally:
                results = benchmark.get_results()
                self.tensor_statistics_map = results["tensor_statistics_map"]
                self.tensors_map = results["tensors_map"]
                for tensor_id, _tensor in self.tensors_map.items():
                    self.traced_tensors.add(tensor_id)

    def _prepare_view_model_from_id_to_view_map(self, model, tensor_id_to_view_map):
        move_to_gpu_processor = MoveUnmappedTensorsToGPUProcessor(self.device_gpu, tensor_id_to_view_map)
        move_to_gpu_processor.apply(model)

        # Track GPU memory used by unmapped tensors (tensors moved to GPU that aren't in view_map)
        self._unmapped_tensors_bytes = move_to_gpu_processor.unmapped_gpu_bytes

        replace_tensor_processor = TensorReplacementProcessor(tensor_id_to_view_map)
        replace_tensor_processor.apply(model)

    def prepare_view_model(self, model):
        new_model = model
        if self.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
            block_controller = self.tensor_layer_loader.allocation_controller
            tensor_id_to_view_map = block_controller.get_tensor_id_to_view_mapping()
            self._prepare_view_model_from_id_to_view_map(new_model, tensor_id_to_view_map)

        return new_model

    def prepare_final_model(self, model):
        if self.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
            return self.prepare_view_model(model)
        if self.direct_enabled:
            return self.prepare_model(model)
        return model

    def prepare_model(self, model):
        new_model = model
        if self.direct_enabled:
            if isinstance(model, dict):
                new_model = ModelDict(self, model)
            elif isinstance(model, torch.nn.Module):
                new_model = prepare_model(model, self)
        return new_model

    def run_profile_suite(self, callback, model=None, direct_mode=True):
        """
        Run all three phases (warmup, profile, direct_mode) in sequence with a callback.

        This method automates the three-phase process:
        1. Warmup phase: Collects initial tensor statistics
        2. Profile phase: Collects detailed layer statistics with tensor loading
        3. Direct mode phase: Profiles with direct tensor access (if enabled)

        Args:
            callback: Function to call for each phase. Should accept model as argument.
            model: The model to prepare
            direct_mode: Whether to enable direct mode profiling

        Returns:
            The inference results

        """
        # Phase 1: Warmup
        self.prepare_warmup_mode()
        results = callback(model)

        # Phase 2: Profile
        self.prepare_profile_mode()
        results = callback(model)

        # Phase 3: Direct mode (if enabled)
        if direct_mode and model is not None:
            prepared_model = self.prepare_profile_direct_mode_model(model)
            self.prepare_profile_direct_mode()
            results = callback(prepared_model)

        return results
