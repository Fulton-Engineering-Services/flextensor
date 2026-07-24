# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import logging
import types
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import torch
from typing_extensions import deprecated as _deprecated

from flextensor._logging import ensure_diagnostics_visible, get_diagnostics_logger
from flextensor.benchmark_tensor_mode import BenchmarkReplace, NoOpBenchmark, TensorBenchmarkMode
from flextensor.collectors import (
    IterativeLayerStatisticsCollector,
    IterativeLayerStatisticsFilter,
    LayerStatistics,
    TensorStatistics,
)
from flextensor.config import ProfileMode
from flextensor.gpu_budget import reserve_strategy_invisible_gpu_budget, resolve_gpu_budget
from flextensor.helpers import ProfilingSuspender, TrapNestingGuard
from flextensor.host_pinning import HostPinner, HostPinRegistry, PinnedMemoryMode, make_host_pinner
from flextensor.instrumentation import instrumentable
from flextensor.layer_statistics_analyzer import report_profiling_quality
from flextensor.loaders import (
    AllocationBlockController,
    PreallocatedBatchTransferTensorLoader,
    PreallocatedBatchTransferTensorLoaderReordered,
    RawBlockController,
    TensorLayerLoader,
    TensorStrategyLoader,
    WarmupDirectTensorLoader,
)
from flextensor.memory_transfer_benchmark import (
    benchmark_memory_transfers,
    extract_memory_transfers_from_layer_stats,
    format_memory_transfer_table,
)
from flextensor.profile_block_controller import ProfileBlockController
from flextensor.state_handler import (
    LoaderInputData,
    TensorManagerState,
    TensorManagerStateHandler,
)
from flextensor.strategy import (
    BlockStrategyData,
    GreedyStrategy,
    KnapsackStrategy,
    NthLayerStrategy,
    Strategy,
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
    get_non_offloaded_tensor_ids,
    get_offload_module_tensor_ids,
    has_offload_modules,
    resolve_include_patterns,
    validate_include_patterns,
)
from flextensor.tensor_processors import (
    BenchmarkTensorProcessor,
    MoveUnmappedTensorsToGPUProcessor,
    TensorReplacementProcessor,
    compute_reachable_tensor_ids,
    create_model_with_shared_tensors,
    preprocess_model,
)
from flextensor.trap_tensor_mode import (
    Trap,
    TrapDirect,
    TrapInfer,
    TrapInferDirect,
    TrapProfileView,
    WarmupTrap,
    WarmupTrapDirect,
)
from flextensor.types import GPUMemoryUsage

_PROFILE_MODES: tuple[str, ...] = get_args(ProfileMode)

if TYPE_CHECKING:
    from flextensor.offload_manager import TensorManagerProtocol as _TensorManagerProtocol

logger = logging.getLogger(__name__)


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
            elif isinstance(tensor_manager.device_gpu, torch.device):
                tensor_names = getattr(tensor_manager, "tensor_id_to_name_map", {})
                tensor_name = (
                    tensor_names.get(tensor_id, "<unknown>") if isinstance(tensor_names, dict) else "<unknown>"
                )
                msg = (
                    "FlexTensor missing materialized tensor for traced parameter "
                    f"{tensor_name} (id={tensor_id}). This usually means a cross-layer dependency "
                    "was not recorded before profiling/inference. Disable skip discovery or seed the dependency."
                )
                raise RuntimeError(msg)
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
    """Convert iterative (collection-time) stats to strict (strategy-time) stats.

    Entries whose ``duration`` is ``None`` — i.e. labels that were discovered
    but never had a duration sample recorded — are dropped here. This is the
    phase boundary where the collector's tolerant shape is narrowed down to
    the strategy package's requirement that every ``LayerStatistics.duration``
    is a real measurement.

    Dropping is silent; which traps were untimed is reported by
    :class:`flextensor.layer_statistics_analyzer.UntimedTrapsReport`.
    """
    layers_stats = []
    for iterative_layer_stat in iterative_layer_statistics:
        if iterative_layer_stat.duration is None:
            continue

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


if TYPE_CHECKING:
    # Keep the structural contract visible near the concrete implementation.
    _TENSOR_MANAGER_PROTOCOL: type[_TensorManagerProtocol]


@instrumentable
class TensorManager:
    def __init__(
        self,
        device_gpu: str | torch.device,
        tensor_manager_load_strategy: Strategy,
        pinned_memory: bool = True,
        pinned_memory_mode: PinnedMemoryMode = "torch",
        loader_type: str = "allocation_block_transfer",
        remove_layers_operations: list[dict[str, Any]] | None = None,
        blocks: int = 4,
        move_top_level_buffers_to_gpu: bool = True,
        use_shm=False,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        enable_diagnostics: bool = False,
        max_gpu_mem_fraction: float | None = None,
        profile_mode: ProfileMode = "view",
        *,
        _direct_mode: bool = True,
        _use_trace_tensor: bool = False,
        _rearrange_transfers: bool = False,
        _compute_transfer_gap: int = 1,
        _enable_untraced_tensor_discovery: bool = True,
    ) -> None:
        """
        Initialize TensorManager with configurable tensor loading strategy.

        Args:
            device_gpu: GPU device to use
            tensor_manager_load_strategy: Strategy for loading tensors
            pinned_memory: Whether to use pinned (page-locked) memory for CPU tensors.
                Required for non-blocking transfers on a separate CUDA stream so that
                weight copies can overlap with GPU computation. Increases host memory
                pressure since pinned pages cannot be swapped. (default: True)
            pinned_memory_mode: How to pin CPU tensors when ``pinned_memory=True``.
                ``"torch"`` uses :meth:`torch.Tensor.pin_memory` (fresh pinned copy);
                ``"host_register"`` uses ``cudaHostRegister`` to pin in place (no copy,
                lower peak host RAM). On a CUDA host whose cudart binding is broken or
                missing, the requested mode falls back to ``"torch"`` with a WARNING
                log naming the cause. On a CPU-only host, ``pinned_memory=True``
                raises ``RuntimeError`` at construction — set ``pinned_memory=False``
                for CPU-only deployments. SHM segments built with
                ``pinned_memory=True`` register in place regardless of this setting;
                SHM segments built with ``pinned_memory=False`` are not registered.
                (default: "torch")
            loader_type: Type of tensor loader to use. Options:
                - "strategy": Uses TensorStrategyLoader
                - "raw_block_transfer": Uses PreallocatedBatchTransferTensorLoader with RawBlockController
                - "allocation_block_transfer": Uses PreallocatedBatchTransferTensorLoader with AllocationBlockController
            remove_layers_operations: List of operations to remove layers from the strategy map
            blocks: Number of blocks to use for the block transfer loaders (default: 4)
            move_top_level_buffers_to_gpu: Whether to move top-level model buffers to GPU
                during discovery initialization (default: True)
            include_patterns: List of patterns to include for offloading. Parameters not matching
                are kept on GPU permanently (default: None → ["*"]).
            exclude_patterns: List of patterns to exclude from offloading. Passed through to tensor
                discovery for parameter-level filtering (default: None)
            enable_diagnostics: Whether to log diagnostic information (layer duration statistics,
                block assignment table) after strategy computation (default: False)
            max_gpu_mem_fraction: Fraction of total GPU memory to use as budget, in (0.0, 1.0].
                Resolved to bytes at compute time via :func:`flextensor.gpu_budget.resolve_gpu_budget`. If None,
                no memory constraint is applied (latency mode). (default: None)
            profile_mode: Profile-phase mechanism. ``"view"``/``"getter"`` are
                profiling variants of the model-patching runtime and affect only
                the profile phase; ``"torch_function"`` selects the indirect
                runtime, which also changes warmup and inference. One of:

                * ``"view"`` — default. Patches the profile model with views
                  into a rotating GPU block plus a shared prefix for tensors
                  reused across labels. Pre-allocates substantial GPU memory
                  during profile — pick ``"getter"`` if you are tight on
                  memory. Incompatible with ``_use_trace_tensor=True``. See
                  ``profile_mode`` in ``docs/explanation/configuration.md`` for
                  the timing model and memory formula.
                * ``"getter"`` — profile model uses property getters that route
                  every parameter access through ``TensorLayerLoader``. Lower
                  profile-phase GPU footprint than ``"view"`` at the cost of
                  attribute-getter overhead in per-trap durations.
                * ``"torch_function"`` — fallback using ``TorchFunctionMode`` traps
                  that rewrite tensor arguments per call, without patching the
                  model. Significant overhead, not torch.compile-compatible; only
                  valid with ``loader_type='strategy'``.
            _direct_mode: Internal — runtime family. ``True`` (default) uses the
                model-patching runtime across warmup/profile/inference; ``False``
                uses the indirect ``TorchFunctionMode`` runtime (requires
                ``loader_type='strategy'``). ``profile_mode='torch_function'``
                forces this ``False``. Not part of the public API.
            _use_trace_tensor: Internal debug flag — enables TraceTensor + BenchmarkReplace
                for per-tensor transfer timing. Not part of the public API.
            _rearrange_transfers: Internal — whether to enable transfer rearrangement.
                Auto-enabled when permanent gap layers are detected. (default: False)
            _compute_transfer_gap: Internal — minimum layer gap between a block's compute
                and its next transfer during rearrangement. (default: 1)
            _enable_untraced_tensor_discovery: Internal — whether to discover untraced
                tensors (e.g., FP8 weights passed to Triton kernels). (default: True)
        """
        if remove_layers_operations is None:
            remove_layers_operations = []
        self.device_gpu = device_gpu
        self.tensor_statistics_map: dict[int, TensorStatistics] = {}
        self.tensors_map: Any = {}
        self.trap_class: Any = None
        self.tensor_layer_loader: Any = None
        self.layer_statistics_collector: Any = None
        self.tensor_manager_load_strategy = tensor_manager_load_strategy
        self.release_tensors = True
        self.pinned_memory = pinned_memory
        self.pinned_memory_mode = pinned_memory_mode
        self.host_pinner: HostPinner = make_host_pinner(pinned_memory, pinned_memory_mode)
        self._benchmark_cls: type[TensorBenchmarkMode] = BenchmarkReplace if _use_trace_tensor else NoOpBenchmark
        if profile_mode not in _PROFILE_MODES:
            raise ValueError(f"Invalid profile_mode={profile_mode!r}; expected one of {_PROFILE_MODES!r}.")
        self.profile_mode: ProfileMode = profile_mode
        # Runtime family: True = model-patching, False = indirect (torch_function).
        self._direct_mode: bool = False if profile_mode == "torch_function" else _direct_mode
        self.traced_tensors: set[int] = set()
        self.transfer_stream_priority = 1
        self.blocks = blocks
        self.move_top_level_buffers_to_gpu = move_top_level_buffers_to_gpu
        self.enable_untraced_tensor_discovery = _enable_untraced_tensor_discovery
        validate_include_patterns(include_patterns)
        self.include_patterns = resolve_include_patterns(include_patterns)
        self.exclude_patterns = exclude_patterns or []
        self.enable_diagnostics = enable_diagnostics
        if self.enable_diagnostics:
            ensure_diagnostics_visible()
        self._max_gpu_mem_fraction = max_gpu_mem_fraction

        self.traps_duration_ms = 0.0
        self.model_ids: set[int] = set()
        self.block_data: BlockStrategyData | None = None

        self.loader_type = loader_type
        self.remove_layers_operations = remove_layers_operations
        self.rearrange_transfers = _rearrange_transfers
        self.min_compute_transfer_gap = _compute_transfer_gap
        self.use_trace_tensor = _use_trace_tensor
        if _use_trace_tensor:
            self.is_traced = self.is_traced_trace_tensor

        self._validate_profile_mode_combination()

        self.model: Any = None
        self.tensor_id_to_name_map: dict[int, str] = {}
        self.tensor_manager_state: Any = None
        self.use_shm = use_shm
        self.shm_namespace: str | None = None
        self._unmapped_tensors_bytes = 0
        self.module_tracker: ModuleTracker | None = None
        self._non_offloaded_tensors_moved = False
        self._layer_stats_computed = False
        # Compiled re-plan state. ``_first_loader_non_destructive`` and
        # ``_replan_source_data``: without these, the first ``prepare_final_model``
        # repoints ``param.data`` onto loader-1's rolling GPU views and orphans the
        # true CPU weights, so the rebuild would copy stale rolling-buffer bytes and
        # corrupt the model.
        self._first_loader_non_destructive: bool = False
        self._replan_source_data: dict[int, torch.Tensor] = {}
        # Retained so teardown can restore dict-container entries patched with views.
        self._profile_view_model: Any = None
        self.memory_transfer_stats: dict[int, float] | None = None
        self.trap_start_event = torch.cuda.Event(enable_timing=True)
        self.trap_end_event = torch.cuda.Event(enable_timing=True)
        self.trap_nesting_guard = TrapNestingGuard()
        self._profiling_suspender = ProfilingSuspender()

    def arm_non_destructive_first_loader(self) -> None:
        """Keep source weights intact on the first inference loader build.

        Call before the INFERENCE transition when a post-compile strategy replan
        may rebuild the loader from compiled timings (pre-replan loader).
        One-shot: consumed by the first inference loader build.
        """
        self._first_loader_non_destructive = True

    def clear_replan_state(self) -> None:
        """Drop replan arming and any retained source-weight snapshot.

        Called when compiled activation is torn down or restarted so a later
        eager/view re-offload cannot keep an obsolete model-sized host copy or
        build another non-destructive loader from a stale flag.
        """
        self._first_loader_non_destructive = False
        self._replan_source_data = {}

    def _validate_profile_mode_combination(self) -> None:
        """Reject impossible direct-mode / loader-type / trace combinations."""
        block_loaders = ("allocation_block_transfer", "raw_block_transfer")

        if not self._direct_mode and self.loader_type in block_loaders:
            msg = (
                f"Indirect mode (profile_mode='torch_function' or _direct_mode=False) "
                f"is incompatible with loader_type={self.loader_type!r}. Use the direct "
                f"runtime (profile_mode='getter'/'view') or set loader_type='strategy'."
            )
            raise ValueError(msg)

        if self._direct_mode and self.profile_mode == "view" and self.use_trace_tensor:
            msg = (
                "profile_mode='view' is incompatible with _use_trace_tensor=True. "
                "Set profile_mode='getter' to use tracing."
            )
            raise ValueError(msg)

    @property
    def direct_enabled(self) -> bool:
        """``True`` when the model-patching ("direct") runtime is active.

        This is the runtime-family axis (warmup/profile/inference), held by
        :attr:`_direct_mode`. ``False`` selects the indirect ``TorchFunctionMode``
        runtime. Independent of the profile-phase variant (``view``/``getter``).
        """
        return self._direct_mode

    @property
    def _profile_uses_views(self) -> bool:
        """``True`` when profile patches the model with views into a rotating block.

        Only meaningful in the direct family; ``view`` is ignored when indirect.
        """
        return self._direct_mode and self.profile_mode == "view"

    @property
    def host_pin_registry(self) -> HostPinRegistry | None:
        """The :class:`HostPinRegistry` backing host_register mode, or ``None`` in torch mode."""
        return self.host_pinner.registry

    def should_pin_in_preprocess(self) -> bool:
        """Whether ``preprocess_model`` should pin during model preparation.

        Block-loader paths (``allocation_block_transfer`` / ``raw_block_transfer``)
        pin their own per-block buffers later, so preprocess pins nothing for
        those loader types regardless of :attr:`pinned_memory`.
        """
        if self.loader_type in ("allocation_block_transfer", "raw_block_transfer"):
            return False
        return self.pinned_memory

    def build_parameters_mapping(self, model):
        self.tensor_id_to_name_map = {}
        if isinstance(model, torch.nn.Module):
            for name, tensor in model.named_parameters():
                self.tensor_id_to_name_map[id(tensor)] = name
        elif isinstance(model, dict):
            for name, tensor in model.items():
                self.tensor_id_to_name_map[id(tensor)] = name

    def _move_non_offloaded_tensors_to_gpu(self):
        """Move non-offloaded tensors to GPU and remove them from offload tracking.

        Identifies tensors not matching include_patterns or matching
        exclude_patterns and moves them to GPU permanently. The processor
        receives a mapping of *offloaded* tensors; parameters absent from
        this mapping (the non-offloaded ones) are treated as unmapped and
        moved to GPU. Their IDs are then removed from tensors_map and
        traced_tensors so the offload system ignores them.

        After completion, ``tensors_map`` is frozen via :class:`types.MappingProxyType`
        to prevent accidental mutation during profile and inference phases.

        This method is idempotent — safe to call multiple times. Guarded by
        the _non_offloaded_tensors_moved flag.
        """
        if self._non_offloaded_tensors_moved:
            return

        non_offloaded_ids = get_non_offloaded_tensor_ids(
            self.model,
            self.tensors_map,
            self.include_patterns,
            self.exclude_patterns,
        )
        if non_offloaded_ids:
            non_offloaded_tensors = [self.tensors_map[tid] for tid in non_offloaded_ids if tid in self.tensors_map]
            non_offloaded_bytes = sum(t.nelement() * t.element_size() for t in non_offloaded_tensors)

            identity_mapping = {tid: t for tid, t in self.tensors_map.items() if tid not in non_offloaded_ids}
            processor = MoveUnmappedTensorsToGPUProcessor(self.device_gpu, identity_mapping)
            try:
                processor.apply(self.model)
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError(
                    f"GPU out of memory while moving {len(non_offloaded_tensors)} non-offloaded tensors "
                    f"({non_offloaded_bytes / 1024**3:.2f} GiB) to GPU. "
                    "Parameters NOT matching `include_patterns` (or matching `exclude_patterns`) "
                    "stay on GPU permanently. Broaden `include_patterns` or narrow "
                    "`exclude_patterns` to reduce permanent GPU memory usage."
                ) from e

            for tid in non_offloaded_ids:
                self.tensors_map.pop(tid, None)
                self.traced_tensors.discard(tid)

            self.build_parameters_mapping(self.model)

        self._non_offloaded_tensors_moved = True
        self.tensors_map = types.MappingProxyType(self.tensors_map)

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
            preprocess_model(
                self.model,
                self,
                self.device_gpu,
                pin_memory=self.should_pin_in_preprocess(),
                host_pinner=self.host_pinner,
                move_top_level_buffers_to_gpu=self.move_top_level_buffers_to_gpu,
            )
            self._move_non_offloaded_tensors_to_gpu()
        self.prepare_model_ids(self.model)  # TODO: Move to preprocess_model after remove benchmark context

        self.prepare_warmup_mode()
        return self.model

    def initialize_profile(self):
        if self.tensor_manager_state is not None:
            return self.model
        self._move_non_offloaded_tensors_to_gpu()
        profile_model = self.model
        if self.direct_enabled:
            try:
                profile_model = self.prepare_profile_direct_mode_model(self.model)
                self.prepare_profile_direct_mode()
            except BaseException:
                # Suppress teardown errors so the original failure propagates.
                try:
                    self._teardown_profile_block_controller()
                except Exception:
                    logger.exception("Profile-setup cleanup failed; preserving the original error.")
                raise
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
        # Fresh discovery cycle -> the next profile setup must rebuild stats.
        self._layer_stats_computed = False
        if self._use_direct_warmup_model():
            label_to_tensor_ids = get_offload_module_tensor_ids(
                self.model,
                self.tensors_map,
                include_patterns=self.include_patterns,
                exclude_patterns=self.exclude_patterns,
            )
            self.tensor_layer_loader = WarmupDirectTensorLoader(
                label_to_tensor_ids,
                self.tensors_map,
                self.device_gpu,
            )
            self.trap_class = WarmupTrapDirect
            return

        self.trap_class = WarmupTrap
        self.tensor_layer_loader = None
        # Initialize module tracker for manual traps (no forward patching).
        # With forward patching, we can discover tensors directly via named_parameters().
        if self.model is not None and isinstance(self.model, torch.nn.Module) and not has_offload_modules(self.model):
            self.module_tracker = ModuleTracker()
            self.module_tracker.register(self.model)

    def _use_direct_warmup_model(self) -> bool:
        return (
            self.direct_enabled
            and not self.use_trace_tensor
            and isinstance(self.model, torch.nn.Module)
            and has_offload_modules(self.model)
        )

    def prepare_profile_direct_mode_model(self, model):
        """Return the profile-phase model for ``profile_mode`` in {``view``, ``getter``}.

        Must be called before :meth:`prepare_profile_direct_mode`. View mode
        patches a copy with block-views and stages the controller into
        ``self.tensor_layer_loader``; getter mode wraps in property getters.
        No-op for ``torch_function``.
        """
        if self._profile_uses_views:
            # Patch a copy regardless of ``loader_type``: the view controller is
            # a self-contained profile-only loader, torn down before inference.
            profile_model = copy.copy(model) if isinstance(model, dict) else create_model_with_shared_tensors(model)
            return self._prepare_view_profile_model(profile_model)
        if self.loader_type in {"allocation_block_transfer", "raw_block_transfer"}:
            profile_model = copy.copy(model) if isinstance(model, dict) else create_model_with_shared_tensors(model)
            return self.prepare_model(profile_model)
        if self.direct_enabled:
            return self.prepare_model(model)
        return model

    def _compute_profile_layer_stats(self) -> None:
        """Materialize ``self.layer_stats`` for profile and unregister the module tracker.

        Idempotent: safe to call from any profile-setup path
        (``prepare_profile_direct_mode_model`` view, ``prepare_profile_direct_mode``
        getter, ``prepare_profile_mode`` torch_function).
        """
        if self._layer_stats_computed:
            return

        self.layer_stats = self.layer_statistics_collector.get_layer_stats()
        self.layer_stats = IterativeLayerStatisticsFilter().filter_by_tensor_ids(
            self.layer_stats,
            set(self.tensors_map.keys()),
        )
        if self.enable_untraced_tensor_discovery:
            self.layer_stats = discover_untraced_tensors_for_layers(
                self.layer_stats,
                self.tensors_map,
                self.model,
                self.tensor_id_to_name_map,
                module_tracker=self.module_tracker,
                include_patterns=self.include_patterns,
                exclude_patterns=self.exclude_patterns,
            )

        if self.module_tracker is not None:
            self.module_tracker.unregister()
            self.module_tracker = None

        self._layer_stats_computed = True

    def _prepare_view_profile_model(self, profile_model):
        """Build the view-mode profile controller and patch ``profile_model`` with its views.

        The controller is installed as ``self.tensor_layer_loader`` immediately
        so the failure-recovery path in :meth:`initialize_profile` (and
        :meth:`_teardown_profile_block_controller`) can find it via the loader
        slot. ``prepare_profile_direct_mode`` later wires the matching trap
        class against the same instance.
        """
        self._compute_profile_layer_stats()

        gpu_budget_bytes = resolve_gpu_budget(self._max_gpu_mem_fraction, self.device_gpu)
        controller = ProfileBlockController(
            self.layer_stats,
            self.tensors_map,
            self.device_gpu,
            pinned_memory=self.pinned_memory,
            pinned_memory_mode=self.pinned_memory_mode,
            gpu_budget_bytes=gpu_budget_bytes,
        )
        self.tensor_layer_loader = controller
        self._prepare_view_model_from_id_to_view_map(profile_model, controller.get_tensor_id_to_view_mapping())
        self._profile_view_model = profile_model
        return profile_model

    def _teardown_profile_block_controller(self) -> None:
        """Restore ``.data`` via ``self.tensors_map`` and drop the view-mode
        controller. No-op for any other profile mode.
        """
        controller = self.tensor_layer_loader
        if not isinstance(controller, ProfileBlockController):
            return

        # Pass the patched model so teardown can restore dict-container entries.
        profile_model = self._profile_view_model if self._profile_view_model is not None else self.model
        try:
            controller.teardown(profile_model, self.tensors_map)
        finally:
            # Drop the refs unconditionally; the block is already released by
            # ``teardown``'s own finally.
            self._profile_view_model = None
            self.tensor_layer_loader = None

    def prepare_profile_direct_mode(self):
        """Wire the profile-phase trap class for ``profile_mode`` in {``view``, ``getter``}.

        Must run after :meth:`prepare_profile_direct_mode_model`. View mode
        raises ``RuntimeError`` if that method hasn't built the controller
        yet; getter mode builds the loader inline.
        """
        if self._profile_uses_views:
            if not isinstance(self.tensor_layer_loader, ProfileBlockController):
                raise RuntimeError("view-mode controller missing; call prepare_profile_direct_mode_model() first.")
            self.trap_class = TrapProfileView
            return

        self._compute_profile_layer_stats()

        self.tensor_layer_loader = TensorLayerLoader(self.layer_stats, self.tensors_map, self.device_gpu)
        self.tensor_layer_loader.set_model_ids(self.model_ids)  # TODO: Fix me, remove

        self.trap_class = TrapDirect

    def prepare_profile_mode(self):
        """Wire the profile-phase trap for ``profile_mode='torch_function'``.

        Single-call setup (the ``torch_function`` path doesn't patch the model).
        """
        self._compute_profile_layer_stats()
        self.tensor_layer_loader = TensorLayerLoader(self.layer_stats, self.tensors_map, self.device_gpu)
        self.tensor_layer_loader.set_model_ids(self.model_ids)  # TODO: Fix me, remove
        self.trap_class = Trap

    def _select_infer_trap_class(self) -> type:
        """Select the eager inference trap class.

        Compiled-offload auto-patched units do **not** use this trap: their
        forwards call ``pre_compute/post_compute`` helpers directly. Manual
        ``offload_block`` keeps the eager traps (:class:`TrapInferDirect` /
        :class:`TrapInfer`).
        """
        return TrapInferDirect if self.direct_enabled else TrapInfer

    def _create_loader(
        self, data: LoaderInputData, *, prepare_state: bool = True, release_tensor_memory: bool = True
    ) -> None:
        """Create tensor layer loader from input data.

        This is the unified loader creation method that works with both fresh computation
        and restored state data.

        Args:
            data: LoaderInputData containing all necessary loader configuration.
            prepare_state: Whether to prepare and store the tensor manager state after
                creating the loader. Set to False when loading from saved state
                (state is already prepared).
            release_tensor_memory: Forwarded to block loaders; when False the source
                weights in ``tensors_map`` are preserved (used by the non-destructive
                pre-replan loader). Ignored by the ``strategy`` loader.
        """
        self.trap_class = self._select_infer_trap_class()

        if self.loader_type == "strategy":
            self._setup_strategy_loader(data, prepare_state=prepare_state)
        elif self.loader_type == "raw_block_transfer":
            self._setup_raw_block_loader(data, prepare_state=prepare_state, release_tensor_memory=release_tensor_memory)
        elif self.loader_type == "allocation_block_transfer":
            self._setup_allocation_block_loader(
                data, prepare_state=prepare_state, release_tensor_memory=release_tensor_memory
            )
        else:
            msg = f"Unknown loader type: {self.loader_type}"
            raise ValueError(msg)

    def prepare_infer_load_mode(self):
        """Prepare inference mode from saved state.

        This method is called when tensor_manager_state has been loaded from a saved profile.
        It extracts LoaderInputData from the state and creates the appropriate loader.

        Unlike :meth:`prepare_infer_mode`, this does not call
        ``report_profiling_quality``: its untimed-trap signal lives in the
        collector, which is dropped by :func:`compute_layer_statistics`
        before state is serialized. The load-time equivalent is the
        unconditional rescue warning in
        :class:`flextensor.loaders.TensorStrategyLoader`.

        When :meth:`arm_non_destructive_first_loader` was set (compiled replan
        after ``offload_from_profile`` / SHM restore), preserves source weights
        and snapshots them into ``_replan_source_data`` — same contract as
        :meth:`prepare_infer_mode`.
        """
        self.load_strategy = self.tensor_manager_state.load_strategy
        self.stats = self.tensor_manager_state.stats

        loader_data = self.tensor_manager_state.to_loader_input_data()
        # Match prepare_infer_mode: destructive by default; keep sources when a
        # post-compile replan may rebuild the loader from compiled timings.
        release_tensor_memory = not self._first_loader_non_destructive
        self._create_loader(
            loader_data,
            prepare_state=False,
            release_tensor_memory=release_tensor_memory,
        )
        if self._first_loader_non_destructive:
            # Snapshot before prepare_final_model repoints params onto rolling views.
            self._replan_source_data = {tid: self.tensors_map[tid].data for tid in self.tensors_map}
            # One-shot: later rebuilds / re-offloads must not inherit this arm.
            self._first_loader_non_destructive = False

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
            reachable_tensor_ids=compute_reachable_tensor_ids(self.model),
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

    def _setup_raw_block_loader(
        self, data: LoaderInputData, *, prepare_state: bool = True, release_tensor_memory: bool = True
    ) -> None:
        """Setup RawBlockController loader for 'raw_block_transfer' loader type.

        Args:
            data: LoaderInputData containing loader configuration.
            prepare_state: Whether to prepare and store state after setup
                (False when loading from saved state).
            release_tensor_memory: When True (default) the controller frees each
                source weight after copying into the CPU block. False preserves
                ``tensors_map`` for a later compiled-duration replan rebuild.
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
            host_pinner=self.host_pinner,
            release_tensor_memory=release_tensor_memory,
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

    def _setup_allocation_block_loader(
        self, data: LoaderInputData, *, prepare_state: bool = True, release_tensor_memory: bool = True
    ) -> None:
        """Setup AllocationBlockController loader for 'allocation_block_transfer' loader type.

        Args:
            data: LoaderInputData containing loader configuration.
            prepare_state: Whether to prepare and store state after setup
                (False when loading from saved state).
            release_tensor_memory: When True (default) the controller frees each
                source weight in ``tensors_map`` after copying it into its pinned
                CPU block. Pass False for a pre-replan loader that keeps source
                weights intact so a later replan can rebuild a destructive loader
                from compiled per-layer timings.
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
            host_pinner=self.host_pinner,
            release_tensor_memory=release_tensor_memory,
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

    def prepare_infer_mode(self):
        """Prepare inference mode from fresh profiling data.

        This method is called after profiling to compute the load strategy and create
        the appropriate loader for inference.
        """
        # View-mode profile owns a per-profile rotating block plus a shared
        # prefix; release them before inference allocates its own per-strategy
        # CPU/GPU blocks. The CUDA caching allocator keeps the freed bytes in
        # its pool, so inference's allocations reuse them without going back
        # to the driver.
        self._teardown_profile_block_controller()

        report_profiling_quality(self.layer_statistics_collector, self.enable_diagnostics)

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
            get_diagnostics_logger().info(
                "Memory transfer statistics:\n%s",
                format_memory_transfer_table(memory_stats),
            )

        # Default: destructive first build. Re-plan arms ``_first_loader_non_destructive``
        # so weights stay intact for the post-compile rebuild.
        release_tensor_memory = not self._first_loader_non_destructive
        self._compute_strategy_and_build_loader(release_tensor_memory=release_tensor_memory)

        if self._first_loader_non_destructive:
            # Snapshot original ``.data`` before ``prepare_final_model`` repoints
            # params onto loader-1's rolling views.
            self._replan_source_data = {tid: self.tensors_map[tid].data for tid in self.tensors_map}
            # One-shot: later rebuilds / re-offloads must not inherit this arm.
            self._first_loader_non_destructive = False

    def _compute_strategy_and_build_loader(self, *, release_tensor_memory: bool = True) -> None:
        """Compute the load strategy from ``self.stats`` and build the inference loader.

        Shared by :meth:`prepare_infer_mode` (first build) and
        :meth:`replan_from_compiled_durations` (corrected rebuild). Reads the
        already-computed ``self.stats`` and ``self.memory_transfer_stats`` so the
        re-plan can call it after rewriting per-layer durations, without
        re-running discovery/profiling.

        Args:
            release_tensor_memory: Forwarded to the block loader. False builds a
                non-destructive pre-replan loader that preserves ``tensors_map``
                until a compiled-duration replan rebuilds the loader.
        """
        memory_stats = self.memory_transfer_stats
        max_gpu_mem_bytes = resolve_gpu_budget(self._max_gpu_mem_fraction, self.device_gpu, logger=logger)
        budget_reservation = reserve_strategy_invisible_gpu_budget(
            max_gpu_mem_bytes,
            model=self.model,
            loader_type=self.loader_type,
            device_gpu=self.device_gpu,
            layer_stats=self.stats,
            tensors_map=self.tensors_map,
        )
        if (
            self.enable_diagnostics
            and budget_reservation.reserved_bytes
            and max_gpu_mem_bytes is not None
            and budget_reservation.effective_budget is not None
        ):
            get_diagnostics_logger().info(
                "GPU memory reserved for always-resident tensors: %.2f MiB (%d tensor(s)). "
                "Strategy budget: %.2f MiB -> %.2f MiB.",
                budget_reservation.reserved_bytes / 1024**2,
                budget_reservation.reserved_count,
                max_gpu_mem_bytes / 1024**2,
                budget_reservation.effective_budget / 1024**2,
            )
        strategy_gpu_budget = budget_reservation.effective_budget
        result = self.tensor_manager_load_strategy.compute(self.stats, memory_stats, strategy_gpu_budget)
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

        self._create_loader(loader_data, prepare_state=True, release_tensor_memory=release_tensor_memory)

    def _rewrite_durations_with_compiled(self, durations_by_label: dict[str, float]) -> tuple[list, int]:
        """Return a new ``stats`` list with compiled durations, plus the rewrite count.

        Labels absent from ``durations_by_label`` (or with a non-positive time)
        keep their existing eager duration.
        """
        new_stats = []
        rewritten = 0
        for stat in self.stats:
            compiled_ms = durations_by_label.get(stat.label)
            if compiled_ms is not None and compiled_ms > 0:
                new_stats.append(stat.model_copy(update={"duration": float(compiled_ms)}))
                rewritten += 1
            else:
                new_stats.append(stat)
        return new_stats, rewritten

    def _restore_original_weights_before_replan(self) -> None:
        """Repoint managed ``param.data`` to ``_replan_source_data`` originals.

        Restores true CPU weights before the destructive rebuild, not stale
        loader-1 rolling-buffer views.
        """
        restored = 0
        for tid, original in self._replan_source_data.items():
            param = self.tensors_map.get(tid)
            if param is not None:
                param.data = original
                restored += 1
        logger.info(
            "FlexTensor re-plan: restored %d/%d params to original weights before rebuild.",
            restored,
            len(self._replan_source_data),
        )

    def replan_from_compiled_durations(
        self,
        durations_by_label: dict[str, float],
        model,
    ) -> bool:
        """Recompute offload strategy from compiled per-layer timings and rebuild the loader.

        Rewrites durations, rebuilds a destructive loader from intact weights,
        repoints ``param.data``, and releases loader-1. Safe without recompile:
        the graph reads ``param.data`` each call and drives the loader via
        ``pre_compute/post_compute`` custom ops.

        Args:
            durations_by_label: ``label -> compiled compute time (ms)``.
            model: Live model to repoint onto the rebuilt block views.

        Returns:
            True if replanned; False if nothing to do.
        """
        if not durations_by_label:
            logger.warning("FlexTensor re-plan: no compiled durations supplied; keeping eager-profile strategy.")
            return False
        if not self.stats:
            logger.warning("FlexTensor re-plan: no layer statistics available; cannot re-plan.")
            return False
        if not self._replan_source_data:
            logger.warning(
                "FlexTensor re-plan: original-weight snapshot is missing "
                "(_first_loader_non_destructive was not armed before the first build); "
                "refusing to rebuild to avoid corrupting weights. Keeping eager-profile strategy."
            )
            return False

        new_stats, rewritten = self._rewrite_durations_with_compiled(durations_by_label)
        if rewritten == 0:
            logger.warning(
                "FlexTensor re-plan: none of the %d compiled durations matched a profiled layer label; "
                "keeping eager-profile strategy.",
                len(durations_by_label),
            )
            return False

        logger.info(
            "FlexTensor re-plan: rewriting %d/%d layer durations with compiled timings and recomputing strategy.",
            rewritten,
            len(self.stats),
        )
        saved_stats = self.stats

        self._restore_original_weights_before_replan()

        # Release loader-1 before rebuild; restore already detached ``param.data``.
        # Frees GPU blocks early so peak memory stays ~1x during re-plan.
        old_loader = self.tensor_layer_loader
        self._release_loader_blocks(old_loader)

        # Recompute strategy + build the destructive inference loader from the
        # (now-restored) source weights. ``self.model`` is temporarily set so the
        # strategy-invisible GPU budget reservation sees the real residents.
        self.model = model
        try:
            self.stats = new_stats
            self._compute_strategy_and_build_loader(release_tensor_memory=True)
            # Repoint the live model's params onto the rebuilt block views.
            self.prepare_final_model(model)
        except Exception as exc:
            self.stats = saved_stats
            raise RuntimeError(
                "FlexTensor compiled-offload: replan failed after releasing the inference loader. "
                "Inference is unsafe. Restart the process; free GPU memory and retry, or use "
                "compiled view-profile (compile_fn + profile_mode='view') instead of replan."
            ) from exc
        finally:
            self.model = None

        # The rebuild has consumed the originals (offloaded weights were copied
        # into the new blocks and emptied); drop the snapshot so the host memory
        # it pinned can be reclaimed.
        self._replan_source_data = {}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def _release_loader_blocks(self, loader) -> None:
        """Release a loader's pinned CPU blocks and GPU allocations.

        Re-plan helper: frees loader-1 before rebuild. Best-effort — failures
        are logged, never raised.
        """
        if loader is None:
            return
        shutdown = getattr(loader, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # best-effort cleanup must not abort inference
                logger.exception("FlexTensor re-plan: error releasing the previous loader's CPU blocks.")
        controller = getattr(loader, "allocation_controller", None)
        release_gpu_blocks = getattr(controller, "release_gpu_blocks", None)
        if callable(release_gpu_blocks):
            try:
                release_gpu_blocks()
            except Exception:  # best-effort cleanup must not abort inference
                logger.exception("FlexTensor re-plan: error releasing the previous loader's GPU blocks.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def load_state(self, profile_directory: str) -> TensorManagerState:
        """Load profile state from directory without restoring.

        Use this when you need access to the state object before restoring,
        e.g., to read state.gpu_tensors_names or state.view_tensors_names.

        Args:
            profile_directory: Directory containing the profile

        Returns:
            The loaded state object
        """
        profile_file = Path(profile_directory) / "profile.json"
        return self._load_state_from_file(profile_file)

    def load_profile(self, profile_directory: str, model: torch.nn.Module) -> None:
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

    def suspend_profiling(self) -> None:
        """Suppress duration recording; see :class:`~flextensor.helpers.ProfilingSuspender`."""
        self._profiling_suspender.suspend()

    def resume_profiling(self) -> None:
        """Release one outstanding :meth:`suspend_profiling` call."""
        self._profiling_suspender.resume()

    # See `flextensor.helpers.ProfilingSuspender.suspended` for the `-> Any` rationale
    # (beartype + @contextmanager cross-version compatibility).
    @contextmanager
    def pause_profiling(self) -> Any:
        """Context-manager form of :meth:`suspend_profiling` / :meth:`resume_profiling`.

        Shares the refcount, so it nests freely with the raw methods.
        """
        with self._profiling_suspender.suspended():
            yield

    def is_profiling_suspended(self) -> bool:
        """``True`` while any :meth:`suspend_profiling` call is outstanding."""
        return self._profiling_suspender.is_suspended()

    def clear_profiling_durations(self) -> None:
        """Clear all accumulated duration measurements.

        No-op if no collector is active.
        """
        if self.layer_statistics_collector is not None:
            self.layer_statistics_collector.clear_duration_measurements()

    def record_tensors(self, label: str, tensor_ids: list[int] | set[int]) -> None:
        """Record tensor IDs for a DISCOVERY-phase trap exit.

        Discovery captures only the tensor-to-layer mapping — ``WarmupTrap``
        does not measure per-iteration durations, so there is no duration
        sample to record here. Suspension is not consulted because the
        mapping is a hard prerequisite for every later phase.

        No-op if no collector is active.
        """
        if self.layer_statistics_collector is None:
            return
        self.layer_statistics_collector.add_tensors(label, tensor_ids)

    def record_all(self, label: str, tensor_ids: list[int] | set[int], duration_ms: float) -> None:
        """Record tensor IDs and duration for a PROFILING-phase trap exit.

        No-op if no collector is active or profiling is suspended. Suspending
        skips both the tensor IDs and the duration sample so a paused warmup
        pass cannot widen per-layer tensor sets on data-dependent models
        (MoE / conditional branches).
        """
        if self.layer_statistics_collector is None:
            return
        if self.is_profiling_suspended():
            return
        self.layer_statistics_collector.add_tensors(label, tensor_ids)
        self.layer_statistics_collector.add_duration(label, duration_ms)

    def record_duration(self, label: str, duration_ms: float) -> None:
        """Record a duration sample unless profiling is suspended.

        No-op if no collector is active or profiling is suspended.
        """
        if self.layer_statistics_collector is None:
            return
        if not self.is_profiling_suspended():
            self.layer_statistics_collector.add_duration(label, duration_ms)

    def trap(self, name):
        return self.trap_class(self, name, self.device_gpu)

    def release_memory(self):
        if self.tensor_layer_loader is not None:
            self.tensor_layer_loader.release_memory()

    def shutdown(self):
        try:
            # If a view-mode profile is still in flight, restore ``.data`` first
            # so model parameters don't end up aliasing freed GPU storage.
            self._teardown_profile_block_controller()
            if self.tensor_layer_loader is not None:
                self.tensor_layer_loader.shutdown()
        finally:
            # Pins must be released even if loader teardown raises, but a
            # release_all() exception inside this finally block would mask
            # the original loader exception — log and swallow instead.
            try:
                self.host_pinner.release_all()
            except Exception:
                logger.exception(
                    "host_pinner.release_all() raised during shutdown; remaining pins will leak until process exit"
                )

    def get_gpu_memory_usage(self) -> GPUMemoryUsage:
        """Get GPU memory usage by FlexTensor in inference mode.

        Returns the memory used by GPU transfer blocks and unmapped tensors
        that were moved to GPU. This method should be called after the manager
        has transitioned to inference mode (after discovery and profiling phases).

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
            host_pinner=self.host_pinner,
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

        # Tensors outside any per-layer block (embeddings, lm_head, top-level
        # norms, etc.) are moved to GPU here (device copy). This is the documented
        # normal pattern, not a rescue: ``get_gpu_memory_usage()`` exposes the
        # footprint via ``unmapped_tensors_bytes``, and profile-coverage gaps
        # for *trapped* labels are surfaced upstream by
        # ``report_profiling_quality`` (fresh runs) and the strategy-loader
        # rescue warning (saved-state load).
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

    @_deprecated(
        "TensorManager.run_profile_suite() is deprecated and will be removed "
        "in v0.4.0. Use OffloadManager / offload() instead."
    )
    def run_profile_suite(
        self,
        callback: Callable[[Any], Any],
        model: Any | None = None,
        direct_mode: bool = True,
    ) -> Any:
        """Run all three phases (discovery, profiling, direct_mode) in sequence with a callback.

        .. deprecated:: v0.3
            Use ``initialize_warmup`` / ``initialize_profile`` /
            ``initialize_inference`` directly. Removed in v0.4.0.

        Args:
            callback: Function to call for each phase. Should accept model as argument.
            model: The model to prepare
            direct_mode: Whether to enable direct mode profiling

        Returns:
            The inference results
        """
        # Phase 1: Discovery
        self.prepare_warmup_mode()
        results = callback(model)

        # Phase 2: Profile
        self.prepare_profile_mode()
        results = callback(model)

        # Phase 3: Direct mode (if enabled)
        if direct_mode and model is not None:
            # Preserve historical semantics: ``direct_mode=True`` always meant
            # the property-getter profile (``profile_mode='getter'``) in the
            # direct family, regardless of the (newer) ``profile_mode`` setting
            # (now defaulting to "view"). Forcing both keeps the deprecation
            # window backwards compatible.
            saved_profile_mode = self.profile_mode
            saved_direct_mode = self._direct_mode
            self.profile_mode = "getter"
            self._direct_mode = True
            try:
                prepared_model = self.prepare_profile_direct_mode_model(model)
                self.prepare_profile_direct_mode()
                results = callback(prepared_model)
            finally:
                self.profile_mode = saved_profile_mode
                self._direct_mode = saved_direct_mode

        return results
