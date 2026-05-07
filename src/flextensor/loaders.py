# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch

from flextensor.allocation_block import AllocationBlock, AllocationManager
from flextensor.host_pinning import HostPinner
from flextensor.instrumentation import instrumentable
from flextensor.strategy_operations import find_transfers_for_preload

from .collectors import IterativeLayerStatistics, LayerStatistics, TensorStatistics
from .utils import clear_and_delete_tensor, delete_tensor, is_dense_layout

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================


def _compute_preload(
    layer_stats: list[LayerStatistics],
    strategy: dict[str, list[TensorStatistics]],
) -> set[int]:
    """Compute tensor IDs that need to be preloaded.

    Args:
        layer_stats: Layer statistics.
        strategy: Strategy mapping layer labels to tensors to offload.

    Returns:
        Set of tensor IDs that are not included in the strategy.
    """
    preload_tensors_ids: set[int] = set()
    strategy_tensors_ids: set[int] = set()
    for layer_stat in layer_stats:
        # add tensors from current layer
        label = layer_stat.label
        if label in strategy:
            layer_strategy = strategy[label]
            for tensor_info in layer_strategy:
                strategy_tensors_ids.add(tensor_info.tensor_id)
        # check tensors which are not included in the strategy layer
        for tensor_info in layer_stat.tensors:
            if tensor_info.tensor_id not in strategy_tensors_ids:
                preload_tensors_ids.add(tensor_info.tensor_id)
    return preload_tensors_ids


def _compute_untimed_traced_preload(
    layer_stats: list[LayerStatistics],
    tensors_map: Mapping[int, torch.Tensor],
) -> set[int]:
    """Tensor IDs in ``tensors_map`` that appear in no layer at all.

    These are *traced but untimed* — their trap fired (so the tensor is
    registered in ``tensors_map`` and ``traced_tensors``) but no duration
    sample was recorded, so the label was dropped by
    :func:`flextensor.tensor_manager.compute_layer_statistics`. Without
    this rescue, ``TensorStrategyLoader.get(...)`` returns ``None`` for
    them and the trap path falls through to the original CPU tensor
    (a device-mismatch crash for parameterized GPU ops, or silent CPU
    compute when all operands happen to be CPU).

    Block loaders don't need an analogous rescue:
    ``MoveUnmappedTensorsToGPUProcessor`` already routes any non-view-mapped
    tensor to GPU as a *routing* decision (mapped → block view, unmapped →
    GPU pin), so the strategy-loader fall-through-to-CPU failure mode
    cannot occur there. The cost on the strategy path is permanent GPU
    residence for the rescued IDs (they sit in ``cpu_to_gpu_map`` for the
    loader's lifetime).

    The typical trigger is a profile-coverage gap (e.g. vLLM's
    ``logits_processor`` firing only at decode-time but profiled with
    prefill-shaped inputs). Anchored by
    ``tests/unit/test_untimed_traps_runtime.py``.

    Note: ``TensorStrategyLoader`` further narrows this set against
    the model's reachable tensor IDs (passed via
    ``reachable_tensor_ids``) so that ``tensors_map`` entries from any
    future non-model code path don't get moved to GPU as a side effect.
    """
    layer_tensor_ids: set[int] = set()
    for layer_stat in layer_stats:
        for tensor_info in layer_stat.tensors:
            layer_tensor_ids.add(tensor_info.tensor_id)
    return {tid for tid in tensors_map if tid not in layer_tensor_ids}


def _compute_peak_memory_from_strategy(
    strategy_map: dict[str, list[TensorStatistics]],
    release_strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
) -> int:
    """Compute peak GPU memory by simulating the sliding window execution.

    This function simulates the tensor loading and release pattern during inference
    to determine the maximum memory required at any point. It iterates through layers
    in execution order, tracking which tensors are loaded (from strategy_map) and
    released (from release_strategy_map).

    Args:
        strategy_map: Dictionary mapping layer labels to lists of TensorStatistics
            representing tensors to load when entering that layer.
        release_strategy_map: Dictionary mapping layer labels to lists of TensorStatistics
            representing tensors to release when exiting that layer.
        layer_stats: List of LayerStatistics defining the execution order.

    Returns:
        Peak GPU memory in bytes - the maximum total size of tensors simultaneously
        loaded at any point during execution.
    """
    if not layer_stats:
        return 0

    # Track currently loaded tensors by their ID to avoid double-counting
    loaded_tensor_sizes: dict[int, int] = {}
    peak_memory = 0
    current_memory = 0

    for layer in layer_stats:
        label = layer.label

        # Load tensors from strategy_map when entering the layer
        if label in strategy_map:
            for tensor_info in strategy_map[label]:
                tensor_id = tensor_info.tensor_id
                if tensor_id not in loaded_tensor_sizes:
                    loaded_tensor_sizes[tensor_id] = tensor_info.size_bytes
                    current_memory += tensor_info.size_bytes

        # Update peak after loading (this is when memory is highest for this layer)
        peak_memory = max(peak_memory, current_memory)

        # Release tensors from release_strategy_map when exiting the layer
        if label in release_strategy_map:
            for tensor_info in release_strategy_map[label]:
                tensor_id = tensor_info.tensor_id
                if tensor_id in loaded_tensor_sizes:
                    current_memory -= loaded_tensor_sizes[tensor_id]
                    del loaded_tensor_sizes[tensor_id]

    return peak_memory


class Loader(ABC):
    @abstractmethod
    def enter(self, label: str) -> None:
        """
        Called when a particular block/tensor group is ready for compute/transfer.
        """

    @abstractmethod
    def exit(self, label: str) -> None:
        """
        Called when a particular block/tensor group is finished with compute/transfer.
        """

    def shutdown(self) -> None:
        """
        Release resources and clean up allocations.
        """
        return None


@instrumentable
class TensorLayerLoader(Loader):
    """
    This loader manages tensor loading for specific layers within a trap/block context.
    It loads all required tensors when entering a layer and releases them upon exit.

    Note: This loader uses clear_and_delete_tensor by default for more aggressive memory release.
    It has been observed that the standard delete_tensor function does not work effectively
    in this context, potentially leading to insufficient cleanup.
    Further investigation is needed.
    """

    def __init__(
        self,
        layer_stats: list[IterativeLayerStatistics],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
        delete_tensor_func: Callable[[torch.Tensor], None] = clear_and_delete_tensor,
    ) -> None:
        self.layer_stats = layer_stats
        self.tensors_map = tensors_map
        self.layer_stats_map = {}
        self.load_time_ms = 0
        for statistics in layer_stats:
            stats_list = []
            if statistics.label in self.layer_stats_map:
                stats_list = self.layer_stats_map[statistics.label]
            stats_list.extend(statistics.tensor_ids)
            self.layer_stats_map[statistics.label] = stats_list
        self.loaded = []
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map = {}
        self.del_tensor = delete_tensor_func

        self.model_ids = set()

    def set_model_ids(self, tensor_ids):
        self.model_ids = tensor_ids

    def enter(self, label: str) -> None:
        """Load tensors for the specified layer."""
        start_time_ns = time.time_ns()
        tensor_ids_list: list[int] = []
        if label in self.layer_stats_map:
            tensor_ids_list = self.layer_stats_map[label]
        # TODO: after consolidating stats, remove unique_tensor_ids
        unique_tensor_ids = set()
        for tensor_id in tensor_ids_list:
            if tensor_id in unique_tensor_ids:
                continue
            unique_tensor_ids.add(tensor_id)
            tensor = self.tensors_map[tensor_id]
            if tensor.device != self.device_gpu:
                gpu_tensor = tensor.to(device=self.device_gpu, copy=True)
                self.cpu_to_gpu_map[tensor_id] = gpu_tensor
        torch.cuda.synchronize()
        end_time_ns = time.time_ns()
        duration_ms = (end_time_ns - start_time_ns) / 1e6
        self.load_time_ms += duration_ms

    def exit(self, label: str) -> None:
        """Release tensors for the specified layer."""
        tensor_ids_list = []
        if label in self.layer_stats_map:
            tensor_ids_list = self.layer_stats_map[label]
        # TODO: after consolidating stats, remove unique_tensor_ids
        unique_tensor_ids = set()
        for tensor_id in tensor_ids_list:
            if tensor_id in unique_tensor_ids:
                continue
            unique_tensor_ids.add(tensor_id)

            gpu_tensor = self.cpu_to_gpu_map[tensor_id]
            if tensor_id in self.model_ids:
                self.del_tensor(gpu_tensor)
            else:
                del gpu_tensor
            del self.cpu_to_gpu_map[tensor_id]

    def get(self, tensor_id: int) -> torch.Tensor | None:
        """Get tensor by ID if loaded."""
        tensor = None
        if tensor_id in self.cpu_to_gpu_map:
            tensor = self.cpu_to_gpu_map[tensor_id]
        return tensor

    def release_memory(self) -> None:
        """Release memory (placeholder implementation)."""


@instrumentable
class TensorStrategyLoader(Loader):
    """
    This loader implements strategic tensor management using configurable loading and release strategies.
    It supports preloading, asynchronous transfers, and scheduled memory cleanup based on defined policies.
    """

    def __init__(
        self,
        layer_stats: list[LayerStatistics],
        strategy_map: dict[str, list[TensorStatistics]],
        release_strategy_map: dict[str, list[TensorStatistics]],
        tensors_map: Mapping[int, torch.Tensor],
        device_gpu: torch.device,
        release_tensors: bool,
        stream_priority: int,
        delete_tensor_func: Callable[[torch.Tensor], None] = delete_tensor,
        *,
        reachable_tensor_ids: set[int] | None = None,
    ) -> None:
        self.device_gpu = device_gpu
        self.layer_stats = layer_stats
        self.strategy_map = strategy_map
        self.release_strategy_map = release_strategy_map
        self.tensors_map = tensors_map
        self.cpu_to_gpu_map = {}
        # create transfer stream with higher priority
        self.transfer_stream = torch.cuda.Stream(device=self.device_gpu, priority=stream_priority)
        self.transfer_ongoing = set()
        self.release_tensors = release_tensors
        self.use_events = True

        # preload tensors
        # TODO: create method for preloading tensors
        self.preload_ids = _compute_preload(layer_stats, strategy_map)
        # Rescue traced-but-untimed tensors so .get(...) doesn't return None
        # and fall through to the CPU tensor at trap time.
        rescue_ids = _compute_untimed_traced_preload(layer_stats, self.tensors_map)
        if reachable_tensor_ids is not None:
            # Narrow the rescue to model-reachable tensors only.
            rescue_ids &= reachable_tensor_ids
        if rescue_ids:
            # Only signal on the saved-state path; report_profiling_quality covers fresh runs.
            rescue_bytes = sum(
                self.tensors_map[tid].numel() * self.tensors_map[tid].element_size() for tid in rescue_ids
            )
            # Untimed-rescued tensors aren't in any layer_stat (that's *why*
            # they're rescued), so layer-name lookup isn't possible here.
            # Surface the sorted tensor IDs (truncated) so a user grepping a
            # profile dump can cross-reference which trapped tensors lost
            # their duration sample.
            sorted_ids = sorted(rescue_ids)
            id_hint = ", ".join(str(tid) for tid in sorted_ids[:8])
            if len(sorted_ids) > 8:
                id_hint += f", ... ({len(sorted_ids) - 8} more)"
            LOGGER.warning(
                "Untimed-trap rescue activated: %d tensor(s) (%.2f MiB) force-pinned to GPU "
                "[ids: %s]. Likely a profile-coverage gap.",
                len(rescue_ids),
                rescue_bytes / (1024 * 1024),
                id_hint,
            )
        self.preload_ids |= rescue_ids
        for tensor_id, tensor in self.tensors_map.items():
            if tensor_id in self.preload_ids:
                gpu_tensor = tensor.to(device=device_gpu, copy=True)
                self.cpu_to_gpu_map[tensor_id] = gpu_tensor

        torch.cuda.synchronize()

        self.scheduled_releases = {}
        self.transfer_events = {}
        self.scheduled_transfers = {}
        self.del_tensor = delete_tensor_func

        self.tensor_id_to_release_label = {}
        self.load_to_release_labels_map = {}
        self.prepare_load_to_release_labels_map()

    def prepare_load_to_release_labels_map(self):
        for label, strategy in self.release_strategy_map.items():
            for tensor_info in strategy:
                self.tensor_id_to_release_label[tensor_info.tensor_id] = label

        for label, strategy in self.strategy_map.items():
            release_labels = set()
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                release_label = self.tensor_id_to_release_label[tensor_id]
                release_labels.add(release_label)
            self.load_to_release_labels_map[label] = release_labels

    def enter(self, label: str) -> None:
        if label not in self.strategy_map:
            return
        strategy = self.strategy_map[label]

        release_labels = self.load_to_release_labels_map[label]
        for release_label in release_labels:
            if release_label in self.scheduled_releases:
                del self.scheduled_releases[release_label]

        with torch.cuda.stream(self.transfer_stream):
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                tensor = self.tensors_map[tensor_id]
                if tensor_id not in self.cpu_to_gpu_map:
                    self.cpu_to_gpu_map[tensor_id] = tensor.to(device=self.device_gpu, non_blocking=True, copy=True)

        # transfer events
        event = torch.cuda.Event()
        event.record(self.transfer_stream)
        self.scheduled_transfers[label] = event
        for tensor_info in strategy:
            tensor_id = tensor_info.tensor_id
            self.transfer_events[tensor_id] = event

    def release_for_label(self, label):
        release_strategy = self.release_strategy_map[label]
        for tensor_info in release_strategy:
            if tensor_info.tensor_id in self.cpu_to_gpu_map:
                gpu_tensor = self.cpu_to_gpu_map[tensor_info.tensor_id]
                self.del_tensor(gpu_tensor)
                del self.cpu_to_gpu_map[tensor_info.tensor_id]

    def release_memory(self):
        if not self.release_tensors:
            return
        labels_passed = []
        for key, ev_item in self.scheduled_releases.items():
            if ev_item.query():
                labels_passed.append(key)
        for key in labels_passed:
            self.release_for_label(key)
            del self.scheduled_releases[key]

        # release transfer events
        for label in labels_passed:
            if label not in self.strategy_map:
                continue
            strategy = self.strategy_map[label]
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                if tensor_id in self.transfer_events:
                    del self.transfer_events[tensor_id]

    def exit(self, label: str) -> None:
        if not self.release_tensors:
            return

        if label not in self.release_strategy_map:
            if self.use_events:
                self.release_memory()
            return
        if self.use_events:
            self.release_memory_with_events(label)
        else:
            self.release_memory_with_stream_synchronize(label)

    def release_memory_with_events(self, label):
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.release_memory()
        self.scheduled_releases[str(label)] = event

    def release_memory_with_stream_synchronize(self, label):
        torch.cuda.current_stream().synchronize()
        self.release_for_label(label)

    def get(self, tensor_id):
        if tensor_id in self.transfer_events:
            event = self.transfer_events[tensor_id]
            torch.cuda.current_stream().wait_event(event)
            del self.transfer_events[tensor_id]

        tensor = None
        if tensor_id in self.cpu_to_gpu_map:
            tensor = self.cpu_to_gpu_map[tensor_id]
        return tensor

    def get_gpu_memory_bytes(self) -> int:
        """Get peak GPU memory used during inference execution.

        Computes the maximum GPU memory that will be used at any point during
        inference by simulating the sliding window pattern of tensor loading
        and releasing.

        Returns:
            Peak GPU memory in bytes - the maximum total size of tensors
            simultaneously loaded at any point during execution.
        """
        return _compute_peak_memory_from_strategy(self.strategy_map, self.release_strategy_map, self.layer_stats)


@instrumentable
class RawBlockController:
    def __init__(
        self,
        label_to_size_map,
        block_sizes,
        device_gpu,
        tensors_map,
        strategy_map,
        label_to_block_id,
        *,
        host_pinner: HostPinner,
    ):
        self.block_map_cpu = {}
        self.block_map_cpu_sizes = {}
        self.block_map_gpu = {}
        self.device_cpu = "cpu"
        self.device_gpu = device_gpu
        self.label_to_block_id = label_to_block_id
        self.host_pinner = host_pinner

        # Process each layer: allocate block, copy tensors, release tensors
        # This interleaved approach reduces peak memory compared to allocating all blocks upfront
        self.block_meta_map = {}
        self.label_to_cpu_tensor_id_map = {}
        for label, strategy in strategy_map.items():
            if len(strategy) == 0:
                continue

            # Allocate CPU block for this layer.
            total_size = label_to_size_map[label]
            block = torch.zeros(total_size, dtype=torch.uint8, device=self.device_cpu)
            block = self.host_pinner.pin(block)
            self.block_map_cpu[label] = block
            self.block_map_cpu_sizes[label] = block.size(0)

            # Collect tensors for this layer
            tensors_list = []
            tensor_ids_list = []
            # TODO: after consolidating stats, remove unique_tensor_ids
            unique_tensor_ids = set()
            for tensor_info in strategy:
                tensor_id = tensor_info.tensor_id
                if tensor_id in unique_tensor_ids:
                    continue
                unique_tensor_ids.add(tensor_id)
                tensor = tensors_map[tensor_id]
                tensors_list.append(tensor)
                tensor_ids_list.append(tensor_id)

            # combine_tensors copies tensor data to block and releases each tensor immediately
            meta = self.combine_tensors(tensors_list, block)
            self.block_meta_map[label] = meta
            self.label_to_cpu_tensor_id_map[label] = tensor_ids_list

        # Allocate GPU blocks after all CPU blocks are populated
        for block_id, total_size in block_sizes.items():
            block = torch.zeros(total_size, dtype=torch.uint8, device=self.device_gpu)
            self.block_map_gpu[block_id] = block

        self.gpu_block_view_map = {}
        for label, block_id in label_to_block_id.items():
            gpu_block = self.block_map_gpu[block_id]
            cpu_block_size = self.block_map_cpu_sizes[label]
            gpu_block_view = gpu_block[:cpu_block_size]
            self.gpu_block_view_map[label] = gpu_block_view

        self.prepare_tensor_id_to_view_mapping()

    def prepare_tensor_id_to_view_mapping(self):
        self.label_to_tensor_views_map = {}
        for label, meta in self.block_meta_map.items():
            block_id = self.label_to_block_id[label]
            gpu_block = self.block_map_gpu[block_id]
            tensor_views = self.reconstruct_original_shapes(gpu_block, meta)
            self.label_to_tensor_views_map[label] = tensor_views

        self.tensor_id_to_view_map = {}
        for label, views in self.label_to_tensor_views_map.items():
            tensor_ids = self.label_to_cpu_tensor_id_map[label]
            for tensor_id, view in zip(tensor_ids, views, strict=False):
                self.tensor_id_to_view_map[tensor_id] = view

    def combine_tensors(self, tensors, combined):
        metadata = []
        current_idx = 0
        for tensor in tensors:
            tensor_shape = tensor.shape
            tensor_dtype = tensor.dtype
            tensor_stride = tensor.stride()
            num_elements = tensor.numel()
            element_size = tensor.element_size()
            nbytes = num_elements * element_size

            start_idx = current_idx
            end_idx = current_idx + nbytes

            if tensor.is_contiguous():
                byte_tensor = tensor.view(torch.uint8).flatten()
            elif is_dense_layout(tensor):
                # Dense non-C-contiguous (e.g. Fortran-contiguous from
                # weight.t()): access raw storage bytes so the original
                # memory layout is preserved in the combined block.
                offset_bytes = tensor.storage_offset() * element_size
                byte_tensor = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                byte_tensor.set_(tensor.untyped_storage(), offset_bytes, (nbytes,))
            else:
                # Non-dense with gaps: copy to contiguous layout first.
                contig = tensor.contiguous()
                tensor_stride = contig.stride()
                byte_tensor = contig.view(torch.uint8).flatten()

            combined[start_idx:end_idx] = byte_tensor

            metadata.append((start_idx, end_idx, tensor_shape, tensor_dtype, tensor_stride))
            current_idx = end_idx

            tensor.data = torch.empty(0, device="cpu", dtype=tensor_dtype)

        return metadata

    def reconstruct_original_shapes(
        self,
        combined_tensor: torch.Tensor,
        metadata: list[tuple[int, int, torch.Size, torch.dtype, tuple[int, ...]]],
    ) -> list[torch.Tensor]:
        """
        Reconstruct tensors with their original shapes, dtypes and strides
        from the uint8 combined tensor.

        Args:
            combined_tensor: The flattened combined uint8 tensor (reinterpreted bytes)
            metadata: List of (start_idx, end_idx, original_shape, dtype, strides) tuples

        Returns:
            List of tensors with original shapes, dtypes and strides
        """
        ust = combined_tensor.untyped_storage()
        reconstructed = []
        for start_idx, _end_idx, original_shape, dtype, original_strides in metadata:
            t = torch.empty(0, dtype=dtype, device=combined_tensor.device)
            storage_offset_elems = start_idx // t.element_size()
            t.set_(ust, storage_offset_elems, original_shape, original_strides)
            reconstructed.append(t)

        return reconstructed

    def get_tensor_id_to_view_mapping(self):
        return self.tensor_id_to_view_map

    def schedule_transfer(self, label, non_blocking=True):
        cpu_block = self.block_map_cpu[label]
        gpu_block_view = self.gpu_block_view_map[label]
        gpu_block_view.copy_(cpu_block, non_blocking=non_blocking)

    def shutdown(self) -> None:
        """Release any resources associated with this controller."""
        # Currently nothing explicit to free; method present for API compatibility.
        return None

    def get_gpu_memory_bytes(self) -> int:
        """Get total GPU memory used by transfer blocks.

        Returns:
            Total GPU memory in bytes allocated for transfer blocks.
        """
        total_bytes = 0
        for block in self.block_map_gpu.values():
            total_bytes += block.numel() * block.element_size()
        return total_bytes


@instrumentable
class AllocationBlockController:
    def __init__(
        self,
        allocation_ordered: dict[int, list[str]],
        device_gpu: torch.device,
        tensors_map: Mapping[int, torch.Tensor],
        strategy_map: dict[str, list[TensorStatistics]],
        label_to_block_id: dict[str, int],
        use_shm: bool = False,
        shm_block_name_map: dict[str, str] | None = None,
        block_name_fn: Callable[[int], str] | None = None,
        *,
        host_pinner: HostPinner,
    ) -> None:
        self.block_map_cpu = {}  # label to cpu block
        self.label_to_cpu_tensor_id_map = {}
        self.device_cpu = "cpu"
        self.device_gpu = device_gpu
        self.device_gpu_str = self._format_device_str(device_gpu)
        self.block_map_gpu = {}  # block_id to gpu block
        self.label_to_gpu_block = {}
        self.use_shm = use_shm
        self.load_model_from_shm = use_shm and shm_block_name_map is not None
        self.shm_block_name_map = shm_block_name_map or {}
        self._block_name_fn = block_name_fn or (lambda index: f"ft_{os.getpid()}_{index}")
        self.host_pinner = host_pinner

        for index, (_block_id, layers) in enumerate(allocation_ordered.items()):
            shm_prefix = None
            if self.use_shm:
                if layers[0] not in self.shm_block_name_map:
                    shm_prefix = self._block_name_fn(index)
                else:
                    shm_prefix = self.shm_block_name_map[layers[0]]
            allocation_manager = AllocationManager(
                shm_block_name_prefix=shm_prefix,
                load_from_shm=self.load_model_from_shm,
                pinned_memory=True,
                host_pinner=self.host_pinner,
                release_tensor_memory=True,
            )
            for label in layers:
                if self.use_shm and label not in self.shm_block_name_map:
                    self.shm_block_name_map[label] = shm_prefix
                strategy = strategy_map[label]
                if len(strategy) == 0:
                    continue
                block = allocation_manager.block()
                self.block_map_cpu[label] = block
                # _add_tensors calls block.allocate() which releases each tensor immediately after copying
                tensors_list, tensor_ids_list = self._add_tensors(label, strategy, tensors_map, block)
                self.label_to_cpu_tensor_id_map[label] = tensor_ids_list
                self.release_memory(tensors_list)

            gpu_block = allocation_manager.create_max_block(device=self.device_gpu_str)
            for label in layers:
                self.label_to_gpu_block[label] = gpu_block

        self.label_to_tensor_views_map = {}
        self.gpu_block_view_map = {}
        for label, cpu_block in self.block_map_cpu.items():
            gpu_block = self.label_to_gpu_block[label]
            gpu_tensor_views, gpu_block_view = cpu_block.project_views(gpu_block)
            self.label_to_tensor_views_map[label] = gpu_tensor_views
            block_id = label_to_block_id[label]
            self.block_map_gpu[block_id] = gpu_block_view  # TODO: REMOVE!
            self.gpu_block_view_map[label] = gpu_block_view

        self.prepare_tensor_id_to_view_mapping()

    def _add_tensors(
        self,
        label: str,
        strategy: list[TensorStatistics],
        tensors_map: Mapping[int, torch.Tensor],
        block: AllocationBlock,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """Add tensors from strategy to an allocation block.

        Collects tensors referenced by the strategy, deduplicates by tensor ID,
        adds them to the allocation block, and allocates the block memory.

        Args:
            label: Layer label identifying this tensor group.
            strategy: List of TensorStatistics describing tensors to add.
            tensors_map: Mapping from tensor IDs to actual tensors.
            block: AllocationBlock to add tensors to and allocate.

        Returns:
            A tuple containing:
                - List of tensors that were added (for subsequent memory release).
                - List of corresponding tensor IDs in the same order.

        Example:
            >>> block = allocation_manager.block()
            >>> tensors, ids = self._add_tensors("layer.0", strategy, tensors_map, block)
            >>> self.release_memory(tensors)
        """
        unique_tensor_ids: set[int] = set()
        tensors_list: list[torch.Tensor] = []
        tensor_ids_list: list[int] = []
        for tensor_info in strategy:
            tensor_id = tensor_info.tensor_id
            # TODO: after consolidating stats, remove unique_tensor_ids
            if tensor_id in unique_tensor_ids:
                continue
            unique_tensor_ids.add(tensor_id)

            tensor = tensors_map[tensor_id]
            tensors_list.append(tensor)
            tensor_ids_list.append(tensor_id)
            block.add(tensor)
        _views = block.allocate()
        self.label_to_cpu_tensor_id_map[label] = tensor_ids_list
        # TODO: FIX release memory!
        return tensors_list, tensor_ids_list

    def _format_device_str(self, device_gpu: torch.device) -> str:
        """Format a torch device to its string representation.

        Args:
            device_gpu: The torch.device to format (e.g., cuda:0, cuda, cpu).

        Returns:
            String representation of the device, including index if present
            (e.g., "cuda:0" or "cuda").
        """
        device_str = str(device_gpu.type)
        if device_gpu.index is not None:
            device_str += ":" + str(device_gpu.index)
        return device_str

    def release_memory(self, tensors_list):
        for tensor in tensors_list:
            del tensor

    def prepare_tensor_id_to_view_mapping(self):
        self.tensor_id_to_view_map = {}
        for label, views in self.label_to_tensor_views_map.items():
            tensor_ids = self.label_to_cpu_tensor_id_map[label]
            for tensor_id, view in zip(tensor_ids, views, strict=False):
                self.tensor_id_to_view_map[tensor_id] = view

    def get_tensor_id_to_view_mapping(self):
        return self.tensor_id_to_view_map

    def schedule_transfer(self, label, non_blocking=True):
        cpu_block = self.block_map_cpu[label]
        gpu_block_view = self.gpu_block_view_map[label]
        cpu_block.copy_to(gpu_block_view, non_blocking=non_blocking)

    def shutdown(self):
        for block in self.block_map_cpu.values():
            block.release()

    def get_gpu_memory_bytes(self) -> int:
        """Get total GPU memory used by transfer blocks.

        Returns:
            Total GPU memory in bytes allocated for transfer blocks.
        """
        total_bytes = 0
        for gpu_block_view in self.block_map_gpu.values():
            # Use untyped_storage().nbytes() to get the full underlying allocation size,
            # not just the view size (views share storage with the full GPU block)
            total_bytes += gpu_block_view.untyped_storage().nbytes()
        return total_bytes


class PreallocatedLoader(Loader):
    def __init__(self, allocation_controller: AllocationBlockController | RawBlockController):
        self.allocation_controller = allocation_controller

    @abstractmethod
    def preload(self) -> None:
        """
        Method to pre-load tensors or data blocks as per loader strategy.
        """

    @abstractmethod
    def prepare(self) -> None:
        """
        Prepare the loader for transfers or computation.
        """

    def shutdown(self) -> None:
        """
        Release resources and clean up allocations.
        """
        self.allocation_controller.shutdown()

    def release_memory(self) -> None:
        """
        Release memory.
        """
        return None

    def get_gpu_memory_bytes(self) -> int:
        """Get total GPU memory used by transfer blocks.

        Delegates to the block controller to compute GPU memory usage.

        Returns:
            Total GPU memory in bytes allocated for transfer blocks.
        """
        return self.allocation_controller.get_gpu_memory_bytes()


@instrumentable
class PreallocatedBatchTransferTensorLoader(PreallocatedLoader):
    def __init__(
        self,
        layer_stats: list[LayerStatistics],
        device_gpu: torch.device,
        label_to_block_id: dict[str, int],
        transfer_to_compute_map: dict[str, str],
        stream_priority: int,
        allocation_controller: AllocationBlockController | RawBlockController,
    ) -> None:
        super().__init__(allocation_controller)
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map = {}

        self.label_to_block_id = label_to_block_id
        self.transfer_to_compute_map = transfer_to_compute_map

        self.transfer_stream = torch.cuda.Stream(device=self.device_gpu, priority=stream_priority)
        self.scheduled_transfers = {}

        self.compute_events = {}  # block id -> event, add in exit
        self.block_events = {}  # block id -> event , add in enter

        self.compute_events_map = {}
        self.last_block_id_to_label_map = {}

        # Track iteration boundary for cross-iteration synchronization
        self.last_iteration_event: torch.cuda.Event | None = None
        self.first_layer_label: str | None = None
        self.last_layer_label: str | None = None

        if layer_stats:
            self.first_layer_label = layer_stats[0].label
            self.last_layer_label = layer_stats[-1].label

        self.preload_transfers = find_transfers_for_preload(self.transfer_to_compute_map, layer_stats)
        self.prepare()

    def preload(self) -> None:
        with torch.cuda.stream(self.transfer_stream):
            for label, _compute_label in self.preload_transfers.items():
                self.allocation_controller.schedule_transfer(label)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # Track which block each preloaded label occupies so enter() can wait
        # for compute to finish before overwriting the block with new data.
        for label, _compute_label in self.preload_transfers.items():
            block_id = self.label_to_block_id[label]
            self.last_block_id_to_label_map[block_id] = label

    def prepare(self) -> None:
        self.preload()

    def enter(self, label):
        if label == self.first_layer_label:
            # Safety net: a previous iteration that aborted before reaching
            # last_layer's exit() leaves stale entries in these maps; the
            # next iteration's lookups would then wait on events recorded
            # in the aborted iteration and produce cross-capture violations
            # under torch.compile / CUDA graph capture. Clear-on-entry costs
            # one dict.clear() per forward and unblocks recovery.
            if self.compute_events_map or self.last_block_id_to_label_map:
                LOGGER.debug(
                    "PreallocatedBatchTransferTensorLoader: clearing %d stale event-map "
                    "entries from aborted previous iteration",
                    len(self.compute_events_map),
                )
                self.compute_events_map.clear()
                self.last_block_id_to_label_map.clear()

            # Fork: bring transfer stream into CUDA graph capture scope.
            # Once forked, the transfer stream stays captured for the entire graph.
            fork_event = torch.cuda.current_stream().record_event()
            self.transfer_stream.wait_event(fork_event)

        if label not in self.label_to_block_id:
            return

        block_id = self.label_to_block_id[label]
        if block_id in self.last_block_id_to_label_map:
            last_label = self.last_block_id_to_label_map[block_id]
            compute_label = self.transfer_to_compute_map[last_label]
            if compute_label in self.compute_events_map:
                event_compute = self.compute_events_map[compute_label]
                self.transfer_stream.wait_event(event_compute)

        self.last_block_id_to_label_map[block_id] = label

        with torch.cuda.stream(self.transfer_stream):
            self.allocation_controller.schedule_transfer(label)

        event = self.transfer_stream.record_event()
        self.scheduled_transfers[label] = event

    def exit(self, label):
        if label in self.scheduled_transfers:
            event = self.scheduled_transfers[label]
            torch.cuda.current_stream().wait_event(event)

        event_compute = torch.cuda.current_stream().record_event()
        self.compute_events_map[label] = event_compute

        # Cross-iteration sync: record event on transfer stream after last layer.
        # Also serves as the final join to bring the transfer stream back to the
        # compute stream, required for CUDA graph capture completeness.
        if label == self.last_layer_label:
            self.last_iteration_event = self.transfer_stream.record_event()
            torch.cuda.current_stream().wait_event(self.last_iteration_event)

            # Clear stale cross-iteration events so the next iteration starts
            # fresh.  Required for CUDA graph capture: without it, the next
            # iteration's enter() would wait on events recorded *before*
            # capture, violating stream-capture isolation rules
            # (cudaErrorStreamCaptureIsolation). Paired with the safety-net
            # clear at first-layer enter() for aborted-mid-iteration recovery.
            self.compute_events_map.clear()
            self.last_block_id_to_label_map.clear()

    def release_memory(self):
        pass


@instrumentable
class PreallocatedBatchTransferTensorLoaderReordered(PreallocatedLoader):
    def __init__(
        self,
        layer_stats: list[LayerStatistics],
        device_gpu: torch.device,
        label_to_block_id: dict[str, int],
        transfer_to_compute_map: dict[str, str],
        stream_priority: int,
        allocation_controller: AllocationBlockController | RawBlockController,
    ) -> None:
        super().__init__(allocation_controller)
        self.device_gpu = device_gpu
        self.cpu_to_gpu_map = {}

        self.label_to_block_id = label_to_block_id
        self.transfer_to_compute_map = transfer_to_compute_map

        self.transfer_stream = torch.cuda.Stream(device=self.device_gpu, priority=stream_priority)
        self.scheduled_transfers = {}

        self.compute_events = {}  # block id -> event, add in exit
        self.block_events = {}  # block id -> event , add in enter

        self.compute_events_map = {}
        self.last_block_id_to_label_map = {}

        # Track iteration boundary for CUDA graph capture
        self.first_layer_label: str | None = None
        self.last_layer_label: str | None = None

        if layer_stats:
            self.first_layer_label = layer_stats[0].label
            self.last_layer_label = layer_stats[-1].label

        self.preload_transfers = find_transfers_for_preload(self.transfer_to_compute_map, layer_stats)
        self.prepare()

    def preload(self) -> None:
        with torch.cuda.stream(self.transfer_stream):
            for label, _compute_label in self.preload_transfers.items():
                self.allocation_controller.schedule_transfer(label)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        for label, compute_label in self.preload_transfers.items():
            block_id = self.label_to_block_id[label]
            self.last_block_id_to_label_map[block_id] = label
            event = self.transfer_stream.record_event()
            self.scheduled_transfers[compute_label] = event

    def prepare(self) -> None:
        self.preload()

    def enter(self, label):
        if label == self.first_layer_label:
            # Safety net: see PreallocatedBatchTransferTensorLoader.enter for
            # the aborted-iteration rationale.
            if self.compute_events_map or self.last_block_id_to_label_map:
                LOGGER.debug(
                    "PreallocatedBatchTransferTensorLoaderReordered: clearing %d stale "
                    "event-map entries from aborted previous iteration",
                    len(self.compute_events_map),
                )
                self.compute_events_map.clear()
                self.last_block_id_to_label_map.clear()

            # Fork: bring transfer stream into CUDA graph capture scope.
            # Once forked, the transfer stream stays captured for the entire graph.
            fork_event = torch.cuda.current_stream().record_event()
            self.transfer_stream.wait_event(fork_event)

        # Wait for the transfer that provides this layer's data before compute reads it.
        if label in self.scheduled_transfers:
            event = self.scheduled_transfers[label]
            torch.cuda.current_stream().wait_event(event)

        if label not in self.label_to_block_id:
            return

        block_id = self.label_to_block_id[label]
        if block_id in self.last_block_id_to_label_map:
            last_label = self.last_block_id_to_label_map[block_id]
            compute_label = self.transfer_to_compute_map[last_label]
            if compute_label in self.compute_events_map:
                event_compute = self.compute_events_map[compute_label]
                self.transfer_stream.wait_event(event_compute)

        self.last_block_id_to_label_map[block_id] = label

        with torch.cuda.stream(self.transfer_stream):
            self.allocation_controller.schedule_transfer(label)
            event = self.transfer_stream.record_event()

        compute_label = self.transfer_to_compute_map[label]
        self.scheduled_transfers[compute_label] = event

    def exit(self, label):
        event_compute = torch.cuda.current_stream().record_event()
        self.compute_events_map[label] = event_compute

        # Join: bring transfer stream back to compute stream after last layer.
        # Required for CUDA graph capture completeness.
        if label == self.last_layer_label:
            last_event = self.transfer_stream.record_event()
            torch.cuda.current_stream().wait_event(last_event)

            # Clear stale cross-iteration events (see comment in
            # PreallocatedBatchTransferTensorLoader.exit for rationale).
            self.compute_events_map.clear()
            self.last_block_id_to_label_map.clear()

    def release_memory(self):
        pass
