# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AllocationBlockController`` unified-memory NVMe eviction path.

Exercises the ``_construct_unified_memory`` code path that writes GPU
tensors directly to NVMe via ``os.pwrite`` with a ctypes buffer view,
freeing each tensor after the write and allocating GPU "hot" blocks
from the freed memory. On non-GPU systems, CPU tensors simulate
unified memory (where CPU and GPU share the same physical DRAM pool).

Also tests the ``_view_from_gpu_block`` static method for correct
typed-view creation from a uint8 block.
"""

from pathlib import Path

import pytest
import torch

from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import NoOpHostPinner
from flextensor.loaders import AllocationBlockController
from flextensor.nvme_transfer import PosixBackend


def _tensor_stats(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=f"t{tensor_id}", size_bytes=size_bytes, load_time_ms=0.0)


def _build_unified_controller(
    tmp_path: Path,
    *,
    allocation_ordered: dict[int, list[str]],
    tensors_map: dict[int, torch.Tensor],
    strategy_map: dict[str, list[TensorStatistics]],
    label_to_block_id: dict[str, int],
    release_tensor_memory: bool = True,
    alignment: int = 512,
) -> AllocationBlockController:
    """Build an AllocationBlockController with the unified-memory path.

    Uses CPU device to simulate unified memory (CPU == GPU pool).
    """
    nvme_dir = tmp_path / "nvme_blocks"
    nvme_dir.mkdir(parents=True, exist_ok=True)
    backend = PosixBackend(alignment=alignment, use_odirect=False)
    controller = AllocationBlockController(
        allocation_ordered=allocation_ordered,
        device_gpu=torch.device("cpu"),
        tensors_map=tensors_map,
        strategy_map=strategy_map,
        label_to_block_id=label_to_block_id,
        host_pinner=NoOpHostPinner(),
        release_tensor_memory=release_tensor_memory,
        nvme_backend=backend,
        nvme_offload_path=str(nvme_dir),
        unified_memory=True,
    )
    return controller


class TestUnifiedMemoryConstruction:
    """Construction-time behaviour of the unified-memory eviction path."""

    def test_writes_to_nvme_and_frees_source_tensor(self, tmp_path: Path) -> None:
        """The unified-memory path must write tensors to NVMe and free the source.

        After construction, the source tensor in ``tensors_map`` must be
        emptied (``release_tensor_memory=True`` default), and an NVMe block
        file must exist on disk.
        """
        label = "layer.0.weight"
        nbytes = 64
        tensor = torch.arange(nbytes, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
            label_to_block_id={label: 0},
        )

        # Source tensor must be freed.
        assert tensor.numel() == 0

        # NVMe block file must exist and be non-empty.
        nvme_files = list((tmp_path / "nvme_blocks").glob("blocks_*.bin"))
        assert len(nvme_files) == 1
        assert nvme_files[0].stat().st_size > 0

        # NVMe block map must be populated. logical_nbytes is the packed
        # (alignment-padded) total, not the raw tensor byte count.
        assert label in controller.nvme_block_map
        assert controller.nvme_block_map[label].logical_nbytes >= nbytes

        controller.shutdown()

    def test_preserves_source_tensor_when_release_disabled(self, tmp_path: Path) -> None:
        """``release_tensor_memory=False`` must keep source tensors intact.

        The profiling/re-plan path needs source weights to survive so a
        corrected destructive controller can be rebuilt.
        """
        label = "layer.0.weight"
        nbytes = 32
        tensor = torch.ones(nbytes, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
            label_to_block_id={label: 0},
            release_tensor_memory=False,
        )

        assert tensor.numel() == nbytes
        assert torch.equal(tensor, torch.ones(nbytes, dtype=torch.uint8))

        controller.shutdown()

    def test_nvme_file_fd_is_open(self, tmp_path: Path) -> None:
        """The NVMe file descriptor must be set after construction."""
        label = "layer.0.weight"
        nbytes = 16
        tensor = torch.zeros(nbytes, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
            label_to_block_id={label: 0},
        )

        assert controller.nvme_file_fd is not None

        controller.shutdown()

        # shutdown must close the fd.
        assert controller.nvme_file_fd is None

    def test_gpu_block_views_populated(self, tmp_path: Path) -> None:
        """GPU block view maps must be populated after construction."""
        label = "layer.0.weight"
        nbytes = 32
        tensor = torch.arange(nbytes, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
            label_to_block_id={label: 0},
        )

        assert label in controller.gpu_block_view_map
        assert label in controller.label_to_gpu_block
        assert 0 in controller.block_map_gpu
        assert label in controller.label_to_tensor_views_map
        assert 1 in controller.get_tensor_id_to_view_mapping()

        controller.shutdown()


class TestUnifiedMemoryRoundtrip:
    """End-to-end: write to NVMe → schedule_transfer → read back → verify."""

    def test_single_tensor_roundtrip(self, tmp_path: Path) -> None:
        """A single tensor written via the unified path must read back correctly."""
        label = "layer.0.weight"
        data = torch.arange(100, dtype=torch.float32)
        expected = data.clone()
        nbytes = data.numel() * data.element_size()
        tensors_map = {1: data}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
            label_to_block_id={label: 0},
        )

        # schedule_transfer reads from NVMe into the GPU block view.
        controller.schedule_transfer(label, non_blocking=False)

        view = controller.get_tensor_id_to_view_mapping()[1]
        assert view.dtype == torch.float32
        assert view.shape == torch.Size([100])
        torch.testing.assert_close(view, expected)

        controller.shutdown()

    def test_multiple_labels_in_one_group_roundtrip(self, tmp_path: Path) -> None:
        """Multiple labels in one allocation group share a GPU block and roundtrip individually.

        Labels in the same allocation group share one GPU block — only one
        label's data is valid at a time after ``schedule_transfer``. This test
        verifies each label can be independently scheduled and read back.
        """
        label_a = "layer.0.weight"
        label_b = "layer.0.bias"
        data_a = torch.arange(50, dtype=torch.float32)
        data_b = torch.arange(30, dtype=torch.int64) * 3
        expected_a = data_a.clone()
        expected_b = data_b.clone()
        tensors_map = {1: data_a, 2: data_b}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label_a, label_b]},
            tensors_map=tensors_map,
            strategy_map={
                label_a: [_tensor_stats(tensor_id=1, size_bytes=data_a.numel() * data_a.element_size())],
                label_b: [_tensor_stats(tensor_id=2, size_bytes=data_b.numel() * data_b.element_size())],
            },
            label_to_block_id={label_a: 0, label_b: 0},
        )

        # Both labels share the same GPU block (same allocation group).
        assert controller.label_to_gpu_block[label_a] is controller.label_to_gpu_block[label_b]

        # Schedule and verify label_a.
        controller.schedule_transfer(label_a, non_blocking=False)
        views = controller.get_tensor_id_to_view_mapping()
        torch.testing.assert_close(views[1], expected_a)

        # Schedule and verify label_b (overwrites label_a's data in the shared block).
        controller.schedule_transfer(label_b, non_blocking=False)
        views = controller.get_tensor_id_to_view_mapping()
        torch.testing.assert_close(views[2], expected_b)

        controller.shutdown()

    def test_multiple_allocation_groups_roundtrip(self, tmp_path: Path) -> None:
        """Multiple allocation groups (blocks) must each have independent GPU blocks."""
        label_a = "layer.0.weight"
        label_b = "layer.1.weight"
        data_a = torch.arange(40, dtype=torch.float32)
        data_b = torch.arange(60, dtype=torch.float32) * 2.0
        expected_a = data_a.clone()
        expected_b = data_b.clone()
        tensors_map = {1: data_a, 2: data_b}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label_a], 1: [label_b]},
            tensors_map=tensors_map,
            strategy_map={
                label_a: [_tensor_stats(tensor_id=1, size_bytes=data_a.numel() * data_a.element_size())],
                label_b: [_tensor_stats(tensor_id=2, size_bytes=data_b.numel() * data_b.element_size())],
            },
            label_to_block_id={label_a: 0, label_b: 1},
        )

        # Each label should have its own GPU block.
        assert controller.label_to_gpu_block[label_a] is not controller.label_to_gpu_block[label_b]

        controller.schedule_transfer(label_a, non_blocking=False)
        controller.schedule_transfer(label_b, non_blocking=False)

        views = controller.get_tensor_id_to_view_mapping()
        torch.testing.assert_close(views[1], expected_a)
        torch.testing.assert_close(views[2], expected_b)

        controller.shutdown()

    def test_multiple_tensors_per_label_roundtrip(self, tmp_path: Path) -> None:
        """Multiple tensors packed into one label must roundtrip correctly."""
        label = "layer.0"
        t1 = torch.arange(25, dtype=torch.float32)
        t2 = torch.arange(20, dtype=torch.float64)
        expected_t1 = t1.clone()
        expected_t2 = t2.clone()
        tensors_map = {1: t1, 2: t2}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={
                label: [
                    _tensor_stats(tensor_id=1, size_bytes=t1.numel() * t1.element_size()),
                    _tensor_stats(tensor_id=2, size_bytes=t2.numel() * t2.element_size()),
                ],
            },
            label_to_block_id={label: 0},
        )

        controller.schedule_transfer(label, non_blocking=False)

        views = controller.get_tensor_id_to_view_mapping()
        torch.testing.assert_close(views[1], expected_t1)
        torch.testing.assert_close(views[2], expected_t2)

        controller.shutdown()


class TestUnifiedMemoryEdgeCases:
    """Edge cases for the unified-memory eviction path."""

    def test_empty_strategy_label_skipped(self, tmp_path: Path) -> None:
        """A label with an empty strategy list must be handled without error."""
        label = "layer.0.weight"
        empty_label = "layer.0.empty"
        tensor = torch.arange(16, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label, empty_label]},
            tensors_map=tensors_map,
            strategy_map={
                label: [_tensor_stats(tensor_id=1, size_bytes=16)],
                empty_label: [],
            },
            label_to_block_id={label: 0, empty_label: 0},
        )

        # Empty label should have an empty view list and no NVMe block.
        assert controller.label_to_tensor_views_map[empty_label] == []
        assert empty_label not in controller.nvme_block_map

        # Non-empty label should work.
        assert label in controller.nvme_block_map
        controller.schedule_transfer(label, non_blocking=False)
        torch.testing.assert_close(
            controller.get_tensor_id_to_view_mapping()[1],
            torch.arange(16, dtype=torch.uint8),
        )

        controller.shutdown()

    def test_nvme_file_has_uuid_suffix(self, tmp_path: Path) -> None:
        """The NVMe block file must be named with a UUID suffix (collision-safe)."""
        label = "layer.0.weight"
        tensor = torch.arange(8, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=8)]},
            label_to_block_id={label: 0},
        )

        nvme_files = list((tmp_path / "nvme_blocks").glob("blocks_*.bin"))
        assert len(nvme_files) == 1
        # UUID hex is 32 chars.
        suffix = nvme_files[0].stem.split("blocks_", 1)[1]
        assert len(suffix) == 32
        assert all(c in "0123456789abcdef" for c in suffix)

        controller.shutdown()

    def test_no_cpu_block_allocations(self, tmp_path: Path) -> None:
        """The unified-memory path must not allocate any CPU blocks.

        ``block_map_cpu`` must remain empty — the whole point is to avoid
        ``device="cpu"`` allocations that would double peak memory.
        """
        label = "layer.0.weight"
        tensor = torch.arange(16, dtype=torch.uint8)
        tensors_map = {1: tensor}

        controller = _build_unified_controller(
            tmp_path,
            allocation_ordered={0: [label]},
            tensors_map=tensors_map,
            strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=16)]},
            label_to_block_id={label: 0},
        )

        assert controller.block_map_cpu == {}

        controller.shutdown()


class TestViewFromGpuBlock:
    """Tests for the ``_view_from_gpu_block`` static method.

    The method creates a typed view from a uint8 block's raw bytes. Tests
    must populate the block with the byte representation of the expected
    typed data — ``_view_from_gpu_block`` reinterprets raw bytes, it does
    not convert values.
    """

    def test_creates_typed_view_at_zero_offset(self) -> None:
        """A view at offset 0 must match the original typed data."""
        data = torch.arange(4, dtype=torch.float32)
        block = data.view(torch.uint8).reshape(-1)
        view = AllocationBlockController._view_from_gpu_block(
            block, torch.float32, torch.Size([4]), 0, None,
        )
        assert view.dtype == torch.float32
        assert view.shape == torch.Size([4])
        torch.testing.assert_close(view, data)

    def test_creates_view_at_nonzero_offset(self) -> None:
        """A view at a non-zero offset must read from the correct position."""
        data = torch.arange(8, dtype=torch.float32)
        block = data.view(torch.uint8).reshape(-1)
        # float32 = 4 bytes; offset 16 bytes = 4 elements -> data[4:8].
        view = AllocationBlockController._view_from_gpu_block(
            block, torch.float32, torch.Size([4]), 16, None,
        )
        torch.testing.assert_close(view, data[4:8])

    def test_preserves_stride(self) -> None:
        """A custom stride must be applied to the view."""
        data = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        block = data.view(torch.uint8).reshape(-1)
        view = AllocationBlockController._view_from_gpu_block(
            block, torch.float32, torch.Size([2, 2]), 0, (2, 1),
        )
        assert view.shape == torch.Size([2, 2])
        torch.testing.assert_close(view, data)

    def test_rejects_misaligned_offset(self) -> None:
        """An offset not divisible by the dtype element size must raise."""
        block = torch.zeros(16, dtype=torch.uint8)
        with pytest.raises(ValueError, match="must be a multiple of dtype element size"):
            AllocationBlockController._view_from_gpu_block(
                block, torch.float32, torch.Size([1]), 1, None,
            )

    def test_works_with_int64(self) -> None:
        """int64 (8-byte elements) views must work at 8-byte-aligned offsets."""
        data = torch.arange(4, dtype=torch.int64)
        block = data.view(torch.uint8).reshape(-1)
        # offset 8 bytes = 1 int64 element -> data[1:3].
        view = AllocationBlockController._view_from_gpu_block(
            block, torch.int64, torch.Size([2]), 8, None,
        )
        torch.testing.assert_close(view, data[1:3])
