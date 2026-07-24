# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AllocationBlockController`` source-weight release semantics.

The inference loader frees each source weight in ``tensors_map`` right after
copying it into its pinned CPU block (``release_tensor_memory=True``) to keep
peak host memory low. The compiled-offload strategy re-plan path needs the
opposite for its first, profiling-phase build: the source weights must survive so
a corrected, destructive loader can be rebuilt from them. These tests pin both
behaviours.
"""

import torch

from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import make_host_pinner
from flextensor.loaders import AllocationBlockController


def _tensor_stats(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=f"t{tensor_id}", size_bytes=size_bytes, load_time_ms=0.0)


def _build_controller(*, release_tensor_memory: bool) -> tuple[AllocationBlockController, dict[int, torch.Tensor]]:
    label = "model.layers.0"
    nbytes = 64
    src = torch.arange(nbytes, dtype=torch.uint8)
    tensors_map = {1: src}
    controller = AllocationBlockController(
        allocation_ordered={0: [label]},
        device_gpu=torch.device("cpu"),
        tensors_map=tensors_map,
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
        label_to_block_id={label: 0},
        host_pinner=make_host_pinner(False, "torch"),
        release_tensor_memory=release_tensor_memory,
    )
    return controller, tensors_map


def test_release_tensor_memory_true_empties_source_weight() -> None:
    """Inference default: the source weight is freed after the block copy."""
    _controller, tensors_map = _build_controller(release_tensor_memory=True)
    assert tensors_map[1].numel() == 0


def test_release_tensor_memory_false_preserves_source_weight() -> None:
    """Re-plan path: the source weight survives the (non-destructive) build."""
    _controller, tensors_map = _build_controller(release_tensor_memory=False)
    assert tensors_map[1].numel() == 64
    # Bytes are intact, so a subsequent destructive rebuild reads real weights.
    assert torch.equal(tensors_map[1], torch.arange(64, dtype=torch.uint8))


def test_release_tensor_memory_defaults_to_true() -> None:
    """Default must stay destructive so existing inference behaviour is unchanged."""
    label = "model.layers.0"
    nbytes = 32
    tensors_map = {1: torch.ones(nbytes, dtype=torch.uint8)}
    AllocationBlockController(
        allocation_ordered={0: [label]},
        device_gpu=torch.device("cpu"),
        tensors_map=tensors_map,
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
        label_to_block_id={label: 0},
        host_pinner=make_host_pinner(False, "torch"),
    )
    assert tensors_map[1].numel() == 0


def test_release_gpu_blocks_drops_all_gpu_references() -> None:
    """``release_gpu_blocks`` must clear every map that aliases the GPU blocks.

    The re-plan calls this to return the first loader's GPU segments to the
    caching allocator before the rebuild; any surviving reference would defeat
    the ~1x peak-memory guard.
    """
    controller, _tensors_map = _build_controller(release_tensor_memory=False)
    # Sanity: the build populated the GPU view maps.
    assert controller.label_to_gpu_block
    assert controller.block_map_gpu
    assert controller.gpu_block_view_map
    assert controller.label_to_tensor_views_map
    assert controller.tensor_id_to_view_map

    controller.release_gpu_blocks()

    assert controller.label_to_gpu_block == {}
    assert controller.block_map_gpu == {}
    assert controller.gpu_block_view_map == {}
    assert controller.label_to_tensor_views_map == {}
    assert controller.tensor_id_to_view_map == {}
