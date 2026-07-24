# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``RawBlockController`` source-weight release and GPU teardown for compiled re-plan."""

import torch

from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import make_host_pinner
from flextensor.loaders import RawBlockController


def _tensor_stats(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=f"t{tensor_id}", size_bytes=size_bytes, load_time_ms=0.0)


def _build_controller(*, release_tensor_memory: bool) -> tuple[RawBlockController, dict[int, torch.Tensor]]:
    label = "model.layers.0"
    nbytes = 64
    src = torch.arange(nbytes, dtype=torch.uint8)
    tensors_map = {1: src}
    controller = RawBlockController(
        label_to_size_map={label: nbytes},
        block_sizes={0: nbytes},
        device_gpu=torch.device("cpu"),
        tensors_map=tensors_map,
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
        label_to_block_id={label: 0},
        host_pinner=make_host_pinner(False, "torch"),
        release_tensor_memory=release_tensor_memory,
    )
    return controller, tensors_map


def test_release_tensor_memory_true_empties_source_weight() -> None:
    _controller, tensors_map = _build_controller(release_tensor_memory=True)
    assert tensors_map[1].numel() == 0


def test_release_tensor_memory_false_preserves_source_weight() -> None:
    _controller, tensors_map = _build_controller(release_tensor_memory=False)
    assert tensors_map[1].numel() == 64
    assert torch.equal(tensors_map[1], torch.arange(64, dtype=torch.uint8))


def test_release_tensor_memory_defaults_to_true() -> None:
    label = "model.layers.0"
    nbytes = 32
    tensors_map = {1: torch.ones(nbytes, dtype=torch.uint8)}
    RawBlockController(
        label_to_size_map={label: nbytes},
        block_sizes={0: nbytes},
        device_gpu=torch.device("cpu"),
        tensors_map=tensors_map,
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
        label_to_block_id={label: 0},
        host_pinner=make_host_pinner(False, "torch"),
    )
    assert tensors_map[1].numel() == 0


def test_release_gpu_blocks_drops_all_gpu_references() -> None:
    controller, _tensors_map = _build_controller(release_tensor_memory=False)
    assert controller.block_map_gpu
    assert controller.gpu_block_view_map
    assert controller.label_to_tensor_views_map
    assert controller.tensor_id_to_view_map

    controller.release_gpu_blocks()

    assert controller.block_map_gpu == {}
    assert controller.gpu_block_view_map == {}
    assert controller.label_to_tensor_views_map == {}
    assert controller.tensor_id_to_view_map == {}
