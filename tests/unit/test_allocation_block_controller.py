# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import NoOpHostPinner
from flextensor.loaders import AllocationBlockController


def _tensor_stats(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=f"t{tensor_id}", size_bytes=size_bytes, load_time_ms=0.0)


def test_allocation_block_controller_has_no_post_copy_release_hook() -> None:
    assert not hasattr(AllocationBlockController, "release_memory")


def test_allocation_block_controller_releases_tensor_storage_during_block_allocation() -> None:
    label = "layer.0.weight"
    tensor = torch.arange(6, dtype=torch.float32)
    expected = tensor.clone()

    class _Controller(AllocationBlockController):
        def release_memory(self, tensors_list: list[torch.Tensor]) -> None:
            raise AssertionError("AllocationBlockController must not use a post-copy release hook")

    controller = _Controller(
        allocation_ordered={0: [label]},
        device_gpu=torch.device("cpu"),
        tensors_map={1: tensor},
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=tensor.numel() * tensor.element_size())]},
        label_to_block_id={label: 0},
        host_pinner=NoOpHostPinner(),
    )

    assert tensor.numel() == 0
    assert tensor.untyped_storage().nbytes() == 0
    controller.schedule_transfer(label, non_blocking=False)
    assert torch.equal(controller.get_tensor_id_to_view_mapping()[1], expected)

    controller.shutdown()
