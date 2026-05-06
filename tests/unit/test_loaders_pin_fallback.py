# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for pin-failure propagation at non-AllocationBlock pin sites.

Pinning is strict: a ``cudaHostRegister`` or ``tensor.pin_memory()`` failure
during block allocation must abort warmup with a ``RuntimeError``, not
silently degrade to pageable transfers and leave the operator with a
30-60% perf cliff to triage from logs. These tests pin that contract for
``RawBlockController`` and the related shutdown invariants.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from flextensor import host_pinning
from flextensor.allocation_block import AllocationBlock
from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import HostPinner, HostPinRegistry
from flextensor.loaders import AllocationBlockController, RawBlockController


def _failing_pinner() -> HostPinner:
    """A host_register-mode pinner whose registry always raises on pin_in_place."""
    registry = MagicMock(spec=HostPinRegistry)
    registry.pin_in_place.side_effect = RuntimeError("CUDA runtime call cudaHostRegister failed with error code 2")
    return HostPinner(registry)


def _tensor_stats(tensor_id: int, size_bytes: int) -> TensorStatistics:
    return TensorStatistics(tensor_id=tensor_id, name=f"t{tensor_id}", size_bytes=size_bytes, load_time_ms=0.0)


def test_raw_block_controller_init_propagates_when_pin_fails():
    """A cudaHostRegister failure during block allocation must propagate as
    a RuntimeError — silent fallback would mask RLIMIT_MEMLOCK / pinned-pool
    misconfigurations as a perf regression that can only be diagnosed from
    transfer-time logs."""
    label = "layer.0.weight"
    block_id = 0
    nbytes = 32

    src_tensor = torch.zeros(nbytes, dtype=torch.uint8)
    tensors_map = {1: src_tensor}
    strategy_map = {label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]}
    label_to_size_map = {label: nbytes}
    block_sizes = {block_id: nbytes}
    label_to_block_id = {label: block_id}

    pinner = _failing_pinner()

    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        RawBlockController(
            label_to_size_map=label_to_size_map,
            block_sizes=block_sizes,
            device_gpu=torch.device("cpu"),
            tensors_map=tensors_map,
            strategy_map=strategy_map,
            label_to_block_id=label_to_block_id,
            host_pinner=pinner,
        )


class _RecordingCudart:
    """Fake cudart that succeeds and records every register/unregister call."""

    def __init__(self) -> None:
        self.register_calls: list[int] = []
        self.unregister_calls: list[int] = []

    def cudaHostRegister(self, ptr, _size, _flags):  # noqa: N802
        self.register_calls.append(int(ptr))

        class _Ok:
            value = 0

        return _Ok()

    def cudaHostUnregister(self, ptr):  # noqa: N802
        self.unregister_calls.append(int(ptr))

        class _Ok:
            value = 0

        return _Ok()


def test_raw_block_controller_shutdown_does_not_unregister_pins(monkeypatch):
    """``RawBlockController.shutdown()`` is intentionally a no-op for pins —
    the :class:`HostPinRegistry` lifecycle is owned by the
    :class:`~flextensor.tensor_manager.TensorManager` and its
    :class:`HostPinner`. The only legitimate unregister path is
    :meth:`HostPinner.release_all`, driven by ``TensorManager.shutdown``;
    controllers must not unregister.

    A future "be tidy at shutdown" refactor that adds an unregister call to
    the controller would either (a) free pinned storage while pages are
    still ``cudaHostRegister``-locked at the kernel level (UAF), or (b)
    cause a double-unregister when ``TensorManager.shutdown`` also fires.
    Both are silent in CI today; this test pins the contract.
    """
    cudart = _RecordingCudart()
    monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: cudart)

    label = "layer.0.weight"
    block_id = 0
    nbytes = 32

    src_tensor = torch.zeros(nbytes, dtype=torch.uint8)
    pinner = HostPinner(HostPinRegistry())

    controller = RawBlockController(
        label_to_size_map={label: nbytes},
        block_sizes={block_id: nbytes},
        device_gpu=torch.device("cpu"),
        tensors_map={1: src_tensor},
        strategy_map={label: [_tensor_stats(tensor_id=1, size_bytes=nbytes)]},
        label_to_block_id={label: block_id},
        host_pinner=pinner,
    )

    # Construction must have pinned the per-label block.
    assert len(cudart.register_calls) == 1, "test precondition: controller construction must have registered the block"
    assert len(pinner.registry) == 1

    controller.shutdown()

    # Critical: shutdown must NOT have invoked cudaHostUnregister and
    # must NOT have removed the entry from the registry.
    assert cudart.unregister_calls == [], (
        "RawBlockController.shutdown() invoked cudaHostUnregister — the controller "
        "must defer all unregister calls to TensorManager.shutdown() to avoid "
        "kernel-level UAF when pages are still page-locked."
    )
    assert len(pinner.registry) == 1, (
        "RawBlockController.shutdown() removed an entry from the host_pin_registry — "
        "this would free CPU storage while the kernel still holds the page locked."
    )

    # The legitimate path (TM.shutdown → host_pinner.release_all) still works.
    pinner.release_all()
    assert len(pinner.registry) == 0
    assert len(cudart.unregister_calls) == 1


def test_allocation_block_controller_shutdown_does_not_unregister_pins(monkeypatch):
    """``AllocationBlockController.shutdown()`` walks each :class:`AllocationBlock`
    and calls ``block.release()`` — but neither the controller nor the block
    is allowed to invoke ``cudaHostUnregister``. The contract is identical
    to :func:`test_raw_block_controller_shutdown_does_not_unregister_pins`,
    just one composition layer up.

    Validates the chain:

    - ``AllocationBlockController.shutdown()`` →
    - ``AllocationBlock.release()`` (per-block loop) →
    - must NOT call ``cudaHostUnregister``.

    A future refactor that adds an unregister call at either layer would
    free CPU storage while pages are still kernel-locked.
    """
    cudart = _RecordingCudart()
    monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: cudart)

    pinner = HostPinner(HostPinRegistry())

    # Bypass the (heavy) AllocationBlockController constructor: build it
    # empty, then attach pre-pinned AllocationBlocks directly. The shutdown
    # contract only depends on what's in `block_map_cpu`.
    controller = AllocationBlockController(
        allocation_ordered={},
        device_gpu=torch.device("cpu"),
        tensors_map={},
        strategy_map={},
        label_to_block_id={},
        host_pinner=pinner,
    )
    for label, nbytes in (("layer.0", 64), ("layer.1", 96), ("layer.2", 128)):
        block = AllocationBlock(
            device="cpu",
            host_pinner=pinner,
            pinned_memory=True,
            block_size=nbytes,
        )
        controller.block_map_cpu[label] = block

    assert len(cudart.register_calls) == 3, (
        "test precondition: each block construction must have registered its base buffer"
    )
    assert len(pinner.registry) == 3

    controller.shutdown()

    assert cudart.unregister_calls == [], (
        "AllocationBlockController.shutdown() invoked cudaHostUnregister via the "
        "block-release loop — controllers must defer all unregister calls to "
        "TensorManager.shutdown() to avoid kernel-level UAF."
    )
    assert len(pinner.registry) == 3, (
        "AllocationBlockController.shutdown() removed entries from the "
        "host_pin_registry — this would free CPU storage while the kernel "
        "still holds the pages locked."
    )

    pinner.release_all()
    assert len(pinner.registry) == 0
    assert len(cudart.unregister_calls) == 3
