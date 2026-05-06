# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TensorManager.get_memory_transfer_stats accessor."""

import pytest
import torch

from flextensor.strategy import KnapsackStrategy
from flextensor.tensor_manager import TensorManager


@pytest.fixture(autouse=True)
def _fake_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend CUDA is available so ``TensorManager(pinned_memory=True)``
    construction doesn't raise on CPU-only CI hosts. These tests only
    exercise the stats accessor; they don't need real CUDA.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


class TestGetMemoryTransferStats:
    """Tests for TensorManager.get_memory_transfer_stats()."""

    @pytest.fixture
    def tensor_manager(self):
        """Create a TensorManager with minimal config (no GPU needed)."""
        strategy = KnapsackStrategy()
        return TensorManager(
            device_gpu="cuda:0",
            tensor_manager_load_strategy=strategy,
        )

    def test_returns_none_before_profiling(self, tensor_manager):
        """Accessor returns None before prepare_infer_mode() has run."""
        assert tensor_manager.get_memory_transfer_stats() is None

    def test_returns_dict_after_assignment(self, tensor_manager):
        """Accessor returns the stored dict after direct assignment."""
        expected = {1024: 0.015, 4096: 0.023}
        tensor_manager.memory_transfer_stats = expected
        assert tensor_manager.get_memory_transfer_stats() == expected
