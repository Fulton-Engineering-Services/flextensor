# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the structural TensorManager contract used by OffloadManager."""

from unittest.mock import MagicMock, patch

import torch

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.helpers import NoOpTensorManager
from flextensor.offload_manager import TensorManagerProtocol
from flextensor.state_handler import TensorManagerState
from flextensor.tensor_manager import TensorManager


def _make_tensor_manager() -> TensorManager:
    """Create a fresh TensorManager without requiring CUDA hardware."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.Event", return_value=MagicMock()),
    ):
        return TensorManager(
            device_gpu="cpu",
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
        )


def _make_state_for_model(model: torch.nn.Module, loader_type: str) -> TensorManagerState:
    tensor_stats = [
        TensorStatistics(
            tensor_id=index,
            name=name,
            size_bytes=tensor.nelement() * tensor.element_size(),
            load_time_ms=0.0,
        )
        for index, (name, tensor) in enumerate(model.named_parameters())
    ]
    return TensorManagerState(
        loader_type=loader_type,
        tensor_id_to_name_map={stat.tensor_id: stat.name for stat in tensor_stats},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"layer_0": tensor_stats},
        release_strategy={"layer_0": tensor_stats},
        label_to_block_id={},
        stats=[LayerStatistics(label="layer_0", tensors=tensor_stats, duration=1.0)],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


def _stub_restored_phase_setup(manager: TensorManager) -> None:
    manager.prepare_infer_load_mode = MagicMock()
    manager.prepare_final_model = MagicMock(side_effect=lambda model: model)


def test_tensor_manager_satisfies_offload_manager_protocol() -> None:
    """The concrete TensorManager must expose the OffloadManager surface."""
    assert isinstance(_make_tensor_manager(), TensorManagerProtocol)


def test_noop_tensor_manager_satisfies_offload_manager_protocol() -> None:
    """The disabled-offload manager must stay in lockstep with TensorManager."""
    assert isinstance(NoOpTensorManager(device_gpu="cpu"), TensorManagerProtocol)


def test_tensor_manager_profile_restore_paths_associate_model() -> None:
    """Profile restore/load associate the model used by later phase calls."""
    restore_manager = _make_tensor_manager()
    restored_model = torch.nn.Linear(2, 2)
    _stub_restored_phase_setup(restore_manager)

    restore_manager.restore_state(
        restored_model,
        _make_state_for_model(restored_model, loader_type=restore_manager.loader_type),
    )
    assert restore_manager.initialize_warmup() is restored_model
    assert restore_manager.initialize_profile() is restored_model
    assert restore_manager.initialize_inference() is restored_model

    load_manager = _make_tensor_manager()
    loaded_model = torch.nn.Linear(2, 2)
    _stub_restored_phase_setup(load_manager)
    with patch.object(
        load_manager,
        "_load_state_from_file",
        return_value=_make_state_for_model(loaded_model, loader_type=load_manager.loader_type),
    ) as load_state:
        load_manager.load_profile("unused", loaded_model)

    load_state.assert_called_once()
    assert load_manager.initialize_warmup() is loaded_model
    assert load_manager.initialize_profile() is loaded_model
    assert load_manager.initialize_inference() is loaded_model


def test_noop_tensor_manager_profile_restore_paths_associate_model() -> None:
    """Disabled offload preserves the profile-load phase sequence contract."""
    manager = NoOpTensorManager(device_gpu="cpu")
    restored_model = object()
    loaded_model = object()

    manager.restore_state(restored_model, object())
    assert manager.initialize_warmup() is restored_model
    assert manager.initialize_profile() is restored_model
    assert manager.initialize_inference() is restored_model

    manager.load_profile("unused", loaded_model)
    assert manager.initialize_warmup() is loaded_model
    assert manager.initialize_profile() is loaded_model
    assert manager.initialize_inference() is loaded_model
