# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for strategy-invisible tensors moved permanently to GPU."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.collectors import IterativeLayerStatistics, LayerStatistics, TensorStatistics
from flextensor.gpu_budget import MIN_GPU_BUDGET_BYTES, reserve_strategy_invisible_gpu_budget
from flextensor.strategy.protocol import BlockStrategyData, StrategyResult
from flextensor.tensor_manager import ModelDict, TensorManager
from flextensor.tensor_processors import MoveUnmappedTensorsToGPUProcessor


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _tensor_stats(tensor: torch.Tensor, name: str) -> TensorStatistics:
    return TensorStatistics(
        tensor_id=id(tensor),
        name=name,
        size_bytes=_tensor_bytes(tensor),
        load_time_ms=0.1,
    )


def _empty_block_data() -> BlockStrategyData:
    return BlockStrategyData(
        label_to_size_map={},
        allocation_ordered={},
        block_sizes={},
        label_to_block_id={},
        transfer_to_compute_map={},
    )


class _ModelWithInvisibleTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visible = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.invisible = nn.Parameter(torch.zeros(8, dtype=torch.float32))


def _make_manager(model: nn.Module, *, loader_type: str = "allocation_block_transfer") -> TensorManager:
    strategy = MagicMock()
    block_data = _empty_block_data() if loader_type in {"allocation_block_transfer", "raw_block_transfer"} else None
    strategy.compute.return_value = StrategyResult(strategy_map={"visible": []}, block_data=block_data)

    tm = TensorManager(
        device_gpu=torch.device("cpu"),
        tensor_manager_load_strategy=strategy,
        pinned_memory=False,
        loader_type=loader_type,
        max_gpu_mem_fraction=0.9,
    )
    tm.model = model
    tm.tensors_map = {
        id(model.visible): model.visible,
        id(model.invisible): model.invisible,
    }
    tm.layer_statistics_collector = MagicMock()
    tm.layer_statistics_collector.get_layer_stats.return_value = [
        IterativeLayerStatistics(label="visible", tensor_ids={id(model.visible)}, duration=1.0)
    ]
    return tm


def test_block_loader_reserves_reachable_tensors_missing_from_layer_stats() -> None:
    """Block-loader strategy budget must include tensors finalization will move."""

    model = _ModelWithInvisibleTensor()
    tm = _make_manager(model)
    orphan = nn.Parameter(torch.zeros(16, dtype=torch.float32))
    tm.tensors_map[id(orphan)] = orphan
    original_budget = MIN_GPU_BUDGET_BYTES + 4096
    invisible_bytes = _tensor_bytes(model.invisible)

    with (
        patch("flextensor.tensor_manager.report_profiling_quality"),
        patch.object(TensorManager, "_benchmark_tensor_statistics") as mock_benchmark,
        patch.object(TensorManager, "_get_memory_transfer_stats", return_value={}),
        patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=original_budget),
        patch.object(TensorManager, "_create_loader"),
    ):
        mock_benchmark.return_value = {
            id(model.visible): _tensor_stats(model.visible, "visible"),
            id(model.invisible): _tensor_stats(model.invisible, "invisible"),
            id(orphan): _tensor_stats(orphan, "orphan"),
        }

        tm.prepare_infer_mode()

    tm.tensor_manager_load_strategy.compute.assert_called_once()
    assert tm.tensor_manager_load_strategy.compute.call_args.args[2] == original_budget - invisible_bytes


def test_strategy_loader_reserves_reachable_traced_tensors_missing_from_layer_stats() -> None:
    """Strategy loader reserves traced tensors its untimed rescue will force-pin."""

    model = _ModelWithInvisibleTensor()
    tm = _make_manager(model, loader_type="strategy")
    original_budget = MIN_GPU_BUDGET_BYTES + 4096
    invisible_bytes = _tensor_bytes(model.invisible)

    with (
        patch("flextensor.tensor_manager.report_profiling_quality"),
        patch.object(TensorManager, "_benchmark_tensor_statistics") as mock_benchmark,
        patch.object(TensorManager, "_get_memory_transfer_stats", return_value={}),
        patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=original_budget),
        patch.object(TensorManager, "_create_loader"),
    ):
        mock_benchmark.return_value = {
            id(model.visible): _tensor_stats(model.visible, "visible"),
            id(model.invisible): _tensor_stats(model.invisible, "invisible"),
        }

        tm.prepare_infer_mode()

    tm.tensor_manager_load_strategy.compute.assert_called_once()
    assert tm.tensor_manager_load_strategy.compute.call_args.args[2] == original_budget - invisible_bytes


def test_strategy_loader_does_not_reserve_reachable_tensors_outside_rescue_set() -> None:
    """Strategy loader must not reserve model tensors it will leave on CPU."""

    model = _ModelWithInvisibleTensor()
    tm = _make_manager(model, loader_type="strategy")
    del tm.tensors_map[id(model.invisible)]
    original_budget = MIN_GPU_BUDGET_BYTES + 4096

    with (
        patch("flextensor.tensor_manager.report_profiling_quality"),
        patch.object(TensorManager, "_benchmark_tensor_statistics") as mock_benchmark,
        patch.object(TensorManager, "_get_memory_transfer_stats", return_value={}),
        patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=original_budget),
        patch.object(TensorManager, "_create_loader"),
    ):
        mock_benchmark.return_value = {
            id(model.visible): _tensor_stats(model.visible, "visible"),
        }

        tm.prepare_infer_mode()

    tm.tensor_manager_load_strategy.compute.assert_called_once()
    assert tm.tensor_manager_load_strategy.compute.call_args.args[2] == original_budget


def test_budget_reservation_supports_dict_models() -> None:
    """Dict models use the same reachable-tensor budget reservation as modules."""

    visible = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    invisible = nn.Parameter(torch.zeros(8, dtype=torch.float32))
    original_budget = MIN_GPU_BUDGET_BYTES + 4096

    reservation = reserve_strategy_invisible_gpu_budget(
        original_budget,
        model={"visible": visible, "invisible": invisible},
        loader_type="allocation_block_transfer",
        device_gpu=torch.device("cpu"),
        layer_stats=[
            LayerStatistics(label="visible", tensors=[_tensor_stats(visible, "visible")], duration=1.0),
        ],
        tensors_map={id(visible): visible, id(invisible): invisible},
        min_gpu_budget_bytes=MIN_GPU_BUDGET_BYTES,
    )

    assert reservation.effective_budget == original_budget - _tensor_bytes(invisible)
    assert reservation.reserved_bytes == _tensor_bytes(invisible)
    assert reservation.reserved_count == 1


def test_budget_reservation_supports_model_dict_models() -> None:
    """ModelDict wrappers use the same reachable-tensor budget reservation."""

    visible = nn.Parameter(torch.zeros(4, dtype=torch.float32))
    invisible = nn.Parameter(torch.zeros(8, dtype=torch.float32))
    original_budget = MIN_GPU_BUDGET_BYTES + 4096

    reservation = reserve_strategy_invisible_gpu_budget(
        original_budget,
        model=ModelDict(model={"visible": visible, "invisible": invisible}),
        loader_type="allocation_block_transfer",
        device_gpu=torch.device("cpu"),
        layer_stats=[
            LayerStatistics(label="visible", tensors=[_tensor_stats(visible, "visible")], duration=1.0),
        ],
        tensors_map={id(visible): visible, id(invisible): invisible},
        min_gpu_budget_bytes=MIN_GPU_BUDGET_BYTES,
    )

    assert reservation.effective_budget == original_budget - _tensor_bytes(invisible)
    assert reservation.reserved_bytes == _tensor_bytes(invisible)
    assert reservation.reserved_count == 1


def test_prepare_infer_mode_raises_when_invisible_tensors_exhaust_budget() -> None:
    """The strategy should not run with an effective budget too small to be meaningful."""

    model = _ModelWithInvisibleTensor()
    tm = _make_manager(model)
    original_budget = MIN_GPU_BUDGET_BYTES + _tensor_bytes(model.invisible) - 1

    with (
        patch("flextensor.tensor_manager.report_profiling_quality"),
        patch.object(TensorManager, "_benchmark_tensor_statistics") as mock_benchmark,
        patch.object(TensorManager, "_get_memory_transfer_stats", return_value={}),
        patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=original_budget),
        patch.object(TensorManager, "_create_loader"),
    ):
        mock_benchmark.return_value = {
            id(model.visible): _tensor_stats(model.visible, "visible"),
            id(model.invisible): _tensor_stats(model.invisible, "invisible"),
        }

        with pytest.raises(RuntimeError, match="strategy-invisible permanent GPU"):
            tm.prepare_infer_mode()

    tm.tensor_manager_load_strategy.compute.assert_not_called()


def test_cuda_tiny_fraction_reproduction_fails_before_strategy_compute() -> None:
    """Synthetic CUDA repro: real budget resolution, clear failure before strategy compute."""

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA for real max_gpu_mem_fraction resolution")

    model = _ModelWithInvisibleTensor()
    tm = _make_manager(model)
    tm.device_gpu = torch.device("cuda:0")
    _, total_bytes = torch.cuda.mem_get_info(tm.device_gpu)
    tm._max_gpu_mem_fraction = (MIN_GPU_BUDGET_BYTES + _tensor_bytes(model.invisible) - 1) / total_bytes

    with (
        patch("flextensor.tensor_manager.report_profiling_quality"),
        patch.object(TensorManager, "_benchmark_tensor_statistics") as mock_benchmark,
        patch.object(TensorManager, "_get_memory_transfer_stats", return_value={}),
        patch.object(TensorManager, "_create_loader"),
        pytest.raises(RuntimeError, match="strategy-invisible permanent GPU"),
    ):
        mock_benchmark.return_value = {
            id(model.visible): _tensor_stats(model.visible, "visible"),
            id(model.invisible): _tensor_stats(model.invisible, "invisible"),
        }

        tm.prepare_infer_mode()

    tm.tensor_manager_load_strategy.compute.assert_not_called()


def test_unmapped_processor_checks_cuda_free_memory_before_move() -> None:
    """Unmapped CUDA moves fail before calling the raw tensor move when memory is short."""

    tensor = nn.Parameter(torch.zeros(8, dtype=torch.float32))
    tensor_bytes = _tensor_bytes(tensor)
    proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cuda:0"), tensor_id_mapping={})

    with (
        patch.object(torch.cuda, "mem_get_info", return_value=(tensor_bytes - 1, tensor_bytes * 2)),
        patch.object(proc.move_to_gpu, "process", return_value=tensor) as mock_move,
        pytest.raises(RuntimeError, match=r"Insufficient GPU memory.*unmapped tensor"),
    ):
        proc.process(tensor)

    mock_move.assert_not_called()


def test_unmapped_processor_moves_when_cuda_free_memory_is_sufficient() -> None:
    """The CUDA free-memory guard allows the raw tensor move when enough memory remains."""

    tensor = nn.Parameter(torch.zeros(8, dtype=torch.float32))
    tensor_bytes = _tensor_bytes(tensor)
    proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cuda:0"), tensor_id_mapping={})

    with (
        patch.object(torch.cuda, "mem_get_info", return_value=(tensor_bytes + 32 * 1024**2, tensor_bytes * 4)),
        patch.object(proc.move_to_gpu, "process", return_value=tensor) as mock_move,
    ):
        result = proc.process(tensor)

    assert result is tensor
    mock_move.assert_called_once()
    assert mock_move.call_args.args[0] is tensor


def test_unmapped_processor_skips_cuda_free_memory_guard_for_tensor_already_on_target_gpu() -> None:
    """Already-resident tensors do not need extra CUDA allocation budget."""

    class _FakeCudaTensor(torch.Tensor):
        @property
        def device(self) -> torch.device:
            return torch.device("cuda:0")

    proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cuda:0"), tensor_id_mapping={})
    tensor = torch.Tensor._make_subclass(_FakeCudaTensor, torch.empty(1), require_grad=False)

    with patch.object(torch.cuda, "mem_get_info") as mock_mem_get_info:
        proc._guard_unmapped_cuda_move(tensor)

    mock_mem_get_info.assert_not_called()
