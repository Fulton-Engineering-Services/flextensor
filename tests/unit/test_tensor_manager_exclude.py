# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TensorManager excluded tensor GPU placement."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.strategy import KnapsackStrategy
from flextensor.tensor_manager import TensorManager


class SimpleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))
        self.bias = torch.nn.Parameter(torch.randn(10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = SimpleModule()
        self.layer2 = SimpleModule()
        self.layer3 = SimpleModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def _make_tensor_manager(
    device_gpu: torch.device,
    exclude_patterns: list[str] | None = None,
) -> TensorManager:
    """Create a TensorManager with sensible defaults for testing."""
    strategy = MagicMock(spec=KnapsackStrategy)
    return TensorManager(
        device_gpu=device_gpu,
        pinned_memory=False,
        tensor_manager_load_strategy=strategy,
        exclude_patterns=exclude_patterns,
    )


class TestMoveExcludedTensorsToGPU:
    """Tests for _move_excluded_tensors_to_gpu()."""

    def setup_method(self):
        self.device_gpu = torch.device("cpu")  # Use CPU as "GPU" for unit tests

    def test_excluded_tensors_removed_from_tracking(self):
        """Excluded tensor IDs are removed from tensors_map and traced_tensors."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["layer1.*"])
        tm.set_model(model)

        # Simulate what preprocess_model does: populate tensors_map and traced_tensors
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        layer1_ids = {id(model.layer1.weight), id(model.layer1.bias)}

        # Verify layer1 IDs are present before
        assert layer1_ids.issubset(tm.tensors_map.keys())
        assert layer1_ids.issubset(tm.traced_tensors)

        tm._move_excluded_tensors_to_gpu()

        # layer1 IDs should be removed
        for tid in layer1_ids:
            assert tid not in tm.tensors_map
            assert tid not in tm.traced_tensors

        # layer2 and layer3 IDs should remain
        layer2_ids = {id(model.layer2.weight), id(model.layer2.bias)}
        layer3_ids = {id(model.layer3.weight), id(model.layer3.bias)}
        assert layer2_ids.issubset(tm.tensors_map.keys())
        assert layer3_ids.issubset(tm.tensors_map.keys())

    def test_idempotent_on_double_call(self):
        """Calling _move_excluded_tensors_to_gpu() twice is a no-op the second time."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["layer1.*"])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        tm._move_excluded_tensors_to_gpu()
        remaining_after_first = dict(tm.tensors_map)

        tm._move_excluded_tensors_to_gpu()
        remaining_after_second = dict(tm.tensors_map)

        assert remaining_after_first.keys() == remaining_after_second.keys()
        assert tm._excluded_tensors_moved is True

    def test_noop_with_empty_patterns(self):
        """No-op when exclude_patterns is empty."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=[])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        original_ids = set(tm.tensors_map.keys())

        tm._move_excluded_tensors_to_gpu()

        assert set(tm.tensors_map.keys()) == original_ids
        assert tm._excluded_tensors_moved is True

    def test_noop_with_no_matching_patterns(self):
        """No-op when patterns don't match any parameters."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["nonexistent.*"])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        original_ids = set(tm.tensors_map.keys())

        tm._move_excluded_tensors_to_gpu()

        assert set(tm.tensors_map.keys()) == original_ids

    def test_flag_set_after_call(self):
        """The _excluded_tensors_moved flag is set after the call."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["layer1.*"])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        assert tm._excluded_tensors_moved is False
        tm._move_excluded_tensors_to_gpu()
        assert tm._excluded_tensors_moved is True

    def test_parameter_mapping_refreshed(self):
        """tensor_id_to_name_map is refreshed after GPU move."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["layer1.*"])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        tm._move_excluded_tensors_to_gpu()

        # After refresh, all current parameter names should be mapped
        current_param_ids = {id(p) for p in model.parameters()}
        assert set(tm.tensor_id_to_name_map.keys()) == current_param_ids

    def test_processor_receives_correct_mapping(self):
        """MoveUnmappedTensorsToGPUProcessor is constructed with non-excluded tensors and apply() is called."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["layer1.*"])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        layer1_ids = {id(model.layer1.weight), id(model.layer1.bias)}
        expected_mapping = {tid: t for tid, t in tm.tensors_map.items() if tid not in layer1_ids}

        with patch("flextensor.tensor_manager.MoveUnmappedTensorsToGPUProcessor") as mock_cls:
            mock_processor = MagicMock()
            mock_cls.return_value = mock_processor

            tm._move_excluded_tensors_to_gpu()

            mock_cls.assert_called_once_with(self.device_gpu, expected_mapping)
            mock_processor.apply.assert_called_once_with(model)

    def test_oom_raises_contextual_error(self):
        """OOM during excluded tensor GPU placement raises an actionable error with chained cause."""
        model = SimpleModel()
        tm = _make_tensor_manager(self.device_gpu, exclude_patterns=["layer1.*"])
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())

        original_oom = torch.cuda.OutOfMemoryError("CUDA out of memory.")
        with patch("flextensor.tensor_manager.MoveUnmappedTensorsToGPUProcessor") as mock_cls:
            mock_processor = MagicMock()
            mock_processor.apply.side_effect = original_oom
            mock_cls.return_value = mock_processor

            with pytest.raises(torch.cuda.OutOfMemoryError, match="exclude_patterns") as exc_info:
                tm._move_excluded_tensors_to_gpu()

            assert exc_info.value.__cause__ is original_oom
            assert "excluded tensors" in str(exc_info.value)
            assert "GiB" in str(exc_info.value)
