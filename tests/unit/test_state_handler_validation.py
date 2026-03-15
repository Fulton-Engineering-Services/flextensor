# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TensorManagerStateHandler validation functionality.

This test suite validates the state compatibility checking between saved profiles
and current models. It verifies:

1. Validation detection: Correctly identifies missing and unexpected tensor names
2. Strict mode: Raises ValueError when tensors are missing from the model
3. Non-strict mode: Logs warnings and skips missing tensors
4. restore_state integration: Validation is properly called during restore
"""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.state_handler import (
    StateValidationResult,
    TensorManagerState,
    TensorManagerStateHandler,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing with known parameter names."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 20)
        self.layer2 = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


def create_tensor_statistics(names: list[str]) -> list[TensorStatistics]:
    """Create a list of TensorStatistics with given names."""
    return [TensorStatistics(tensor_id=i, name=name, size_bytes=1000, load_time_ms=0.1) for i, name in enumerate(names)]


def create_layer_statistics(names: list[str]) -> list[LayerStatistics]:
    """Create a LayerStatistics with given tensor names."""
    tensors = create_tensor_statistics(names)
    return [LayerStatistics(label="layer_0", tensors=tensors, duration=1.0)]


def create_mock_state(
    tensor_names: list[str],
    view_tensor_names: list[str] | None = None,
) -> TensorManagerState:
    """Create a mock TensorManagerState with the given tensor names."""
    if view_tensor_names is None:
        view_tensor_names = []

    return TensorManagerState(
        loader_type="strategy",
        tensor_id_to_name_map=dict(enumerate(tensor_names)),
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"layer_0": create_tensor_statistics(tensor_names)},
        release_strategy={"layer_0": create_tensor_statistics(tensor_names)},
        label_to_block_id={},
        stats=create_layer_statistics(tensor_names),
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=view_tensor_names,
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


class TestStateValidationResult:
    """Tests for the StateValidationResult dataclass."""

    def test_empty_result_is_falsy(self):
        """An empty result (no missing/unexpected) should be falsy."""
        result = StateValidationResult()
        assert not result
        assert result.missing_keys == []
        assert result.unexpected_keys == []

    def test_result_with_missing_keys_is_truthy(self):
        """A result with missing keys should be truthy."""
        result = StateValidationResult(missing_keys=["layer.weight"])
        assert result
        assert result.missing_keys == ["layer.weight"]
        assert result.unexpected_keys == []

    def test_result_with_unexpected_keys_is_truthy(self):
        """A result with unexpected keys should be truthy."""
        result = StateValidationResult(unexpected_keys=["new_layer.weight"])
        assert result
        assert result.missing_keys == []
        assert result.unexpected_keys == ["new_layer.weight"]

    def test_result_with_both_is_truthy(self):
        """A result with both missing and unexpected keys should be truthy."""
        result = StateValidationResult(
            missing_keys=["old.weight"],
            unexpected_keys=["new.weight"],
        )
        assert result


class TestValidateStateCompatibility:
    """Tests for the validate_state_compatibility method."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        # Model has: layer1.weight, layer1.bias, layer2.weight, layer2.bias
        self.model_tensor_names = {name for name, _ in self.model.named_parameters()}

        # Create mock tensor manager
        self.mock_tm = MagicMock()
        self.handler = TensorManagerStateHandler(self.mock_tm)

    def test_exact_match_no_issues(self):
        """When state tensors exactly match model tensors, no issues reported."""
        state = create_mock_state(list(self.model_tensor_names))

        result = self.handler.validate_state_compatibility(self.model, state)

        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert not result

    def test_missing_keys_detected(self):
        """Tensors in state but not in model are reported as missing."""
        # Add a tensor name that doesn't exist in the model
        tensor_names = [*self.model_tensor_names, "nonexistent.weight", "another.bias"]
        state = create_mock_state(tensor_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "nonexistent.weight" in result.missing_keys
        assert "another.bias" in result.missing_keys
        assert len(result.missing_keys) == 2

    def test_unexpected_keys_detected(self):
        """Tensors in model but not in state are reported as unexpected."""
        # Only include some of the model's tensors in the state
        partial_names = ["layer1.weight", "layer1.bias"]
        state = create_mock_state(partial_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "layer2.weight" in result.unexpected_keys
        assert "layer2.bias" in result.unexpected_keys
        assert len(result.unexpected_keys) == 2

    def test_both_missing_and_unexpected(self):
        """Both missing and unexpected keys can be detected simultaneously."""
        # State has some tensors model doesn't, model has some tensors state doesn't
        state_names = ["layer1.weight", "old_layer.weight"]
        state = create_mock_state(state_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "old_layer.weight" in result.missing_keys
        assert "layer1.bias" in result.unexpected_keys
        assert "layer2.weight" in result.unexpected_keys
        assert "layer2.bias" in result.unexpected_keys

    def test_view_tensors_included_in_validation(self):
        """View tensor names from state are also validated."""
        # Model tensors plus a view tensor that doesn't exist
        state = create_mock_state(
            list(self.model_tensor_names),
            view_tensor_names=["missing_view.weight"],
        )

        result = self.handler.validate_state_compatibility(self.model, state)

        assert "missing_view.weight" in result.missing_keys

    def test_dict_model_supported(self):
        """Validation works with dict models (state_dict style)."""
        model_dict = {name: torch.randn(10) for name in ["a.weight", "b.weight"]}
        state = create_mock_state(["a.weight", "c.weight"])

        result = self.handler.validate_state_compatibility(model_dict, state)

        assert "c.weight" in result.missing_keys
        assert "b.weight" in result.unexpected_keys

    def test_keys_are_sorted(self):
        """Missing and unexpected keys are returned sorted for consistency."""
        state_names = ["z.weight", "a.weight", "m.weight"]
        state = create_mock_state(state_names)

        result = self.handler.validate_state_compatibility(self.model, state)

        # Missing keys should be sorted
        assert result.missing_keys == sorted(result.missing_keys)
        # Unexpected keys should be sorted
        assert result.unexpected_keys == sorted(result.unexpected_keys)


class TestCheckValidationResult:
    """Tests for the _check_validation_result static method."""

    def test_no_issues_does_not_raise(self):
        """When there are no issues, no exception is raised."""
        result = StateValidationResult()
        # Should not raise
        TensorManagerStateHandler._check_validation_result(result, strict=True)

    def test_missing_keys_strict_raises_value_error(self):
        """In strict mode, missing keys raise ValueError."""
        result = StateValidationResult(missing_keys=["layer.weight", "layer.bias"])

        with pytest.raises(ValueError) as exc_info:
            TensorManagerStateHandler._check_validation_result(result, strict=True)

        error_msg = str(exc_info.value)
        assert "2 tensor(s)" in error_msg
        assert "layer.weight" in error_msg
        assert "strict=False" in error_msg

    def test_missing_keys_non_strict_logs_warning(self, caplog):
        """In non-strict mode, missing keys log a warning but don't raise."""
        result = StateValidationResult(missing_keys=["layer.weight"])

        with caplog.at_level(logging.WARNING):
            # Should not raise
            TensorManagerStateHandler._check_validation_result(result, strict=False)

        assert "1 tensor(s)" in caplog.text
        assert "layer.weight" in caplog.text
        assert "Skipping" in caplog.text

    def test_unexpected_keys_logs_info(self, caplog):
        """Unexpected keys log an info message (both strict and non-strict)."""
        result = StateValidationResult(unexpected_keys=["new.weight", "new.bias"])

        with caplog.at_level(logging.INFO):
            TensorManagerStateHandler._check_validation_result(result, strict=True)

        assert "2 tensor(s)" in caplog.text
        assert "new.weight" in caplog.text

    def test_many_missing_keys_shows_sample_with_ellipsis(self):
        """When there are many missing keys, only a sample is shown."""
        many_keys = [f"layer{i}.weight" for i in range(10)]
        result = StateValidationResult(missing_keys=many_keys)

        with pytest.raises(ValueError) as exc_info:
            TensorManagerStateHandler._check_validation_result(result, strict=True)

        error_msg = str(exc_info.value)
        assert "10 tensor(s)" in error_msg
        assert "..." in error_msg  # Ellipsis indicates truncation


class TestLoaderTypeMismatch:
    """Tests for loader_type mismatch detection in restore_state."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model_tensor_names = [name for name, _ in self.model.named_parameters()]

    def test_restore_state_raises_on_loader_type_mismatch(self):
        """restore_state raises ValueError when loader_type doesn't match."""
        # Create mock tensor manager with strategy loader type
        mock_tm = MagicMock()
        mock_tm.use_trace_tensor = True
        mock_tm.loader_type = "strategy"

        handler = TensorManagerStateHandler(mock_tm)

        # Create state with different loader_type (allocation_block_transfer)
        state = create_mock_state(self.model_tensor_names)
        state.loader_type = "allocation_block_transfer"

        with pytest.raises(ValueError) as exc_info:
            handler.restore_state(self.model, state)

        error_msg = str(exc_info.value)
        assert "Saved profile uses loader_type='allocation_block_transfer'" in error_msg
        assert "TensorManager is configured with loader_type='strategy'" in error_msg
        assert "incompatible tensor mappings" in error_msg
        assert "re-profile" in error_msg

    def test_restore_state_succeeds_on_matching_loader_type(self):
        """restore_state succeeds when loader_type matches."""
        mock_tm = MagicMock()
        mock_tm.use_trace_tensor = True
        mock_tm.loader_type = "allocation_block_transfer"

        handler = TensorManagerStateHandler(mock_tm)

        state = create_mock_state(self.model_tensor_names)
        state.loader_type = "allocation_block_transfer"

        # Should not raise
        result = handler.restore_state(self.model, state, strict=True)
        assert isinstance(result, StateValidationResult)

    def test_restore_state_loader_type_mismatch_all_combinations(self):
        """Test loader_type mismatch detection for various loader type combinations."""
        loader_types = ["strategy", "allocation_block_transfer", "raw_block_transfer"]

        for manager_loader_type in loader_types:
            for state_loader_type in loader_types:
                mock_tm = MagicMock()
                mock_tm.use_trace_tensor = True
                mock_tm.loader_type = manager_loader_type

                handler = TensorManagerStateHandler(mock_tm)

                state = create_mock_state(self.model_tensor_names)
                state.loader_type = state_loader_type

                if manager_loader_type == state_loader_type:
                    # Should not raise
                    handler.restore_state(self.model, state, strict=True)
                else:
                    # Should raise
                    with pytest.raises(ValueError, match="Saved profile uses loader_type="):
                        handler.restore_state(self.model, state, strict=True)


class TestRestoreStateWithValidation:
    """Integration tests for restore_state with validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model_tensor_names = [name for name, _ in self.model.named_parameters()]

        # Create mock tensor manager with required attributes
        self.mock_tm = MagicMock()
        self.mock_tm.use_trace_tensor = True  # Skip preprocess_model
        self.mock_tm.loader_type = "strategy"

        self.handler = TensorManagerStateHandler(self.mock_tm)

    def test_restore_state_strict_raises_on_missing(self):
        """restore_state with strict=True raises on missing tensors."""
        state = create_mock_state([*self.model_tensor_names, "missing.weight"])

        with pytest.raises(ValueError, match=r"missing\.weight"):
            self.handler.restore_state(self.model, state, strict=True)

    def test_restore_state_non_strict_skips_missing(self, caplog):
        """restore_state with strict=False skips missing tensors."""
        extra_tensor = "missing.weight"
        state = create_mock_state([*self.model_tensor_names, extra_tensor])

        with caplog.at_level(logging.WARNING):
            result = self.handler.restore_state(self.model, state, strict=False)

        # Should have logged a warning
        assert "missing.weight" in caplog.text

        # Result should indicate what was missing
        assert extra_tensor in result.missing_keys

        # The restored state (deep-copied inside restore_state) should exclude the missing tensor
        restored_state = self.mock_tm.tensor_manager_state
        for s_stats in restored_state.load_strategy.values():
            tensor_names = [s.name for s in s_stats]
            assert extra_tensor not in tensor_names

    def test_restore_state_returns_validation_result(self):
        """restore_state returns StateValidationResult for inspection."""
        # Use exact match - should have no missing but maybe unexpected
        partial_names = ["layer1.weight", "layer1.bias"]
        state = create_mock_state(partial_names)

        result = self.handler.restore_state(self.model, state, strict=True)

        assert isinstance(result, StateValidationResult)
        assert result.missing_keys == []
        assert "layer2.weight" in result.unexpected_keys

    def test_restore_state_default_is_strict(self):
        """restore_state defaults to strict=True."""
        state = create_mock_state([*self.model_tensor_names, "missing.weight"])

        # Should raise without explicit strict parameter
        with pytest.raises(ValueError):
            self.handler.restore_state(self.model, state)

    def test_restore_state_filters_view_tensors(self):
        """restore_state filters out missing view tensor names."""
        state = create_mock_state(
            self.model_tensor_names,
            view_tensor_names=["valid.view", "missing.view"],
        )
        # Add valid.view to model for partial match
        # (In this case, both will be filtered since SimpleModel doesn't have them)

        _result = self.handler.restore_state(self.model, state, strict=False)

        # The restored state (deep-copied inside restore_state) should have filtered view tensors
        # Since neither exists in SimpleModel, both should be removed
        restored_state = self.mock_tm.tensor_manager_state
        assert "missing.view" not in restored_state.view_tensors_names
        assert "valid.view" not in restored_state.view_tensors_names
