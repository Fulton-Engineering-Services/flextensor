# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadModelProxy serialization and saving compatibility.

This test suite validates that OffloadModelProxy properly handles model saving,
loading, and serialization operations that are critical for:
- Model checkpointing
- Fine-tuning workflows
- Model distribution
- HuggingFace model hub integration

Key behaviors tested:
- state_dict() access through proxy
- torch.save() and torch.load() compatibility
- Pickle serialization (if supported)
- HuggingFace save_pretrained() and from_pretrained() (if available)
- Parameter access for optimization
"""

import io
import pickle  # noqa: S403
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.offload_manager import OffloadConfig, OffloadManager


# Test models
class SimpleLinearModel(nn.Module):
    """Simple model for testing serialization."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ModelWithCustomState(nn.Module):
    """Model with custom state for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.custom_buffer = None
        self.register_buffer("buffer", torch.zeros(10))

    def forward(self, x):
        return self.linear(x) + self.buffer


def _is_transformers_available() -> bool:
    """Check if transformers library is available."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestOffloadProxyStateDictAccess:
    """Test cases for state_dict() access through the proxy."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.x = torch.randn(4, 10)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_state_dict_accessible(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that state_dict() is accessible through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        original_state_dict = model.state_dict()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test state_dict access
        proxy_state_dict = proxy_model.state_dict()

        assert proxy_state_dict is not None, "state_dict() should be accessible"
        assert isinstance(proxy_state_dict, dict), "state_dict() should return a dict"
        assert len(proxy_state_dict) > 0, "state_dict() should not be empty"

        # Verify all keys are present
        for key in original_state_dict:
            assert key in proxy_state_dict, f"Key '{key}' should be in proxy state_dict"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_state_dict_contains_correct_parameters(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that state_dict() contains the correct parameters."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        original_state_dict = model.state_dict()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Get state_dict from proxy
        proxy_state_dict = proxy_model.state_dict()

        # Compare state dicts
        assert set(original_state_dict.keys()) == set(proxy_state_dict.keys()), "Keys should match"

        # Compare tensor values
        for key in original_state_dict:
            assert torch.allclose(original_state_dict[key], proxy_state_dict[key]), f"Values for '{key}' should match"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_state_dict_across_state_transitions(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that state_dict() works across state transitions."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        # Create models for each state
        warmup_model = SimpleLinearModel()
        profile_model = SimpleLinearModel()
        inference_model = SimpleLinearModel()

        # Make them have the same weights
        state = warmup_model.state_dict()
        profile_model.load_state_dict(state)
        inference_model.load_state_dict(state)

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True, warmup_iters=1, profile_iters=1)

        # Offload the model
        proxy_model = om.offload(SimpleLinearModel(), config=config)

        # Get state_dict in WARMUP
        warmup_state = proxy_model.state_dict()
        assert warmup_state is not None, "state_dict() should work in WARMUP"

        # Transition to PROFILE
        with torch.no_grad():
            _ = proxy_model(self.x)
            _ = proxy_model(self.x)

        # Get state_dict in PROFILE
        profile_state = proxy_model.state_dict()
        assert profile_state is not None, "state_dict() should work in PROFILE"

        # Transition to INFERENCE
        with torch.no_grad():
            _ = proxy_model(self.x)
            _ = proxy_model(self.x)

        # Get state_dict in INFERENCE
        inference_state = proxy_model.state_dict()
        assert inference_state is not None, "state_dict() should work in INFERENCE"


class TestOffloadProxyTorchSave:
    """Test cases for torch.save() and torch.load() with proxy."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_torch_save_state_dict(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that torch.save() works with proxy.state_dict()."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        original_state_dict = model.state_dict()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Save state_dict to in-memory buffer
        try:
            buffer = io.BytesIO()
            torch.save(proxy_model.state_dict(), buffer)
            buffer.seek(0)

            # Load state_dict from buffer
            loaded_state_dict = torch.load(buffer, weights_only=True)
            assert isinstance(loaded_state_dict, dict), "Loaded state should be a dict"

            # Verify contents
            for key in original_state_dict:
                assert key in loaded_state_dict, f"Key '{key}' should be in loaded state"
                assert torch.allclose(original_state_dict[key], loaded_state_dict[key]), (
                    f"Values for '{key}' should match"
                )

        except Exception as e:
            pytest.fail(f"torch.save() failed with proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_load_state_dict_into_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that load_state_dict() works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Create new state dict with different values
        new_state_dict = {k: v + 1.0 for k, v in model.state_dict().items()}

        # Load new state into proxy
        try:
            proxy_model.load_state_dict(new_state_dict)

            # Verify the state was updated
            current_state = proxy_model.state_dict()
            for key in new_state_dict:
                assert torch.allclose(current_state[key], new_state_dict[key]), f"State for '{key}' should be updated"

        except Exception as e:
            pytest.fail(f"load_state_dict() failed with proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_and_load_full_checkpoint(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test saving and loading a full checkpoint with optimizer state."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Create optimizer
        optimizer = torch.optim.Adam(proxy_model.parameters(), lr=0.001)

        # Save checkpoint
        checkpoint = {
            "model_state_dict": proxy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 5,
        }

        try:
            # Save to in-memory buffer
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)

            # Load checkpoint from buffer
            loaded_checkpoint = torch.load(buffer, weights_only=False)

            assert "model_state_dict" in loaded_checkpoint
            assert "optimizer_state_dict" in loaded_checkpoint
            assert loaded_checkpoint["epoch"] == 5

            # Load state back into model
            proxy_model.load_state_dict(loaded_checkpoint["model_state_dict"])

        except Exception as e:
            pytest.fail(f"Full checkpoint save/load failed: {e}")


class TestOffloadProxyParametersAccess:
    """Test cases for parameters() and named_parameters() access."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.x = torch.randn(4, 10)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_parameters_accessible(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that parameters() is accessible through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        original_params = list(model.parameters())
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Get parameters through proxy
        proxy_params = list(proxy_model.parameters())

        assert len(proxy_params) > 0, "parameters() should not be empty"
        assert len(proxy_params) == len(original_params), "Parameter count should match"

        # Verify parameters are tensors
        for param in proxy_params:
            assert isinstance(param, torch.Tensor), "Each parameter should be a tensor"
            assert param.requires_grad, "Parameters should require grad"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_named_parameters_accessible(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that named_parameters() is accessible through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        original_named_params = dict(model.named_parameters())
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Get named parameters through proxy
        proxy_named_params = dict(proxy_model.named_parameters())

        assert len(proxy_named_params) > 0, "named_parameters() should not be empty"
        assert len(proxy_named_params) == len(original_named_params), "Parameter count should match"

        # Verify parameter names match
        assert set(proxy_named_params.keys()) == set(original_named_params.keys()), "Parameter names should match"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_optimizer_initialization_with_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that an optimizer can be initialized with proxy.parameters()."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Create optimizer with proxy parameters
        try:
            optimizer = torch.optim.Adam(proxy_model.parameters(), lr=0.001)
            assert optimizer is not None, "Optimizer should be created"

            # Verify optimizer has param groups
            assert len(optimizer.param_groups) > 0, "Optimizer should have param groups"
            assert len(optimizer.param_groups[0]["params"]) > 0, "Optimizer should have parameters"

        except Exception as e:
            pytest.fail(f"Optimizer initialization failed with proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_optimizer_step_with_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that optimizer.step() works with proxy parameters."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Create optimizer
        optimizer = torch.optim.SGD(proxy_model.parameters(), lr=0.01)

        # Get initial parameter values
        initial_params = {name: param.clone() for name, param in proxy_model.named_parameters()}

        # Simulate training step
        try:
            x = torch.randn(4, 10)
            target = torch.randn(4, 10)

            # Forward pass
            output = proxy_model(x)
            loss = nn.functional.mse_loss(output, target)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Verify parameters were updated
            updated_params = {name: param.clone() for name, param in proxy_model.named_parameters()}

            # At least some parameters should have changed
            params_changed = False
            for name in initial_params:
                if not torch.allclose(initial_params[name], updated_params[name]):
                    params_changed = True
                    break

            assert params_changed, "At least some parameters should have been updated by optimizer"

        except Exception as e:
            pytest.fail(f"Optimizer step failed with proxy: {e}")


class TestOffloadProxyPickleSerialization:
    """Test cases for pickle serialization (if supported)."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_pickle_state_dict(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that state_dict() can be pickled."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleLinearModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Get state_dict
        state_dict = proxy_model.state_dict()

        # Try to pickle state_dict
        try:
            pickled = pickle.dumps(state_dict)
            assert pickled is not None, "state_dict should be picklable"

            # Try to unpickle
            unpickled = pickle.loads(pickled)  # noqa: S301
            assert isinstance(unpickled, dict), "Unpickled object should be a dict"
            assert len(unpickled) == len(state_dict), "Unpickled state should have same length"

        except Exception as e:
            pytest.fail(f"Pickle serialization of state_dict failed: {e}")


class TestOffloadProxyHuggingFaceSerialization:
    """Test cases for HuggingFace save_pretrained() and from_pretrained()."""

    @pytest.mark.skipif(
        not _is_transformers_available(),
        reason="transformers not available",
    )
    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_save_pretrained_through_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that save_pretrained() is accessible through the proxy."""
        from transformers import PreTrainedModel

        # Create a mock PreTrainedModel
        class MockModel(PreTrainedModel):
            config_class = MagicMock

            def __init__(self):
                nn.Module.__init__(self)
                self.linear = nn.Linear(10, 10)
                self.config = MagicMock()

            def forward(self, x):
                return self.linear(x)

        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MockModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Verify that the method is accessible
        assert hasattr(proxy_model, "save_pretrained"), "save_pretrained should be accessible"
        assert callable(proxy_model.save_pretrained), "save_pretrained should be callable"

    @pytest.mark.skipif(
        not _is_transformers_available(),
        reason="transformers not available",
    )
    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_state_dict_compatible_with_huggingface(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that state_dict() is compatible with HuggingFace patterns."""
        from transformers import PreTrainedModel

        # Create a mock PreTrainedModel
        class MockModel(PreTrainedModel):
            config_class = MagicMock

            def __init__(self):
                nn.Module.__init__(self)
                self.linear = nn.Linear(10, 10)
                self.config = MagicMock()

            def forward(self, x):
                return self.linear(x)

        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MockModel()
        original_state = model.state_dict()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Get state_dict through proxy
        proxy_state = proxy_model.state_dict()

        # Verify compatibility
        assert isinstance(proxy_state, dict), "state_dict should be a dict"
        assert set(proxy_state.keys()) == set(original_state.keys()), "Keys should match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
