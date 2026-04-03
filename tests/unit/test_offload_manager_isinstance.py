# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadModelProxy isinstance compatibility.

This test suite validates that OffloadModelProxy properly preserves type information
for compatibility with libraries that do type checking (like HuggingFace Transformers).

Key behaviors tested:
- isinstance() checks work through the proxy
- type() checks reflect the underlying model
- hasattr() and getattr() work properly
- Class hierarchy is preserved
- Compatibility with transformers.PreTrainedModel (if available)
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.offload_manager import OffloadConfig, OffloadManager, OffloadModelProxy


# Simple test models with inheritance hierarchy
class BaseModel(nn.Module):
    """Base model class for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class DerivedModel(BaseModel):
    """Derived model class for testing inheritance."""

    def __init__(self):
        super().__init__()
        self.extra_layer = nn.Linear(10, 10)

    def forward(self, x):
        x = super().forward(x)
        return self.extra_layer(x)


class CustomModel(nn.Module):
    """Custom model with specific attributes for testing."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(10, 10)
        self.decoder = nn.Linear(10, 10)
        self.custom_attr = "test_value"

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def custom_method(self):
        return "custom_method_result"


class TestOffloadProxyInstanceOf:
    """Test cases for isinstance checks with OffloadModelProxy."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.x = torch.randn(4, 10)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_isinstance_nn_module(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that proxy passes isinstance check for nn.Module."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = BaseModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test isinstance for nn.Module
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) check"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_isinstance_base_class(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that proxy passes isinstance check for base class."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = DerivedModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test isinstance for base class and derived class
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) check"
        assert isinstance(proxy_model, BaseModel), "Proxy should pass isinstance(BaseModel) check"
        assert isinstance(proxy_model, DerivedModel), "Proxy should pass isinstance(DerivedModel) check"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_isinstance_specific_model_type(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that proxy passes isinstance check for specific model type."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test isinstance for specific model type
        assert isinstance(proxy_model, CustomModel), "Proxy should pass isinstance(CustomModel) check"
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) check"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_type_reflection(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that type() returns information about the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test type information
        proxy_type = type(proxy_model)
        print(f"Proxy type: {proxy_type}")
        print(f"Proxy type name: {proxy_type.__name__}")

        # The proxy should be OffloadModelProxy
        assert proxy_type == OffloadModelProxy or "Proxy" in proxy_type.__name__

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hasattr_through_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hasattr works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test hasattr for various attributes
        assert hasattr(proxy_model, "encoder"), "Proxy should have 'encoder' attribute"
        assert hasattr(proxy_model, "decoder"), "Proxy should have 'decoder' attribute"
        assert hasattr(proxy_model, "custom_attr"), "Proxy should have 'custom_attr' attribute"
        assert hasattr(proxy_model, "custom_method"), "Proxy should have 'custom_method' attribute"
        assert hasattr(proxy_model, "forward"), "Proxy should have 'forward' attribute"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_getattr_through_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that getattr works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        # Use empty include_patterns to not wrap any modules (original behavior when module_paths=None)
        config = OffloadConfig(offload_on=True, include_patterns=[])

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test getattr for various attributes
        encoder = getattr(proxy_model, "encoder")  # noqa: B009
        assert isinstance(encoder, nn.Linear), "Should get encoder layer"

        decoder = getattr(proxy_model, "decoder")  # noqa: B009
        assert isinstance(decoder, nn.Linear), "Should get decoder layer"

        custom_attr = getattr(proxy_model, "custom_attr")  # noqa: B009
        assert custom_attr == "test_value", "Should get custom attribute value"

        custom_method = getattr(proxy_model, "custom_method")  # noqa: B009
        assert callable(custom_method), "Should get custom method"
        assert custom_method() == "custom_method_result", "Custom method should work"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_dir_includes_model_attributes(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that dir() includes underlying model attributes."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test dir() includes model attributes
        attrs = dir(proxy_model)
        assert "encoder" in attrs, "dir() should include 'encoder'"
        assert "decoder" in attrs, "dir() should include 'decoder'"
        assert "forward" in attrs, "dir() should include 'forward'"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_isinstance_across_state_transitions(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that isinstance works correctly across state transitions."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        # Create three different model instances (as happens in real code)
        warmup_model = CustomModel()
        profile_model = CustomModel()
        inference_model = CustomModel()

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True, warmup_iters=1, profile_iters=1)

        # Offload the model
        proxy_model = om.offload(CustomModel(), config=config)

        # Test isinstance in WARMUP state
        assert isinstance(proxy_model, CustomModel), "Proxy should pass isinstance in WARMUP"
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) in WARMUP"

        # Transition to PROFILE
        with torch.no_grad():
            _ = proxy_model(self.x)
            _ = proxy_model(self.x)

        # Test isinstance in PROFILE state
        assert isinstance(proxy_model, CustomModel), "Proxy should pass isinstance in PROFILE"
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) in PROFILE"

        # Transition to INFERENCE
        with torch.no_grad():
            _ = proxy_model(self.x)
            _ = proxy_model(self.x)

        # Test isinstance in INFERENCE state
        assert isinstance(proxy_model, CustomModel), "Proxy should pass isinstance in INFERENCE"
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) in INFERENCE"


def _is_transformers_available() -> bool:
    """Check if transformers library is available."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


class TestOffloadProxyHuggingFaceCompatibility:
    """Test cases for HuggingFace transformers compatibility."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.x = torch.randn(4, 10)

    @pytest.mark.skipif(
        not _is_transformers_available(),
        reason="transformers not available",
    )
    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_huggingface_pretrained_model_isinstance(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test isinstance with HuggingFace PreTrainedModel."""
        from transformers import PreTrainedModel

        # Create a mock PreTrainedModel
        class MockPreTrainedModel(PreTrainedModel):
            config_class = MagicMock

            def __init__(self):
                # Skip parent __init__ to avoid config requirements
                nn.Module.__init__(self)
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MockPreTrainedModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test isinstance for PreTrainedModel
        assert isinstance(proxy_model, PreTrainedModel), "Proxy should pass isinstance(PreTrainedModel) check"
        assert isinstance(proxy_model, nn.Module), "Proxy should pass isinstance(nn.Module) check"

    @pytest.mark.skipif(
        not _is_transformers_available(),
        reason="transformers not available",
    )
    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_huggingface_type_checking_pattern(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test HuggingFace-style type checking pattern."""
        from transformers import PreTrainedModel

        # Create a mock PreTrainedModel
        class MockGPT2Model(PreTrainedModel):
            config_class = MagicMock

            def __init__(self):
                # Skip parent __init__ to avoid config requirements
                nn.Module.__init__(self)
                self.transformer = nn.Linear(10, 10)

            def forward(self, x):
                return self.transformer(x)

        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MockGPT2Model()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Simulate HuggingFace pipeline type checking
        def check_model_type(model):
            """Simulate HuggingFace type checking."""
            if not isinstance(model, PreTrainedModel):
                raise TypeError(f"model must be of type PreTrainedModel, got {type(model)}")
            return True

        # This should NOT raise an exception
        try:
            result = check_model_type(proxy_model)
            assert result is True, "Type checking should pass"
        except TypeError as e:
            pytest.fail(f"Type checking failed with proxy: {e}")


class TestOffloadProxyCallable:
    """Test that proxy is callable and works as a model."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.x = torch.randn(4, 10)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_proxy_is_callable(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that proxy can be called like a normal model."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test that proxy is callable
        assert callable(proxy_model), "Proxy should be callable"

        # Test calling the proxy
        with torch.no_grad():
            output = proxy_model(self.x)
            assert output is not None, "Proxy should return output"
            assert isinstance(output, torch.Tensor), "Output should be a tensor"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_proxy_preserves_model_behavior(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that proxy preserves model behavior."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CustomModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(offload_on=True)

        # Get output from original model
        with torch.no_grad():
            original_output = model(self.x)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Get output from proxy model
        with torch.no_grad():
            proxy_output = proxy_model(self.x)

        # Outputs should match
        assert torch.allclose(original_output, proxy_output), "Proxy should preserve model behavior"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
