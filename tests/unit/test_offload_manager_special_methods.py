# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadModelProxy special methods and dunder method delegation.

This test suite validates that OffloadModelProxy properly handles special methods
(dunder methods) that are not automatically delegated via __getattr__.

Python's __getattr__ does NOT intercept special methods like:
- __len__, __getitem__, __setitem__
- __iter__, __next__
- __contains__
- __enter__, __exit__ (context managers)
- __call__ (already handled for forward)

Key behaviors tested:
- Subscriptable models (__getitem__)
- Iterable models (__iter__, __len__)
- Context manager models (__enter__, __exit__)
- Container models (__contains__)
- Comparison operations
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.offload_manager import OffloadConfig, OffloadManager


# Test models with special methods
class SubscriptableModel(nn.Module):
    """Model that supports indexing with __getitem__."""

    def __init__(self, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(num_layers)])

    def __getitem__(self, idx):
        """Allow indexing to access layers."""
        return self.layers[idx]

    def __len__(self):
        """Return number of layers."""
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class IterableModel(nn.Module):
    """Model that supports iteration with __iter__."""

    def __init__(self, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(num_layers)])

    def __iter__(self):
        """Allow iteration over layers."""
        return iter(self.layers)

    def __len__(self):
        """Return number of layers."""
        return len(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ContextManagerModel(nn.Module):
    """Model that implements context manager protocol."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.setup_called = False
        self.cleanup_called = False

    def __enter__(self):
        """Setup when entering context."""
        self.setup_called = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting context."""
        self.cleanup_called = True
        return False

    def forward(self, x):
        return self.linear(x)


class ContainerModel(nn.Module):
    """Model that implements container protocol."""

    def __init__(self):
        super().__init__()
        self.layer_names = {"encoder", "decoder", "head"}
        self.encoder = nn.Linear(10, 20)
        self.decoder = nn.Linear(20, 10)
        self.head = nn.Linear(10, 5)

    def __contains__(self, item):
        """Check if layer name exists."""
        return item in self.layer_names

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.head(x)


class CallableLayerModel(nn.Module):
    """Model where layers are callable objects."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    def __call__(self, x, custom_arg=None):
        """Custom __call__ that takes extra arguments."""
        x = self.layer1(x)
        if custom_arg is not None:
            x = x * custom_arg
        return self.layer2(x)

    def forward(self, x):
        return self.__call__(x)


class TestOffloadProxySubscriptable:
    """Test cases for subscriptable models (__getitem__, __len__)."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_getitem_access(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that __getitem__ works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test __getitem__ access
        try:
            # Access first layer
            layer_0 = proxy_model[0]
            assert layer_0 is not None, "Should be able to access layer via index"
            assert isinstance(layer_0, nn.Linear), "Should return Linear layer"

            # Access middle layer
            layer_2 = proxy_model[2]
            assert layer_2 is not None, "Should be able to access middle layer"

            # Access last layer
            layer_4 = proxy_model[4]
            assert layer_4 is not None, "Should be able to access last layer"

        except TypeError as e:
            pytest.fail(f"__getitem__ not supported through proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_len_function(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that len() works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test len()
        try:
            length = len(proxy_model)
            assert length == 5, f"len() should return 5, got {length}"
        except TypeError as e:
            pytest.fail(f"len() not supported through proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_negative_indexing(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that negative indexing works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test negative indexing
        try:
            last_layer = proxy_model[-1]
            assert last_layer is not None, "Should access last layer with -1"
            assert isinstance(last_layer, nn.Linear), "Should return Linear layer"
        except (TypeError, IndexError) as e:
            pytest.fail(f"Negative indexing not supported: {e}")


class TestOffloadProxyIterable:
    """Test cases for iterable models (__iter__)."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_iter_function(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that iter() works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = IterableModel(num_layers=3)
        layers = list(iter(model))
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        layers = list(iter(proxy_model))
        # Test iteration
        try:
            layers = list(iter(proxy_model))
            assert len(layers) == 3, f"Should iterate over 3 layers, got {len(layers)}"
            assert all(isinstance(layer, nn.Linear) for layer in layers), "All items should be Linear layers"
        except TypeError as e:
            pytest.fail(f"iter() not supported through proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_for_loop_iteration(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that for loop iteration works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = IterableModel(num_layers=3)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test for loop
        try:
            count = 0
            for layer in proxy_model:
                assert isinstance(layer, nn.Linear), "Each item should be a Linear layer"
                count += 1
            assert count == 3, f"Should iterate 3 times, got {count}"
        except TypeError as e:
            pytest.fail(f"for loop iteration not supported through proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_len_with_iterable(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that len() works with iterable models."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = IterableModel(num_layers=3)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test len()
        try:
            length = len(proxy_model)
            assert length == 3, f"len() should return 3, got {length}"
        except TypeError as e:
            pytest.fail(f"len() not supported through proxy: {e}")


class TestOffloadProxyContextManager:
    """Test cases for context manager protocol (__enter__, __exit__)."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_context_manager_protocol(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that context manager protocol works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = ContextManagerModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test context manager
        try:
            with proxy_model as m:
                assert m is not None, "Context manager should return model"
                # Verify setup was called
                assert model.setup_called, "Setup should have been called"

            # Verify cleanup was called
            assert model.cleanup_called, "Cleanup should have been called"

        except AttributeError as e:
            pytest.fail(f"Context manager protocol not supported through proxy: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_context_manager_with_forward(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test using context manager with forward pass."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = ContextManagerModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test context manager with forward
        try:
            x = torch.randn(4, 10)
            with proxy_model as m:
                output = m(x)
                assert output is not None, "Should be able to call forward in context"
                assert output.shape == (4, 10), "Output shape should match"

            assert model.setup_called, "Setup should have been called"
            assert model.cleanup_called, "Cleanup should have been called"

        except (AttributeError, RuntimeError) as e:
            pytest.fail(f"Context manager with forward not supported: {e}")


class TestOffloadProxyContainerProtocol:
    """Test cases for container protocol (__contains__)."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_contains_operator(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that 'in' operator (__contains__) works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = ContainerModel()
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test 'in' operator
        try:
            assert "encoder" in proxy_model, "'encoder' should be in model"
            assert "decoder" in proxy_model, "'decoder' should be in model"
            assert "head" in proxy_model, "'head' should be in model"
            assert "nonexistent" not in proxy_model, "'nonexistent' should not be in model"

        except TypeError as e:
            pytest.fail(f"'in' operator (__contains__) not supported through proxy: {e}")


class TestOffloadProxyCustomCall:
    """Test cases for custom __call__ implementations."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_custom_call_with_kwargs(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that custom __call__ with kwargs works through the proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = CallableLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test calling with custom argument
        try:
            x = torch.randn(4, 10)

            # Call without custom arg
            output1 = proxy_model(x)
            assert output1 is not None, "Should be able to call without custom arg"

            # Call with custom arg
            output2 = proxy_model(x, custom_arg=2.0)
            assert output2 is not None, "Should be able to call with custom arg"

        except TypeError as e:
            pytest.fail(f"Custom __call__ not properly supported: {e}")


class TestOffloadProxySlicing:
    """Test cases for slicing operations."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_slicing_support(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that slicing works if model supports it."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test slicing (if supported by underlying model)
        try:
            # Note: This will only work if the model's __getitem__ supports slices
            # Our test model might not support it, so we catch both success and expected failure
            _ = proxy_model[0:2]
            # If we get here, slicing works
        except TypeError as e:
            # Expected if model doesn't support slicing
            if "slice" not in str(e).lower():
                pytest.fail(f"Unexpected error with slicing: {e}")


class TestOffloadProxyComparisonOps:
    """Test cases for comparison operations."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_equality_comparison(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that equality comparison is possible."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test equality
        try:
            # Compare with itself
            result = proxy_model == proxy_model
            # Just verify it doesn't crash
            assert result is not None

            # Compare with None
            result = proxy_model == None  # noqa: E711
            assert result is not None

        except Exception as e:
            pytest.fail(f"Equality comparison failed: {e}")


class TestOffloadProxyBoolConversion:
    """Test cases for bool conversion."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_bool_conversion(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that bool conversion works."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test bool conversion
        try:
            # Should be truthy (model exists)
            if proxy_model:
                assert True, "Model should be truthy"
            else:
                pytest.fail("Model should be truthy")

            # Test in boolean context
            result = proxy_model and True
            assert result is not None

        except Exception as e:
            pytest.fail(f"Bool conversion failed: {e}")


class TestOffloadProxyIntrospection:
    """Test cases for introspection methods like __dir__."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_dir_includes_model_attributes(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that dir() includes both proxy and underlying model attributes."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test dir()
        try:
            attrs = dir(proxy_model)

            # Should include model's attributes
            assert "layers" in attrs, "Should include model's 'layers' attribute"
            assert "forward" in attrs, "Should include model's 'forward' method"

            # Should include special methods from underlying model
            assert "__getitem__" in attrs, "Should include __getitem__ from model"
            assert "__len__" in attrs, "Should include __len__ from model"

            # Verify it returns a list
            assert isinstance(attrs, list), "dir() should return a list"

            # Verify list is sorted
            assert attrs == sorted(attrs), "dir() should return sorted list"

        except Exception as e:
            pytest.fail(f"dir() introspection failed: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_dir_includes_proxy_methods(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that dir() includes proxy's own methods."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SubscriptableModel(num_layers=5)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test dir()
        try:
            attrs = dir(proxy_model)

            # Should include proxy's own methods
            assert "register_forward_hook" in attrs, "Should include proxy's register_forward_hook method"
            assert "register_forward_pre_hook" in attrs, "Should include proxy's register_forward_pre_hook method"
            assert "register_full_backward_hook" in attrs, "Should include proxy's register_full_backward_hook method"

        except Exception as e:
            pytest.fail(f"dir() introspection of proxy methods failed: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_dir_with_iterable_model(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that dir() works with different model types."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = IterableModel(num_layers=3)
        mock_tensor_manager.initialize_warmup.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Offload the model
        proxy_model = om.offload(model, config=config)

        # Test dir()
        try:
            attrs = dir(proxy_model)

            # Should include iterable model's attributes
            assert "layers" in attrs, "Should include model's 'layers' attribute"
            assert "__iter__" in attrs, "Should include __iter__ from model"

            # Should still be sorted
            assert attrs == sorted(attrs), "dir() should return sorted list"

        except Exception as e:
            pytest.fail(f"dir() with iterable model failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
