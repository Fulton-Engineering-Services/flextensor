# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadModelProxy compatibility with PyTorch hooks.

This test suite validates that hooks work correctly with the OffloadModelProxy:

Critical behaviors tested:
- Forward hook registration
- Backward hook registration
- Activation capture
- Gradient manipulation
- Hook removal
- Multiple hooks
- Hooks on submodules
- Hooks through proxy vs underlying model

Hooks are essential for:
- Activation visualization
- Gradient clipping
- Feature extraction
- Custom regularization
- Debugging and monitoring
- Advanced training techniques
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.offload_manager import OffloadConfig, OffloadManager


# Test models
class SimpleHookModel(nn.Module):
    """Simple model for hook testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


class MultiLayerModel(nn.Module):
    """Model with multiple named layers for hook testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        return self.layer3(x)


class TestForwardHooks:
    """Test forward hook registration and execution."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_register_forward_hook_on_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering a forward hook on the proxy model."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register hook
        hook_called = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_called.append(True)

        try:
            handle = proxy_model.register_forward_hook(forward_hook)
            assert handle is not None, "Should return hook handle"

            # Run forward pass
            x = torch.randn(4, 10)
            with torch.no_grad():
                _ = proxy_model(x)

            # Check hook was called
            assert len(hook_called) > 0, "Forward hook should be called"

            # Remove hook
            handle.remove()

        except Exception as e:
            pytest.fail(f"Failed to register forward hook: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_forward_hook_captures_output(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that forward hook can capture model output."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Capture output
        captured_output = []

        def capture_hook(module, input, output):  # noqa: A002
            captured_output.append(output.detach().clone())

        handle = proxy_model.register_forward_hook(capture_hook)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            actual_output = proxy_model(x)

        # Check captured output matches
        assert len(captured_output) == 1, "Should capture one output"
        assert torch.allclose(captured_output[0], actual_output), "Captured output should match actual"

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_forward_hook_on_submodule(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering forward hook on a specific submodule."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register hook on specific layer
        hook_called = []

        def layer_hook(module, input, output):  # noqa: A002
            hook_called.append(output.shape)

        # Access submodule through proxy
        handle = proxy_model.layer2.register_forward_hook(layer_hook)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # Check hook was called with correct shape
        assert len(hook_called) == 1, "Hook should be called once"
        assert hook_called[0] == torch.Size([4, 30]), f"Expected shape (4, 30), got {hook_called[0]}"

        handle.remove()


class TestBackwardHooks:
    """Test backward hook registration and execution."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_register_backward_hook_on_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering a backward hook on the proxy model."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register backward hook
        hook_called = []

        def backward_hook(module, grad_input, grad_output):
            hook_called.append(True)

        try:
            handle = proxy_model.register_full_backward_hook(backward_hook)
            assert handle is not None, "Should return hook handle"

            # Run forward and backward
            x = torch.randn(4, 10, requires_grad=True)
            output = proxy_model(x)
            loss = output.sum()
            loss.backward()

            # Check hook was called
            assert len(hook_called) > 0, "Backward hook should be called"

            handle.remove()

        except Exception as e:
            pytest.fail(f"Failed to register backward hook: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_backward_hook_gradient_clipping(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test using backward hook for gradient clipping."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register gradient clipping hook
        clipped_grads = []

        def clip_hook(module, grad_input, grad_output):
            # Capture gradient magnitudes
            for grad in grad_output:
                if grad is not None:
                    clipped_grads.append(grad.abs().max().item())

        handle = proxy_model.register_full_backward_hook(clip_hook)

        # Run forward and backward with large loss
        x = torch.randn(4, 10, requires_grad=True)
        output = proxy_model(x)
        loss = output.sum() * 1000  # Large multiplier to create large gradients
        loss.backward()

        # Check gradients were captured
        assert len(clipped_grads) > 0, "Should capture gradients"

        handle.remove()


class TestActivationCapture:
    """Test activation capture use case."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_capture_activations_from_all_layers(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test capturing activations from all layers."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        # Use empty include_patterns to not wrap any modules (original behavior when module_paths=None)
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, include_patterns=[])

        proxy_model = om.offload(model, config=config)

        # Activation capture
        activations = {}

        def make_capture_hook(name):
            def hook(module, input, output):  # noqa: A002
                activations[name] = output.detach().clone()

            return hook

        # Register hooks on all Linear layers
        handles = []
        for name, module in proxy_model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(make_capture_hook(name))
                handles.append(handle)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # Check activations were captured
        assert len(activations) > 0, "Should capture activations"
        assert "layer1" in activations, "Should capture layer1 activations"
        assert "layer2" in activations, "Should capture layer2 activations"
        assert "layer3" in activations, "Should capture layer3 activations"

        # Check shapes
        assert activations["layer1"].shape == torch.Size([4, 20]), "layer1 output should be (4, 20)"
        assert activations["layer2"].shape == torch.Size([4, 30]), "layer2 output should be (4, 30)"
        assert activations["layer3"].shape == torch.Size([4, 10]), "layer3 output should be (4, 10)"

        # Cleanup
        for handle in handles:
            handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_capture_specific_layer_activations(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test capturing activations from a specific layer."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Capture only layer2
        layer2_output = []

        def capture_layer2(module, input, output):  # noqa: A002
            layer2_output.append(output.detach().clone())

        handle = proxy_model.layer2.register_forward_hook(capture_layer2)

        # Run multiple forward passes
        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 10)

        with torch.no_grad():
            _ = proxy_model(x1)
            _ = proxy_model(x2)

        # Check we captured two outputs
        assert len(layer2_output) == 2, "Should capture output from both forward passes"

        handle.remove()


class TestGradientManipulation:
    """Test gradient manipulation through hooks."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_gradient_clipping_via_hook(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test gradient clipping through backward hook."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Test gradient clipping on a specific layer
        gradient_norms = []

        def clip_gradient_hook(module, grad_input, grad_output):
            for grad in grad_output:
                if grad is not None:
                    gradient_norms.append(grad.norm().item())

        # Register on specific layer
        handle = proxy_model.fc1.register_full_backward_hook(clip_gradient_hook)

        # Run training step
        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 10)

        output = proxy_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients were captured
        assert len(gradient_norms) > 0, "Should capture gradient norms"

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_gradient_monitoring(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test monitoring gradients through hooks."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        # Use empty include_patterns to not wrap any modules (original behavior when module_paths=None)
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, include_patterns=[])

        proxy_model = om.offload(model, config=config)

        # Monitor gradients for all layers
        gradient_stats = {}

        def make_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                stats = {}
                for i, grad in enumerate(grad_output):
                    if grad is not None:
                        stats[f"output_{i}"] = {
                            "mean": grad.mean().item(),
                            "std": grad.std().item(),
                            "max": grad.max().item(),
                            "min": grad.min().item(),
                        }
                gradient_stats[name] = stats

            return hook

        # Register hooks
        handles = []
        for name, module in proxy_model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_full_backward_hook(make_gradient_hook(name))
                handles.append(handle)

        # Training step
        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 10)

        output = proxy_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradient stats were collected
        assert len(gradient_stats) > 0, "Should collect gradient statistics"
        assert "layer1" in gradient_stats, "Should have stats for layer1"

        # Cleanup
        for handle in handles:
            handle.remove()


class TestMultipleHooks:
    """Test multiple hooks on the same module."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_multiple_forward_hooks(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering multiple forward hooks on the same module."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register multiple hooks
        hook1_called = []
        hook2_called = []
        hook3_called = []

        def hook1(module, input, output):  # noqa: A002
            hook1_called.append(True)

        def hook2(module, input, output):  # noqa: A002
            hook2_called.append(True)

        def hook3(module, input, output):  # noqa: A002
            hook3_called.append(True)

        handle1 = proxy_model.register_forward_hook(hook1)
        handle2 = proxy_model.register_forward_hook(hook2)
        handle3 = proxy_model.register_forward_hook(hook3)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # All hooks should be called
        assert len(hook1_called) > 0, "Hook 1 should be called"
        assert len(hook2_called) > 0, "Hook 2 should be called"
        assert len(hook3_called) > 0, "Hook 3 should be called"

        # Cleanup
        handle1.remove()
        handle2.remove()
        handle3.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hook_execution_order(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks execute in registration order."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Track execution order
        execution_order = []

        def make_hook(hook_id):
            def hook(module, input, output):  # noqa: A002
                execution_order.append(hook_id)

            return hook

        handle1 = proxy_model.register_forward_hook(make_hook(1))
        handle2 = proxy_model.register_forward_hook(make_hook(2))
        handle3 = proxy_model.register_forward_hook(make_hook(3))

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # Check execution order
        assert execution_order == [1, 2, 3], f"Expected [1, 2, 3], got {execution_order}"

        # Cleanup
        handle1.remove()
        handle2.remove()
        handle3.remove()


class TestHookRemoval:
    """Test hook removal."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_remove_forward_hook(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test removing a forward hook."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register and remove hook
        hook_called = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_called.append(True)

        handle = proxy_model.register_forward_hook(forward_hook)

        # First forward pass - hook should be called
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_called) == 1, "Hook should be called once"

        # Remove hook
        handle.remove()

        # Second forward pass - hook should NOT be called
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_called) == 1, "Hook should still be called only once (not after removal)"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_remove_one_of_multiple_hooks(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test removing one hook while others remain."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Register multiple hooks
        hook1_count = []
        hook2_count = []

        def hook1(module, input, output):  # noqa: A002
            hook1_count.append(1)

        def hook2(module, input, output):  # noqa: A002
            hook2_count.append(1)

        handle1 = proxy_model.register_forward_hook(hook1)
        handle2 = proxy_model.register_forward_hook(hook2)

        # First pass - both should be called
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook1_count) == 1, "Hook 1 should be called"
        assert len(hook2_count) == 1, "Hook 2 should be called"

        # Remove hook1
        handle1.remove()

        # Second pass - only hook2 should be called
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook1_count) == 1, "Hook 1 should not be called after removal"
        assert len(hook2_count) == 2, "Hook 2 should still be called"

        # Cleanup
        handle2.remove()


class TestHooksAcrossStateTransitions:
    """Test hook behavior across state transitions."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hooks_persist_across_state_transitions(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks registered on model modules persist across state transitions."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        # Different model instances for each state (simulating model swaps)
        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()

        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=2)

        # Offload the warmup model (initialize_warmup will return it)
        proxy_model = om.offload(warmup_model, config=config)

        # Register hook on the underlying model's fc1 layer (not the proxy)
        hook_calls = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls.append(om._current_phase.name)

        # Register on the actual module, not the proxy
        handle = proxy_model.fc1.register_forward_hook(forward_hook)

        # Run through state transitions
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)  # DISCOVERY (count=1, >= 1, transition after)
            _ = proxy_model(x)  # PROFILING (count=1)
            _ = proxy_model(x)  # PROFILING (count=2, >= 2, transition after)
            _ = proxy_model(x)  # INFERENCE
            _ = proxy_model(x)  # INFERENCE

        # With automatic hook transfer, hooks should persist across transitions
        assert len(hook_calls) == 5, f"Hook should be called 5 times, got {len(hook_calls)}"
        assert hook_calls == ["DISCOVERY", "PROFILING", "PROFILING", "INFERENCE", "INFERENCE"], (
            f"Hook should be called in all states, got {hook_calls}"
        )

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hooks_on_proxy_work_with_explicit_delegation(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks registered directly on proxy work with explicit delegation."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()

        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=2)

        proxy_model = om.offload(warmup_model, config=config)

        # Register hook on PROXY itself (now works with explicit delegation)
        hook_calls = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls.append(om._current_phase.name)

        handle = proxy_model.register_forward_hook(forward_hook)

        # Run forward passes through state transitions
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)  # DISCOVERY (count=1, >= 1, transition after)
            _ = proxy_model(x)  # PROFILING (count=1)
            _ = proxy_model(x)  # PROFILING (count=2, >= 2, transition after)
            _ = proxy_model(x)  # INFERENCE
            _ = proxy_model(x)  # INFERENCE

        # With explicit delegation and hook transfer, hooks should work and persist
        assert len(hook_calls) == 5, f"Hook should be called 5 times, got {len(hook_calls)}"
        assert hook_calls == ["DISCOVERY", "PROFILING", "PROFILING", "INFERENCE", "INFERENCE"], (
            f"Hook should be called in all states, got {hook_calls}"
        )

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_all_hook_types_on_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test all hook types (forward, forward_pre, backward, backward_pre) on proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Track hook calls
        hook_calls = {"forward_pre": [], "forward": [], "backward": []}

        def forward_pre_hook(module, input):  # noqa: A002
            hook_calls["forward_pre"].append("pre")
            return None

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls["forward"].append("forward")
            return None

        def backward_hook(module, grad_input, grad_output):
            hook_calls["backward"].append("backward")
            return None

        # Register all hook types on proxy
        handle_pre = proxy_model.register_forward_pre_hook(forward_pre_hook)
        handle_fwd = proxy_model.register_forward_hook(forward_hook)
        handle_bwd = proxy_model.register_full_backward_hook(backward_hook)

        # Run forward and backward pass
        x = torch.randn(4, 10, requires_grad=True)
        output = proxy_model(x)
        loss = output.sum()
        loss.backward()

        # Verify all hooks were called
        assert len(hook_calls["forward_pre"]) == 1, "Forward pre-hook should be called once"
        assert len(hook_calls["forward"]) == 1, "Forward hook should be called once"
        assert len(hook_calls["backward"]) == 1, "Backward hook should be called once"

        # Clean up
        handle_pre.remove()
        handle_fwd.remove()
        handle_bwd.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_backward_pre_hook_on_proxy_if_available(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test backward pre-hook on proxy (if available in PyTorch version)."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        proxy_model = om.offload(model, config=config)

        # Check if backward pre-hook is available in this PyTorch version
        if not hasattr(model, "register_full_backward_pre_hook"):
            pytest.skip("register_full_backward_pre_hook not available in this PyTorch version")

        hook_calls = []

        def backward_pre_hook(module, grad_output):
            hook_calls.append("backward_pre")
            return None

        # Register backward pre-hook on proxy
        handle = proxy_model.register_full_backward_pre_hook(backward_pre_hook)

        # Run forward and backward pass
        x = torch.randn(4, 10, requires_grad=True)
        output = proxy_model(x)
        loss = output.sum()
        loss.backward()

        # Verify hook was called
        assert len(hook_calls) == 1, "Backward pre-hook should be called once"

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hooks_persist_across_multiple_offload_calls(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks persist when offload() is called multiple times (re-initialization)."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        # Create different model instances for each offload call
        model1 = SimpleHookModel()
        model2 = SimpleHookModel()
        model2.load_state_dict(model1.state_dict())

        # First offload call uses model1
        mock_tensor_manager.initialize_warmup.return_value = model1
        mock_tensor_manager.initialize_profile.return_value = model1
        mock_tensor_manager.initialize_inference.return_value = model1

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0)

        # First offload call
        proxy_model = om.offload(model1, config=config)

        # Register hook on a submodule
        hook_calls = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls.append("called")

        handle = proxy_model.fc1.register_forward_hook(forward_hook)

        # Run forward pass - hook should work
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_calls) == 1, "Hook should be called once after first offload()"

        # Second offload call (re-initialization) uses model2
        mock_tensor_manager.initialize_warmup.return_value = model2

        # Call offload again - this triggers _transition_to_warmup with hook transfer
        proxy_model = om.offload(model2)

        # Run forward pass - hook should still work after re-initialization
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_calls) == 2, (
            f"Hook should be called twice (once before, once after re-offload), got {len(hook_calls)}"
        )

        handle.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
