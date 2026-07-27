# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadManager phase transitions and lifecycle.

This test suite validates the OffloadManager's state machine logic by mocking the
TensorManager and trap calls. It verifies:

1. State Transitions: Proper transitions through DISCOVERY -> PROFILING -> INFERENCE states
2. Iteration Counting: Correct tracking of iterations in each state
3. Automatic State Tracking: Via ManagedModelWrapper (new) or forward hooks (old)
4. Configuration: Different discovery and profiling iteration counts
5. Pattern Matching: Multiple module patterns for offloading

Key behaviors tested:
- State transitions occur AFTER completing N iterations (on the N+1th forward pass)
- Iteration counters reset to 0 when transitioning to a new state
- Automatic state transitions work without manual update_state() calls
- NoOpTensorManager is used when enabled=False

Test Compatibility:
- Most tests work with both old (hooks) and new (wrapper) implementations
- test_automatic_transitions_with_changing_model_objects specifically tests the
  bug that exists in the old version where hooks don't work when model objects
  change during state transitions

Example flow with discovery_iters=2, profiling_iters=3:
  Forward 0: DISCOVERY (count 0 -> 1)
  Forward 1: DISCOVERY (count 1 -> 2)
  Forward 2: DISCOVERY -> PROFILING transition (count reset to 0)
  Forward 3: PROFILING (count 0 -> 1)
  Forward 4: PROFILING (count 1 -> 2)
  Forward 5: PROFILING (count 2 -> 3)
  Forward 6: PROFILING -> INFERENCE transition (count reset to 0)
  Forward 7+: INFERENCE (no counting)
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.compile import COMPILED_EAGER_PROFILE_FORWARDS
from flextensor.offload_manager import (
    OffloadConfig,
    OffloadManager,
    OffloadModelProxy,
    OffloadPhase,
)


# Simple model for testing
class SubmoduleL2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class SubmoduleL1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule_l2 = SubmoduleL2()
        self.submodule_l3 = SubmoduleL2()

    def forward(self, x):
        for _i in range(5):
            x = self.submodule_l2(x)
        return self.submodule_l3(x)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule_l1 = SubmoduleL1()
        self.submodule_l2 = SubmoduleL2()
        self.module_list = torch.nn.ModuleList([SubmoduleL2() for _ in range(5)])

    def forward(self, x):
        x = self.submodule_l1(x)
        x = self.submodule_l2(x)
        for module in self.module_list:
            x = module(x)
        return x


class MockTrap:
    """Mock trap context manager for testing."""

    def __init__(self, name):
        self.name = name
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


class DelegatingWrapper(torch.nn.Module):
    """A wrapper that delegates to another model without registering it as a child.

    This simulates the pattern where forward hooks fail:
    - The wrapper's forward immediately calls another module
    - That other module is not a registered child (accessed via external reference)
    - PyTorch hooks don't fire because of this delegation pattern
    """

    def __init__(self, target_model):
        super().__init__()
        # Store model reference but DON'T register as child
        self._target = target_model

    def forward(self, x):
        # Direct delegation to external model - hooks won't fire here
        return self._target(x)


class TestMaxGpuMemResolution:
    """Tests that _initialize_tensor_manager passes max_gpu_mem_fraction to TensorManager.

    Budget resolution (fraction → bytes) now happens inside resolve_gpu_budget()
    at compute time, not at strategy construction time. These tests verify the fraction
    is forwarded correctly to TensorManager.
    """

    def test_fraction_passed_to_tensor_manager(self):
        """max_gpu_mem_fraction is forwarded to TensorManager."""
        config = OffloadConfig(max_gpu_mem_fraction=0.8)
        om = OffloadManager("test_fraction_resolved")
        om.set_config(config)

        with (
            patch("flextensor.offload_manager.AdaptiveStrategy"),
            patch("flextensor.tensor_manager.TensorManager") as mock_tm,
        ):
            om._initialize_tensor_manager()

        mock_tm.assert_called_once()
        _, kwargs = mock_tm.call_args
        assert kwargs["max_gpu_mem_fraction"] == pytest.approx(0.8)

    def test_fraction_none_passed_to_tensor_manager(self):
        """max_gpu_mem_fraction=None (latency mode) is forwarded to TensorManager."""
        config = OffloadConfig(max_gpu_mem_fraction=None)
        om = OffloadManager("test_fraction_none")
        om.set_config(config)

        with (
            patch("flextensor.offload_manager.AdaptiveStrategy"),
            patch("flextensor.tensor_manager.TensorManager") as mock_tm,
        ):
            om._initialize_tensor_manager()

        _, kwargs = mock_tm.call_args
        assert kwargs["max_gpu_mem_fraction"] is None

    def test_fraction_uses_correct_gpu_device(self):
        """gpu_device index is forwarded to TensorManager."""
        config = OffloadConfig(gpu_device=1, max_gpu_mem_fraction=0.5)
        om = OffloadManager("test_fraction_gpu_device")
        om.set_config(config)

        with (
            patch("flextensor.offload_manager.AdaptiveStrategy"),
            patch("flextensor.tensor_manager.TensorManager") as mock_tm,
        ):
            om._initialize_tensor_manager()

        _, kwargs = mock_tm.call_args
        assert kwargs["max_gpu_mem_fraction"] == pytest.approx(0.5)

    def test_profile_mode_default_forwarded(self):
        """profile_mode defaults to 'view' and is forwarded to TensorManager."""
        config = OffloadConfig()
        om = OffloadManager("test_profile_mode_default")
        om.set_config(config)

        with (
            patch("flextensor.offload_manager.AdaptiveStrategy"),
            patch("flextensor.tensor_manager.TensorManager") as mock_tm,
        ):
            om._initialize_tensor_manager()

        _, kwargs = mock_tm.call_args
        assert kwargs["profile_mode"] == "view"

    def test_profile_mode_view_forwarded(self):
        """profile_mode='view' is forwarded to TensorManager."""
        config = OffloadConfig(profile_mode="view")
        om = OffloadManager("test_profile_mode_view")
        om.set_config(config)

        with (
            patch("flextensor.offload_manager.AdaptiveStrategy"),
            patch("flextensor.tensor_manager.TensorManager") as mock_tm,
        ):
            om._initialize_tensor_manager()

        _, kwargs = mock_tm.call_args
        assert kwargs["profile_mode"] == "view"


class TestOffloadManagerStateMachine:
    """Test cases for OffloadManager state machine logic."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()

        # Create test input
        self.x = torch.randn(4, 10)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_phase_transitions_discovery_to_profiling_to_inference(
        self,
        mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test state transitions from NOT_INITIALIZED -> DISCOVERY -> PROFILING -> INFERENCE."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager

        # Mock trap to track calls
        trap_calls = []

        def mock_trap(name):
            trap = MockTrap(name)
            trap_calls.append(trap)
            return trap

        mock_tensor_manager.trap = mock_trap
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        # Configure offload manager with specific iteration counts
        discovery_iters = 2
        profiling_iters = 7

        om = OffloadManager("test")
        config = OffloadConfig(
            enabled=True,
            discovery_iters=discovery_iters,
            profiling_iters=profiling_iters,
        )

        # Offload the model
        config = config.model_copy(update={"include_patterns": ["submoduleL1.submoduleL2"]})
        model = om.offload(self.model, config=config)

        # Verify initial state
        assert om._current_phase == OffloadPhase.DISCOVERY
        assert om._iteration_count == 0

        # Run discovery iterations - wrapper automatically calls update_state()
        for _ in range(discovery_iters):
            with torch.no_grad():
                _ = model(self.x)  # Wrapper automatically calls update_state()

        # Should have transitioned to PROFILING after discovery_iters iterations
        assert om._current_phase == OffloadPhase.PROFILING
        assert om._iteration_count == 0  # Counter resets on transition

        # Verify transition to profile was called
        mock_tensor_manager.initialize_profile.assert_called_once()

        # Run profiling iterations
        for _ in range(profiling_iters):
            with torch.no_grad():
                _ = model(self.x)  # Wrapper automatically calls update_state()

        # Should have transitioned to INFERENCE after profiling_iters iterations
        assert om._current_phase == OffloadPhase.INFERENCE
        assert om._iteration_count == 0  # Counter resets on transition

        # Verify transition to inference was called
        mock_tensor_manager.initialize_inference.assert_called_once()

        # Run inference iterations (should stay in INFERENCE state)
        for _i in range(5):
            with torch.no_grad():
                _ = model(self.x)  # Wrapper automatically calls update_state()
            assert om._current_phase == OffloadPhase.INFERENCE

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_automatic_state_transitions_via_forward(
        self,
        mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test that state transitions happen automatically during forward passes.

        This test verifies that calling the model's forward method triggers
        automatic state transitions without requiring manual update_state() calls.

        Note: This test works with both implementations:
        - Old version: Uses forward hooks (works on simple models, may fail on complex delegation)
        - New version: Uses ManagedModelWrapper (works reliably in all cases)
        """
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        # Configure with minimal iterations
        discovery_iters = 1
        profiling_iters = 1

        om = OffloadManager("test")
        config = OffloadConfig(
            enabled=True,
            discovery_iters=discovery_iters,
            profiling_iters=profiling_iters,
        )

        # Offload the model
        config = config.model_copy(update={"include_patterns": ["submoduleL1.submoduleL2"]})
        model = om.offload(self.model, config=config)

        # Verify model is a torch.nn.Module (could be original or wrapper)
        assert isinstance(model, torch.nn.Module)

        # Initial state should be DISCOVERY
        assert om._current_phase == OffloadPhase.DISCOVERY
        assert om._iteration_count == 0

        # Run discovery iterations to trigger transition
        # State transitions should happen automatically during forward passes
        for _ in range(discovery_iters):
            with torch.no_grad():
                _ = model(self.x)

        # Should have transitioned to PROFILING
        assert om._current_phase == OffloadPhase.PROFILING
        assert om._iteration_count == 0

        # Run profiling iterations to trigger transition
        for _ in range(profiling_iters):
            with torch.no_grad():
                _ = model(self.x)

        # Should now be in INFERENCE
        assert om._current_phase == OffloadPhase.INFERENCE
        assert om._iteration_count == 0

        # Verify all transition methods were called
        mock_tensor_manager.initialize_warmup.assert_called_once()
        mock_tensor_manager.initialize_profile.assert_called_once()
        mock_tensor_manager.initialize_inference.assert_called_once()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_automatic_transitions_with_changing_model_objects(
        self,
        mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test automatic state transitions when model objects change during transitions.

        This test simulates the real failure scenario:
        - initialize_warmup() returns model object A
        - Hook is registered on model A
        - initialize_profile() returns model object B (different instance)
        - Hook on model A doesn't fire when calling model B
        - User is calling model B but hook is stuck on model A

        Expected behavior:
        - Old version (hooks): FAILS - hook on model A, but calling model B
        - New version (wrapper): PASSES - wrapper delegates to current model
        """
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)

        # Create THREE DIFFERENT model instances (as happens in real code)
        warmup_model = SimpleModel()
        profile_model = SimpleModel()  # Different instance!
        inference_model = SimpleModel()  # Different instance!

        # Each transition returns a DIFFERENT model object
        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        # Configure with minimal iterations
        discovery_iters = 1
        profiling_iters = 1

        om = OffloadManager("test_changing_models")
        config = OffloadConfig(
            enabled=True,
            discovery_iters=discovery_iters,
            profiling_iters=profiling_iters,
        )

        # Offload - KEY DIFFERENCE:
        # Old version: returns self.model (original), but hook is on warmup_model!
        # New version: returns wrapper that tracks om._model
        _original_model_id = id(self.model)
        config = config.model_copy(update={"include_patterns": ["submoduleL1.submoduleL2"]})
        returned_model = om.offload(self.model, config=config)

        # OLD VERSION BUG: returned_model is original, but hook is on warmup_model
        # This causes hooks to never fire because user calls returned_model
        # NEW VERSION FIX: returned_model is wrapper that always calls om._model

        # Initial state should be DISCOVERY
        assert om._current_phase == OffloadPhase.DISCOVERY
        assert om._iteration_count == 0

        # Run discovery iterations to trigger transition
        # After transition, om._model becomes profile_model
        # Old version: still calling warmup_model (hook fires)
        # New version: wrapper calls om._model (profile_model)
        for _ in range(discovery_iters):
            with torch.no_grad():
                _ = returned_model(self.x)

        # Should have transitioned to PROFILING
        # Old version: This PASSES because hook is still on warmup_model
        # But in REAL usage, user would call the internal model which changes!
        assert om._current_phase == OffloadPhase.PROFILING
        assert om._iteration_count == 0

        # Run profiling iterations to trigger transition
        # After transition, om._model becomes inference_model
        # Old version: still calling warmup_model (hook still fires)
        # New version: wrapper calls om._model (inference_model)
        for _ in range(profiling_iters):
            with torch.no_grad():
                _ = returned_model(self.x)

        # Should now be in INFERENCE
        assert om._current_phase == OffloadPhase.INFERENCE
        assert om._iteration_count == 0

        # Verify all transition methods were called
        mock_tensor_manager.initialize_warmup.assert_called_once()
        mock_tensor_manager.initialize_profile.assert_called_once()
        mock_tensor_manager.initialize_inference.assert_called_once()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_iteration_count_tracking(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that iteration counts are tracked correctly during state transitions."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        discovery_iters = 3
        profiling_iters = 5

        om = OffloadManager("test")
        config = OffloadConfig(
            enabled=True,
            discovery_iters=discovery_iters,
            profiling_iters=profiling_iters,
            include_patterns=["submoduleL1.submoduleL2"],
        )
        model = om.offload(self.model, config=config)

        # Track iteration count during discovery
        for _ in range(discovery_iters):
            with torch.no_grad():
                _ = model(self.x)  # Wrapper automatically calls update_state()

        # Should have transitioned to PROFILING
        assert om._current_phase == OffloadPhase.PROFILING
        assert om._iteration_count == 0

        # Track iteration count during profiling
        for _ in range(profiling_iters):
            with torch.no_grad():
                _ = model(self.x)  # Wrapper automatically calls update_state()

        # Should have transitioned to INFERENCE
        assert om._current_phase == OffloadPhase.INFERENCE
        assert om._iteration_count == 0

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_custom_discovery_and_profiling_iterations(
        self,
        mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test with custom discovery and profiling iteration counts."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        # Test different configurations
        test_cases = [
            (1, 1),
            (2, 7),
            (5, 10),
            (10, 5),
        ]

        for discovery_iters, profiling_iters in test_cases:
            om = OffloadManager(f"test_{discovery_iters}_{profiling_iters}")
            config = OffloadConfig(
                enabled=True,
                discovery_iters=discovery_iters,
                profiling_iters=profiling_iters,
                include_patterns=["submoduleL1.submoduleL2"],
            )
            model = om.offload(self.model, config=config)

            # Run discovery iterations to trigger transition
            for _ in range(discovery_iters):
                with torch.no_grad():
                    _ = model(self.x)  # Wrapper automatically calls update_state()

            # Should have transitioned to PROFILING
            assert om._current_phase == OffloadPhase.PROFILING

            # Run profiling iterations to trigger transition
            for _ in range(profiling_iters):
                with torch.no_grad():
                    _ = model(self.x)  # Wrapper automatically calls update_state()

            # Should have transitioned to INFERENCE
            assert om._current_phase == OffloadPhase.INFERENCE

            # Verify initialize methods were called
            mock_tensor_manager.initialize_warmup.assert_called()
            mock_tensor_manager.initialize_profile.assert_called()
            mock_tensor_manager.initialize_inference.assert_called()

            # Reset mocks for next test case
            mock_tensor_manager.reset_mock()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_offload_block_returns_trap(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that offload_block returns the correct trap object."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager

        mock_trap = MockTrap("test_trap")
        mock_tensor_manager.trap.return_value = mock_trap
        mock_tensor_manager.initialize_warmup.return_value = self.model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, include_patterns=["submoduleL1.submoduleL2"])
        om.offload(self.model, config=config)

        # Get trap from offload_block
        trap = om.offload_block("test_block")

        # Verify it's the mock trap
        assert trap is mock_trap
        mock_tensor_manager.trap.assert_called_with("test_block")

        # Test context manager usage
        with trap:
            assert trap.entered is True
        assert trap.exited is True

    @patch("flextensor.offload_manager.NoOpTensorManager")
    def test_offload_disabled_uses_noop_manager(self, mock_noop_manager_cls):
        """Test that when enabled=False, NoOpTensorManager is used."""
        # Setup mock
        mock_noop_manager = MagicMock()
        mock_noop_manager_cls.return_value = mock_noop_manager
        mock_noop_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_noop_manager.initialize_warmup.return_value = self.model
        mock_noop_manager.initialize_profile.return_value = self.model
        mock_noop_manager.initialize_inference.return_value = self.model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=False, include_patterns=["submoduleL1.submoduleL2"])
        model = om.offload(self.model, config=config)

        # Verify NoOpTensorManager was created
        mock_noop_manager_cls.assert_called_once()

        # Run some iterations - should not transition states normally
        for _i in range(5):
            with torch.no_grad():
                _ = model(self.x)

        # Should still be in discovery or have transitioned minimally
        # (NoOp manager doesn't do real state management)
        assert om._tensor_manager is mock_noop_manager

    def test_offload_disabled_with_real_noop_manager(self):
        """`offload(enabled=False)` must work without mocking `NoOpTensorManager`.

        Regression guard: ``OffloadManager.offload()`` calls
        ``self._tensor_manager.build_parameters_mapping(model)`` unconditionally,
        so the real ``NoOpTensorManager`` needs that method. Other tests patch
        ``NoOpTensorManager`` with a ``MagicMock`` and would mask a missing
        attribute.
        """
        om = OffloadManager("test_noop_real")
        config = OffloadConfig(enabled=False, include_patterns=["submodule_l1.submodule_l2"])

        model = om.offload(self.model, config=config)

        assert om._tensor_manager is not None
        assert om._tensor_manager.__class__.__name__ == "NoOpTensorManager"
        with torch.no_grad():
            _ = model(self.x)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_multiple_offload_patterns(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test offloading multiple module patterns."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model

        om = OffloadManager("test")

        # Offload multiple patterns via config
        patterns = [
            "submoduleL1.submoduleL2",
            "submoduleL1.submoduleL3",
            "submoduleL2",
            "module_list.*",
        ]
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=patterns)
        om.offload(self.model, config=config)

        # Count how many modules were patched
        patched_module_count = 0
        for _name, module in self.model.named_modules():
            if hasattr(module, "_ft_original_forward_func"):
                patched_module_count += 1

        # Should have patched the matching modules
        assert patched_module_count > 0

    def test_model_none_forward_raises_runtime_error(self):
        """Test that calling proxy.forward() explicitly with None model raises RuntimeError."""
        om = OffloadManager("test_none_forward")
        # Create a dummy model for the proxy (but offload_manager.model will be None)
        dummy_model = SimpleModel()
        proxy = OffloadModelProxy(dummy_model, om)

        # Model is None because offload() was never called
        assert om.model is None

        # Calling forward() explicitly should raise RuntimeError
        with pytest.raises(RuntimeError, match="Model not initialized"):
            proxy.forward(self.x)

    def test_model_none_getattr_raises_attribute_error(self):
        """Test that proxy delegates to ObjectWrapper when offload_manager.model is None."""
        om = OffloadManager("test_none_getattr")
        # Create a dummy model for the proxy (but offload_manager.model will be None)
        dummy_model = SimpleModel()
        proxy = OffloadModelProxy(dummy_model, om)

        # Model is None because offload() was never called
        assert om.model is None

        # Attribute access works via ObjectWrapper (delegates to dummy_model)
        assert hasattr(proxy, "submodule_l1")

        # But calling forward explicitly should raise RuntimeError
        with pytest.raises(RuntimeError, match="Model not initialized"):
            proxy.forward(self.x)

    def test_model_none_after_release(self):
        """Test that model becomes None after release() and raises appropriate errors."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.release_memory = MagicMock()

        om = OffloadManager("test_release")
        om._tensor_manager = mock_tensor_manager
        om._model = self.model

        # Create proxy with the model
        proxy = OffloadModelProxy(self.model, om)

        # Model should work before release
        assert om.model is not None

        # Release the manager
        om.release()

        # Model should be None after release
        assert om.model is None

        # Calling forward explicitly should raise RuntimeError
        with pytest.raises(RuntimeError, match="Model not initialized"):
            proxy.forward(self.x)

    @patch("flextensor.tensor_manager.TensorManager")
    def test_offload_uses_default_wildcard_pattern(self, mock_tensor_manager_cls):
        """Test that offload uses default ['*'] pattern when not specified in config."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True)

        # Verify default pattern is ["*"]
        assert config.include_patterns == ["*"]

        # Offload without specifying patterns - should use default
        model = om.offload(self.model, config=config)

        # Verify model was wrapped
        assert isinstance(model, OffloadModelProxy)
        assert om._current_phase == OffloadPhase.DISCOVERY

    @patch("flextensor.tensor_manager.TensorManager")
    def test_offload_with_custom_patterns_in_config(self, mock_tensor_manager_cls):
        """Test that offload uses custom patterns from config."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model

        om = OffloadManager("test")
        config = OffloadConfig(
            enabled=True,
            include_patterns=["submoduleL1", "submoduleL2"],
        )

        model = om.offload(self.model, config=config)

        # Verify model was wrapped
        assert isinstance(model, OffloadModelProxy)
        assert om._current_phase == OffloadPhase.DISCOVERY


class TestOffloadManagerConfig:
    """Test cases for OffloadConfig and configuration management."""

    def test_config_pre_inference_iters_property(self):
        """Test that pre_inference_iters property returns sum of discovery and profiling iters."""
        config = OffloadConfig(discovery_iters=3, profiling_iters=7)
        assert config.pre_inference_iters == 10  # 3 + 7

        config = OffloadConfig(discovery_iters=1, profiling_iters=10)
        assert config.pre_inference_iters == 11  # 1 + 10

        config = OffloadConfig(discovery_iters=5, profiling_iters=5)
        assert config.pre_inference_iters == 10  # 5 + 5

    def test_config_defaults(self):
        """Test default configuration values."""
        config = OffloadConfig()

        assert config.enabled is True
        assert config.gpu_device == 0
        assert config.discovery_iters == 1
        assert config.profiling_iters == 10
        assert config.pinned_memory is True
        assert config.shm_enabled is False

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = OffloadConfig(
            enabled=False,
            gpu_device=1,
            discovery_iters=5,
            profiling_iters=15,
        )

        assert config.enabled is False
        assert config.gpu_device == 1
        assert config.discovery_iters == 5
        assert config.profiling_iters == 15
        assert config.pre_inference_iters == 20  # 5 + 15


class TestEagerProfilingBudget:
    """Single profiling_iters knob: eager seed (compile path) + compiled measure window."""

    def _manager(self, *, activated_compiled, profiling_iters=10, replan=True):
        om = OffloadManager("test_eager_budget")
        om.config = OffloadConfig(enabled=True, profiling_iters=profiling_iters)
        om._compiled.active = activated_compiled
        om._compiled.replan_active = replan if activated_compiled else False
        return om

    def test_full_budget_when_not_compile_path(self):
        """Non-compile path uses the full profiling_iters (no seed cap)."""
        om = self._manager(activated_compiled=False, profiling_iters=10)
        assert om._eager_profiling_iters() == 10

    def test_reduced_budget_on_compile_path_with_replan(self):
        """Compile path + replan caps the eager budget to the fixed seed constant."""
        om = self._manager(activated_compiled=True, profiling_iters=10)
        assert om._eager_profiling_iters() == COMPILED_EAGER_PROFILE_FORWARDS

    def test_full_budget_on_compile_path_when_replan_disabled(self):
        """With replan off the eager plan is final, so keep full profiling_iters."""
        om = self._manager(activated_compiled=True, profiling_iters=10, replan=False)
        assert om._eager_profiling_iters() == 10

    def test_eager_seed_is_fixed_regardless_of_profiling_iters(self):
        """Eager seed is a fixed constant, independent of profiling_iters (no clamp)."""
        om_low = self._manager(activated_compiled=True, profiling_iters=1)
        assert om_low._eager_profiling_iters() == COMPILED_EAGER_PROFILE_FORWARDS
        om_high = self._manager(activated_compiled=True, profiling_iters=50)
        assert om_high._eager_profiling_iters() == COMPILED_EAGER_PROFILE_FORWARDS

    def test_compiled_measure_window_equals_profiling_iters(self):
        """profiling_iters sizes the compiled measure window directly (no floor)."""
        om_low = self._manager(activated_compiled=True, profiling_iters=5)
        assert om_low._compiled.measure_forwards() == 5
        om_high = self._manager(activated_compiled=True, profiling_iters=25)
        assert om_high._compiled.measure_forwards() == 25

    def test_manager_iters_before_inference_plain_path(self):
        """Plain path: discovery_iters + profiling_iters (matches config)."""
        om = OffloadManager("test_iters_plain")
        om.config = OffloadConfig(enabled=True, discovery_iters=2, profiling_iters=7)
        om._compiled.active = False
        om._compiled.replan_active = False
        assert om.iters_before_inference == 9
        assert om.iters_before_inference == om.config.pre_inference_iters

    def test_manager_iters_before_inference_compile_path(self):
        """Compile path without compile_fn: eager seed only (no replan tail in count)."""
        om = OffloadManager("test_iters_compile")
        om.config = OffloadConfig(enabled=True, discovery_iters=2, profiling_iters=50)
        om._compiled.active = True
        om._compiled.replan_active = True
        assert om.iters_before_inference == 2 + COMPILED_EAGER_PROFILE_FORWARDS
        assert om.iters_before_inference != om.config.pre_inference_iters

    def test_manager_iters_before_inference_compile_fn_view_no_replan_tail(self):
        """Default compile_fn + view: warmup + full profile budget, no post-INFERENCE replan count."""
        from flextensor.compile.lifecycle import PROFILE_COMPILE_WARMUP_FORWARDS

        om = OffloadManager("test_iters_compile_fn_view")
        om.config = OffloadConfig(enabled=True, discovery_iters=2, profiling_iters=10, profile_mode="view")
        om._compiled.active = True
        om._compiled.replan_active = False
        om._compiled.profile_active = True
        om._compiled.compile_fn = lambda m: m
        assert om.iters_before_inference == 2 + PROFILE_COMPILE_WARMUP_FORWARDS + 10

    def test_manager_iters_before_inference_compile_fn_non_view_uses_eager_seed(self):
        """compile_fn + non-view: eager seed only; replan measure is post-INFERENCE."""
        om = OffloadManager("test_iters_compile_fn_getter")
        om.config = OffloadConfig(enabled=True, discovery_iters=2, profiling_iters=10, profile_mode="getter")
        om._compiled.active = True
        om._compiled.replan_active = True
        om._compiled.profile_active = False
        om._compiled.compile_fn = lambda m: m
        assert om.iters_before_inference == 2 + COMPILED_EAGER_PROFILE_FORWARDS

    def test_update_state_transitions_at_reduced_budget(self):
        """PROFILING -> INFERENCE fires after the seed, not full profiling_iters."""
        om = self._manager(activated_compiled=True, profiling_iters=10)
        om._current_phase = OffloadPhase.PROFILING
        om._iteration_count = 0
        mock_tm = MagicMock()
        mock_tm.is_profiling_suspended.return_value = False
        om._tensor_manager = mock_tm

        with (
            patch.object(om, "_transition_to_inference") as mock_transition,
        ):
            for _ in range(COMPILED_EAGER_PROFILE_FORWARDS - 1):
                om.update_state()
            mock_transition.assert_not_called()
            om.update_state()  # forward that hits the seed budget
            mock_transition.assert_called_once()
