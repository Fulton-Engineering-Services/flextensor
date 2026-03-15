# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for forward method patching approach.

This test suite validates that forward method patching preserves model
hierarchy, isinstance checks, and supports proper cleanup.
"""

import pytest
import torch
from torch import nn

from flextensor.config import OffloadConfig
from flextensor.offload_manager import OffloadManager


class SimpleLayer(nn.Module):
    """A simple layer for testing."""

    def __init__(self, in_features: int = 10, out_features: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.custom_attr = "test_value"

    def forward(self, x):
        return self.linear(x)


class ModelWithLayers(nn.Module):
    """Model with multiple layers for testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = SimpleLayer()
        self.layer2 = SimpleLayer()
        self.layer3 = SimpleLayer()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class TestForwardPatching:
    """Test cases for forward method patching."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_preserves_isinstance(self):
        """Test that isinstance checks still work after patching."""
        model = ModelWithLayers()
        om = OffloadManager("test_isinstance")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1"])

        # Verify isinstance works before patching
        assert isinstance(model.layer1, SimpleLayer)
        assert isinstance(model.layer1, nn.Module)

        # Offload and patch
        om.offload(model, config=config)

        # isinstance should still work after patching
        assert isinstance(model.layer1, SimpleLayer)
        assert isinstance(model.layer1, nn.Module)

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_preserves_hierarchy(self):
        """Test that model hierarchy is preserved after patching."""
        model = ModelWithLayers()
        om = OffloadManager("test_hierarchy")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1"])

        # Get references before patching
        layer1_before = model.layer1
        layer2_before = model.layer2

        # Offload and patch
        om.offload(model, config=config)

        # Hierarchy should be preserved - same objects
        assert model.layer1 is layer1_before
        assert model.layer2 is layer2_before

        # Can still access nested attributes
        assert hasattr(model.layer1, "linear")
        assert isinstance(model.layer1.linear, nn.Linear)

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_release_restores_original_forward(self):
        """Test that release() restores original forward methods."""
        model = ModelWithLayers()
        om = OffloadManager("test_restore")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1"])

        # Save original forward
        original_forward = model.layer1.forward

        # Offload and patch
        om.offload(model, config=config)

        # Forward should be patched
        assert model.layer1.forward != original_forward
        assert hasattr(model.layer1, "_ft_original_forward_func")
        # _ft_original_forward_func stores the unbound class method
        assert model.layer1._ft_original_forward_func == type(model.layer1).forward  # type: ignore[attr-defined]

        # Release and check restoration
        om.release()

        # Forward should be restored (class method is accessible again)
        assert model.layer1.forward.__func__ == type(model.layer1).forward  # type: ignore[attr-defined]
        assert not hasattr(model.layer1, "_ft_original_forward_func")
        assert not hasattr(model.layer1, "_ft_offload_name")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_state_dict(self):
        """Test that state_dict() works correctly with patched modules."""
        model = ModelWithLayers()
        om = OffloadManager("test_state_dict")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1", "layer2"])

        # Get state_dict before patching
        state_dict_before = model.state_dict()

        # Offload and patch
        om.offload(model, config=config)

        # State dict should be identical (no wrapper in hierarchy)
        state_dict_after = model.state_dict()
        assert set(state_dict_before.keys()) == set(state_dict_after.keys())

        # Values should match
        for key in state_dict_before:
            assert torch.equal(state_dict_before[key], state_dict_after[key])

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_custom_attributes(self):
        """Test that custom attributes are still accessible after patching."""
        model = ModelWithLayers()
        om = OffloadManager("test_attributes")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1"])

        # Offload and patch
        om.offload(model, config=config)

        # Custom attributes should be accessible
        assert model.layer1.custom_attr == "test_value"
        assert hasattr(model.layer1, "linear")
        assert isinstance(model.layer1.linear, nn.Linear)

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_multiple_patches_same_module(self):
        """Test that patching the same module twice is handled correctly."""
        model = ModelWithLayers()
        om = OffloadManager("test_double_patch")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1"])

        # Offload and patch
        om.offload(model, config=config)

        # Try patching again (should be skipped)
        om._patch_module_forward(model.layer1, "test_name")

        # Should still have original stored (unbound class method)
        assert hasattr(model.layer1, "_ft_original_forward_func")
        assert model.layer1._ft_original_forward_func == type(model.layer1).forward  # type: ignore[attr-defined]

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_named_modules(self):
        """Test that named_modules() works correctly with patched modules."""
        model = ModelWithLayers()
        om = OffloadManager("test_named_modules")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1", "layer2"])

        # Get named modules before patching
        named_modules_before = dict(model.named_modules())

        # Offload and patch
        om.offload(model, config=config)

        # named_modules should return same structure
        named_modules_after = dict(model.named_modules())
        assert set(named_modules_before.keys()) == set(named_modules_after.keys())

        # Modules should be the same objects
        for name in named_modules_before:
            assert named_modules_before[name] is named_modules_after[name]

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_forward_callable(self):
        """Test that patched forward is callable and wraps the original."""
        model = ModelWithLayers()
        om = OffloadManager("test_callable")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1"])

        # Save original forward
        original_forward = model.layer1.forward

        # Offload and patch
        om.offload(model, config=config)

        # Forward should be patched and callable
        assert model.layer1.forward != original_forward
        assert callable(model.layer1.forward)

        # Should have the original stored (unbound class method)
        assert hasattr(model.layer1, "_ft_original_forward_func")
        assert model.layer1._ft_original_forward_func == type(model.layer1).forward  # type: ignore[attr-defined]

        # Forward method should have proper name (from functools.wraps)
        assert model.layer1.forward.__name__ == original_forward.__name__

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_modules_tracking(self):
        """Test that patched modules are tracked correctly."""
        model = ModelWithLayers()
        om = OffloadManager("test_tracking")
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, module_patterns=["layer1", "layer2"])

        # Initially no patched modules
        assert len(om._patched_modules) == 0

        # Offload and patch
        om.offload(model, config=config)

        # Should have tracked patched modules
        assert len(om._patched_modules) == 2
        assert model.layer1 in om._patched_modules
        assert model.layer2 in om._patched_modules

        # Release should clear tracking
        om.release()
        assert len(om._patched_modules) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
