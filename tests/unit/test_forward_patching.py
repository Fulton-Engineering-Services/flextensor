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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1", "layer2"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1", "layer2"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layer1", "layer2"])

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


# =============================================================================
# Nested model fixtures for ancestor guard tests
# =============================================================================


class AttentionBlock(nn.Module):
    """Attention block with sub-projections."""

    def __init__(self, dim: int = 10):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.q_proj(x) + self.k_proj(x)


class TransformerLayer(nn.Module):
    """Transformer layer with attention sub-module."""

    def __init__(self, dim: int = 10):
        super().__init__()
        self.attn = AttentionBlock(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.attn(x) + x)


class NestedModel(nn.Module):
    """Model with nested layers for ancestor guard testing."""

    def __init__(self, num_layers: int = 2, dim: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# =============================================================================
# Ancestor guard tests
# =============================================================================


class TestAncestorGuard:
    """Test cases for the ancestor guard in _offload_modules."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_ancestor_guard_skips_nested_modules(self):
        """Overlapping patterns only patch the outermost module."""
        model = NestedModel()
        om = OffloadManager("test_ancestor_skip")
        config = OffloadConfig(
            enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layers.*", "layers.*.attn"]
        )

        om.offload(model, config=config)

        # layers.0 and layers.1 should be patched (they are offload units)
        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")

        # layers.0.attn and layers.1.attn should NOT be patched (ancestor already patched)
        assert not hasattr(model.layers[0].attn, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].attn, "_ft_original_forward_func")

        # Only 2 modules should be patched total
        assert len(om._patched_modules) == 2

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_ancestor_guard_non_overlapping_patterns(self):
        """Non-overlapping patterns patch all matches."""
        model = NestedModel()
        om = OffloadManager("test_ancestor_non_overlap")
        config = OffloadConfig(
            enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layers.*.attn", "head"]
        )

        om.offload(model, config=config)

        # attn modules should be patched (no ancestor conflict)
        assert hasattr(model.layers[0].attn, "_ft_original_forward_func")
        assert hasattr(model.layers[1].attn, "_ft_original_forward_func")
        # head should be patched
        assert hasattr(model.head, "_ft_original_forward_func")

        # layers themselves should NOT be patched (not in patterns)
        assert not hasattr(model.layers[0], "_ft_original_forward_func")
        assert not hasattr(model.layers[1], "_ft_original_forward_func")

        assert len(om._patched_modules) == 3

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_ancestor_guard_deeply_nested(self):
        """Deeply nested patterns are skipped when ancestor is patched."""
        model = NestedModel()
        om = OffloadManager("test_ancestor_deep")
        config = OffloadConfig(
            enabled=True, warmup_iters=1, profile_iters=1, include_patterns=["layers.*", "layers.*.attn.q_proj"]
        )

        om.offload(model, config=config)

        # layers.0 and layers.1 should be patched
        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")

        # q_proj should NOT be patched (ancestor layers.0 is patched)
        assert not hasattr(model.layers[0].attn.q_proj, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].attn.q_proj, "_ft_original_forward_func")

        assert len(om._patched_modules) == 2

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_ancestor_guard_with_exclude_interaction(self):
        """Exclude on a never-patched descendant is a no-op at module level."""
        model = NestedModel()
        om = OffloadManager("test_ancestor_exclude")
        config = OffloadConfig(
            enabled=True,
            warmup_iters=1,
            profile_iters=1,
            include_patterns=["layers.*"],
            exclude_patterns=["layers.*.attn"],
        )

        om.offload(model, config=config)

        # layers.0 and layers.1 should be patched (they are offload units)
        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")

        # attn was never patched, so exclude is a no-op at module level
        assert not hasattr(model.layers[0].attn, "_ft_original_forward_func")

        assert len(om._patched_modules) == 2

        om.release()

    def test_exclude_modules_double_call_no_error(self):
        """Calling _exclude_modules twice does not raise ValueError on remove."""
        model = NestedModel()
        om = OffloadManager("test_exclude_double")

        # Simulate patching layers.0 (must set both attrs that _restore_module_forward deletes)
        model.layers[0]._ft_original_forward_func = type(model.layers[0]).forward
        model.layers[0]._ft_offload_name = "TransformerLayer.0"
        om._patched_modules.append(model.layers[0])

        # First call un-patches and removes from _patched_modules
        om._exclude_modules(model, ["layers.0"])
        assert model.layers[0] not in om._patched_modules

        # Second call should be a no-op (no ValueError)
        om._exclude_modules(model, ["layers.0"])
        assert model.layers[0] not in om._patched_modules

    def test_has_patched_ancestor_directly(self):
        """Unit test _has_patched_ancestor helper without CUDA."""
        model = NestedModel()
        om = OffloadManager("test_has_patched_ancestor")

        # Nothing patched yet
        assert not om._has_patched_ancestor(model, "layers.0.attn")
        assert not om._has_patched_ancestor(model, "layers.0")

        # Simulate patching layers.0
        model.layers[0]._ft_original_forward_func = type(model.layers[0]).forward

        # Now layers.0.attn should detect its ancestor is patched
        assert om._has_patched_ancestor(model, "layers.0.attn")
        assert om._has_patched_ancestor(model, "layers.0.attn.q_proj")

        # layers.0 itself should not (we only check strict ancestors)
        assert not om._has_patched_ancestor(model, "layers.0")

        # layers.1 and its descendants should not be affected
        assert not om._has_patched_ancestor(model, "layers.1.attn")

        # Clean up
        delattr(model.layers[0], "_ft_original_forward_func")


class TestPatternMatchWarnings:
    """Test that unmatched include/exclude patterns emit warnings."""

    def test_include_pattern_no_match_warns(self, caplog):
        """An include pattern that matches nothing logs a warning."""
        model = NestedModel()
        om = OffloadManager("test_include_warn")

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._offload_modules(model, ["nonexistent_module"])

        assert any("Include pattern 'nonexistent_module' did not match any modules" in m for m in caplog.messages)

    def test_include_pattern_match_no_warning(self, caplog):
        """An include pattern that matches does not log a warning."""
        model = NestedModel()
        om = OffloadManager("test_include_no_warn")

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._offload_modules(model, ["layers.*"])

        assert not any("did not match" in m for m in caplog.messages)
        om.release()

    def test_exclude_pattern_no_match_warns(self, caplog):
        """An exclude pattern that matches nothing logs a warning."""
        model = NestedModel()
        om = OffloadManager("test_exclude_warn")

        # Simulate patching layers.0
        model.layers[0]._ft_original_forward_func = type(model.layers[0]).forward
        model.layers[0]._ft_offload_name = "TransformerLayer.0"
        om._patched_modules.append(model.layers[0])

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._exclude_modules(model, ["nonexistent_module"])

        assert any(
            "Exclude pattern 'nonexistent_module' did not match any modules or parameters" in m for m in caplog.messages
        )

    def test_exclude_pattern_match_no_warning(self, caplog):
        """An exclude pattern that matches does not log a warning."""
        model = NestedModel()
        om = OffloadManager("test_exclude_no_warn")

        # Simulate patching layers.0
        model.layers[0]._ft_original_forward_func = type(model.layers[0]).forward
        model.layers[0]._ft_offload_name = "TransformerLayer.0"
        om._patched_modules.append(model.layers[0])

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._exclude_modules(model, ["layers.0"])

        assert not any("did not match" in m for m in caplog.messages)

    def test_exclude_parameter_pattern_no_false_warning(self, caplog):
        """An exclude pattern targeting parameters (not modules) does not falsely warn."""
        model = NestedModel()
        om = OffloadManager("test_exclude_param_no_warn")

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._exclude_modules(model, ["*.weight"])

        assert not any("did not match" in m for m in caplog.messages)

    def test_mixed_patterns_warns_only_unmatched(self, caplog):
        """When some patterns match and others don't, only unmatched ones warn."""
        model = NestedModel()
        om = OffloadManager("test_mixed_warn")

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._offload_modules(model, ["layers.*", "bogus_layer"])

        assert any("Include pattern 'bogus_layer' did not match any modules" in m for m in caplog.messages)
        assert not any("layers.*" in m for m in caplog.messages)
        om.release()


class TestAggregatePatternWarnings:
    """Test aggregate error/warning signals when offloading is effectively disabled."""

    def test_no_include_matches_logs_error(self, caplog):
        """All include patterns failing to match emits an ERROR log."""
        model = NestedModel()
        om = OffloadManager("test_agg_include")
        om.config = OffloadConfig(include_patterns=["bogus", "also_bogus"])

        with caplog.at_level("ERROR", logger="flextensor.offload_manager"):
            om._offload_modules(model, om.config.include_patterns)
            om._check_no_modules_patched()

        assert any("Offloading is effectively disabled" in m for m in caplog.messages)

    def test_no_error_when_modules_match(self, caplog):
        """No aggregate error when at least one module is patched."""
        model = NestedModel()
        om = OffloadManager("test_agg_ok")
        om.config = OffloadConfig(include_patterns=["layers.*"])

        with caplog.at_level("ERROR", logger="flextensor.offload_manager"):
            om._offload_modules(model, om.config.include_patterns)
            om._check_no_modules_patched()

        assert not any("Offloading is effectively disabled" in m for m in caplog.messages)
        om.release()

    def test_default_wildcard_still_errors_if_nothing_patched(self, caplog):
        """Default ['*'] with no patched modules still emits the aggregate error."""
        om = OffloadManager("test_agg_wildcard")
        om.config = OffloadConfig(include_patterns=["*"])

        with caplog.at_level("ERROR", logger="flextensor.offload_manager"):
            om._check_no_modules_patched()

        assert any("Offloading is effectively disabled" in m for m in caplog.messages)

    def test_all_excluded_logs_error(self, caplog):
        """Excluding all included modules emits an ERROR log."""
        model = NestedModel()
        om = OffloadManager("test_agg_exclude")

        # Simulate patching layers.0 and layers.1
        for i in range(2):
            layer = model.layers[i]
            layer._ft_original_forward_func = type(layer).forward
            layer._ft_offload_name = f"TransformerLayer.{i}"
            layer.__dict__["forward"] = lambda self, x: x  # noqa: ARG005
            om._patched_modules.append(layer)

        with caplog.at_level("ERROR", logger="flextensor.offload_manager"):
            om._exclude_modules(model, ["layers.*"])

        assert any("All 2 included modules were removed by exclude_patterns" in m for m in caplog.messages)

    def test_partial_exclude_no_error(self, caplog):
        """Excluding some (but not all) modules does not emit an aggregate error."""
        model = NestedModel()
        om = OffloadManager("test_partial_exclude")

        # Simulate patching layers.0 and layers.1
        for i in range(2):
            layer = model.layers[i]
            layer._ft_original_forward_func = type(layer).forward
            layer._ft_offload_name = f"TransformerLayer.{i}"
            layer.__dict__["forward"] = lambda self, x: x  # noqa: ARG005
            om._patched_modules.append(layer)

        with caplog.at_level("ERROR", logger="flextensor.offload_manager"):
            om._exclude_modules(model, ["layers.0"])

        assert not any("All" in m and "removed by exclude_patterns" in m for m in caplog.messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
