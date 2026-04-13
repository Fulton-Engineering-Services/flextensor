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
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1", "layer2"]
        )

        # Get state_dict before patching
        state_dict_before = model.state_dict()

        # Offload and patch
        om.offload(model, config=config)

        # State dict should be identical (no wrapper in hierarchy)
        state_dict_after = model.state_dict()
        assert set(state_dict_before.keys()) == set(state_dict_after.keys())

        # Values should match (offload may move tensors to CUDA, so compare on CPU)
        for key in state_dict_before:
            assert torch.equal(state_dict_before[key].cpu(), state_dict_after[key].cpu())

        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_patched_module_custom_attributes(self):
        """Test that custom attributes are still accessible after patching."""
        model = ModelWithLayers()
        om = OffloadManager("test_attributes")
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1", "layer2"]
        )

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
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1"])

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
        config = OffloadConfig(
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layer1", "layer2"]
        )

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


class WrappedNestedModel(nn.Module):
    """Wrapper-shaped model (e.g., model.layers.0.attn...) for regression testing."""

    def __init__(self, num_layers: int = 2, dim: int = 10):
        super().__init__()
        self.model = NestedModel(num_layers, dim)

    def forward(self, x):
        return self.model(x)


class LayerWithDirectParam(nn.Module):
    """Layer with both a direct parameter and a child module owning parameters."""

    def __init__(self, dim: int = 10):
        super().__init__()
        self.x = nn.Parameter(torch.randn(dim))
        self.linear = nn.Linear(dim, dim)

    def forward(self, inp):
        return self.linear(inp) + self.x


class ModelWithDirectParams(nn.Module):
    """Model whose layers have both direct parameters and child modules."""

    def __init__(self, num_layers: int = 2, dim: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([LayerWithDirectParam(dim) for _ in range(num_layers)])

    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
        return inp


class LinearWithScale(nn.Module):
    """Linear layer mimicking the DeepSeek-V3 inner-field pattern.

    ``self.weight.scale = self.scale = nn.Parameter(...)`` creates a tensor
    attribute on weight that aliases a module parameter.
    """

    def __init__(self, dim: int = 10):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.weight.scale = self.scale = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class ModelWithInnerFieldLayers(nn.Module):
    """Model with indexed LinearWithScale layers (DeepSeek-V3 pattern)."""

    def __init__(self, num_layers: int = 2, dim: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([LinearWithScale(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layers.*", "layers.*.attn"]
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
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layers.*.attn", "head"]
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
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["layers.*", "layers.*.attn.q_proj"]
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
            discovery_iters=1,
            profiling_iters=1,
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

    def test_deep_param_pattern_no_false_warning(self, caplog):
        """A pattern like layers.*.weight should not warn when it matches deep params via recursive_star.

        NestedModel has parameters like layers.0.attn.q_proj.weight — with
        recursive_star=True the pattern layers.*.weight matches these, so the
        validation pass must not emit "did not match" warnings.
        """
        model = NestedModel()
        om = OffloadManager("test_deep_param_no_warn")

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._offload_modules(model, ["layers.*.weight"])

        assert not any("did not match" in m for m in caplog.messages)
        om.release()

    def test_star_weight_no_false_warning(self, caplog):
        """*.weight must not warn on a wrapper model where it matches deep params.

        Exercises the interaction of both fixes: param_recursive_star=True
        in _find_matched_patterns and the per-parameter fallback in
        _derive_module_patterns.
        """
        model = WrappedNestedModel()
        om = OffloadManager("test_star_weight_no_warn")

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._offload_modules(model, ["*.weight"])

        assert not any("did not match" in m for m in caplog.messages)
        om.release()


# =============================================================================
# Tests for _derive_module_patterns and parameter-level include patterns
# =============================================================================


class TestDeriveModulePatterns:
    """Tests for _derive_module_patterns() auto-truncation logic."""

    def test_module_level_pattern_unchanged(self):
        """A pattern matching a module directly is preserved."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = ModelWithLayers()
        result = _derive_module_patterns(["layer1"], get_module_paths(model))
        assert "layer1" in result

    def test_wildcard_module_pattern_unchanged(self):
        """Wildcard module-level pattern is kept as-is."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = NestedModel()
        result = _derive_module_patterns(["layers.*"], get_module_paths(model))
        assert "layers.*" in result

    def test_param_pattern_derives_module(self):
        """A parameter-level pattern derives the module-level prefix."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = ModelWithLayers()
        result = _derive_module_patterns(["layer1.linear.weight"], get_module_paths(model))
        assert "layer1.linear" in result or "layer1" in result

    def test_wildcard_param_pattern_derives_module(self):
        """layers.*.weight derives layers.* for module patching."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = NestedModel()
        result = _derive_module_patterns(["layers.*.weight"], get_module_paths(model))
        assert "layers.*" in result

    def test_star_pattern_unchanged(self):
        """The catch-all '*' pattern is preserved."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = ModelWithLayers()
        result = _derive_module_patterns(["*"], get_module_paths(model))
        assert "*" in result

    def test_deduplication(self):
        """Multiple patterns deriving the same module produce a single entry."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = NestedModel()
        result = _derive_module_patterns(["layers.*.weight", "layers.*.bias"], get_module_paths(model))
        assert result.count("layers.*") <= 1  # set-based, no duplicates

    def test_star_weight_does_not_collapse_to_star(self):
        """*.weight must not truncate to bare * on a wrapper model."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = WrappedNestedModel()
        module_paths = get_module_paths(model)
        result = _derive_module_patterns(["*.weight"], module_paths, model)
        assert "*" not in result
        assert len(result) > 1, "should derive multiple module-level targets, not a single catch-all"

    def test_star_weight_derives_correct_param_owners(self):
        """*.weight fallback must derive the exact modules that own weight parameters."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = WrappedNestedModel()
        result = set(_derive_module_patterns(["*.weight"], get_module_paths(model), model))
        expected = {
            "model.layers.0.attn.q_proj",
            "model.layers.0.attn.k_proj",
            "model.layers.0.norm",
            "model.layers.1.attn.q_proj",
            "model.layers.1.attn.k_proj",
            "model.layers.1.norm",
            "model.head",
        }
        assert result == expected

    def test_star_weight_flat_model_derives_linear_modules(self):
        """*.weight on a flat model (ModelWithLayers) derives the Linear sub-modules."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = ModelWithLayers()
        result = set(_derive_module_patterns(["*.weight"], get_module_paths(model), model))
        assert "*" not in result
        assert result == {"layer1.linear", "layer2.linear", "layer3.linear"}

    def test_star_weight_without_model_gracefully_skips(self):
        """*.weight without model arg falls through truncation and is skipped (no crash)."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = WrappedNestedModel()
        result = _derive_module_patterns(["*.weight"], get_module_paths(model))
        assert result == []

    def test_mixed_truncation_and_fallback(self):
        """Patterns that truncate fine and patterns that need fallback coexist."""
        from flextensor.offload_manager import _derive_module_patterns
        from flextensor.utils import get_module_paths

        model = WrappedNestedModel()
        module_paths = get_module_paths(model)
        result = set(
            _derive_module_patterns(
                ["model.layers.*", "*.bias"],
                module_paths,
                model,
            )
        )
        assert "model.layers.*" in result, "truncation-derived pattern should survive"
        assert "*" not in result, "*.bias must not collapse to *"
        assert "model.head" in result, "model.head owns bias params"


class TestFindMatchedPatterns:
    """Direct tests for _find_matched_patterns with split recursive_star."""

    def test_param_recursive_star_finds_deep_params(self):
        """param_recursive_star=True matches deep parameters that recursive_star=False misses."""
        from flextensor.offload_manager import _find_matched_patterns
        from flextensor.utils import get_module_paths

        model = NestedModel()
        module_paths = get_module_paths(model)

        without = _find_matched_patterns(
            model,
            ["layers.*.weight"],
            module_paths,
            recursive_star=False,
            include_parameters=True,
        )
        with_param = _find_matched_patterns(
            model,
            ["layers.*.weight"],
            module_paths,
            recursive_star=False,
            param_recursive_star=True,
            include_parameters=True,
        )
        assert without == set(), "recursive_star=False should miss deep parameter paths"
        assert with_param == {"layers.*.weight"}, "param_recursive_star=True should match"

    def test_param_recursive_star_defaults_to_recursive_star(self):
        """When param_recursive_star is not set, it falls back to recursive_star."""
        from flextensor.offload_manager import _find_matched_patterns
        from flextensor.utils import get_module_paths

        model = NestedModel()
        module_paths = get_module_paths(model)

        with_recursive = _find_matched_patterns(
            model,
            ["layers.*.weight"],
            module_paths,
            recursive_star=True,
            include_parameters=True,
        )
        assert with_recursive == {"layers.*.weight"}, (
            "recursive_star=True without param_recursive_star should match via fallback"
        )

    def test_module_matching_unaffected_by_param_recursive_star(self):
        """param_recursive_star does not change module-level matching semantics."""
        from flextensor.offload_manager import _find_matched_patterns
        from flextensor.utils import get_module_paths

        model = NestedModel()
        module_paths = get_module_paths(model)

        result = _find_matched_patterns(
            model,
            ["layers.*"],
            module_paths,
            recursive_star=False,
            param_recursive_star=True,
            include_parameters=True,
        )
        assert "layers.*" in result, "module-level pattern should still match"


class TestParameterLevelIncludePatching:
    """Tests that parameter-level include patterns correctly patch modules."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_param_pattern_patches_parent_module(self):
        """include_patterns=['layers.*.weight'] patches layers.0 and layers.1."""
        model = NestedModel()
        om = OffloadManager("test_param_patch")

        om._offload_modules(model, ["layers.*.weight"])

        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")
        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_default_star_patches_all(self):
        """Default ['*'] patches all modules (backward compat)."""
        model = ModelWithLayers()
        om = OffloadManager("test_default_star")

        om._offload_modules(model, ["*"])

        assert hasattr(model.layer1, "_ft_original_forward_func")
        assert hasattr(model.layer2, "_ft_original_forward_func")
        assert hasattr(model.layer3, "_ft_original_forward_func")
        om.release()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
    def test_star_weight_wrapper_model_not_single_unit(self):
        """*.weight on a wrapper model must NOT collapse into one offload unit.

        Regression: truncation used to derive bare ``*``, which patched the
        wrapper's single top-level child (``model``) and the ancestor guard
        suppressed everything else — creating one giant offload unit.
        """
        model = WrappedNestedModel()
        om = OffloadManager("test_star_weight_wrapper")

        om._offload_modules(model, ["*.weight"])

        assert len(om._patched_modules) > 1, (
            f"expected multiple offload units but got {len(om._patched_modules)}; pattern likely collapsed to '*'"
        )
        assert not hasattr(model.model, "_ft_original_forward_func"), (
            "top-level wrapper child 'model' should not be patched as a single unit"
        )
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


# =============================================================================
# Offload label uniqueness and pattern merging tests
# =============================================================================


class TestOffloadLabelUniqueness:
    """Verify offload label uniqueness and proper merging of derived patterns.

    Scenario 1 demonstrates the naming bug: sibling leaf modules derived from
    parameter-level patterns receive identical offload labels, merging their
    profiler stats and loader schedules.

    Scenarios 2 and 3 verify that the ancestor guard correctly merges
    overlapping patterns and that mixing module- and parameter-level patterns
    produces consistent patching.
    """

    def test_derived_leaf_modules_get_unique_labels(self):
        """Scenario 1: *.weight on ModelWithLayers must produce unique labels.

        _derive_module_patterns maps *.weight to {layer1.linear, layer2.linear,
        layer3.linear}.  Using module_path as the label ensures each gets a
        distinct name.
        """
        model = ModelWithLayers()
        om = OffloadManager("test_label_unique")

        om._offload_modules(model, ["*.weight"])

        offload_names = [m._ft_offload_name for m in om._patched_modules]
        assert len(offload_names) == 3, f"expected 3 patched modules, got {len(offload_names)}"
        assert len(set(offload_names)) == len(offload_names), (
            f"offload labels must be unique but got duplicates: {offload_names}"
        )

        om.release()

    def test_overlapping_param_patterns_merge_to_ancestor(self):
        """Scenario 2: *.x + *.weight merge into one trap per layer.

        layers.0 owns parameter 'x' directly while layers.0.linear owns
        'weight'.  Both patterns derive modules under the same subtree, so the
        ancestor guard must collapse them into a single trap on layers.0 /
        layers.1 rather than also patching layers.0.linear / layers.1.linear.
        """
        model = ModelWithDirectParams()
        om = OffloadManager("test_merge_ancestor")

        om._offload_modules(model, ["*.x", "*.weight"])

        assert len(om._patched_modules) == 2
        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")
        assert not hasattr(model.layers[0].linear, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].linear, "_ft_original_forward_func")

        om.release()

    def test_module_and_param_patterns_consistent(self):
        """Scenario 3: layers.* + *.weight does not split layers into sub-traps.

        When layers.* already matches layers.0 and layers.1, adding *.weight
        must not create extra traps on descendants like layers.0.attn.q_proj.
        The result should be identical to layers.* alone for the layers subtree.
        """
        model = NestedModel()
        om = OffloadManager("test_consistent")

        om._offload_modules(model, ["layers.*", "*.weight"])

        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")

        # No descendant within layers should be separately patched
        assert not hasattr(model.layers[0].attn, "_ft_original_forward_func")
        assert not hasattr(model.layers[0].attn.q_proj, "_ft_original_forward_func")
        assert not hasattr(model.layers[0].attn.k_proj, "_ft_original_forward_func")
        assert not hasattr(model.layers[0].norm, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].attn, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].attn.q_proj, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].attn.k_proj, "_ft_original_forward_func")
        assert not hasattr(model.layers[1].norm, "_ft_original_forward_func")

        om.release()


# =============================================================================
# DeepSeek-V3 inner field pattern tests (include/exclude)
# =============================================================================


class TestInnerFieldPatterns:
    """Tests for include/exclude patterns with the DeepSeek-V3 inner field pattern.

    The pattern ``self.weight.scale = self.scale = nn.Parameter(...)`` creates
    a tensor attribute on weight that aliases a registered module parameter.
    ``scale`` is visible to both ``named_parameters()`` (as ``layers.0.scale``)
    and ``get_inner_tensor_field_ids()`` (as an attribute of ``weight``).
    """

    @staticmethod
    def _build_tensors_map(model):
        return {id(p): p for p in model.parameters()}

    def test_scale_alias_identity(self):
        """Sanity check: weight.scale and scale are the same object."""
        model = ModelWithInnerFieldLayers()
        for i in range(2):
            layer = model.layers[i]
            assert layer.scale is layer.weight.scale, (
                f"layers.{i}.scale and layers.{i}.weight.scale must be the same object"
            )

    def test_include_star_scale_patches_correct_modules(self):
        """*.scale derives modules from the scale parameter and patches them."""
        model = ModelWithInnerFieldLayers()
        om = OffloadManager("test_inner_include_scale")

        om._offload_modules(model, ["*.scale"])

        assert len(om._patched_modules) == 2
        assert hasattr(model.layers[0], "_ft_original_forward_func")
        assert hasattr(model.layers[1], "_ft_original_forward_func")

        om.release()

    def test_include_star_weight_discovers_inner_scale(self):
        """*.weight includes weight.scale via inner field discovery."""
        from flextensor.tensor_discovery import get_offload_module_tensor_ids

        model = ModelWithInnerFieldLayers()
        om = OffloadManager("test_inner_weight_discovers")
        om._offload_modules(model, ["*.weight"])
        tensors_map = self._build_tensors_map(model)

        label_to_ids = get_offload_module_tensor_ids(
            model,
            tensors_map,
            include_patterns=["*.weight"],
        )

        for i in range(2):
            layer = model.layers[i]
            label = layer._ft_offload_name
            assert label in label_to_ids
            assert id(layer.weight) in label_to_ids[label], "weight should be offloaded"
            assert id(layer.scale) in label_to_ids[label], "weight.scale should be discovered as inner field of weight"

        om.release()

    def test_include_star_scale_only_offloads_scale(self):
        """*.scale offloads only scale, not weight."""
        from flextensor.tensor_discovery import get_offload_module_tensor_ids

        model = ModelWithInnerFieldLayers()
        om = OffloadManager("test_inner_scale_only")
        om._offload_modules(model, ["*.scale"])
        tensors_map = self._build_tensors_map(model)

        label_to_ids = get_offload_module_tensor_ids(
            model,
            tensors_map,
            include_patterns=["*.scale"],
        )

        for i in range(2):
            layer = model.layers[i]
            label = layer._ft_offload_name
            assert label in label_to_ids
            assert id(layer.scale) in label_to_ids[label], "scale should be offloaded"
            assert id(layer.weight) not in label_to_ids[label], (
                "weight should NOT be offloaded when only *.scale is included"
            )

        om.release()

    def test_exclude_star_scale_marks_non_offloaded(self):
        """exclude_patterns=['*.scale'] correctly identifies scale as non-offloaded."""
        from flextensor.tensor_discovery import get_non_offloaded_tensor_ids

        model = ModelWithInnerFieldLayers()
        tensors_map = self._build_tensors_map(model)

        non_offloaded = get_non_offloaded_tensor_ids(
            model,
            tensors_map,
            include_patterns=["*"],
            exclude_patterns=["*.scale"],
        )

        for i in range(2):
            layer = model.layers[i]
            assert id(layer.scale) in non_offloaded, f"layers.{i}.scale should be non-offloaded (excluded)"
            assert id(layer.weight) not in non_offloaded, f"layers.{i}.weight should remain offloaded"

    def test_exclude_star_scale_with_include_star_weight(self):
        """include=['*.weight'] + exclude=['*.scale']: weight offloaded, scale excluded.

        scale does not match *.weight so it is already excluded by the include
        filter.  The explicit exclude is redundant but should not cause errors.
        """
        from flextensor.tensor_discovery import get_non_offloaded_tensor_ids, get_offload_module_tensor_ids

        model = ModelWithInnerFieldLayers()
        om = OffloadManager("test_inner_weight_excl_scale")
        om._offload_modules(model, ["*.weight"])
        tensors_map = self._build_tensors_map(model)

        non_offloaded = get_non_offloaded_tensor_ids(
            model,
            tensors_map,
            include_patterns=["*.weight"],
            exclude_patterns=["*.scale"],
        )

        # scale does not match include *.weight → non-offloaded
        for i in range(2):
            layer = model.layers[i]
            assert id(layer.scale) in non_offloaded
            assert id(layer.weight) not in non_offloaded

        # After removing non-offloaded from tensors_map (simulating
        # _move_non_offloaded_tensors_to_gpu), inner field should not re-add scale
        filtered_map = {tid: t for tid, t in tensors_map.items() if tid not in non_offloaded}
        label_to_ids = get_offload_module_tensor_ids(
            model,
            filtered_map,
            include_patterns=["*.weight"],
            exclude_patterns=["*.scale"],
        )

        for i in range(2):
            layer = model.layers[i]
            label = layer._ft_offload_name
            assert label in label_to_ids
            assert id(layer.weight) in label_to_ids[label]
            assert id(layer.scale) not in label_to_ids[label], (
                "scale must not leak back via inner field after removal from tensors_map"
            )

        om.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
