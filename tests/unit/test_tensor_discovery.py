# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensor discovery utilities."""

import torch

from flextensor.collectors import IterativeLayerStatistics
from flextensor.tensor_discovery import (
    ModuleTracker,
    _any_prefix_matches,
    _should_offload_param,
    discover_untraced_tensors_for_layers,
    get_non_offloaded_tensor_ids,
    get_offload_module_tensor_ids,
    get_offload_name,
    has_offload_modules,
    is_offload_patched_module,
)


class SimpleModule(torch.nn.Module):
    """Simple test module."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10))
        self.bias = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        return x @ self.weight + self.bias


class SimpleModel(torch.nn.Module):
    """Simple test model with multiple layers."""

    def __init__(self):
        super().__init__()
        self.layer1 = SimpleModule()
        self.layer2 = SimpleModule()
        self.layer3 = SimpleModule()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# =============================================================================
# Tests for helper functions
# =============================================================================


def test_is_offload_patched_module():
    """Test detection of patched modules."""
    module = SimpleModule()
    assert not is_offload_patched_module(module)

    # Simulate patching by adding the marker attribute
    module._ft_original_forward_func = module.forward
    assert is_offload_patched_module(module)


def test_get_offload_name():
    """Test retrieving offload name from patched modules."""
    module = SimpleModule()
    assert get_offload_name(module) is None

    # Simulate patching
    module._ft_offload_name = "test_layer"
    assert get_offload_name(module) == "test_layer"


def test_has_offload_modules_with_none():
    """Test has_offload_modules with None model."""
    assert not has_offload_modules(None)


def test_has_offload_modules_with_dict():
    """Test has_offload_modules with dict model."""
    assert not has_offload_modules({})
    assert not has_offload_modules({"key": "value"})


def test_has_offload_modules_with_unpatched_model():
    """Test has_offload_modules with unpatched model."""
    model = SimpleModel()
    assert not has_offload_modules(model)


def test_has_offload_modules_with_patched_model():
    """Test has_offload_modules with patched model."""
    model = SimpleModel()
    # Simulate patching one layer
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "layer1"
    assert has_offload_modules(model)


# =============================================================================
# Tests for get_offload_module_tensor_ids
# =============================================================================


def test_get_offload_module_tensor_ids_with_none():
    """Test get_offload_module_tensor_ids with None model."""
    result = get_offload_module_tensor_ids(None, {})
    assert result == {}


def test_get_offload_module_tensor_ids_with_dict():
    """Test get_offload_module_tensor_ids with dict model."""
    result = get_offload_module_tensor_ids({}, {})
    assert result == {}


def test_get_offload_module_tensor_ids_with_unpatched_model():
    """Test get_offload_module_tensor_ids with unpatched model."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    result = get_offload_module_tensor_ids(model, tensors_map)
    assert result == {}


def test_get_offload_module_tensor_ids_with_patched_model():
    """Test get_offload_module_tensor_ids with patched model."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Simulate patching layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(model, tensors_map)

    assert "SimpleModel.layer1" in result
    # layer1 has weight and bias (2 parameters)
    assert len(result["SimpleModel.layer1"]) == 2

    # Verify the tensor IDs match layer1's parameters
    layer1_param_ids = {id(p) for p in model.layer1.parameters()}
    assert result["SimpleModel.layer1"] == layer1_param_ids


def test_get_offload_module_tensor_ids_with_multiple_patched_modules():
    """Test get_offload_module_tensor_ids with multiple patched modules."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Simulate patching layer1 and layer2
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"
    model.layer2._ft_original_forward_func = model.layer2.forward
    model.layer2._ft_offload_name = "SimpleModel.layer2"

    result = get_offload_module_tensor_ids(model, tensors_map)

    assert "SimpleModel.layer1" in result
    assert "SimpleModel.layer2" in result
    assert len(result["SimpleModel.layer1"]) == 2
    assert len(result["SimpleModel.layer2"]) == 2

    # Verify no overlap
    assert result["SimpleModel.layer1"].isdisjoint(result["SimpleModel.layer2"])


def test_get_offload_module_tensor_ids_filters_unknown_tensors():
    """Test that get_offload_module_tensor_ids only returns tensors in tensors_map."""
    model = SimpleModel()
    # Only include layer1's weight in tensors_map, not bias
    tensors_map = {id(model.layer1.weight): model.layer1.weight}

    # Patch layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(model, tensors_map)

    assert "SimpleModel.layer1" in result
    # Should only have 1 tensor (weight), not 2 (weight + bias)
    assert len(result["SimpleModel.layer1"]) == 1
    assert id(model.layer1.weight) in result["SimpleModel.layer1"]


# =============================================================================
# Tests for ModuleTracker
# =============================================================================


def test_module_tracker_track_and_get():
    """Test ModuleTracker tracking and retrieval."""
    tracker = ModuleTracker()
    model = SimpleModel()
    tracker.register(model)

    # Track modules for different labels
    tracker.enter_trap("label1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("label1")

    tracker.enter_trap("label2")
    _ = model.layer2(torch.randn(5, 10))
    tracker.exit_trap("label2")

    # Verify tracked modules
    trap_to_modules = tracker.get_trap_to_modules()

    assert "label1" in trap_to_modules
    assert "label2" in trap_to_modules
    assert model.layer1 in trap_to_modules["label1"]
    assert model.layer1 not in trap_to_modules["label2"]
    assert model.layer2 in trap_to_modules["label2"]
    assert model.layer2 not in trap_to_modules["label1"]

    tracker.unregister()


def test_module_tracker_nested_tracking():
    """Test ModuleTracker with nested module calls."""
    tracker = ModuleTracker()
    model = SimpleModel()
    tracker.register(model)

    tracker.enter_trap("full_forward")
    _ = model(torch.randn(5, 10))
    tracker.exit_trap("full_forward")

    # Should track all three layers
    trap_to_modules = tracker.get_trap_to_modules()
    assert "full_forward" in trap_to_modules
    assert model.layer1 in trap_to_modules["full_forward"]
    assert model.layer2 in trap_to_modules["full_forward"]
    assert model.layer3 in trap_to_modules["full_forward"]

    tracker.unregister()


# =============================================================================
# Tests for discover_untraced_tensors_for_layers
# =============================================================================


def test_discover_with_empty_inputs():
    """Test augmentation with empty inputs."""
    # Empty layer_stats
    result = discover_untraced_tensors_for_layers([], {}, None, {})
    assert result == []

    # Empty tensors_map
    layer_stats = [IterativeLayerStatistics(label="layer1", tensor_ids={1, 2}, duration=0.1)]
    result = discover_untraced_tensors_for_layers(layer_stats, {}, None, {})
    assert result == layer_stats


def test_discover_with_no_untraced_tensors():
    """Test discovery when all tensors are already traced."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # All tensors already traced
    all_tensor_ids = set(tensors_map.keys())
    layer_stats = [
        IterativeLayerStatistics(label="layer1", tensor_ids=all_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(layer_stats, tensors_map, model, tensor_id_to_name_map)

    # Should return unchanged
    assert len(result) == 1
    assert result[0].tensor_ids == all_tensor_ids


def test_discover_with_forward_patching():
    """Test discovery using forward patching strategy."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # Patch layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    # Only trace layer2 and layer3's tensors
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(layer_stats, tensors_map, model, tensor_id_to_name_map)

    # layer1 should now have its tensors added via forward patching
    layer1_param_ids = {id(p) for p in model.layer1.parameters()}
    assert result[0].tensor_ids == layer1_param_ids
    assert result[1].tensor_ids == layer2_tensor_ids
    assert result[2].tensor_ids == layer3_tensor_ids


def test_discover_with_module_tracker():
    """Test discovery using ModuleTracker strategy."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # Create tracker and track layer1
    tracker = ModuleTracker()
    tracker.register(model)
    tracker.enter_trap("SimpleModel.layer1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("SimpleModel.layer1")
    # Don't unregister yet - augmentation needs the tracked data

    # Only trace layer2 and layer3's tensors
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats, tensors_map, model, tensor_id_to_name_map, module_tracker=tracker
    )

    # layer1 should now have its tensors added via ModuleTracker
    layer1_param_ids = {id(p) for p in model.layer1.parameters()}
    assert result[0].tensor_ids == layer1_param_ids
    assert result[1].tensor_ids == layer2_tensor_ids
    assert result[2].tensor_ids == layer3_tensor_ids

    # Clean up
    tracker.unregister()


def test_discover_strategy_priority():
    """Test that forward patching is tried before ModuleTracker."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # Patch layer1 (should be found by forward patching)
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    # Track layer1 with tracker too (but should not be used since patching succeeds)
    tracker = ModuleTracker()
    tracker.register(model)
    tracker.enter_trap("SimpleModel.layer1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("SimpleModel.layer1")
    # Don't unregister yet

    # Only trace layer2 and layer3's tensors
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats, tensors_map, model, tensor_id_to_name_map, module_tracker=tracker
    )

    # layer1 should have its tensors (found via forward patching, not tracker)
    layer1_param_ids = {id(p) for p in model.layer1.parameters()}
    assert result[0].tensor_ids == layer1_param_ids

    # Clean up
    tracker.unregister()


def test_discover_no_fallback_to_all_layers():
    """Test that unmatched tensors are NOT added to all layers."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # Don't patch anything and don't provide tracker
    # Only trace layer2's tensors, leaving layer1 and layer3 untraced
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=set(), duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(layer_stats, tensors_map, model, tensor_id_to_name_map)

    # Prefix matching might add some tensors based on names
    # But we should verify that tensors are NOT blindly added to ALL layers
    # Each layer should have different tensor sets (not all the same)
    assert result[0].tensor_ids != result[2].tensor_ids or len(result[0].tensor_ids) == 0


def test_discover_preserves_original_tensors():
    """Test that discovery preserves originally traced tensors."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    # Patch layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    # Manually add one tensor to layer1 (simulating partial tracing)
    manually_traced_id = id(model.layer1.weight)
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids={manually_traced_id}, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(layer_stats, tensors_map, model, tensor_id_to_name_map)

    # layer1 should have both manually traced and augmented tensors
    layer1_param_ids = {id(p) for p in model.layer1.parameters()}
    assert manually_traced_id in result[0].tensor_ids
    assert result[0].tensor_ids == layer1_param_ids


# =============================================================================
# Tests for discover_untraced_tensors_for_layers with include_patterns
# =============================================================================


def test_discover_with_include_patterns_weight_only():
    """Forward-patching discovery filters parameters by include_patterns."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    layer0 = model.foo["layers"][0]
    layer0._ft_original_forward_func = layer0.forward
    layer0._ft_offload_name = "InnerLayer.0"

    layer1 = model.foo["layers"][1]
    layer1._ft_original_forward_func = layer1.forward
    layer1._ft_offload_name = "InnerLayer.1"

    layer_stats = [
        IterativeLayerStatistics(label="InnerLayer.0", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="InnerLayer.1", tensor_ids=set(), duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        include_patterns=["*.weight"],
    )

    weight_ids_0 = {id(layer0.bar.linear.weight), id(layer0.norm.weight)}
    weight_ids_1 = {id(layer1.bar.linear.weight), id(layer1.norm.weight)}
    assert result[0].tensor_ids == weight_ids_0
    assert result[1].tensor_ids == weight_ids_1


def test_discover_with_include_patterns_subtree():
    """Forward-patching discovery with mid-level module pattern includes only that subtree."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    layer0 = model.foo["layers"][0]
    layer0._ft_original_forward_func = layer0.forward
    layer0._ft_offload_name = "InnerLayer.0"

    layer_stats = [
        IterativeLayerStatistics(label="InnerLayer.0", tensor_ids=set(), duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        include_patterns=["foo.layers.*.bar"],
    )

    bar_ids = {id(layer0.bar.linear.weight), id(layer0.bar.linear.bias)}
    assert result[0].tensor_ids == bar_ids


def test_discover_with_include_patterns_infix_wildcard():
    """Forward-patching discovery with infix wildcard *layer1* targets specific module."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        include_patterns=["*layer1*"],
    )

    layer1_param_ids = {id(p) for p in model.layer1.parameters()}
    assert result[0].tensor_ids == layer1_param_ids


def test_discover_with_exclude_patterns_infix_wildcard():
    """Forward-patching discovery with infix wildcard *layer1* in exclude skips layer1 params."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        exclude_patterns=["*layer1*"],
    )

    assert result is layer_stats


def test_discover_with_combined_include_and_exclude_patterns():
    """Forward-patching discovery applies both include and exclude patterns together."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    layer0 = model.foo["layers"][0]
    layer0._ft_original_forward_func = layer0.forward
    layer0._ft_offload_name = "InnerLayer.0"

    layer1 = model.foo["layers"][1]
    layer1._ft_original_forward_func = layer1.forward
    layer1._ft_offload_name = "InnerLayer.1"

    layer_stats = [
        IterativeLayerStatistics(label="InnerLayer.0", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="InnerLayer.1", tensor_ids=set(), duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        include_patterns=["*.weight"],
        exclude_patterns=["foo.layers.0.*"],
    )

    # layer0: all weights excluded by exclude_patterns → nothing discovered
    assert result[0].tensor_ids == set()

    # layer1: only weights included (biases filtered by include), none excluded
    expected_layer1 = {id(layer1.bar.linear.weight), id(layer1.norm.weight)}
    assert result[1].tensor_ids == expected_layer1


# =============================================================================
# Tests for include/exclude filtering on ModuleTracker discovery (Strategy 2)
# =============================================================================


def test_module_tracker_discovery_respects_exclude_patterns():
    """ModuleTracker discovery filters out excluded parameters."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    tracker = ModuleTracker()
    tracker.register(model)
    tracker.enter_trap("SimpleModel.layer1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("SimpleModel.layer1")

    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        module_tracker=tracker,
        exclude_patterns=["layer1.weight"],
    )

    assert id(model.layer1.weight) not in result[0].tensor_ids
    assert id(model.layer1.bias) in result[0].tensor_ids

    tracker.unregister()


def test_module_tracker_discovery_respects_include_patterns():
    """ModuleTracker discovery only includes parameters matching include_patterns."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    tracker = ModuleTracker()
    tracker.register(model)
    tracker.enter_trap("SimpleModel.layer1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("SimpleModel.layer1")

    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        module_tracker=tracker,
        include_patterns=["*.bias"],
    )

    assert id(model.layer1.weight) not in result[0].tensor_ids
    assert id(model.layer1.bias) in result[0].tensor_ids

    tracker.unregister()


def test_module_tracker_discovery_exclude_all_params_yields_empty():
    """When exclude_patterns matches ALL untraced tensors, pre-filter removes them
    and the function early-returns the original layer_stats unchanged."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    tracker = ModuleTracker()
    tracker.register(model)
    tracker.enter_trap("SimpleModel.layer1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("SimpleModel.layer1")

    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        module_tracker=tracker,
        exclude_patterns=["layer1.*"],
    )

    # All untraced tensors were excluded → result is the original layer_stats
    assert result is layer_stats

    tracker.unregister()


def test_module_tracker_discovery_exclude_partial_still_discovers():
    """When exclude_patterns matches SOME untraced tensors, the remaining
    tensors are still discovered by strategies (not early-returned)."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    tracker = ModuleTracker()
    tracker.register(model)
    tracker.enter_trap("SimpleModel.layer1")
    _ = model.layer1(torch.randn(5, 10))
    tracker.exit_trap("SimpleModel.layer1")

    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(label="SimpleModel.layer1", tensor_ids=set(), duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        module_tracker=tracker,
        exclude_patterns=["layer1.weight"],
    )

    # Not early-returned — new IterativeLayerStatistics objects were created
    assert result is not layer_stats
    # Excluded weight is absent, non-excluded bias was discovered
    assert id(model.layer1.weight) not in result[0].tensor_ids
    assert id(model.layer1.bias) in result[0].tensor_ids

    tracker.unregister()


# =============================================================================
# Tests for include/exclude filtering on prefix matching discovery (Strategy 3)
# =============================================================================


def test_prefix_matching_discovery_respects_exclude_patterns():
    """Prefix-matching discovery filters out excluded parameters."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    layer1_weight_id = id(model.layer1.weight)
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(
            label="SimpleModel.layer1",
            tensor_ids={layer1_weight_id},
            duration=0.1,
        ),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        exclude_patterns=["layer1.bias"],
    )

    assert layer1_weight_id in result[0].tensor_ids
    assert id(model.layer1.bias) not in result[0].tensor_ids


def test_prefix_matching_discovery_respects_include_patterns():
    """Prefix-matching discovery only includes parameters matching include_patterns."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}
    tensor_id_to_name_map = {id(p): name for name, p in model.named_parameters()}

    layer1_weight_id = id(model.layer1.weight)
    layer2_tensor_ids = {id(p) for p in model.layer2.parameters()}
    layer3_tensor_ids = {id(p) for p in model.layer3.parameters()}

    layer_stats = [
        IterativeLayerStatistics(
            label="SimpleModel.layer1",
            tensor_ids={layer1_weight_id},
            duration=0.1,
        ),
        IterativeLayerStatistics(label="SimpleModel.layer2", tensor_ids=layer2_tensor_ids, duration=0.1),
        IterativeLayerStatistics(label="SimpleModel.layer3", tensor_ids=layer3_tensor_ids, duration=0.1),
    ]

    result = discover_untraced_tensors_for_layers(
        layer_stats,
        tensors_map,
        model,
        tensor_id_to_name_map,
        include_patterns=["*.weight"],
    )

    assert layer1_weight_id in result[0].tensor_ids
    assert id(model.layer1.bias) not in result[0].tensor_ids


# =============================================================================
# Tests for exclude_patterns in get_offload_module_tensor_ids
# =============================================================================


def test_get_offload_module_tensor_ids_with_exclude_patterns():
    """Test that exclude_patterns filters out matching parameters."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Patch layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    # Exclude all weight parameters
    result = get_offload_module_tensor_ids(model, tensors_map, exclude_patterns=["*.weight"])

    assert "SimpleModel.layer1" in result
    # Only bias should remain (weight is excluded)
    assert len(result["SimpleModel.layer1"]) == 1
    assert id(model.layer1.bias) in result["SimpleModel.layer1"]
    assert id(model.layer1.weight) not in result["SimpleModel.layer1"]


def test_get_offload_module_tensor_ids_exclude_infix():
    """Infix wildcard *layer1* in exclude removes all layer1 params from offload."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(
        model,
        tensors_map,
        exclude_patterns=["*layer1*"],
    )

    assert "SimpleModel.layer1" in result
    assert len(result["SimpleModel.layer1"]) == 0


def test_get_offload_module_tensor_ids_exclude_specific_param():
    """Test excluding a specific parameter by full path."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Patch layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    # Exclude only layer1's bias
    result = get_offload_module_tensor_ids(model, tensors_map, exclude_patterns=["layer1.bias"])

    assert "SimpleModel.layer1" in result
    # Only weight should remain (bias is excluded)
    assert len(result["SimpleModel.layer1"]) == 1
    assert id(model.layer1.weight) in result["SimpleModel.layer1"]
    assert id(model.layer1.bias) not in result["SimpleModel.layer1"]


def test_get_offload_module_tensor_ids_empty_exclude():
    """Test that empty exclude_patterns changes nothing."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Patch layer1
    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(model, tensors_map, exclude_patterns=[])

    assert "SimpleModel.layer1" in result
    assert len(result["SimpleModel.layer1"]) == 2


# =============================================================================
# Tests for module-level exclude within offload units
# =============================================================================


class BarModule(torch.nn.Module):
    """Sub-module with weight and bias."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class InnerLayer(torch.nn.Module):
    """Layer containing a bar sub-module and a norm."""

    def __init__(self):
        super().__init__()
        self.bar = BarModule()
        self.norm = torch.nn.LayerNorm(10)

    def forward(self, x):
        return self.norm(self.bar(x))


class NestedOffloadModel(torch.nn.Module):
    """Model with nested structure for testing module-level exclude within offload units."""

    def __init__(self):
        super().__init__()
        self.foo = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([InnerLayer(), InnerLayer()]),
        })

    def forward(self, x):
        for layer in self.foo["layers"]:
            x = layer(x)
        return x


def test_exclude_module_within_offload_unit():
    """Excluding a sub-module path within an offload unit removes its parameters."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Patch foo.layers.0 as an offload unit
    layer0 = model.foo["layers"][0]
    layer0._ft_original_forward_func = type(layer0).forward
    layer0._ft_offload_name = "ModuleList.0"

    # Exclude the bar sub-module within the offload unit
    result = get_offload_module_tensor_ids(model, tensors_map, exclude_patterns=["foo.layers.0.bar"])

    assert "ModuleList.0" in result
    # bar's parameters (linear.weight, linear.bias) should be excluded
    bar_param_ids = {id(p) for p in layer0.bar.parameters()}
    for pid in bar_param_ids:
        assert pid not in result["ModuleList.0"]

    # norm's parameters should still be included
    norm_param_ids = {id(p) for p in layer0.norm.parameters()}
    for pid in norm_param_ids:
        if pid in tensors_map:
            assert pid in result["ModuleList.0"]


def test_exclude_module_cascades_to_parameters():
    """Excluding a module path excludes ALL its parameters (weight, bias, etc.)."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}

    # Patch foo.layers.0 as an offload unit
    layer0 = model.foo["layers"][0]
    layer0._ft_original_forward_func = type(layer0).forward
    layer0._ft_offload_name = "ModuleList.0"

    # Exclude with a wildcard pattern that matches bar sub-modules in all layers
    result = get_offload_module_tensor_ids(model, tensors_map, exclude_patterns=["foo.layers.*.bar"])

    assert "ModuleList.0" in result
    # bar's parameters should be excluded (both weight and bias of bar.linear)
    bar_param_ids = {id(p) for p in layer0.bar.parameters()}
    for pid in bar_param_ids:
        assert pid not in result["ModuleList.0"]

    # norm's parameters should still be present
    norm_param_ids = {id(p) for p in layer0.norm.parameters()}
    for pid in norm_param_ids:
        if pid in tensors_map:
            assert pid in result["ModuleList.0"]


# =============================================================================
# Tests for include_patterns in get_offload_module_tensor_ids
# =============================================================================


def test_get_offload_module_tensor_ids_include_weight_only():
    """Include patterns filter to only matching parameters."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(
        model,
        tensors_map,
        include_patterns=["*.weight"],
    )

    assert "SimpleModel.layer1" in result
    assert id(model.layer1.weight) in result["SimpleModel.layer1"]
    assert id(model.layer1.bias) not in result["SimpleModel.layer1"]


def test_get_offload_module_tensor_ids_default_include_all():
    """Default include_patterns=['*'] includes all parameters."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(model, tensors_map)

    assert "SimpleModel.layer1" in result
    assert id(model.layer1.weight) in result["SimpleModel.layer1"]
    assert id(model.layer1.bias) in result["SimpleModel.layer1"]


def test_get_offload_module_tensor_ids_include_and_exclude():
    """Include weights only + exclude layer1.weight → layer1 has no tensors."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"

    result = get_offload_module_tensor_ids(
        model,
        tensors_map,
        include_patterns=["*.weight"],
        exclude_patterns=["layer1.weight"],
    )

    assert "SimpleModel.layer1" in result
    assert len(result["SimpleModel.layer1"]) == 0


def test_get_offload_module_tensor_ids_include_infix():
    """Infix wildcard *layer1* includes only layer1 parameters in offload modules."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    model.layer1._ft_original_forward_func = model.layer1.forward
    model.layer1._ft_offload_name = "SimpleModel.layer1"
    model.layer2._ft_original_forward_func = model.layer2.forward
    model.layer2._ft_offload_name = "SimpleModel.layer2"

    result = get_offload_module_tensor_ids(
        model,
        tensors_map,
        include_patterns=["*layer1*"],
    )

    assert "SimpleModel.layer1" in result
    assert id(model.layer1.weight) in result["SimpleModel.layer1"]
    assert id(model.layer1.bias) in result["SimpleModel.layer1"]
    assert "SimpleModel.layer2" in result
    assert len(result["SimpleModel.layer2"]) == 0


# =============================================================================
# Tests for get_non_offloaded_tensor_ids (exclude patterns)
# =============================================================================


def test_get_non_offloaded_basic_exclude_match():
    """Excluded parameters are returned as non-offloaded."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(model, tensors_map, exclude_patterns=["*.weight"])

    expected = {id(model.layer1.weight), id(model.layer2.weight), id(model.layer3.weight)}
    assert non_offloaded == expected


def test_get_non_offloaded_empty_patterns():
    """Empty patterns return empty set."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    assert get_non_offloaded_tensor_ids(model, tensors_map, exclude_patterns=[]) == set()
    assert get_non_offloaded_tensor_ids(model, tensors_map, exclude_patterns=None) == set()


def test_get_non_offloaded_no_match():
    """Non-matching patterns return empty set."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(model, tensors_map, exclude_patterns=["nonexistent.*"])
    assert non_offloaded == set()


def test_get_non_offloaded_dict_model():
    """Dict models support exclude patterns via dict keys."""
    t1 = torch.randn(10)
    t2 = torch.randn(10)
    t3 = torch.randn(10)
    model_dict = {
        "layers.0.attention.wq.weight": t1,
        "layers.0.attention.wk.weight": t2,
        "norm.weight": t3,
    }
    tensors_map = {id(t1): t1, id(t2): t2, id(t3): t3}

    non_offloaded = get_non_offloaded_tensor_ids(model_dict, tensors_map, exclude_patterns=["norm.*"])
    assert non_offloaded == {id(t3)}

    non_offloaded_all = get_non_offloaded_tensor_ids(model_dict, tensors_map, exclude_patterns=["*"])
    assert non_offloaded_all == {id(t1), id(t2), id(t3)}

    non_offloaded_none = get_non_offloaded_tensor_ids(model_dict, tensors_map, exclude_patterns=["nonexistent"])
    assert non_offloaded_none == set()


def test_get_non_offloaded_dict_model_infix_exclude():
    """Infix wildcard *attn* in exclude_patterns excludes matching keys."""
    t_attn = torch.randn(10)
    t_mlp = torch.randn(10)
    t_norm = torch.randn(10)
    model_dict = {
        "self_attn.weight": t_attn,
        "mlp.weight": t_mlp,
        "norm.weight": t_norm,
    }
    tensors_map = {id(t_attn): t_attn, id(t_mlp): t_mlp, id(t_norm): t_norm}

    non_offloaded = get_non_offloaded_tensor_ids(
        model_dict,
        tensors_map,
        exclude_patterns=["*attn*"],
    )

    assert non_offloaded == {id(t_attn)}


def test_get_non_offloaded_none_model():
    """None model returns empty set."""
    non_offloaded = get_non_offloaded_tensor_ids(None, {}, exclude_patterns=["*"])
    assert non_offloaded == set()


def test_get_non_offloaded_only_returns_ids_in_tensors_map():
    """Only IDs present in tensors_map are returned."""
    model = SimpleModel()
    tensors_map = {id(model.layer1.weight): model.layer1.weight}

    non_offloaded = get_non_offloaded_tensor_ids(model, tensors_map, exclude_patterns=["*.weight"])

    assert non_offloaded == {id(model.layer1.weight)}


# =============================================================================
# Tests for _any_prefix_matches
# =============================================================================


def test_any_prefix_matches_wildcard_all():
    """Default ['*'] includes every parameter."""
    assert _any_prefix_matches("layers.0.weight", ["*"])
    assert _any_prefix_matches("head.bias", ["*"])


def test_any_prefix_matches_module_level():
    """Module-level pattern includes all parameters of that module."""
    assert _any_prefix_matches("layers.0.weight", ["layers.*"])
    assert _any_prefix_matches("layers.0.bias", ["layers.*"])
    assert not _any_prefix_matches("head.weight", ["layers.*"])


def test_any_prefix_matches_param_level():
    """Parameter-level pattern only includes matching parameters."""
    assert _any_prefix_matches("layers.0.weight", ["*.weight"])
    assert not _any_prefix_matches("layers.0.bias", ["*.weight"])


def test_any_prefix_matches_specific_pattern():
    """Specific parameter path pattern works."""
    assert _any_prefix_matches("layers.0.attn.weight", ["layers.*.attn.weight"])
    assert not _any_prefix_matches("layers.0.attn.bias", ["layers.*.attn.weight"])
    assert not _any_prefix_matches("layers.0.mlp.weight", ["layers.*.attn.weight"])


def test_any_prefix_matches_deep_nesting_module_pattern():
    """Module-level pattern cascades to all deeply nested parameters."""
    assert _any_prefix_matches("layers.0.attn.q_proj.weight", ["layers.*"])
    assert _any_prefix_matches("layers.0.attn.q_proj.bias", ["layers.*"])
    assert _any_prefix_matches("layers.0.mlp.gate.weight", ["layers.*"])
    assert not _any_prefix_matches("embed.weight", ["layers.*"])


def test_any_prefix_matches_deep_nesting_param_pattern():
    """Parameter-level pattern matches only exact paths, not siblings at same depth."""
    assert _any_prefix_matches("layers.0.attn.q_proj.weight", ["*.weight"])
    assert not _any_prefix_matches("layers.0.attn.q_proj.bias", ["*.weight"])
    assert _any_prefix_matches("layers.0.mlp.gate.weight", ["*.weight"])


def test_any_prefix_matches_mid_level_module_pattern():
    """Pattern targeting an intermediate module includes its descendants."""
    assert _any_prefix_matches("layers.0.attn.q_proj.weight", ["layers.*.attn"])
    assert _any_prefix_matches("layers.0.attn.k_proj.bias", ["layers.*.attn"])
    assert not _any_prefix_matches("layers.0.mlp.gate.weight", ["layers.*.attn"])


def test_any_prefix_matches_infix_wildcard():
    """Infix wildcard *layer* matches segments containing 'layer' as substring."""
    assert _any_prefix_matches("layer1.weight", ["*layer*"])
    assert _any_prefix_matches("my_layer.bias", ["*layer*"])
    assert _any_prefix_matches("layers.0.weight", ["*layer*"])
    assert not _any_prefix_matches("head.weight", ["*layer*"])
    assert not _any_prefix_matches("embed.bias", ["*layer*"])


# =============================================================================
# Tests for _should_offload_param
# =============================================================================


def test_should_offload_param_included_and_not_excluded():
    """Included AND not excluded → True."""
    assert _should_offload_param("layers.0.weight", ["*"], [])


def test_should_offload_param_not_included():
    """Not matching include → False, regardless of exclude."""
    assert not _should_offload_param("layers.0.weight", ["*.bias"], [])
    assert not _should_offload_param("layers.0.weight", ["*.bias"], ["*.weight"])


def test_should_offload_param_included_but_excluded():
    """Matching both include and exclude → False (exclude wins)."""
    assert not _should_offload_param("layers.0.weight", ["*"], ["*.weight"])


def test_should_offload_param_empty_exclude_list():
    """Empty exclude list never excludes anything."""
    assert _should_offload_param("layers.0.weight", ["*"], [])
    assert _should_offload_param("layers.0.bias", ["*.bias"], [])


def test_should_offload_param_module_level_include():
    """Module-level include cascades to nested parameters."""
    assert _should_offload_param("layers.0.attn.q.weight", ["layers.*"], [])
    assert not _should_offload_param("embed.weight", ["layers.*"], [])


def test_should_offload_param_module_level_exclude():
    """Module-level exclude cascades to nested parameters."""
    assert not _should_offload_param("layers.0.attn.q.weight", ["*"], ["layers.*"])
    assert _should_offload_param("embed.weight", ["*"], ["layers.*"])


def test_should_offload_param_include_and_exclude_same_scope():
    """Include a subtree, then exclude a deeper subtree within it."""
    assert _should_offload_param("layers.0.mlp.weight", ["layers.*"], ["layers.*.attn"])
    assert not _should_offload_param("layers.0.attn.q.weight", ["layers.*"], ["layers.*.attn"])


def test_should_offload_param_include_narrow_exclude_broad():
    """Narrow include + broad exclude that overlaps → excluded."""
    assert not _should_offload_param("layers.0.weight", ["*.weight"], ["layers.0.*"])


def test_should_offload_param_infix_wildcards():
    """Infix wildcards match segments containing the substring."""
    assert _should_offload_param("my_layer.weight", ["*layer*"], [])
    assert not _should_offload_param("my_layer.weight", ["*layer*"], ["*layer*"])
    # *weight* only matches a segment containing "weight" at a prefix level;
    # "weight" as the leaf segment is never a standalone prefix, so it does
    # NOT exclude "my_layer.weight".
    assert _should_offload_param("my_layer.weight", ["*layer*"], ["*weight*"])


# =============================================================================
# Tests for get_non_offloaded_tensor_ids (include patterns)
# =============================================================================


def test_get_non_offloaded_include_weight_only():
    """Only weight parameters are offloaded; biases stay on GPU."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(
        model,
        tensors_map,
        include_patterns=["*.weight"],
    )

    expected_bias_ids = {id(model.layer1.bias), id(model.layer2.bias), id(model.layer3.bias)}
    assert non_offloaded == expected_bias_ids


def test_get_non_offloaded_include_single_layer():
    """Only layer1 parameters are offloaded; layer2 and layer3 stay on GPU."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(
        model,
        tensors_map,
        include_patterns=["layer1"],
    )

    expected = {
        id(model.layer2.weight),
        id(model.layer2.bias),
        id(model.layer3.weight),
        id(model.layer3.bias),
    }
    assert non_offloaded == expected


def test_get_non_offloaded_include_and_exclude_combined():
    """Include+exclude: include only weights, then exclude layer1."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(
        model,
        tensors_map,
        include_patterns=["*.weight"],
        exclude_patterns=["layer1.*"],
    )

    # Non-offloaded = all biases (not included) + layer1 params (excluded)
    expected = {
        id(model.layer1.weight),
        id(model.layer1.bias),
        id(model.layer2.bias),
        id(model.layer3.bias),
    }
    assert non_offloaded == expected


def test_get_non_offloaded_default_includes_backward_compat():
    """Default include_patterns=['*'] with no excludes returns empty set."""
    model = SimpleModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(model, tensors_map)
    assert non_offloaded == set()


def test_get_non_offloaded_deep_nesting_weight_only():
    """Parameter-level include with deeply nested model only offloads matching params."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(
        model,
        tensors_map,
        include_patterns=["*.weight"],
    )

    expected_bias_ids = {
        id(model.foo["layers"][0].bar.linear.bias),
        id(model.foo["layers"][0].norm.bias),
        id(model.foo["layers"][1].bar.linear.bias),
        id(model.foo["layers"][1].norm.bias),
    }
    assert non_offloaded == expected_bias_ids


def test_get_non_offloaded_deep_nesting_subtree():
    """Mid-level module pattern offloads only that subtree's params."""
    model = NestedOffloadModel()
    tensors_map = {id(p): p for p in model.parameters()}

    non_offloaded = get_non_offloaded_tensor_ids(
        model,
        tensors_map,
        include_patterns=["foo.layers.*.bar"],
    )

    expected_norm_ids = {
        id(model.foo["layers"][0].norm.weight),
        id(model.foo["layers"][0].norm.bias),
        id(model.foo["layers"][1].norm.weight),
        id(model.foo["layers"][1].norm.bias),
    }
    assert non_offloaded == expected_norm_ids


def test_get_non_offloaded_dict_model_include_patterns():
    """Dict models support include patterns."""
    t1 = torch.randn(10)
    t2 = torch.randn(10)
    t3 = torch.randn(10)
    model_dict = {
        "layers.0.attention.wq.weight": t1,
        "layers.0.attention.wk.weight": t2,
        "norm.weight": t3,
    }
    tensors_map = {id(t1): t1, id(t2): t2, id(t3): t3}

    non_offloaded = get_non_offloaded_tensor_ids(
        model_dict,
        tensors_map,
        include_patterns=["layers.*"],
    )
    assert non_offloaded == {id(t3)}


def test_get_non_offloaded_dict_model_infix_include():
    """Infix wildcard *attn* includes only dict keys whose first segment contains 'attn'."""
    t_attn = torch.randn(10)
    t_mlp = torch.randn(10)
    t_norm = torch.randn(10)
    model_dict = {
        "self_attn.weight": t_attn,
        "mlp.weight": t_mlp,
        "norm.weight": t_norm,
    }
    tensors_map = {id(t_attn): t_attn, id(t_mlp): t_mlp, id(t_norm): t_norm}

    non_offloaded = get_non_offloaded_tensor_ids(
        model_dict,
        tensors_map,
        include_patterns=["*attn*"],
    )

    assert non_offloaded == {id(t_mlp), id(t_norm)}


class TestEmptyIncludePatterns:
    """Tests for include_patterns=[] vs None semantics."""

    def test_empty_include_patterns_warns(self, caplog):
        """Passing include_patterns=[] logs a warning via validate_include_patterns."""
        import logging

        from flextensor.tensor_discovery import validate_include_patterns

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_discovery"):
            validate_include_patterns([])
        assert "include_patterns is an empty list" in caplog.text
        assert "pattern-based parameter filtering is disabled" in caplog.text

    def test_none_include_patterns_no_warning(self, caplog):
        """Passing include_patterns=None uses default ['*'] without warning."""
        import logging

        model = SimpleModule()
        tensors_map = {id(p): p for p in model.parameters()}

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_discovery"):
            result = get_non_offloaded_tensor_ids(model, tensors_map, include_patterns=None)
        assert "include_patterns is an empty list" not in caplog.text
        assert result == set()

    def test_empty_include_patterns_disables_filtering(self):
        """With include_patterns=[], no tensors are classified as non-offloaded (manual-trap mode)."""
        model = SimpleModel()
        tensors_map = {id(p): p for p in model.parameters()}

        non_offloaded = get_non_offloaded_tensor_ids(model, tensors_map, include_patterns=[])
        assert non_offloaded == set()
        assert len(tensors_map) > 0

    def test_empty_include_patterns_dict_model(self):
        """With include_patterns=[] on a dict model, no tensors are classified as non-offloaded."""
        t1 = torch.randn(4, 4)
        t2 = torch.randn(4, 4)
        model_dict = {"layers.0.weight": t1, "layers.1.weight": t2}
        tensors_map = {id(t1): t1, id(t2): t2}

        non_offloaded = get_non_offloaded_tensor_ids(model_dict, tensors_map, include_patterns=[])
        assert non_offloaded == set()

    def test_validate_include_patterns_no_warning_for_none(self, caplog):
        """validate_include_patterns does not warn when include_patterns is None."""
        import logging

        from flextensor.tensor_discovery import validate_include_patterns

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_discovery"):
            validate_include_patterns(None)
        assert "include_patterns is an empty list" not in caplog.text
