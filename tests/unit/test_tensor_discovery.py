# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tensor discovery utilities."""

import torch

from flextensor.collectors import IterativeLayerStatistics
from flextensor.tensor_discovery import (
    ModuleTracker,
    discover_untraced_tensors_for_layers,
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
