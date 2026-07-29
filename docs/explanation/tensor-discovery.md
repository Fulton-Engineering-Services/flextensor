<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Understanding Untraced Tensor Discovery

This document explains how FlexTensor discovers tensors that aren't traced through normal PyTorch function interception. Understanding this mechanism helps explain why certain model architectures work seamlessly with FlexTensor and aids in debugging edge cases.

## The Problem: Untraced Tensors

During the discovery phase, FlexTensor builds a map of which tensors belong to which layers. In indirect paths it does this by intercepting PyTorch operations via `__torch_function__`. In the default direct path, forward-patched modules also provide a structural mapping from trap labels to owned tensors, and FlexTensor materializes those tensors while the trap executes. Some tensors can still slip through:

### Why Some Tensors Go Untraced

1. **Direct kernel calls**: Custom CUDA or Triton kernels that bypass PyTorch's dispatch system. FP8 (8-bit floating point) inference frameworks like Transformer Engine often use Triton kernels that access weight tensors directly.

2. **Optimized fused operations**: Fused attention or MLP implementations may access parameters through C++ bindings that don't invoke `__torch_function__`.

3. **Lazy tensor access**: Parameters that are only accessed under certain conditions during inference but not during discovery.

The consequence: if FlexTensor doesn't know a tensor belongs to a layer, it can't offload it properly. The tensor either stays on GPU (wasting memory) or gets offloaded incorrectly (causing errors).

Direct warmup/profile also handle the common raw-parameter case by temporarily pointing the original parameter storage at the active materialized tensor. That covers code that reads `self.weight` directly. It does not remove the need for discovery: FlexTensor still has to know which tensor IDs to materialize for each trap.

For custom kernels and fused backends such as vLLM MoE, the supported shape is:

1. Wrap the module that owns the kernel weights with forward patching (`offload()` plus matching `include_patterns`).
2. Launch the custom kernel from inside that patched module's `forward`.
3. Pass weights by reading module parameters while the trap is active, for example `self.weight`, `self.w13_weight`, or `self.w2_weight`.

In that shape, direct warmup/profile materialize the raw parameter storage before the kernel launch, so the backend sees active GPU storage even if the kernel bypasses PyTorch dispatch. Manual `offload_block()` can work, but it relies on ModuleTracker or prefix matching rather than the direct forward-patched ownership path, so it is less robust for unconventional MoE parameter layouts. Avoid caching original CPU tensor or `Parameter` objects before `offload()` and using those cached references later; FlexTensor cannot rewrite opaque external pointers that never go through module ownership, direct getters, or discovery.

### Example: FP8 Weight Scales

Consider a model using FP8 quantization:

```python
class FP8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(...)  # FP8 tensor
        self.weight.scale = torch.tensor(...)  # Scale factor for dequantization

    def forward(self, x):
        # Triton kernel accesses weight.scale directly
        return triton_fp8_matmul(x, self.weight, self.weight.scale)
```

During discovery, FlexTensor traces `self.weight` through PyTorch operations. But `self.weight.scale` is accessed by the Triton kernel, which doesn't trigger `__torch_function__`. Without tensor discovery, the scale tensor would be left on CPU when the weight is loaded to GPU, causing a device mismatch error.

## The Solution: Multi-Strategy Discovery

FlexTensor uses three discovery strategies, tried in sequence. Each strategy succeeds for different model configurations, and the system falls back to the next if one doesn't find all untraced tensors.

```
┌─────────────────────────────────────────────────────────────┐
│                    Untraced Tensors                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Strategy 1: Auto Trap      │
              │  (Forward Patching)         │
              │  Most accurate, uses        │
              │  module.named_parameters()  │
              └─────────────────────────────┘
                            │
                      Found all? ──Yes──▶ Done
                            │
                           No
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Strategy 2: Module Tracker │
              │  (Manual Traps)             │
              │  Forward hooks record which │
              │  modules execute per trap   │
              └─────────────────────────────┘
                            │
                      Found all? ──Yes──▶ Done
                            │
                           No
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Strategy 3: Prefix Match   │
              │  (Fallback)                 │
              │  Match tensor names by      │
              │  patterns from traced ones  │
              └─────────────────────────────┘
                            │
                            ▼
                          Done
```

## Strategy 1: Auto Trap Discovery

### When It Works

This strategy applies when you use FlexTensor's `offload()` API with include patterns:

```python
from flextensor import offload, OffloadConfig

config = OffloadConfig(include_patterns=["layers.*"])
model = offload(model, config=config)
```

The `offload()` function patches each matched module's `forward` method and marks it with `_ft_original_forward_func` and `_ft_offload_name` attributes. This is called "forward patching" or "auto trap."

### How It Works

For each patched module, FlexTensor:

1. Identifies modules with `_ft_original_forward_func` attribute (marker for forward patching)
2. Retrieves the offload name from `_ft_offload_name`
3. Calls `module.named_parameters()` to get all tensor IDs
4. Includes inner tensor fields (like `weight.scale`) by inspecting custom attributes

```python
# Internal discovery logic (simplified)
for module in model.modules():
    if hasattr(module, "_ft_original_forward_func"):
        label = module._ft_offload_name
        for param in module.parameters():
            tensor_ids[label].add(id(param))
            # Also add inner fields like weight.scale
            for field in get_custom_tensor_fields(param):
                tensor_ids[label].add(id(field))
```

### Why It's Most Accurate

This strategy has a direct, structural relationship: each patched module knows exactly which tensors it owns. There's no guessing or pattern matching—just iterating `named_parameters()`.

## Strategy 2: Module Tracker Discovery

### When It Works

This strategy applies when you use manual traps (context managers) instead of forward patching. **Requires `skip_discovery=False`** (the default) — setting `skip_discovery=True` short-circuits DISCOVERY and never captures the tensor mappings that manual blocks enumerate, so `offload_block()` raises a `RuntimeError`:

```python
config = OffloadConfig(...)  # skip_discovery=False by default
manager = get_offload_manager()
model = manager.offload(model, config)

for batch in dataloader:
    with manager.offload_block("layer_0"):
        x = model.layer_0(x)
    with manager.offload_block("layer_1"):
        x = model.layer_1(x)
```

In this mode, FlexTensor doesn't know which module corresponds to which trap name—the user defines that relationship through code structure.

### How It Works

The `ModuleTracker` class solves this by using forward hooks:

1. **Registration**: At discovery start, forward hooks are registered on all modules with parameters
2. **Tracking**: When a trap context is entered, the tracker records the current trap name
3. **Recording**: Forward hooks fire as modules execute, recording which modules ran under which trap
4. **Discovery**: After the discovery phase, each trap's modules are known, and their parameters can be retrieved

```python
# How ModuleTracker builds the mapping (simplified)
class ModuleTracker:
    def enter_trap(self, trap_name):
        self._current_trap = trap_name

    def _forward_hook(self, module, inputs, output):
        if self._current_trap:
            self._trap_to_modules[self._current_trap].add(module)

    def get_trap_tensor_ids(self, tensors_map):
        result = {}
        for trap_name, modules in self._trap_to_modules.items():
            tensor_ids = set()
            for module in modules:
                for param in module.parameters(recurse=False):
                    if id(param) in tensors_map:
                        tensor_ids.add(id(param))
            result[trap_name] = tensor_ids
        return result
```

### Why It's Needed

Manual traps offer flexibility—you can group arbitrary code under a trap name. But this flexibility means FlexTensor can't statically determine which tensors belong where. The ModuleTracker bridges this gap by observing actual execution.

## Strategy 3: Prefix Matching Discovery

### When It Works

This is the fallback strategy for edge cases where:
- Forward patching isn't used
- Module execution doesn't cleanly map to trap contexts
- Tensors exist outside standard module hierarchies

### How It Works

The strategy infers tensor naming patterns from tensors that *were* traced:

1. **Collect traced tensor names**: For each layer, gather names of traced tensors (e.g., `"layers.0.attn.q.weight"`, `"layers.0.attn.k.weight"`)
2. **Find common prefix**: Identify the shared prefix up to a structural boundary (typically a numeric index like `"layers.0"`)
3. **Match untraced tensors**: Any untraced tensor whose name starts with that prefix belongs to that layer

```python
# Example prefix inference
traced_names = ["layers.0.attn.q.weight", "layers.0.attn.k.weight", "layers.0.ffn.w1.weight"]
# Inferred prefix: "layers.0"

# Untraced tensor "layers.0.attn.q.scale" matches prefix → belongs to layer "0"
```

### Why It's a Fallback

Prefix matching relies on naming conventions. It assumes:
- Tensor names follow a hierarchical structure
- Tensors in the same layer share a common prefix
- The prefix boundary is at a numeric index (like `layers.0`)

These assumptions hold for most transformer architectures but may fail for unconventional naming schemes or models with flat parameter hierarchies.

## Inner Tensor Fields

A cross-cutting concern across all strategies is handling "inner tensor fields"—tensors attached as attributes to other tensors.

### What They Are

Some quantization formats store metadata as tensor attributes:

```python
weight = torch.nn.Parameter(torch.randn(10, 10))
weight.scale = torch.tensor(1.0)  # Inner tensor field
weight.zero_point = torch.tensor(0)  # Another inner field
```

### How They're Discovered

All three strategies call `get_inner_tensor_field_ids()` after finding a parameter:

```python
def get_inner_tensor_field_ids(tensor):
    """Find tensor attributes that aren't part of base torch.Tensor."""
    inner_ids = []
    custom_fields = set(dir(tensor)) - set(dir(torch.Tensor))
    for field_name in custom_fields:
        if not field_name.startswith("_"):
            field = getattr(tensor, field_name, None)
            if isinstance(field, torch.Tensor):
                inner_ids.append(id(field))
    return inner_ids
```

This ensures that when a weight tensor is offloaded, its scale and other metadata tensors move together.

## Configuration

Both untraced tensor discovery and ModuleTracker are always enabled (hardcoded). No configuration is required. If you need to disable untraced tensor discovery for benchmarking or debugging, use the private `TensorManager` parameter `_enable_untraced_tensor_discovery`.

## Integration with FlexTensor Phases

Tensor discovery happens at a specific point in the discovery → profiling transition:

```
DISCOVERY PHASE
├── WarmupTrapDirect materializes known trap tensors in direct mode
├── WarmupTrap records operation-level tensor IDs in indirect mode
├── Traced tensors recorded via direct getters and __torch_function__
├── ModuleTracker records module → trap mapping (Strategy 2)
└── Discovery complete

TRANSITION TO PROFILING
├── discover_untraced_tensors_for_layers() called
│   ├── Strategy 1: Check for forward-patched modules
│   ├── Strategy 2: Query ModuleTracker for module mappings
│   └── Strategy 3: Prefix match remaining tensors
├── Layer statistics augmented with discovered tensors
├── ModuleTracker hooks removed (no longer needed)
└── Profiling phase begins with complete tensor mapping

PROFILING PHASE
└── All tensors (traced + discovered) properly loaded per layer
```

The discovery runs once, at the discovery-to-profiling boundary. By the time inference runs, FlexTensor has a complete picture of which tensors belong to which layers.

## Debugging Untraced Tensors

If you suspect tensor discovery issues:

### Symptoms

- CUDA device mismatch errors during inference
- Missing tensors (tensor is on CPU when kernel expects GPU)
- Memory not being reclaimed (tensors stuck on GPU)

### Verification

Check the tensor mapping after discovery:

```python
manager = get_offload_manager()
model = manager.offload(model, config)

# Run discovery. Needed when `skip_discovery=False`, and also when
# `skip_discovery=True` was requested but no patched modules were reachable —
# the manager then falls back to a real discovery phase and
# `skip_discovery_honored` is False. Only when the skip actually fired are
# layer stats built statically at `offload()` time and this loop unnecessary.
if not (config.skip_discovery and manager.skip_discovery_honored):
    for _ in range(config.discovery_iters):
        model(dummy_input)

# Examine layer statistics
# Note: _tensor_manager is an internal attribute intended for debugging only.
# It is not part of the public API and may change without notice.
layer_stats = manager._tensor_manager.layer_statistics_collector.get_layer_stats()
for stat in layer_stats:
    print(f"Layer {stat.label}: {len(stat.tensor_ids)} tensors")
```

If a layer has fewer tensors than expected, discovery may have missed some.

### Solutions

1. **Use forward patching** when possible—it's the most reliable strategy
2. **Verify module structure** ensures modules execute within the correct trap context
3. **Check tensor naming** follows hierarchical conventions if using prefix matching

## Summary

| Strategy | Trigger Condition | Accuracy | Performance |
|----------|-------------------|----------|-------------|
| Auto Trap (Forward Patching) | Model uses `offload()` API | Highest | Fast |
| Module Tracker | Manual traps + no forward patching | High | Moderate |
| Prefix Matching | Fallback when above don't cover all | Variable | Fast |

Tensor discovery ensures FlexTensor handles modern model architectures—including FP8 quantization and custom kernels—without requiring users to manually specify tensor-to-layer mappings. The multi-strategy approach provides robustness across different usage patterns while maintaining accuracy for the most common cases.
