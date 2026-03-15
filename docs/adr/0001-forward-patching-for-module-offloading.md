<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# ADR-0001: Forward Patching for Module Offloading

**Date**: 2026-01-23

**Status**: Accepted

## Context

FlexTensor provides tensor offloading capabilities by intercepting module forward passes to wrap them with offload context managers. The initial implementation used an `OffloadModule` wrapper class that inherits from `nn.Module` and wraps target modules.

However, this wrapper approach has several significant issues:

1. **Hierarchy changes**: When a module is wrapped, `model.layers[0]` returns an `OffloadModule` instance instead of the actual layer (e.g., `TransformerBlock`). This breaks code that navigates the model hierarchy expecting specific types.

2. **isinstance checks fail**: Framework code that uses `isinstance(module, SpecificLayerType)` to identify layer types fails because the module is wrapped in `OffloadModule`, which is not a subclass of the original layer type. This is critical for vLLM integration, where the framework extensively checks layer types:
   - Pipeline parallelism checks: [`vllm/model_executor/models/utils.py:255`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/utils.py#L255) - `isinstance(module, (StageMissingLayer, PPMissingLayer))`
   - MoE layer detection: [`vllm/model_executor/models/qwen3_moe.py:679-683`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py#L679-L683)
   - Our tests demonstrate this works with patching: [`test_forward_patching.py::test_patched_module_preserves_isinstance`](../../tests/unit/test_forward_patching.py)

3. **Complex attribute delegation**: The wrapper requires custom `__getattr__` implementation to delegate attribute access to the wrapped module. This is error-prone and doesn't handle all edge cases:
   - Bug discovered with vLLM's `quant_method` attribute access - see deleted test file that documented this issue
   - PyTorch's `__getattr__` delegation must handle `_modules`, `_parameters`, and `_buffers` specially to avoid recursion

4. **Serialization complications**: `state_dict()` and `torch.save()` require special handling because the wrapper adds an extra layer to the module hierarchy. With wrappers, `state_dict()` keys include the wrapper (`module.layer1.module.weight` instead of `module.layer1.weight`). Our tests verify patching preserves this: [`test_forward_patching.py::test_patched_module_state_dict`](../../tests/unit/test_forward_patching.py).

5. **Debug complexity**: Stack traces and model inspection tools show `OffloadModule` wrappers rather than the actual module types, making debugging harder.

The FlexTensor team explored forward patching in earlier experiments (note: the `experiments/` directory has since been removed), where forward patching was used successfully, but the production implementation initially chose wrappers for perceived simplicity.

## Decision

We will **replace the `OffloadModule` wrapper with direct forward method patching**.

Implementation approach:

1. **Patch the forward method**: Replace `module.forward` with a method that wraps the original forward in an offload context manager
2. **Store original for restoration**: Save the original forward as `module._ft_original_forward_func` (unbound class method) to enable cleanup
3. **Track patched modules**: Maintain a list of patched modules in `OffloadManager._patched_modules` for proper cleanup during `release()`
4. **Use unbound class methods**: Store the unbound forward function from the class, not a bound method, to ensure correct behavior when models are copied during state transitions
5. **Use types.MethodType**: Bind the patched forward as a method so it receives `self` correctly

Key implementation details:

```python
def _patch_module_forward(self, module: nn.Module, offload_name: str) -> None:
    """Patch module's forward to include offload context."""
    import functools
    import types

    # Get the UNBOUND forward function from the class, not a bound method.
    # This is critical: bound methods capture `self`, which breaks when models
    # are copied during state transitions (warmup → profile → inference).
    original_forward_func = type(module).forward
    offload_manager = self

    def patched_forward(self_module, *args, **kwargs):
        with offload_manager.offload_block(offload_name):
            # Call the unbound forward function with self_module explicitly
            return original_forward_func(self_module, *args, **kwargs)

    patched_forward = functools.wraps(original_forward_func)(patched_forward)

    # Store original for restoration (unbound function, not bound method)
    module._ft_original_forward_func = original_forward_func
    module._ft_offload_name = offload_name

    # Bind patched_forward as a method to the module
    module.forward = types.MethodType(patched_forward, module)
```

**Why unbound functions instead of bound methods?**

FlexTensor transitions through multiple states (warmup → profile → inference), and during these transitions, the model may be copied or replaced with a prepared version that has property getters for tensor offloading. If we captured a bound method in the closure, it would remain bound to the OLD module even after the model is replaced, causing tensor accesses to use the wrong module (one without property getters). By using unbound functions and binding `self` dynamically via `types.MethodType`, the patched forward always uses the correct current module.

This approach is simpler, more direct, and avoids the complexity of wrapper objects while correctly handling model state transitions.

## Alternatives Considered

We evaluated three main approaches for intercepting module forward passes:

### Alternative 1: Module Wrapper (Current/Original Implementation)

**Approach**: Create an `OffloadModule` class that inherits from `nn.Module` and wraps the target module, delegating calls through `__getattr__`.

**Pros**:
- Object-oriented and feels "Pythonic" with a dedicated class
- Clear encapsulation of offload logic
- Easy to identify wrapped modules via `isinstance(module, OffloadModule)`

**Cons**:
- Changes model hierarchy (`model.layers[0]` returns `OffloadModule`, not actual layer)
- Breaks `isinstance(module, OriginalType)` checks - wrapper is not a subclass of the original module type
- Requires complex `__getattr__` delegation to forward attribute access to wrapped module
- Complicates serialization - `state_dict()` keys include wrapper hierarchy
- Harder to debug - stack traces show `OffloadModule.forward` instead of actual layer name
- Issues with frameworks that inspect module types (vLLM's `PPMissingLayer` checks fail)

**Why rejected**: The hierarchy changes and `isinstance` breakage are fundamental issues that complicate integration with external frameworks. The delegation complexity adds maintenance burden.

### Alternative 2: PyTorch Hooks

**Approach**: Use `register_forward_pre_hook()` and `register_forward_hook()` to wrap execution in offload context.

**Pros**:
- Official PyTorch API with [documented behavior](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
- Preserves model hierarchy completely
- Hooks can be registered/removed dynamically
- No custom `__getattr__` or class wrapping needed
- Works well with PyTorch internals

**Cons**:
- Context manager pattern is awkward (enter in pre-hook, exit in post-hook)
- Need to store context manager state between hooks (e.g., in dict)
- Hooks are called for every forward pass, adding slight overhead
- Pre-hook and post-hook separation makes code harder to follow
- Error handling between pre/post hooks is complex

**Why rejected**: While technically sound, the split between pre/post hooks makes the context manager pattern awkward. The need to manage state between hooks adds complexity without clear benefits over direct patching.

### Alternative 3: Dynamic Subclassing

**Approach**: Dynamically change the module's `__class__` to a subclass that overrides `forward()`.

**Pros**:
- Preserves most hierarchy aspects
- `isinstance` checks still work (subclass relationship maintained)
- No delegation needed
- Can override other methods if needed

**Cons**:
- Unusual Python pattern (changing `__class__` at runtime)
- Creates new classes dynamically (memory overhead)
- Potential issues with pickling/serialization
- May confuse static analysis tools
- Less clear in debugging (shows dynamically created class)
- Unclear interaction with `torch.compile` and JIT

**Why rejected**: While technically possible, changing `__class__` at runtime is unconventional and could cause subtle issues with serialization, introspection tools, and compilation. The uncertainty around edge cases outweighs the benefits.

### Alternative 4: Forward Method Patching (Selected)

**Approach**: Replace `module.forward` with a closure that wraps the original forward in an offload context manager.

**Pros**:
- Simplest implementation (direct function replacement with closure)
- Preserves model hierarchy completely (module identity unchanged)
- `isinstance` checks work perfectly (module class unchanged)
- No attribute delegation needed (all attributes directly accessible)
- Native serialization works without changes (`state_dict()` keys unchanged)
- Clear in stack traces (shows actual module type)
- Well-established pattern in Python (decorators, monkey-patching) and used internally by PyTorch hooks
- Easy to restore original behavior (store `_ft_original_forward_func`)
- Compatible with `functools.wraps` for preserving function metadata

**Cons**:
- Function assignment is less "object-oriented"
- Type checkers may warn (suppressed with `# type: ignore`)
- Less conventional than subclassing

**Why selected**: This approach provides the best balance of simplicity, correctness, and compatibility. It directly solves the hierarchy and `isinstance` issues without introducing new complexity. The pattern is well-established in Python (decorators, hooks) and PyTorch internals.

## Consequences

### Positive

- **Preserves model hierarchy**: `model.layers[0]` returns the actual layer, not a wrapper. Model navigation works as expected.
- **isinstance checks work**: Framework code using `isinstance(module, LayerClass)` continues to work correctly.
- **No attribute delegation needed**: All attributes are directly accessible without custom `__getattr__` logic.
- **Native serialization**: `state_dict()`, `torch.save()`, and `torch.load()` work without special handling.
- **Better debugging**: Stack traces and introspection show actual module types, not wrappers.
- **Simpler code**: Less code to maintain, fewer edge cases to handle.
- **torch.compile compatibility**: More likely to work with compilation modes that inspect module structure.

### Negative

- **Breaking change**: Code that imports `OffloadModule` directly will break. This is acceptable because:
  - `OffloadModule` was not documented as part of the public API
  - The high-level `flextensor.offload()` API remains unchanged
  - Internal use is limited to FlexTensor itself
- **Function assignment in Python**: Assigning to `module.forward` is less conventional than subclassing, though it's a well-established pattern (used in PyTorch hooks, decorators, etc.)
- **Type checking challenges**: Static type checkers may warn about assigning to `forward` method. We suppress these with `# type: ignore[method-assign]` comments.

### Neutral

- **Different debugging experience**: When stepping through code, debuggers will show the patched closure rather than a class. This is neither better nor worse, just different.
- **Restoration required**: We must track and restore original forwards during cleanup. The wrapper approach also required cleanup (removing wrappers from parent modules).
- **Attribute storage**: Using `_ft_*` prefixed attributes on modules is unconventional but safe. Alternative approaches (WeakKeyDictionary) would be more complex.

## References

### Internal Code References

- Forward patching implementation: [`flextensor/offload_manager.py::_patch_module_forward`](../../src/flextensor/offload_manager.py)
- Test suite for patching behavior: [`tests/unit/test_forward_patching.py`](../../tests/unit/test_forward_patching.py)
- vLLM integration using patching: [`flextensor/contrib/vllm/worker.py`](../../src/flextensor/contrib/vllm/worker.py)
- Original experiment with forward patching: `experiments/202504-vllm_model_offload/README.md` (note: this file has been removed)

### External References

- PyTorch Module hooks documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- PyTorch `state_dict` documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict
- vLLM model layer type checking examples:
  - Pipeline parallelism checks: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/utils.py#L255
  - MoE layer detection: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py#L679-L683
- Python `functools.wraps` for preserving function metadata: https://docs.python.org/3/library/functools.html#functools.wraps
