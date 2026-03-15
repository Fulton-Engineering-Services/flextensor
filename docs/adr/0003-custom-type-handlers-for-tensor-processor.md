<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# ADR-0003: Custom Type Handlers for TensorProcessor

**Date**: 2026-02-16

**Status**: Accepted

## Context

`TensorProcessor` is the core mechanism for walking a PyTorch model's attribute tree and applying per-tensor transformations (pinning to memory, moving to GPU, benchmarking, etc.).  The base class's `_apply_on()` method previously contained a hardcoded cascade of `isinstance` checks to decide how each module attribute should be processed:

```python
# Before (hardcoded in _apply_on)
if isinstance(attr_value, torch.Tensor):
    updated_attr = self.process(attr_value)
if isinstance(attr_value, set) and len(attr_value) > 0:
    updated_attr = set()
    for value in attr_value:
        updated_attr.add(self.process(value))
if isinstance(attr_value, collections.OrderedDict):
    ...
elif isinstance(attr_value, dict):
    ...
```

This design has several problems:

1. **Not extensible for new attribute types**: Real-world models (especially in vLLM, DeepSeek, and Megatron-LM) use custom data structures on modules — shared weight containers, exotic parameter wrappers, metadata objects holding tensors.  When `_apply_on()` encounters these, it silently skips the tensors inside them, leading to missed offloading, stale device placements, or crashes.

2. **No way for users to intervene**: The only extension point was subclassing `TensorProcessor` and overriding `_apply_on()` entirely — a fragile approach that forces users to duplicate the whole iteration and type-check logic and breaks whenever FlexTensor adds new built-in handling.

3. **`nn.Parameter` preservation was ad-hoc**: Operations like `tensor.to(device)` and `tensor.pin_memory()` strip the `nn.Parameter` wrapper, returning a plain `torch.Tensor`.  Each subclass had to remember to re-wrap parameters manually.  Missing this caused silent bugs in id()-based tensor tracking (`tensors_map`, `traced_tensors`).

4. **Hardcoded type checks are fragile**: Adding support for a single new container type (e.g. `NamedTuple`, `dataclass`) required editing the base class — violating the Open/Closed Principle and risking regressions in all existing subclasses.

## Decision

We will **replace the hardcoded `isinstance` cascade in `_apply_on()` with a type handler dispatch system**, allowing custom attribute types to be processed without modifying `TensorProcessor` internals.

### Core abstractions

1. **`TypeHandler` Protocol** — A lightweight interface with two methods:
   - `can_handle(value) -> bool` — returns `True` if the handler should process this attribute value.
   - `process_attribute(value, ctx) -> Any` — processes the value, using `ctx` for tensor operations.

2. **`ProcessingContext`** — A context object passed to handlers that wraps the owning processor and exposes safe helper methods (`process()`, `process_and_preserve()`, `dispatch()`).  This prevents handlers from depending on processor internals while still giving them full access to tensor processing capabilities including `nn.Parameter` preservation.

3. **Built-in type handlers** — The three formerly-hardcoded cases are refactored into public handler classes (`TensorTypeHandler`, `SetTypeHandler`, `DictTypeHandler`) that sit at the end of the dispatch chain.

### Handler priority chain

When `_apply_on()` processes a module attribute, handlers are checked in order:

1. **Instance-level custom handlers** (registered via `register_type_handler()` or the `type_handlers` constructor parameter)
2. **Global handlers** (registered via `TensorProcessor.register_global_type_handler()`)
3. **Built-in handlers** (tensor, set, dict)

The first handler whose `can_handle()` returns `True` wins.  Within each tier, later registrations take higher priority (inserted at index 0).

### `nn.Parameter` preservation

A new `preserve_parameter_type()` function centralizes the logic to re-wrap `nn.Parameter` after processing.  Two modes are supported via the `force_update_nn_parameters` constructor flag:

- **`False` (default)**: Updates `Parameter.data` in-place, preserving object identity for id()-based tracking.
- **`True`**: Creates a new `nn.Parameter`, used when building independent model copies (e.g. profile models) where identity should intentionally differ.

`ProcessingContext.process_and_preserve()` exposes this to handlers as a single convenient call.

### Registration API

```python
# Per-instance handler (highest priority)
processor = MoveToGPUTensorProcessor(device)
processor.register_type_handler(MyHandler())

# Or via constructor
processor = MoveToGPUTensorProcessor(device, type_handlers=[MyHandler()])

# Global handler (applies to all processors)
TensorProcessor.register_global_type_handler(MyHandler())
TensorProcessor.clear_global_type_handlers()  # cleanup
```

### Naming rationale

We chose **"Type Handler"** over alternatives:

- **"Plugin"**: Too generic; implies a broader extensibility architecture.  This is specifically about dispatching attribute processing by type.
- **"Custom Ops"**: Strongly collides with PyTorch's `torch.ops` / custom operator concept.  Would be confusing for PyTorch users.
- **"Attribute Processor"**: Collides with the existing `TensorProcessor` naming.
- **"Value Handler"**: Less descriptive — "type" signals the dispatch-by-type pattern.

"Type Handler" follows established patterns (Django form field handlers, marshmallow type serializers) and precisely describes what the abstraction does.

## Alternatives Considered

### Alternative 1: Override `_apply_on()` in subclasses

**Approach**: Each subclass that needs custom type handling overrides `_apply_on()` and adds its own `isinstance` checks alongside the base ones.

**Pros**:
- No new abstractions needed
- Standard inheritance pattern

**Cons**:
- Forces duplicating the entire iteration and type-check logic in each subclass
- Breaks whenever the base class adds new built-in type handling
- No way for users to add handling without subclassing every processor they use
- Violates Open/Closed Principle

**Why rejected**: Fragile and not user-extensible.  Users integrating custom model architectures (vLLM workers, DeepSeek inference) would need to subclass *every* processor variant.

### Alternative 2: `functools.singledispatch` based dispatch

**Approach**: Use Python's `singledispatch` to register processing functions per type.

**Pros**:
- Standard library mechanism
- Familiar pattern for Pythonistas
- Type-safe dispatch

**Cons**:
- Dispatches on exact type, not on predicates — cannot handle "any object with attribute X" patterns
- No support for priority ordering between handlers
- Global mutable state (dispatch table) with no instance-level overrides
- Awkward integration with `ProcessingContext` (no natural way to pass context)
- Cannot handle duck-typed custom containers common in ML frameworks

**Why rejected**: ML model attributes frequently use duck-typing and custom wrappers where predicate-based dispatch (`can_handle()`) is more appropriate than exact-type dispatch.

### Alternative 3: Visitor Pattern

**Approach**: Define a `TensorVisitor` with `visit_tensor()`, `visit_dict()`, `visit_set()`, etc. methods.

**Pros**:
- Well-known design pattern
- Clear separation of traversal and processing

**Cons**:
- Requires a `visit_*` method for every type, defeating extensibility for unknown types
- Adding a new type requires updating the visitor interface (same problem as hardcoded checks)
- Overly formal for the simple dispatch needed here
- No predicate-based matching for custom types

**Why rejected**: The Visitor pattern works well when the set of types is fixed and known.  Our problem is specifically that the set of types is *open* and determined by external model architectures.

### Alternative 4: Type Handler Dispatch (Selected)

**Approach**: A `TypeHandler` protocol with `can_handle()` / `process_attribute()` methods, checked in priority order (instance → global → built-in).

**Pros**:
- Open for extension without modifying base class (Open/Closed Principle)
- Predicate-based dispatch handles duck-typed and custom types
- Priority ordering gives users control over handler precedence
- `ProcessingContext` provides safe, documented access to processor capabilities
- Built-in handlers are just regular handlers, no special casing
- `Protocol` class provides structural typing without requiring inheritance
- Instance-level and global scopes cover both targeted and broad use cases

**Cons**:
- New abstraction to learn (`TypeHandler`, `ProcessingContext`)
- Linear scan of handlers per attribute (negligible for realistic handler counts)

**Why selected**: Best fit for the open-type problem.  Predicate dispatch is the right tool when the type set is unbounded.  The handler chain is simple to understand and extend.

## Consequences

### Positive

- **User-extensible**: Custom model architectures (vLLM shared weights, DeepSeek MoE containers, Megatron distributed parameters) can be handled without forking FlexTensor.
- **Cleaner `_apply_on()`**: The method body shrinks from a multi-branch `isinstance` cascade to a simple handler dispatch loop.
- **Centralised `nn.Parameter` preservation**: `preserve_parameter_type()` and `ProcessingContext.process_and_preserve()` eliminate a class of bugs where parameter wrappers were silently stripped.
- **Testable in isolation**: Each handler can be unit-tested independently via `ProcessingContext`.
- **Backward compatible**: Existing `TensorProcessor` subclasses continue to work unchanged — they only override `process()`, not `_apply_on()`.

### Negative

- **New concepts to learn**: Contributors need to understand `TypeHandler`, `ProcessingContext`, and the priority chain.  Mitigated by docstrings and the Protocol pattern being well-known in Python.
- **Global handler state**: `_global_type_handlers` is mutable class-level state.  Tests must call `clear_global_type_handlers()` in teardown.  This is an accepted trade-off for the convenience of framework-wide handler registration.

### Intentional behavioural improvement: set element processing

The original hardcoded set handling called `self.process(value)` directly on each element — no handler dispatch, no `nn.Parameter` preservation.  The new built-in `SetTypeHandler` routes elements through `ctx.dispatch()`, which:

1. Checks custom type handlers for each element (enabling recursive dispatch for exotic types nested inside sets).
2. Falls back to `process_and_preserve()` for tensors, which applies `nn.Parameter` re-wrapping.

This is an intentional improvement: the original omission of Parameter preservation in sets was an oversight, not a deliberate choice, and it was inconsistent with how dict values were handled.

**Escape hatch** — If the improved behaviour causes issues with a specific model, register `LegacySetTypeHandler` on the processor instance to restore the original `self.process()`-only semantics:

```python
from flextensor.tensor_processors import LegacySetTypeHandler

processor = MoveToGPUTensorProcessor(device)
processor.register_type_handler(LegacySetTypeHandler())
```

### Neutral

- **Performance**: The handler dispatch adds a linear scan per attribute.  With the typical 3 built-in + 0–2 custom handlers, this is negligible compared to tensor transfer times.
- **`_apply_on()` complexity shifts**: Instead of hardcoded branches, the complexity now lives in the handler chain assembly.  The total complexity is similar, just better organized.

## References

### Internal Code References

- Type handler implementation: [`flextensor/tensor_processors.py::TypeHandler`](../../src/flextensor/tensor_processors.py)
- `ProcessingContext` implementation: [`flextensor/tensor_processors.py::ProcessingContext`](../../src/flextensor/tensor_processors.py)
- Built-in handlers: [`flextensor/tensor_processors.py::TensorTypeHandler`, `SetTypeHandler`, `DictTypeHandler`](../../src/flextensor/tensor_processors.py)
- Handler dispatch in `_apply_on()`: [`flextensor/tensor_processors.py::TensorProcessor._apply_on`](../../src/flextensor/tensor_processors.py)
- Unit tests: [`tests/unit/test_tensor_processors.py::TestTypeHandlerSystem`](../../tests/unit/test_tensor_processors.py)

### External References

- Python `Protocol` class documentation: https://docs.python.org/3/library/typing.html#typing.Protocol
- Chain of Responsibility pattern: https://refactoring.guru/design-patterns/chain-of-responsibility
- PyTorch `nn.Parameter` documentation: https://pytorch.org/docs/stable/generated/torch.nn.Parameter.html
