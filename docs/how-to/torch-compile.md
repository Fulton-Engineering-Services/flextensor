<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use `torch.compile` with FlexTensor offload

FlexTensor's offloaded proxy composes with `torch.compile` under one
constraint: **discovery must run eagerly.** Profile and inference can run
under compile.

The trap boundaries around each offloaded block (``TrapDirect`` /
``TrapInferDirect``) begin and end with ``_graph_break()``. Dynamo compiles
the layer's tensor ops *between* the breaks; the loader's tensor movement
runs eagerly in resume functions between subgraphs. Profile timing
therefore reflects compiled-kernel execution.

## Supported flow — within a single process

```python
import torch
import flextensor as ft

model = MyModel().to("cpu")
config = ft.OffloadConfig(include_patterns=["layers.*"], ...)
proxy = ft.offload(model, config)

# Discovery: eager.  WarmupTrap is a TorchFunctionMode that does not compose
# with torch.compile — see "Why discovery cannot run under torch.compile"
# below.
for _ in range(config.discovery_iters):
    proxy(x)

# Profile + inference: compiled.
compiled = torch.compile(proxy, backend="inductor")
for _ in range(config.profiling_iters):
    compiled(x)          # profile under compile; timing reflects compiled kernels
compiled(x)              # inference under compile
```

## Supported flow — two processes with profile save / restore

**Process 1** — offload, drive discovery eagerly, compile, profile, save:

```python
import torch
import flextensor as ft

model = MyModel().to("cpu")
config = ft.OffloadConfig(include_patterns=["layers.*"], ...)
proxy = ft.offload(model, config)

for _ in range(config.discovery_iters):
    proxy(x)                          # discovery eager

compiled = torch.compile(proxy, backend="inductor")
for _ in range(config.profiling_iters):
    compiled(x)                       # profile under compile

ft.save_profile("/path/to/profile_dir")
```

**Process 2** — load the profile, compile immediately, infer:

```python
import torch
import flextensor as ft

model = MyModel().to("cpu")
# include_patterns / exclude_patterns must match the config used in
# Process 1 — they are not serialized into the saved profile.
config = ft.OffloadConfig(include_patterns=["layers.*"], ...)
proxy = ft.offload_from_profile(model, "/path/to/profile_dir", config)

compiled = torch.compile(proxy, backend="inductor")
compiled(x)                            # inference under compile
```

`offload_from_profile` skips discovery / profiling and hands the model
straight to inference-ready state. The next `compiled(x)` compiles once
and reuses the cached graph from then on.

!!! warning "Pattern fields must match across processes"
    The saved profile stores the *result* of offload planning (tensor
    names, block assignments, layer stats) but not `include_patterns` /
    `exclude_patterns`.  Process 2 must construct its `OffloadConfig`
    so that its patterns patch the same set of modules as Process 1.

    A divergent pattern set is **silently accepted** at load time —
    `validate_state_compatibility` only checks that the tensors named in
    the profile exist on the new model, not that the configs are
    equivalent.  The mismatch surfaces only at runtime, typically as a
    `KeyError` on the first forward (loader looks up a label for a
    module Process 2 never patched) or, in the absence of an early
    lookup, as wrong output from skipped transfers.  Reproduce the
    Process 1 `OffloadConfig` exactly to avoid this class of bug.

## `torch.compile` backends and modes

FlexTensor's role is to insert graph breaks at trap boundaries so that
`torch.compile` has something compilable between them — it does not
constrain which backend or mode you pass to `torch.compile`. Any
`backend=` / `mode=` combination supported by your PyTorch version can
be used; choosing among them is up to you and outside the scope of this
guide. The only hard constraint is the one below: discovery must stay
eager.

## Limits

- **Don't compile before discovery completes.** Calling `compiled(x)` while
  the manager is still in `DISCOVERY` raises `Unhandled FakeTensor Device
  Propagation for aten.mm.default` on the first op of the first patched
  layer (`WarmupTrap`'s `id()`-based staging is incompatible with Dynamo's
  FakeTensor tracer).
- **Don't compile across a phase transition.** `torch.compile` specializes
  on the underlying `nn.Module` live at compile time; FlexTensor swaps the
  proxy's underlying model on each transition (discovery → profile →
  inference), so Dynamo's guards fail and the next call triggers a full
  recompile under the new specialization. Compile *after* the transitions
  you care about, as the supported flows above show.

??? note "Why discovery cannot run under `torch.compile`"

    Discovery uses `WarmupTrap`, a `TorchFunctionMode` that intercepts each
    torch op and stages tensors from CPU to GPU on demand. Simplified excerpt
    (see `src/flextensor/trap_tensor_mode.py::WarmupTrap.__torch_function__`
    for the full implementation):

    ```python
    def __torch_function__(self, func, _types, args, kwargs=None):
        for arg in args:
            if self.tensor_manager.is_traced(arg) and arg.device != self.device_gpu:
                new_arg = arg.to(device=self.device_gpu, copy=True)
                torch.cuda.synchronize()
                ...
        return func(*new_args, **(new_kwargs or {}))
    ```

    Under Dynamo tracing this fails for two compounding reasons:

    1. `tensor_manager.is_traced(arg)` is `id(arg) in tracked_ids`. During
       tracing, `arg` is a `FakeTensor` with a different Python `id()` than
       the real parameter, so every lookup returns `False` and no substitution
       is recorded in the traced graph.
    2. `FakeTensorProp` then sees `aten.mm.default(gpu_input_fake, cpu_weight_fake)`
       and raises `Unhandled FakeTensor Device Propagation for aten.mm.default,
       found two different devices cuda:0, cpu`.

## Related tests

Integration tests exercising these flows live in
`tests/integration/L0_compile_profile_roundtrip/test_compile_profile_roundtrip.py`:

- `TestCompileWrappedProxy::test_user_two_process_flow` — the full save /
  restore round-trip above.
- `TestCompileWrappedProxy::test_compile_after_lifecycle_matches_eager` —
  single-process compile applied after the entire lifecycle.
- `TestCompileWrappedProxy::test_compile_on_restored_profile_matches_reference`
  — restored profile then compile.
