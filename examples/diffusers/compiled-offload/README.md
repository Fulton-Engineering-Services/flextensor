<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Compiled Offload with FlexTensor

This example runs a diffusion transformer with FlexTensor **compiled offload**: the
same block-by-block weight streaming as the [quickstart](../quickstart/), but each
offloaded unit is also `torch.compile`d. It uses
[Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
with the Hugging Face [Diffusers](https://github.com/huggingface/diffusers) library.

## Quick Start

```bash
pip install -r requirements.txt

# Default: compile_fn + compiled view-profile (no replan)
python wan_t2v_compiled.py

# External torch.compile after INFERENCE + replan
python wan_t2v_external_compile.py
```

## Recommended path: `compile_fn`

For most users, compiled offload is enabled by **one argument** — `compile_fn` — on
the existing `flextensor.offload()` call. There is no new entry point and teardown
stays `flextensor.release()`.

```python
def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
    return torch.compile(module, fullgraph=True)

pipe.transformer = flextensor.offload(
    pipe.transformer,
    config=offload_config,
    name="transformer",
    compile_fn=compile_fn,   # <-- the whole compiled-offload API
)
```

Everything else — `include_patterns`, `OffloadConfig`, and the driving loop — is
identical to `../quickstart/wan_t2v.py`. Omitting `compile_fn` gives exactly the
plain eager offload from the quickstart, with no behavior change.

With `compile_fn` and `profile_mode='view'` (default), FlexTensor **auto-enables
compiled view-profile** (profile under `compile_fn`), so the offload strategy is
built from compiled timings and **no** `request_strategy_replan()` is needed.
Call `request_strategy_replan()` only after eager/direct profiling or external
`torch.compile`.

`compile_fn` is any callable `module -> module`. FlexTensor calls it **once per
offloaded unit** at the INFERENCE transition (or during view profile), so each
unit becomes its own compiled graph.

## Advanced: external `torch.compile`

Use this when compile already lives **outside** FlexTensor — e.g. your pipeline calls
`torch.compile` itself and you need FlexTensor to stay out of that step.

```bash
python wan_t2v_external_compile.py
```

The script sets `external_compile=True` (no `compile_fn`), runs one eager `pipe(...)`,
compiles each block, calls `request_strategy_replan()`, runs one more `pipe(...)` to
finish the re-plan tail (many transformer forwards per call), then exports from a
third steady-state run.

Prefer `compile_fn` (`wan_t2v_compiled.py`) in general: FlexTensor compiles **one
offloaded unit per graph** (per-block, not whole-model), so compile granularity
matches offload granularity and rolling weight slots never alias inside a graph —
use external compile only when `torch.compile` must stay outside FlexTensor.

## Backend independence

`compile_fn` is not tied to `torch.compile`. Any `module -> module` callable works:

```python
# Torch-TensorRT:
compile_fn = lambda m: torch_tensorrt.compile(m, ir="dynamo")

# Bring-your-own tuner:
compile_fn = my_tuner.optimize

# Identity compiler — no torch.compile, but still uses the compiled-offload
# custom-op path (not plain eager offload; omit compile_fn for that):
compile_fn = lambda m: m
```

It can also be a full multi-line function that branches per unit (e.g. skip tiny
modules by returning them unchanged, pick a backend by module type, or wrap in
`try/except` to fall back to eager for any unit that fails to compile).

## Prerequisites

- A CUDA-capable GPU. Only non-transformer components (VAE, text encoder) stay
  resident on the GPU; transformer weights are streamed from CPU.
- Model weights are downloaded automatically from Hugging Face Hub on first run.

## Troubleshooting

- **Corrupt video with whole-transformer compile.** Expected under offload — compile
  per block, not the whole transformer.
- **Graph breaks under `fullgraph=True`.** Small functional modules can occasionally
  fail to trace to a single graph. Either drop `fullgraph=True`, or wrap the compile
  in `try/except` and return the module unchanged so that one unit falls back to
  eager while the rest stay compiled.
- General offload tuning (out of memory, performance) is covered in the
  [Troubleshooting Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/how-to/troubleshooting.md).
