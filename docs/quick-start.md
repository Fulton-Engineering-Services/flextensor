<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Quick Start

Get FlexTensor running in minutes. This guide covers installation, basic usage, and key concepts.

## Installation

Install FlexTensor from PyPI:

```bash
pip install flextensor
```

Verify CUDA is available:

```python
import torch
assert torch.cuda.is_available(), "CUDA required"
```

## How It Works

FlexTensor manages weight transfers automatically through a short learning phase (discovery and profiling) before switching to optimized inference. You don't manage these phases directly — FlexTensor handles them during the first few iterations.

For a deeper explanation of the discovery, profiling, and inference phases and the decisions made during each, see [Internal Phases](explanation/phases.md).

## Basic Usage

```python
import flextensor
from flextensor import OffloadConfig

# Your existing model
model = YourModel()

# Configure offloading
config = OffloadConfig(
    gpu_device=0,                   # GPU to use
    profiling_iters=10,             # Iterations for timing measurement
    include_patterns=["layers.*"],  # Which modules to offload
)

# Patch the model
model = flextensor.offload(model, config=config)

# Use normally — the first few forwards warm the manager
# (`discovery_iters` + `profiling_iters` under the default
# `skip_discovery=False`; set `skip_discovery=True` to skip discovery).
# `flextensor.get_offload_manager().iters_before_inference` is path-aware and
# always returns the right count if you need to know it.
for batch in dataloader:
    output = model(batch)  # FlexTensor handles everything (discovery→profile→inference)
```

!!! warning "Single-thread only"
    FlexTensor is **not thread-safe**. All stages — offloading setup, discovery, profiling, and inference — must run on the same thread. Do not call `offload()`, run forward passes on a patched model, or access the offload manager from multiple threads in parallel. If you need per-thread offloading, create a separate named manager and model per thread.

## Include Patterns

The `include_patterns` field in `OffloadConfig` specifies which modules to offload. Each entry is one of three forms:

| Pattern | Matches |
|---------|---------|
| `"layers.*"` | All modules under `model.layers` (name-based) |
| `"encoder.block_*"` | `encoder.block_0`, `encoder.block_1`, etc. |
| `"attention.?"` | Single-character suffixes like `attention.q` |
| `"class:SharedExpertMLP"` | Every module whose class is `SharedExpertMLP`, regardless of its path |

```python
config = OffloadConfig(
    include_patterns=[
        "embed",                    # Exact match (name)
        "layers.*",                 # Wildcard (name)
        "class:SharedExpertMLP",    # Class-based — useful for hybrid architectures
        "head",
    ],
)
model = flextensor.offload(model, config=config)
```

`exclude_patterns` accepts the same three forms and removes matching modules or parameters from the offload set — see [Exclude Patterns](how-to/exclude-patterns.md).

Include patterns can also be set via the `FT_INCLUDE_PATTERNS` environment variable as a comma-separated list:

```bash
FT_INCLUDE_PATTERNS="layers.*,embed,head,class:SharedExpertMLP" python my_script.py
```

## Key Configuration Options

The most commonly tuned options are:

- **`include_patterns`** — which modules to offload (supports `*` and `?` wildcards plus `class:` selectors, default `["*"]`; use specific patterns such as `class:*DecoderLayer` or `model.layers.*` for better per-layer pipelining)
- **`skip_discovery`** — skip the discovery phase and jump straight to profiling (default `False`; when `True`, the manager derives tensor-to-layer mappings statically from the patched modules). Leave at `False` if the model uses manual `offload_block()` blocks.
- **`profiling_iters`** — iterations for timing measurement (default `10`)
- **`discovery_iters`** — only consulted when `skip_discovery=False` (default `1`)

See [Configuration](explanation/configuration.md) for the full list of options and explanations.

## Profile Caching

Skip discovery/profiling on subsequent runs by saving and loading profiles:

```python
om = flextensor.get_offload_manager()

# First run: save profile after discovery completes
config = OffloadConfig(
    include_patterns=["layers.*"],
    profile_read_only=False,  # Allow saving profiles
)
model = om.offload(model, config=config)
for _ in range(om.iters_before_inference):
    model(sample_input)
om.save_profile("/tmp/profiles/my_model")

# Later runs: load profile, skip discovery/profiling
model = flextensor.offload_from_profile(
    model,
    "/tmp/profiles/my_model",
    config=config,
)
```

`offload_from_profile` combines `init`, `load_profile`, and `offload` into a single call —
the model is ready for inference immediately with no discovery or profiling overhead.

## Verify It's Working

```python
usage = flextensor.get_gpu_memory_usage()
print(f"GPU memory: {usage.total_mb:.1f} MB")
```

## Next Steps

- [Configuration](explanation/configuration.md) -- All options explained
- [Troubleshooting](how-to/troubleshooting.md) -- Debug issues
- [Internal Phases](explanation/phases.md) -- How the state machine works
- [Tensor Discovery](explanation/tensor-discovery.md) -- How untraced tensors are found
