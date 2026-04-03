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

FlexTensor manages weight transfers automatically through a short learning phase (warmup and profile) before switching to optimized inference. You don't manage these phases directly — FlexTensor handles them during the first few iterations.

For a deeper explanation of the warmup, profile, and inference states and the decisions made during each, see [Internal States](explanation/states.md).

## Basic Usage

```python
import flextensor
from flextensor import OffloadConfig

# Your existing model
model = YourModel()

# Configure offloading
config = OffloadConfig(
    gpu_device=0,              # GPU to use
    warmup_iters=1,            # Iterations for parameter discovery
    profile_iters=10,          # Iterations for timing measurement
    include_patterns=["layers.*"],  # Which modules to offload
)

# Patch the model
model = flextensor.offload(model, config=config)

# Use normally - first warmup_iters + profile_iters iterations are warmup/profile
for batch in dataloader:
    output = model(batch)  # FlexTensor handles everything
```

!!! warning "Single-thread only"
    FlexTensor is **not thread-safe**. All stages — offloading setup, warmup, profiling, and inference — must run on the same thread. Do not call `offload()`, run forward passes on a patched model, or access the offload manager from multiple threads in parallel. If you need per-thread offloading, create a separate named manager and model per thread.

## Module Path Patterns

The `include_patterns` field in `OffloadConfig` specifies which modules to offload using path patterns:

| Pattern | Matches |
|---------|---------|
| `"layers.*"` | All modules under `model.layers` |
| `"encoder.block_*"` | `encoder.block_0`, `encoder.block_1`, etc. |
| `"attention.?"` | Single-character suffixes like `attention.q` |

```python
config = OffloadConfig(
    include_patterns=[
        "embed",           # Exact match
        "layers.*",        # Wildcard
        "head",
    ],
)
model = flextensor.offload(model, config=config)
```

Module patterns can also be set via the `FT_INCLUDE_PATTERNS` environment variable as a comma-separated list:

```bash
FT_INCLUDE_PATTERNS="layers.*,embed,head" python my_script.py
```

## Key Configuration Options

The most commonly tuned options are:

- **`include_patterns`** — which modules to offload (supports `*` and `?` wildcards, default `["*"]`; use specific patterns such as `model.layers.*` for better per-layer pipelining)
- **`warmup_iters`** — iterations for tensor discovery (default `1`)
- **`profile_iters`** — iterations for timing measurement (default `10`)

See [Configuration](explanation/configuration.md) for the full list of options and explanations.

## Profile Caching

Skip warmup/profile on subsequent runs by saving and loading profiles:

```python
om = flextensor.get_offload_manager()

# First run: save profile after warmup completes
config = OffloadConfig(
    include_patterns=["layers.*"],
    profile_read_only=False,  # Allow saving profiles
)
model = om.offload(model, config=config)
for _ in range(config.warmup_iters + config.profile_iters):
    model(sample_input)
om.save_profile("/tmp/profiles/my_model")

# Later runs: load profile, skip warmup/profile
model = flextensor.offload_from_profile(
    model,
    "/tmp/profiles/my_model",
    config=config,
)
```

`offload_from_profile` combines `init`, `load_profile`, and `offload` into a single call —
the model is ready for inference immediately with no warmup or profiling overhead.

## Verify It's Working

```python
usage = flextensor.get_gpu_memory_usage()
print(f"GPU memory: {usage.total_mb:.1f} MB")
```

## Next Steps

- [Configuration](explanation/configuration.md) -- All options explained
- [Troubleshooting](how-to/troubleshooting.md) -- Debug issues
- [Internal States](explanation/states.md) -- How the state machine works
- [Tensor Discovery](explanation/tensor-discovery.md) -- How untraced tensors are found
