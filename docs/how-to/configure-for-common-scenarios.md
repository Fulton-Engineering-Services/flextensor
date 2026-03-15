<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# How to Configure FlexTensor for Common Scenarios

This guide shows you how to apply starting configurations for common deployment scenarios. It assumes you are already familiar with `OffloadConfig` and the basic `offload()` workflow. If you are new to FlexTensor, start with the [Quick Start](../quick-start.md) first.

## Prerequisites

- FlexTensor installed
- A PyTorch model ready to offload
- Basic familiarity with `OffloadConfig` options (see [Configuration](../explanation/configuration.md) for full option reference)

---

## Memory-Constrained Systems

Use this configuration when GPU memory is the primary constraint and you need to maximize memory savings, even at the cost of some throughput.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    release_tensors=True,       # Release GPU memory promptly
    pinned_memory=True,         # Fast CPU-GPU transfers
    profile_iters=15,           # Accurate profiling
    max_gpu_mem_fraction=0.8,   # Use at most 80% of total GPU memory
    min_blocks=2,               # Fewer blocks to save GPU memory
)

model = offload(model, config=config)
```

Key choices:

- `release_tensors=True` frees GPU memory after each layer executes, reducing peak usage.
- `pinned_memory=True` uses page-locked CPU memory to accelerate transfers via Direct Memory Access (DMA).
- `max_gpu_mem_fraction=0.8` switches the strategy to memory mode, keeping peak GPU usage within 80% of total device memory. Using a fraction rather than a byte count makes the config portable across GPU SKUs.
- `min_blocks=2` lets the optimizer try fewer memory blocks, reducing GPU memory consumed by FlexTensor itself.

---

## Performance-Focused

Use this configuration when latency matters more than memory savings and you want to minimize the overhead that offloading adds to the model forward pass.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    transfer_mode="allocation_block_transfer",
    num_blocks=8,               # More blocks for parallelism
    enable_direct_mode=True,    # Lower overhead
    rearrange_transfers=True,   # Optimize transfer scheduling
)

model = offload(model, config=config)
```

Key choices:

- `num_blocks=8` increases the number of pre-allocated GPU memory blocks, allowing more transfers to overlap with computation.
- `rearrange_transfers=True` lets FlexTensor reorder transfer scheduling to better hide latency.

---

## Quick Experimentation

Use this configuration during development when you need fast iteration and startup time, and profiling accuracy is less important.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    warmup_iters=1,
    profile_iters=3,            # Minimal profiling for fast startup
)

model = offload(model, config=config)
```

Key choices:

- `profile_iters=3` reduces the startup cost before reaching inference state.
- This produces a less accurate offloading strategy than higher iteration counts, so it is not recommended for production.

---

## Production Deployment

Use this configuration when deploying to a stable environment where startup time is less critical and you want accurate, reproducible offloading behavior.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    warmup_iters=1,
    profile_iters=20,           # Accurate initial profiling
)

model = offload(model, config=config)
```

Key choices:

- `profile_iters=20` gives the profiler more timing samples, which is especially important on shared or cloud GPUs where timing noise is higher.

!!! tip "Use profile persistence for faster startup"
    Save profiles after the initial warmup/profile run using `save_profile()`, then load them on subsequent runs with `load_profile()` to skip the warmup and profile phases entirely. Set `profile_storage_dir` in your config and ensure `profile_read_only=False` to enable saving.

---

## Next Steps

- [Configuration reference](../explanation/configuration.md) — full description of every `OffloadConfig` option
- [Troubleshooting](troubleshooting.md) — diagnose performance and memory issues
