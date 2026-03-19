<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Understanding FlexTensor Configuration

This document explains the configuration system in FlexTensor—how it works, what each option controls, and how to make informed decisions about configuring tensor offloading for your workload.

## Overview

FlexTensor uses `OffloadConfig` to control all aspects of tensor offloading behavior. The configuration system is designed around three principles:

1. **Sensible defaults**: Works out-of-the-box for most use cases
2. **Progressive disclosure**: Simple cases need few options; advanced tuning is available when needed
3. **Multiple sources**: Configure via Python code, files, or environment variables

```python
from flextensor import OffloadConfig, offload

# Minimal configuration - uses sensible defaults
config = OffloadConfig()

# Tuned configuration for specific requirements
config = OffloadConfig(
    gpu_device=0,
    warmup_iters=1,
    profile_iters=10,
)

model = offload(model, config=config)
```

!!! warning "Single-thread only"
    FlexTensor is **not thread-safe**. The entire offloading lifecycle — setup, warmup, profiling, and inference — must run on one thread. Do not call `offload()`, run forward passes on a patched model, or access the same offload manager from multiple threads concurrently. If you need per-thread offloading, use a separate named manager and model per thread (see [Troubleshooting](../how-to/troubleshooting.md#fix-cross-thread-manager-access-errors)).

## Configuration Loading

FlexTensor supports three ways to load configuration, with clear precedence rules.

### Precedence Order

When multiple sources provide the same option, higher precedence wins:

1. **Explicit kwargs** (highest) — `OffloadConfig(gpu_device=1)`
2. **Environment variables** — `FT_GPU_DEVICE=1`
3. **Configuration file** — `gpu_device = 1` in file
4. **Default values** (lowest) — Built-in defaults

### Loading from Environment Variables

Environment variables use the `FT_` prefix by default:

```bash
export FT_ENABLED=1
export FT_GPU_DEVICE=0
export FT_WARMUP_ITERS=2
```

```python
from flextensor import load_config_from_env

config = load_config_from_env()  # Reads FT_* variables
```

!!! note "Environment-only Loading"
    When loading from environment variables without a config file, `enabled` defaults to `False`. You must explicitly set `FT_ENABLED=1` to enable offloading. This prevents accidental offloading when environment variables are partially set.

### Loading from Files

FlexTensor supports INI, JSON, and YAML formats:

=== "INI Format"

    ```ini
    ; flextensor.conf
    [flextensor]
    enabled = true
    gpu_device = 0
    warmup_iters = 1
    profile_iters = 10
    ```

=== "YAML Format"

    ```yaml
    # flextensor.yaml
    enabled: true
    gpu_device: 0
    warmup_iters: 1
    profile_iters: 10
    ```

=== "JSON Format"

    ```json
    {
      "enabled": true,
      "gpu_device": 0,
      "warmup_iters": 1,
      "profile_iters": 10
    }
    ```

```python
from flextensor import load_config, load_config_from_file

# Auto-detect format from extension
config = load_config("flextensor.yaml")

# Or use the file-only loader (no env override)
config = load_config_from_file("flextensor.conf")
```

### Combined Loading

The `load_config()` function combines all sources with proper precedence:

```python
from flextensor import load_config

# File + env vars + kwargs (precedence: kwargs > env > file > defaults)
config = load_config(
    config_path="flextensor.yaml",  # Base configuration
    use_env=True,                     # Override with FT_* env vars
    gpu_device=1,                     # Override with explicit value
)
```

## Core Configuration Options

### Enabling Offloading

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `True` | Master switch for offloading |
| `gpu_device` | int | `0` | GPU device index to use |
| `module_patterns` | list[str] | `["*"]` | Module path patterns to offload (supports `*` and `?` wildcards). Default `["*"]` works for quick experimentation; use specific patterns (e.g., `model.layers.*`) for transformer pipelining. |

#### Module Patterns

The `module_patterns` option specifies which modules in the model to offload. Patterns support `*` (match any sequence) and `?` (match a single character) wildcards:

```python
config = OffloadConfig(
    module_patterns=["layers.*", "embed", "head"],
)
model = offload(model, config=config)
```

Patterns can also be set via environment variable as a comma-separated list:

```bash
FT_MODULE_PATTERNS="layers.*,embed,head" python my_script.py
```

!!! tip "vLLM worker default patterns"
    The `FlexTensorOffloadWorker` uses these patterns by default when no custom `FT_MODULE_PATTERNS` is set. They are designed for decoder-only transformer layouts as served by vLLM:

    ```python
    config = OffloadConfig(
        module_patterns=[
            "model.embed_tokens",
            "model.layers.*",
            "model.norm",
            "lm_head",
            "logits_processor",
        ]
    )
    ```

    These patterns give each transformer layer its own offload trap, enabling FlexTensor to overlap CPU→GPU transfers with GPU computation. For models with a different layout, inspect module names with `model.named_modules()` and set `FT_MODULE_PATTERNS` accordingly.

The default `["*"]` matches all top-level child modules. It is a reasonable starting point for quick experimentation, but for transformer models in production, prefer specific patterns such as `model.layers.*`. With `["*"]`, every top-level child is wrapped in a single coarse trap, which prevents per-layer pipelining. With patterns like `model.layers.*`, each layer gets its own trap, and FlexTensor can overlap CPU→GPU transfers for layer N+1 while the GPU computes layer N.

When `enabled=False`, FlexTensor passes through to normal PyTorch execution with no overhead. This is useful for:

- Toggling offloading in production without code changes
- A/B testing offloading vs. baseline
- Disabling offloading on machines with sufficient GPU memory

### Memory Management

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pinned_memory` | bool | `True` | Use pinned memory for CPU tensors |
| `shm_enabled` | bool | `False` | Enable cross-process weight sharing via POSIX shared memory |
| `shm_namespace` | str \| None | `None` | Base namespace for SHM blocks (auto-derived if None) |
| `shm_wait_timeout` | float | `0.0` | Hard timeout (seconds) for followers waiting on creator |
| `release_tensors` | bool | `True` | Release GPU tensors after layer execution |
| `max_gpu_mem_fraction` | float \| None | `0.9` | GPU memory budget as a fraction of total device memory (e.g. `0.9` = 90%). `None` = latency mode (no constraint). |
| `max_gpu_mem_bytes` | int \| None | `None` | Deprecated since v0.4 — use `max_gpu_mem_fraction` instead. Will be removed in v0.5. |

#### Pinned Memory

When `pinned_memory=True`, CPU tensors use page-locked memory, which enables:

- Faster CPU↔GPU transfers (DMA — Direct Memory Access — without CPU involvement)
- Overlapping computation with transfers

The trade-off is increased CPU memory usage and allocation time. Disable for systems with limited CPU memory:

```python
config = OffloadConfig(pinned_memory=False)  # Lower CPU memory overhead
```

#### GPU Memory Limit

`max_gpu_mem_fraction` controls how much of the GPU's total memory FlexTensor may use. It accepts a `float` in the range `(0.0, 1.0]`, where `0.9` means "use at most 90% of total device memory."

When set to a fraction, the strategy operates in *memory mode* and keeps peak GPU usage within that budget. The fraction is resolved to an absolute byte count at runtime via `torch.cuda.get_device_properties()`, so the same config works portably across GPU SKUs with different memory capacities.

```python
config = OffloadConfig(
    max_gpu_mem_fraction=0.9,  # Default: 90% of total GPU memory
)
```

When set to `None`, the strategy switches to *latency mode* and optimises for throughput based on the `knapsack_scale` factor, with no explicit memory cap:

```python
config = OffloadConfig(
    max_gpu_mem_fraction=None,  # Latency mode: no memory constraint
)
```

!!! note "Default changed in v0.4"
    Before v0.4, `max_gpu_mem_fraction` did not exist and `max_gpu_mem_bytes` defaulted to `None`
    (latency mode). As of v0.4, the default is `0.9` (memory mode). If your workload previously
    relied on latency mode by default, set `max_gpu_mem_fraction=None` explicitly.

This option can also be set via the `FT_MAX_GPU_MEM_FRACTION` environment variable:

```bash
FT_MAX_GPU_MEM_FRACTION=0.8   # Use 80% of GPU memory (memory mode)
FT_MAX_GPU_MEM_FRACTION=none   # Switch to latency mode (also accepts "null" or "")
```

!!! warning "Deprecated: `max_gpu_mem_bytes`"
    `max_gpu_mem_bytes` and its env var `FT_MAX_GPU_MEM_BYTES` are deprecated since v0.4 and will
    be removed in v0.5. Replace any use of `max_gpu_mem_bytes=N` with
    `max_gpu_mem_fraction=N / total_gpu_bytes`, or use `max_gpu_mem_fraction=0.9` to target 90%
    of device memory portably.

#### Shared Memory (Cross-Process Weight Sharing)

When `shm_enabled=True`, FlexTensor stores model weights in POSIX shared memory so
multiple processes (e.g., vLLM data-parallel replicas) can share a single copy of the
weights in CPU RAM. The first process to start becomes the *creator* and loads the model
normally; subsequent processes become *followers* and attach to existing shared memory
blocks without re-loading from disk.

```python
config = OffloadConfig(shm_enabled=True)  # Enable cross-process weight sharing
```

The `shm_namespace` is auto-derived from the model path, config fields, and manager
name, ensuring that different models, config variations, or named `OffloadManager`
instances get separate shared memory regions. Override with an explicit namespace when
needed:

```python
config = OffloadConfig(shm_enabled=True, shm_namespace="my_model_v1")
```

When `shm_wait_timeout` is `0.0` (the default), followers rely on heartbeat liveness
detection rather than a hard wall-clock timeout.

These options can also be set via `FT_SHM_ENABLED`, `FT_SHM_NAMESPACE`, and
`FT_SHM_WAIT_TIMEOUT` environment variables.

### Profiling Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `warmup_iters` | int | `1` | Iterations to discover tensor-layer relationships |
| `profile_iters` | int | `10` | Iterations to measure execution timing |

FlexTensor learns your model's behavior during initial iterations:

```
First N iterations:  WARMUP (warmup_iters) → PROFILE (profile_iters) → INFERENCE
Remaining iterations: Optimized execution with learned strategy
```

#### Tuning Iteration Counts

**`warmup_iters`**: Usually 1 is sufficient. Increase if your model has:

- Dynamic control flow affecting tensor access patterns
- Variable-length inputs that change which tensors are used

**`profile_iters`**: More iterations = more accurate timing estimates. Consider:

- **Noisy environments**: Increase to 20-50 for shared/cloud GPUs
- **Deterministic workloads**: 5-10 is often sufficient
- **Quick experimentation**: Reduce to 3-5 for faster iteration

```python
# Production deployment (accurate profiling)
config = OffloadConfig(warmup_iters=1, profile_iters=20)

# Development/debugging (fast iteration)
config = OffloadConfig(warmup_iters=1, profile_iters=3)
```

### Transfer Modes

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `transfer_mode` | str | `"allocation_block_transfer"` | Tensor transfer strategy |
| `num_blocks` | int | `4` | Number of memory blocks for block transfer |
| `min_blocks` | int | `4` | Minimum blocks for assignment optimization |
| `enable_direct_mode` | bool | `True` | Use direct tensor access (lower overhead) |

FlexTensor supports different transfer strategies:

| Mode | Description | Best For |
|------|-------------|----------|
| `strategy` | Basic strategy-based loading | Simple models, debugging |
| `allocation_block_transfer` | Pre-allocated GPU blocks | Most production workloads |
| `raw_block_transfer` | Direct memory management | Maximum efficiency |

The default `allocation_block_transfer` provides a good balance of performance and memory efficiency by pre-allocating GPU memory blocks and reusing them across layers.

#### Minimum Blocks

`min_blocks` sets the lower bound of the block-count search range used by the assignment optimizer. Lower values give the optimizer more freedom and reduce GPU memory consumed by FlexTensor's memory blocks, but may hurt pipelining throughput. The value must be at least 2 (pipelined execution requires two blocks).

```python
config = OffloadConfig(
    num_blocks=4,
    min_blocks=2,  # Let optimizer try fewer blocks to save GPU memory
)
```

This option can also be set via the `FT_MIN_BLOCKS` environment variable.

#### Gap Layers and Transfer Windows

Gap layers are layers that contain no offloadable tensors. When a model has sequential gap layers between offloadable layers, the effective transfer window — the time available to pre-fetch the next layer's tensors — is larger than a single layer's execution time.

`GapAwareWindow` exploits this by computing transfer windows that span gap layers, allowing FlexTensor to start pre-fetching earlier. `strategy_has_transfer_gaps()` detects whether a computed strategy contains such gaps. When gap layers are detected during inference setup, `rearrange_transfers` is automatically enabled to take advantage of the extended windows.

### Strategy Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `knapsack_scale` | float | `1.0` | Scale factor for knapsack algorithm |
| `load_strategy` | Strategy | `None` | Override automatic strategy selection |

#### Understanding Strategies

FlexTensor includes several offloading strategies. The table below is a conceptual comparison of the most commonly used strategies; it is not a complete list. See the [Strategies reference page](../api/strategies.md) for the full API.

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `KnapsackStrategy` | Dynamic programming optimization | General use |
| `GreedyStrategy` | Largest tensors first | Quick approximation |
| `NthLayerStrategy` | Offload every Nth layer | Predictable patterns |
| `AdaptiveKnapsackStrategy` | Knapsack with runtime adaptation | Variable workloads |
| `AdaptiveStrategy` | Evaluates multiple candidates, selects best | Default automatic selection |
| `GlobalOffloadStrategy` | Global optimizer across all layers | Maximizing overall memory reduction |
| `GlobalTensorSelectionStrategy` | Metaheuristic search (`"DE"` or `"SA"`) | Highest-quality solution when runtime allows |

When `load_strategy=None` (the default), FlexTensor runs `AdaptiveStrategy` internally. `AdaptiveStrategy` evaluates multiple candidate strategies and selects the one with the lowest estimated overhead that satisfies the memory constraint. To include slower but potentially higher-quality `GlobalTensorSelectionStrategy` candidates in that evaluation, create an `AdaptiveStrategy` with `extra_optimization=True` and assign it to `load_strategy`:

```python
from flextensor import OffloadConfig, AdaptiveStrategy

config = OffloadConfig(
    load_strategy=AdaptiveStrategy(extra_optimization=True)
)
```

Override `load_strategy` only when you have specific requirements:

```python
from flextensor import OffloadConfig, GreedyStrategy

# Force greedy strategy for faster startup (less optimal offloading)
config = OffloadConfig(load_strategy=GreedyStrategy())
```

### Advanced Transfer Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rearrange_transfers` | bool | `False` | Optimize transfer scheduling |
| `compute_transfer_gap` | int | `1` | Minimum gap between compute and transfer |

These options control transfer scheduling optimizations:

- **`rearrange_transfers`**: When enabled, FlexTensor may reorder tensor transfers to better overlap with computation. FlexTensor also auto-enables `rearrange_transfers` when it detects permanent gap layers (layers with no offloadable tensors) during inference setup. You may therefore observe this behavior even when you have not set the option explicitly.
- **`compute_transfer_gap`**: Ensures transfers are initiated N layers before tensors are needed, hiding transfer latency

### Profile Storage

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `profile_storage_dir` | str | `None` | Directory for profile persistence |
| `profile_read_only` | bool | `False` | Only load profiles, don't save |

Profile storage enables skipping the warmup/profile phases on subsequent runs by saving and loading offload profiles. Profiles are stored as JSON files in the specified directory.

```python
config = OffloadConfig(
    profile_storage_dir="/tmp/flextensor_profiles",
    profile_read_only=False,  # Allow saving profiles (default)
)

# First run: warmup → profile → save → inference
om = flextensor.get_offload_manager()
model = om.offload(model, config=config)
for _ in range(config.warmup_iters + config.profile_iters):
    model(sample_input)
om.save_profile()  # Saves to profile_storage_dir

# Subsequent runs: load profile → inference (faster startup)
om = flextensor.get_offload_manager()
om.set_config(config)
om.load_profile(model=model)  # Loads from profile_storage_dir
```

Set `profile_read_only=True` to prevent accidental profile overwrites in production deployments.

### Debugging Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_tracing` | bool | `False` | Enable tensor access tracing |
| `enable_instrumentation` | bool | `False` | Capture component initialization args |
| `instrumentation_output_dir` | str | `".flextensor/instrumentation"` | Instrumentation output directory |
| `enable_diagnostics` | bool | `False` | Log memory transfer statistics, layer duration statistics, and block assignment table after strategy computation |

These options help diagnose offloading behavior:

- **`enable_tracing`**: Records tensor access patterns during profiling.
- **`enable_instrumentation`**: Captures the arguments passed to FlexTensor components at initialization time and writes them to `instrumentation_output_dir`. Useful for reproducing configuration state.
- **`enable_diagnostics`**: Logs three tables after strategy computation: a Memory Transfer Statistics table (tensor size → transfer time and bandwidth), a Layer Duration Statistics table (per-trap timing: min, max, median, avg, std, coefficient of variation), and the block assignment table (at NOTICE level 25). The Layer Duration Statistics table lists every offload trap created during profiling — it is the authoritative way to confirm which module patterns were applied as traps. Useful for diagnosing per-layer pipelining setup and inspecting why specific tensors were assigned to specific pipeline blocks.

The distinction between `enable_diagnostics` and `enable_instrumentation` is scope: `enable_diagnostics` reports strategy decisions (which tensors landed in which block and why); `enable_instrumentation` captures component initialization arguments (how each component was configured).

```python
# Enable detailed tracing for debugging
config = OffloadConfig(
    enable_tracing=True,
    enable_instrumentation=True,
)

# Inspect strategy decisions
config = OffloadConfig(enable_diagnostics=True)
```

!!! warning "Performance Impact"
    Tracing and instrumentation add significant overhead. Use only for debugging, not production.

### Tensor Discovery

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_untraced_tensor_discovery` | bool | `True` | Discover untraced tensors (e.g., FP8 weights passed to Triton kernels) |
| `enable_module_tracker` | bool | `True` | Enable ModuleTracker for manual traps when forward patching is not used |

Tensor discovery ensures that all tensors are properly tracked, even those accessed by custom kernels that bypass PyTorch's dispatch system (such as Triton kernels used in FP8 inference). In most cases, leave both options enabled.

You might disable `enable_untraced_tensor_discovery` if:

- Your model does not use FP8 or custom kernels
- You are debugging and want to isolate issues
- You have verified all tensors are traced normally

See [Tensor Discovery](tensor-discovery.md) for a detailed explanation of how the discovery mechanism works.

## Configuration Recipes

For ready-to-use starting configurations covering memory-constrained systems, performance-focused deployments, quick experimentation, and production, see [How to Configure FlexTensor for Common Scenarios](../how-to/configure-for-common-scenarios.md).

## Summary

| Category | Key Options | Typical Tuning |
|----------|-------------|----------------|
| **Core** | `enabled`, `gpu_device` | Set based on deployment |
| **Memory** | `pinned_memory`, `release_tensors`, `max_gpu_mem_fraction`, `shm_enabled` | Balance memory vs. performance |
| **Profiling** | `warmup_iters`, `profile_iters` | More iters = more accurate |
| **Transfer** | `transfer_mode`, `num_blocks`, `min_blocks` | Default works for most cases |
| **Debug** | `enable_tracing` | Only when troubleshooting |

Start with defaults, measure performance, and tune based on your specific workload characteristics. The profiling system will adapt to your model's actual behavior, so explicit strategy configuration is rarely needed.
