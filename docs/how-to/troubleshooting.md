<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# How to Troubleshoot FlexTensor

This guide provides practical solutions to common problems you may encounter when using FlexTensor. Each section focuses on a specific troubleshooting scenario with step-by-step instructions to diagnose and resolve issues.

**When to use this guide**: You're experiencing unexpected behavior, errors, or need to inspect FlexTensor's internal operations for debugging.

## Prerequisites

- FlexTensor installed
- A model configured for offloading
- Basic familiarity with FlexTensor's configuration (see [Quick Start](../quick-start.md))

---

## Debug Component Initialization

**Problem**: You need to understand what configuration values FlexTensor is using internally, verify that components are initialized correctly, or debug unexpected behavior during offloading setup.

**Solution**: Use FlexTensor's instrumentation system to capture all initialization arguments and configuration values used by internal components.

### When to Use This Approach

Use instrumentation debugging when:
- Component initialization fails with unclear error messages
- You suspect configuration values aren't being applied correctly
- You need to verify which strategy or loader is being used
- You're reporting a bug and need detailed diagnostic information

### Steps

#### 1. Enable Instrumentation in Your Config

Set `enable_instrumentation=True` in your `OffloadConfig`:

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    enable_instrumentation=True,
    include_patterns=["layers.*"],
    # ... your other settings
)

model = offload(model, config=config)
```

**Alternative: Use environment variables**

You can also enable instrumentation via environment variables without modifying code:

```bash
export FT_ENABLE_INSTRUMENTATION=1
export FT_INSTRUMENTATION_OUTPUT_DIR=/tmp/my_debug_output  # optional
```

Then load the config from environment:

```python
from flextensor import load_config, offload

config = load_config()  # reads FT_* environment variables
model = offload(model, config=config)
```

#### 2. Run Your Model Through Discovery and Profiling

Execute your model until it transitions to inference. Use the path-aware
`iters_before_inference` count — it accounts for `skip_discovery` (default
`False`), compiled offload, and replan paths:

```python
om = flextensor.get_offload_manager()
for _ in range(om.iters_before_inference):
    output = model(input_data)
```

When FlexTensor transitions to inference phase, it automatically dumps the instrumentation data.

#### 3. Locate the Output File

The instrumentation data is written to the `instrumentation_output_dir` directory (default: `.flextensor/instrumentation`). The file structure is:

```
.flextensor/
└── instrumentation/
    └── 20251209_143052/
        └──── components.20251209_143052_pid12345.json
```

Each run creates a timestamped subdirectory with the JSON file inside.

#### 4. Interpret the Captured Data

The JSON file contains:

```json
{
  "timestamp": "2025-12-09T14:30:52.123456",
  "flextensor_version": "0.1.0",
  "host_memory": {
    "host_memory_total": 270122237952,
    "host_memory_used": 13204619264,
    "host_memory_available": 256917618688,
    "swap_total": 8589934592,
    "swap_used": 107374182,
    "swap_free": 8482560410
  },
  "components": [
    {
      "class_name": "KnapsackStrategy",
      "module_path": "flextensor.strategy.KnapsackStrategy",
      "init_timestamp": "2025-12-09T14:30:50.100000",
      "args": {
        "scale": 1.0,
        "cyclic": false,
        "group_size": 1,
        "threshold_mb": 0.1
      }
    },
    {
      "class_name": "TensorManager",
      "module_path": "flextensor.tensor_manager.TensorManager",
      "init_timestamp": "2025-12-09T14:30:50.200000",
      "args": {
        "device_gpu": "cuda:0",
        "tensor_manager_load_strategy": {
          "_type": "AdaptiveStrategy",
          "_module": "flextensor.strategy"
        },
        "pinned_memory": true,
        "pinned_memory_mode": "torch",
        "loader_type": "allocation_block_transfer",
        "remove_layers_operations": [],
        "blocks": 4,
        "move_top_level_buffers_to_gpu": true,
        "use_shm": false,
        "enable_diagnostics": false,
        "max_gpu_mem_fraction": null,
        "profile_mode": "view",
        "_use_trace_tensor": false,
        "_rearrange_transfers": false,
        "_compute_transfer_gap": 1,
        "_enable_untraced_tensor_discovery": true
      }
    }
  ],
  "memory_transfer_stats": {
    "1024": 0.015,
    "4096": 0.023,
    "1048576": 0.187
  }
}
```

Each component record includes:
- `class_name`: The component class
- `module_path`: Full module path
- `init_timestamp`: When the component was initialized
- `args`: All initialization arguments **and their default values**. The decorator captures every parameter, including those not explicitly passed by the caller.

> **Note:** Fields prefixed with `_` (e.g. `_use_trace_tensor`, `_rearrange_transfers`) are internal debug parameters. Their presence in instrumentation output is expected. In normal operation they remain at their defaults. The exact set of fields may vary across components and versions.

The `host_memory` object captures a point-in-time snapshot of host physical memory (`host_memory_*`) and swap space (`swap_*`) at the time the dump is written. All values are in bytes. Divide by `1024 ** 3` to convert to GiB.

The `memory_transfer_stats` object maps tensor sizes (bytes) to GPU↔CPU transfer times (ms), as measured during profiling. This key is present when the dump is written at inference transition (the automatic path); it may be absent in manual dumps.

### Advanced Options

#### Manual Dump

To dump instrumentation data at any point (not just at inference transition):

```python
from flextensor.instrumentation import dump_instrumentation

dump_instrumentation("/path/to/dir")
```

#### Custom Output Directory

Change the output location via config:

```python
config = OffloadConfig(
    enable_instrumentation=True,
    instrumentation_output_dir="/tmp/my_debug_output",
)
```

#### Disable Instrumentation

For production, ensure instrumentation is disabled (the default):

```python
config = OffloadConfig(
    enable_instrumentation=False,  # default
)
```

Or simply omit the `enable_instrumentation` parameter.

---

## Resolve GPU Out-of-Memory Errors

**Problem**: `torch.cuda.OutOfMemoryError` or `CUDA out of memory` during discovery, profiling, or inference.

### Why this happens

FlexTensor pre-allocates GPU memory blocks sized to the largest trapped module's weights. With `include_patterns=["*"]`, container modules (e.g., the top-level `model` or a `Sequential`) are also trapped — their traps see all child weights, inflating block sizes.

### Step 1: Check for competing GPU processes

Verify that no other processes are holding GPU memory:

```bash
nvidia-smi
```

If another process is consuming significant memory, stop it or move your workload to a free device.

!!! note
    Setting `max_gpu_mem_fraction` does not fully eliminate this risk. The budget is capped to available memory at the time the strategy is computed, but a competing process can allocate GPU memory between that query and the actual block allocations, causing an OOM.

### Step 2: Narrow the include patterns

If `include_patterns=["*"]` is offloading embedding or output layers that need to
stay on GPU, or container modules are inflating block sizes, narrow the scope.
For vLLM, prefer leaving `FT_INCLUDE_PATTERNS` / `FT_EXCLUDE_PATTERNS` unset
unless you need a model-specific override; the worker defaults use
decoder-layer class includes and MoE sidecar excludes. The exact defaults live
in `src/flextensor/contrib/vllm/worker.py`.

**Why this helps:** Offloaded layers are distributed across `num_blocks` GPU memory blocks, each sized to the largest layer assigned to it. With `["*"]`, container modules become the "largest layer" and inflate every memory block to the full model's size. Per-layer patterns remove the containers, so memory blocks are sized to individual layers.

For other architectures, use `model.named_modules()` to find the right pattern names.

### Step 3: Reduce the number of memory blocks

Step 2 reduces each memory block's *size*; this step reduces the *count*. Fewer memory blocks means less total GPU memory reserved:

```python
config = OffloadConfig(
    transfer_mode="allocation_block_transfer",
    num_blocks=2,   # Reduce from the default of 4
)
```

### Step 4: Check for "Insufficient free GPU memory" error

If you see:

```
RuntimeError: Insufficient free GPU memory: X.XX GiB available
(free=..., reserved=..., allocated=...), minimum required: 0.25 GiB
```

The GPU has less than 256 MiB free when the strategy runs — typically consumed by other processes, the CUDA context, or a large KV cache.

**Resolve:**

- Free other GPU processes or reduce KV cache / batch size.
- In vLLM or TRT-LLM, lower GPU memory reservation so FlexTensor has room.
- Set `max_gpu_mem_fraction=None` to switch to latency mode — this bypasses the memory budget check, but if the GPU genuinely has no free memory, block allocations will still fail with a CUDA OOM.

If you see:

```
RuntimeError: Insufficient strategy GPU budget after reserving
strategy-invisible permanent GPU tensors: original_budget=...,
reserved_strategy_invisible_permanent_gpu=...,
effective_strategy_budget=..., minimum_required=...
```

Some tensors that FlexTensor will keep on GPU are absent from the profiled layer
statistics, so FlexTensor must reserve room for them before choosing a transfer
strategy. This can happen during block-loader finalization or strategy-loader
untimed-tensor rescue. The error means the remaining strategy budget is too
small after that reservation. Reduce other GPU consumers (for example KV cache /
batch size or competing processes), increase `max_gpu_mem_fraction` when it is
set too low for the model's required permanent GPU tensors, or adjust
`include_patterns` so tensors intended for offloading are actually profiled
instead of left as permanent GPU tensors.

---

## Resolve Host Out-of-Memory Errors

**Problem**: Python `MemoryError`, the Linux OOM killer terminating the process (`Killed`), or severe swap thrashing during discovery or profiling.

### Why this happens

FlexTensor copies offloaded weights into pinned (page-locked) CPU memory, which cannot be swapped — it consumes physical RAM exclusively. With `include_patterns=["*"]`, every discovered weight is pinned, which can exhaust host RAM on large models.

Since `pin_memory()` copies data, the original and pinned weights coexist briefly — peak host memory is roughly model size plus one layer's worth of weights. PyTorch may also round pinned allocations to the next power of two ([pytorch#150517](https://github.com/pytorch/pytorch/issues/150517)).

### Step 1: Check host memory pressure

```bash
free -h              # available memory and swap
dmesg | grep -i oom  # check for OOM killer
```

You can also inspect the `host_memory` object in FlexTensor's [instrumentation output](#debug-component-initialization) for a memory snapshot at inference transition.

### Step 2: Narrow the include patterns

Narrow `include_patterns` from `["*"]` to target specific layers. See [Step 2: Narrow the include patterns](#step-2-narrow-the-include-patterns) under [Resolve GPU Out-of-Memory Errors](#resolve-gpu-out-of-memory-errors) for a vLLM worker-aligned example.

**Why this helps:** Fewer trapped modules means fewer weights pinned in host RAM.

### Step 3: Switch to in-place pinning

The default `pin_memory()` path copies each weight into a fresh allocation from PyTorch's caching pinned allocator, which rounds up to the next power of two ([pytorch#150517](https://github.com/pytorch/pytorch/issues/150517)) and briefly holds both the source and pinned copy in RAM. Switching to `host_register` mode pins the *existing* allocation in place via `cudaHostRegister` — no copy, no rounding doubling:

```python
config = OffloadConfig(
    pinned_memory=True,
    pinned_memory_mode="host_register",
)
```

This keeps the performance benefit of pinned memory (non-blocking DMA) while removing the peak-RAM spike. On a CUDA host with a broken `cudart` binding, falls back to `"torch"` with a warning. On a CPU-only host, `pinned_memory=True` raises at `TensorManager` construction time — offloading without a GPU has no purpose, and silently degrading would mask the misconfiguration as a perf regression. Set `pinned_memory=False` on CPU-only hosts.

!!! note "SHM segments are already pinned in place"
    `pinned_memory_mode` only affects the non-SHM allocator path. When `shm_enabled=True`, the SHM segment itself is always registered in place via `cudaHostRegister` (regardless of mode), because POSIX shared-memory buffers can't be re-allocated through PyTorch's pinned allocator. Switching to `"host_register"` mode is what lets the *non-SHM* per-layer blocks avoid the doubling described above.

!!! note "Pinning `RuntimeError` from `cudaHostRegister` or `tensor.pin_memory()`"
    Pinning is strict at every site that pins host memory — model preparation (`preprocess_model`), warmup, profiling, and benchmark passes. The first failure aborts the offending phase with a `RuntimeError` that names the operation, pointer, size, and cudart rc. Common causes:

    - `RLIMIT_MEMLOCK` exhaustion (most frequent in containers / unprivileged processes).
    - Pinned-pool pressure (PyTorch's caching pinned allocator out of headroom).
    - A non-zero `cudaHostRegister` rc (rare; usually surfaces as `cudaError… 712 / 717`).

    Mitigations, in rough order of preference:

    - Raise the lock budget — `ulimit -l unlimited` (or `LimitMEMLOCK=infinity` for systemd units, `--ulimit memlock=-1` for Docker).
    - Reduce pinned footprint — narrow `include_patterns` (or add `exclude_patterns`) to offload fewer modules.
    - Switch modes — `pinned_memory_mode="torch"` uses PyTorch's allocator instead of `cudaHostRegister`, but it still locks pages so it doesn't relieve `RLIMIT_MEMLOCK`; only worth trying when the failure is specific to in-place registration.
    - Disable pinning entirely — `pinned_memory=False`. Loses non-blocking transfer overlap but lets the pipeline run.

### Step 4: Disable pinned memory as a last resort

If host memory is still exhausted, disable pinned memory entirely:

```python
config = OffloadConfig(
    pinned_memory=False,
)
```

!!! warning "Performance trade-off"
    Unpinned memory can be swapped (preventing OOM) but disables asynchronous DMA, making CPU↔GPU transfers slower.
---

## Fix CUDA Device Mismatch Errors

**Problem**: A runtime error such as `RuntimeError: Expected all tensors to be on the same device` during inference.

This typically occurs when FlexTensor offloads a parameter but a custom kernel (for example, a Triton or CUDA kernel) accesses a related metadata attribute (such as a scale tensor) that was not mapped during the discovery phase.

### Step 1: Verify tensor discovery is enabled

Both untraced tensor discovery and ModuleTracker are always enabled (hardcoded). No configuration is required.

### Step 2: Use the `offload()` API with forward patching

Auto trap discovery (the most reliable strategy) only activates when you use `flextensor.offload()` with explicit `include_patterns`:

```python
config = OffloadConfig(include_patterns=["layers.*"])
model = flextensor.offload(model, config=config)
```

Using manual `offload_block` context managers without forward patching relies on Module Tracker or Prefix Matching, which may miss tensors with unconventional naming.

For custom kernels or fused backends such as vLLM MoE, include the module that owns the kernel weights and launch the backend while that module's trap is active. Direct warmup/profile temporarily materialize raw parameter storage, so calls that read `self.weight`-style attributes inside the patched `forward` see active GPU storage. Avoid using tensor references cached before `offload()`: FlexTensor cannot update opaque pointers that bypass module ownership and discovery.

### Step 3: Debug with instrumentation

Enable debug instrumentation to capture component initialization details and verify the parameter-to-trap mapping:

```python
config = OffloadConfig(
    enable_instrumentation=True,
    include_patterns=["layers.*"],
)
model = offload(model, config=config)
```

See [Debug Component Initialization](#debug-component-initialization) for how to interpret the output.

For a detailed explanation of how tensor discovery works, see [Tensor Discovery](../explanation/tensor-discovery.md).

---

## Diagnose High Performance Overhead

**Problem**: Model inference is slower than expected. FlexTensor achieves low latency overhead by pipelining weight transfers to overlap with GPU computation, under the assumptions described [below](#when-to-expect-higher-overhead).

### Step 1: Check that inference phase has been reached

Discovery and profiling iterations are intentionally slower—overhead measurements are only meaningful after all discovery and profiling iterations complete:

```python
config = OffloadConfig(profiling_iters=10)
model = flextensor.offload(model, config=config)
om = flextensor.get_offload_manager()

# First `om.iters_before_inference` iterations are measurement phases.
# Under the default `skip_discovery=False` that is
# `discovery_iters + profiling_iters`; with `skip_discovery=True` it is
# just `profiling_iters`.
warmup = om.iters_before_inference
for i, batch in enumerate(dataloader):
    output = model(batch)
    if i == warmup:
        print("Inference phase reached — timing is now production overhead")
```

### Step 2: Increase profile iterations for noisy environments

On shared or cloud GPUs, thermal throttling and multi-tenancy can cause noisy timing data. Increase `profiling_iters` for a more accurate strategy:

```python
config = OffloadConfig(profiling_iters=20)  # default: 10
```

### Step 3: Tune transfer mode and block count

The `allocation_block_transfer` mode with more blocks can reduce stalls by overlapping transfers with computation:

```python
config = OffloadConfig(
    transfer_mode="allocation_block_transfer",
    num_blocks=8,                  # More blocks = more parallelism
)
```

Transfer rearrangement is auto-enabled when gap layers are detected.

### When to expect higher overhead

Low overhead depends on the CPU-to-GPU interconnect bandwidth being sufficient to transfer offloaded weights within the available compute time. This condition may not hold when:

- **Low concurrency / small batch decode**: At batch=1, per-layer compute can be very short (1-3 ms for typical LLMs), while transferring hundreds of MB of weights may take 15-25 ms depending on interconnect bandwidth. The transfer cannot be overlapped with such short compute.
- **High offload ratio**: When a large fraction of model weights are offloaded (e.g., >40%), most layers require significant weight transfers. Even modest stalls per layer accumulate across dozens of layers.
- **Profiling/production mismatch**: FlexTensor profiles at a specific batch size to measure per-layer compute time and plan transfers accordingly. If the production workload has a substantially different batch size (e.g., profiling at large prefill size but serving single-token decode at low concurrency), the transfer schedule may be planned for a compute window that does not exist at serving time.

**How to check**: Enable `FT_ENABLE_DIAGNOSTICS=1` and examine the block assignment table in the log output. Compare the "Compute" column (per-layer compute time in ms) with the transfer sizes. If the time to transfer a layer's offloaded weights exceeds the preceding layer's compute time, significant overhead is expected.

**Mitigation**:

- Increase request concurrency — larger decode batches increase per-layer compute, improving transfer overlap.
- Reduce the offload ratio by increasing `max_gpu_mem_fraction` (if KV-cache budget allows).
- Accept the latency tradeoff when serving a model that otherwise would not fit in GPU memory — FlexTensor enables serving, but the interconnect bandwidth constrains throughput.

---

## Fix Unexpected Behavior After Modifying a Patched Model

**Problem**: You added, removed, or replaced a module in your model after calling `offload()`, and the new module is not being offloaded, or you observe incorrect behavior such as device mismatch errors or missing tensor coverage.

### Why This Happens

When you call `flextensor.offload(model, config=config)`, FlexTensor patches the `forward` methods of matched modules and builds a parameter-to-trap map during discovery. This map is fixed at the end of the discovery phase. Any module added to the model after `offload()` is called is not part of that map and will not be offloaded.

```python
# BAD: new layer is not covered by offloading
config = flextensor.OffloadConfig(include_patterns=["layers.*"])
model = flextensor.offload(model, config=config)
model.layers.append(NewLayer())  # NewLayer will not be offloaded
```

### Solution: Release and Re-apply Offloading

Call `release()` on the offload manager to remove all patches and clear the parameter map, modify the model, then call `offload()` again:

```python
import flextensor

config = flextensor.OffloadConfig(include_patterns=["layers.*"])

# GOOD: patch after all modifications are done
model.layers.append(NewLayer())
model = flextensor.offload(model, config=config)
```

If you have already called `offload()` and need to modify the model afterward:

```python
import flextensor

config = flextensor.OffloadConfig(include_patterns=["layers.*"])
model = flextensor.offload(model, config=config)

# Later: modify the model
om = flextensor.get_offload_manager()
om.release()                          # Remove all patches and tensor map
model.layers.append(NewLayer())
model = flextensor.offload(model, config=config)  # Re-apply
```

### Prevention

The safest approach is to finalize your model architecture before calling `offload()`. Apply all `append`, `insert`, `pop`, or attribute assignments to the model before patching it.

---

## Fix Cross-Thread Manager Access Errors

**Problem**: A `RuntimeError` with a message like `OffloadManager 'default' belongs to thread 12345, but accessed from thread 67890`, or silent data corruption when using FlexTensor from multiple threads.

!!! danger "FlexTensor is not thread-safe"
    The entire offloading lifecycle — setup, discovery, profiling, **and inference** — must run on a single thread per manager. FlexTensor's internal state (tensor maps, CUDA streams, memory blocks, profiling counters) is not protected against concurrent access. Using a patched model from multiple threads in parallel can cause silent data corruption, CUDA errors, or incorrect inference results even if no `RuntimeError` is raised.

The `get_offload_manager()` thread-ownership guard catches the most common mistake (calling the API from the wrong thread), but it does **not** protect against running forward passes on a patched model concurrently.

### Step 1: Identify the conflicting access

The error message includes both thread IDs. Check your code for places where `get_offload_manager()` (or the convenience functions `offload()`, `init()`, `set_config()`, etc.) is called from a background thread, callback, or worker that differs from the thread that originally created the manager. Also check for forward passes on the patched model from multiple threads.

### Step 2: Use a separate name, model, and thread per manager

If you need parallel offloading, each thread must have its own manager **and its own model instance**:

```python
import threading
import flextensor

def worker(thread_name: str, model):
    config = flextensor.OffloadConfig(include_patterns=["layers.*"])
    model = flextensor.offload(model, config=config, name=thread_name)
    for batch in dataloader:
        output = model(batch)  # Safe: single thread owns this manager + model

t1 = threading.Thread(target=worker, args=("worker-1", build_model()))
t2 = threading.Thread(target=worker, args=("worker-2", build_model()))
t1.start()
t2.start()
```

### Step 3: Restructure to single-thread access

If threads must share a single model, perform all FlexTensor operations (offloading, discovery, profiling, and inference) on one thread and dispatch results to other threads afterward. Do not run forward passes on a patched model from multiple threads concurrently.

---

## Collect GPU Memory Snapshots in vLLM

**Problem**: You need a record of GPU memory state at each vLLM worker lifecycle point (after device initialization, model load, KV cache allocation, and warmup) to understand memory pressure, verify headroom, or compare baseline versus offloading deployments.

**Solution**: Use one of the snapshot worker classes, which capture `MemorySnapshot` readings at each lifecycle stage and write them to a JSON file.

### Choose the Right Worker Class

| Scenario | Worker class |
|----------|--------------|
| Standard vLLM, no offloading | `SnapshotWorker` |
| vLLM with FlexTensor offloading | `FlexTensorSnapshotWorker` |

Both classes collect snapshots at the same lifecycle points: after `init_device`, `load_model`, `determine_available_memory`, `initialize_from_config` (KV cache), and `compile_or_warm_up_model`.

### Steps

#### 1. Set the output directory

Snapshots are only written to disk when `FT_VLLM_SNAPSHOT_OUTPUT_DIR` is set. If the variable is unset or empty, snapshots are collected in memory but discarded when the process exits.

```bash
export FT_VLLM_SNAPSHOT_OUTPUT_DIR=/tmp/gpu_snapshots
```

#### 2. Start vLLM with the snapshot worker

For a standard vLLM deployment (no FlexTensor offloading):

```bash
vllm serve <model> \
    --worker-cls flextensor.contrib.vllm.snapshot.SnapshotWorker
```

For a FlexTensor offloading deployment:

```bash
FT_ENABLED=1 vllm serve <model> \
    --worker-cls flextensor.contrib.vllm.snapshot.FlexTensorSnapshotWorker
```

#### 3. Inspect the output

After the final warmup step, each worker rank writes a JSON file to `FT_VLLM_SNAPSHOT_OUTPUT_DIR`:

```text
/tmp/gpu_snapshots/
└── gpu_snapshots_rank0_device0_20260303_120000.json
```

The file contains metadata about the worker and a list of snapshot entries, one per lifecycle label:

```json
{
  "worker_type": "FlexTensorSnapshotWorker",
  "model": "Qwen/Qwen2.5-7B",
  "rank": 0,
  "local_rank": 0,
  "device": "cuda:0",
  "snapshots": [
    {
      "label": "after_init_device",
      "gpu_memory": {
        "free_memory": 79285829632,
        "cuda_memory": 1073741824,
        "torch_memory": 536870912,
        "non_torch_memory": 536870912,
        "total_memory": 85899345920,
        "torch_peak": 536870912,
        "timestamp": 1741003201.123456
      },
      "host_memory": {
        "host_memory_total": 270122237952,
        "host_memory_used": 13204619264,
        "host_memory_available": 256917618688,
        "swap_total": 8589934592,
        "swap_used": 107374182,
        "swap_free": 8482560410
      }
    }
  ]
}
```

All fields in `gpu_memory` and `host_memory` are in bytes. Divide by `1024 ** 3` to convert to GiB.

!!! note "Multi-rank deployments"
    In tensor-parallel deployments, each rank writes its own file. The filename includes `rank` and `device` to distinguish them.
