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
    profiling_iters=15,           # Accurate profiling
    max_gpu_mem_fraction=0.8,   # Use at most 80% of total GPU memory
    min_blocks=2,               # Fewer blocks to save GPU memory
)

model = offload(model, config=config)
```

Key choices:

- `max_gpu_mem_fraction=0.8` switches the strategy to memory mode, targeting at most 80% of total device memory (the effective budget may be lower if other consumers have already used GPU memory). Using a fraction rather than a byte count makes the config portable across GPU SKUs.
- `min_blocks=2` lets the optimizer try fewer memory blocks, reducing GPU memory consumed by FlexTensor itself.

---

## Performance-Focused

Use this configuration when latency matters more than memory savings and you want to minimize the overhead that offloading adds to the model forward pass.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    transfer_mode="allocation_block_transfer",
    num_blocks=8,               # More blocks for parallelism
)

model = offload(model, config=config)
```

Key choices:

- `num_blocks=8` increases the number of pre-allocated GPU memory blocks, allowing more transfers to overlap with computation.
- Transfer rearrangement is auto-enabled when gap layers are detected.

---

## Quick Experimentation

Use this configuration during development when you need fast iteration and startup time, and profiling accuracy is less important.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    profiling_iters=3,            # Minimal profiling for fast startup
)

model = offload(model, config=config)
```

Key choices:

- Under the default `skip_discovery=False`, `discovery_iters + profiling_iters` drive startup cost. Set `skip_discovery=True` to bypass discovery when using forward patching.
- `profiling_iters=3` reduces the startup cost before reaching inference phase.
- This produces a less accurate offloading strategy than higher iteration counts, so it is not recommended for production.

---

## Production Deployment

Use this configuration when deploying to a stable environment where startup time is less critical and you want accurate, reproducible offloading behavior.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    profiling_iters=20,           # Accurate initial profiling
)

model = offload(model, config=config)
```

Key choices:

- `profiling_iters=20` gives the profiler more timing samples, which is especially important on shared or cloud GPUs where timing noise is higher.

!!! tip "Use profile persistence for faster startup"
    Save profiles after the initial discovery/profiling run using `save_profile()`, then load them on subsequent runs with `load_profile()` to skip the discovery and profiling phases entirely. Set `profile_storage_dir` in your config and ensure `profile_read_only=False` to enable saving.

### Refresh a vLLM worker-v2 profile from serving traffic

Worker v2 can bootstrap from the previous `profile.json`, then replace its compute timings after ten
matching serving batches. This example measures CUDA-graph replay:

```bash
export FT_ENABLED=1
export FT_VLLM_USE_V2_WORKER=1
export FT_PROFILING_ITERS=10
export FT_PROFILE_STORAGE_DIR=/persistent/flextensor/qwen3-decode

vllm serve Qwen/Qwen3-30B-A3B-FP8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 512 \
  --no-enable-prefix-caching \
  --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker
```

Worker v2 derives and overwrites `external_compile` from vLLM's compilation mode and
`offload_timing` from the CUDA-graph mode requested at model load. It warns before overriding an
explicitly configured conflicting value. vLLM may later downgrade CUDA-graph mode for attention
backend compatibility; external timing events also support eager execution, and each sampled
`execute_model()` call selects CUDA-graph or eager finalization from actual replay activity.
`FT_VLLM_TIMING_BATCH` defaults to `decode` for pure decode iterations; set it to `prefill` for pure
prefill iterations. Mixed prefill/decode iterations are ignored. Worker v2 uses the first
`max(1, FT_PROFILING_ITERS)` matching samples; the target must not exceed the 1024-entry timing-store
limit. `FT_PROFILE_STORAGE_DIR` defaults to `None`, which disables persistence; set
`FT_PROFILE_READ_ONLY=1` for load-only operation.

On restart, worker v2 adopts only compatible saved timing statistics from `profile.json` and ignores
its stale strategy. A fresh model scan, the current model/config, and current GPU budget remain
authoritative when it computes a new strategy conservatively. It warns and falls back to conservative
statistics when the saved profile is missing, invalid, or incompatible; incomplete timing collection
leaves an existing file unchanged. The active server never replans or recaptures; refreshed timings
affect strategy computation only on the next bootstrap. Delete `profile.json` or use another storage
directory when switching between decode and prefill measurements.

---

## NVMe Disk Offload

Use this configuration when both GPU memory and host RAM are constrained — for
example, running a large model on a GB10 Grace-Blackwell system where the GPU
and CPU share unified DRAM. NVMe disk offload evicts cold weights to a local
NVMe SSD, freeing host memory while keeping weights accessible for on-demand
GPU reads.

```python
from flextensor import OffloadConfig, offload

config = OffloadConfig(
    nvme_offload_enabled=True,
    nvme_offload_path="/mnt/nvme/flextensor_weights",
    nvme_transfer_mode="cufile",           # GDS direct-to-GPU; auto-falls back to POSIX
    transfer_mode="allocation_block_transfer",  # required — NVMe needs a block loader
)

model = offload(model, config=config)
```

Key choices:

- `nvme_offload_enabled=True` activates the NVMe eviction path. During block
  controller construction, all packed weight blocks are written to a single
  file (`{nvme_offload_path}/blocks.bin`) at alignment-aligned offsets, then the
  pinned CPU blocks are freed. During inference, `schedule_transfer()` reads from
  the NVMe file instead of copying from CPU memory.
- `nvme_offload_path` must point to a writable directory on a local NVMe
  filesystem (not NFS). Files are opened with `O_DIRECT` for cuFile; the POSIX
  backend also prefers `O_DIRECT` but falls back to buffered I/O when the
  filesystem does not support it.
- `nvme_transfer_mode="cufile"` selects GPU Direct Storage (`cuFileRead`) for
  direct NVMe→GPU DMA. When `nvidia-fs` is not loaded, `libcufile.so` is
  unavailable, or the GPU model does not support GDS P2P, FlexTensor logs a
  `WARNING` and falls back to `PosixBackend` automatically. Set
  `nvme_transfer_mode="posix"` to use the `pread`+`copy_` path explicitly.
- A block `transfer_mode` (`allocation_block_transfer` or `raw_block_transfer`)
  is required. NVMe backing is implemented by the block controllers, which the
  `strategy` loader does not use. Config construction raises `ValueError` if
  `nvme_offload_enabled=True` is combined with `transfer_mode="strategy"`.

!!! note "GB10 (Grace-Blackwell) unified memory"
    On GB10, cuFile/GDS is fundamentally unsupported — `cuFileHandleRegister`
    fails with `CU_FILE_IO_NOT_SUPPORTED` (rc=5008) because no PCIe P2P path
    exists from the NVMe controller to the GPU's unified memory. The auto-fallback
    to `PosixBackend` handles this transparently. On GB10, `PosixBackend` is
    the only viable backend: `pread` into pinned CPU memory, then `copy_` to the
    GPU block view. Since the pinned CPU buffer shares the same physical DRAM as
    the GPU on unified-memory systems, the `copy_` is near-zero-cost.

    Set `nvme_transfer_mode="posix"` explicitly to suppress the fallback
    warning.

!!! warning "CUDA graph replay with POSIX NVMe backend"
    CUDA graph capture succeeds with NVMe-backed offload, but **graph replay
    correctness is not guaranteed with the POSIX backend**. `os.pread` is a
    CPU-side operation not captured in the CUDA graph, so on replay the shared
    pinned staging buffer contains stale data from the last capture-time read.
    cuFile (GDS) would work because it writes directly to GPU memory (capturable),
    but cuFile is unavailable on GB10. Use eager inference (no graph capture) or
    avoid graph replay with NVMe offload on GB10. See
    [Troubleshooting](troubleshooting.md#troubleshoot-nvme-disk-offload) for
    details.

---

## Measure transfer overlap during inference

Use this when you want per-trap **transfer / compute / wait** timings while serving — without rebuilding the offload strategy.

```python
import flextensor as ft
from flextensor import OffloadConfig, format_offload_timing_table, get_offload_manager, offload

config = OffloadConfig(
    offload_timing="eager",  # or "cuda_graph" under CUDA-graph replay
    # Requires a block transfer_mode (default allocation_block_transfer).
)
model = offload(model, config=config)
om = get_offload_manager()

# Reach INFERENCE first — do not reset/collect during discovery/profiling.
for _ in range(om.iters_before_inference):
    model(x)

# Clear so the report covers only the serving window you care about:
ft.reset_offload_timing()

for _ in range(20):
    model(x)

report = ft.collect_offload_timing()
if report is not None:
    print(format_offload_timing_table(report))
```

Key choices:

- `offload_timing="eager"` (or env `FT_OFFLOAD_TIMING=eager`) arms the collector for module forwards.
- Requires a block `transfer_mode` (`allocation_block_transfer` / `raw_block_transfer`); `transfer_mode="strategy"` is rejected — that loader has no enter/exit timing hooks.
- Eager / normal module forwards publish each pass automatically; call `collect_offload_timing()` after the window ends (drains the durable store).
- The durable store is an internal ring buffer (default cap 1024 passes).
- `wait_ms ≈ 0` means H2D finished before compute needed the data (fully hidden).
- CUDA-graph replay needs `offload_timing="cuda_graph"`, then `update_offload_timing()` after each `graph.replay()`, then `collect_offload_timing()`. See [torch.compile](torch-compile.md#cuda-graphs-request_strategy_replanmanual_update_statetrue).

---

## Next Steps

- [Configuration reference](../explanation/configuration.md) — full description of every `OffloadConfig` option
- [Troubleshooting](troubleshooting.md) — diagnose performance and memory issues
- [torch.compile](torch-compile.md) — compiled offload and CUDA-graph measure / replan
