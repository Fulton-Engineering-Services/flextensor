<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `dump_to_file()`, `dump_to_directory()`, and `dump_instrumentation()` accept a keyword-only
  `extra: dict[str, Any] | None` parameter. The dict is merged flat into the top-level JSON
  output; a `ValueError` is raised if any key collides with built-in keys.

- `TensorManager.get_memory_transfer_stats()` — public accessor returning the memory transfer
  statistics (`dict[int, float]`: tensor size in bytes → transfer time in ms) computed during
  profiling, or `None` before profiling completes.

- Instrumentation dump at inference transition now includes a `memory_transfer_stats` top-level
  key (via the new `extra` parameter), recording the GPU↔CPU transfer bandwidth data used by the
  offloading strategy.

- `enable_diagnostics=True` now also logs a Memory Transfer Statistics table (tensor size,
  human-readable size, transfer time, bandwidth in GB/s) after profiling completes.

## [0.3.6] — 2026-03-05

### Added

- `OffloadConfig.max_gpu_mem_fraction` (`float | None`, default `0.9`) — GPU memory budget as a
  fraction of total device memory (e.g. `0.9` = 90%). Replaces `max_gpu_mem_bytes`. Set to `None`
  for latency mode (no memory constraint). Env var: `FT_MAX_GPU_MEM_FRACTION`.

  **Behavior change**: the default switches from `None` (latency mode) to `0.9` (memory mode).
  Users relying on latency mode by default must now explicitly set `max_gpu_mem_fraction=None`.

- `ShmCoordinator` (`flex_tensor.shm`) — creator/follower orchestration for cross-process profile
  sharing. The first process (creator) writes its profile to a shared memory coordination block and
  signals readiness; follower processes validate version compatibility and read the shared profile,
  skipping warmup and profiling entirely.

- `OffloadManager._initialize_from_shm()` — follower processes jump directly from
  `NOT_INITIALIZED` to `INFERENCE` via shared memory, avoiding redundant profiling when multiple
  processes serve the same model.

- `TensorManagerStateHandler.save_state_to_bytes()` / `load_state_from_bytes()` — serialize and
  deserialize manager state as length-prefixed bytes (4-byte big-endian uint32 + compact JSON)
  for direct use with shared memory buffers. Accepts `bytes`, `bytearray`, and `memoryview`.

### Deprecated

- `OffloadConfig.max_gpu_mem_bytes` — use `max_gpu_mem_fraction` instead. Will be removed in v0.5.
  Env var `FT_MAX_GPU_MEM_BYTES` is similarly deprecated; use `FT_MAX_GPU_MEM_FRACTION`.

## [0.3.5] — 2026-03-03

### Added

- `capture_host_resources()` (`flex_tensor.instrumentation`) — point-in-time snapshot of host
  physical memory (`host_memory_total`, `host_memory_used`, `host_memory_available`) and swap
  space (`swap_total`, `swap_used`, `swap_free`); all values in bytes via `psutil`.

- `MemorySnapshotMixin` with `_take_snapshot(label)` and `_dump_snapshots()` — opt-in GPU + host
  memory snapshot collection during vLLM worker lifecycle stages.

- `SnapshotWorker` and `FlexTensorSnapshotWorker` (`flex_tensor.contrib.vllm`) — apply
  `MemorySnapshotMixin` to the standard vLLM `Worker` and `FlexTensorOffloadWorker` respectively.
  Set `FT_VLLM_SNAPSHOT_OUTPUT_DIR` to write JSON snapshots after the final warmup step.

- `dump_to_file()` now includes a `host_memory` key alongside `components` in instrumentation
  JSON output.

- `OffloadConfig.shm_enabled`, `shm_namespace`, `shm_wait_timeout` — configuration fields for
  cross-process shared memory coordination. `shm_enabled` replaces the deprecated
  `use_shared_memory`.

- `CoordBlockHeader` — SHM version gate; prevents mismatched FlexTensor versions from sharing
  memory across processes.

- `DEFAULT_MANAGER_NAME` constant (`"default"`) exported from `flex_tensor` — replaces the
  previous PID-based default key so named managers are stable across processes.

- Thread-ownership guard on named `OffloadManager` instances — each name is bound to the thread
  that created it; cross-thread access raises `RuntimeError` with both thread IDs. Registry
  access is protected by `_MANAGER_MAP_LOCK` to prevent TOCTOU races.

### Deprecated

- `OffloadConfig.use_shared_memory` — use `shm_enabled` instead. Will be removed in v0.5.

## [0.3.4] — 2026-03-02

### Fixed

- `AllocationBlock` now preserves the original tensor stride pattern when packing tensors into a
  block. Previously, views were created with C-contiguous strides, causing `copy_()` to
  rearrange Fortran-contiguous (column-major) data. This broke vLLM modelopt FP8 quantized
  weights on SM100 (flashinfer CUTLASS FP8 GEMM expects column-major layout).

## [0.3.3] — 2026-02-27

### Changed

- `FlexTensorOffloadWorker` now defaults to LLaMA-style per-layer module patterns
  (`model.embed_tokens`, `model.layers.*`, `model.norm`, `lm_head`, `logits_processor`) instead
  of the generic `["*"]` wildcard. The wildcard created a single coarse trap around every
  top-level child, preventing per-layer pipelining. Override via `FT_MODULE_PATTERNS` for
  non-standard model layouts.

## [0.3.2] — 2026-02-27

### Added

- `OffloadConfig.max_gpu_mem_bytes` (`int | None`, default `None`) — hard GPU memory limit in
  bytes. When set, the strategy switches from latency mode to memory mode and keeps peak GPU usage
  within the specified budget. Env var: `FT_MAX_GPU_MEM_BYTES`.

- `OffloadConfig.min_block_count` — minimum number of allocation blocks the strategy must produce.

## [0.3.1] — 2026-02-26

### Fixed

- `GlobalTensorSelectionStrategy` now reuses the block assignment computed by the optimizer
  instead of re-deriving it from the transfer pipeline. This ensures peak memory reported after
  optimization is consistent with the constraint the optimizer was solving against.

### Removed

- Three unused `OffloadConfig` attributes removed to reduce API surface.
