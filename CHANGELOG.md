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

- `offload_from_profile()` convenience API that combines `init`, `load_profile`, and `offload`
  into a single call for loading a saved profile and going straight to inference.
- Diffusers examples (`examples/diffusers/quickstart/` and `examples/diffusers/profile-reuse/`)
  demonstrating FlexTensor offloading with the Wan2.2 text-to-video model, including profile
  save/load workflow.

### Removed

- **BREAKING**: Removed internal-only `OffloadConfig` fields that should not have been
  user-facing: `release_tensors`, `enable_direct_mode`, `enable_tracing`,
  `rearrange_transfers`, `compute_transfer_gap`, `enable_untraced_tensor_discovery`,
  and `enable_module_tracker`.
  `release_tensors` is now hardcoded to `True`. `rearrange_transfers` is hardcoded to
  `False` (auto-enabled when gap layers are detected). `enable_module_tracker` is
  hardcoded to `True`. The corresponding environment variables are no longer recognised.
  `compute_transfer_gap` and `enable_untraced_tensor_discovery` remain available as
  private debug parameters on `TensorManager` (`_compute_transfer_gap`,
  `_enable_untraced_tensor_discovery`).

### Fixed

- Profiling traps now use CUDA events instead of `time.time_ns()` for timing. Host-side
  wall-clock time included kernel-launch overhead and scheduler jitter; CUDA events record
  timestamps on the GPU timeline, giving accurate device-side compute duration for strategy
  decisions.
- GPU memory budget (`max_gpu_mem_fraction`) is now capped by actual available GPU memory
  at strategy compute time. Previously, `total * fraction` could exceed available memory
  when CUDA context, KV cache, or framework buffers had already consumed GPU memory,
  causing OOM. (#104)

## [0.1.0] — 2026-03-16

<!-- Initial public release. This entry is a release summary (declaration of state), not an
     incremental changelog. Keep a Changelog format applies from the next version onward. -->

FlexTensor 0.1.0 is the initial public release of a PyTorch tensor-offloading library for
running large models on limited GPU memory by dynamically moving tensors between GPU and CPU.

**Core offloading** — `OffloadManager` / `offload()` high-level API with five built-in
strategies (Knapsack, Greedy, NthLayer, Adaptive, Global). GPU memory budget controlled via
`max_gpu_mem_fraction` (default `0.9`); module-level granularity via wildcard
`module_patterns`. Forward methods are patched directly — no proxy wrappers — preserving model
hierarchy, `isinstance` checks, and serialization.

**Profile persistence** — `save_profile()` / `load_profile()` to skip warmup and profiling on
subsequent runs. `load_model_from_profile()` for memory-efficient model loading from saved
profiles (meta-device parameters, selective safetensors weight loading).

**Cross-process shared memory** — `ShmCoordinator` for creator/follower orchestration: the
first process profiles and writes results to shared memory; followers read the shared profile
and jump straight to inference. Opt-in via `shm_enabled=True`. Version-gated to prevent
mismatched FlexTensor versions from sharing state.

**vLLM integration** — `FlexTensorOffloadWorker` with per-layer module patterns for
transformer models. `SnapshotWorker` / `FlexTensorSnapshotWorker` for opt-in GPU and host
memory snapshot collection during worker lifecycle.

**Known limitations** — inference only (no training or backward pass); no data parallelism;
no MoE support; not thread-safe (one thread per manager instance).

[Unreleased]: https://github.com/ai-dynamo/flextensor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ai-dynamo/flextensor/releases/tag/v0.1.0
