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

- Profiling data control API: `flextensor.clear_profiling_durations()`,
  `suspend_profiling()` / `resume_profiling()`, and the `pause_profiling()`
  context manager. Lets backends (e.g. vLLM) bracket mixed-batch warmup so
  paused passes contribute neither duration samples nor tensor IDs to the
  PROFILING input — important on data-dependent models (MoE experts,
  conditional branches, mixed-batch shapes) where a paused pass might
  exercise a different parameter set than the real workload. The PROFILING
  iteration counter is also frozen so suppressed passes don't consume the
  `profiling_iters` budget. Suspensions are reference-counted, so
  independent callers can nest without stepping on each other. See
  [Profiling Data Control](docs/explanation/phases.md#profiling-data-control).

### Changed

- `UntimedTrapsReport` is now emitted at `WARNING` whenever any trap has
  tensor IDs but no duration samples, regardless of `enable_diagnostics`.
  These labels are silently dropped from the strategy input; users need
  visibility in production. The verbose layer-duration table remains
  gated on `enable_diagnostics`.
- Internal: `WarmupTrap` no longer measures per-iteration CUDA-event
  durations (they were wiped before profiling began anyway), and the new
  `TensorManager.record_tensors(label, tensor_ids)` is the DISCOVERY-only
  recorder. Affects plugin authors who subclassed `WarmupTrap` or called
  the recorders directly.

### Fixed

- `FlexTensorOffloadWorker` now pushes the speculative-decoding drafter
  (e.g. MTP / Eagle) to GPU before `flextensor.offload()` runs, so drafter
  weights don't remain on CPU where FT's loader leaves them. This resolves
  the Dynamo `cpu/cuda` device mismatch in vLLM's `@torch.compile`
  layernorm helper that previously crashed warmup on every MTP + FT tier. (#140)
- Instrumentation dumps are now valid JSON when components hold frozen maps
  (e.g. `TensorManager.tensors_map`, a `MappingProxyType`); serializer failures
  and non-JSON values in `dump_instrumentation(extra=…)` degrade gracefully
  instead of aborting the dump or propagating out of decorated `__init__`. (#141)
- FlexTensor no longer crashes during model discovery on modules or tensor
  subclasses whose attribute access raises a non-`AttributeError` (e.g. vLLM
  0.18.x `StageMissingLayer` raising `KeyError`). Probes that walk arbitrary
  model trees now fail closed and log at `DEBUG` instead of aborting the run.
- Diagnostic tables (block assignment, memory transfer) now appear under
  vLLM and standalone when `FT_ENABLE_DIAGNOSTICS=1`, regardless of the
  host application's log level.

## [0.2.0] — 2026-04-16

### Added

- `offload_from_profile()` convenience API that combines `init`, `load_profile`, and `offload`
  into a single call for loading a saved profile and going straight to inference.
- Diffusers examples (`examples/diffusers/quickstart/` and `examples/diffusers/profile-reuse/`)
  demonstrating FlexTensor offloading with the Wan2.2 text-to-video model, including profile
  save/load workflow.

- `scale` parameter on `GreedyStrategy` for parity with `KnapsackStrategy` and
  `AdaptiveStrategy`. Multiplies the cumulative compute budget (< 1 adds safety
  margin, > 1 allows more transfers). Defaults to `1.0` (no change in behaviour).
- `exclude_patterns` config option for keeping specific modules or parameters on GPU
  (e.g., `lm_head`, `*.norm`, `*.scale`). Applied after `include_patterns`.
- `include_patterns` now accepts parameter-level patterns such as `*.weight` or
  `layers.*.attn.q_proj.weight`, allowing users to offload specific parameters
  while keeping others (e.g., biases, norms) on GPU for faster inference.
  Module-level patterns (e.g., `layers.*`) continue to include all parameters
  of the matched module.

### Deprecated

- `OffloadState` class — use `OffloadPhase` instead. Will be removed in v0.4.0.
- `OffloadPhase.WARMUP` member — use `OffloadPhase.DISCOVERY` instead. Will be removed in v0.4.0.
- `OffloadPhase.PROFILE` member — use `OffloadPhase.PROFILING` instead. Will be removed in v0.4.0.
- `warmup_iters` config field — use `discovery_iters` instead. Will be removed in v0.4.0.
- `profile_iters` config field — use `profiling_iters` instead. Will be removed in v0.4.0.
- `all_warmup_iters` property — use `pre_inference_iters` instead. Will be removed in v0.4.0.
- `knapsack_scale` config option — use `transfer_budget_scale` instead. Will be removed in v0.3.
- `module_patterns` config option — use `include_patterns` instead. Will be removed in v0.3.
- `max_gpu_mem_bytes` config option — use `max_gpu_mem_fraction` instead. Will be removed in v0.3.
- `use_shared_memory` config option — use `shm_enabled` instead. Will be removed in v0.3.

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

### Changed

- **BREAKING**: Renamed `OffloadState` enum to `OffloadPhase` with members
  `DISCOVERY` (was `WARMUP`) and `PROFILING` (was `PROFILE`). Config fields
  `warmup_iters`/`profile_iters` renamed to `discovery_iters`/`profiling_iters`.
  `all_warmup_iters` property renamed to `pre_inference_iters`. Old names
  are deprecated aliases with `DeprecationWarning` — removal in v0.4.0.
- `knapsack_scale` renamed to `transfer_budget_scale`. The environment
  variable `FT_KNAPSACK_SCALE` is now `FT_TRANSFER_BUDGET_SCALE`. The old field name
  and env var still work but emit a deprecation warning (removed in v0.3).
- `module_patterns` renamed to `include_patterns`. The environment variable
  `FT_MODULE_PATTERNS` is now `FT_INCLUDE_PATTERNS`. The old names still work but
  emit a deprecation warning.

- vLLM example updated to v0.19.0 with shell scripts replacing the Dockerfile.

- **BREAKING**: `SHM_PROTOCOL_VERSION` bumped to 2. Existing shared-memory profiles are
  invalidated due to namespace hash key changes (`module_patterns` → `include_patterns`,
  added `exclude_patterns`). Multi-process deployments will re-profile on first run after
  upgrade.

- **BREAKING**: Profile `SCHEMA_VERSION` bumped to 2. Offload block labels now use the
  module path (e.g., `layers.0.attn`) instead of `ParentClassName.module_name`
  (e.g., `TransformerLayer.attn`). The old format was incompatible with the new
  parameter-level `include_patterns`. Saved profiles must be re-generated.

### Fixed

- vLLM integration now supports v0.11.2 through v0.19.0. Previously broke on
  v0.16+ due to `vllm.attention` module being moved to
  `vllm.model_executor.layers.attention`, and on v0.18+ due to a new
  `subfolder` parameter added to the model loader's weight preparation method.
- FlexTensor log messages (offloading config, loader phases) now visible on
  vLLM v0.18+ where spawn-mode child processes don't inherit parent logging
  state. Loggers use the `vllm.*` namespace to flow through vLLM's handler.
- vLLM snapshot dumps now serialize module parameters as metadata instead of raw
  tensor values, which produced megabytes of unbounded output.
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

[Unreleased]: https://github.com/ai-dynamo/flextensor/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ai-dynamo/flextensor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ai-dynamo/flextensor/releases/tag/v0.1.0
