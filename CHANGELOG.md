<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Avoid raw CUDA OOM during inference setup by budgeting for available GPU
  memory and permanent GPU tensors before finalizing offloaded models.
- Better direct-offload support for custom kernels and cross-layer parameter getters.

## [0.2.1] — 2026-05-19

### Added

- `pinned_memory_mode` option (`"torch"` default, `"host_register"` opt-in) to
  pin existing allocations in place via `cudaHostRegister` and avoid PyTorch's
  peak-memory doubling on power-of-two boundaries
  ([pytorch#150517](https://github.com/pytorch/pytorch/issues/150517)).
  See [Switch to in-place pinning](docs/how-to/troubleshooting.md#step-3-switch-to-in-place-pinning).
- Profiling data control API (`clear_profiling_durations()`,
  `suspend_profiling()` / `resume_profiling()`, `pause_profiling()` context
  manager) so backends can bracket mixed-batch warmup without polluting the
  PROFILING input or consuming the `profiling_iters` budget. See
  [Profiling Data Control](docs/explanation/phases.md#profiling-data-control).
- `torch.compile` support for offloaded models (profile and inference;
  discovery must stay eager). See [docs/how-to/torch-compile.md](docs/how-to/torch-compile.md).
- `class:<glob>` prefix on `include_patterns` / `exclude_patterns` that matches
  on the module's class (short and fully-qualified) instead of its path.
  A `name:<glob>` prefix is accepted for symmetry; bare patterns remain
  name-based. See [Pattern Matching](docs/explanation/pattern-matching.md).

### Changed

- `pinned_memory=True` on a CPU-only host now raises `RuntimeError` at
  `TensorManager` construction (was a silent no-op).
- `UntimedTrapsReport` is emitted at `WARNING` whenever a trap has tensor IDs
  but no duration samples, regardless of `enable_diagnostics`.
- User-registered forward hooks on the top-level offloaded model now always
  observe the **post-transition** phase (previously order-dependent on whether
  the hook was registered before or after `offload()`). Sub-module hooks are
  unchanged.
- `OffloadConfig.include_patterns` / `exclude_patterns` now reject non-string,
  empty, and typo-prefixed entries (`clas:`, `Class:`, …) at construction
  with `ValueError`; whitespace is stripped.
- Internal: `WarmupTrap` no longer records per-iteration durations, and the
  DISCOVERY-only recorder is `TensorManager.record_tensors(label, tensor_ids)`.
  `BenchmarkReplace.__init__` now requires a keyword-only `host_pinner`.
  Affects plugin authors who subclassed `WarmupTrap` or called the recorders
  directly.

### Fixed

- Include pattern derivation now skips unmatched sibling module patterns
  instead of truncating them to broad ancestors, preserving per-layer traps on
  wrapper models.
- `FlexTensorOffloadWorker` pushes the speculative-decoding drafter (MTP /
  Eagle) to GPU before `offload()` runs, fixing the Dynamo `cpu/cuda` device
  mismatch in vLLM's `@torch.compile` layernorm helper on MTP + FT.
- `dump_instrumentation()` now degrades gracefully on frozen maps (e.g.
  `MappingProxyType`) and non-JSON values instead of aborting.
- Model discovery no longer crashes on modules / tensor subclasses whose
  attribute access raises non-`AttributeError` (e.g. vLLM 0.18.x
  `StageMissingLayer` raising `KeyError`); probes fail closed and log at
  `DEBUG`.
- Diagnostic tables (block assignment, memory transfer) now appear under vLLM
  and standalone when `FT_ENABLE_DIAGNOSTICS=1`, regardless of host log level.
- vLLM online quantization loading now works with FlexTensor's CPU-first
  loader for BF16 checkpoints larger than target GPU memory, by deferring
  vLLM's CUDA-only weight processing until the layer-by-layer GPU phase.

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
  causing OOM.

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

[Unreleased]: https://github.com/ai-dynamo/flextensor/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/ai-dynamo/flextensor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ai-dynamo/flextensor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ai-dynamo/flextensor/releases/tag/v0.1.0
