<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Advanced API

Lower-level components for custom workflows. In most cases, use the
[Simplified API](simplified.md) or [`OffloadManager`](offload-manager.md) instead.

`TensorManager` provides direct control over discovery, profiling, and inference phases.
See [Internal Phases](../explanation/phases.md) for details on how these components work together.

::: flextensor.tensor_manager.TensorManager
    options:
      members:
        - set_model
        - initialize_warmup
        - initialize_profile
        - initialize_inference
        - prepare_warmup_mode
        - prepare_profile_mode
        - prepare_infer_mode
        - trap
        - release_memory
        - get_gpu_memory_usage
        - get_memory_transfer_stats
        - benchmark_context
        - save_profile
        - load_profile
        - load_state
        - restore_state
        - plan_state_adoption
        - execute_state_adoption
        - restore_adopted_state

## Benchmark Modes

These are used internally by `OffloadManager` and exposed for users who need custom benchmark instrumentation.

::: flextensor.benchmark_tensor_mode.TensorBenchmarkMode
    options:
      members:
        - get_results

::: flextensor.benchmark_tensor_mode.BenchmarkReplace
    options:
      members:
        - get_results

::: flextensor.benchmark_tensor_mode.PreloadToDevice
    options:
      members:
        - get_results

## Host Pinning

Lower-level helpers for `pinned_memory_mode` dispatch and `cudaHostRegister`
registrations. See [Configuration](../explanation/configuration.md#pinned-memory)
for user-facing semantics and
[Troubleshooting](../how-to/troubleshooting.md#step-3-switch-to-in-place-pinning)
for when to switch modes.

::: flextensor.host_pinning
    options:
      members:
        - PinnedMemoryMode
        - HostPinner
        - HostPinRegistry
        - NoOpHostPinner
        - is_available
        - make_host_pinner
