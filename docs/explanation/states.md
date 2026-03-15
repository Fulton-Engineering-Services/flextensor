<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Understanding FlexTensor's Internal States

This document explains the internal state machine that FlexTensor uses to manage tensor offloading. These states are implementation details that users don't need to interact with directly—FlexTensor handles state transitions automatically. However, understanding them helps explain what happens during model execution and aids in debugging.

## Overview

FlexTensor uses an internal state machine with four states: **Not Initialized**, **Warmup**, **Profile**, and **Inference**. After initialization, the system automatically progresses through three active states (Warmup, Profile, Inference) during the first few model iterations.

### Why States?

Tensor offloading presents a fundamental challenge: to efficiently move tensors between CPU and GPU, we need to know *which* tensors to offload and *when* to transfer them. Making poor decisions leads to either memory exhaustion (offloading too little) or performance degradation (offloading too much or at the wrong time).

FlexTensor solves this by learning about your model's tensor usage patterns before optimizing:

1. **Discovers** which tensors belong to which layers (Warmup state)
2. **Measures** execution timing with tensor loading (Profile state)
3. **Applies** an optimized offloading strategy (Inference state)

This learning process happens transparently during the first few iterations of model execution.

## State Progression

```
NOT_INITIALIZED → WARMUP → PROFILE → INFERENCE
                    ↓          ↓
              (warmup_iters) (profile_iters)
```

The `OffloadManager` tracks the current state internally and transitions automatically based on iteration counts configured via `OffloadConfig`:

```python
config = OffloadConfig(
    warmup_iters=1,    # Iterations in warmup state
    profile_iters=10,  # Iterations in profile state
)
```

## Warmup State

### Purpose

The warmup state identifies the relationship between layers and tensors. It answers the question: "Which tensors does each layer use?"

### What Happens Internally

1. **Model Preprocessing**: Tensors are moved to CPU and optionally pinned in memory for faster transfers
2. **Tensor Tracking**: Each tensor operation is intercepted via `WarmupTrap`
3. **On-Demand Transfer**: When a layer needs a tensor, it's copied to GPU, used, then released
4. **Statistics Collection**: The system records which tensor IDs are accessed by each layer

### Internal Mechanics

During warmup, the `WarmupTrap` context manager:

- Intercepts all PyTorch operations via `__torch_function__`
- Tracks tensor IDs used in each operation
- Copies tensors to GPU on-demand when needed for computation
- Measures baseline layer execution time
- Collects tensor-to-layer mapping data

```python
# Internal flow during warmup (simplified)
with WarmupTrap(tensor_manager, layer_name, device_gpu):
    # Tensors are copied to GPU as needed
    # Tensor IDs are recorded
    # Timing is measured
    output = layer(input)
```

### Why It Matters

Without warmup, we wouldn't know which tensors to preload for each layer. This mapping is essential for the profile state to measure realistic transfer overhead.

### Outputs and Model State

At the end of the warmup state, FlexTensor has collected:

| Output | Description |
|--------|-------------|
| `tensors_map` | Dictionary mapping tensor IDs to CPU tensor references |
| `layer_statistics_collector` | Contains tensor-to-layer mappings and baseline timing per layer |
| `model_ids` | Set of all tensor IDs belonging to the model |

**Model state after warmup**:
- All model tensors reside on **CPU memory**
- Tensors are **pinned** (if `pinned_memory=True`) for faster GPU transfers
- The model structure is unchanged—only tensor locations have moved
- No GPU memory is permanently allocated yet (tensors were copied on-demand and released)

This CPU-resident model with collected statistics is the foundation for the profile state.

## Profile State

### Purpose

The profile state measures how long each layer takes to execute *with* tensor loading overhead included. It answers: "How much time does each layer need, including tensor transfers?"

### What Happens Internally

1. **Statistics Initialization**: Layer statistics from warmup are processed
2. **Tensor Layer Loader Setup**: A loader is configured with the tensor-to-layer mapping
3. **Detailed Timing**: Each layer's execution time is measured with CUDA synchronization
4. **Duration Collection**: Statistics are accumulated across multiple iterations for accuracy

### Internal Mechanics

The profile state uses either `Trap` (indirect mode) or `TrapDirect` (direct mode):

**Indirect Mode (`Trap`)**:
- Uses PyTorch's `TorchFunctionMode` to intercept operations
- Replaces CPU tensor references with GPU copies on-the-fly
- Suitable for models where tensor access patterns vary

**Direct Mode (`TrapDirect`)**:
- Uses regular context managers without function interception
- Model is pre-patched to use GPU tensor references directly
- Lower overhead, suitable for most transformer architectures

```python
# Internal flow during profile (simplified)
with Trap(tensor_manager, layer_name, device_gpu):
    # tensor_layer_loader.enter() - preload tensors
    timer_start = time.time_ns()

    output = layer(input)

    torch.cuda.synchronize()
    timer_end = time.time_ns()
    # tensor_layer_loader.exit() - cleanup
    # Duration recorded for layer
```

### Why It Matters

Profile data enables the offloading strategy (e.g., Knapsack, Greedy, Adaptive, Global) to make informed decisions about:

- Which tensors to keep on GPU vs. offload to CPU
- When to initiate transfers to hide latency
- How to batch transfers for efficiency

### Outputs and Model State

At the end of the profile state, FlexTensor has collected:

| Output | Description |
|--------|-------------|
| `layer_stats` | Filtered layer statistics with tensor IDs and accumulated timing data |
| `tensor_statistics_map` | Per-tensor statistics (size, transfer time estimates) |
| `tensor_layer_loader` | Configured loader with tensor-to-layer mapping |

**Model state after profile**:
- Model tensors still reside on **CPU memory**
- In direct mode: model may have been copied with shared tensor references for profiling
- `layer_statistics_collector` contains averaged timing across all profile iterations
- System is ready to compute the optimal offloading strategy

The collected timing data is the input for strategy computation in the inference state.

## Inference State

### Purpose

The inference state applies the computed offloading strategy for production execution. It answers: "How do we execute optimally based on what we learned?"

### What Happens Internally

1. **Strategy Computation**: The load strategy (e.g., Knapsack) determines which tensors to load at each layer
2. **Release Strategy**: Complementary strategy determines when to release GPU memory
3. **Tensor Loader Selection**: The appropriate loader is configured based on `transfer_mode`
4. **Model Finalization**: The model is patched with optimized tensor access patterns

### Internal Mechanics

The inference state uses `TrapInfer` or `TrapInferDirect`:

**Key Differences from Profile**:
- No timing collection (removes measurement overhead)
- Tensor loading follows the precomputed schedule
- Memory is released according to the release strategy
- Optimized for throughput rather than data collection

### Transfer Modes

FlexTensor supports different transfer strategies internally:

| Mode | Description |
|------|-------------|
| `strategy` | Basic strategy-based loading and releasing |
| `allocation_block_transfer` | Pre-allocated GPU blocks with batch transfers |
| `raw_block_transfer` | Direct memory management for maximum efficiency |

### Why It Matters

The inference state is where the performance gains are realized. By using the learned statistics and optimized strategies, FlexTensor achieves:

- Minimal GPU memory usage (tensors offloaded to CPU when not needed)
- Low latency overhead (transfers scheduled to overlap with computation)
- Consistent performance (deterministic based on learned patterns)

### Outputs and Model State

At the end of state transition to inference, FlexTensor has computed:

| Output | Description |
|--------|-------------|
| `load_strategy` | Per-layer decisions on which tensors to load to GPU |
| `release_strategy` | Per-layer decisions on which tensors to release from GPU |
| `tensor_layer_loader` | Production-optimized loader (e.g., `TensorStrategyLoader` or block-based) |
| `stats` | Computed layer statistics with sizes and timing |

**Model state in inference**:
- Model is **finalized** via `prepare_final_model()` with optimized tensor access
- Tensors are loaded to GPU **on-demand** per the computed schedule
- GPU memory is **dynamically allocated and released** according to the release strategy
- The model reference in tensor manager is cleared (`self.model = None`) as it's no longer needed for profiling

**Runtime behavior**:
- Tensors are preloaded to GPU before each layer executes (`tensor_layer_loader.enter()`)
- After layer execution, tensors may be released based on the release strategy (`tensor_layer_loader.exit()`)
- No timing collection overhead—optimized for production throughput

This is the steady-state for all subsequent model executions.

## State Transitions

The `OffloadManager` handles state transitions automatically and internally:

```python
def update_state(self):
    """Check and update state if necessary."""
    if self._current_state in {OffloadState.NOT_INITIALIZED, OffloadState.INFERENCE}:
        return
    self._iteration_count += 1

    if self._current_state == OffloadState.WARMUP and self._iteration_count >= self.config.warmup_iters:
        self._transition_to_profile()
    elif self._current_state == OffloadState.PROFILE and self._iteration_count >= self.config.profile_iters:
        self._transition_to_inference()
```

Each transition:
1. Prepares the tensor manager for the new state
2. Updates the model with appropriate hooks
3. Resets the iteration counter

Users don't need to call this method—it's invoked automatically when using `offload_block` or patched modules.

## Configuration

While users don't interact with states directly, they can influence state behavior through configuration:

### Warmup Iterations (`warmup_iters`)

- **Default**: 1
- **Purpose**: How many iterations to run in warmup state
- **Guidance**: Usually 1 is sufficient unless your model has dynamic tensor access patterns

### Profile Iterations (`profile_iters`)

- **Default**: 10
- **Purpose**: How many iterations to collect timing data
- **Guidance**: More iterations = more accurate statistics, but longer startup time

### Example

```python
from flextensor import OffloadConfig, get_offload_manager, offload

# Configuration affects internal state duration
config = OffloadConfig(
    warmup_iters=1,
    profile_iters=5,
    module_patterns=["layers.*"],
)

model = offload(model, config=config)

# First warmup_iters + profile_iters iterations: internal learning
# Subsequent iterations: optimized inference
for batch in dataloader:
    output = model(batch)
```

## Design Consequences

Understanding the state machine explains some behaviors that might otherwise seem surprising.

**Profile iteration count affects strategy quality.** The profile state accumulates timing statistics across `profile_iters` iterations. Too few iterations can produce inaccurate estimates, especially on systems with thermal throttling or when sharing the GPU with other processes. If you observe variable inference performance, increasing `profile_iters` gives the strategy more data to work with.

**Tensor-to-layer mapping is fixed after warmup.** The warmup state builds a complete map of which tensors belong to which layers. Any module added to the model after `offload()` is called will not be part of this map and will not be offloaded. The system does not re-run warmup when the model changes.

**State transitions are irreversible within a run.** Once FlexTensor enters inference state, it does not return to warmup or profile. To re-profile (for example, after changing the model), call `release()` and then `offload()` again.

**Profile persistence can skip warmup and profile.** By saving the offloading profile after the first run (using `save_profile()`), subsequent runs can load the saved state (using `load_profile()`) and skip the warmup and profile phases entirely. The loaded state contains the tensor-to-layer mappings, timing statistics, and computed strategy, allowing the system to proceed directly to inference-ready configuration. See `OffloadConfig.profile_storage_dir` for configuration.

For practical guidance on avoiding mistakes related to these behaviors, see [Troubleshooting](../how-to/troubleshooting.md).

## Summary

| State | Purpose | Internal Trap Class | Key Output |
|-------|---------|---------------------|------------|
| Warmup | Discover tensor-layer relationships | `WarmupTrap` | Tensor ID mapping |
| Profile | Measure execution timing | `Trap` / `TrapDirect` | Layer statistics |
| Inference | Apply optimized strategy | `TrapInfer` / `TrapInferDirect` | Production execution |

This state machine approach enables FlexTensor to intelligently offload tensors without requiring manual configuration. By learning from your model's actual behavior, it achieves near-optimal memory efficiency with minimal performance overhead -- all transparently managed as an internal implementation detail.
