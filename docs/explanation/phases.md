<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Understanding FlexTensor's Internal Phases

This document explains the internal state machine that FlexTensor uses to manage tensor offloading. These phases are implementation details that users don't need to interact with directly—FlexTensor handles phase transitions automatically. However, understanding them helps explain what happens during model execution and aids in debugging.

## Overview

FlexTensor uses an internal state machine with four phases: **Not Initialized**, **Discovery**, **Profiling**, and **Inference**. After initialization, the system automatically progresses through three active phases (Discovery, Profiling, Inference) during the first few model iterations.

### Why Phases?

Tensor offloading presents a fundamental challenge: to efficiently move model weights between CPU and GPU, we need to know *which* parameters to offload and *when* to transfer them. Making poor decisions leads to either memory exhaustion (offloading too little) or performance degradation (offloading too much or at the wrong time).

FlexTensor solves this by learning about your model's tensor usage patterns before optimizing:

1. **Discovers** which parameters belong to which traps (Discovery phase)
2. **Measures** per-trap execution timing with weight loading (Profiling phase)
3. **Applies** an optimized offloading strategy (Inference phase)

This learning process happens transparently during the first few iterations of model execution.

## State Progression

```
NOT_INITIALIZED → DISCOVERY → PROFILING → INFERENCE
                       ↓           ↓
                 (discovery_iters) (profiling_iters)
```

The `OffloadManager` tracks the current phase internally and transitions automatically based on iteration counts configured via `OffloadConfig`:

```python
config = OffloadConfig(
    discovery_iters=1,    # Iterations in discovery phase
    profiling_iters=10,  # Iterations in profiling phase
)
```

## Discovery Phase

### Purpose

The discovery phase identifies the relationship between modules and their parameters. It answers the question: "Which parameters does each module use?"

### What Happens Internally

1. **Model Preprocessing**: Parameters are moved to CPU and optionally pinned in memory for faster transfers
2. **Tensor Tracking**: Each tensor operation is intercepted via `WarmupTrap`
3. **On-Demand Transfer**: When a module needs a weight, it's copied to GPU, used, then released
4. **Statistics Collection**: The system records which tensor IDs are accessed by each trap

### Internal Mechanics

During discovery, the `WarmupTrap` context manager:

- Intercepts all PyTorch operations via `__torch_function__`
- Tracks tensor IDs used in each operation
- Copies weights to GPU on-demand when needed for computation
- Measures baseline module execution time
- Collects parameter-to-trap mapping data

```python
# Internal flow during discovery (simplified)
with WarmupTrap(tensor_manager, layer_name, device_gpu):
    # Tensors are copied to GPU as needed
    # Tensor IDs are recorded
    # Timing is measured
    output = layer(input)
```

### Why It Matters

Without discovery, we wouldn't know which weights to preload for each trap. This mapping is essential for the profiling phase to measure realistic transfer overhead.

### Outputs and Model State

At the end of the discovery phase, FlexTensor has collected:

| Output | Description |
|--------|-------------|
| `tensors_map` | Dictionary mapping tensor IDs to CPU tensor references |
| `layer_statistics_collector` | Contains parameter-to-trap mappings and baseline timing per trap |
| `model_ids` | Set of all tensor IDs belonging to the model |

**Model state after discovery**:
- All model tensors reside on **CPU memory**
- For the strategy loader, tensors are **pinned** (if `pinned_memory=True`) at this stage; for block transfer loaders, pinned buffers are allocated later during inference setup
- The model structure is unchanged—only tensor locations have moved
- No GPU memory is permanently allocated yet (weights were copied on-demand and released)

This CPU-resident model with collected statistics is the foundation for the profiling phase.

## Profiling Phase

### Purpose

The profiling phase measures how long each trap takes to execute *with* weight loading overhead included. It answers: "How much time does each module need, including weight transfers?"

### What Happens Internally

1. **Statistics Initialization**: Per-trap statistics from discovery are processed
2. **Weight Loader Setup**: A loader is configured with the parameter-to-trap mapping
3. **Detailed Timing**: Each trap's execution time is measured using CUDA events (`start_event.record()` / `end_event.record()`)
4. **Duration Collection**: Statistics are accumulated across multiple iterations for accuracy

### Internal Mechanics

The profiling phase uses either `Trap` (indirect mode) or `TrapDirect` (direct mode):

**Indirect Mode (`Trap`)**:
- Uses PyTorch's `TorchFunctionMode` to intercept operations
- Replaces CPU tensor references with GPU copies on-the-fly
- Suitable for models where tensor access patterns vary

**Direct Mode (`TrapDirect`)**:
- Uses regular context managers without function interception
- Model is pre-patched to use GPU tensor references directly
- Lower overhead, suitable for most transformer architectures

```python
# Internal flow during profiling (simplified)
with Trap(tensor_manager, layer_name, device_gpu):
    # trap_nesting_guard.acquire() - prevent nested traps
    # tensor_layer_loader.enter() - preload tensors
    start_event.record()

    output = layer(input)

    end_event.record()
    end_event.synchronize()
    duration_ms = start_event.elapsed_time(end_event)
    # tensor_layer_loader.exit() - cleanup
    # Duration recorded for layer
    # trap_nesting_guard.release()
```

### Why It Matters

Profile data enables the offloading strategy (e.g., Knapsack, Greedy, Adaptive, Global) to make informed decisions about:

- Which parameters to keep on GPU vs. offload to CPU
- When to initiate transfers to hide latency
- How to batch transfers for efficiency

### Outputs and Model State

At the end of the profiling phase, FlexTensor has collected:

| Output | Description |
|--------|-------------|
| `layer_stats` | Filtered per-trap statistics with tensor IDs and accumulated timing data |
| `tensor_statistics_map` | Per-tensor statistics (size, transfer time estimates) |
| `tensor_layer_loader` | Configured loader with parameter-to-trap mapping |

**Model state after profiling**:
- Model tensors still reside on **CPU memory**
- In direct mode: model may have been copied with shared tensor references for profiling
- `layer_statistics_collector` contains averaged per-trap timing across all profiling iterations
- System is ready to compute the optimal offloading strategy

The collected timing data is the input for strategy computation in the inference phase.

## Inference Phase

### Purpose

The inference phase applies the computed offloading strategy for production execution. It answers: "How do we execute optimally based on what we learned?"

### What Happens Internally

1. **Strategy Computation**: The load strategy (e.g., Knapsack) determines which weights to load at each trap
2. **Release Strategy**: Complementary strategy determines when to release GPU memory
3. **Tensor Loader Selection**: The appropriate loader is configured based on `transfer_mode`
4. **Model Finalization**: The model is patched with optimized tensor access patterns

### Internal Mechanics

The inference phase uses `TrapInfer` or `TrapInferDirect`:

**Key Differences from Profile**:
- No timing collection (removes measurement overhead)
- Weight loading follows the precomputed schedule
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

The inference phase is where the performance gains are realized. By using the learned statistics and optimized strategies, FlexTensor achieves:

- Minimal GPU memory usage (weights offloaded to CPU when not needed)
- Low latency overhead (transfers scheduled to overlap with computation)
- Consistent performance (deterministic based on learned patterns)

### Outputs and Model State

At the end of phase transition to inference, FlexTensor has computed:

| Output | Description |
|--------|-------------|
| `load_strategy` | Per-trap decisions on which weights to load to GPU |
| `release_strategy` | Per-trap decisions on which weights to release from GPU |
| `tensor_layer_loader` | Production-optimized loader (e.g., `TensorStrategyLoader` or block-based) |
| `stats` | Computed per-trap statistics with sizes and timing |

**Model state in inference**:
- Model is **finalized** via `prepare_final_model()` with optimized tensor access
- Weights are loaded to GPU **on-demand** per the computed schedule
- GPU memory is **dynamically allocated and released** according to the release strategy
- The model reference in tensor manager is cleared (`self.model = None`) as it's no longer needed for profiling

**Runtime behavior**:
- Weights are preloaded to GPU before each module executes (`tensor_layer_loader.enter()`)
- After module execution, weights may be released based on the release strategy (`tensor_layer_loader.exit()`)
- No timing collection overhead—optimized for production throughput

This is the steady-state for all subsequent model executions.

## State Transitions

The `OffloadManager` handles phase transitions automatically and internally:

```python
def update_state(self):
    """Check and update state if necessary."""
    if self._current_phase in {OffloadPhase.NOT_INITIALIZED, OffloadPhase.INFERENCE}:
        return
    self._iteration_count += 1

    if self._current_phase == OffloadPhase.DISCOVERY and self._iteration_count >= self.config.discovery_iters:
        self._transition_to_profile()
    elif self._current_phase == OffloadPhase.PROFILING and self._iteration_count >= self.config.profiling_iters:
        self._transition_to_inference()
```

Each transition:
1. Prepares the tensor manager for the new phase
2. Updates the model with appropriate hooks
3. Resets the iteration counter

Users don't need to call this method—it's invoked automatically when using `offload_block` or patched modules.

## Configuration

While users don't interact with phases directly, they can influence phase behavior through configuration:

### Discovery Iterations (`discovery_iters`)

- **Default**: 1
- **Purpose**: How many iterations to run in discovery phase
- **Guidance**: Usually 1 is sufficient unless your model has dynamic tensor access patterns

### Profiling Iterations (`profiling_iters`)

- **Default**: 10
- **Purpose**: How many iterations to collect timing data
- **Guidance**: More iterations = more accurate statistics, but longer startup time

### Example

```python
from flextensor import OffloadConfig, get_offload_manager, offload

# Configuration affects internal phase duration
config = OffloadConfig(
    discovery_iters=1,
    profiling_iters=5,
    include_patterns=["layers.*"],
)

model = offload(model, config=config)

# First discovery_iters + profiling_iters iterations: internal learning
# Subsequent iterations: optimized inference
for batch in dataloader:
    output = model(batch)
```

## Design Consequences

Understanding the state machine explains some behaviors that might otherwise seem surprising.

**Profiling iteration count affects strategy quality.** The profiling phase accumulates timing statistics across `profiling_iters` iterations. Too few iterations can produce inaccurate estimates, especially on systems with thermal throttling or when sharing the GPU with other processes. If you observe variable inference performance, increasing `profiling_iters` gives the strategy more data to work with.

**Parameter-to-trap mapping is fixed after discovery.** The discovery phase builds a complete map of which parameters belong to which traps. Any module added to the model after `offload()` is called will not be part of this map and will not be offloaded. The system does not re-run discovery when the model changes.

**Phase transitions are irreversible within a run.** Once FlexTensor enters inference phase, it does not return to discovery or profiling. To re-profile (for example, after changing the model), call `release()` and then `offload()` again.

**Profile persistence can skip discovery and profiling.** By saving the offloading profile after the first run (using `save_profile()`), subsequent runs can load the saved state (using `load_profile()`) and skip the discovery and profiling phases entirely. The loaded state contains the parameter-to-trap mappings, timing statistics, and computed strategy, allowing the system to proceed directly to inference-ready configuration. See `OffloadConfig.profile_storage_dir` for configuration.

For practical guidance on avoiding mistakes related to these behaviors, see [Troubleshooting](../how-to/troubleshooting.md).

## Summary

| Phase | Purpose | Internal Trap Class | Key Output |
|-------|---------|---------------------|------------|
| Discovery | Discover parameter-to-trap relationships | `WarmupTrap` | Parameter ID mapping |
| Profiling | Measure per-trap execution timing | `Trap` / `TrapDirect` | Per-trap statistics |
| Inference | Apply optimized strategy | `TrapInfer` / `TrapInferDirect` | Production execution |

This state machine approach enables FlexTensor to intelligently offload model weights without requiring manual configuration. By learning from your model's actual behavior, it achieves near-optimal memory efficiency with minimal performance overhead -- all transparently managed as an internal implementation detail.
