<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# FlexTensor

FlexTensor is a tensor offloading library for PyTorch that enables running large models on limited GPU memory by intelligently offloading model weights between GPU and CPU memory.

## Features

- **Automatic Model Patching**: Offload model layers without modifying model code
- **Manual Control**: Fine-grained control with `offload_block` context managers
- **Smart Profiling**: Automatic warmup and profiling for optimal performance
- **Wildcard Support**: Use patterns like `"layers.*"` to offload multiple modules
- **Profile Persistence**: Save and load offloading profiles for faster startup
- **Lazy Model Initialization**: Load models from saved profiles with optimized weight loading
- **Shared Memory**: Optional shared memory subsystem for cross-process tensor coordination

## Quick Example

```python
import torch
import flextensor
from flextensor import OffloadConfig

# Load your model
model = YourModel()

# Configure offloading
config = OffloadConfig(
    gpu_device=0,
    warmup_iters=1,
    profile_iters=10,
    module_patterns=["embed", "layers.*", "head"],
)

# Patch model - no code changes needed
model = flextensor.offload(model, config=config)

# Use model normally
output = model(input_tensor)
```

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    Install FlexTensor and get up and running in minutes

    [:octicons-arrow-right-24: Quick Start](quick-start.md)

-   :material-cog:{ .lg .middle } **Configuration**

    ---

    Understand all configuration options and how to tune them

    [:octicons-arrow-right-24: Configuration](explanation/configuration.md)

-   :material-tools:{ .lg .middle } **How-To Guides**

    ---

    Solve specific problems with step-by-step instructions

    [:octicons-arrow-right-24: Troubleshooting](how-to/troubleshooting.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete technical reference for all functions and classes

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>

## Performance

FlexTensor targets less than 5% latency overhead compared to a baseline without offloading, when the CPU-to-GPU interconnect bandwidth is sufficient to transfer offloaded weights within available compute time. Overhead increases when this condition is not met — for example, at low request concurrency with high offload ratios. See [Troubleshooting](how-to/troubleshooting.md#when-to-expect-higher-overhead) for details on when overhead may exceed this target.

The <5% target assumes:

- **Interconnect bandwidth matches workload** — the CPU-to-GPU bandwidth is sufficient to complete weight transfers within the compute time available to overlap them.
- **Sufficient GPU memory for double-buffering** — the GPU can hold the weights currently being computed and the weights being transferred simultaneously, plus activations and KV-cache.
- **Representative profiling** — the batch size used during profiling reflects the production workload. If profiling runs at large prefill size but production is dominated by single-token decode, the offloading strategy may be suboptimal.
