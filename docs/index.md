<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# FlexTensor

FlexTensor is a tensor offloading library for PyTorch that enables running large models on limited GPU memory by intelligently offloading tensors between GPU and CPU memory.

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

FlexTensor targets less than 5% latency overhead compared to a baseline without offloading. See the [Dashboard](https://github.com/ai-dynamo/flextensor) for benchmark results.
