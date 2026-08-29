<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# FlexTensor

[![Documentation](https://img.shields.io/badge/Documentation-View-green)](https://github.com/ai-dynamo/flextensor)
[![Dashboard](https://img.shields.io/badge/Dashboard-View-blue)](https://github.com/ai-dynamo/flextensor)

FlexTensor is a tensor offloading and management library for PyTorch that enables running large models on limited GPU memory by intelligently offloading tensors between GPU and CPU memory.

## Features

- **Simplified API**: Easy-to-use high-level API for automatic tensor offloading
- **Automatic Model Patching**: Offload model layers without modifying model code
- **Manual Control**: Fine-grained control with `offload_block` context managers
- **Smart Profiling**: Automatic discovery and profiling for optimal performance
- **Wildcard Support**: Use patterns like `"layers.*"` to offload multiple modules
- **Profile Persistence**: Save and load offloading profiles for faster startup
- **Lazy Model Initialization**: Load models from saved profiles with optimized weight loading
- **Shared Memory**: Optional shared memory subsystem for cross-process tensor coordination
- **NVMe Disk Offload**: Evict cold weights to a local NVMe SSD with cuFile (GDS) direct-to-GPU reads or POSIX `pread` fallback, reducing host memory pressure when both GPU and CPU RAM are constrained

## Documentation

For detailed guides, API reference, and more, visit our [Documentation](https://github.com/ai-dynamo/flextensor).

## Quick Installation

To install FlexTensor from PyPI:

```bash
pip install flextensor
```

For more installation options (source and development), see the [Installation Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/installation.md).

## Quick Example

```python
import flextensor
from flextensor import OffloadConfig

# Your existing model
model = YourModel()

# Configure offloading
config = OffloadConfig(
    gpu_device=0,                   # GPU to use
    profiling_iters=10,             # Iterations for timing measurement
    include_patterns=["layers.*"],  # Which modules to offload
)

# Patch the model
model = flextensor.offload(model, config=config)

# Use normally — the first few iterations warm the manager
# (`discovery_iters` + `profiling_iters` under the default
# `skip_discovery=False`; query
# `flextensor.get_offload_manager().iters_before_inference` for
# the exact path-aware count).
for batch in dataloader:
    output = model(batch)  # FlexTensor handles everything
```

See the [Quick Start](https://github.com/ai-dynamo/flextensor/blob/main/docs/quick-start.md) for more examples.

## License

FlexTensor is licensed under the [Apache License 2.0](LICENSE). See
[NOTICE](NOTICE) for the project notice, [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for
third-party dependency attributions, and
[EXTERNAL_MATERIALS.md](EXTERNAL_MATERIALS.md) for external materials.
