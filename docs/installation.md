<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Installation

## Requirements

- Python 3.10 or higher
- PyTorch 2.5 or higher (with CUDA support)
- CUDA-capable GPU

## Install from Package

FlexTensor is available on PyPI. Use the following command to install:

```bash
pip install flextensor
```

## Install from Source

Clone the repository and install with:

```bash
git clone https://github.com/ai-dynamo/flextensor.git
cd flex-tensor
pip install .
```

## Development Installation

For development, install with all optional dependencies:

```bash
pip install -e ".[all]"
```

This includes:

- Testing tools (pytest, pytest-cov)
- Pre-commit hooks
- Debugging tools (pudb)
- Documentation generation tools (mkdocs, mike)

## Verify Installation

```python
import flextensor
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"FlexTensor version: {flextensor.__version__}")
```
