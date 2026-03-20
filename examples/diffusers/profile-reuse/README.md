<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Diffusers Advanced: Profile Save & Load

Profile once, then generate as many videos as you want without re-profiling. This example splits the workflow into a one-time profiling step and a fast inference step, so subsequent runs skip warmup and profiling entirely.

This example uses the [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) 14B-parameter text-to-video model with the Hugging Face [Diffusers](https://github.com/huggingface/diffusers) library.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

**Step 1 — Profile and save** (run once):

```bash
python run_profile.py --profile-dir ./wan_profile
```

**Step 2 — Inference with saved profile** (run as many times as you like):

```bash
python run_infer.py \
    --profile-dir ./wan_profile \
    --prompt "A kitten curled up next to a crackling fireplace, snowflakes drifting past a frost-covered window in the background" \
    --output kitten.mp4
```

Generate more videos without re-profiling:

```bash
python run_infer.py \
    --profile-dir ./wan_profile \
    --prompt "An astronaut floating above Earth, slowly turning to face the camera as sunlight breaks over the planet's horizon" \
    --output astronaut.mp4

python run_infer.py \
    --profile-dir ./wan_profile \
    --prompt "An eagle soaring through a canyon at dawn, its shadow racing along the red sandstone walls below" \
    --output eagle.mp4
```

## Prerequisites

- A CUDA-capable GPU.
- Model weights are downloaded automatically from Hugging Face Hub on first run.

## Overview

The workflow is split into two scripts:

| Script | Purpose |
|--------|---------|
| `run_profile.py` | Runs warmup + profiling, then saves the profile to disk |
| `run_infer.py` | Loads the saved profile and runs inference immediately |

The profiling step creates a `wan_profile/` directory with separate subdirectories for each transformer's offload schedule:

```
wan_profile/
├── transformer/
│   └── profile.json
└── transformer2/
    └── profile.json
```

## How It Works

### Profiling (`run_profile.py`)

1. Loads the Wan2.2 pipeline and moves non-transformer components to GPU.
2. Wraps both transformers with `flextensor.offload()`.
3. Runs a generation pass — FlexTensor observes layer execution patterns and tensor access timings.
4. Calls `flextensor.save_profile()` to serialize the learned offload schedule to disk.

### Inference (`run_infer.py`)

1. Loads the pipeline and moves non-transformer components to GPU.
2. Calls `flextensor.offload_from_profile()` for each transformer — this loads the saved profile and offloads the model in one step, skipping warmup and profiling entirely.
3. Runs inference at full speed.

### Key API

```python
# After profiling completes, save each profile to its own directory:
flextensor.save_profile("./wan_profile/transformer", name="transformer")
flextensor.save_profile("./wan_profile/transformer2", name="transformer2")

# In a later session, load and offload in one call:
pipe.transformer = flextensor.offload_from_profile(
    pipe.transformer, "./wan_profile/transformer", config=offload_config, name="transformer",
)
```

## Configuration

| Parameter | `run_profile.py` | `run_infer.py` |
|-----------|-------------|------------|
| `profile_iters` | `20` | not needed |
| `min_blocks` | `2` | `2` (default is `4`; lowered to reduce GPU memory usage) |
| `module_patterns` | see scripts | must match between profiling and inference |

For the full list of configuration options, see the [Configuration Reference](https://github.com/ai-dynamo/flextensor/blob/main/docs/api/configuration.md).

## Troubleshooting

For troubleshooting tips (out of memory, performance tuning, debugging), see the [Troubleshooting Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/how-to/troubleshooting.md).
