<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Diffusers Model Weight Offloading with FlexTensor

Run large diffusion video-generation models on a single GPU with limited memory. FlexTensor offloads transformer weights to CPU and streams them back block-by-block during inference, overlapping transfers with computation so the GPU stays busy.

This is a minimal example showing how FlexTensor can be integrated with any Diffusers pipeline. It uses the [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) 14B-parameter text-to-video model with the Hugging Face [Diffusers](https://github.com/huggingface/diffusers) library.

## Quick Start

Install dependencies and run:

```bash
pip install -r requirements.txt
python wan_t2v.py
```

The script will:

1. Load the Wan2.2-T2V pipeline and move non-transformer components to GPU.
2. Apply FlexTensor offloading to the transformer blocks.
3. Run a warmup/profiling pass to learn the optimal prefetch schedule.
4. Run a second inference pass and export the result to `wan-t2v.mp4`.

## Prerequisites

- A CUDA-capable GPU. Only non-transformer components (VAE, text encoder) stay resident on the GPU; transformer weights are streamed from CPU, significantly reducing VRAM requirements.
- Model weights are downloaded automatically from Hugging Face Hub on first run.

**Note:** This example uses Wan2.2 which has two transformers (`transformer` and `transformer_2`). For models with a single transformer (e.g. Wan2.1), simply remove the `transformer_2` offload and release calls.

## How It Works

Instead of calling `pipe.to("cuda")` (which would load the entire model onto the GPU), only the lightweight components are moved to the GPU. The transformer — by far the largest component — stays on CPU and is wrapped with `flextensor.offload()`:

```python
pipe.transformer = flextensor.offload(
    pipe.transformer,
    config=offload_config,
    name="transformer",
)
```

FlexTensor then streams transformer weights block-by-block to the GPU during forward passes, prefetching the next block while the current one executes.

### Include Patterns

The `include_patterns` list controls which submodules are individually offloaded:

```python
include_patterns = [
    "rope",
    "patch_embedding",
    "condition_embedder",
    "blocks.*",
    "norm_out",
    "proj_out",
]
```

Each pattern becomes a separate offload unit. `blocks.*` expands to one unit per transformer block, enabling fine-grained pipelining where the next block is prefetched while the current one executes.

### Configuration

Key `OffloadConfig` parameters used in this example:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `profile_iters` | `20` | Number of diffusion steps used for profiling |
| `min_blocks` | `2` | Minimum number of blocks to keep on GPU (default is `4`; diffusion models are compute-heavy, so transfers easily hide behind computation — lowering to `2` saves GPU memory with minimal performance impact) |
| `include_patterns` | see above | Submodule patterns to offload |

For the full list of configuration options, see the [Configuration Reference](https://github.com/ai-dynamo/flextensor/blob/main/docs/api/configuration.md).

## Troubleshooting

For troubleshooting tips (out of memory, performance tuning, debugging), see the [Troubleshooting Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/how-to/troubleshooting.md).
