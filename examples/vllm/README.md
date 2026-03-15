<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# vLLM Model Weight Offloading with FlexTensor

This example demonstrates how to use FlexTensor's tensor offloading with vLLM to run larger models with limited GPU memory.

## Prerequisites and Hardware Requirements

**GPU memory**: A single GPU with at least 40 GB of VRAM is recommended for 70B-parameter models. With FlexTensor offloading, models that would normally require multiple high-memory GPUs can run on a single GPU by keeping only the active layer's tensors on the GPU at any time. Exact requirements depend on the model size, sequence length, and KV cache allocation.

**Compatible vLLM versions**: This integration requires vLLM 0.11.x or later (uses the `vllm.v1` worker API). The `FlexTensorOffloadWorker` class extends vLLM's internal worker API, which may change between major vLLM releases.

## Feature Support

| Feature | Status | Notes |
|---------|--------|-------|
| Tensor Parallelism (TP) | Supported | |
| Pipeline Parallelism (PP) | Supported | |
| Data Parallelism (DP) | In progress | |
| Expert Parallelism (EP) | In progress | |
| CUDA Graphs | In progress | Currently requires `--enforce-eager` |
| Multi-Token Prediction (MTP) | Not supported | MTP targets decode; FlexTensor offloading targets prefill |

## Quick Start

After [installing dependencies](#installation):

```bash
# Enable offloading and serve a model
FT_ENABLED=1 FT_MAX_GPU_MEM_FRACTION=0.7 vllm serve Qwen/Qwen2.5-72B-Instruct \
    --enforce-eager \
    --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker
```

## Installation

Install example requirements:

```bash
pip install -r examples/vllm/requirements.txt
```

## Configuration

FlexTensor loads configuration from environment variables (`FT_*`).

Every `OffloadConfig` field can be set via an environment variable named `FT_` + the uppercase field name. For example, `warmup_iters` → `FT_WARMUP_ITERS`. Values follow standard Python literals (`true`/`false`, integers, floats, comma-separated lists).

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FT_ENABLED` | Enable offloading | `false` |
| `FT_MAX_GPU_MEM_FRACTION` | Fraction of GPU memory to use for model weights (e.g. `0.7` = 70%); `None` disables the memory constraint | `0.9` |
| `FT_MODULE_PATTERNS` | Comma-separated module patterns to offload | see below |
| `FT_ENABLE_DIAGNOSTICS` | Log Layer Duration Statistics and block assignment table after profiling | `false` |

The worker defaults to LLaMA-style transformer patterns that give each layer its own
offload trap, enabling fine-grained pipelining:

```
model.embed_tokens,model.layers.*,model.norm,lm_head,logits_processor
```

These patterns cover most HuggingFace models (Llama, Mistral, Qwen2, Gemma, Phi-3, …).
Set `FT_MODULE_PATTERNS` explicitly only when your model uses a different module layout.

For the full list of configuration options (config files, warmup/profile tuning, transfer modes), see the [Configuration Reference](https://github.com/ai-dynamo/flextensor/blob/main/docs/api/configuration.md).

## Usage Examples

### Testing the Deployment

Once the server is running, test it with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "user", "content": "The capital of France is"}
        ],
        "max_tokens": 10
    }'

# Check available models
curl http://localhost:8000/v1/models
```

## Docker

Build and run with Docker:

```bash
# Build
docker build -t vllm-flextensor examples/vllm/

# Run
docker run --gpus all -p 8000:8000 -e FT_ENABLED=1 vllm-flextensor \
    Qwen/Qwen2.5-72B-Instruct
```

## How It Works

The `FlexTensorOffloadWorker` extends vLLM's GPU worker to:

1. Load configuration from environment variables or config file
2. Apply FlexTensor offloading to model layers after loading
3. Automatically manage tensor movement between CPU and GPU

When `FT_ENABLED=1` is set, the worker applies offloading using the default LLaMA-style
module patterns (or those set via `FT_MODULE_PATTERNS`). Otherwise it behaves like the
standard vLLM worker. Look for these log messages to confirm offloading is active:

```
FlexTensor offloading enabled with config: ...
FlexTensor offloading applied (GPU usage: X.XX GiB)
```

## Troubleshooting

For troubleshooting tips (out of memory, performance tuning, debugging), see the [Troubleshooting Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/how-to/troubleshooting.md).
