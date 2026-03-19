<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# vLLM Model Weight Offloading with FlexTensor

This example demonstrates how to use FlexTensor's tensor offloading with vLLM to run larger models with limited GPU memory.

## Prerequisites and Hardware Requirements

**GPU memory**: A CUDA-capable GPU is required. Peak VRAM depends on model size, sequence length, and KV cache; FlexTensor offloads weights to CPU, significantly reducing requirements compared to a full model load.

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

The worker defaults to decoder-only transformer patterns that give each layer its own
offload trap, enabling fine-grained pipelining:

```
model.embed_tokens,model.layers.*,model.norm,lm_head,logits_processor
```

These patterns work out of the box for most post-2023 decoder-only models: LLaMA 2–4, Mistral, Mixtral, Qwen2/2.5/3, Phi-3/4, Gemma 2/3, DeepSeek V2/V3, Nemotron, OLMo/OLMo2, Granite, StarCoder2, and others that follow the same `model.*` layout.

For models with a different module layout, set `FT_MODULE_PATTERNS` explicitly:

| Family | `FT_MODULE_PATTERNS` |
|---|---|
| GPT-2 / GPT-J / Falcon v1 / BLOOM / Qwen1 | `transformer.wte,transformer.h.*,transformer.ln_f,lm_head,logits_processor` |
| GPT-NeoX / Pythia | `gpt_neox.embed_in,gpt_neox.layers.*,gpt_neox.final_layer_norm,embed_out,logits_processor` |
| MPT / DBRX | `transformer.wte,transformer.blocks.*,transformer.norm_f,lm_head,logits_processor` |
| Mamba / Mamba2 | `backbone.embeddings,backbone.layers.*,backbone.norm_f,lm_head,logits_processor` |
| Phi-1 / Phi-2 / Jamba / Bamba | `model.embed_tokens,model.layers.*,model.final_layernorm,lm_head,logits_processor` |
| InternLM2 | `model.tok_embeddings,model.layers.*,model.norm,output,logits_processor` |
| OPT | `model.decoder.embed_tokens,model.decoder.layers.*,model.decoder.final_layer_norm,lm_head,logits_processor` |
| ChatGLM v1–3 | `transformer.embedding,transformer.encoder.layers.*,transformer.encoder.final_layernorm,lm_head,logits_processor` |

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

When `FT_ENABLED=1` is set, the worker applies offloading using the default decoder-only transformer
module patterns (or those set via `FT_MODULE_PATTERNS`). Otherwise it behaves like the
standard vLLM worker. Look for these log messages to confirm offloading is active:

```
FlexTensor offloading enabled with config: ...
FlexTensor offloading applied (GPU usage: X.XX GiB)
```

## Troubleshooting

For troubleshooting tips (out of memory, performance tuning, debugging), see the [Troubleshooting Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/how-to/troubleshooting.md).
