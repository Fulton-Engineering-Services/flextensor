<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# vLLM Model Weight Offloading with FlexTensor [Coming Soon]

This example demonstrates how to use FlexTensor's tensor offloading with vLLM to run larger models with limited GPU memory.

## Prerequisites and Hardware Requirements

**GPU memory**: A CUDA-capable GPU is required. Peak VRAM depends on model size, sequence length, and KV cache; FlexTensor offloads weights to CPU, significantly reducing requirements compared to a full model load.

**Compatible vLLM versions**: This integration requires vLLM 0.11.x or later (uses the `vllm.v1` worker API). The `FlexTensorOffloadWorker` class extends vLLM's internal worker API, which may change between major vLLM releases.

**Docker image**: This example uses the official [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai) image (`v0.19.0`).

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

Start the server inside the vLLM container:

```bash
docker run --gpus all -p 8000:8000 \
    -v ./examples/vllm:/workspace \
    vllm/vllm-openai:v0.19.0 \
    bash -c "/workspace/install.sh && /workspace/serve.sh Qwen/Qwen2.5-72B-Instruct"
```

Pass `FT_*` environment variables with `-e` to configure offloading:

```bash
docker run --gpus all -p 8000:8000 \
    -e FT_MAX_GPU_MEM_FRACTION=0.7 \
    -e FT_ENABLE_DIAGNOSTICS=1 \
    -v ./examples/vllm:/workspace \
    vllm/vllm-openai:v0.19.0 \
    bash -c "/workspace/install.sh && /workspace/serve.sh Qwen/Qwen2.5-72B-Instruct"
```

Test it from the host:

```bash
bash examples/vllm/client.sh Qwen/Qwen2.5-72B-Instruct
```

## Scripts

### install.sh

Installs FlexTensor and its dependencies inside the container:

```bash
bash /workspace/install.sh
```

### serve.sh

Starts vLLM with the FlexTensor offload worker. Takes the model name as the first argument:

```bash
bash /workspace/serve.sh MODEL_NAME
```

### client.sh

Sends a chat completion request and lists available models. Takes the model name as the first argument and an optional port (default 8000):

```bash
bash examples/vllm/client.sh MODEL_NAME [PORT]
```

## Configuration

FlexTensor loads configuration from environment variables (`FT_*`).

Every `OffloadConfig` field can be set via an environment variable named `FT_` + the uppercase field name. For example, `discovery_iters` → `FT_DISCOVERY_ITERS`. Values follow standard Python literals (`true`/`false`, integers, floats, comma-separated lists).

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FT_ENABLED` | Enable offloading | `false` |
| `FT_MAX_GPU_MEM_FRACTION` | Fraction of GPU memory to use for model weights (e.g. `0.7` = 70%); `None` disables the memory constraint | `0.9` |
| `FT_INCLUDE_PATTERNS` | Comma-separated include patterns to offload | see below |
| `FT_EXCLUDE_PATTERNS` | Comma-separated patterns to keep GPU-resident | see below |
| `FT_ENABLE_DIAGNOSTICS` | Log Layer Duration Statistics and block assignment table after profiling | `false` |

Unlike standalone `OffloadConfig`, which defaults to latency mode, the vLLM
worker keeps an integration-specific `0.9` default when
`FT_MAX_GPU_MEM_FRACTION` is omitted. Set it to `none` to request latency mode
explicitly. The worker does not derive the FlexTensor weight budget from vLLM's
`gpu_memory_utilization`, because that budget must also leave room for KV cache
and runtime allocations.

The worker defaults to vLLM-oriented patterns when `FT_INCLUDE_PATTERNS` /
`FT_EXCLUDE_PATTERNS` are not customized: decoder-layer class includes, common
embedding/norm/head paths, and excludes for known MoE sidecars and tiny
router/gating tensors that should stay GPU-resident. The exact defaults live in
`VLLM_DEFAULT_INCLUDE_PATTERNS` and `VLLM_DEFAULT_EXCLUDE_PATTERNS` in
`src/flextensor/contrib/vllm/worker.py`.

For models with a different module layout, set `FT_INCLUDE_PATTERNS` explicitly:

| Family | `FT_INCLUDE_PATTERNS` |
|---|---|
| GPT-2 / GPT-J / Falcon v1 / BLOOM / Qwen1 | `class:*Block,transformer.wte,transformer.ln_f,lm_head,logits_processor` |
| GPT-NeoX / Pythia | `class:*Layer,gpt_neox.embed_in,gpt_neox.final_layer_norm,embed_out,logits_processor` |
| MPT / DBRX | `class:*Block,transformer.wte,transformer.norm_f,lm_head,logits_processor` |
| Mamba / Mamba2 | `class:*DecoderLayer,backbone.embeddings,backbone.norm_f,lm_head,logits_processor` |
| Phi-1 / Phi-2 / Jamba / Bamba | `class:*DecoderLayer,model.embed_tokens,model.final_layernorm,lm_head,logits_processor` |
| InternLM2 | `class:*DecoderLayer,model.tok_embeddings,model.norm,output,logits_processor` |
| OPT | `class:*DecoderLayer,model.decoder.embed_tokens,model.decoder.final_layer_norm,lm_head,logits_processor` |
| ChatGLM v1–3 | `class:*Layer,transformer.embedding,transformer.encoder.final_layernorm,lm_head,logits_processor` |

For the full list of configuration options (config files, discovery/profiling tuning, transfer modes), see the [Configuration Reference](https://github.com/ai-dynamo/flextensor/blob/main/docs/api/configuration.md).

## How It Works

The `FlexTensorOffloadWorker` extends vLLM's GPU worker to:

1. Load configuration from environment variables or config file
2. Apply FlexTensor offloading to model layers after loading
3. Automatically manage tensor movement between CPU and GPU

When `FT_ENABLED=1` is set, the worker applies offloading using the default decoder-only transformer
include patterns (or those set via `FT_INCLUDE_PATTERNS`). Otherwise it behaves like the
standard vLLM worker. Look for these log messages to confirm offloading is active:

```
FlexTensor offloading enabled with config: ...
FlexTensor offloading applied (GPU usage: X.XX GiB)
```

## Troubleshooting

For troubleshooting tips (out of memory, performance tuning, debugging), see the [Troubleshooting Guide](https://github.com/ai-dynamo/flextensor/blob/main/docs/how-to/troubleshooting.md).
