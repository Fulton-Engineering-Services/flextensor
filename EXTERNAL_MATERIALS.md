<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# External Materials

For third-party dependency attributions, see [ATTRIBUTIONS.md](ATTRIBUTIONS.md).

This software includes configurations (integration tests, development
containers, and examples) that automatically retrieve external materials listed
below. Those retrieved materials are not distributed with this software and are
governed solely by their own terms, conditions, and licenses.

## NVIDIA Materials

| Material | Source |
| --- | --- |
| NVIDIA PyTorch Container | <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch> |
| NVIDIA vLLM Container | <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm> |

## Third-Party Container Images

| Material | Source |
| --- | --- |
| vllm/vllm-openai | <https://hub.docker.com/r/vllm/vllm-openai> |

## Hugging Face Models and Datasets

The following models are downloaded from <https://huggingface.co> during
integration tests and examples.

| Model ID | License | Used In |
| --- | --- | --- |
| TinyLlama/TinyLlama_v1.1 | Apache 2.0 | Integration tests |
| Qwen/Qwen3-0.6B | Apache 2.0 | Integration tests |
| Qwen/Qwen3-32B | Apache 2.0 | Integration tests |
| Qwen/Qwen2.5-0.5B-Instruct | Apache 2.0 | Integration tests |
| Qwen/Qwen2.5-7B-Instruct | Apache 2.0 | Integration tests |
| Qwen/Qwen2.5-32B-Instruct | Apache 2.0 | Integration tests |
| Qwen/Qwen3.6-35B-A3B | Apache 2.0 | Integration tests |
| Qwen/Qwen3.6-35B-A3B-FP8 | Apache 2.0 | Integration tests |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 | NVIDIA Nemotron Open Model License | Integration tests |
| nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | NVIDIA Nemotron Open Model License | Integration tests |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 | NVIDIA Nemotron Open Model License | Integration tests |
| Qwen/Qwen2.5-72B-Instruct | Qwen License | Examples |
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | Apache 2.0 | Examples |

## Notice and Disclaimer

This software automatically retrieves, accesses, or interacts with external
materials. Those retrieved materials are not distributed with this software and
are governed solely by separate terms, conditions, and licenses. You are solely
responsible for finding, reviewing, and complying with all applicable terms,
conditions, and licenses, and for verifying the security, integrity, and
suitability of any retrieved materials for your specific use case.

This software is provided "AS IS", without warranty of any kind. The author
makes no representations or warranties regarding any retrieved materials, and
assumes no liability for any losses, damages, liabilities, or legal consequences
from your use or inability to use this software or any retrieved materials. Use
this software and the retrieved materials at your own risk.
