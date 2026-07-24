<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Diffusers + FlexTensor Examples

These examples show how to run large [Diffusers](https://github.com/huggingface/diffusers) models on a single GPU with limited memory using FlexTensor weight offloading.

| Example | Description |
|---------|-------------|
| [`quickstart/`](quickstart/) | Minimal single-script example showing how to integrate FlexTensor with a Diffusers pipeline. Profiles inline on every launch. |
| [`compiled-offload/`](compiled-offload/) | Same weight streaming as quickstart, plus `torch.compile` per offloaded unit — via `compile_fn` or external compile after INFERENCE. |
| [`profile-reuse/`](profile-reuse/) | Two-step workflow: profile once with `run_profile.py`, then generate videos with `run_infer.py` without re-profiling. |

Start with **quickstart** to see how FlexTensor integrates with Diffusers. Use **compiled-offload** when you also want compiled kernels under offload. Move to **profile-reuse** when you want to avoid the profiling overhead on repeated runs.
