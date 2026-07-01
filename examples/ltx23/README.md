<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# LTX 2.3 + FlexTensor Examples

These examples show how to use FlexTensor weight streaming with LTX 2.3 video generation pipelines that would otherwise be difficult to serve on memory-constrained single-GPU systems.

| Example | Description |
|---------|-------------|
| [`lipdub/`](lipdub/) | Serves the LTX 2.3 LipDub IC-LoRA with one FlexTensor manager per diffusion stage and optional Gemma text-encoder offload. |
| [`outpaint/`](outpaint/) | Serves the LTX 2.3 Outpaint IC-LoRA, including letterbox preparation and single-GPU or NGINX-backed multi-GPU helper scripts. |

Start with **lipdub** for A100 40 GB viability experiments. Use **outpaint** for canvas-extension experiments and warmed-worker serving helpers.

## External Artifacts and Licenses

The examples may download LTX and Gemma artifacts from Hugging Face. Those
artifacts are not distributed with FlexTensor and are governed by their upstream
terms. In particular, LTX artifacts are governed by the
[LTX-2 Community License Agreement](https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE),
which includes commercial-use and acceptable-use restrictions. Review and
comply with the applicable upstream terms before downloading or running the
default artifacts.
