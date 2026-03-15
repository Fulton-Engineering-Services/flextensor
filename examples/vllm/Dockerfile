# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG FROM_IMAGE=nvcr.io/nvidia/vllm:25.12-py3
FROM ${FROM_IMAGE}

ENV UV_SYSTEM_PYTHON=true \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=240

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

# Use FlexTensor worker with native vllm serve
ENTRYPOINT ["vllm", "serve", "--enforce-eager", "--worker-cls", "flextensor.contrib.vllm.worker.FlexTensorOffloadWorker"]
