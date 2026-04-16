#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL=${1:?Usage: serve.sh MODEL_NAME}

# exec replaces the shell so SIGTERM reaches vllm directly (needed for clean shutdown)
set -x
exec env FT_ENABLED=1 vllm serve "$MODEL" \
  --enforce-eager \
  --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker
