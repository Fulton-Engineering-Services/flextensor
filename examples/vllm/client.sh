#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

MODEL=${1:?Usage: client.sh MODEL_NAME}
PORT=${2:-8000}

set -x
curl -sf --connect-timeout 5 --max-time 30 "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
        \"model\": \"${MODEL}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"The capital of France is\"}],
        \"max_tokens\": 10
    }"

echo -e "\n\n=== Models ==="
curl -sf --connect-timeout 5 --max-time 10 "http://localhost:${PORT}/v1/models"
