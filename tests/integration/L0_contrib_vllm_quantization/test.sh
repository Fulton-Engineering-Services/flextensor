#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../gpu_diagnostics.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

require_nvidia_gpu

# Install test-specific requirements if not in CI (CI installs with flextensor for proper dependency resolution)
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
  pip list
fi

TIMEOUT=${TIMEOUT:-7200}
echo "Running vLLM quantization integration tests with timeout ${TIMEOUT}s..."

PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
if [ -n "${PYTEST_MARKER:-}" ]; then
  echo "Filtering tests with marker: $PYTEST_MARKER"
  PYTEST_ARGS+=(-m "$PYTEST_MARKER")
fi

timeout "$TIMEOUT" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

echo "vLLM quantization integration tests completed successfully!"
