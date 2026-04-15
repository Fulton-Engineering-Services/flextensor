#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: NVIDIA GPU not detected. These tests require a GPU."
  exit 1
fi

# Install test-specific requirements if not in CI (CI installs with flextensor for proper dependency resolution)
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
  pip list
fi

TIMEOUT=${TIMEOUT:-3600}
echo "Running vLLM example integration tests with timeout ${TIMEOUT}s..."

# Build pytest command with optional marker filter
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
if [ -n "${PYTEST_MARKER:-}" ]; then
  echo "Filtering tests with marker: $PYTEST_MARKER"
  PYTEST_ARGS+=(-m "$PYTEST_MARKER")
fi

timeout "$TIMEOUT" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"

echo "Integration tests completed successfully!"
