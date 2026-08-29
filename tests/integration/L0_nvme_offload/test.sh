#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run NVMe disk offload integration tests.
#
# Tests cover:
#   1. PosixBackend and CuFileBackend GPU roundtrips (direct, no FlexTensor)
#   2. Full OffloadManager lifecycle with NVMe weight eviction
#   3. CUDA graph capture + replay with NVMe-backed offload
#   4. Profile save/restore roundtrip with NVMe backing
#
# Environment variables:
#   FT_NVME_TEST_PATH  Base directory for NVMe test files (default: pytest tmp_path)
#   CUDA_VISIBLE_DEVICES  GPU index to use (default: 0)

set -xeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Install test-specific requirements if not in CI.
if [ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
  pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
fi

PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")

timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"