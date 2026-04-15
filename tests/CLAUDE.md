<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Tests

## Integration Tests

Structure: `integration/L{0,1,2}_<name>/` — each directory contains `.dockerimages`, `requirements.txt`, `test.sh`.

**GPU memory markers**: `@pytest.mark.gpu_mem_{24g,40g,48g,80g,96g}` — CI auto-generates jobs per tier; unmarked → highest-tier runner.

### test.sh Pattern

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi >/dev/null 2>&1 || { echo "ERROR: No GPU."; exit 1; }
[ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ] && pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")
timeout "${TIMEOUT:-1800}" python3 -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"
```
