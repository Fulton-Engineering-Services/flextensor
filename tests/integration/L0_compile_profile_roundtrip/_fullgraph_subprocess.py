# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess driver for ``test_fullgraph_fails_with_offloaded_model``.

The companion test invokes this script and asserts a non-zero exit. Running
the compile attempt in a child process isolates the parent pytest worker
from torch 2.10's ``reset_user_object_tracking`` segfault — see
``_has_torch_2_10_weakref_bug`` in the parent test module for the canonical
description of the bug. Any failure mode — Python exception or fatal
signal — counts as the expected outcome.

Exit codes
----------
0  ``torch.compile(fullgraph=True)`` returned without raising. ``UNEXPECTED_SUCCESS``
   is printed to stderr; the parent's stderr-sentinel assertion catches this.
1  A Python exception escaped — either during lifecycle setup (before
   ``REACHED_COMPILE``) or from compile itself (after, expected:
   ``torch._dynamo.exc.Unsupported``). The parent uses the ``REACHED_COMPILE``
   sentinel to distinguish the two.
*  Any other non-zero exit (e.g. SIGSEGV producing 139, or ``SystemExit(N)``
   propagating) — also accepted as "fullgraph did not silently succeed".
"""

from __future__ import annotations

import sys
from pathlib import Path

# Re-add the project root: when launched as ``python <script>``, sys.path[0] is
# this file's directory, not the project root that pytest's rootdir discovery
# normally provides — so ``tests.integration._compile_helpers`` would not
# resolve.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from flextensor import get_offload_manager  # noqa: E402
from tests.integration._compile_helpers import (  # noqa: E402
    make_offload_config,
    make_simple_model,
    run_offload_lifecycle,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SEED = 42
NUM_LAYERS = 20
DIM = 256
INTER_DIM = 512
NUM_EXPERTS = 2
BATCH = 1
SEQ_LEN = 64
MODULE_PATTERNS = ["input_projection", "layers.*", "output_projection"]


def main() -> int:
    config = make_offload_config(
        discovery_iters=1,
        profiling_iters=3,
        feedback_iters=2,
        module_patterns=MODULE_PATTERNS,
    )
    model = make_simple_model(
        num_layers=NUM_LAYERS,
        dim=DIM,
        inter_dim=INTER_DIM,
        num_experts=NUM_EXPERTS,
        dtype=DTYPE,
        device=torch.device("cpu"),
        seed=SEED,
    )
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=DEVICE, dtype=DTYPE)

    om = get_offload_manager("subprocess_fullgraph")
    proxy = om.offload(model, config)
    try:
        run_offload_lifecycle(proxy, x, discovery_iters=1, profiling_iters=3, feedback_iters=2)
        torch._dynamo.reset()
        # Sentinel: the parent asserts this string in stderr to confirm the
        # subprocess made it through lifecycle setup before any failure.
        print("REACHED_COMPILE", file=sys.stderr, flush=True)
        compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            compiled(x)
        # Sentinel must be flushed *before* ``finally: om.release()`` runs —
        # if release() crashes after a silent compile success, this print is
        # the only signal the parent has that fullgraph=True did not raise.
        print("UNEXPECTED_SUCCESS: fullgraph=True returned without raising", file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001 - any compile-time failure is expected
        # KeyboardInterrupt / SystemExit / other BaseExceptions intentionally
        # propagate so the parent sees them as a non-zero exit rather than
        # masking them as "compile failed".
        print(f"COMPILE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        om.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())
