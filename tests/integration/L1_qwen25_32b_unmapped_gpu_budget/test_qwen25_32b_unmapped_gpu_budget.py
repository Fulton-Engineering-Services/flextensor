# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-32B unmapped GPU tensor budget coverage for vLLM.

This high-residency run exercises unmapped tensor finalization on A100-40GB CI.
Fixed code must avoid raw CUDA OOM and may either reach vLLM's controlled
KV-cache capacity error or fail earlier with a clear FlexTensor budget error.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import requests

from tests.integration._vllm_server import sanitize_test_name
from tests.integration.L0_contrib_vllm.vllm_utils import (
    make_chat_request,
    parse_memory_profiling_logs,
    start_vllm_server,
    stop_vllm_server,
    wait_for_server,
)

pytestmark = [pytest.mark.gpu_vram_40g, pytest.mark.gpu_sm_80]

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
INCLUDE_PATTERNS = "model.embed_tokens,model.layers.*,model.norm,lm_head,logits_processor"
RAW_CUDA_OOM_MARKERS = (
    "torch.OutOfMemoryError",
    "CUDA out of memory",
)
EXPECTED_MEMORY_PRESSURE_FAILURE_MARKERS = (
    "No available memory for the cache blocks",
    "larger than the available KV cache memory",
)
STRATEGY_BUDGET_FAILURE_MARKER = "Insufficient strategy GPU budget after reserving"
# Emitted by src/flextensor/contrib/vllm/worker.py after FlexTensor finishes
# warmup/profile/offload setup and before vLLM proceeds into KV-cache sizing.
OFFLOAD_APPLIED_MARKER = "FlexTensor offloading applied"


@pytest.fixture
def test_output_dir(request: pytest.FixtureRequest) -> Path:
    test_dir = Path(__file__).parent
    output_dir = test_dir / "test_results" / sanitize_test_name(request.node.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def test_qwen25_32b_budgets_unmapped_gpu_tensors_without_raw_cuda_oom(test_output_dir: Path) -> None:
    process = None
    log_lines: list[str] = []
    summary: dict[str, object] = {
        "model": MODEL_NAME,
        "result": "unknown",
    }

    try:
        process, log_lines = start_vllm_server(
            MODEL_NAME,
            offload_enabled=True,
            additional_cli_args=["--enforce-eager", "--max-num-seqs", "1", "--max-model-len", "128"],
            additional_env_vars={
                "VLLM_NO_USAGE_STATS": "1",
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "FT_ENABLE_DIAGNOSTICS": "1",
                "FT_MAX_GPU_MEM_FRACTION": "0.95",
                "FT_INCLUDE_PATTERNS": INCLUDE_PATTERNS,
                "FT_DEBUG_LOG_PATH": str(test_output_dir / "debug.log"),
            },
        )

        ready = wait_for_server(timeout=2700, process=process)
        if not ready and process.poll() is not None:
            time.sleep(1)

        log_text = "\n".join(log_lines)
        memory_metrics = parse_memory_profiling_logs(log_lines)

        assert not any(marker in log_text for marker in RAW_CUDA_OOM_MARKERS), (
            "Qwen2.5-32B hit raw CUDA OOM in unmapped tensor finalization"
        )

        if ready:
            chat_response = make_chat_request(
                messages=[{"role": "user", "content": "Say OK."}],
                model=MODEL_NAME,
                max_tokens=1,
                timeout=120,
            )
            assert chat_response.get("choices"), "server became ready but returned no choices"
            models_response = requests.get("http://localhost:8000/v1/models", timeout=10)
            models_response.raise_for_status()
            summary["result"] = "server_started"
            return

        assert process is not None
        accepted_clear_budget_failure = STRATEGY_BUDGET_FAILURE_MARKER in log_text
        accepted_kv_failure_after_transition = OFFLOAD_APPLIED_MARKER in log_text and any(
            marker in log_text for marker in EXPECTED_MEMORY_PRESSURE_FAILURE_MARKERS
        )
        summary.update({
            "result": "process_exited_before_ready",
            "exit_code": process.poll(),
            "memory_profiling": memory_metrics.to_dict(),
            "accepted_clear_budget_failure": accepted_clear_budget_failure,
            "accepted_kv_failure_after_transition": accepted_kv_failure_after_transition,
        })

        assert accepted_clear_budget_failure or accepted_kv_failure_after_transition, (
            "server did not start, but logs did not show a clear budget failure or post-transition KV-cache failure"
        )
    finally:
        if process is not None:
            stop_vllm_server(process)
            time.sleep(1)
        (test_output_dir / "server.log").write_text("\n".join(log_lines))
        (test_output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
