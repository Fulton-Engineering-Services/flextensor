# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron 3 Nano vLLM functional smoke tests."""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.integration._vllm_server import (
    MemoryProfilingMetrics,
    VllmBenchmarkConfig,
    VllmOffloadSmokeCase,
    run_vllm_server_test,
)
from tests.integration._vllm_utils import (
    assert_moe_backend_selection,
    assert_no_triton_cpu_pointer_failure,
    assert_rejected_moe_backend_reason,
    sanitize_test_name,
)

NEMOTRON_3_NANO_FP8_SMOKE_CASE = VllmOffloadSmokeCase(
    model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    output_dir_name="nemotron3_nano_fp8",
    cli_args=(
        "--enforce-eager",
        "--trust-remote-code",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--kv-cache-dtype",
        "fp8",
    ),
    extra_env_vars=(("FT_MAX_GPU_MEM_FRACTION", "0.75"),),
).with_flextensor_offload()
NEMOTRON_3_NANO_FP8_REQUIRED_MOE_BACKENDS = (
    "TRITON",
    "FLASHINFER_TRTLLM",
    "FLASHINFER_CUTLASS",
)

NEMOTRON_3_NANO_NVFP4_SMOKE_CASE = VllmOffloadSmokeCase(
    model_name="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    output_dir_name="nemotron3_nano_nvfp4",
    cli_args=(
        "--enforce-eager",
        "--trust-remote-code",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
    ),
    extra_env_vars=(("FT_MAX_GPU_MEM_FRACTION", "0.75"),),
).with_flextensor_offload()
NEMOTRON_3_NANO_NVFP4_REQUIRED_MOE_BACKENDS = (
    "FLASHINFER_TRTLLM",
    "FLASHINFER_CUTLASS",
)
NEMOTRON_3_NANO_BENCHMARK_CONFIG = VllmBenchmarkConfig(
    request_count=2,
    input_tokens=64,
    output_tokens=4,
    max_concurrency=1,
    timeout=600,
)


@pytest.fixture
def test_output_dir(request) -> Path:
    """Fixture that provides a unique output directory for each test case."""
    test_dir = Path(__file__).parent
    sanitized_name = sanitize_test_name(request.node.name)
    output_dir = test_dir / "test_results" / sanitized_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def assert_successful_offload_request(
    case: VllmOffloadSmokeCase,
    offload_memory: MemoryProfilingMetrics,
    offload_metrics: dict[str, Any],
) -> None:
    """Assert that vLLM served the smoke request and reported usable KV cache when available."""
    assert offload_metrics["usage"].get("completion_tokens", 0) > 0, f"{case.model_name} returned no generated tokens"
    if offload_memory.available_kv_cache_memory_gib is not None:
        assert offload_memory.available_kv_cache_memory_gib > 0, f"{case.model_name} reported no available KV cache"


def write_offload_metrics(
    test_output_dir: Path,
    case: VllmOffloadSmokeCase,
    offload_metrics: dict[str, Any],
) -> Path:
    """Write the per-case offload metrics artifact."""
    metrics_file = test_output_dir / f"{case.output_dir_name}_metrics.json"
    with metrics_file.open("w") as f:
        json.dump(
            {
                "model_name": case.model_name,
                "offload": offload_metrics,
            },
            f,
            indent=2,
        )
    return metrics_file


class TestNemotron3Nano:
    """Nemotron 3 Nano functional coverage for FlexTensor's vLLM worker."""

    @pytest.mark.gpu_vram_24g
    @pytest.mark.gpu_sm_89
    def test_nemotron3_nano_fp8_moe_serves_with_offloading_on_l4(
        self,
        test_output_dir: Path,
    ) -> None:
        """Smoke-test FP8 MoE offloading on the 24GB L4 target."""
        case = NEMOTRON_3_NANO_FP8_SMOKE_CASE

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, offload_logs = run_vllm_server_test(
            case,
            output_dir=output_dir,
            # One token is enough to validate the offload path on a small VRAM target.
            chat_max_tokens=1,
            chat_request_timeout=180,
            benchmark_config=NEMOTRON_3_NANO_BENCHMARK_CONFIG,
        )

        assert_successful_offload_request(case, offload_memory, offload_metrics)
        assert_no_triton_cpu_pointer_failure(offload_logs)
        backend_evidence = offload_metrics["backend_evidence"]
        assert_moe_backend_selection(
            backend_evidence,
            expected_backend="TRITON",
            expected_family="Fp8",
            expected_potential_backends=NEMOTRON_3_NANO_FP8_REQUIRED_MOE_BACKENDS,
        )
        assert_rejected_moe_backend_reason(
            backend_evidence,
            backend="FLASHINFER_TRTLLM",
            reason_contains="kernel does not support current device cuda",
        )
        assert_rejected_moe_backend_reason(
            backend_evidence,
            backend="FLASHINFER_CUTLASS",
            reason_contains="kernel does not support current device cuda",
        )
        metrics_file = write_offload_metrics(test_output_dir, case, offload_metrics)
        print(f"\nNemotron 3 Nano FP8 metrics saved to: {metrics_file}")

    @pytest.mark.gpu_vram_96g
    @pytest.mark.gpu_sm_min_120
    def test_nemotron3_nano_nvfp4_moe_serves_with_offloading(
        self,
        test_output_dir: Path,
    ) -> None:
        """Smoke-test Nemotron 3 Nano NVFP4 MoE serving through the vLLM worker.

        This is offload-only by design: the functional signal is that the
        NVFP4/hybrid-MoE checkpoint loads through FlexTensor's CPU-first loader,
        completes FT warmup/profiling, and serves a chat request. Baseline memory
        comparison remains covered by the smaller 40GB Qwen test.
        """
        case = NEMOTRON_3_NANO_NVFP4_SMOKE_CASE

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, _ = run_vllm_server_test(
            case,
            output_dir=output_dir,
            benchmark_config=NEMOTRON_3_NANO_BENCHMARK_CONFIG,
        )

        assert_successful_offload_request(case, offload_memory, offload_metrics)
        backend_evidence = offload_metrics["backend_evidence"]
        assert_moe_backend_selection(
            backend_evidence,
            expected_backend="FLASHINFER_CUTLASS",
            expected_family="NvFp4",
            expected_potential_backends=NEMOTRON_3_NANO_NVFP4_REQUIRED_MOE_BACKENDS,
        )
        assert_rejected_moe_backend_reason(
            backend_evidence,
            backend="FLASHINFER_TRTLLM",
            reason_contains="kernel does not support current device cuda",
        )
        metrics_file = write_offload_metrics(test_output_dir, case, offload_metrics)
        print(f"\nNemotron 3 Nano NVFP4 metrics saved to: {metrics_file}")
