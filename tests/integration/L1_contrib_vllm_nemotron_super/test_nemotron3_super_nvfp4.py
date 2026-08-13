# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron 3 Super NVFP4 vLLM instrumentation smoke tests."""

import json
import shutil
from pathlib import Path

import pytest

from tests.integration._vllm_server import (
    VllmCorrectnessCheck,
    VllmOffloadSmokeCase,
    run_vllm_server_test,
    with_hf_reasoning_parser,
)
from tests.integration._vllm_utils import (
    assert_moe_backend_selection,
    assert_rejected_moe_backend_reason,
    load_latest_instrumentation_dump,
    sanitize_test_name,
)

NEMOTRON_3_SUPER_NVFP4_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

NEMOTRON_3_SUPER_NVFP4_SMOKE_CASE = VllmOffloadSmokeCase(
    model_name=NEMOTRON_3_SUPER_NVFP4_MODEL,
    output_dir_name="nemotron3_super_nvfp4",
    cli_args=(
        "--trust-remote-code",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
    ),
    extra_env_vars=(
        ("FT_MAX_GPU_MEM_FRACTION", "0.70"),
        ("FT_PROFILING_ITERS", "2"),
    ),
).with_flextensor_offload()
NEMOTRON_3_SUPER_NVFP4_REQUIRED_MOE_BACKENDS = (
    "FLASHINFER_TRTLLM",
    "FLASHINFER_CUTLASS",
)
NEMOTRON_3_SUPER_NON_THINKING_CHECK = VllmCorrectnessCheck(
    max_tokens=20,
    timeout=180,
    temperature=1.0,
    chat_template_kwargs={"enable_thinking": False},
)
NEMOTRON_3_SUPER_REASONING_PARSER = "super_v3"
NEMOTRON_3_SUPER_REASONING_PARSER_FILENAME = "super_v3_reasoning_parser.py"


@pytest.fixture
def test_output_dir(request) -> Path:
    """Fixture that provides a unique output directory for each test case."""
    test_dir = Path(__file__).parent
    sanitized_name = sanitize_test_name(request.node.name)
    output_dir = test_dir / "test_results" / sanitized_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestNemotron3SuperNvfp4Instrumentation:
    """Nemotron 3 Super NVFP4 instrumentation coverage for FlexTensor's vLLM worker."""

    # Checkpoint shards are ~74.8 GiB, below the 96GB runner; this test covers
    # Super instrumentation rather than smaller-than-checkpoint VRAM pressure.
    @pytest.mark.gpu_vram_96g
    @pytest.mark.gpu_sm_min_120
    def test_nemotron3_super_nvfp4_moe_serves_and_dumps_instrumentation(
        self,
        test_output_dir: Path,
    ) -> None:
        """Smoke-test Super serving and verify FlexTensor emits instrumentation."""
        case = with_hf_reasoning_parser(
            NEMOTRON_3_SUPER_NVFP4_SMOKE_CASE,
            filename=NEMOTRON_3_SUPER_REASONING_PARSER_FILENAME,
            parser_name=NEMOTRON_3_SUPER_REASONING_PARSER,
        )

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(output_dir / "instrumentation", ignore_errors=True)

        offload_memory, offload_metrics, _ = run_vllm_server_test(
            case,
            output_dir=output_dir,
            correctness_check=NEMOTRON_3_SUPER_NON_THINKING_CHECK,
        )

        assert offload_metrics["usage"].get("completion_tokens", 0) > 0, (
            f"{case.model_name} returned no generated tokens"
        )
        backend_evidence = offload_metrics["backend_evidence"]
        assert_moe_backend_selection(
            backend_evidence,
            expected_backend="FLASHINFER_CUTLASS",
            expected_family="NvFp4",
            expected_potential_backends=NEMOTRON_3_SUPER_NVFP4_REQUIRED_MOE_BACKENDS,
        )
        assert_rejected_moe_backend_reason(
            backend_evidence,
            backend="FLASHINFER_TRTLLM",
            reason_contains="kernel does not support current device cuda",
        )
        if offload_memory.available_kv_cache_memory_gib is not None:
            assert offload_memory.available_kv_cache_memory_gib > 0, f"{case.model_name} reported no available KV cache"

        instrumentation_path, instrumentation_payload = load_latest_instrumentation_dump(output_dir)
        components = instrumentation_payload.get("components")
        assert isinstance(components, list) and components, "Instrumentation dump is missing components"
        component_class_names = {component.get("class_name") for component in components}
        assert "TensorManager" in component_class_names, "Instrumentation dump is missing TensorManager"
        assert "host_memory" in instrumentation_payload, "Instrumentation dump is missing host memory"
        assert isinstance(instrumentation_payload.get("memory_transfer_stats"), dict), (
            "Instrumentation dump is missing FlexTensor memory transfer stats"
        )

        metrics_file = test_output_dir / f"{case.output_dir_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(
                {
                    "model_name": case.model_name,
                    "offload": offload_metrics,
                    "instrumentation_dump": str(instrumentation_path),
                },
                f,
                indent=2,
            )

        print(f"\nNemotron 3 Super NVFP4 metrics saved to: {metrics_file}")
