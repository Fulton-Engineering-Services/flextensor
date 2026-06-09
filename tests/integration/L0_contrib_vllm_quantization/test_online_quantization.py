# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online quantization smoke coverage for FlexTensor's vLLM worker."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from beartype import beartype

from tests.integration._vllm_server import (
    VllmOffloadSmokeCase,
    run_vllm_server_test,
)
from tests.integration._vllm_utils import (
    sanitize_test_name,
)

BASE_CLI_ARGS = (
    "--enforce-eager",
    "--tensor-parallel-size",
    "1",
    "--gpu-memory-utilization",
    "0.95",
    "--max-model-len",
    "1024",
    "--max-num-seqs",
    "1",
)

ONLINE_QUANTIZATION_ENV_VARS = (
    ("FT_ENABLE_DIAGNOSTICS", "1"),
    ("FT_MAX_GPU_MEM_FRACTION", "0.90"),
)


@dataclass(frozen=True)
class OnlineQuantizationSmokeCase:
    """vLLM online quantization smoke-test configuration."""

    model_name: str
    output_dir_name: str
    quantization: str
    residency_note: str

    def to_vllm_case(self) -> VllmOffloadSmokeCase:
        return VllmOffloadSmokeCase(
            model_name=self.model_name,
            output_dir_name=self.output_dir_name,
            cli_args=(*BASE_CLI_ARGS, "--quantization", self.quantization),
            extra_env_vars=ONLINE_QUANTIZATION_ENV_VARS,
        ).with_flextensor_offload()


ONLINE_QUANTIZATION_CASES = (
    pytest.param(
        OnlineQuantizationSmokeCase(
            model_name="Qwen/Qwen3-32B",
            output_dir_name="qwen3_32b_fp8",
            quantization="fp8",
            residency_note="BF16 checkpoint is larger than the 40GB A100 CI target",
        ),
        marks=(pytest.mark.gpu_vram_40g, pytest.mark.gpu_sm_80),
        id="qwen3-32b-fp8-sm80-40g",
    ),
    pytest.param(
        OnlineQuantizationSmokeCase(
            model_name="Qwen/Qwen3-32B",
            output_dir_name="qwen3_32b_fp8_per_tensor",
            quantization="fp8_per_tensor",
            residency_note="BF16 checkpoint is larger than the 24GB L4 CI target",
        ),
        marks=(pytest.mark.gpu_vram_24g, pytest.mark.gpu_sm_89),
        id="qwen3-32b-fp8-per-tensor-sm89-24g",
    ),
    pytest.param(
        OnlineQuantizationSmokeCase(
            model_name="Qwen/Qwen3-32B",
            output_dir_name="qwen3_32b_fp8_per_block",
            quantization="fp8_per_block",
            residency_note="BF16 checkpoint is larger than the 24GB L4 CI target",
        ),
        marks=(pytest.mark.gpu_vram_24g, pytest.mark.gpu_sm_89),
        id="qwen3-32b-fp8-per-block-sm89-24g",
    ),
    pytest.param(
        OnlineQuantizationSmokeCase(
            model_name="Qwen/Qwen3-32B",
            output_dir_name="qwen3_32b_mxfp8",
            quantization="mxfp8",
            residency_note="Smaller BF16 checkpoint exercises MXFP8 on 96GB Blackwell without host-RAM pressure",
        ),
        marks=(pytest.mark.gpu_vram_96g, pytest.mark.gpu_sm_120),
        id="qwen3-32b-mxfp8-sm120-96g",
    ),
)


def require_vllm_quantization(quantization: str) -> None:
    """Skip when the installed vLLM package does not accept ``quantization``."""
    try:
        version = importlib.metadata.version("vllm")
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    except (ImportError, importlib.metadata.PackageNotFoundError):
        pytest.skip(f"online quantization {quantization!r} requires vLLM")

    if quantization not in QUANTIZATION_METHODS:
        pytest.skip(f"online quantization {quantization!r} is not supported by vLLM {version}")


def run_online_quantization_smoke(case: OnlineQuantizationSmokeCase, test_output_dir: Path) -> None:
    """Run a vLLM online quantization smoke test for a BF16 checkpoint."""
    output_dir = test_output_dir / case.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    vllm_case = case.to_vllm_case()

    offload_memory, offload_metrics, log_lines = run_vllm_server_test(
        vllm_case,
        output_dir=output_dir,
        chat_request_timeout=180,
    )

    assert offload_metrics["usage"].get("completion_tokens", 0) > 0, f"{case.model_name} returned no generated tokens"
    assert any("FlexTensorModelLoader: Processed" in line for line in log_lines), (
        "FlexTensor loader did not report layer-by-layer GPU processing"
    )
    assert offload_memory.available_kv_cache_memory_gib is not None, (
        "KV cache memory metric unavailable after online quantization offload processing"
    )
    assert offload_memory.available_kv_cache_memory_gib > 0

    metrics_file = output_dir / "metrics.json"
    with metrics_file.open("w") as f:
        json.dump(
            {
                "model_name": case.model_name,
                "quantization": case.quantization,
                "residency_note": case.residency_note,
                "offload": offload_metrics,
            },
            f,
            indent=2,
        )

    print(f"\n{case.model_name} {case.quantization} metrics saved to: {metrics_file}")


@pytest.fixture
@beartype
def test_output_dir(request: pytest.FixtureRequest) -> Path:
    """Return a unique output directory for the current test case."""
    test_dir = Path(__file__).parent
    output_dir = test_dir / "test_results" / sanitize_test_name(request.node.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
@beartype
def device_gpu() -> torch.device:
    """Return the CUDA device used by this integration test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


class TestVllmOnlineQuantization:
    """FlexTensor online quantization coverage for vLLM."""

    @pytest.mark.parametrize("case", ONLINE_QUANTIZATION_CASES)
    @beartype
    def test_bf16_online_quantization_serves_with_flextensor_offload(
        self,
        case: OnlineQuantizationSmokeCase,
        device_gpu: torch.device,
        test_output_dir: Path,
    ) -> None:
        """Smoke-test online quantization with a BF16 checkpoint larger than CI VRAM."""
        require_vllm_quantization(case.quantization)
        assert device_gpu.type == "cuda"

        run_online_quantization_smoke(case, test_output_dir)
