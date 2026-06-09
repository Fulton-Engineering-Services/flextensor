# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen 3.6 BF16 MoE vLLM functional smoke tests."""

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.integration._vllm_server import (
    VllmCorrectnessCheck,
    VllmOffloadSmokeCase,
    run_vllm_server_test,
)
from tests.integration._vllm_utils import (
    assert_moe_backend_family,
    assert_moe_backend_selection,
    assert_moe_backend_selection_in,
    assert_no_triton_cpu_pointer_failure,
    assert_rejected_moe_backend_reason,
    sanitize_test_name,
)

QWEN3_6_35B_A3B_BF16_MODEL = "Qwen/Qwen3.6-35B-A3B"
QWEN3_6_35B_A3B_FP8_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
# HF model.safetensors.index.json metadata.total_size: ~67.0 GiB.
QWEN3_6_35B_A3B_BF16_WEIGHT_BYTES = 71_903_645_408


@dataclass(frozen=True)
class Qwen36QuantizedMoeCase:
    """Qwen 3.6 MoE quantized smoke-test launch configuration."""

    model_name: str
    output_dir_name: str
    quantization: str | None
    expected_moe_family: str
    residency_note: str

    def to_vllm_case(self) -> VllmOffloadSmokeCase:
        cli_args = list(QWEN3_6_35B_A3B_BF16_SMOKE_CASE.cli_args)
        if self.quantization is not None:
            cli_args.extend(("--quantization", self.quantization))
        return VllmOffloadSmokeCase(
            model_name=self.model_name,
            output_dir_name=self.output_dir_name,
            cli_args=tuple(cli_args),
            extra_env_vars=(
                *QWEN3_6_35B_A3B_BF16_SMOKE_CASE.extra_env_vars,
                ("VLLM_LOGGING_LEVEL", "INFO"),
            ),
        )


QWEN3_6_35B_A3B_BF16_SMOKE_CASE = VllmOffloadSmokeCase(
    model_name=QWEN3_6_35B_A3B_BF16_MODEL,
    output_dir_name="qwen3_6_35b_a3b_bf16",
    cli_args=(
        "--enforce-eager",
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "512",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--moe-backend",
        "auto",
    ),
    extra_env_vars=(
        ("FT_MAX_GPU_MEM_FRACTION", "0.70"),
        ("FT_PROFILING_ITERS", "2"),
    ),
).with_flextensor_offload()
QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK = VllmCorrectnessCheck(
    max_tokens=1,
    timeout=180,
    chat_template_kwargs={"enable_thinking": False},
)
QWEN3_6_UNQUANTIZED_MOE_BACKENDS = ("FlashInfer TRTLLM", "FlashInfer CUTLASS", "TRITON")
QWEN3_6_FP8_PER_BLOCK_SKIP_REASON = (
    "vLLM 0.20.2 Qwen 3.6 fp8_per_block selects TRITON Fp8 on SM120, then fails warmup in "
    "cutlass_scaled_mm with Invalid status"
)
QWEN3_6_MXFP8_SKIP_REASON = (
    "vLLM rejects Qwen 3.6 online MXFP8 because input_size_per_partition 4304 is not divisible by 32"
)
QWEN3_6_FP8_CHECKPOINT_SKIP_REASON = (
    "Qwen/Qwen3.6-35B-A3B-FP8 is not cached in FT CI, and the job runs Hugging Face in offline mode"
)
QWEN3_6_QUANTIZED_MOE_CASES = (
    pytest.param(
        Qwen36QuantizedMoeCase(
            model_name=QWEN3_6_35B_A3B_BF16_MODEL,
            output_dir_name="qwen3_6_35b_a3b_bf16_moe_online_fp8",
            quantization="fp8",
            expected_moe_family="Fp8",
            residency_note="Online FP8 quantization of BF16 Qwen 3.6 MoE on the 40GB SM80 target.",
        ),
        marks=(pytest.mark.gpu_vram_40g, pytest.mark.gpu_sm_80),
        id="online-fp8-sm80-40g",
    ),
    pytest.param(
        Qwen36QuantizedMoeCase(
            model_name=QWEN3_6_35B_A3B_BF16_MODEL,
            output_dir_name="qwen3_6_35b_a3b_bf16_moe_online_fp8_per_tensor",
            quantization="fp8_per_tensor",
            expected_moe_family="Fp8",
            residency_note="Online per-tensor FP8 quantization of BF16 Qwen 3.6 MoE on the SM120 target.",
        ),
        marks=(pytest.mark.gpu_vram_96g, pytest.mark.gpu_sm_min_120),
        id="online-fp8-per-tensor-sm120-96g",
    ),
    pytest.param(
        Qwen36QuantizedMoeCase(
            model_name=QWEN3_6_35B_A3B_BF16_MODEL,
            output_dir_name="qwen3_6_35b_a3b_bf16_moe_online_fp8_per_block",
            quantization="fp8_per_block",
            expected_moe_family="Fp8",
            residency_note="Online per-block FP8 quantization of BF16 Qwen 3.6 MoE on the SM120 target.",
        ),
        marks=(
            pytest.mark.gpu_vram_96g,
            pytest.mark.gpu_sm_min_120,
            pytest.mark.skip(reason=QWEN3_6_FP8_PER_BLOCK_SKIP_REASON),
        ),
        id="online-fp8-per-block-sm120-96g",
    ),
    pytest.param(
        Qwen36QuantizedMoeCase(
            model_name=QWEN3_6_35B_A3B_BF16_MODEL,
            output_dir_name="qwen3_6_35b_a3b_bf16_moe_online_mxfp8",
            quantization="mxfp8",
            expected_moe_family="MxFp8",
            residency_note="Online MXFP8 quantization of BF16 Qwen 3.6 MoE on the SM120 target.",
        ),
        marks=(
            pytest.mark.gpu_vram_96g,
            pytest.mark.gpu_sm_min_120,
            pytest.mark.skip(reason=QWEN3_6_MXFP8_SKIP_REASON),
        ),
        id="online-mxfp8-sm120-96g",
    ),
    pytest.param(
        Qwen36QuantizedMoeCase(
            model_name=QWEN3_6_35B_A3B_FP8_MODEL,
            output_dir_name="qwen3_6_35b_a3b_fp8_moe_checkpoint",
            quantization=None,
            expected_moe_family="Fp8",
            residency_note="Official Qwen 3.6 FP8 checkpoint with fine-grained block-128 FP8 weights.",
        ),
        marks=(
            pytest.mark.gpu_vram_96g,
            pytest.mark.gpu_sm_min_120,
            pytest.mark.skip(reason=QWEN3_6_FP8_CHECKPOINT_SKIP_REASON),
        ),
        id="checkpoint-fp8-sm120-96g",
    ),
)


def qwen3_6_35b_a3b_bf16_case_with_moe_backend(
    moe_backend: str,
    *,
    output_dir_name: str,
) -> VllmOffloadSmokeCase:
    """Return the Qwen 3.6 smoke case with an explicit MoE backend."""
    cli_args = list(QWEN3_6_35B_A3B_BF16_SMOKE_CASE.cli_args)
    try:
        backend_index = cli_args.index("--moe-backend") + 1
    except ValueError as exc:
        raise AssertionError("Qwen 3.6 BF16 smoke case must define --moe-backend") from exc
    cli_args[backend_index] = moe_backend
    return VllmOffloadSmokeCase(
        model_name=QWEN3_6_35B_A3B_BF16_SMOKE_CASE.model_name,
        output_dir_name=output_dir_name,
        cli_args=tuple(cli_args),
        extra_env_vars=QWEN3_6_35B_A3B_BF16_SMOKE_CASE.extra_env_vars,
    )


def require_vllm_quantization(quantization: str | None) -> None:
    """Skip when the installed vLLM package does not accept ``quantization``."""
    if quantization is None:
        return

    try:
        version = importlib.metadata.version("vllm")
        from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    except (ImportError, importlib.metadata.PackageNotFoundError):
        pytest.skip(f"online quantization {quantization!r} requires vLLM")

    if quantization not in QUANTIZATION_METHODS:
        pytest.skip(f"online quantization {quantization!r} is not supported by vLLM {version}")


@pytest.fixture
def test_output_dir(request) -> Path:
    """Fixture that provides a unique output directory for each test case."""
    test_dir = Path(__file__).parent
    sanitized_name = sanitize_test_name(request.node.name)
    output_dir = test_dir / "test_results" / sanitized_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestQwen36Bf16Moe:
    """Qwen 3.6 BF16 MoE functional coverage for FlexTensor's vLLM worker."""

    @pytest.mark.gpu_vram_40g
    @pytest.mark.gpu_sm_80
    def test_qwen3_6_35b_a3b_bf16_moe_serves_with_offloading(
        self,
        test_output_dir: Path,
    ) -> None:
        """Smoke-test Qwen 3.6 BF16 MoE serving through the vLLM worker.

        This checkpoint has ~67.0 GiB of HF safetensor shards, larger than
        the 40GB A100 target. A successful request validates that FlexTensor
        can offload a newer BF16 MoE model without test-specific include
        pattern overrides.
        """
        case = QWEN3_6_35B_A3B_BF16_SMOKE_CASE

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, offload_logs = run_vllm_server_test(
            case,
            output_dir=output_dir,
            # Qwen thinking mode can spend the one-token smoke budget on reasoning,
            # leaving empty final content even when generation succeeds.
            correctness_check=QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK,
        )

        assert offload_metrics["usage"].get("completion_tokens", 0) > 0, (
            f"{case.model_name} returned no generated tokens"
        )
        assert_no_triton_cpu_pointer_failure(offload_logs)
        backend_evidence = offload_metrics["backend_evidence"]
        assert_moe_backend_selection(
            backend_evidence,
            expected_backend="TRITON",
            expected_family="Unquantized",
            expected_potential_backends=QWEN3_6_UNQUANTIZED_MOE_BACKENDS,
        )
        assert_rejected_moe_backend_reason(
            backend_evidence,
            backend="FlashInfer TRTLLM",
            reason_contains="kernel does not support current device cuda",
        )
        assert_rejected_moe_backend_reason(
            backend_evidence,
            backend="FlashInfer CUTLASS",
            reason_contains="kernel does not support current device cuda",
        )
        if offload_memory.available_kv_cache_memory_gib is not None:
            assert offload_memory.available_kv_cache_memory_gib > 0, f"{case.model_name} reported no available KV cache"

        metrics_file = test_output_dir / f"{case.output_dir_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(
                {
                    "model_name": case.model_name,
                    "checkpoint_weight_bytes": QWEN3_6_35B_A3B_BF16_WEIGHT_BYTES,
                    "offload": offload_metrics,
                },
                f,
                indent=2,
            )

        print(f"\nQwen 3.6 BF16 MoE metrics saved to: {metrics_file}")

    @pytest.mark.gpu_vram_96g
    @pytest.mark.gpu_sm_min_120
    def test_qwen3_6_35b_a3b_bf16_moe_records_auto_backend_on_sm120(
        self,
        test_output_dir: Path,
    ) -> None:
        """Record the Blackwell oracle choice for Qwen 3.6 BF16 MoE."""
        case = QWEN3_6_35B_A3B_BF16_SMOKE_CASE

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, offload_logs = run_vllm_server_test(
            case,
            output_dir=output_dir,
            correctness_check=QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK,
        )

        assert offload_metrics["usage"].get("completion_tokens", 0) > 0, (
            f"{case.model_name} returned no generated tokens"
        )
        assert_no_triton_cpu_pointer_failure(offload_logs)
        assert_moe_backend_selection_in(
            offload_metrics["backend_evidence"],
            expected_backends=QWEN3_6_UNQUANTIZED_MOE_BACKENDS,
            expected_family="Unquantized",
            expected_potential_backends=QWEN3_6_UNQUANTIZED_MOE_BACKENDS,
        )
        if offload_memory.available_kv_cache_memory_gib is not None:
            assert offload_memory.available_kv_cache_memory_gib > 0, f"{case.model_name} reported no available KV cache"

        metrics_file = test_output_dir / f"{case.output_dir_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(
                {
                    "model_name": case.model_name,
                    "checkpoint_weight_bytes": QWEN3_6_35B_A3B_BF16_WEIGHT_BYTES,
                    "offload": offload_metrics,
                },
                f,
                indent=2,
            )

        print(f"\nQwen 3.6 BF16 MoE SM120 metrics saved to: {metrics_file}")

    @pytest.mark.gpu_vram_40g
    @pytest.mark.gpu_sm_80
    def test_qwen3_6_35b_a3b_bf16_moe_explicit_triton_on_sm80(
        self,
        test_output_dir: Path,
    ) -> None:
        """Probe explicit Triton MoE backend selection on A100."""
        case = qwen3_6_35b_a3b_bf16_case_with_moe_backend(
            "triton",
            output_dir_name="qwen3_6_35b_a3b_bf16_moe_triton",
        )

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, offload_logs = run_vllm_server_test(
            case,
            output_dir=output_dir,
            correctness_check=QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK,
        )

        assert offload_metrics["usage"].get("completion_tokens", 0) > 0, (
            f"{case.model_name} returned no generated tokens"
        )
        assert_no_triton_cpu_pointer_failure(offload_logs)
        assert_moe_backend_selection(
            offload_metrics["backend_evidence"],
            expected_backend="TRITON",
            expected_family="Unquantized",
        )
        if offload_memory.available_kv_cache_memory_gib is not None:
            assert offload_memory.available_kv_cache_memory_gib > 0, f"{case.model_name} reported no available KV cache"

        metrics_file = test_output_dir / f"{case.output_dir_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(
                {
                    "model_name": case.model_name,
                    "checkpoint_weight_bytes": QWEN3_6_35B_A3B_BF16_WEIGHT_BYTES,
                    "offload": offload_metrics,
                },
                f,
                indent=2,
            )

        print(f"\nQwen 3.6 BF16 MoE explicit Triton metrics saved to: {metrics_file}")

    @pytest.mark.gpu_vram_96g
    @pytest.mark.gpu_sm_min_120
    def test_qwen3_6_35b_a3b_bf16_moe_explicit_flashinfer_cutlass_on_sm120(
        self,
        test_output_dir: Path,
    ) -> None:
        """Probe explicit FlashInfer CUTLASS MoE backend selection on Blackwell."""
        case = qwen3_6_35b_a3b_bf16_case_with_moe_backend(
            "flashinfer_cutlass",
            output_dir_name="qwen3_6_35b_a3b_bf16_moe_flashinfer_cutlass",
        )

        output_dir = test_output_dir / case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, offload_logs = run_vllm_server_test(
            case,
            output_dir=output_dir,
            correctness_check=QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK,
        )

        assert offload_metrics["usage"].get("completion_tokens", 0) > 0, (
            f"{case.model_name} returned no generated tokens"
        )
        assert_no_triton_cpu_pointer_failure(offload_logs)
        assert_moe_backend_selection(
            offload_metrics["backend_evidence"],
            expected_backend="FlashInfer CUTLASS",
            expected_family="Unquantized",
        )
        if offload_memory.available_kv_cache_memory_gib is not None:
            assert offload_memory.available_kv_cache_memory_gib > 0, f"{case.model_name} reported no available KV cache"

        metrics_file = test_output_dir / f"{case.output_dir_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(
                {
                    "model_name": case.model_name,
                    "checkpoint_weight_bytes": QWEN3_6_35B_A3B_BF16_WEIGHT_BYTES,
                    "offload": offload_metrics,
                },
                f,
                indent=2,
            )

        print(f"\nQwen 3.6 BF16 MoE explicit FlashInfer CUTLASS metrics saved to: {metrics_file}")

    @pytest.mark.parametrize("case", QWEN3_6_QUANTIZED_MOE_CASES)
    def test_qwen3_6_35b_a3b_quantized_moe_records_backend_family(
        self,
        case: Qwen36QuantizedMoeCase,
        test_output_dir: Path,
    ) -> None:
        """Probe Qwen 3.6 MoE quantized oracle paths accepted by vLLM."""
        require_vllm_quantization(case.quantization)
        vllm_case = case.to_vllm_case()

        output_dir = test_output_dir / vllm_case.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, offload_logs = run_vllm_server_test(
            vllm_case,
            output_dir=output_dir,
            correctness_check=QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK,
        )

        assert offload_metrics["usage"].get("completion_tokens", 0) > 0, (
            f"{vllm_case.model_name} returned no generated tokens"
        )
        assert_no_triton_cpu_pointer_failure(offload_logs)
        backend_evidence = offload_metrics["backend_evidence"]
        assert_moe_backend_family(backend_evidence, expected_family=case.expected_moe_family)
        if case.quantization is not None:
            assert backend_evidence["quantization"] == case.quantization
        if offload_memory.available_kv_cache_memory_gib is not None:
            assert offload_memory.available_kv_cache_memory_gib > 0, (
                f"{vllm_case.model_name} reported no available KV cache"
            )

        metrics_file = test_output_dir / f"{vllm_case.output_dir_name}_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(
                {
                    "model_name": vllm_case.model_name,
                    "quantization": case.quantization,
                    "expected_moe_family": case.expected_moe_family,
                    "residency_note": case.residency_note,
                    "offload": offload_metrics,
                },
                f,
                indent=2,
            )

        selected_backend = backend_evidence["selected_moe_backend"]
        selected_family = backend_evidence["selected_moe_backend_family"]
        print(
            f"\nQwen 3.6 quantized MoE metrics saved to: {metrics_file}; selected {selected_backend} {selected_family}"
        )

    @pytest.mark.gpu_vram_96g
    @pytest.mark.gpu_sm_min_120
    @pytest.mark.skip(
        reason=(
            "vLLM rejects Qwen 3.6 BF16 --moe-backend flashinfer_trtllm on SM120: "
            "kernel does not support current device cuda"
        )
    )
    def test_qwen3_6_35b_a3b_bf16_moe_explicit_flashinfer_trtllm_unsupported_on_sm120(self) -> None:
        """Record the current explicit FlashInfer TRTLLM rejection on SM120."""
