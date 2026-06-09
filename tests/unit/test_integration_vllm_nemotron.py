# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style tests for Nemotron vLLM integration smoke configurations."""

from tests.integration._vllm_utils import assert_moe_backend_selection
from tests.integration.L0_contrib_vllm_nemotron.test_nemotron3_nano import (
    NEMOTRON_3_NANO_FP8_REQUIRED_MOE_BACKENDS,
    NEMOTRON_3_NANO_FP8_SMOKE_CASE,
)


def test_nemotron3_nano_fp8_case_does_not_force_flashinfer_fp8_moe() -> None:
    env_vars = dict(NEMOTRON_3_NANO_FP8_SMOKE_CASE.extra_env_vars)

    assert "VLLM_USE_FLASHINFER_MOE_FP8" not in env_vars


def test_nemotron3_nano_fp8_expectations_match_vllm_fp8_oracle_names() -> None:
    evidence = {
        "selected_moe_backend": "TRITON",
        "selected_moe_backend_family": "Fp8",
        "potential_moe_backends": [
            "AITER",
            "FLASHINFER_TRTLLM",
            "FLASHINFER_CUTLASS",
            "DEEPGEMM",
            "TRITON",
            "MARLIN",
        ],
    }

    assert_moe_backend_selection(
        evidence,
        expected_backend="TRITON",
        expected_family="Fp8",
        expected_potential_backends=NEMOTRON_3_NANO_FP8_REQUIRED_MOE_BACKENDS,
    )
