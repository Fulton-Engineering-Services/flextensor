# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style coverage for Qwen MoE integration helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.integration.L1_contrib_vllm_qwen_moe import test_qwen3_6_bf16 as qwen_bf16


def test_qwen3_fp8_sm120_case_disables_unsupported_deepgemm() -> None:
    env_vars = dict(qwen_bf16.QWEN3_30B_A3B_FP8_CASE.extra_env_vars)

    assert env_vars["VLLM_USE_DEEP_GEMM"] == "0"


def test_qwen3_6_bf16_case_pins_bootstrap_storage_in_place() -> None:
    env_vars = dict(qwen_bf16.QWEN3_6_35B_A3B_BF16_SMOKE_CASE.extra_env_vars)

    assert env_vars["FT_PINNED_MEMORY_MODE"] == "host_register"


def test_qwen3_6_server_test_allows_slow_large_model_startup(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}

    def _fake_run(case, **kwargs):
        observed["case"] = case
        observed.update(kwargs)
        return "memory", "metrics", "logs"

    monkeypatch.setattr(qwen_bf16, "run_vllm_server_test", _fake_run)

    result = qwen_bf16.run_qwen3_6_server_test(
        qwen_bf16.QWEN3_6_35B_A3B_BF16_SMOKE_CASE,
        output_dir=tmp_path,
    )

    assert result == ("memory", "metrics", "logs")
    assert observed["server_ready_timeout"] == 1800
    assert observed["correctness_check"] == qwen_bf16.QWEN3_6_ONE_TOKEN_NON_THINKING_CHECK


def test_qwen3_6_bf16_case_with_moe_backend_replaces_backend() -> None:
    case = qwen_bf16.qwen3_6_35b_a3b_bf16_case_with_moe_backend(
        "triton",
        output_dir_name="qwen_triton",
    )

    cli_args = list(case.cli_args)
    assert cli_args[cli_args.index("--moe-backend") + 1] == "triton"
    assert case.output_dir_name == "qwen_triton"
    assert case.extra_env_vars == qwen_bf16.QWEN3_6_35B_A3B_BF16_SMOKE_CASE.extra_env_vars


def test_qwen3_6_bf16_case_with_moe_backend_requires_base_backend_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    cli_args = list(qwen_bf16.QWEN3_6_35B_A3B_BF16_SMOKE_CASE.cli_args)
    backend_index = cli_args.index("--moe-backend")
    del cli_args[backend_index : backend_index + 2]
    monkeypatch.setattr(
        qwen_bf16,
        "QWEN3_6_35B_A3B_BF16_SMOKE_CASE",
        replace(qwen_bf16.QWEN3_6_35B_A3B_BF16_SMOKE_CASE, cli_args=tuple(cli_args)),
    )

    with pytest.raises(AssertionError, match="must define --moe-backend"):
        qwen_bf16.qwen3_6_35b_a3b_bf16_case_with_moe_backend(
            "triton",
            output_dir_name="qwen_triton",
        )
