# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style coverage for Qwen MoE integration helpers."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.integration.L0_contrib_vllm_qwen_moe import test_qwen3_6_bf16 as qwen_bf16


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
