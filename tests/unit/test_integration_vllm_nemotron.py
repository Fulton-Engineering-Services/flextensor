# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-style tests for Nemotron vLLM integration smoke configurations."""

import pytest

from tests.integration import _vllm_server
from tests.integration._vllm_utils import assert_moe_backend_selection
from tests.integration.L1_contrib_vllm_nemotron import test_nemotron3_nano
from tests.integration.L1_contrib_vllm_nemotron.test_nemotron3_nano import (
    NEMOTRON_3_NANO_FP8_REQUIRED_MOE_BACKENDS,
    NEMOTRON_3_NANO_FP8_SMOKE_CASE,
)
from tests.integration.L1_contrib_vllm_nemotron_super import test_nemotron3_super_nvfp4


class _RequestCapturedError(Exception):
    pass


@pytest.mark.parametrize(
    ("module", "run_smoke", "temperature"),
    [
        (
            test_nemotron3_nano,
            test_nemotron3_nano.TestNemotron3Nano().test_nemotron3_nano_fp8_moe_serves_with_offloading_on_l4,
            0.0,
        ),
        (
            test_nemotron3_super_nvfp4,
            test_nemotron3_super_nvfp4.TestNemotron3SuperNvfp4Instrumentation().test_nemotron3_super_nvfp4_moe_serves_and_dumps_instrumentation,
            1.0,
        ),
    ],
)
def test_nemotron_smoke_requests_disable_thinking(monkeypatch, tmp_path, module, run_smoke, temperature) -> None:
    captured = {}

    def capture_request(*_args, **kwargs):
        captured.update(kwargs)
        raise _RequestCapturedError

    monkeypatch.setattr(_vllm_server, "resolve_hf_reasoning_parser", lambda *_args: tmp_path / "parser.py")
    monkeypatch.setattr(module, "run_vllm_server_test", capture_request)

    with pytest.raises(_RequestCapturedError):
        run_smoke(tmp_path)

    correctness = captured["correctness_check"]
    assert correctness.max_tokens == 20
    assert correctness.timeout == 180
    assert correctness.temperature == temperature
    assert correctness.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("module", "run_smoke", "plugin_filename", "parser_name"),
    [
        (
            test_nemotron3_nano,
            test_nemotron3_nano.TestNemotron3Nano().test_nemotron3_nano_fp8_moe_serves_with_offloading_on_l4,
            "nano_v3_reasoning_parser.py",
            "nano_v3",
        ),
        (
            test_nemotron3_super_nvfp4,
            test_nemotron3_super_nvfp4.TestNemotron3SuperNvfp4Instrumentation().test_nemotron3_super_nvfp4_moe_serves_and_dumps_instrumentation,
            "super_v3_reasoning_parser.py",
            "super_v3",
        ),
    ],
)
def test_nemotron_smoke_uses_model_reasoning_parser(
    monkeypatch,
    tmp_path,
    module,
    run_smoke,
    plugin_filename,
    parser_name,
) -> None:
    captured = {}
    parser_path = tmp_path / plugin_filename

    def capture_request(case, **_kwargs):
        captured["case"] = case
        raise _RequestCapturedError

    def resolve_parser(model_name, filename):
        captured["parser_source"] = (model_name, filename)
        return parser_path

    monkeypatch.setattr(_vllm_server, "resolve_hf_reasoning_parser", resolve_parser)
    monkeypatch.setattr(module, "run_vllm_server_test", capture_request)

    with pytest.raises(_RequestCapturedError):
        run_smoke(tmp_path)

    case = captured["case"]
    assert captured["parser_source"] == (case.model_name, plugin_filename)
    assert case.cli_args[-4:] == (
        "--reasoning-parser-plugin",
        str(parser_path),
        "--reasoning-parser",
        parser_name,
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
