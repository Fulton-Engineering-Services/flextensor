# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared vLLM integration-test server helpers."""

from __future__ import annotations

import importlib.metadata
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import requests

from tests.integration.L0_contrib_vllm.vllm_utils import (
    MemoryProfilingMetrics,
    make_chat_request,
    parse_memory_profiling_logs,
    start_vllm_server,
    stop_vllm_server,
    wait_for_server,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class VllmOffloadSmokeCase:
    """Model-specific vLLM offload smoke-test configuration."""

    model_name: str
    output_dir_name: str
    cli_args: tuple[str, ...] = ()
    extra_env_vars: tuple[tuple[str, str], ...] = ()


def sanitize_test_name(test_name: str) -> str:
    """Sanitize a pytest test name so it can be used as a directory name."""
    sanitized = re.sub(r'[<>:"/\\|?*\[\]]', "_", test_name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def parse_version_triplet(version: str) -> tuple[int, int, int] | None:
    """Parse a leading ``major.minor.patch`` triplet from a package version."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def require_vllm_min_version(min_version: tuple[int, int, int], reason: str) -> None:
    """Skip when the installed vLLM package is older than ``min_version``."""
    min_version_text = ".".join(str(part) for part in min_version)
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{reason}; requires vLLM >= {min_version_text}")

    parsed_version = parse_version_triplet(version)
    if parsed_version is None or parsed_version < min_version:
        pytest.skip(f"{reason}; requires vLLM >= {min_version_text}, found {version}")


def run_vllm_server_test(
    model_name: str,
    enable_offload: bool,
    cli_args: list[str],
    output_dir: Path,
    port: int = 8000,
    extra_env_vars: dict[str, str] | None = None,
    chat_max_tokens: int = 20,
    chat_request_timeout: int = 60,
) -> tuple[MemoryProfilingMetrics, dict, list[str]]:
    """Run a single vLLM server test and return metrics.

    Args:
        model_name: Model to test.
        enable_offload: Whether to enable FlexTensor offloading.
        cli_args: Additional CLI arguments to pass to vLLM serve.
        output_dir: Directory for test output.
        port: Port to run server on.
        extra_env_vars: Additional environment variables to set.

    Returns:
        Tuple of parsed memory metrics, request metrics, and raw server log lines.
    """
    process = None
    log_lines: list[str] = []
    offload_status = "with offloading" if enable_offload else "without offloading (baseline)"
    result_metrics: dict = {
        "model_name": model_name,
        "offload_enabled": enable_offload,
    }

    success = False
    try:
        instrumentation_dir = output_dir / "instrumentation"
        instrumentation_dir.mkdir(parents=True, exist_ok=True)

        additional_env_vars = {
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "VLLM_NO_USAGE_STATS": "1",
            "FT_DEBUG_LOG_PATH": str(output_dir / "debug.log"),
            "FT_ENABLE_INSTRUMENTATION": "1",
            "FT_INSTRUMENTATION_OUTPUT_DIR": str(instrumentation_dir),
        }
        if extra_env_vars:
            assert "FT_ENABLED" not in extra_env_vars, "Use the enable_offload parameter instead"
            additional_env_vars.update(extra_env_vars)

        cli_args = ["--enforce-eager", *cli_args]
        process, log_lines = start_vllm_server(
            model_name,
            offload_enabled=enable_offload,
            port=port,
            additional_cli_args=cli_args,
            additional_env_vars=additional_env_vars,
        )
        ready = wait_for_server(port=port, timeout=600, process=process)

        assert ready, f"Server failed to start within timeout ({offload_status})"

        start_time = time.time()
        chat_response = make_chat_request(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model=model_name,
            port=port,
            max_tokens=chat_max_tokens,
            timeout=chat_request_timeout,
        )
        request_latency = time.time() - start_time

        assert "choices" in chat_response, "No choices in chat response"
        assert len(chat_response["choices"]) > 0, "Empty choices in chat response"
        assert "message" in chat_response["choices"][0], "No message in chat choice"

        print(f"{offload_status} - Chat response: {chat_response['choices'][0]['message']['content']}")

        models_url = f"http://localhost:{port}/v1/models"
        models_response = requests.get(models_url, timeout=10)
        models_response.raise_for_status()

        models_data = models_response.json()
        assert "data" in models_data, "No data in models response"
        assert len(models_data["data"]) > 0, "No models returned"

        model_ids = [model["id"] for model in models_data["data"]]
        print(f"{offload_status} - Available models: {model_ids}")
        assert model_name in model_ids, f"Model {model_name} not found in {model_ids}"

        result_metrics["request_latency_seconds"] = request_latency
        result_metrics["usage"] = chat_response.get("usage", {})
        result_metrics["response_content"] = chat_response["choices"][0]["message"]["content"]
        success = True

    finally:
        if process is not None:
            stop_vllm_server(process)
            time.sleep(1)

            if success:
                log_text = "\n".join(log_lines)
                offload_config_found = "FlexTensor offloading enabled with config" in log_text
                offload_applied_found = "FlexTensor offloading applied" in log_text

                if enable_offload:
                    assert offload_config_found, "FlexTensor offloading config not found in logs"
                    assert offload_applied_found, "FlexTensor offloading applied message not found"
                else:
                    assert not offload_config_found, "Unexpected offloading config in baseline"
                    assert not offload_applied_found, "Unexpected offloading applied in baseline"

    memory_metrics = parse_memory_profiling_logs(log_lines)

    print(f"\n{offload_status} - Memory profiling metrics from vLLM server:")
    print(f"  Initial free memory: {memory_metrics.initial_free_memory_gib} GiB")
    print(f"  Weights memory: {memory_metrics.weights_memory_gib} GiB")
    print(f"  Available KV cache memory: {memory_metrics.available_kv_cache_memory_gib} GiB")
    print(f"  Total non KV cache memory: {memory_metrics.total_non_kv_cache_memory_gib} GiB")

    result_metrics["memory_profiling"] = memory_metrics.to_dict()

    return memory_metrics, result_metrics, log_lines
