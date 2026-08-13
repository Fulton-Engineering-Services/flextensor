# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional integration test for the examples/vllm/ shell scripts.

Validates that serve.sh and client.sh work end-to-end inside the
vllm/vllm-openai container by starting a real vLLM server with FlexTensor
offloading and hitting the OpenAI-compatible API. install.sh is run by
test.sh before this module executes.
"""

import contextlib
import os
import signal
import subprocess  # noqa: S404 - subprocess needed for test utilities
import threading
import time
from pathlib import Path

import pytest
import requests

from tests.integration._vllm_utils import (
    assert_vllm_cuda_platform_logged,
    validate_latest_memory_transfer_stats_for_current_bus,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples" / "vllm"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SERVER_PORT = 8000
SERVER_TIMEOUT = 600  # seconds to wait for server startup


def test_serve_sh_forwards_vllm_arguments(tmp_path: Path) -> None:
    """Catch helpers that drop model-specific arguments or force eager mode."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    args_file = tmp_path / "vllm-args.txt"
    fake_vllm = bin_dir / "vllm"
    fake_vllm.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$ARGS_FILE"\n')
    fake_vllm.chmod(0o755)

    env = os.environ.copy()
    env.update({"ARGS_FILE": str(args_file), "PATH": f"{bin_dir}:{env['PATH']}"})
    result = subprocess.run(  # noqa: S603 - inputs are controlled by test
        [
            "/bin/bash",
            str(EXAMPLES_DIR / "serve.sh"),
            MODEL_NAME,
            "--tensor-parallel-size",
            "2",
            "--max-model-len",
            "4096",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert args_file.read_text().splitlines() == [
        "serve",
        MODEL_NAME,
        "--worker-cls",
        "flextensor.contrib.vllm.worker.FlexTensorOffloadWorker",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "4096",
    ]


def _log_reader_thread(process: subprocess.Popen[bytes], log_lines: list[str]) -> None:
    """Collect and display server logs in real-time.

    Appends each line to log_lines for later assertion by the test.
    """
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, b""):
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                log_lines.append(decoded)
                print(f"[serve] {decoded}", flush=True)
    except Exception as e:
        msg = f"[LOG READER ERROR] {type(e).__name__}: {e}"
        log_lines.append(msg)
        print(msg, flush=True)


def _wait_for_server(port: int, timeout: int, process: subprocess.Popen[bytes]) -> bool:
    """Wait for vLLM server to be ready by polling /health."""
    health_url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        if process.poll() is not None:
            print(f"Server exited with code {process.returncode} before becoming ready")
            return False
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                print(f"Server ready after {time.time() - start:.1f}s")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass  # Expected while server is booting
        except requests.exceptions.RequestException as e:
            print(f"Unexpected error polling health endpoint: {e}")
        time.sleep(3)
    print(f"Server did not become ready within {timeout}s")
    return False


def _stop_server(process: subprocess.Popen[bytes], timeout: int = 30) -> None:
    """Stop server process group gracefully."""
    if process.poll() is None:
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return  # Process exited between poll() and getpgid()/killpg()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            process.wait()


class TestExampleVLLM:
    """Functional tests for the examples/vllm/ shell scripts."""

    def _run_serve_and_validate(
        self,
        output_dir: Path,
        extra_env: dict[str, str] | None = None,
        run_client: bool = True,
        vllm_args: tuple[str, ...] = (),
    ) -> list[str]:
        """Start serve.sh, optionally run client.sh, and return collected log lines.

        Args:
            output_dir: Directory for instrumentation output.
            extra_env: Additional environment variables passed to serve.sh.
                These simulate ``docker run -e KEY=VALUE`` passthrough.
            run_client: Whether to run client.sh and validate API responses.
            vllm_args: Model-specific arguments forwarded to ``vllm serve``.

        Returns:
            Collected server log lines for further assertion by the caller.
        """
        serve_script = str(EXAMPLES_DIR / "serve.sh")
        client_script = str(EXAMPLES_DIR / "client.sh")

        from tests.integration.conftest import enable_offline_if_cached

        enable_offline_if_cached(MODEL_NAME)

        instrumentation_dir = output_dir / "instrumentation"
        instrumentation_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "FT_ENABLE_INSTRUMENTATION": "1",
            "FT_INSTRUMENTATION_OUTPUT_DIR": str(instrumentation_dir),
        })
        if extra_env:
            env.update(extra_env)
        use_v2_worker = env.get("FT_VLLM_USE_V2_WORKER", "1") == "1"

        log_lines: list[str] = []
        process = subprocess.Popen(  # noqa: S603 - inputs are controlled by test
            ["bash", serve_script, MODEL_NAME, *vllm_args],  # noqa: S607
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=True,
        )

        log_thread = threading.Thread(target=_log_reader_thread, args=(process, log_lines), daemon=True)
        log_thread.start()

        try:
            ready = _wait_for_server(SERVER_PORT, SERVER_TIMEOUT, process)
            assert ready, "serve.sh failed to start a healthy server"

            if run_client:
                # Smoke test: run client.sh and check exit code
                client_result = subprocess.run(  # noqa: S603
                    ["bash", client_script, MODEL_NAME, str(SERVER_PORT)],  # noqa: S607
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                assert client_result.returncode == 0, (
                    f"client.sh failed with exit code {client_result.returncode}\n"
                    f"stdout: {client_result.stdout}\n"
                    f"stderr: {client_result.stderr}"
                )
                print(f"client.sh output:\n{client_result.stdout}")

                # Structured assertions via Python requests
                chat_url = f"http://localhost:{SERVER_PORT}/v1/chat/completions"
                chat_payload = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "The capital of France is"}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                }
                chat_resp = requests.post(chat_url, json=chat_payload, timeout=60)
                chat_resp.raise_for_status()
                chat_data = chat_resp.json()

                assert "choices" in chat_data, f"No choices in response: {chat_data}"
                assert len(chat_data["choices"]) > 0, "Empty choices"
                assert "message" in chat_data["choices"][0], "No message in first choice"
                assert chat_data["choices"][0]["message"]["content"], "Empty response content"
                print(f"Chat response: {chat_data['choices'][0]['message']['content']}")

                models_url = f"http://localhost:{SERVER_PORT}/v1/models"
                models_resp = requests.get(models_url, timeout=10)
                models_resp.raise_for_status()
                models_data = models_resp.json()

                assert "data" in models_data, f"No data in models response: {models_data}"
                model_ids = [m["id"] for m in models_data["data"]]
                assert MODEL_NAME in model_ids, f"{MODEL_NAME} not in {model_ids}"
                print(f"Available models: {model_ids}")

            # Verify FlexTensor offloading was active — the v2 takeover message
            # is emitted during model loading (well before /health), but the
            # reader thread may still be catching up. Join with a timeout to
            # ensure all buffered output has been consumed before asserting.
            log_thread.join(timeout=10)
            if log_thread.is_alive():
                print("WARNING: log reader thread did not finish within 10s")

            log_text = "\n".join(log_lines)
            assert_vllm_cuda_platform_logged(log_lines)
            if use_v2_worker:
                assert "FlexTensor vLLM integration v2 state takeover complete" in log_text, (
                    "FlexTensor v2 state takeover message not found in server logs"
                )
            else:
                assert "FlexTensor offloading enabled with config" in log_text, (
                    "FlexTensor legacy offloading config not found in server logs"
                )
                assert "FlexTensor offloading applied" in log_text, (
                    "FlexTensor legacy offloading applied message not found in server logs"
                )

        finally:
            _stop_server(process)
            log_thread.join(timeout=5)
            # Dump last 50 lines of server logs for debugging on failure
            if log_lines:
                print(f"\n{'=' * 60}\nServer log ({len(log_lines)} lines, last 50):\n{'=' * 60}")
                for line in log_lines[-50:]:
                    print(f"  {line}")

        if not use_v2_worker:
            transfer_validation = validate_latest_memory_transfer_stats_for_current_bus(instrumentation_dir)
            print("\nexamples/vllm - Memory transfer validation:")
            print(
                "  Bus: "
                f"PCIe Gen{transfer_validation['pcie_link_gen']} x{transfer_validation['pcie_link_width']} "
                f"({transfer_validation['pci_bus_id']}), CPU affinity={transfer_validation['cpu_affinity']}, "
                f"NUMA affinity={transfer_validation['numa_affinity']}"
            )
            print(
                "  Observed bandwidth: "
                f"{transfer_validation['observed_bandwidth_gbps']:.2f} GB/s "
                f"(expected {transfer_validation['min_expected_bandwidth_gbps']:.2f}-"
                f"{transfer_validation['max_expected_bandwidth_gbps']:.2f} GB/s)"
            )

        return log_lines

    @pytest.mark.gpu_vram_24g
    def test_example_scripts_end_to_end(self, tmp_path: Path) -> None:
        """Test that serve.sh starts a working server and client.sh gets valid responses.

        This test exercises the exact same scripts that users run, validating
        the example end-to-end:
        1. Start serve.sh with a small model as a background process
        2. Wait for the server to become healthy
        3. Run client.sh and assert it exits successfully (smoke test)
        4. Make structured Python requests for detailed assertions
        5. Assert the FlexTensor v2 state-takeover log message is present
        6. Stop the server
        """
        self._run_serve_and_validate(tmp_path)

    @pytest.mark.gpu_vram_24g
    def test_env_var_passthrough(self, tmp_path: Path) -> None:
        """Test that FT_* environment variables pass through to FlexTensor.

        Simulates ``docker run -e`` by selecting the legacy worker and enabling
        diagnostics in the subprocess environment. The legacy-worker selection
        is verified by ``_run_serve_and_validate``; the BLOCK ASSIGNMENT table
        verifies that diagnostics also passed through.
        """
        log_lines = self._run_serve_and_validate(
            tmp_path,
            extra_env={"FT_VLLM_USE_V2_WORKER": "0", "FT_ENABLE_DIAGNOSTICS": "1"},
            run_client=False,
            vllm_args=("--enforce-eager",),
        )

        log_text = "\n".join(log_lines)
        assert "BLOCK ASSIGNMENT" in log_text, (
            "FT_ENABLE_DIAGNOSTICS=1 was set but BLOCK ASSIGNMENT table not "
            "found in server logs — env var passthrough may be broken"
        )
