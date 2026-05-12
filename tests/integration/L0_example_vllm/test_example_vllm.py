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

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples" / "vllm"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SERVER_PORT = 8000
SERVER_TIMEOUT = 600  # seconds to wait for server startup


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
        extra_env: dict[str, str] | None = None,
        run_client: bool = True,
    ) -> list[str]:
        """Start serve.sh, optionally run client.sh, and return collected log lines.

        Args:
            extra_env: Additional environment variables passed to serve.sh.
                These simulate ``docker run -e KEY=VALUE`` passthrough.
            run_client: Whether to run client.sh and validate API responses.

        Returns:
            Collected server log lines for further assertion by the caller.
        """
        serve_script = str(EXAMPLES_DIR / "serve.sh")
        client_script = str(EXAMPLES_DIR / "client.sh")

        from tests.integration.conftest import enable_offline_if_cached

        enable_offline_if_cached(MODEL_NAME)

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        log_lines: list[str] = []
        process = subprocess.Popen(  # noqa: S603 - inputs are controlled by test
            ["bash", serve_script, MODEL_NAME],  # noqa: S607
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

            # Verify FlexTensor offloading was active — offloading messages
            # are emitted during model loading (well before /health), but the
            # reader thread may still be catching up. Join with a timeout to
            # ensure all buffered output has been consumed before asserting.
            log_thread.join(timeout=10)
            if log_thread.is_alive():
                print("WARNING: log reader thread did not finish within 10s")

            log_text = "\n".join(log_lines)
            assert "FlexTensor offloading enabled with config" in log_text, (
                "FlexTensor offloading config not found in server logs"
            )
            assert "FlexTensor offloading applied" in log_text, (
                "FlexTensor offloading applied message not found in server logs"
            )

        finally:
            _stop_server(process)
            log_thread.join(timeout=5)
            # Dump last 50 lines of server logs for debugging on failure
            if log_lines:
                print(f"\n{'=' * 60}\nServer log ({len(log_lines)} lines, last 50):\n{'=' * 60}")
                for line in log_lines[-50:]:
                    print(f"  {line}")

        return log_lines

    @pytest.mark.gpu_vram_24g
    def test_example_scripts_end_to_end(self) -> None:
        """Test that serve.sh starts a working server and client.sh gets valid responses.

        This test exercises the exact same scripts that users run, validating
        the example end-to-end:
        1. Start serve.sh with a small model as a background process
        2. Wait for the server to become healthy
        3. Run client.sh and assert it exits successfully (smoke test)
        4. Make structured Python requests for detailed assertions
        5. Assert FlexTensor offloading log messages are present
        6. Stop the server
        """
        self._run_serve_and_validate()

    @pytest.mark.gpu_vram_24g
    def test_env_var_passthrough(self) -> None:
        """Test that FT_* environment variables pass through to FlexTensor.

        Simulates ``docker run -e FT_ENABLE_DIAGNOSTICS=1`` by setting the
        variable in the subprocess environment. Verifies that FlexTensor
        receives the variable by checking for the BLOCK ASSIGNMENT table in
        server logs, which is only emitted when diagnostics are on and does
        not depend on profiling-data consistency.
        """
        log_lines = self._run_serve_and_validate(
            extra_env={"FT_ENABLE_DIAGNOSTICS": "1"},
            run_client=False,
        )

        log_text = "\n".join(log_lines)
        assert "BLOCK ASSIGNMENT" in log_text, (
            "FT_ENABLE_DIAGNOSTICS=1 was set but BLOCK ASSIGNMENT table not "
            "found in server logs — env var passthrough may be broken"
        )
