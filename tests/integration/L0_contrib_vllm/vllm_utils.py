# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for vLLM integration tests with FlexTensor."""

import json
import os
import re
import signal
import subprocess  # noqa: S404 - subprocess needed for test utilities
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class MemoryProfilingMetrics:
    """Memory profiling metrics extracted from vLLM server logs."""

    initial_free_memory_gib: float | None = None
    requested_memory_util: float | None = None
    requested_memory_gib: float | None = None
    free_memory_after_profiling_total_gib: float | None = None
    free_memory_after_profiling_within_requested_gib: float | None = None
    total_non_kv_cache_memory_gib: float | None = None
    torch_peak_memory_increase_gib: float | None = None
    non_torch_forward_increase_memory_gib: float | None = None
    weights_memory_gib: float | None = None
    available_kv_cache_memory_gib: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "initial_free_memory_gib": self.initial_free_memory_gib,
            "requested_memory_util": self.requested_memory_util,
            "requested_memory_gib": self.requested_memory_gib,
            "free_memory_after_profiling_total_gib": self.free_memory_after_profiling_total_gib,
            "free_memory_after_profiling_within_requested_gib": self.free_memory_after_profiling_within_requested_gib,
            "total_non_kv_cache_memory_gib": self.total_non_kv_cache_memory_gib,
            "torch_peak_memory_increase_gib": self.torch_peak_memory_increase_gib,
            "non_torch_forward_increase_memory_gib": self.non_torch_forward_increase_memory_gib,
            "weights_memory_gib": self.weights_memory_gib,
            "available_kv_cache_memory_gib": self.available_kv_cache_memory_gib,
        }


def parse_memory_profiling_logs(log_lines: list[str]) -> MemoryProfilingMetrics:
    """Parse vLLM server logs to extract memory profiling metrics.

    Parses log lines from vLLM's gpu_worker.py memory profiling output:
    - Initial free memory and requested memory
    - Free memory after profiling
    - Total non KV cache memory breakdown (weights, torch peak, non-torch forward)
    - Available KV cache memory

    Args:
        log_lines: List of log lines captured from vLLM server

    Returns:
        MemoryProfilingMetrics dataclass with extracted values (None if not found)
    """
    metrics = MemoryProfilingMetrics()
    log_text = "\n".join(log_lines)

    # Pattern: Initial free memory: 44.04 GiB; Requested memory: 0.90 (util), 40.07 GiB
    initial_memory_pattern = (
        r"Initial free memory:\s*([\d.]+)\s*GiB;\s*"
        r"Requested memory:\s*([\d.]+)\s*\(util\),\s*([\d.]+)\s*GiB"
    )
    match = re.search(initial_memory_pattern, log_text)
    if match:
        metrics.initial_free_memory_gib = float(match.group(1))
        metrics.requested_memory_util = float(match.group(2))
        metrics.requested_memory_gib = float(match.group(3))

    # Pattern: Free memory after profiling: 28.96 GiB (total), 25.00 GiB (within requested)
    free_after_profiling_pattern = (
        r"Free memory after profiling:\s*([\d.]+)\s*GiB\s*\(total\),\s*"
        r"([\d.]+)\s*GiB\s*\(within requested\)"
    )
    match = re.search(free_after_profiling_pattern, log_text)
    if match:
        metrics.free_memory_after_profiling_total_gib = float(match.group(1))
        metrics.free_memory_after_profiling_within_requested_gib = float(match.group(2))

    # Pattern: Total non KV cache memory: 16.22GiB; torch peak memory increase: 1.19GiB;
    #          non-torch forward increase memory: 0.04GiB; weights memory: 14.99GiB.
    # Note: GiB may or may not have space before it
    memory_breakdown_pattern = (
        r"Total non KV cache memory:\s*([\d.]+)\s*GiB;\s*"
        r"torch peak memory increase:\s*([\d.]+)\s*GiB;\s*"
        r"non-torch forward increase memory:\s*([\d.]+)\s*GiB;\s*"
        r"weights memory:\s*([\d.]+)\s*GiB"
    )
    match = re.search(memory_breakdown_pattern, log_text)
    if match:
        metrics.total_non_kv_cache_memory_gib = float(match.group(1))
        metrics.torch_peak_memory_increase_gib = float(match.group(2))
        metrics.non_torch_forward_increase_memory_gib = float(match.group(3))
        metrics.weights_memory_gib = float(match.group(4))

    # Pattern: Available KV cache memory: 23.86 GiB
    kv_cache_pattern = r"Available KV cache memory:\s*([\d.]+)\s*GiB"
    match = re.search(kv_cache_pattern, log_text)
    if match:
        metrics.available_kv_cache_memory_gib = float(match.group(1))

    return metrics


def parse_layer_duration_stats(log_lines: list[str]) -> dict[str, dict[str, float | int]]:
    """Parse the Layer Duration Statistics table from FlexTensor log output.

    The table is emitted by FlexTensor after profiling and shows every offload trap
    with its timing statistics. It is the authoritative source for which module
    patterns were actually applied as traps.

    The table appears after every profiling run when FT_ENABLE_DIAGNOSTICS=1 (INFO path,
    consistent measurements), and also when any layer has insufficient samples or high
    unexplained CV (WARNING path, inconsistent measurements). Exactly one of these two
    paths fires per run.

    Args:
        log_lines: List of log lines captured from vLLM server

    Returns:
        Dict mapping layer_name to timing stats dict with keys:
        min, max, median, avg, std (all in ms), count (int), and cv — the coefficient
        of variation as a percentage value (e.g., 25.0 means 25%). Note: this differs
        from LayerDurationStatistics.coefficient_of_variation, which is a ratio (e.g., 0.25).
        Returns empty dict if the table is not found in the logs.

    Example:
        >>> stats = parse_layer_duration_stats(log_lines)
        >>> any(".layers." in k or re.search(r"\\.\\d+$", k) for k in stats)
        True
        >>> "LlamaForCausalLM.model" in stats  # coarse trap — bad
        False
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    # Row: LayerName  float float float float float  float%  int
    row_re = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)$")
    in_table = False
    result: dict[str, dict[str, float | int]] = {}

    for raw_line in log_lines:
        # Strip ANSI codes first
        line = ansi_escape.sub("", raw_line)
        # log_lines stores the raw vLLM subprocess output (the test print adds "[vLLM]" to
        # the screen only). Each line starts with "(ProcessName pid=N) " from vLLM's logger.
        # Strip "[vLLM] " tag (present if line came from a nested test print) then the
        # "(ProcessName pid=N) " prefix that vLLM always emits.
        line = re.sub(r"^\[vLLM\]\s*", "", line)
        line = re.sub(r"^\(\S+ pid=\d+\)\s*", "", line)
        # Strip optional FlexTensor timestamp: [2026-01-01 ...] LEVEL module.py:N:
        line = re.sub(r"^\[\d{4}-\d{2}-\d{2}.*?\]\s*\w+\s+\S+:\s*", "", line).strip()

        if "Layer Duration Statistics" in line:
            in_table = True
            continue

        if in_table:
            m = row_re.match(line)
            if m:
                result[m.group(1)] = {
                    "min": float(m.group(2)),
                    "max": float(m.group(3)),
                    "median": float(m.group(4)),
                    "avg": float(m.group(5)),
                    "std": float(m.group(6)),
                    "cv": float(m.group(7)),
                    "count": int(m.group(8)),
                }
            elif line.startswith(("=", "-", "Layer")):
                continue  # separator or column header
            elif line:
                in_table = False  # non-data line ends the table

    return result


def _log_reader_thread(process: subprocess.Popen[bytes], log_lines: list[str], prefix: str = "[vLLM]") -> None:
    """Background thread to read and display server logs in real-time.

    Args:
        process: Server process to read logs from
        log_lines: Shared list to store log lines for later inspection
        prefix: Prefix to add to each log line when printing
    """
    if process.stdout is None:
        return

    for line in iter(process.stdout.readline, b""):
        if not line:
            break
        decoded_line = line.decode("utf-8", errors="replace").rstrip()
        if decoded_line:
            log_lines.append(decoded_line)
            print(f"{prefix} {decoded_line}", flush=True)

    # Read any remaining output
    remaining = process.stdout.read()
    if remaining:
        for line in remaining.decode("utf-8", errors="replace").splitlines():
            if line:
                log_lines.append(line)
                print(f"{prefix} {line}", flush=True)


def start_vllm_server(
    model_name: str,
    offload_enabled: bool = True,
    port: int = 8000,
    additional_cli_args: list[str] | None = None,
    additional_env_vars: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[bytes], list[str]]:
    """Start vLLM server in background with real-time log monitoring.

    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        offload_enabled: Whether to enable FlexTensor offloading
        port: Port to run server on
        additional_cli_args: Additional CLI arguments to pass to vllm serve
        additional_env_vars: Additional environment variables to set

    Returns:
        Tuple of (subprocess.Popen object for the server process, shared log lines list)
    """

    env_vars = {}
    cmd = ["vllm", "serve", model_name, "--port", str(port)]
    if offload_enabled:
        env_vars["FT_ENABLED"] = "1"
        cmd += ["--worker-cls", "flextensor.contrib.vllm.worker.FlexTensorOffloadWorker"]

    if additional_env_vars:
        env_vars.update(additional_env_vars)

    if additional_cli_args:
        cmd.extend(additional_cli_args)
    print(f"\nStarting vLLM server with command: {' '.join(cmd)}")
    print(f"Additional environment variables: {env_vars}")

    env = os.environ.copy()
    env.update(env_vars)

    # Start server with output redirected to subprocess
    process = subprocess.Popen(  # noqa: S603 - inputs are controlled by test parametrization
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=1,
    )

    # Start background thread to read and display logs in real-time
    log_lines: list[str] = []
    log_thread = threading.Thread(target=_log_reader_thread, args=(process, log_lines), daemon=True)
    log_thread.start()

    return process, log_lines


def wait_for_server(
    port: int = 8000, timeout: int = 600, check_interval: int = 5, process: subprocess.Popen[bytes] | None = None
) -> bool:
    """Wait for vLLM server to be ready.

    Args:
        port: Port server is running on
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        process: Optional server process to monitor for crashes

    Returns:
        True if server is ready, False if timeout occurred or process crashed
    """
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()

    print(f"Waiting for server to be ready at {health_url}...")

    while time.time() - start_time < timeout:
        # Check if process has crashed
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                print(f"Server process exited with code {exit_code} before becoming ready")
                print("Check the server logs above for error details")
                return False

        try:
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                print(f"Server is ready after {time.time() - start_time:.1f}s")
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(max(check_interval - 2, 0.1))

    print(f"Server did not become ready within {timeout}s")
    return False


def stop_vllm_server(process: subprocess.Popen[bytes], timeout: int = 30) -> None:
    """Stop vLLM server process gracefully.

    Args:
        process: Server process to stop
        timeout: Maximum time to wait for graceful shutdown

    Note:
        Log lines are captured by the background thread started in start_vllm_server.
        This function only handles process termination.
    """
    if process.poll() is None:
        print("\nStopping vLLM server...")

        # Try graceful shutdown first
        process.send_signal(signal.SIGTERM)

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("Server did not stop gracefully, forcing...")
            process.kill()
            process.wait()


def load_gpu_memory_snapshots(snapshot_dir: Path) -> list[dict]:
    """Load GPU memory snapshots from a directory written by MemorySnapshotMixin.

    Reads all JSON files matching ``gpu_snapshots_rank*_device*.json`` in the
    given directory and returns the snapshots list from the first (and expected
    only) file found. Suitable for rank-0 single-device validation.

    Args:
        snapshot_dir: Directory passed as FT_VLLM_SNAPSHOT_OUTPUT_DIR.

    Returns:
        List of snapshot dicts, each with ``label``, ``gpu_memory``, and
        ``host_memory`` keys. Returns an empty list if no file is found.

    Example:
        >>> snapshots = load_gpu_memory_snapshots(Path("/tmp/gpu_snapshots"))
        >>> labels = [s["label"] for s in snapshots]
        >>> assert "after_load_model" in labels
    """
    json_files = sorted(snapshot_dir.glob("gpu_snapshots_rank0_device*.json"))
    if not json_files:
        return []
    data = json.loads(json_files[0].read_text())
    return data.get("snapshots", [])


def make_chat_request(
    messages: list[dict[str, str]], model: str, port: int = 8000, max_tokens: int = 100, timeout: int = 60
) -> dict[str, Any]:
    """Make a chat completion request to vLLM API.

    Args:
        messages: Chat messages
        model: Model name
        port: Server port
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        JSON response from API

    Raises:
        requests.exceptions.RequestException: If request fails
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}

    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
