#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess integration test for real diffusers example scripts.

The test follows the PyTriton example-test shape: execute the user-facing
example scripts as commands and validate their outputs. Unlike a unit smoke
test, this uses the real ``diffusers.WanPipeline`` and downloads/loads the Wan
model through Hugging Face when it is not already cached.
"""

from __future__ import annotations

import argparse
import codecs
import os
import signal
import subprocess  # noqa: S404 - subprocess is the point of this example test
import sys
import threading
from pathlib import Path
from typing import BinaryIO, TextIO

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent
TEST_RESULTS_DIR = TEST_DIR / "test_results"
QUICKSTART_SCRIPT = REPO_ROOT / "examples" / "diffusers" / "quickstart" / "wan_t2v.py"
PROFILE_SCRIPT = REPO_ROOT / "examples" / "diffusers" / "profile-reuse" / "run_profile.py"
INFER_SCRIPT = REPO_ROOT / "examples" / "diffusers" / "profile-reuse" / "run_infer.py"

TEST_PROMPT = "A small blue cube rotating on a plain background"
MIN_AVAILABLE_RAM_GIB_ENV = "FT_DIFFUSERS_MIN_AVAILABLE_RAM_GIB"
DEFAULT_MIN_AVAILABLE_RAM_GIB = 110.0
pytestmark = pytest.mark.gpu_vram_40g


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _resolve_work_root(work_dir: str | None) -> Path:
    if work_dir is not None:
        return Path(work_dir)
    return TEST_RESULTS_DIR


def _read_meminfo_gib() -> dict[str, float]:
    meminfo = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError as exc:
        print(f"WARNING: could not read /proc/meminfo for RAM check: {exc}", flush=True)  # noqa: T201
        return meminfo

    for line in lines:
        key, _, value = line.partition(":")
        if key not in {"MemTotal", "MemAvailable"}:
            continue
        fields = value.split()
        if len(fields) < 2 or fields[1] != "kB":
            continue
        meminfo[key] = int(fields[0]) / 1024 / 1024
    return meminfo


def _minimum_available_ram_gib() -> float:
    raw_value = os.environ.get(MIN_AVAILABLE_RAM_GIB_ENV, str(DEFAULT_MIN_AVAILABLE_RAM_GIB))
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{MIN_AVAILABLE_RAM_GIB_ENV} must be a number of GiB, got: {raw_value}") from exc


def _format_gib(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f} GiB"


def _check_available_ram(stage: str, *, require: bool) -> None:
    meminfo = _read_meminfo_gib()
    total_gib = meminfo.get("MemTotal")
    available_gib = meminfo.get("MemAvailable")
    minimum_gib = _minimum_available_ram_gib()
    print(  # noqa: T201
        f"[{stage}] RAM: total={_format_gib(total_gib)} available={_format_gib(available_gib)} "
        f"minimum_available={minimum_gib:.1f} GiB",
        flush=True,
    )
    if available_gib is None:
        if require:
            raise RuntimeError(f"Could not determine available RAM before {stage}")
        return
    if require and minimum_gib > 0 and available_gib < minimum_gib:
        raise RuntimeError(
            f"Insufficient available RAM before {stage}: {available_gib:.1f} GiB available, "
            f"need at least {minimum_gib:.1f} GiB. Override with {MIN_AVAILABLE_RAM_GIB_ENV} if needed."
        )


def _tee_stream(stream: BinaryIO, destination: TextIO, log_file: TextIO, log_lock: threading.Lock) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while chunk := os.read(stream.fileno(), 4096):
        text = decoder.decode(chunk)
        destination.write(text)
        destination.flush()
        with log_lock:
            log_file.write(text)
            log_file.flush()
    if text := decoder.decode(b"", final=True):
        destination.write(text)
        destination.flush()
        with log_lock:
            log_file.write(text)
            log_file.flush()
    stream.close()


def _run_example_command(name: str, command: list[str], cwd: Path, timeout_s: float) -> str:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    env.setdefault("PYTHONUNBUFFERED", "1")

    log_path = cwd / f"{name}.log"
    command_path = cwd / f"{name}.command.txt"
    instrumentation_dir = cwd / f"{name}-instrumentation"
    instrumentation_dir.mkdir(exist_ok=True)
    command_path.write_text(" ".join(command) + "\n")
    env.setdefault("FT_ENABLE_DIAGNOSTICS", "1")
    env.setdefault("FT_ENABLE_INSTRUMENTATION", "1")
    env.setdefault("FT_INSTRUMENTATION_OUTPUT_DIR", str(instrumentation_dir))

    print(f"[{name}] cwd: {cwd}", flush=True)  # noqa: T201
    print(f"[{name}] command: {' '.join(command)}", flush=True)  # noqa: T201
    print(f"[{name}] combined log: {log_path}", flush=True)  # noqa: T201
    print(f"[{name}] instrumentation dir: {instrumentation_dir}", flush=True)  # noqa: T201
    _check_available_ram(f"{name} pre-start", require=True)

    with log_path.open("w") as log_file:
        log_lock = threading.Lock()
        process = subprocess.Popen(  # noqa: S603 - command is controlled by this test
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Subprocess pipes were not created")
        stream_threads = [
            threading.Thread(target=_tee_stream, args=(process.stdout, sys.stdout, log_file, log_lock)),
            threading.Thread(target=_tee_stream, args=(process.stderr, sys.stderr, log_file, log_lock)),
        ]
        for thread in stream_threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            for thread in stream_threads:
                thread.join()
            msg = (
                f"Command timed out after {timeout_s}s: {command}\n"
                f"combined log: {log_path}\n"
                f"combined log tail:\n{_tail(log_path)}"
            )
            raise TimeoutError(msg) from exc
        for thread in stream_threads:
            thread.join()

    print(f"[{name}] exited with return code {returncode}", flush=True)  # noqa: T201
    _check_available_ram(f"{name} post-exit", require=False)
    output = log_path.read_text(errors="replace")
    if returncode != 0:
        msg = (
            f"Command failed with return code {returncode}: {command}\n"
            f"combined log: {log_path}\n"
            f"combined log tail:\n{_tail(log_path)}"
        )
        raise RuntimeError(msg)
    return output


def _assert_output_file(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Expected output file was not created: {path}")
    if path.stat().st_size == 0:
        raise AssertionError(f"Expected output file is empty: {path}")


def test_quickstart_command(work_root: Path, timeout_s: float) -> None:
    cwd = work_root / "quickstart"
    cwd.mkdir(parents=True, exist_ok=True)
    output = cwd / "wan-t2v.mp4"
    output.unlink(missing_ok=True)

    command = [
        sys.executable,
        str(QUICKSTART_SCRIPT),
    ]
    _run_example_command("quickstart", command, cwd, timeout_s)

    _assert_output_file(output)


def test_profile_and_infer_commands(work_root: Path, timeout_s: float) -> None:
    cwd = work_root / "profile-reuse"
    cwd.mkdir(parents=True, exist_ok=True)
    profile_dir = cwd / "wan_profile"
    output = cwd / "infer.mp4"
    output.unlink(missing_ok=True)

    profile_command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--profile-dir",
        str(profile_dir),
    ]
    profile_output = _run_example_command("profile", profile_command, cwd, timeout_s)
    if "Profiles saved to" not in profile_output:
        raise AssertionError(f"Profile command did not report saved profiles:\n{profile_output}")

    infer_command = [
        sys.executable,
        str(INFER_SCRIPT),
        "--prompt",
        TEST_PROMPT,
        "--negative-prompt",
        "",
        "--num-frames",
        "1",
        "--profile-dir",
        str(profile_dir),
        "--output",
        str(output),
    ]
    infer_output = _run_example_command("infer", infer_command, cwd, timeout_s)

    _assert_output_file(output)
    if "Video saved to" not in infer_output:
        raise AssertionError(f"Infer command did not report output path:\n{infer_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real diffusers example integration tests")
    parser.add_argument("--timeout-s", default=7200.0, type=float, help="Timeout per example command")
    parser.add_argument("--work-dir", default=None, help="Directory for logs and generated artifacts")
    args = parser.parse_args()

    work_root = _resolve_work_root(args.work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {work_root}", flush=True)  # noqa: T201
    test_quickstart_command(work_root, args.timeout_s)
    test_profile_and_infer_commands(work_root, args.timeout_s)


if __name__ == "__main__":
    main()
