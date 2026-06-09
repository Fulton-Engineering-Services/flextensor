# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Python-owned integration GPU requirements."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration import conftest as integration_conftest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_require_nvidia_gpu_fails_when_nvidia_smi_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(integration_conftest.shutil, "which", lambda _name: None)

    with pytest.raises(pytest.fail.Exception, match="nvidia-smi not found"):
        integration_conftest.require_integration_nvidia_gpu.__wrapped__()


def test_require_nvidia_gpu_prints_summary_pcie_and_topology(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> integration_conftest.subprocess.CompletedProcess[str]:
        assert check is False
        assert capture_output is True
        assert text is True
        assert timeout == 15
        calls.append(tuple(args))
        return integration_conftest.subprocess.CompletedProcess(
            args,
            0,
            stdout=f"output for {' '.join(args)}",
            stderr="",
        )

    monkeypatch.setattr(integration_conftest.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(integration_conftest.subprocess, "run", fake_run)

    integration_conftest.require_integration_nvidia_gpu.__wrapped__()

    assert calls == [
        ("nvidia-smi",),
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,pcie.link.gen.gpucurrent,pcie.link.gen.max,"
            "pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max,"
            "driver_version",
            "--format=csv",
        ),
        ("nvidia-smi", "topo", "-m"),
    ]
    output = capsys.readouterr().out
    assert "=== nvidia-smi ===" in output
    assert "=== nvidia-smi PCIe details ===" in output
    assert "=== nvidia-smi topo -m ===" in output


def test_integration_shell_wrappers_do_not_call_shell_gpu_requirement() -> None:
    offenders = []
    for path in sorted((REPO_ROOT / "tests" / "integration").glob("L*_*/test.sh")):
        content = path.read_text()
        if "gpu_diagnostics.sh" in content or "require_nvidia_gpu" in content or "nvidia-smi >/dev/null" in content:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
