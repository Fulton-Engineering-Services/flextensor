# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ensure_drafter_on_device``.

See ``flextensor.contrib.vllm._drafter_device`` for the crash these tests
are guarding against.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from flextensor.contrib.vllm._drafter_device import ensure_drafter_on_device

_HELPER_LOGGER = "flextensor.contrib.vllm._drafter_device"


class _RecordingModel:
    """``nn.Module`` stand-in that records the device passed to ``.to()``."""

    def __init__(self) -> None:
        self.to_device: object = None

    def to(self, device: object) -> "_RecordingModel":
        self.to_device = device
        return self


class _NonMovableModel:
    """Stand-in with a non-callable ``.to`` attribute."""

    to: object = "not callable"


class TestEnsureDrafterOnDevice:
    def test_moves_drafter_model_to_device(self) -> None:
        """Happy path calls ``drafter.model.to(device)``."""
        model = _RecordingModel()
        runner = SimpleNamespace(drafter=SimpleNamespace(model=model))

        ensure_drafter_on_device(runner, "cuda:0")

        assert model.to_device == "cuda:0"

    def test_no_op_when_runner_has_no_drafter_attr(self) -> None:
        """Non-speculative runs have no ``drafter`` attribute."""
        runner = SimpleNamespace()

        ensure_drafter_on_device(runner, "cuda:0")

    def test_no_op_when_drafter_is_none(self) -> None:
        """``drafter`` may be ``None`` before the proposer attaches."""
        runner = SimpleNamespace(drafter=None)

        ensure_drafter_on_device(runner, "cuda:0")

    def test_no_op_when_drafter_has_no_model(self) -> None:
        """Proposer without a ``.model`` attribute is tolerated."""
        runner = SimpleNamespace(drafter=SimpleNamespace())

        ensure_drafter_on_device(runner, "cuda:0")

    def test_no_op_when_drafter_model_is_none(self) -> None:
        """``drafter.model`` may be ``None`` during proposer pre-init."""
        runner = SimpleNamespace(drafter=SimpleNamespace(model=None))

        ensure_drafter_on_device(runner, "cuda:0")

    def test_no_op_when_drafter_model_not_movable(self) -> None:
        """``drafter.model`` with a non-callable ``.to`` is tolerated."""
        runner = SimpleNamespace(drafter=SimpleNamespace(model=_NonMovableModel()))

        ensure_drafter_on_device(runner, "cuda:0")

    def test_exceptions_from_to_propagate(self) -> None:
        """Real ``.to()`` failures (OOM, invalid device) surface to the caller."""

        class _RaisingModel:
            def to(self, device: object) -> None:
                raise RuntimeError("synthetic device failure")

        runner = SimpleNamespace(drafter=SimpleNamespace(model=_RaisingModel()))

        with pytest.raises(RuntimeError, match="synthetic device failure"):
            ensure_drafter_on_device(runner, "cuda:0")


class TestLogging:
    """Logging contract for ``ensure_drafter_on_device``.

    If a future vLLM refactor renames the drafter attribute, the helper
    silently becomes a no-op and warmup crashes with the original CPU/CUDA
    mismatch. The ``WARNING`` records asserted here are the breadcrumb a
    debugger would otherwise be missing.
    """

    def test_logs_debug_on_happy_path(self, caplog: pytest.LogCaptureFixture) -> None:
        """Happy path emits a DEBUG record naming the target device."""
        model = _RecordingModel()
        runner = SimpleNamespace(drafter=SimpleNamespace(model=model))

        with caplog.at_level(logging.DEBUG, logger=_HELPER_LOGGER):
            ensure_drafter_on_device(runner, "cuda:0")

        debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("cuda:0" in m for m in debug_messages), f"no DEBUG record mentioning device: {debug_messages}"

    def test_debug_only_when_drafter_absent(self, caplog: pytest.LogCaptureFixture) -> None:
        """Non-speculative runs log at DEBUG only — never WARN."""
        runner = SimpleNamespace()

        with caplog.at_level(logging.DEBUG, logger=_HELPER_LOGGER):
            ensure_drafter_on_device(runner, "cuda:0")

        assert any(r.levelno == logging.DEBUG for r in caplog.records), "expected a DEBUG record"
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), "non-speculative runs must not WARN"

    def test_warns_when_drafter_model_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Drafter without a ``.model`` emits a WARNING."""
        runner = SimpleNamespace(drafter=SimpleNamespace())

        with caplog.at_level(logging.DEBUG, logger=_HELPER_LOGGER):
            ensure_drafter_on_device(runner, "cuda:0")

        assert any(r.levelno == logging.WARNING for r in caplog.records), (
            "expected a WARNING when drafter present but .model missing"
        )

    def test_warns_when_drafter_model_not_movable(self, caplog: pytest.LogCaptureFixture) -> None:
        """``drafter.model`` with a non-callable ``.to`` emits a WARNING."""
        runner = SimpleNamespace(drafter=SimpleNamespace(model=_NonMovableModel()))

        with caplog.at_level(logging.DEBUG, logger=_HELPER_LOGGER):
            ensure_drafter_on_device(runner, "cuda:0")

        assert any(r.levelno == logging.WARNING for r in caplog.records), (
            "expected a WARNING when drafter.model is not nn.Module-like"
        )


class TestLoadModelOrdering:
    """Structural guard on ``FlexTensorOffloadWorker.load_model`` call order.

    ``ensure_drafter_on_device`` must run before ``flextensor.offload()`` so
    that identity-shared drafter submodules are GPU-resident when FT
    installs forward patches. This test reads ``worker.py`` via AST so it
    runs without vLLM or a GPU.
    """

    def _load_model_ast(self) -> ast.FunctionDef:
        worker_path = Path(__file__).parents[4] / "src" / "flextensor" / "contrib" / "vllm" / "worker.py"
        tree = ast.parse(worker_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "FlexTensorOffloadWorker":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "load_model":
                        return item
        pytest.fail("FlexTensorOffloadWorker.load_model not found in worker.py")

    def test_ensure_drafter_on_device_called_before_flextensor_offload(self) -> None:
        load_model = self._load_model_ast()

        drafter_line: int | None = None
        offload_line: int | None = None
        for node in ast.walk(load_model):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "ensure_drafter_on_device":
                drafter_line = node.lineno if drafter_line is None else drafter_line
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "offload"
                and isinstance(func.value, ast.Name)
                and func.value.id == "flextensor"
            ):
                offload_line = node.lineno if offload_line is None else offload_line

        assert drafter_line is not None, "FlexTensorOffloadWorker.load_model must call ensure_drafter_on_device"
        assert offload_line is not None, "FlexTensorOffloadWorker.load_model must call flextensor.offload"
        assert drafter_line < offload_line, (
            f"ensure_drafter_on_device (line {drafter_line}) must precede "
            f"flextensor.offload (line {offload_line}): drafter submodules can "
            f"be identity-shared with the main model and must be GPU-resident "
            f"before FT installs forward patches."
        )
