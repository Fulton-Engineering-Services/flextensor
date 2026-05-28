# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def test_compiler_hooks_module_exists() -> None:
    assert importlib.util.find_spec("flextensor.compiler_hooks") is not None


def test_disable_delegates_to_public_torch_compiler_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    fn = Mock()
    disabled_fn = Mock()
    compiler = SimpleNamespace(
        disable=Mock(return_value=disabled_fn),
    )

    monkeypatch.setattr(torch, "compiler", compiler)

    assert compiler_hooks.disable(fn) is disabled_fn
    compiler.disable.assert_called_once_with(fn)


def test_disable_is_noop_when_public_disable_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    fn = Mock()
    monkeypatch.setattr(torch, "compiler", SimpleNamespace())
    monkeypatch.setattr(compiler_hooks, "_dynamo", None)

    assert compiler_hooks.disable(fn) is fn


def test_disable_falls_back_to_dynamo_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    fn = Mock()
    disabled_fn = Mock()
    dynamo = SimpleNamespace(disable=Mock(return_value=disabled_fn))
    monkeypatch.setattr(torch, "compiler", SimpleNamespace())
    monkeypatch.setattr(compiler_hooks, "_dynamo", dynamo)

    assert compiler_hooks.disable(fn) is disabled_fn
    dynamo.disable.assert_called_once_with(fn)


def test_is_compiling_uses_public_torch_compiler_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    compiler = SimpleNamespace(is_compiling=Mock(return_value=True))
    monkeypatch.setattr(torch, "compiler", compiler)

    assert compiler_hooks.is_compiling() is True
    compiler.is_compiling.assert_called_once_with()


def test_is_compiling_falls_back_to_dynamo(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    dynamo = SimpleNamespace(is_compiling=Mock(return_value=True))
    monkeypatch.setattr(torch, "compiler", SimpleNamespace())
    monkeypatch.setattr(compiler_hooks, "_dynamo", dynamo)

    assert compiler_hooks.is_compiling() is True
    dynamo.is_compiling.assert_called_once_with()


def test_graph_break_delegates_to_dynamo(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    dynamo = SimpleNamespace(graph_break=Mock())
    monkeypatch.setattr(compiler_hooks, "_dynamo", dynamo)

    compiler_hooks.graph_break()

    dynamo.graph_break.assert_called_once_with()


def test_graph_break_is_noop_when_dynamo_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    monkeypatch.setattr(compiler_hooks, "_dynamo", None)

    compiler_hooks.graph_break()


def test_preallocated_block_loader_boundaries_are_compiler_disabled() -> None:
    from flextensor.loaders import (
        PreallocatedBatchTransferTensorLoader,
        PreallocatedBatchTransferTensorLoaderReordered,
    )

    for loader_cls in (PreallocatedBatchTransferTensorLoader, PreallocatedBatchTransferTensorLoaderReordered):
        for method_name in ("enter", "exit"):
            method = getattr(loader_cls, method_name)
            assert getattr(method, "_torchdynamo_disable", False) is True


def test_find_torch_compile_wrapper_returns_optimized_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    @dataclass
    class OptimizedModule:
        _orig_mod: object

    target = object()
    wrapper = OptimizedModule(target)
    dynamo = SimpleNamespace(eval_frame=SimpleNamespace(OptimizedModule=OptimizedModule))
    monkeypatch.setattr(compiler_hooks, "_dynamo", dynamo)

    assert compiler_hooks.find_torch_compile_wrapper(target) is wrapper


def test_find_torch_compile_wrapper_returns_none_without_dynamo(monkeypatch: pytest.MonkeyPatch) -> None:
    from flextensor import compiler_hooks

    monkeypatch.setattr(compiler_hooks, "_dynamo", None)

    assert compiler_hooks.find_torch_compile_wrapper(object()) is None
