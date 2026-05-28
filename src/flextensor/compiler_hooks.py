# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiler integration hooks for FlexTensor runtime boundaries."""

import gc
from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch

_dynamo: Any | None
try:
    import torch._dynamo as _imported_dynamo
except ImportError:  # pragma: no cover - torch built without dynamo
    _dynamo = None
else:
    _dynamo = _imported_dynamo

F = TypeVar("F", bound=Callable[..., object])


def disable(func: F) -> F:
    """Disable torch compiler tracing for ``func`` when the public API exists."""
    compiler = getattr(torch, "compiler", None)
    compiler_disable = getattr(compiler, "disable", None) if compiler is not None else None
    if compiler_disable is not None:
        return cast("F", compiler_disable(func))
    dynamo_disable = getattr(_dynamo, "disable", None)
    if dynamo_disable is None:
        return func
    return cast("F", dynamo_disable(func))


def is_compiling() -> bool:
    """Return whether PyTorch is currently tracing/compiling user code."""
    compiler = getattr(torch, "compiler", None)
    compiler_is_compiling = getattr(compiler, "is_compiling", None) if compiler is not None else None
    if compiler_is_compiling is not None and compiler_is_compiling():
        return True
    dynamo_is_compiling = getattr(_dynamo, "is_compiling", None)
    return bool(dynamo_is_compiling is not None and dynamo_is_compiling())


def graph_break() -> None:
    """Insert a Dynamo graph break when Dynamo is available."""
    if _dynamo is None:
        return
    _dynamo.graph_break()


def find_torch_compile_wrapper(target: object) -> object | None:
    """Return a ``torch.compile`` OptimizedModule wrapper around ``target``."""
    if _dynamo is None:
        return None
    eval_frame = getattr(_dynamo, "eval_frame", None)
    optimized_cls = getattr(eval_frame, "OptimizedModule", None) if eval_frame is not None else None
    if optimized_cls is None:
        return None
    for obj in gc.get_objects():
        if type(obj) is not optimized_cls:
            continue
        try:
            orig = obj._orig_mod  # noqa: SLF001 - Dynamo exposes the wrapped module here
        except Exception:  # noqa: S112 - skip partially initialized Dynamo internals
            continue
        if orig is target:
            return cast("object", obj)
    return None
