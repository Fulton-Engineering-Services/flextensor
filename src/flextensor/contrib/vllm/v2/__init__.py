# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose the FlexTensor vLLM integration v2 worker without importing vLLM eagerly."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    FlexTensorOffloadWorker: Any

__all__ = ["FlexTensorOffloadWorker"]


def __getattr__(name: str) -> Any:
    if name == "FlexTensorOffloadWorker":
        from flextensor.contrib.vllm.v2.worker import FlexTensorOffloadWorker

        return FlexTensorOffloadWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
