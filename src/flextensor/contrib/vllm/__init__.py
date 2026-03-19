# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlexTensor vLLM integration.

This module provides a custom vLLM worker and model loader with FlexTensor
offloading support. The custom loader initializes and loads weights on CPU,
then processes CUDA-only operations (FP8 quantization, MLA attention) on GPU
layer-by-layer before returning the model on CPU for FlexTensor management.

Usage:
    FT_ENABLED=1 vllm serve Qwen/Qwen2.5-7B \\
        --worker-cls flextensor.contrib.vllm.worker.FlexTensorOffloadWorker

Components:
    FlexTensorOffloadWorker: Custom worker that applies FlexTensor offloading
    FlexTensorModelLoader: Custom model loader with 2-phase CPU-first loading strategy
"""

__all__ = [  # noqa: F822 - provided by __getattr__
    "FlexTensorModelLoader",
    "FlexTensorOffloadWorker",
    "FlexTensorSnapshotWorker",
    "MemorySnapshotMixin",
    "SnapshotWorker",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy imports to avoid requiring vllm at package-init time."""
    if name == "FlexTensorModelLoader":
        from flextensor.contrib.vllm.loader import FlexTensorModelLoader

        return FlexTensorModelLoader
    if name == "FlexTensorOffloadWorker":
        from flextensor.contrib.vllm.worker import FlexTensorOffloadWorker

        return FlexTensorOffloadWorker
    if name in ("FlexTensorSnapshotWorker", "MemorySnapshotMixin", "SnapshotWorker"):
        from flextensor.contrib.vllm import snapshot

        return getattr(snapshot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
