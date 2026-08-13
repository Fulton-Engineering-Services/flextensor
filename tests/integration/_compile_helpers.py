# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the four ``L0_*`` compile / CUDA-graph integration suites.

Consolidates the model definition, lifecycle driver, seeding, checksums and
CUDA-graph capture that were previously copy-pasted into each suite. Each
suite still owns its own constants (``NUM_LAYERS``, ``DIM``, …) and passes
them in, so the helpers stay agnostic of any individual suite's sizing.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from flextensor import OffloadConfig

DEFAULT_MODULE_PATTERNS: list[str] = ["input_projection", "layers.*", "output_projection"]


def set_seed(seed: int = 42) -> None:
    """Seed Python ``random``, NumPy and PyTorch (CPU + CUDA, if available)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def tensor_checksum(tensor: torch.Tensor) -> str:
    """Stable MD5 of a tensor's CPU-side bytes (bf16/fp16 promoted to fp32)."""
    t = tensor.detach().cpu().contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    return hashlib.md5(t.numpy().tobytes(), usedforsecurity=False).hexdigest()


# ---------------------------------------------------------------------------
# Model definition
#
# The L0 compile / CUDA-graph suites share a fixed-topology MoE-style model:
# stable shapes (no data-dependent control flow), one ``input_projection``,
# a ``ModuleList`` of expert-bearing layers, and one ``output_projection``.
# Module patterns target those names so the offload strategy can decide which
# layers stay on GPU.
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ExpertLayer(nn.Module):
    def __init__(self, num_experts: int, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, inter_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for expert in self.experts:
            out = out + expert(x)
        return out


class SimpleModel(nn.Module):
    """Fixed-topology MoE-style model used by every compile/CUDA-graph suite."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        inter_dim: int,
        num_experts: int = 2,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([ExpertLayer(num_experts, dim, inter_dim) for _ in range(num_layers)])
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_projection(x)
        return x


def make_simple_model(
    *,
    num_layers: int,
    dim: int,
    inter_dim: int,
    num_experts: int = 2,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> SimpleModel:
    """Construct a freshly-seeded ``SimpleModel`` on the requested device/dtype."""
    set_seed(seed)
    with torch.device("cpu"):
        model = SimpleModel(num_layers=num_layers, dim=dim, inter_dim=inter_dim, num_experts=num_experts)
    return model.to(dtype=dtype, device=torch.device(device)).eval()


def make_offload_config(
    *,
    discovery_iters: int,
    profiling_iters: int,
    feedback_iters: int,
    module_patterns: list[str] | None = None,
    transfer_mode: str = "allocation_block_transfer",
    num_blocks: int = 4,
    pinned_memory: bool = True,
) -> OffloadConfig:
    """Build the ``OffloadConfig`` shape every compile/CUDA-graph suite uses.

    ``skip_discovery=False`` is pinned explicitly because ``run_offload_lifecycle``
    below drives ``discovery_iters`` eager forwards through DISCOVERY; with
    ``skip_discovery=True`` those forwards would land in PROFILING
    instead and skew the phase accounting these suites assert.
    """
    return OffloadConfig(
        include_patterns=module_patterns if module_patterns is not None else DEFAULT_MODULE_PATTERNS,
        discovery_iters=discovery_iters * feedback_iters,
        profiling_iters=profiling_iters * feedback_iters,
        transfer_mode=transfer_mode,
        num_blocks=num_blocks,
        pinned_memory=pinned_memory,
        skip_discovery=False,
    )


def run_offload_lifecycle(
    proxy: nn.Module,
    x: torch.Tensor,
    *,
    discovery_iters: int,
    profiling_iters: int,
    feedback_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drive discovery → profiling → inference; return each phase's output."""
    with torch.no_grad():
        res_warmup = x
        for _ in range(discovery_iters):
            for _ in range(feedback_iters):
                res_warmup = proxy(res_warmup)

        res_profile: torch.Tensor | None = None
        for i in range(profiling_iters):
            res = x
            for _ in range(feedback_iters):
                res = proxy(res)
            if i == 0:
                res_profile = res

        res_inference = x
        for _ in range(feedback_iters):
            res_inference = proxy(res_inference)

    assert res_profile is not None, "profiling_iters must be >= 1"
    return res_warmup, res_profile, res_inference


def capture_cuda_graph(
    model: nn.Module,
    static_input: torch.Tensor,
    *,
    feedback_iters: int,
    warmup_runs: int = 3,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Capture a CUDA graph over ``feedback_iters`` invocations of ``model``.

    Args:
        model: Model (or proxy) to capture.
        static_input: Pre-allocated GPU tensor reused on every replay.
        feedback_iters: Number of sequential forward passes per capture.
        warmup_runs: Pre-capture runs to let the CUDA allocator settle.

    Returns:
        ``(graph, static_output)`` where ``static_output`` aliases the buffer
        the captured graph writes to. Replay with ``graph.replay()``.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(warmup_runs):
            out = static_input
            for _ in range(feedback_iters):
                out = model(out)
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s), torch.no_grad():
        static_output = static_input
        for _ in range(feedback_iters):
            static_output = model(static_output)

    return graph, static_output


# Re-export for callers that prefer a typed alias.
LifecycleFn = Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def compile_transformer_blocks(
    model: nn.Module,
    *,
    blocks_attr: str = "transformer_blocks",
    backend: str = "inductor",
    mode: str = "default",
    fullgraph: bool = False,
    trt_enabled_precisions: set[torch.dtype] | None = None,
) -> nn.Module:
    """Per-block ``torch.compile`` on ``model.<blocks_attr>`` (synthetic-DiT path).

    Mirrors the ``scope=per-block`` branch of the diffusers benchmark helper so
    each offloaded block is its own graph — slot-alias safe under rolling offload.
    """
    blocks = getattr(model, blocks_attr, None)
    if blocks is None:
        if backend == "inductor":
            return torch.compile(model, mode=mode, fullgraph=fullgraph)
        options = {"enabled_precisions": trt_enabled_precisions or {torch.float32}}
        return torch.compile(model, backend="torch_tensorrt", options=options)

    for idx in range(len(blocks)):
        if backend == "inductor":
            blocks[idx] = torch.compile(blocks[idx], mode=mode, fullgraph=fullgraph)
        elif backend in ("torch_tensorrt", "tensorrt"):
            options = {
                "enabled_precisions": trt_enabled_precisions or {torch.float32},
                "truncate_long_and_double": True,
                "min_block_size": 1,
            }
            blocks[idx] = torch.compile(blocks[idx], backend="torch_tensorrt", options=options)
        else:
            msg = f"unsupported compile backend: {backend!r}"
            raise ValueError(msg)
    return model
