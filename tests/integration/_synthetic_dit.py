# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small TRT-friendly synthetic transformer used by compiled-offload integration tests.

Extracted from ``examples/diffusers/compile-benchmark/synthetic_offload_check.py`` so
CI can validate offload + ``torch.compile`` / Torch-TensorRT without pulling in the
full diffusers benchmark harness.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SyntheticDiTBlock(nn.Module):
    """One transformer block: LayerNorm + SDPA + MLP (all TRT-friendly ops)."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_mult: int,
        *,
        compute_repeat: int = 0,
        elementwise_repeat: int = 0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.compute_repeat = compute_repeat
        self.elementwise_repeat = elementwise_repeat
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_mult * dim)
        self.fc2 = nn.Linear(mlp_mult * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(attn)

        n_mlp = 1 + self.compute_repeat
        mlp_scale = 1.0 / n_mlp
        for _ in range(n_mlp):
            h = self.norm2(x)
            x = x + mlp_scale * self.fc2(F.gelu(self.fc1(h)))

        if self.elementwise_repeat:
            base = x
            acc = torch.zeros_like(base)
            for k in range(self.elementwise_repeat):
                phase = 1.0 + 1e-3 * k
                acc = acc + torch.tanh(F.gelu(base * phase) * torch.sigmoid(base))
            x = base + (0.05 / self.elementwise_repeat) * acc
        return x


class SyntheticDiT(nn.Module):
    """Stack of :class:`SyntheticDiTBlock` modules under ``transformer_blocks``."""

    def __init__(
        self,
        *,
        layers: int,
        dim: int,
        heads: int,
        mlp_mult: int = 4,
        compute_repeat: int = 0,
        elementwise_repeat: int = 0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            msg = f"dim ({dim}) must be divisible by heads ({heads})"
            raise ValueError(msg)
        self.transformer_blocks = nn.ModuleList([
            SyntheticDiTBlock(
                dim,
                heads,
                mlp_mult,
                compute_repeat=compute_repeat,
                elementwise_repeat=elementwise_repeat,
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


def make_synthetic_dit(
    *,
    layers: int = 4,
    dim: int = 256,
    heads: int = 4,
    mlp_mult: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> SyntheticDiT:
    """Build a seeded ``SyntheticDiT`` on ``device`` in ``dtype``."""
    torch.manual_seed(seed)
    model = SyntheticDiT(layers=layers, dim=dim, heads=heads, mlp_mult=mlp_mult)
    return model.to(dtype=dtype, device=torch.device(device)).eval()


def make_synthetic_input(
    *,
    batch: int = 1,
    seq: int = 32,
    dim: int = 256,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
    seed: int = 1,
) -> torch.Tensor:
    """Random input tensor for :class:`SyntheticDiT`."""
    torch.manual_seed(seed)
    return torch.randn(batch, seq, dim, device=torch.device(device), dtype=dtype)
