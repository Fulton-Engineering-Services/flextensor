# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

import torch


class TraceTensor(torch.Tensor):
    def to(self, device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format):
        return super().to(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
            copy=copy,
            memory_format=memory_format,
        )

    @staticmethod
    def __new__(cls, data, name: str | None = None):
        return super().__new__(cls, data)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        res = super().__torch_function__(func, types, args, kwargs)
        if func.__name__[-1] == "_":
            return res
        if isinstance(res, TraceTensor):
            return torch.Tensor(res)
        return res


def wrap_trace_tensor(tensor):
    if isinstance(tensor, TraceTensor):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return TraceTensor(tensor)
    return tensor
