# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation layer types and data structures for FlexTensor.

This module contains core data types used across the FlexTensor library.
These are Foundation layer types with no internal dependencies.
"""

from dataclasses import dataclass


@dataclass
class GPUMemoryUsage:
    """GPU memory usage by FlexTensor in inference mode.

    This dataclass provides a breakdown of GPU memory allocated by FlexTensor
    for tensor offloading operations. It includes memory used for transfer blocks
    and unmapped tensors that are moved to GPU.

    Attributes:
        blocks_bytes: Memory in bytes used by GPU transfer blocks for
            CPU-to-GPU tensor transfers.
        unmapped_tensors_bytes: Memory in bytes used by tensors that are
            not managed by the offload strategy and were moved directly to GPU.
        total_bytes: Total GPU memory in bytes used by FlexTensor.

    Examples:
        >>> import flextensor
        >>> from flextensor import OffloadConfig
        >>> om = flextensor.get_offload_manager()
        >>> config = OffloadConfig(include_patterns=["layers.*"])
        >>> model = om.offload(model, config=config)
        >>> # Run discovery and profiling iterations...
        >>> for _ in range(config.pre_inference_iters):
        ...     model(input)
        >>> # Now in inference mode, get memory usage
        >>> usage = om.get_gpu_memory_usage()
        >>> print(f"FlexTensor using {usage.total_mb:.2f} MB GPU memory")
    """

    blocks_bytes: int
    unmapped_tensors_bytes: int
    total_bytes: int

    @property
    def blocks_mb(self) -> float:
        """Memory used by GPU transfer blocks in megabytes."""
        return self.blocks_bytes / (1024 * 1024)

    @property
    def unmapped_tensors_mb(self) -> float:
        """Memory used by unmapped tensors in megabytes."""
        return self.unmapped_tensors_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        """Total GPU memory used in megabytes."""
        return self.total_bytes / (1024 * 1024)
