# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy protocols and result types for FlexTensor offloading strategies.

This module provides the base Protocol and data types used by all offloading strategies.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from flextensor.collectors import LayerStatistics, TensorStatistics

# =============================================================================
# Exceptions
# =============================================================================


class StrategyComputeError(Exception):
    """Raised when a strategy cannot produce a valid result for the given inputs.

    This covers recoverable, expected failures such as insufficient layers for
    pipelining or missing memory-transfer data.  :class:`AdaptiveStrategy`
    catches this exception to skip a failing candidate and try the next one.

    Programming errors (``TypeError``, ``AttributeError``, etc.) should **not**
    be wrapped in this exception — they must propagate so that bugs surface
    during development.
    """


# =============================================================================
# Strategy Result Types
# =============================================================================


@dataclass
class BlockStrategyData:
    """Block allocation data computed by strategies.

    This contains all data needed for block-based loaders to execute
    pipelined CPU→GPU transfers.
    """

    label_to_size_map: dict[str, int]
    """Maps layer label to total size of tensors to offload."""

    allocation_ordered: dict[int, list[str]]
    """Maps block_id to ordered list of layer labels assigned to it."""

    block_sizes: dict[int, int] | list[int]
    """Size of each block in bytes (max layer size per block)."""

    label_to_block_id: dict[str, int]
    """Maps layer label to assigned block index."""

    transfer_to_compute_map: dict[str, str]
    """Maps transfer layer label to compute layer label for pipelining."""


@dataclass
class StrategyResult:
    """Result from strategy.compute().

    All strategies return this structure for consistency.
    TensorManager uses the appropriate fields based on loader_type.
    """

    strategy_map: dict[str, list[TensorStatistics]]
    """Maps layer label to list of tensors to offload."""

    block_data: BlockStrategyData | None = None
    """Block allocation data. None if n_blocks=0 or not computed."""


# =============================================================================
# Strategy Protocol
# =============================================================================


@runtime_checkable
class Strategy(Protocol):
    """Protocol for offload strategies.

    All offload strategies must implement this protocol to be usable
    with the OffloadConfig and TensorManager.
    """

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
    ) -> StrategyResult:
        """Compute offload strategy.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Optional dict mapping tensor size (bytes) to transfer time (ms).

        Returns:
            StrategyResult containing strategy_map and optional block_data.
        """
        ...
