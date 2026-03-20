# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for FlexTensor offloading strategies.

This module provides shared utility functions used by multiple strategy implementations.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.memory_block_planner import MemoryBlockPlanner

from .protocol import BlockStrategyData

logger = logging.getLogger(__name__)

_block_table_logger = logging.getLogger("flextensor.block_table")
if not _block_table_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _block_table_logger.addHandler(_handler)
    _block_table_logger.setLevel(logging.INFO)
    _block_table_logger.propagate = False


# =============================================================================
# Optimizer Early Stopping
# =============================================================================


def validate_memory_params(scale: float) -> None:
    """Validate memory parameters common to all strategies.

    Args:
        scale: Duration multiplier; must be positive.

    Raises:
        ValueError: If ``scale <= 0``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


class EarlyStopCallback:
    """Callback for scipy optimizers that stops after max_stall iterations without improvement.

    If the best objective value has not improved for ``max_stall`` consecutive
    iterations, the optimization is terminated by returning ``True`` from the
    callback.

    Compatible with both ``differential_evolution`` (callback signature: ``(xk, convergence)``)
    and ``dual_annealing`` (callback signature: ``(x, f, context)``).

    Args:
        max_stall: Maximum iterations without improvement before stopping.
        objective_func: The objective function to evaluate the current best solution.
            Used by ``differential_evolution`` which does not pass the function value
            to the callback. Ignored by ``dual_annealing`` which provides ``f`` directly.

    Example:
        >>> callback = EarlyStopCallback(max_stall=75, objective_func=my_func)
        >>> result = differential_evolution(my_func, bounds, callback=callback)

        >>> callback = EarlyStopCallback(max_stall=25, objective_func=my_func)
        >>> result = dual_annealing(my_func, bounds, callback=callback)
    """

    def __init__(
        self,
        max_stall: int,
        objective_func: Callable[[np.ndarray], float],
    ) -> None:
        if max_stall < 1:
            raise ValueError(f"max_stall must be >= 1, got {max_stall}")
        self.max_stall = max_stall
        self.objective_func = objective_func
        self._best_value = float("inf")
        self._stall_count = 0

    def __call__(self, xk: np.ndarray, f_or_convergence: float = 0.0, context: int | None = None) -> bool:
        """Evaluate and check for early stopping.

        Handles two callback signatures:
        - ``differential_evolution``: ``callback(xk, convergence)`` — re-evaluates objective.
        - ``dual_annealing``: ``callback(x, f, context)`` — uses provided ``f`` directly.

        Args:
            xk: Current best solution vector.
            f_or_convergence: Function value (dual_annealing) or convergence fraction (DE).
            context: If not None, indicates dual_annealing callback (context is an int).

        Returns:
            True if optimization should stop, False otherwise.
        """
        current = float(f_or_convergence) if context is not None else self.objective_func(xk)

        if current < self._best_value - 1e-10:
            self._best_value = current
            self._stall_count = 0
        else:
            self._stall_count += 1

        return self._stall_count >= self.max_stall


# =============================================================================
# Block Strategy Helpers
# =============================================================================


def _filter_zero_sizes_layers(ordered_map: OrderedDict[str, int]) -> OrderedDict[str, int]:
    """
    Filter out items with zero size from an OrderedDict while preserving the original order.

    Args:
        ordered_map: OrderedDict mapping layer names to their sizes in bytes

    Returns:
        New OrderedDict with zero-size items removed, preserving original order
    """
    filtered_map: OrderedDict[str, int] = OrderedDict()

    for key, value in ordered_map.items():
        if value != 0:
            filtered_map[key] = value

    return filtered_map


def compute_label_to_size_map(
    layer_stats: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
) -> OrderedDict[str, int]:
    """Compute mapping from layer labels to total tensor sizes.

    Args:
        layer_stats: Layer statistics.
        strategy_map: Strategy mapping layer labels to tensors.

    Returns:
        OrderedDict mapping layer labels to total sizes (excluding zero-size entries).
    """
    label_to_size_map: OrderedDict[str, int] = OrderedDict()
    for layer_statistic in layer_stats:
        label = layer_statistic.label
        size_bytes = 0
        if label in strategy_map:
            strategy = strategy_map[label]
            for tensor_info in strategy:
                size_bytes += tensor_info.size_bytes
        label_to_size_map[label] = size_bytes
    return _filter_zero_sizes_layers(label_to_size_map)


def _prepare_label_to_block_id(
    allocation_ordered: dict[int, list[str]],
) -> dict[str, int]:
    """Prepare mapping from layer labels to block IDs.

    Args:
        allocation_ordered: Mapping from block ID to list of layer labels.

    Returns:
        Mapping from layer label to block ID.
    """
    label_to_block_id: dict[str, int] = {}
    for block_id, labels in allocation_ordered.items():
        for label in labels:
            label_to_block_id[label] = block_id
    return label_to_block_id


def calculate_transfer_to_compute_map(
    layer_stats: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
) -> dict[str, str]:
    """Calculate mapping from transfer layer to compute layer.

    Args:
        layer_stats: Layer statistics.
        strategy_map: Strategy mapping layer labels to tensors.

    Returns:
        Mapping from transfer layer label to compute layer label.
    """
    tensor_id_to_compute_label_map: dict[int, str] = {}
    for layer_info in layer_stats:
        for tensor_info in layer_info.tensors:
            tensor_id_to_compute_label_map[tensor_info.tensor_id] = layer_info.label
    transfer_to_compute_map: dict[str, str] = {}
    for label, strategy in strategy_map.items():
        if len(strategy) > 0:
            tensor_info = strategy[0]
            compute_label = tensor_id_to_compute_label_map[tensor_info.tensor_id]
            transfer_to_compute_map[label] = compute_label
    return transfer_to_compute_map


def reverse_transfer_to_compute_map(t2c_map: dict[str, str]) -> dict[str, str]:
    """Reverse a transfer-to-compute map into a compute-to-transfer map.

    Args:
        t2c_map: Mapping from transfer layer label to compute layer label.

    Returns:
        Mapping from compute layer label to transfer layer label.

    Raises:
        ValueError: If two transfer labels map to the same compute label,
            which indicates a broken strategy pipeline.
    """
    result: dict[str, str] = {}
    for transfer, compute in t2c_map.items():
        if compute in result:
            raise ValueError(f"Compute label '{compute}' mapped from both '{result[compute]}' and '{transfer}'")
        result[compute] = transfer
    return result


def _prepare_block_strategy(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    blocks: int | None,
) -> tuple[
    OrderedDict[str, int],
    dict[int, list[str]],
    dict[int, int],
    dict[str, int],
    dict[str, str],
]:
    """Prepare block strategy data structures.

    Args:
        strategy_map: Strategy mapping layer labels to tensors.
        layer_stats: Layer statistics.
        blocks: Number of blocks, or None for automatic.

    Returns:
        Tuple of (label_to_size_map, allocation_ordered, block_sizes,
                  label_to_block_id, transfer_to_compute_map).

    Note:
        This does not support repeated tensors in layers. Consider checking for
        repeated tensors and throwing an exception, or assigning repeated tensors
        to a common block ID.
    """
    # FIXME: this do not support repeated tensors in layers!
    # TODO: check for repeated tensors and throw exception, or assign repeated tensors to common block id
    label_to_size_map = compute_label_to_size_map(layer_stats, strategy_map)

    memory_block_planner = MemoryBlockPlanner(label_to_size_map)

    if blocks is None:
        _, allocation_ordered = memory_block_planner.find_minimum_blocks()
    else:
        allocation_ordered = memory_block_planner.optimize_block_distribution(blocks)

    block_sizes = memory_block_planner.find_minimum_block_sizes(allocation_ordered)

    # prepare inverted map: label to block_id
    label_to_block_id = _prepare_label_to_block_id(allocation_ordered)
    transfer_to_compute_map = calculate_transfer_to_compute_map(layer_stats, strategy_map)
    return label_to_size_map, allocation_ordered, block_sizes, label_to_block_id, transfer_to_compute_map


def _build_block_data(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
    n_blocks: int,
) -> BlockStrategyData | None:
    """Build BlockStrategyData from strategy_map using _prepare_block_strategy.

    Args:
        strategy_map: Layer label to tensors mapping.
        layer_stats: Layer statistics.
        n_blocks: Number of blocks (0 to skip).

    Returns:
        BlockStrategyData or None if n_blocks <= 0.
    """
    if n_blocks <= 0:
        return None

    label_to_size_map, allocation_ordered, block_sizes, label_to_block_id, transfer_to_compute_map = (
        _prepare_block_strategy(strategy_map, layer_stats, n_blocks)
    )
    return BlockStrategyData(
        label_to_size_map=label_to_size_map,
        allocation_ordered=allocation_ordered,
        block_sizes=block_sizes,
        label_to_block_id=label_to_block_id,
        transfer_to_compute_map=transfer_to_compute_map,
    )


# =============================================================================
# Block table formatting
# =============================================================================


def _format_bytes(size_bytes: int | float) -> str:
    """Format byte size to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_block_table(  # noqa: C901
    layer_stats: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
    block_data: BlockStrategyData | None,
    strategy_name: str = "",
) -> str:
    """Format a block assignment table showing per-layer transfer and block info.

    Args:
        layer_stats: Layer statistics from profiling.
        strategy_map: Mapping of layer label to tensors transferred during that layer.
        block_data: Block strategy data (may be None for non-block strategies).
        strategy_name: Name of the strategy for the table header.

    Returns:
        Formatted multi-line string with the block assignment table.
    """
    if block_data is not None:
        layer_to_block: dict[str, int] = dict(block_data.label_to_block_id)
        block_sizes_map = block_data.block_sizes
        if isinstance(block_sizes_map, dict):
            max_block = max(block_sizes_map.keys(), default=-1) + 1
            block_sizes = [block_sizes_map.get(i, 0) for i in range(max_block)]
        else:
            block_sizes = list(block_sizes_map)
    else:
        layer_to_block = {}
        block_sizes = []

    # Build compute block map and reverse transfer map using
    # transfer_to_compute_map (accurate) or fall back to naive prev-label
    # heuristic when block_data is unavailable.
    compute_block: dict[str, int | None] = {}
    compute_to_transfer: dict[str, str] = {}
    if block_data is not None and block_data.transfer_to_compute_map:
        compute_to_transfer = reverse_transfer_to_compute_map(block_data.transfer_to_compute_map)
        for layer in layer_stats:
            transfer_label = compute_to_transfer.get(layer.label)
            compute_block[layer.label] = layer_to_block.get(transfer_label) if transfer_label else None
    else:
        for i, layer in enumerate(layer_stats):
            if i == 0:
                compute_block[layer.label] = None
            else:
                prev_label = layer_stats[i - 1].label
                compute_block[layer.label] = layer_to_block.get(prev_label)

    durations = [layer.duration for layer in layer_stats]

    lines: list[str] = []
    w = 130
    lines.append("")
    lines.append("=" * w)
    lines.append(f"BLOCK ASSIGNMENT: {strategy_name}")
    lines.append("=" * w)
    lines.append(
        f"{'Layer':<12} {'Layer Size':>10} {'Offload':>10} {'Transfer':>10} | "
        f"{'C.Blk':>5} {'T.Blk':>5} {'Blk Size':>10} | {'Pipeline':<30} {'Compute':>10}"
    )
    lines.append("-" * w)

    total_layer_size = 0
    total_offload = 0

    for i, layer in enumerate(layer_stats):
        layer_size = sum(t.size_bytes for t in layer.tensors)
        total_layer_size += layer_size
        t_blk = layer_to_block.get(layer.label)
        c_blk = compute_block[layer.label]

        t_str = str(t_blk) if t_blk is not None else "-"
        c_str = str(c_blk) if c_blk is not None else "-"

        bs = _format_bytes(block_sizes[t_blk]) if t_blk is not None and t_blk < len(block_sizes) else "-"

        dur_str = f"{layer.duration:.2f}ms"

        # Offload: tensors transferred FOR this layer's compute.
        # Use transfer_to_compute_map (via reverse lookup) when available,
        # otherwise fall back to prev-label heuristic.
        if compute_to_transfer:
            transfer_label = compute_to_transfer.get(layer.label)
            offload_tensors = strategy_map.get(transfer_label, []) if transfer_label else []
            offload_bytes = sum(t.size_bytes for t in offload_tensors)
        elif i > 0:
            prev_label = layer_stats[i - 1].label
            offload_tensors = strategy_map.get(prev_label, [])
            offload_bytes = sum(t.size_bytes for t in offload_tensors)
        else:
            offload_bytes = 0
        total_offload += offload_bytes
        offload_str = _format_bytes(offload_bytes) if offload_bytes > 0 else "-"

        # Transfer: data being transferred DURING this layer's execution (for next layer)
        transfer_tensors = strategy_map.get(layer.label, [])
        transfer_bytes = sum(t.size_bytes for t in transfer_tensors)
        transfer_str = _format_bytes(transfer_bytes) if transfer_bytes > 0 else "-"

        if c_blk is not None and t_blk is not None:
            pipe = f"read blk {c_blk}, fill blk {t_blk}"
        elif c_blk is not None:
            pipe = f"read blk {c_blk}"
        elif t_blk is not None:
            pipe = f"fill blk {t_blk} (1st transfer)"
        else:
            pipe = "on GPU"

        lines.append(
            f"{layer.label:<12} {_format_bytes(layer_size):>10} {offload_str:>10} {transfer_str:>10} | "
            f"{c_str:>5} {t_str:>5} {bs:>10} | {pipe:<30} {dur_str:>10}"
        )

    lines.append("-" * w)
    lines.append(
        f"Total: layer_size={_format_bytes(total_layer_size)}, "
        f"offload={_format_bytes(total_offload)}, "
        f"compute={sum(durations):.2f}ms"
    )
    if durations:
        lines.append(
            f"Compute: min={min(durations):.2f}ms, max={max(durations):.2f}ms, "
            f"avg={sum(durations) / len(durations):.2f}ms"
        )

    if block_sizes:
        lines.append("Block Sizes:")
        for idx, size in enumerate(block_sizes):
            if size > 0:
                transfer_labels = [ls.label for ls in layer_stats if layer_to_block.get(ls.label) == idx]
                compute_labels = [ls.label for ls in layer_stats if compute_block.get(ls.label) == idx]
                lines.append(
                    f"  Block {idx}: {_format_bytes(size):>10}  "
                    f"(transfers={len(transfer_labels)}: {', '.join(transfer_labels[:6])}"
                    f"{'...' if len(transfer_labels) > 6 else ''}"
                    f" | computes={len(compute_labels)}: {', '.join(compute_labels[:6])}"
                    f"{'...' if len(compute_labels) > 6 else ''})"
                )

    return "\n".join(lines)


def log_block_table(
    layer_stats: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
    block_data: BlockStrategyData | None,
    strategy_name: str = "",
) -> None:
    """Log the block assignment table at INFO level.

    Args:
        layer_stats: Layer statistics from profiling.
        strategy_map: Mapping of layer label to tensors transferred during that layer.
        block_data: Block strategy data (may be None for non-block strategies).
        strategy_name: Name of the strategy for the table header.
    """
    table = format_block_table(layer_stats, strategy_map, block_data, strategy_name)
    _block_table_logger.info(table)


def strategy_has_transfer_gaps(
    strategy_map: dict[str, list[TensorStatistics]],
    layer_stats: list[LayerStatistics],
) -> bool:
    """Check if a strategy has unused layer slots where transfer rearrangement would help.

    In the pipelined model, each of the first ``N-1`` layers can serve as a
    transfer slot (transferring the next layer's data during its compute).
    If some of those slots are unused (because the corresponding next layer
    has no offloaded tensors), ``rearrange_transfers`` can move other
    transfers into those gaps for better amortisation.

    Args:
        strategy_map: Mapping of layer labels to tensors being transferred
            during that layer's compute.
        layer_stats: Layer statistics for all layers in the model.

    Returns:
        ``True`` if there are free slots that rearrangement could exploit.
    """
    if len(layer_stats) <= 1 or not strategy_map:
        return False
    transfer_labels = {label for label, tensors in strategy_map.items() if tensors}
    max_possible_transfers = len(layer_stats) - 1
    return len(transfer_labels) < max_possible_transfers
