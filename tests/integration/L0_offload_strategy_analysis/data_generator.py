# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data generator for synthetic model data used in offload strategy testing.

This module provides functions to generate realistic model data based on
statistical distributions from real model profiling (e.g., DeepSeek R1).

Usage:
    from data_generator import generate_deepseek_r1_data, load_data

    # Generate synthetic data with default distributions
    layer_stats = generate_deepseek_r1_data(seed=42)

    # Load data (from file if available, otherwise generate)
    layer_stats, memory_stats, interpolator = load_data(use_file=False)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator

# Path to test data files
TEST_DATA_DIR = Path(__file__).parent
# Use minimal file (filtered to essential fields only) - 60% smaller
TENSOR_MANAGER_STATE_FILE = TEST_DATA_DIR / "tensor_manager_state_minimal.json"
# Fallback to full file if minimal doesn't exist
TENSOR_MANAGER_STATE_FILE_FULL = TEST_DATA_DIR / "tensor_manager_state.json"
MEMORY_STATISTICS_FILE = TEST_DATA_DIR / "memory_statistics.json"


# Default memory transfer statistics (based on real GPU benchmarks)
# Maps tensor size in bytes -> transfer time in milliseconds
DEFAULT_MEMORY_STATS: dict[int, float] = {
    536870912: 4.30,  # 512MB -> 4.3ms
    1073741824: 8.57,  # 1GB -> 8.57ms
    2147483648: 11.91,  # 2GB -> 11.91ms
    4294967296: 30.75,  # 4GB -> 30.75ms
}


def scale_memory_stats(memory_stats: dict[int, float], ratio: float) -> dict[int, float]:
    """Scale transfer bandwidth by a ratio.

    Args:
        memory_stats: Base transfer statistics mapping bytes -> duration_ms.
        ratio: Speed ratio where 1.0 = original speed, 0.1 = 10% speed (10x slower).

    Returns:
        New dict with transfer times divided by ratio.
    """
    if ratio <= 0:
        msg = f"ratio must be positive, got {ratio}"
        raise ValueError(msg)
    return {size: time_ms / ratio for size, time_ms in memory_stats.items()}


def _interpolate_transfer_time(size_bytes: int) -> float:
    """Estimate transfer time using linear interpolation from benchmark data.

    Uses DEFAULT_MEMORY_STATS to interpolate transfer time for any tensor size.
    This provides realistic transfer times consistent with GPU benchmarks (~100+ GB/s).

    Args:
        size_bytes: Size of tensor in bytes.

    Returns:
        Estimated transfer time in milliseconds.
    """
    if size_bytes <= 0:
        return 0.0

    # Sort benchmark points by size
    sizes = sorted(DEFAULT_MEMORY_STATS.keys())
    times = [DEFAULT_MEMORY_STATS[s] for s in sizes]

    # Handle edge cases
    if size_bytes <= sizes[0]:
        # Extrapolate from first point (linear from origin)
        rate = sizes[0] / times[0]  # bytes per ms
        return size_bytes / rate
    if size_bytes >= sizes[-1]:
        # Extrapolate from last two points
        rate = (sizes[-1] - sizes[-2]) / (times[-1] - times[-2])
        return times[-1] + (size_bytes - sizes[-1]) / rate

    # Linear interpolation between two closest points
    for i in range(len(sizes) - 1):
        if sizes[i] <= size_bytes <= sizes[i + 1]:
            # Interpolate between sizes[i] and sizes[i+1]
            ratio = (size_bytes - sizes[i]) / (sizes[i + 1] - sizes[i])
            return times[i] + ratio * (times[i + 1] - times[i])

    # Fallback (should not reach here)
    return size_bytes / (100 * 1024**3) * 1000  # ~100 GB/s


@dataclass
class LayerDistribution:
    """Statistical distribution parameters for layer generation.

    Attributes:
        size_mean_gb: Mean layer size in GB.
        size_stddev_gb: Standard deviation of layer size in GB.
        duration_mean_ms: Mean compute duration in milliseconds.
        duration_stddev_ms: Standard deviation of compute duration.
        num_tensors_mean: Mean number of tensors per layer.
        num_tensors_stddev: Standard deviation of tensor count.
    """

    size_mean_gb: float
    size_stddev_gb: float
    duration_mean_ms: float
    duration_stddev_ms: float
    num_tensors_mean: int
    num_tensors_stddev: int


# Default distributions based on real DeepSeek R1 profiling data
DEEPSEEK_R1_DISTRIBUTIONS: dict[str, LayerDistribution] = {
    "embed": LayerDistribution(
        size_mean_gb=1.73,
        size_stddev_gb=0.0,  # Single embedding layer, no variance
        duration_mean_ms=0.23,
        duration_stddev_ms=0.05,
        num_tensors_mean=1,
        num_tensors_stddev=0,
    ),
    "pre_moe": LayerDistribution(
        size_mean_gb=0.54,
        size_stddev_gb=0.02,
        duration_mean_ms=110.0,
        duration_stddev_ms=5.0,
        num_tensors_mean=20,
        num_tensors_stddev=2,
    ),
    "moe": LayerDistribution(
        size_mean_gb=10.72,
        size_stddev_gb=0.15,
        duration_mean_ms=227.0,
        duration_stddev_ms=10.0,
        num_tensors_mean=1558,
        num_tensors_stddev=20,
    ),
    "normalize": LayerDistribution(
        size_mean_gb=1.73,
        size_stddev_gb=0.0,
        duration_mean_ms=0.4,
        duration_stddev_ms=0.1,
        num_tensors_mean=2,
        num_tensors_stddev=0,
    ),
}


def _sample_positive(mean: float, stddev: float, min_val: float = 0.0) -> float:
    """Sample from normal distribution, clamped to positive values."""
    if stddev <= 0:
        return mean
    value = random.gauss(mean, stddev)
    return max(min_val, value)


def _sample_positive_int(mean: int, stddev: int, min_val: int = 1) -> int:
    """Sample integer from normal distribution, clamped to minimum."""
    if stddev <= 0:
        return mean
    value = round(random.gauss(mean, stddev))
    return max(min_val, value)


def _create_tensors(
    num_tensors: int,
    total_size_bytes: int,
    layer_name: str,
    tensor_id_start: int,
) -> tuple[list[TensorStatistics], int]:
    """Create tensors for a layer with randomized size distribution.

    Args:
        num_tensors: Number of tensors to create.
        total_size_bytes: Total size of all tensors combined.
        layer_name: Name prefix for tensor names.
        tensor_id_start: Starting tensor ID.

    Returns:
        Tuple of (list of tensors, next tensor ID).
    """
    tensors = []
    tensor_id = tensor_id_start

    if num_tensors <= 0:
        return tensors, tensor_id

    # Generate random proportions for tensor sizes
    proportions = [random.random() for _ in range(num_tensors)]  # noqa: S311
    total_prop = sum(proportions)
    proportions = [p / total_prop for p in proportions]

    for i, prop in enumerate(proportions):
        size = max(1, int(total_size_bytes * prop))
        # Estimate load time using interpolated benchmark data (~100+ GB/s)
        load_time_ms = _interpolate_transfer_time(size)
        tensors.append(
            TensorStatistics(
                tensor_id=tensor_id,
                name=f"{layer_name}.tensor_{i}",
                size_bytes=size,
                load_time_ms=load_time_ms,
            )
        )
        tensor_id += 1

    return tensors, tensor_id


def generate_deepseek_r1_data(
    num_layers: int = 61,
    pre_moe_layers: int = 3,
    distributions: dict[str, LayerDistribution] | None = None,
    seed: int | None = None,
    gap_layers: list[int] | None = None,
) -> list[LayerStatistics]:
    """Generate synthetic DeepSeek R1-like model data with statistical variation.

    DeepSeek R1 structure:
    - embed layer: ~1.73 GB, 1 tensor, very fast (~0.23ms)
    - pre-MoE layers (0-2): ~0.54 GB each, ~20 tensors, ~110ms
    - MoE layers (3-60): ~10.72 GB each, ~1558 tensors, ~227ms
    - normalize layer: ~1.73 GB, 2 tensors, very fast (~0.4ms)

    Data is generated using normal distributions with the specified mean and stddev
    for each layer type, allowing for realistic variation in test data.

    Args:
        num_layers: Total number of transformer layers (default: 61 for layers 0-60).
        pre_moe_layers: Number of pre-MoE layers (default: 3).
        distributions: Custom distributions for each layer type. If None, uses
            DEEPSEEK_R1_DISTRIBUTIONS based on real profiling data.
        seed: Random seed for reproducible data generation. If None, uses random seed.
        gap_layers: Layer indices (0-based within transformer layers) that should have
            no tensors, simulating non-offloadable layers. These layers retain compute
            duration but have empty tensor lists. Useful for testing gap-aware strategies.

    Returns:
        List of LayerStatistics for the generated model.

    Example:
        >>> # Generate reproducible test data
        >>> layers = generate_deepseek_r1_data(num_layers=10, seed=42)
        >>> len(layers)
        12  # embed + 10 layers + normalize

        >>> # Generate data with gap layers (layers 5 and 10 have no tensors)
        >>> layers = generate_deepseek_r1_data(num_layers=20, gap_layers=[5, 10], seed=42)

        >>> # Custom distributions for smaller model
        >>> custom_dist = {
        ...     "embed": LayerDistribution(0.5, 0.0, 0.1, 0.01, 1, 0),
        ...     "pre_moe": LayerDistribution(0.2, 0.01, 50.0, 5.0, 10, 1),
        ...     "moe": LayerDistribution(2.0, 0.1, 100.0, 10.0, 100, 10),
        ...     "normalize": LayerDistribution(0.5, 0.0, 0.1, 0.01, 2, 0),
        ... }
        >>> layers = generate_deepseek_r1_data(distributions=custom_dist)
    """
    if seed is not None:
        random.seed(seed)

    if distributions is None:
        distributions = DEEPSEEK_R1_DISTRIBUTIONS

    layer_stats_list = []
    tensor_id = 1000000

    # Embed layer
    embed_dist = distributions["embed"]
    embed_size_bytes = int(_sample_positive(embed_dist.size_mean_gb, embed_dist.size_stddev_gb) * 1024**3)
    embed_duration = _sample_positive(embed_dist.duration_mean_ms, embed_dist.duration_stddev_ms, 0.01)
    embed_tensors = [
        TensorStatistics(
            tensor_id=tensor_id,
            name="embed.weight",
            size_bytes=embed_size_bytes,
            load_time_ms=_interpolate_transfer_time(embed_size_bytes),
        )
    ]
    tensor_id += 1
    layer_stats_list.append(LayerStatistics(label="embed", tensors=embed_tensors, duration=embed_duration))

    # Pre-MoE layers
    pre_moe_dist = distributions["pre_moe"]
    for i in range(pre_moe_layers):
        size_gb = _sample_positive(pre_moe_dist.size_mean_gb, pre_moe_dist.size_stddev_gb, 0.01)
        size_bytes = int(size_gb * 1024**3)
        duration = _sample_positive(pre_moe_dist.duration_mean_ms, pre_moe_dist.duration_stddev_ms, 0.1)
        num_tensors = _sample_positive_int(pre_moe_dist.num_tensors_mean, pre_moe_dist.num_tensors_stddev)
        tensors, tensor_id = _create_tensors(num_tensors, size_bytes, f"layer_{i}", tensor_id)
        layer_stats_list.append(LayerStatistics(label=str(i), tensors=tensors, duration=duration))

    # MoE layers
    moe_dist = distributions["moe"]
    for i in range(pre_moe_layers, num_layers):
        size_gb = _sample_positive(moe_dist.size_mean_gb, moe_dist.size_stddev_gb, 0.01)
        size_bytes = int(size_gb * 1024**3)
        duration = _sample_positive(moe_dist.duration_mean_ms, moe_dist.duration_stddev_ms, 0.1)
        num_tensors = _sample_positive_int(moe_dist.num_tensors_mean, moe_dist.num_tensors_stddev)
        tensors, tensor_id = _create_tensors(num_tensors, size_bytes, f"layer_{i}", tensor_id)
        layer_stats_list.append(LayerStatistics(label=str(i), tensors=tensors, duration=duration))

    # Normalize layer
    norm_dist = distributions["normalize"]
    norm_size_bytes = int(_sample_positive(norm_dist.size_mean_gb, norm_dist.size_stddev_gb) * 1024**3)
    norm_duration = _sample_positive(norm_dist.duration_mean_ms, norm_dist.duration_stddev_ms, 0.01)
    num_norm_tensors = _sample_positive_int(norm_dist.num_tensors_mean, norm_dist.num_tensors_stddev)
    normalize_tensors, tensor_id = _create_tensors(num_norm_tensors, norm_size_bytes, "normalize", tensor_id)
    layer_stats_list.append(LayerStatistics(label="normalize", tensors=normalize_tensors, duration=norm_duration))

    # Apply gap layers: replace with empty-tensor copies for specified indices
    if gap_layers:
        gap_set = set(gap_layers)
        layer_stats_list = [
            LayerStatistics(label=layer.label, tensors=[], duration=layer.duration)
            if layer.label.isdigit() and int(layer.label) in gap_set
            else layer
            for layer in layer_stats_list
        ]

    return layer_stats_list


def load_data(
    use_file: bool = True,
    num_layers: int = 61,
    pre_moe_layers: int = 3,
    seed: int | None = None,
    distributions: dict[str, LayerDistribution] | None = None,
    gap_layers: list[int] | None = None,
) -> tuple[list[LayerStatistics], dict[int, float], MemoryTransferInterpolator]:
    """Load or generate test data.

    This function provides a unified interface for loading real profiled data
    from JSON files or generating synthetic data for testing.

    Args:
        use_file: If True, load data from JSON files (if they exist).
            If False, always generate synthetic data.
        num_layers: Number of transformer layers for generated data (default: 61).
        pre_moe_layers: Number of pre-MoE layers for generated data (default: 3).
        seed: Random seed for reproducible generated data. If None, uses random seed.
        distributions: Custom distributions for generated data. If None, uses defaults.
        gap_layers: Transformer layer indices that should have no tensors (gaps).
            Only applies to generated data.

    Returns:
        Tuple of (layer_stats_list, memory_stats, interpolator)

    Example:
        >>> # Load from file (falls back to generated if file missing)
        >>> layers, mem_stats, interp = load_data()

        >>> # Always generate synthetic data
        >>> layers, mem_stats, interp = load_data(use_file=False, seed=42)

        >>> # Generate smaller model for quick tests
        >>> layers, mem_stats, interp = load_data(use_file=False, num_layers=10)
    """
    # Load memory statistics (small file, always use real data if available)
    if MEMORY_STATISTICS_FILE.exists():
        with MEMORY_STATISTICS_FILE.open() as f:
            memory_stats_raw = json.load(f)
            memory_stats = {int(k): v for k, v in memory_stats_raw.items()}
    else:
        memory_stats = DEFAULT_MEMORY_STATS.copy()

    # Try minimal file first, fall back to full file
    tensor_file = None
    if use_file:
        if TENSOR_MANAGER_STATE_FILE.exists():
            tensor_file = TENSOR_MANAGER_STATE_FILE
        elif TENSOR_MANAGER_STATE_FILE_FULL.exists():
            tensor_file = TENSOR_MANAGER_STATE_FILE_FULL

    if tensor_file:
        # Load from file
        with tensor_file.open() as f:
            data = json.load(f)

        # Parse layer statistics
        stats = data.get("stats", [])
        layer_stats_list = []

        for entry in stats:
            tensors = [
                TensorStatistics(
                    tensor_id=t["tensor_id"],
                    name=t["name"],
                    size_bytes=t.get("size_bytes", 0),
                    load_time_ms=t.get("load_time_ms", 0.0),
                )
                for t in entry["tensors"]
            ]
            layer_stats_list.append(
                LayerStatistics(
                    label=entry["label"],
                    tensors=tensors,
                    duration=entry["duration"],
                )
            )
    else:
        # Generate synthetic data with statistical variation
        layer_stats_list = generate_deepseek_r1_data(
            num_layers=num_layers,
            pre_moe_layers=pre_moe_layers,
            distributions=distributions,
            seed=seed,
            gap_layers=gap_layers,
        )

    # Create interpolator from memory statistics
    interpolator = MemoryTransferInterpolator(memory_stats)

    return layer_stats_list, memory_stats, interpolator


def print_data_summary(layer_stats_list: list[LayerStatistics]) -> None:
    """Print a summary of the generated/loaded data.

    Args:
        layer_stats_list: List of layer statistics to summarize.
    """
    total_size = sum(sum(t.size_bytes for t in layer.tensors) for layer in layer_stats_list)
    total_tensors = sum(len(layer.tensors) for layer in layer_stats_list)
    total_duration = sum(layer.duration for layer in layer_stats_list)

    print("Data Summary:")
    print(f"  Layers: {len(layer_stats_list)}")
    print(f"  Total tensors: {total_tensors}")
    print(f"  Total size: {total_size / 1024**3:.2f} GB")
    print(f"  Total compute time: {total_duration:.2f} ms")
    print()
    print("  Layer breakdown:")
    for layer in layer_stats_list[:5]:
        size_gb = sum(t.size_bytes for t in layer.tensors) / 1024**3
        print(f"    {layer.label}: {size_gb:.2f} GB, {layer.duration:.1f}ms, {len(layer.tensors)} tensors")
    if len(layer_stats_list) > 7:
        print("    ...")
    for layer in layer_stats_list[-2:]:
        size_gb = sum(t.size_bytes for t in layer.tensors) / 1024**3
        print(f"    {layer.label}: {size_gb:.2f} GB, {layer.duration:.1f}ms, {len(layer.tensors)} tensors")


if __name__ == "__main__":
    # Demo: Generate and display data
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic model data")
    parser.add_argument("--num-layers", type=int, default=61, help="Number of layers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--use-file", action="store_true", help="Load from file if available")
    args = parser.parse_args()

    print(f"Generating data with {args.num_layers} layers (seed={args.seed})...")
    layers, mem_stats, interpolator = load_data(
        use_file=args.use_file,
        num_layers=args.num_layers,
        seed=args.seed,
    )
    print_data_summary(layers)
