# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict

import numpy as np
from pydantic import BaseModel, ConfigDict


class TensorStatistics(BaseModel):
    """Statistics for tensor operations including size and load time metrics."""

    model_config = ConfigDict(frozen=True)

    tensor_id: int
    name: str
    size_bytes: int
    load_time_ms: float


# TODO: change name
class LayerStatistics(BaseModel):
    """Statistics for layer operations including tensors and duration metrics."""

    model_config = ConfigDict(frozen=True)

    label: str
    tensors: list[TensorStatistics]
    duration: float


class LayerStatisticsCollector:
    """Collector for layer statistics with Pydantic model integration."""

    def __init__(self, tensor_statistics_map: dict[int, TensorStatistics] | None = None) -> None:
        self.tensor_statistics_map = tensor_statistics_map or {}
        self.stats_map: OrderedDict[str, LayerStatistics] = OrderedDict()

    def add_all(self, label: str, tensor_ids: list[int] | set[int], duration: float) -> None:
        for tensor_id in tensor_ids:
            self.add(label, tensor_id, duration)

    def add(self, label: str, tensor_id: int, duration: float) -> None:
        """Add tensor statistics for a given label."""
        if tensor_id in self.tensor_statistics_map:
            tensor_statistics = self.tensor_statistics_map[tensor_id]
            stats = LayerStatistics(label=label, tensors=[], duration=duration)
            if label in self.stats_map:
                stats = self.stats_map[label]
            stats.tensors.append(tensor_statistics)
            self.stats_map[label] = stats

    def get_layer_stats(self) -> list[LayerStatistics]:
        """Get all layer statistics as a list."""
        layer_stats = []
        for _name, stats in self.stats_map.items():
            layer_stats.append(stats)
        return layer_stats


class IterativeTensorStatistics(BaseModel):
    """Statistics for tensor operations including size and load time metrics."""

    model_config = ConfigDict(frozen=True)

    tensor_id: int
    name: str
    size_bytes: int
    load_time_ms: float


class IterativeLayerStatistics(BaseModel):
    """Statistics for layer operations including tensors and duration metrics."""

    model_config = ConfigDict(frozen=True)

    label: str
    tensor_ids: list[int] | set[int]
    duration: float


class IterativeLayerStatisticsCollector:
    """Collector for layer statistics with Pydantic model integration."""

    def __init__(self) -> None:
        self.tensor_measurements: dict[str, list[list[int] | set[int]]] = {}
        self.duration_measurements: dict[str, list[float]] = {}

    def add_tensors(self, label: str, tensor_ids: list[int] | set[int]) -> None:
        if label not in self.tensor_measurements:
            self.tensor_measurements[label] = []
        self.tensor_measurements[label].append(tensor_ids)

    def add_duration(self, label: str, duration_ms: float) -> None:
        if label not in self.duration_measurements:
            self.duration_measurements[label] = []
        self.duration_measurements[label].append(duration_ms)

    def clear_duration_measurements(self) -> None:
        self.duration_measurements = {}

    def add_all(self, label: str, tensor_ids: list[int] | set[int], duration_ms: float) -> None:
        self.add_tensors(label, tensor_ids)
        self.add_duration(label, duration_ms)

    def get_median_duration_ms(self) -> dict[str, float]:
        median_duration_ms: dict[str, float] = {}
        for label in self.tensor_measurements:
            median_duration_ms[label] = np.median(self.duration_measurements[label])
        return median_duration_ms

    def get_min_duration_ms(self) -> dict[str, float]:
        min_duration_ms: dict[str, float] = {}
        for label in self.tensor_measurements:
            min_duration_ms[label] = np.min(self.duration_measurements[label])
        return min_duration_ms

    def get_union_tensor_ids(self) -> dict[str, set[int]]:
        union_tensor_ids: dict[str, set[int]] = {}
        for label in self.tensor_measurements:
            union_tensor_ids[label] = set.union(*self.tensor_measurements[label])
        return union_tensor_ids

    def get_layer_stats(self) -> list[IterativeLayerStatistics]:
        stats: list[IterativeLayerStatistics] = []
        durations_map = self.get_median_duration_ms()
        union_tensor_ids = self.get_union_tensor_ids()
        for label in union_tensor_ids:
            stats.append(
                IterativeLayerStatistics(
                    label=label,
                    tensor_ids=union_tensor_ids[label],
                    duration=durations_map[label],
                ),
            )
        return stats


class IterativeLayerStatisticsFilter:
    """Filter for IterativeLayerStatistics based on tensor IDs."""

    @staticmethod
    def filter_by_tensor_ids(
        stats_list: list[IterativeLayerStatistics],
        tensor_ids_set: set[int],
    ) -> list[IterativeLayerStatistics]:
        """
        Filter a list of IterativeLayerStatistics by filtering the tensor_ids within each object
        to keep only those tensor IDs present in the given tensor_ids_set.

        Args:
            stats_list: List of IterativeLayerStatistics to filter
            tensor_ids_set: Set of tensor IDs to filter by

        Returns:
            New list of IterativeLayerStatistics with filtered tensor_ids. For lists,
            order and duplicates are preserved. For sets, intersection is used.
        """
        filtered_stats = []

        for stats in stats_list:
            # Filter tensor_ids based on their type while preserving structure
            if isinstance(stats.tensor_ids, list):
                # For lists, preserve order and duplicates by filtering each element
                filtered_tensor_ids = [tensor_id for tensor_id in stats.tensor_ids if tensor_id in tensor_ids_set]
            else:
                # For sets, use efficient set intersection
                filtered_tensor_ids = stats.tensor_ids & tensor_ids_set

            # Create new MultiLayerStatistics with filtered tensor_ids
            filtered_stats.append(
                IterativeLayerStatistics(
                    label=stats.label,
                    tensor_ids=filtered_tensor_ids,
                    duration=stats.duration,
                ),
            )

        return filtered_stats

    @staticmethod
    def filter_excluding_tensor_ids(
        stats_list: list[IterativeLayerStatistics],
        tensor_ids_set: set[int],
    ) -> list[IterativeLayerStatistics]:
        """
        Filter a list of IterativeLayerStatistics by filtering the tensor_ids within each object
        to keep only those tensor IDs NOT present in the given tensor_ids_set.

        Args:
            stats_list: List of IterativeLayerStatistics to filter
            tensor_ids_set: Set of tensor IDs to exclude

        Returns:
            New list of IterativeLayerStatistics with filtered tensor_ids. For lists,
            order and duplicates are preserved. For sets, set difference is used.
        """
        filtered_stats = []

        for stats in stats_list:
            # Filter tensor_ids based on their type while preserving structure
            if isinstance(stats.tensor_ids, list):
                # For lists, preserve order and duplicates by filtering each element
                filtered_tensor_ids = [tensor_id for tensor_id in stats.tensor_ids if tensor_id not in tensor_ids_set]
            else:
                # For sets, use efficient set difference
                filtered_tensor_ids = stats.tensor_ids - tensor_ids_set

            # Create new MultiLayerStatistics with filtered tensor_ids
            filtered_stats.append(
                IterativeLayerStatistics(
                    label=stats.label,
                    tensor_ids=filtered_tensor_ids,
                    duration=stats.duration,
                ),
            )

        return filtered_stats
