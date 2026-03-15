# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemoryBlockPlanner."""

from collections import OrderedDict

import pytest

from flextensor.memory_block_planner import MemoryBlockPlanner


def _make_layers(n: int) -> OrderedDict[str, int]:
    return OrderedDict((f"layer_{i}", 1024) for i in range(n))


class TestBuildAdjacencyGraph:
    """Verify _build_adjacency_graph produces a path graph (no circular edge)."""

    def test_three_layers_no_circular_edge(self):
        planner = MemoryBlockPlanner(_make_layers(3))
        graph = planner.adjacency_graph
        assert "layer_0" not in graph.get("layer_2", set())
        assert "layer_2" not in graph.get("layer_0", set())

    def test_five_layers_no_circular_edge(self):
        planner = MemoryBlockPlanner(_make_layers(5))
        graph = planner.adjacency_graph
        assert "layer_0" not in graph.get("layer_4", set())
        assert "layer_4" not in graph.get("layer_0", set())

    def test_five_layers_has_consecutive_edges(self):
        planner = MemoryBlockPlanner(_make_layers(5))
        graph = planner.adjacency_graph
        for i in range(4):
            assert f"layer_{i + 1}" in graph[f"layer_{i}"]
            assert f"layer_{i}" in graph[f"layer_{i + 1}"]


@pytest.mark.parametrize(
    ("n_layers", "expected_blocks"),
    [(0, 0), (1, 1), (2, 2), (3, 2), (5, 2)],
    ids=["empty", "single", "two", "three-odd", "five-odd"],
)
class TestFindMinimumBlocksSequential:
    def test_minimum_blocks(self, n_layers: int, expected_blocks: int):
        planner = MemoryBlockPlanner(_make_layers(n_layers))
        assert planner.find_minimum_blocks_sequential() == expected_blocks


class TestOptimizeBlockDistribution:
    def test_odd_layer_count_with_two_blocks(self):
        planner = MemoryBlockPlanner(_make_layers(3))
        blocks = planner.optimize_block_distribution(2)
        all_layers = {name for layers in blocks.values() for name in layers}
        assert all_layers == {"layer_0", "layer_1", "layer_2"}

    def test_five_layers_with_two_blocks_no_adjacent_in_same_block(self):
        planner = MemoryBlockPlanner(_make_layers(5))
        blocks = planner.optimize_block_distribution(2)
        for layers in blocks.values():
            names = set(layers)
            for name in layers:
                idx = int(name.split("_")[1])
                if idx > 0:
                    assert f"layer_{idx - 1}" not in names
                if idx < 4:
                    assert f"layer_{idx + 1}" not in names


class TestEdgeCases:
    def test_empty_input(self):
        planner = MemoryBlockPlanner(OrderedDict())
        assert planner.find_minimum_blocks_sequential() == 0
        assert planner.adjacency_graph == {}

    def test_single_layer(self):
        planner = MemoryBlockPlanner(OrderedDict([("only", 512)]))
        assert planner.find_minimum_blocks_sequential() == 1
        blocks = planner.optimize_block_distribution(1)
        assert blocks == {0: ["only"]}
