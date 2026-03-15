# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for create_allocation_ordered function."""

import pytest

from flextensor.collectors import LayerStatistics
from flextensor.strategy_operations import create_allocation_ordered


class TestCreateAllocationOrdered:
    """Test cases for create_allocation_ordered function."""

    @pytest.fixture
    def sample_layer_stats(self):
        """Create sample layer statistics for testing."""
        return [
            LayerStatistics(label="layer.0", tensors=[], duration=1.0),
            LayerStatistics(label="layer.1", tensors=[], duration=1.0),
            LayerStatistics(label="layer.2", tensors=[], duration=1.0),
            LayerStatistics(label="layer.3", tensors=[], duration=1.0),
            LayerStatistics(label="layer.4", tensors=[], duration=1.0),
            LayerStatistics(label="layer.5", tensors=[], duration=1.0),
            LayerStatistics(label="layer.6", tensors=[], duration=1.0),
            LayerStatistics(label="layer.7", tensors=[], duration=1.0),
            LayerStatistics(label="layer.8", tensors=[], duration=1.0),
            LayerStatistics(label="layer.9", tensors=[], duration=1.0),
            LayerStatistics(label="layer.10", tensors=[], duration=1.0),
            LayerStatistics(label="layer.11", tensors=[], duration=1.0),
            LayerStatistics(label="layer.12", tensors=[], duration=1.0),
            LayerStatistics(label="layer.13", tensors=[], duration=1.0),
            LayerStatistics(label="layer.14", tensors=[], duration=1.0),
        ]

    def test_basic_allocation_ordered(self, sample_layer_stats):
        """Test basic functionality with the provided example."""
        label_to_block_id = {
            "layer.0": 0,
            "layer.1": 1,
            "layer.2": 2,
            "layer.3": 3,
            "layer.4": 0,
            "layer.5": 1,
            "layer.6": 2,
            "layer.7": 3,
            "layer.8": 0,
            "layer.9": 1,
            "layer.10": 2,
            "layer.11": 3,
            "layer.12": 0,
            "layer.13": 1,
            "layer.14": 2,
        }

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {
            0: ["layer.0", "layer.4", "layer.8", "layer.12"],
            1: ["layer.1", "layer.5", "layer.9", "layer.13"],
            2: ["layer.2", "layer.6", "layer.10", "layer.14"],
            3: ["layer.3", "layer.7", "layer.11"],
        }

        assert result == expected

    def test_shuffled_input_order(self, sample_layer_stats):
        """Test that function respects layer_stats order regardless of input dictionary order."""
        # Shuffled input order
        label_to_block_id = {
            "layer.14": 2,
            "layer.3": 3,
            "layer.0": 0,
            "layer.7": 3,
            "layer.10": 2,
            "layer.1": 1,
            "layer.8": 0,
            "layer.11": 3,
            "layer.4": 0,
            "layer.13": 1,
            "layer.2": 2,
            "layer.9": 1,
            "layer.12": 0,
            "layer.5": 1,
            "layer.6": 2,
        }

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {
            0: ["layer.0", "layer.4", "layer.8", "layer.12"],
            1: ["layer.1", "layer.5", "layer.9", "layer.13"],
            2: ["layer.2", "layer.6", "layer.10", "layer.14"],
            3: ["layer.3", "layer.7", "layer.11"],
        }

        assert result == expected

    def test_single_block_allocation(self, sample_layer_stats):
        """Test allocation where all layers go to the same block."""
        label_to_block_id = {
            "layer.0": 0,
            "layer.1": 0,
            "layer.2": 0,
            "layer.3": 0,
            "layer.4": 0,
        }

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {
            0: ["layer.0", "layer.1", "layer.2", "layer.3", "layer.4"],
        }

        assert result == expected

    def test_sequential_block_allocation(self, sample_layer_stats):
        """Test allocation where each layer goes to a different block."""
        label_to_block_id = {
            "layer.0": 0,
            "layer.1": 1,
            "layer.2": 2,
            "layer.3": 3,
            "layer.4": 4,
        }

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {
            0: ["layer.0"],
            1: ["layer.1"],
            2: ["layer.2"],
            3: ["layer.3"],
            4: ["layer.4"],
        }

        assert result == expected

    def test_empty_input(self, sample_layer_stats):
        """Test with empty input dictionary."""
        label_to_block_id = {}

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        assert result == {}

    def test_missing_layers_in_layer_stats(self, sample_layer_stats):
        """Test with layers that are not in layer_stats."""
        # Create layer_stats with only some layers
        limited_layer_stats = sample_layer_stats[:5]  # Only layer.0 to layer.4

        label_to_block_id = {
            "layer.0": 0,
            "layer.1": 0,
            "layer.2": 0,
            "layer.10": 0,
            "layer.11": 0,  # These are not in limited_layer_stats
        }

        result = create_allocation_ordered(label_to_block_id, limited_layer_stats)

        # Missing layers should be placed at the end
        expected = {
            0: ["layer.0", "layer.1", "layer.2", "layer.10", "layer.11"],
        }

        assert result == expected

    def test_non_sequential_block_ids(self, sample_layer_stats):
        """Test with non-sequential block IDs."""
        label_to_block_id = {
            "layer.0": 5,
            "layer.1": 2,
            "layer.2": 8,
            "layer.3": 2,
            "layer.4": 5,
        }

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {
            2: ["layer.1", "layer.3"],
            5: ["layer.0", "layer.4"],
            8: ["layer.2"],
        }

        assert result == expected

    def test_negative_block_ids(self, sample_layer_stats):
        """Test with negative block IDs."""
        label_to_block_id = {
            "layer.0": -1,
            "layer.1": 0,
            "layer.2": -1,
            "layer.3": 1,
        }

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {
            -1: ["layer.0", "layer.2"],
            0: ["layer.1"],
            1: ["layer.3"],
        }

        assert result == expected

    def test_single_layer(self, sample_layer_stats):
        """Test with single layer."""
        label_to_block_id = {"layer.0": 0}

        result = create_allocation_ordered(label_to_block_id, sample_layer_stats)

        expected = {0: ["layer.0"]}

        assert result == expected

    def test_empty_layer_stats(self):
        """Test with empty layer_stats."""
        layer_stats = []
        label_to_block_id = {"layer.0": 0, "layer.1": 0}

        result = create_allocation_ordered(label_to_block_id, layer_stats)

        # All layers should be placed at the end since they're not in layer_stats
        expected = {0: ["layer.0", "layer.1"]}

        assert result == expected

    def test_layers_without_numbers(self):
        """Test order of layers without numerical suffixes."""
        # Create layer stats with non-numerical layer names
        layer_stats = [
            LayerStatistics(label="embedding", tensors=[], duration=1.0),
            LayerStatistics(label="attention", tensors=[], duration=1.0),
            LayerStatistics(label="feedforward", tensors=[], duration=1.0),
            LayerStatistics(label="normalization", tensors=[], duration=1.0),
            LayerStatistics(label="output", tensors=[], duration=1.0),
        ]

        # Test with layers assigned to different blocks
        label_to_block_id = {
            "embedding": 0,
            "attention": 1,
            "feedforward": 0,
            "normalization": 1,
            "output": 2,
        }

        result = create_allocation_ordered(label_to_block_id, layer_stats)

        # Verify that layers are ordered according to their position in layer_stats
        expected = {
            0: ["embedding", "feedforward"],  # embedding comes before feedforward in layer_stats
            1: ["attention", "normalization"],  # attention comes before normalization in layer_stats
            2: ["output"],
        }

        assert result == expected

    def test_mixed_numerical_and_non_numerical_layers(self):
        """Test order with mixed layer naming conventions."""
        layer_stats = [
            LayerStatistics(label="input", tensors=[], duration=1.0),
            LayerStatistics(label="layer.0", tensors=[], duration=1.0),
            LayerStatistics(label="attention", tensors=[], duration=1.0),
            LayerStatistics(label="layer.1", tensors=[], duration=1.0),
            LayerStatistics(label="output", tensors=[], duration=1.0),
        ]

        label_to_block_id = {
            "input": 0,
            "layer.0": 0,
            "attention": 1,
            "layer.1": 1,
            "output": 0,
        }

        result = create_allocation_ordered(label_to_block_id, layer_stats)

        # Verify that layers maintain their order from layer_stats within each block
        expected = {
            0: ["input", "layer.0", "output"],  # input -> layer.0 -> output order from layer_stats
            1: ["attention", "layer.1"],  # attention -> layer.1 order from layer_stats
        }

        assert result == expected
