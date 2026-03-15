# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for remap_strategy function."""

import pytest

from flextensor.collectors import TensorStatistics
from flextensor.strategy_operations import remap_strategy


class TestRemapStrategy:
    """Test cases for remap_strategy function."""

    @pytest.fixture
    def sample_tensor_stats(self):
        """Create sample tensor statistics for testing."""
        return [
            TensorStatistics(tensor_id=1, name="tensor1", size_bytes=1024, load_time_ms=1.0),
            TensorStatistics(tensor_id=2, name="tensor2", size_bytes=2048, load_time_ms=2.0),
            TensorStatistics(tensor_id=3, name="tensor3", size_bytes=4096, load_time_ms=3.0),
        ]

    @pytest.fixture
    def sample_strategy_map(self, sample_tensor_stats):
        """Create sample strategy map for testing."""
        return {
            "layer.0": [sample_tensor_stats[0]],
            "layer.1": [sample_tensor_stats[1]],
            "layer.2": [sample_tensor_stats[2]],
            "layer.3": [sample_tensor_stats[0], sample_tensor_stats[1]],
        }

    def test_basic_remap(self, sample_strategy_map):
        """Test basic key remapping functionality."""
        key_remap = {
            "layer.0": "new_layer.0",
            "layer.2": "new_layer.2",
        }

        result = remap_strategy(sample_strategy_map, key_remap)

        expected = {
            "new_layer.0": [sample_strategy_map["layer.0"][0]],
            "layer.1": [sample_strategy_map["layer.1"][0]],
            "new_layer.2": [sample_strategy_map["layer.2"][0]],
            "layer.3": [sample_strategy_map["layer.3"][0], sample_strategy_map["layer.3"][1]],
        }

        assert result == expected

    def test_partial_remap(self, sample_strategy_map):
        """Test remapping only some keys."""
        key_remap = {
            "layer.1": "renamed_layer.1",
        }

        result = remap_strategy(sample_strategy_map, key_remap)

        expected = {
            "layer.0": [sample_strategy_map["layer.0"][0]],
            "renamed_layer.1": [sample_strategy_map["layer.1"][0]],
            "layer.2": [sample_strategy_map["layer.2"][0]],
            "layer.3": [sample_strategy_map["layer.3"][0], sample_strategy_map["layer.3"][1]],
        }

        assert result == expected

    def test_no_remap_keys(self, sample_strategy_map):
        """Test with empty remap dictionary."""
        key_remap = {}

        result = remap_strategy(sample_strategy_map, key_remap)

        # Should return identical copy
        assert result == sample_strategy_map
        assert result is not sample_strategy_map  # Should be a copy

    def test_remap_nonexistent_keys(self, sample_strategy_map):
        """Test remapping with keys that don't exist in strategy map."""
        key_remap = {
            "nonexistent_layer": "new_nonexistent_layer",
            "layer.1": "renamed_layer.1",
        }

        result = remap_strategy(sample_strategy_map, key_remap)

        expected = {
            "layer.0": [sample_strategy_map["layer.0"][0]],
            "renamed_layer.1": [sample_strategy_map["layer.1"][0]],
            "layer.2": [sample_strategy_map["layer.2"][0]],
            "layer.3": [sample_strategy_map["layer.3"][0], sample_strategy_map["layer.3"][1]],
        }

        assert result == expected

    def test_empty_strategy_map(self):
        """Test with empty strategy map."""
        strategy_map = {}
        key_remap = {"layer.0": "new_layer.0"}

        result = remap_strategy(strategy_map, key_remap)

        assert result == {}
        assert result is not strategy_map  # Should be a copy

    def test_none_strategy_map(self):
        """Test with None strategy map."""
        strategy_map = None
        key_remap = {"layer.0": "new_layer.0"}

        result = remap_strategy(strategy_map, key_remap)

        assert result == {}

    def test_all_keys_remapped(self, sample_strategy_map):
        """Test remapping all keys in the strategy map."""
        key_remap = {
            "layer.0": "new_layer.0",
            "layer.1": "new_layer.1",
            "layer.2": "new_layer.2",
            "layer.3": "new_layer.3",
        }

        result = remap_strategy(sample_strategy_map, key_remap)

        expected = {
            "new_layer.0": [sample_strategy_map["layer.0"][0]],
            "new_layer.1": [sample_strategy_map["layer.1"][0]],
            "new_layer.2": [sample_strategy_map["layer.2"][0]],
            "new_layer.3": [sample_strategy_map["layer.3"][0], sample_strategy_map["layer.3"][1]],
        }

        assert result == expected

    def test_duplicate_new_keys(self, sample_strategy_map):
        """Test remapping to duplicate new keys (should overwrite)."""
        key_remap = {
            "layer.0": "duplicate_key",
            "layer.1": "duplicate_key",
        }

        result = remap_strategy(sample_strategy_map, key_remap)

        # Should have only one "duplicate_key" entry with the last value
        assert "duplicate_key" in result
        assert len(result) == 3  # layer.2, layer.3, and duplicate_key
        assert result["duplicate_key"] == [sample_strategy_map["layer.1"][0]]

    def test_preserve_tensor_statistics(self, sample_strategy_map):
        """Test that tensor statistics are preserved (not copied)."""
        key_remap = {"layer.0": "new_layer.0"}

        result = remap_strategy(sample_strategy_map, key_remap)

        # Check that the same tensor statistics objects are preserved
        assert result["new_layer.0"] is sample_strategy_map["layer.0"]
        assert result["layer.1"] is sample_strategy_map["layer.1"]

    def test_complex_remap_scenario(self, sample_tensor_stats):
        """Test complex remapping scenario with multiple operations."""
        strategy_map = {
            "conv.0": [sample_tensor_stats[0]],
            "bn.0": [sample_tensor_stats[1]],
            "relu.0": [sample_tensor_stats[2]],
            "conv.1": [sample_tensor_stats[0], sample_tensor_stats[1]],
            "pool.0": [sample_tensor_stats[2]],
        }

        key_remap = {
            "conv.0": "convolution.0",
            "bn.0": "batch_norm.0",
            "relu.0": "activation.0",
            "pool.0": "pooling.0",
            # conv.1 not remapped
        }

        result = remap_strategy(strategy_map, key_remap)

        expected = {
            "convolution.0": [sample_tensor_stats[0]],
            "batch_norm.0": [sample_tensor_stats[1]],
            "activation.0": [sample_tensor_stats[2]],
            "conv.1": [sample_tensor_stats[0], sample_tensor_stats[1]],
            "pooling.0": [sample_tensor_stats[2]],
        }

        assert result == expected

    def test_remap_with_special_characters(self, sample_tensor_stats):
        """Test remapping with special characters in keys."""
        strategy_map = {
            "layer-0": [sample_tensor_stats[0]],
            "layer_1": [sample_tensor_stats[1]],
            "layer.2": [sample_tensor_stats[2]],
        }

        key_remap = {
            "layer-0": "layer_with_dash",
            "layer_1": "layer_with_underscore",
            "layer.2": "layer_with_dot",
        }

        result = remap_strategy(strategy_map, key_remap)

        expected = {
            "layer_with_dash": [sample_tensor_stats[0]],
            "layer_with_underscore": [sample_tensor_stats[1]],
            "layer_with_dot": [sample_tensor_stats[2]],
        }

        assert result == expected

    def test_identity_remap(self, sample_strategy_map):
        """Test remapping with identity mappings (old_key == new_key)."""
        key_remap = {
            "layer.0": "layer.0",  # Identity mapping - should preserve original
            "layer.1": "new_layer.1",  # Regular mapping
        }

        result = remap_strategy(sample_strategy_map, key_remap)

        expected = {
            "layer.0": [sample_strategy_map["layer.0"][0]],  # Preserved due to identity mapping
            "new_layer.1": [sample_strategy_map["layer.1"][0]],  # Mapped to new key
            "layer.2": [sample_strategy_map["layer.2"][0]],  # Unchanged
            "layer.3": [sample_strategy_map["layer.3"][0], sample_strategy_map["layer.3"][1]],  # Unchanged
        }

        assert result == expected
        # Verify that the original layer.0 key is preserved (not removed)
        assert "layer.0" in result
        # Verify that layer.1 was removed and replaced with new_layer.1
        assert "layer.1" not in result
        assert "new_layer.1" in result
