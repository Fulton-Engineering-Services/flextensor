# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for find_transfers_for_preload function."""

import pytest

from flextensor.collectors import LayerStatistics
from flextensor.strategy_operations import find_transfers_for_preload


class TestFindTransfersForPreload:
    """Test cases for find_transfers_for_preload function."""

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
            LayerStatistics(label="layer.15", tensors=[], duration=1.0),
        ]

    def test_example_1(self, sample_layer_stats):
        """Test example 1 from the requirements."""
        # Example 1: layer.11 -> layer.0 should be the only transfer returned
        transfer_to_compute_map = {
            "layer.0": "layer.2",  # transfer at 0, compute at 2 -> transfer before compute
            "layer.1": "layer.4",  # transfer at 1, compute at 4 -> transfer before compute
            "layer.2": "layer.6",  # transfer at 2, compute at 6 -> transfer before compute
            "layer.3": "layer.8",  # transfer at 3, compute at 8 -> transfer before compute
            "layer.5": "layer.10",  # transfer at 5, compute at 10 -> transfer before compute
            "layer.7": "layer.12",  # transfer at 7, compute at 12 -> transfer before compute
            "layer.9": "layer.14",  # transfer at 9, compute at 14 -> transfer before compute
            "layer.11": "layer.0",  # transfer at 11, compute at 0 -> transfer after compute (cyclic)
        }

        expected_result = {
            "layer.11": "layer.0",  # Only this transfer happens after its compute
        }

        result = find_transfers_for_preload(transfer_to_compute_map, sample_layer_stats)
        assert result == expected_result

    def test_example_2(self, sample_layer_stats):
        """Test example 2 from the requirements."""
        # Example 2: layer.14 -> layer.0 and layer.15 -> layer.2 should be returned
        transfer_to_compute_map = {
            "layer.3": "layer.4",  # transfer at 3, compute at 4 -> transfer before compute
            "layer.5": "layer.6",  # transfer at 5, compute at 6 -> transfer before compute
            "layer.7": "layer.8",  # transfer at 7, compute at 8 -> transfer before compute
            "layer.9": "layer.10",  # transfer at 9, compute at 10 -> transfer before compute
            "layer.11": "layer.12",  # transfer at 11, compute at 12 -> transfer before compute
            "layer.13": "layer.14",  # transfer at 13, compute at 14 -> transfer before compute
            "layer.14": "layer.0",  # transfer at 14, compute at 0 -> transfer after compute (cyclic)
            "layer.15": "layer.2",  # transfer at 15, compute at 2 -> transfer after compute (cyclic)
        }

        expected_result = {
            "layer.14": "layer.0",  # transfer after compute (cyclic)
            "layer.15": "layer.2",  # transfer after compute (cyclic)
        }

        result = find_transfers_for_preload(transfer_to_compute_map, sample_layer_stats)
        assert result == expected_result

    def test_empty_inputs(self):
        """Test with empty inputs."""
        assert find_transfers_for_preload({}, []) == {}
        assert find_transfers_for_preload({}, []) == {}
        assert find_transfers_for_preload({"layer.0": "layer.1"}, []) == {}

    def test_no_transfers_after_compute(self, sample_layer_stats):
        """Test case where no transfers happen after their compute."""
        transfer_to_compute_map = {
            "layer.0": "layer.2",  # transfer at 0, compute at 2 -> transfer before compute
            "layer.1": "layer.4",  # transfer at 1, compute at 4 -> transfer before compute
            "layer.3": "layer.6",  # transfer at 3, compute at 6 -> transfer before compute
        }

        expected_result = {}

        result = find_transfers_for_preload(transfer_to_compute_map, sample_layer_stats)
        assert result == expected_result

    def test_all_transfers_after_compute(self, sample_layer_stats):
        """Test case where all transfers happen after their compute."""
        transfer_to_compute_map = {
            "layer.5": "layer.2",  # transfer at 5, compute at 2 -> transfer after compute
            "layer.7": "layer.4",  # transfer at 7, compute at 4 -> transfer after compute
            "layer.9": "layer.6",  # transfer at 9, compute at 6 -> transfer after compute
        }

        expected_result = {
            "layer.5": "layer.2",
            "layer.7": "layer.4",
            "layer.9": "layer.6",
        }

        result = find_transfers_for_preload(transfer_to_compute_map, sample_layer_stats)
        assert result == expected_result

    def test_missing_layers_in_layer_stats(self, sample_layer_stats):
        """Test case where some layers are missing from layer_stats."""
        transfer_to_compute_map = {
            "layer.5": "layer.2",  # both exist, transfer after compute (cyclic)
            "layer.7": "layer.4",  # both exist, transfer after compute (cyclic)
            "missing_layer": "layer.6",  # transfer missing
            "layer.9": "missing_compute",  # compute missing
        }

        expected_result = {
            "layer.5": "layer.2",  # transfer after compute (cyclic)
            "layer.7": "layer.4",  # transfer after compute (cyclic)
        }

        result = find_transfers_for_preload(transfer_to_compute_map, sample_layer_stats)
        assert result == expected_result

    def test_cyclic_case_detailed(self, sample_layer_stats):
        """Test cyclic case in more detail."""
        # Create a scenario where we have a clear cyclic case
        transfer_to_compute_map = {
            "layer.10": "layer.2",  # transfer at 10, compute at 2 -> transfer after compute (cyclic)
            "layer.12": "layer.4",  # transfer at 12, compute at 4 -> transfer after compute (cyclic)
            "layer.1": "layer.8",  # transfer at 1, compute at 8 -> transfer before compute
        }

        expected_result = {
            "layer.10": "layer.2",  # cyclic case: 10 > 2
            "layer.12": "layer.4",  # cyclic case: 12 > 4
        }

        result = find_transfers_for_preload(transfer_to_compute_map, sample_layer_stats)
        assert result == expected_result
