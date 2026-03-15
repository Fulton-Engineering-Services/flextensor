# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rearrange_transfers function."""

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy_operations import rearrange_transfers


def _make_layer(label: str, duration: float = 1.0) -> LayerStatistics:
    """Create a minimal LayerStatistics for testing."""
    return LayerStatistics(
        label=label,
        duration=duration,
        tensors=[
            TensorStatistics(
                tensor_id=hash(label) & 0x7FFFFFFF, name=f"{label}.weight", size_bytes=1024, load_time_ms=0.1
            )
        ],
    )


class TestRearrangeTransfersEmptyInputs:
    """Early-return path: any empty input should return a valid 3-tuple."""

    def test_all_empty(self):
        t2c, block, remapped = rearrange_transfers({}, {}, [])
        assert t2c == {}
        assert block == {}
        assert remapped == {}

    def test_empty_transfer_map(self):
        layers = [_make_layer("layer.0")]
        t2c, block, remapped = rearrange_transfers({}, {"layer.0": 0}, layers)
        assert t2c == {}
        assert block == {"layer.0": 0}
        assert remapped == {}

    def test_empty_block_map(self):
        layers = [_make_layer("layer.0")]
        t2c, block, remapped = rearrange_transfers({"layer.0": "layer.1"}, {}, layers)
        assert t2c == {"layer.0": "layer.1"}
        assert block == {}
        assert remapped == {}

    def test_empty_layer_stats(self):
        t2c, block, remapped = rearrange_transfers({"layer.0": "layer.1"}, {"layer.0": 0}, [])
        assert t2c == {"layer.0": "layer.1"}
        assert block == {"layer.0": 0}
        assert remapped == {}

    def test_returns_copies(self):
        """Returned dicts should be copies, not the originals."""
        original_t2c: dict[str, str] = {}
        original_block: dict[str, int] = {}
        t2c, block, _ = rearrange_transfers(original_t2c, original_block, [])
        assert t2c is not original_t2c
        assert block is not original_block


class TestRearrangeTransfersBasic:
    """Smoke tests for the normal (non-empty) path."""

    @pytest.fixture
    def three_layers(self):
        return [_make_layer("layer.0"), _make_layer("layer.1"), _make_layer("layer.2")]

    def test_identity_when_already_optimal(self, three_layers):
        t2c_in = {"layer.0": "layer.1"}
        block_in = {"layer.0": 0}
        t2c, block, remapped = rearrange_transfers(t2c_in, block_in, three_layers)
        assert isinstance(t2c, dict)
        assert isinstance(block, dict)
        assert isinstance(remapped, dict)
