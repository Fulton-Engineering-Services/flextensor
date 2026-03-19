# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemorySnapshotMixin.

This test suite validates the MemorySnapshotMixin class in isolation by
mocking vLLM's MemorySnapshot. No GPU required.

Key behaviors tested:
- _take_snapshot() captures memory fields and labels correctly
- _take_snapshot() accumulates multiple snapshots
- _dump_snapshots() writes JSON with correct structure and filename
- _dump_snapshots() handles empty snapshot list
"""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_memory_snapshot():
    """Create a mock MemorySnapshot with realistic field values."""
    snapshot = MagicMock()
    snapshot.torch_peak = 1000000
    snapshot.free_memory = 80000000000
    snapshot.total_memory = 85899345920
    snapshot.cuda_memory = 5899345920
    snapshot.torch_memory = 5000000000
    snapshot.non_torch_memory = 899345920
    snapshot.timestamp = 1708185000.123456
    return snapshot


@pytest.fixture()
def mixin_instance():
    """Create a MemorySnapshotMixin instance with mock worker attributes."""
    from flextensor.contrib.vllm.snapshot import MemorySnapshotMixin

    instance = MemorySnapshotMixin()
    instance.rank = 0
    instance.local_rank = 0
    instance.device = MagicMock()
    instance.device.__str__ = lambda self: "cuda:0"
    instance.vllm_config = MagicMock()
    instance.vllm_config.model_config.model = "Qwen/Qwen2.5-7B"
    return instance


class TestTakeSnapshot:
    """Tests for _take_snapshot method."""

    def test_take_snapshot_appends_to_list(self, mixin_instance, mock_memory_snapshot):
        """Verify calling _take_snapshot adds one entry to _snapshots."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        assert len(mixin_instance._snapshots) == 1

    def test_take_snapshot_captures_label(self, mixin_instance, mock_memory_snapshot):
        """Verify the label field matches what was passed."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_load_model")

        assert mixin_instance._snapshots[0]["label"] == "after_load_model"

    def test_take_snapshot_captures_memory_fields(self, mixin_instance, mock_memory_snapshot):
        """Verify all 7 GPU memory fields are nested under gpu_memory."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        gpu = mixin_instance._snapshots[0]["gpu_memory"]
        assert gpu["torch_peak"] == 1000000
        assert gpu["free_memory"] == 80000000000
        assert gpu["total_memory"] == 85899345920
        assert gpu["cuda_memory"] == 5899345920
        assert gpu["torch_memory"] == 5000000000
        assert gpu["non_torch_memory"] == 899345920
        assert gpu["timestamp"] == pytest.approx(1708185000.123456)

    def test_take_snapshot_includes_host_resources(self, mixin_instance, mock_memory_snapshot):
        """Verify host_memory key is present and contains expected memory fields."""
        mock_host = {
            "host_memory_total": 274_877_906_944,
            "host_memory_used": 68_719_476_736,
            "host_memory_available": 206_158_430_208,
            "swap_total": 0,
            "swap_used": 0,
            "swap_free": 0,
        }
        with (
            patch("flextensor.contrib.vllm.snapshot.MemorySnapshot", return_value=mock_memory_snapshot),
            patch("flextensor.contrib.vllm.snapshot.capture_host_resources", return_value=mock_host),
        ):
            mixin_instance._take_snapshot("after_load_model")

        snap = mixin_instance._snapshots[0]
        assert snap["host_memory"] == mock_host

    def test_take_multiple_snapshots(self, mixin_instance, mock_memory_snapshot):
        """Verify calling 3 times accumulates 3 entries with correct labels."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")
            mixin_instance._take_snapshot("after_load_model")
            mixin_instance._take_snapshot("after_determine_available_memory")

        assert len(mixin_instance._snapshots) == 3
        labels = [s["label"] for s in mixin_instance._snapshots]
        assert labels == ["after_init_device", "after_load_model", "after_determine_available_memory"]


class TestDumpSnapshots:
    """Tests for _dump_snapshots method."""

    def test_dump_creates_json_file(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify a JSON file is created in the output directory."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1

    def test_dump_filename_contains_rank_and_device(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify filename contains rank2 and device1 when rank=2, local_rank=1."""
        mixin_instance.rank = 2
        mixin_instance.local_rank = 1

        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1
        filename = json_files[0].name
        assert "rank2" in filename
        assert "device1" in filename

    def test_dump_json_structure(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify the JSON contains worker_type, model, rank, local_rank, device, and snapshots array."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")
            mixin_instance._take_snapshot("after_load_model")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        data = json.loads(json_files[0].read_text())

        assert data["worker_type"] == "MemorySnapshotMixin"
        assert data["model"] == "Qwen/Qwen2.5-7B"
        assert data["rank"] == 0
        assert data["local_rank"] == 0
        assert data["device"] == "cuda:0"
        assert len(data["snapshots"]) == 2
        assert data["snapshots"][0]["label"] == "after_init_device"
        assert data["snapshots"][1]["label"] == "after_load_model"

    def test_dump_with_no_snapshots(self, mixin_instance, tmp_path):
        """Verify it still creates file with empty snapshots array."""
        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["snapshots"] == []

    def test_dump_skips_when_env_var_unset(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify no file is written when FT_VLLM_SNAPSHOT_OUTPUT_DIR is not set."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {}, clear=False):
            # Ensure env var is NOT set
            import os

            os.environ.pop("FT_VLLM_SNAPSHOT_OUTPUT_DIR", None)
            mixin_instance._dump_snapshots()

        # No files should be created anywhere
        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 0

    def test_dump_writes_to_env_var_directory(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify file is written to FT_VLLM_SNAPSHOT_OUTPUT_DIR when set."""
        output_dir = tmp_path / "gpu_snapshots"

        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(output_dir)}):
            mixin_instance._dump_snapshots()

        json_files = list(output_dir.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["worker_type"] == "MemorySnapshotMixin"
        assert len(data["snapshots"]) == 1

    def test_dump_creates_output_directory(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "gpu_snapshots"

        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(output_dir)}):
            mixin_instance._dump_snapshots()

        assert output_dir.exists()
        json_files = list(output_dir.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1

    def test_dump_skips_when_env_var_empty(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify no file is written when FT_VLLM_SNAPSHOT_OUTPUT_DIR is empty string."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": ""}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 0
