# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FlexTensorSnapshotWorker dump-timing behavior.

These tests verify that _dump_snapshots() is called exactly once — after the
final vLLM-orchestrated compile_or_warm_up_model() — and NOT during the
internal compile_or_warm_up_model() calls that FlexTensorOffloadWorker makes
inside warmup_and_profile_model() during model loading.

The production class (FlexTensorSnapshotWorker) requires vLLM and an actual
GPU. These tests use a FakeFTSnapshotWorker that reproduces the exact same
method-resolution-order and override pattern, allowing the fix to be validated
without hardware dependencies.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from flextensor.contrib.vllm.snapshot import MemorySnapshotMixin

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeBaseWorker:
    """Simulates vllm.v1.worker.gpu_worker.Worker lifecycle methods."""

    rank = 0
    local_rank = 0

    def __init__(self):
        self.device = MagicMock()
        self.device.__str__ = lambda _: "cuda:0"
        self.vllm_config = MagicMock()
        self.vllm_config.model_config.model = "Qwen/Qwen2.5-7B"

    def init_device(self) -> None:
        pass

    def load_model(self) -> None:
        pass

    def compile_or_warm_up_model(self) -> None:
        pass

    def determine_available_memory(self) -> int:
        return 0

    def initialize_from_config(self, kv_cache_config) -> None:
        pass


class FakeFTOffloadWorker(FakeBaseWorker):
    """Simulates FlexTensorOffloadWorker lifecycle.

    load_model() calls warmup_and_profile_model() internally, which in turn
    calls compile_or_warm_up_model() twice — reproducing the pattern that
    caused the premature dump bug (issue #88).
    """

    def load_model(self) -> None:
        """Load model, then run internal warmup (simulating FT offload setup)."""
        super().load_model()
        self.warmup_and_profile_model()

    def warmup_and_profile_model(self) -> None:
        """Run two compile_or_warm_up_model() passes internally."""
        self.compile_or_warm_up_model()  # 1st internal call
        self.compile_or_warm_up_model()  # 2nd internal call


class FakeFTSnapshotWorker(MemorySnapshotMixin, FakeFTOffloadWorker):
    """Applies MemorySnapshotMixin with the same overrides as FlexTensorSnapshotWorker.

    Mirrors FlexTensorSnapshotWorker exactly (including the fix), so these
    unit tests validate the correct override pattern without requiring vLLM.
    """

    def init_device(self) -> None:
        super().init_device()
        self._take_snapshot("after_init_device")

    def load_model(self) -> None:
        super().load_model()
        self._take_snapshot("after_load_model")

    def determine_available_memory(self) -> int:
        result = super().determine_available_memory()
        self._take_snapshot("after_determine_available_memory")
        return result

    def initialize_from_config(self, kv_cache_config) -> None:
        super().initialize_from_config(kv_cache_config)
        self._take_snapshot("after_kv_cache_init")

    def warmup_and_profile_model(self) -> None:
        """Run FlexTensor warmup/profiling without triggering snapshot dump."""
        self._in_ft_warmup = True
        try:
            super().warmup_and_profile_model()
        finally:
            self._in_ft_warmup = False

    def compile_or_warm_up_model(self) -> None:
        """Warm up model; capture snapshot and dump only on the final vLLM call."""
        super().compile_or_warm_up_model()
        if not getattr(self, "_in_ft_warmup", False):
            self._take_snapshot("after_compile_warmup")
            self._dump_snapshots()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_memory_snapshot():
    """Mock MemorySnapshot with realistic field values."""
    snap = MagicMock()
    snap.torch_peak = 1_000_000
    snap.free_memory = 80_000_000_000
    snap.total_memory = 85_899_345_920
    snap.cuda_memory = 5_899_345_920
    snap.torch_memory = 5_000_000_000
    snap.non_torch_memory = 899_345_920
    snap.timestamp = 1_708_185_000.0
    return snap


@pytest.fixture()
def worker(mock_memory_snapshot):
    """FakeFTSnapshotWorker with patched MemorySnapshot and capture_host_resources."""
    instance = FakeFTSnapshotWorker()
    with (
        patch("flextensor.contrib.vllm.snapshot.MemorySnapshot", return_value=mock_memory_snapshot),
        patch("flextensor.contrib.vllm.snapshot.capture_host_resources", return_value={}),
    ):
        yield instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlexTensorSnapshotWorkerDumpTiming:
    """Verify _dump_snapshots() fires only on the final compile_or_warm_up_model call.

    Issue #88: the internal compile_or_warm_up_model() calls inside
    warmup_and_profile_model() must NOT produce dump files.  The single dump
    must happen after all vLLM lifecycle stages have completed.
    """

    def test_no_dump_during_warmup_and_profile(self, worker, tmp_path):
        """warmup_and_profile_model() must not write any JSON dump files.

        Before the fix: two dump files are created (one per internal
        compile_or_warm_up_model() call). After the fix: zero files.
        """
        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            worker.warmup_and_profile_model()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 0, (
            f"Expected 0 dump files during warmup_and_profile_model, got {len(json_files)}. "
            "compile_or_warm_up_model() must not dump when called internally."
        )

    def test_single_dump_after_full_lifecycle(self, worker, tmp_path):
        """Full lifecycle produces exactly one JSON dump file.

        Before the fix: three files (two from internal calls + one final).
        After the fix: exactly one file (the final external call).
        """
        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            worker.init_device()
            worker.load_model()
            worker.determine_available_memory()
            worker.initialize_from_config(kv_cache_config=None)
            worker.compile_or_warm_up_model()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1, (
            f"Expected exactly 1 dump file after full lifecycle, got {len(json_files)}. "
            "Premature dumps during internal warmup must be suppressed."
        )

    def test_final_dump_contains_all_five_labels(self, worker, tmp_path):
        """The single dump file contains all 5 lifecycle snapshot labels in order."""
        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            worker.init_device()
            worker.load_model()
            worker.determine_available_memory()
            worker.initialize_from_config(kv_cache_config=None)
            worker.compile_or_warm_up_model()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        labels = [s["label"] for s in data["snapshots"]]

        expected = [
            "after_init_device",
            "after_load_model",
            "after_determine_available_memory",
            "after_kv_cache_init",
            "after_compile_warmup",
        ]
        assert labels == expected, (
            f"Expected labels {expected}, got {labels}. "
            "Missing or duplicate snapshot labels indicate wrong dump timing."
        )

    def test_no_duplicate_compile_warmup_labels(self, worker, tmp_path):
        """No duplicate after_compile_warmup labels in the final dump.

        Before the fix: two spurious after_compile_warmup entries appear from
        the internal warmup calls, resulting in 3 total instead of 1.
        """
        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            worker.init_device()
            worker.load_model()
            worker.determine_available_memory()
            worker.initialize_from_config(kv_cache_config=None)
            worker.compile_or_warm_up_model()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        warmup_labels = [s["label"] for s in data["snapshots"] if s["label"] == "after_compile_warmup"]

        assert len(warmup_labels) == 1, (
            f"Expected exactly 1 after_compile_warmup snapshot, got {len(warmup_labels)}. "
            "Duplicate entries from internal warmup calls must be suppressed."
        )
