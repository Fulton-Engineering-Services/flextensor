# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MemorySnapshotMixin and parameter serialization helpers.

This test suite validates the MemorySnapshotMixin class in isolation by
mocking vLLM's MemorySnapshot. No GPU required.

Key behaviors tested:
- _take_snapshot() captures memory fields and labels correctly
- _take_snapshot() accumulates multiple snapshots
- _dump_snapshots() writes JSON with correct structure and filename
- _dump_snapshots() handles empty snapshot list
- _serialize_module_parameters() produces metadata-only output (no raw tensor values)
- _collect_model_modules() filters parameterless modules
"""

import importlib
import json
import sys
import types
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch


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

    def test_dump_includes_modules_metadata(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify dump includes modules key with metadata-only parameter info."""
        model = torch.nn.Linear(10, 5, bias=False)
        mixin_instance.model_runner = MagicMock()
        mixin_instance.model_runner.model = model

        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_load_model")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        data = json.loads(json_files[0].read_text())

        assert "modules" in data
        assert len(data["modules"]) > 0
        # Verify metadata-only — no raw tensor values
        serialized = json.dumps(data["modules"])
        assert "tensor(" not in serialized
        assert "Parameter containing" not in serialized

    def test_dump_without_model_runner_omits_modules(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify dump works without model_runner — modules key omitted gracefully."""
        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        data = json.loads(json_files[0].read_text())
        assert "modules" not in data

    def test_dump_with_model_runner_model_none_omits_modules(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify dump works when model_runner.model is None — modules key omitted."""
        mixin_instance.model_runner = MagicMock()
        mixin_instance.model_runner.model = None

        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_init_device")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        data = json.loads(json_files[0].read_text())
        assert "modules" not in data

    def test_dump_still_writes_when_module_collection_fails(self, mixin_instance, mock_memory_snapshot, tmp_path):
        """Verify snapshot file is written even when _collect_model_modules raises."""
        model = MagicMock()
        model.named_modules.side_effect = RuntimeError("wrapped model")
        mixin_instance.model_runner = MagicMock()
        mixin_instance.model_runner.model = model

        with patch(
            "flextensor.contrib.vllm.snapshot.MemorySnapshot",
            return_value=mock_memory_snapshot,
        ):
            mixin_instance._take_snapshot("after_load_model")

        with patch.dict("os.environ", {"FT_VLLM_SNAPSHOT_OUTPUT_DIR": str(tmp_path)}):
            mixin_instance._dump_snapshots()

        json_files = list(tmp_path.glob("gpu_snapshots_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "modules" not in data
        assert len(data["snapshots"]) == 1


class TestLogitsProcessorSerialization:
    """Functional test reproducing issue #128.

    Simulates a vLLM-like model with a logits_processor module containing
    large bfloat16 parameters. Verifies that _collect_model_modules produces
    JSON-safe output where str(module._parameters) would not.
    """

    @pytest.fixture()
    def vllm_like_model(self):
        """Build a model mimicking vLLM's structure with a logits_processor."""
        model = torch.nn.Module()
        # Model layers (normal model weights)
        model.model = torch.nn.Sequential(
            torch.nn.Linear(4096, 4096, dtype=torch.bfloat16, bias=False),
        )
        # vLLM-added logits_processor with large vocab projection
        model.logits_processor = torch.nn.Linear(4096, 32000, dtype=torch.bfloat16, bias=False)
        return model

    def test_str_parameters_produces_raw_tensor_values(self, vllm_like_model):
        """Demonstrate the broken approach: str(module._parameters) dumps raw tensor data."""
        broken_output = str(vllm_like_model.logits_processor._parameters)
        assert "Parameter containing" in broken_output
        assert "tensor(" in broken_output

    def test_collect_model_modules_produces_no_raw_tensor_values(self, vllm_like_model):
        """Verify _collect_model_modules never includes raw tensor values."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        result = _collect_model_modules(vllm_like_model)
        serialized = json.dumps(result)

        assert "tensor(" not in serialized
        assert "Parameter containing" not in serialized
        assert "mappingproxy" not in serialized

    def test_serialized_output_fits_es_flattened_field(self, vllm_like_model):
        """Verify serialized output is bounded — no unbounded tensor repr growth."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        result = _collect_model_modules(vllm_like_model)
        serialized = json.dumps(result)

        # str(module._parameters) for a 32000x4096 bf16 tensor would produce
        # megabytes of output. Metadata-only serialization should be tiny.
        assert len(serialized) < 2000

    def test_logits_processor_metadata_is_correct(self, vllm_like_model):
        """Verify logits_processor parameter metadata matches actual tensor properties."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        result = _collect_model_modules(vllm_like_model)
        lp_entry = next(m for m in result if m["name"] == "logits_processor")

        weight = lp_entry["tensors_map"]["weight"]
        assert weight["shape"] == [32000, 4096]
        assert weight["dtype"] == "torch.bfloat16"
        assert weight["numel"] == 32000 * 4096
        assert weight["size_bytes"] == 32000 * 4096 * 2  # bfloat16 = 2 bytes


class TestSerializeModuleParameters:
    """Tests for _serialize_module_parameters helper."""

    def test_serializes_parameter_metadata(self):
        """Verify shape, dtype, numel, and size_bytes are captured correctly."""
        from flextensor.contrib.vllm.snapshot import _serialize_module_parameters

        param = torch.nn.Parameter(torch.zeros(32, 64, dtype=torch.float32))
        params = OrderedDict({"weight": param})
        result = _serialize_module_parameters(params)

        assert "weight" in result
        assert result["weight"]["shape"] == [32, 64]
        assert result["weight"]["dtype"] == "torch.float32"
        assert result["weight"]["numel"] == 32 * 64
        assert result["weight"]["size_bytes"] == 32 * 64 * 4  # float32 = 4 bytes

    def test_skips_none_parameters(self):
        """Verify None parameters (e.g. optional bias) are excluded."""
        from flextensor.contrib.vllm.snapshot import _serialize_module_parameters

        params = OrderedDict({"weight": torch.nn.Parameter(torch.zeros(4)), "bias": None})
        result = _serialize_module_parameters(params)

        assert "weight" in result
        assert "bias" not in result

    def test_empty_parameters(self):
        """Verify empty _parameters mapping returns empty dict."""
        from flextensor.contrib.vllm.snapshot import _serialize_module_parameters

        result = _serialize_module_parameters(OrderedDict())
        assert result == {}

    def test_no_raw_tensor_values_in_output(self):
        """Verify serialized output contains no raw tensor data — the core fix for #128."""
        from flextensor.contrib.vllm.snapshot import _serialize_module_parameters

        param = torch.nn.Parameter(torch.randn(100, 200, dtype=torch.bfloat16))
        params = OrderedDict({"weight": param})
        result = _serialize_module_parameters(params)

        serialized = json.dumps(result)
        assert "tensor(" not in serialized
        assert "Parameter containing" not in serialized
        assert "mappingproxy" not in serialized

    def test_output_is_json_serializable(self):
        """Verify the output can be serialized to JSON without TypeError."""
        from flextensor.contrib.vllm.snapshot import _serialize_module_parameters

        param = torch.nn.Parameter(torch.zeros(8, 16, dtype=torch.bfloat16))
        params = OrderedDict({"weight": param})
        result = _serialize_module_parameters(params)

        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert roundtripped["weight"]["shape"] == [8, 16]

    def test_bfloat16_size_bytes(self):
        """Verify size_bytes is correct for bfloat16 (2 bytes per element)."""
        from flextensor.contrib.vllm.snapshot import _serialize_module_parameters

        param = torch.nn.Parameter(torch.zeros(1000, dtype=torch.bfloat16))
        params = OrderedDict({"weight": param})
        result = _serialize_module_parameters(params)

        assert result["weight"]["size_bytes"] == 1000 * 2


class TestCollectModelModules:
    """Tests for _collect_model_modules helper."""

    def test_collects_modules_with_parameters(self):
        """Verify modules with parameters are included."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
        result = _collect_model_modules(model)

        names = [m["name"] for m in result]
        assert "0" in names  # first Linear
        assert "2" in names  # second Linear

    def test_excludes_parameterless_modules(self):
        """Verify modules without parameters (e.g. ReLU) are excluded."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU())
        result = _collect_model_modules(model)

        names = [m["name"] for m in result]
        assert "1" not in names  # ReLU has no parameters

    def test_tensors_map_contains_metadata(self):
        """Verify each module's tensors_map has correct parameter metadata."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        model = torch.nn.Linear(10, 5, bias=True)
        result = _collect_model_modules(model)

        # The Linear module itself should be in the list (named "")
        linear_entry = next(m for m in result if "weight" in m["tensors_map"])
        assert linear_entry["tensors_map"]["weight"]["shape"] == [5, 10]
        assert "bias" in linear_entry["tensors_map"]
        assert linear_entry["tensors_map"]["bias"]["shape"] == [5]

    def test_empty_model(self):
        """Verify empty model returns empty list."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        model = torch.nn.Sequential()
        result = _collect_model_modules(model)

        # Sequential itself has no parameters, so only the container
        assert all(m["tensors_map"] for m in result)  # no empty tensors_maps

    def test_output_is_json_serializable(self):
        """Verify full module collection output can round-trip through JSON."""
        from flextensor.contrib.vllm.snapshot import _collect_model_modules

        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 2))
        result = _collect_model_modules(model)

        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)
        assert len(roundtripped) == len(result)


class TestSnapshotImportScope:
    """Regression guards for the module-level try/except in snapshot.py."""

    def test_non_vllm_module_not_found_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ModuleNotFoundError for a FlexTensor-internal module must not be silently swallowed.

        The module-level ``try/except ModuleNotFoundError`` was intended to
        detect "vLLM not installed" and set ``MemorySnapshot``/``KVCacheConfig``/
        ``Worker`` to ``None`` in unit-test environments. If the except scope
        widens to cover the FlexTensor-internal bridge import, a rename of
        ``flextensor.contrib.vllm._logging`` would be misclassified as
        "vLLM not installed" while vLLM is actually present — silently
        disabling ``SnapshotWorker`` / ``FlexTensorSnapshotWorker`` and
        producing confusing errors at distant call sites.
        """
        # Stub vLLM submodules so the vLLM imports inside snapshot.py succeed.
        vllm_stubs: dict[str, types.ModuleType] = {
            "vllm": types.ModuleType("vllm"),
            "vllm.utils": types.ModuleType("vllm.utils"),
            "vllm.utils.mem_utils": types.ModuleType("vllm.utils.mem_utils"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.kv_cache_interface": types.ModuleType("vllm.v1.kv_cache_interface"),
            "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu_worker": types.ModuleType("vllm.v1.worker.gpu_worker"),
        }
        vllm_stubs["vllm.utils.mem_utils"].MemorySnapshot = type("_MemorySnapshot", (), {})  # type: ignore[attr-defined]
        vllm_stubs["vllm.v1.kv_cache_interface"].KVCacheConfig = type("_KVCacheConfig", (), {})  # type: ignore[attr-defined]
        vllm_stubs["vllm.v1.worker.gpu_worker"].Worker = type("_Worker", (), {})  # type: ignore[attr-defined]
        for name, mod in vllm_stubs.items():
            monkeypatch.setitem(sys.modules, name, mod)

        # Force snapshot and the FT-internal bridge to re-import under our finder.
        monkeypatch.delitem(sys.modules, "flextensor.contrib.vllm.snapshot", raising=False)
        monkeypatch.delitem(sys.modules, "flextensor.contrib.vllm._logging", raising=False)

        blocked = "flextensor.contrib.vllm._logging"

        class BlockingFinder:
            def find_spec(self, name: str, path: object = None, target: object = None) -> None:
                if name == blocked:
                    raise ModuleNotFoundError(
                        f"No module named {name!r} (simulated by test)",
                        name=name,
                    )
                return None

        finder = BlockingFinder()
        sys.meta_path.insert(0, finder)
        try:
            with pytest.raises(ModuleNotFoundError, match=r"flextensor\.contrib\.vllm\._logging"):
                importlib.import_module("flextensor.contrib.vllm.snapshot")
        finally:
            sys.meta_path.remove(finder)
            # Let monkeypatch restore the real module on teardown.
            sys.modules.pop("flextensor.contrib.vllm.snapshot", None)
