# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for instrumentation dumper."""

import json
import os
import tempfile
from pathlib import Path

import pytest

import flextensor
from flextensor.instrumentation import dump_instrumentation, get_registry
from flextensor.instrumentation.dumper import dump_to_directory, dump_to_file


@pytest.fixture
def registry():
    """Fixture that provides a clean registry for each test."""
    reg = get_registry()
    reg.clear()
    original_enabled = reg.enabled
    reg.enabled = True
    yield reg
    reg.enabled = original_enabled
    reg.clear()


@pytest.fixture
def temp_dir():
    """Fixture that provides a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDumpToFile:
    """Tests for dump_to_file function."""

    def test_dump_empty_registry(self, registry, temp_dir):
        """Test dumping when registry is empty."""
        output_path = temp_dir / "empty.json"
        dump_to_file(output_path)

        assert output_path.exists()

        with output_path.open() as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "flextensor_version" in data
        assert data["flextensor_version"] == flextensor.__version__
        assert data["components"] == []

    def test_dump_with_records(self, registry, temp_dir):
        """Test dumping with registered components."""
        registry.register("Class1", "module.Class1", {"arg1": "value1"})
        registry.register("Class2", "module.Class2", {"arg2": 42})

        output_path = temp_dir / "records.json"
        dump_to_file(output_path)

        with output_path.open() as f:
            data = json.load(f)

        assert len(data["components"]) == 2
        assert data["components"][0]["class_name"] == "Class1"
        assert data["components"][0]["args"]["arg1"] == "value1"
        assert data["components"][1]["class_name"] == "Class2"
        assert data["components"][1]["args"]["arg2"] == 42

    def test_dump_creates_parent_directories(self, temp_dir):
        """Test that dump_to_file creates parent directories."""
        nested_path = temp_dir / "nested" / "dirs" / "output.json"
        dump_to_file(nested_path)

        assert nested_path.exists()

    def test_dump_overwrites_existing_file(self, registry, temp_dir):
        """Test that dumping overwrites existing file."""
        output_path = temp_dir / "overwrite.json"

        # First dump
        registry.register("FirstClass", "module.FirstClass", {})
        dump_to_file(output_path)

        # Clear and add different data
        registry.clear()
        registry.register("SecondClass", "module.SecondClass", {})
        dump_to_file(output_path)

        with output_path.open() as f:
            data = json.load(f)

        # Should only have the second record
        assert len(data["components"]) == 1
        assert data["components"][0]["class_name"] == "SecondClass"


class TestDumpToDirectory:
    """Tests for dump_to_directory function."""

    def test_dump_creates_timestamped_file(self, registry, temp_dir):
        """Test that dump_to_directory creates a timestamped file."""
        registry.register("TestClass", "module.TestClass", {"x": 1})

        output_path = dump_to_directory(temp_dir)

        assert output_path.exists()
        assert output_path.suffix == ".json"
        assert "components." in output_path.name
        assert f"pid{os.getpid()}" in output_path.name

    def test_dump_creates_directory_if_not_exists(self, registry, temp_dir):
        """Test that dump_to_directory creates the output directory."""
        new_dir = temp_dir / "new_output_dir"
        assert not new_dir.exists()

        output_path = dump_to_directory(new_dir)

        assert new_dir.exists()
        assert output_path.exists()


class TestDumpInstrumentationConvenience:
    """Tests for dump_instrumentation convenience function."""

    def test_dump_instrumentation(self, registry, temp_dir):
        """Test the dump_instrumentation convenience function."""
        registry.register("ConvClass", "module.ConvClass", {"conv": True})

        output_dir = temp_dir / "convenience"
        dump_instrumentation(str(output_dir))

        assert output_dir.exists()
        assert len(list(output_dir.rglob("*.json"))) > 0


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_output_is_valid_json(self, registry, temp_dir):
        """Test that output is valid, properly formatted JSON."""
        registry.register(
            "TestClass",
            "module.TestClass",
            {
                "nested": {"key": "value"},
                "list": [1, 2, 3],
                "null_value": None,
            },
        )

        output_path = temp_dir / "format.json"
        dump_to_file(output_path)

        # Read raw content to verify formatting
        content = output_path.read_text()

        # Should be indented (pretty-printed)
        assert "\n" in content
        assert "  " in content  # Indentation

        # Should be valid JSON
        data = json.loads(content)
        assert data is not None

    def test_timestamp_format(self, registry, temp_dir):
        """Test that timestamp is in ISO format."""
        output_path = temp_dir / "timestamp.json"
        dump_to_file(output_path)

        with output_path.open() as f:
            data = json.load(f)

        timestamp = data["timestamp"]
        # ISO format should contain T separator and colons
        assert "T" in timestamp
        assert ":" in timestamp


class TestDumpExtra:
    """Tests for the extra parameter on dump functions."""

    def test_extra_keys_appear_flat_in_output(self, registry, temp_dir):
        """Test that extra dict keys are merged flat into top-level JSON."""
        output_path = temp_dir / "extra.json"
        dump_to_file(output_path, extra={"memory_transfer_stats": {1024: 0.015, 4096: 0.023}})

        with output_path.open() as f:
            data = json.load(f)

        assert "memory_transfer_stats" in data
        assert data["memory_transfer_stats"] == {"1024": 0.015, "4096": 0.023}
        # Built-in keys still present
        assert "timestamp" in data
        assert "components" in data
        assert "host_memory" in data

    def test_extra_none_produces_no_extra_keys(self, registry, temp_dir):
        """Test that extra=None produces same output as before (no extra keys)."""
        output_path = temp_dir / "no_extra.json"
        dump_to_file(output_path, extra=None)

        with output_path.open() as f:
            data = json.load(f)

        assert set(data.keys()) == {"timestamp", "flextensor_version", "components", "host_memory"}

    def test_extra_default_produces_no_extra_keys(self, registry, temp_dir):
        """Test that omitting extra produces same output as extra=None."""
        output_path = temp_dir / "default.json"
        dump_to_file(output_path)

        with output_path.open() as f:
            data = json.load(f)

        assert set(data.keys()) == {"timestamp", "flextensor_version", "components", "host_memory"}

    def test_extra_collision_raises_value_error(self, registry, temp_dir):
        """Test that extra keys colliding with built-in keys raise ValueError."""
        output_path = temp_dir / "collision.json"
        with pytest.raises(ValueError, match="collide with built-in keys"):
            dump_to_file(output_path, extra={"components": "bad"})

    def test_extra_collision_reports_colliding_keys(self, registry, temp_dir):
        """Test that the ValueError message includes the colliding key names."""
        output_path = temp_dir / "collision_detail.json"
        with pytest.raises(ValueError, match="host_memory"):
            dump_to_file(output_path, extra={"host_memory": {}, "custom_key": 42})

    def test_extra_multiple_keys(self, registry, temp_dir):
        """Test that multiple extra keys are all merged."""
        output_path = temp_dir / "multi.json"
        dump_to_file(
            output_path,
            extra={"key_a": [1, 2, 3], "key_b": {"nested": True}},
        )

        with output_path.open() as f:
            data = json.load(f)

        assert data["key_a"] == [1, 2, 3]
        assert data["key_b"] == {"nested": True}

    def test_dump_to_directory_passes_extra_through(self, registry, temp_dir):
        """Test that dump_to_directory forwards extra to dump_to_file."""
        output_path = dump_to_directory(temp_dir, extra={"custom": "value"})

        with output_path.open() as f:
            data = json.load(f)

        assert data["custom"] == "value"

    def test_dump_instrumentation_passes_extra_through(self, registry, temp_dir):
        """Test that dump_instrumentation forwards extra to dump_to_directory."""
        output_dir = temp_dir / "convenience_extra"
        dump_instrumentation(str(output_dir), extra={"custom": 123})

        json_files = list(output_dir.rglob("*.json"))
        assert len(json_files) > 0

        with json_files[0].open() as f:
            data = json.load(f)

        assert data["custom"] == 123
