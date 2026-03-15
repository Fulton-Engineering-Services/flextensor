# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for InstrumentationRegistry."""

import pytest

from flextensor.instrumentation import get_registry
from flextensor.instrumentation.registry import ComponentRecord, InstrumentationRegistry


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


class TestInstrumentationRegistry:
    """Tests for InstrumentationRegistry."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        registry1 = InstrumentationRegistry.get_instance()
        registry2 = InstrumentationRegistry.get_instance()
        assert registry1 is registry2

    def test_get_registry_convenience_function(self):
        """Test that get_registry returns the singleton."""
        registry1 = get_registry()
        registry2 = InstrumentationRegistry.get_instance()
        assert registry1 is registry2

    def test_enabled_property(self, registry):
        """Test enabling and disabling instrumentation."""
        registry.enabled = False
        assert not registry.enabled

        registry.enabled = True
        assert registry.enabled

    def test_register_when_disabled(self, registry):
        """Test that registration is skipped when disabled."""
        registry.enabled = False
        registry.register("TestClass", "test.module.TestClass", {"arg1": "value1"})
        assert len(registry) == 0

    def test_register_when_enabled(self, registry):
        """Test successful registration when enabled."""
        registry.register("TestClass", "test.module.TestClass", {"arg1": "value1"})
        assert len(registry) == 1

        records = registry.get_records()
        assert len(records) == 1
        assert records[0].class_name == "TestClass"
        assert records[0].module_path == "test.module.TestClass"
        assert records[0].args == {"arg1": "value1"}
        assert records[0].init_timestamp is not None

    def test_register_multiple_components(self, registry):
        """Test registering multiple components."""
        registry.register("Class1", "module.Class1", {"a": 1})
        registry.register("Class2", "module.Class2", {"b": 2})
        registry.register("Class3", "module.Class3", {"c": 3})

        assert len(registry) == 3
        records = registry.get_records()
        class_names = [r.class_name for r in records]
        assert class_names == ["Class1", "Class2", "Class3"]

    def test_clear_records(self, registry):
        """Test clearing all records."""
        registry.register("TestClass", "test.TestClass", {"arg": "value"})
        assert len(registry) > 0

        registry.clear()
        assert len(registry) == 0
        assert registry.get_records() == []

    def test_get_records_returns_copy(self, registry):
        """Test that get_records returns a copy of the records list."""
        registry.register("TestClass", "test.TestClass", {"arg": "value"})
        records1 = registry.get_records()
        records2 = registry.get_records()

        assert records1 == records2
        assert records1 is not records2  # Different list objects

    def test_component_record_dataclass(self):
        """Test ComponentRecord dataclass."""
        record = ComponentRecord(
            class_name="MyClass",
            module_path="my.module.MyClass",
            init_timestamp="2025-12-09T10:00:00",
            args={"param1": 42, "param2": "test"},
        )

        assert record.class_name == "MyClass"
        assert record.module_path == "my.module.MyClass"
        assert record.init_timestamp == "2025-12-09T10:00:00"
        assert record.args == {"param1": 42, "param2": "test"}

    def test_component_record_default_args(self):
        """Test ComponentRecord with default empty args."""
        record = ComponentRecord(
            class_name="MyClass",
            module_path="my.module.MyClass",
            init_timestamp="2025-12-09T10:00:00",
        )
        assert record.args == {}
