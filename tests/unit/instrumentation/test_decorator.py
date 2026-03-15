# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for @instrumentable decorator."""

import pytest

from flextensor.instrumentation import get_registry, instrumentable


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


class TestInstrumentableDecorator:
    """Tests for the @instrumentable decorator."""

    def test_basic_decoration(self, registry):
        """Test basic class decoration with positional and keyword args."""

        @instrumentable
        class SimpleClass:
            def __init__(self, arg1, arg2="default"):
                self.arg1 = arg1
                self.arg2 = arg2

        obj = SimpleClass("value1", arg2="value2")

        # Verify object was created correctly
        assert obj.arg1 == "value1"
        assert obj.arg2 == "value2"

        # Verify instrumentation captured args
        records = registry.get_records()
        assert len(records) == 1
        assert records[0].class_name == "SimpleClass"
        assert records[0].args["arg1"] == "value1"
        assert records[0].args["arg2"] == "value2"

    def test_decoration_with_defaults(self, registry):
        """Test that default values are captured."""

        @instrumentable
        class ClassWithDefaults:
            def __init__(self, required, optional="default_value", number=42):
                self.required = required
                self.optional = optional
                self.number = number

        obj = ClassWithDefaults("req_val")

        assert obj.required == "req_val"
        assert obj.optional == "default_value"
        assert obj.number == 42

        records = registry.get_records()
        assert len(records) == 1
        assert records[0].args["required"] == "req_val"
        assert records[0].args["optional"] == "default_value"
        assert records[0].args["number"] == 42

    def test_decoration_disabled(self, registry):
        """Test that nothing is captured when instrumentation is disabled."""
        registry.enabled = False

        @instrumentable
        class DisabledClass:
            def __init__(self, arg):
                self.arg = arg

        DisabledClass("value")

        assert len(registry) == 0

    def test_multiple_instantiations(self, registry):
        """Test that each instantiation is captured."""

        @instrumentable
        class MultiInstance:
            def __init__(self, value):
                self.value = value

        MultiInstance(1)
        MultiInstance(2)
        MultiInstance(3)

        records = registry.get_records()
        assert len(records) == 3
        values = [r.args["value"] for r in records]
        assert values == [1, 2, 3]

    def test_class_preserves_functionality(self, registry):
        """Test that decoration doesn't break class functionality."""

        @instrumentable
        class FullClass:
            class_attr = "class_level"

            def __init__(self, name):
                self.name = name

            def get_name(self):
                return self.name

            @classmethod
            def get_class_attr(cls):
                return cls.class_attr

            @staticmethod
            def static_method():
                return "static"

        obj = FullClass("test_name")

        assert obj.name == "test_name"
        assert obj.get_name() == "test_name"
        assert FullClass.get_class_attr() == "class_level"
        assert FullClass.static_method() == "static"

    def test_module_path_captured(self, registry):
        """Test that module path is correctly captured."""

        @instrumentable
        class ModulePathClass:
            def __init__(self):
                pass

        ModulePathClass()

        records = registry.get_records()
        assert len(records) == 1
        # Should include the test module path
        assert "ModulePathClass" in records[0].module_path

    def test_kwargs_only(self, registry):
        """Test handling of keyword-only arguments."""

        @instrumentable
        class KwargsOnlyClass:
            def __init__(self, *, required_kwarg, optional_kwarg="default"):
                self.required_kwarg = required_kwarg
                self.optional_kwarg = optional_kwarg

        obj = KwargsOnlyClass(required_kwarg="required_value")

        assert obj.required_kwarg == "required_value"
        assert obj.optional_kwarg == "default"

        records = registry.get_records()
        assert records[0].args["required_kwarg"] == "required_value"
        assert records[0].args["optional_kwarg"] == "default"

    def test_varargs_not_captured(self, registry):
        """Test that *args are not included in captured arguments."""

        @instrumentable
        class VarargsClass:
            def __init__(self, first, *args):
                self.first = first
                self.args = args

        obj = VarargsClass("one", "two", "three")

        assert obj.first == "one"
        assert obj.args == ("two", "three")

        records = registry.get_records()
        # Should capture 'first' but not the varargs
        assert "first" in records[0].args
        assert records[0].args["first"] == "one"
