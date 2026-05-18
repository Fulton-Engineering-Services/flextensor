# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for utils module."""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from flextensor.utils import (
    PartitionedPatterns,
    _compile_segment_regex,
    any_path_matches_pattern,
    atomic_write_json,
    calculate_tensor_size,
    get_class_matched_module_paths,
    matches_any_class_pattern,
    matches_any_pattern,
    partition_patterns,
)


class TestAtomicWriteJson:
    """Test cases for atomic_write_json function."""

    def test_basic_write(self, tmp_path: Path):
        """Test basic atomic write to new file."""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        atomic_write_json(file_path, data)

        assert file_path.exists()
        with file_path.open() as f:
            loaded = json.load(f)
        assert loaded == data

    def test_overwrite_existing_file(self, tmp_path: Path):
        """Test atomic write replaces existing file."""
        file_path = tmp_path / "test.json"

        # Write initial data
        initial_data = {"version": 1}
        with file_path.open("w") as f:
            json.dump(initial_data, f)

        # Overwrite with new data
        new_data = {"version": 2, "new_field": "value"}
        atomic_write_json(file_path, new_data)

        # Verify new data is present
        with file_path.open() as f:
            loaded = json.load(f)
        assert loaded == new_data
        assert loaded != initial_data

    def test_creates_parent_directories(self, tmp_path: Path):
        """Test that parent directories are created if they don't exist."""
        file_path = tmp_path / "subdir1" / "subdir2" / "test.json"
        data = {"nested": "directory"}

        atomic_write_json(file_path, data)

        assert file_path.exists()
        with file_path.open() as f:
            loaded = json.load(f)
        assert loaded == data

    def test_large_data(self, tmp_path: Path):
        """Test atomic write with large data structure."""
        file_path = tmp_path / "large.json"

        # Create large nested structure
        data = {
            f"layer_{i}": {
                f"tensor_{j}": {"size": 1024 * 1024, "dtype": "float32", "values": list(range(100))} for j in range(10)
            }
            for i in range(100)
        }

        atomic_write_json(file_path, data)

        assert file_path.exists()
        with file_path.open() as f:
            loaded = json.load(f)
        assert loaded == data

    def test_temp_file_cleaned_up_on_success(self, tmp_path: Path):
        """Test that temporary file is removed after successful write."""
        file_path = tmp_path / "test.json"
        data = {"test": "data"}

        atomic_write_json(file_path, data)

        # Check no temp files remain
        temp_files = list(tmp_path.glob(".test.json.*.tmp"))
        assert len(temp_files) == 0

    def test_temp_file_cleaned_up_on_json_error(self, tmp_path: Path):
        """Test that temporary file is cleaned up if json.dump fails."""
        file_path = tmp_path / "test.json"

        # Create un-serializable data (set objects can't be serialized)
        class UnserializableClass:
            pass

        data = {"obj": UnserializableClass()}

        with pytest.raises(TypeError):
            atomic_write_json(file_path, data)

        # Verify target file was not created
        assert not file_path.exists()

        # Verify temp file was cleaned up
        temp_files = list(tmp_path.glob(".test.json.*.tmp"))
        assert len(temp_files) == 0

    def test_original_file_untouched_on_error(self, tmp_path: Path):
        """Test that original file is not modified if write fails."""
        file_path = tmp_path / "test.json"

        # Write initial valid data
        original_data = {"version": 1, "status": "good"}
        with file_path.open("w") as f:
            json.dump(original_data, f)

        # Try to write invalid data
        class UnserializableClass:
            pass

        invalid_data = {"obj": UnserializableClass()}

        with pytest.raises(TypeError):
            atomic_write_json(file_path, invalid_data)

        # Verify original file is unchanged
        with file_path.open() as f:
            loaded = json.load(f)
        assert loaded == original_data

    def test_pathlib_path_input(self, tmp_path: Path):
        """Test that function accepts pathlib.Path objects."""
        file_path = tmp_path / "test.json"
        data = {"path": "pathlib"}

        atomic_write_json(file_path, data)

        assert file_path.exists()

    def test_string_path_input(self, tmp_path: Path):
        """Test that function accepts string paths."""
        file_path = str(tmp_path / "test.json")
        data = {"path": "string"}

        atomic_write_json(file_path, data)

        assert Path(file_path).exists()

    def test_temp_file_cleanup_warning_on_oserror(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """Test that OSError during cleanup is logged as warning."""
        file_path = tmp_path / "test.json"
        data = {"test": "data"}

        with (
            patch.object(Path, "replace", side_effect=OSError("Replace failed")),
            patch.object(Path, "unlink", side_effect=OSError("Permission denied")),
            caplog.at_level(logging.WARNING, logger="flextensor.utils"),
            pytest.raises(OSError, match="Replace failed"),
        ):
            atomic_write_json(file_path, data)

        assert "Failed to clean up temp file" in caplog.text
        assert "Permission denied" in caplog.text


class TestCalculateTensorSize:
    """Test cases for calculate_tensor_size function."""

    def test_calculate_tensor_size(self):
        """Test tensor size calculation."""
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.numel.return_value = 1024 * 1024  # 1M elements
        mock_tensor.element_size.return_value = 4  # 4 bytes per element

        size_bytes = calculate_tensor_size(mock_tensor)

        expected_size = 1024 * 1024 * 4  # 4 MB in bytes
        assert size_bytes == expected_size

    def test_calculate_tensor_size_small(self):
        """Test tensor size calculation for small tensor."""
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.numel.return_value = 100
        mock_tensor.element_size.return_value = 4

        size_bytes = calculate_tensor_size(mock_tensor)

        expected_size = 100 * 4  # 400 bytes
        assert size_bytes == expected_size

    def test_calculate_tensor_size_empty_tensor(self):
        """Test tensor size calculation for empty tensor."""
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.numel.return_value = 0
        mock_tensor.element_size.return_value = 4

        size_bytes = calculate_tensor_size(mock_tensor)

        assert size_bytes == 0

    def test_calculate_tensor_size_different_dtypes(self):
        """Test tensor size calculation for different data types."""
        # Test float16 (2 bytes)
        mock_tensor_f16 = Mock(spec=torch.Tensor)
        mock_tensor_f16.numel.return_value = 1024
        mock_tensor_f16.element_size.return_value = 2

        size_bytes_f16 = calculate_tensor_size(mock_tensor_f16)
        expected_f16 = 1024 * 2
        assert size_bytes_f16 == expected_f16

        # Test float64 (8 bytes)
        mock_tensor_f64 = Mock(spec=torch.Tensor)
        mock_tensor_f64.numel.return_value = 1024
        mock_tensor_f64.element_size.return_value = 8

        size_bytes_f64 = calculate_tensor_size(mock_tensor_f64)
        expected_f64 = 1024 * 8
        assert size_bytes_f64 == expected_f64

    def test_calculate_tensor_size_large_tensor(self):
        """Test tensor size calculation for large tensor."""
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.numel.return_value = 1024 * 1024 * 1024  # 1B elements
        mock_tensor.element_size.return_value = 4  # 4 bytes per element

        size_bytes = calculate_tensor_size(mock_tensor)

        expected_size = 1024 * 1024 * 1024 * 4  # 4GB in bytes
        assert size_bytes == expected_size


class TestCompileSegmentRegex:
    """Test cases for the _compile_segment_regex helper."""

    def test_literal_segment(self):
        """Literal segment matches itself exactly."""
        rx = _compile_segment_regex("layers")
        assert rx.fullmatch("layers")
        assert not rx.fullmatch("layer")
        assert not rx.fullmatch("layers0")

    def test_star_matches_any_chars(self):
        """* maps to .* and matches zero or more characters."""
        rx = _compile_segment_regex("layer_*")
        assert rx.fullmatch("layer_0")
        assert rx.fullmatch("layer_abc")
        assert rx.fullmatch("layer_")
        assert not rx.fullmatch("block_0")

    def test_standalone_star(self):
        """A bare * matches any string (one segment worth of characters)."""
        rx = _compile_segment_regex("*")
        assert rx.fullmatch("")
        assert rx.fullmatch("anything")
        assert rx.fullmatch("layer_0")

    def test_question_mark_matches_single_char(self):
        """? maps to . and matches exactly one character."""
        rx = _compile_segment_regex("layer?")
        assert rx.fullmatch("layer0")
        assert rx.fullmatch("layerX")
        assert not rx.fullmatch("layer")
        assert not rx.fullmatch("layer01")

    def test_multiple_question_marks(self):
        rx = _compile_segment_regex("l??r")
        assert rx.fullmatch("lXYr")
        assert not rx.fullmatch("lXr")
        assert not rx.fullmatch("lXYZr")

    def test_combined_star_and_question_mark(self):
        rx = _compile_segment_regex("a?b*c")
        assert rx.fullmatch("aXbc")
        assert rx.fullmatch("aXbXYZc")
        assert not rx.fullmatch("abc")

    def test_regex_special_chars_escaped(self):
        """Characters that are special in regex (e.g. '.', '+', '(') are escaped."""
        rx = _compile_segment_regex("a.b")
        assert rx.fullmatch("a.b")
        assert not rx.fullmatch("aXb")

        rx2 = _compile_segment_regex("x+y")
        assert rx2.fullmatch("x+y")
        assert not rx2.fullmatch("xy")
        assert not rx2.fullmatch("xxy")

    def test_numeric_segment(self):
        rx = _compile_segment_regex("0")
        assert rx.fullmatch("0")
        assert not rx.fullmatch("1")

    def test_return_type_is_compiled_pattern(self):
        import re

        rx = _compile_segment_regex("foo")
        assert isinstance(rx, re.Pattern)


class TestMatchesAnyPattern:
    """Test cases for matches_any_pattern function."""

    # --- recursive_star=False (include patterns) ---

    def test_exact_match(self):
        """Test exact segment matching."""
        assert matches_any_pattern("layers", ["layers"], recursive_star=False)

    def test_exact_match_multi_segment(self):
        """Test exact multi-segment matching."""
        assert matches_any_pattern("layers.0", ["layers.0"], recursive_star=False)

    def test_single_star_matches_one_segment(self):
        """Test that * matches exactly one segment when recursive_star=False."""
        assert matches_any_pattern("layers.0", ["layers.*"], recursive_star=False)
        assert matches_any_pattern("layers.1", ["layers.*"], recursive_star=False)

    def test_single_star_does_not_match_nested(self):
        """Test that * does not match nested segments when recursive_star=False."""
        assert not matches_any_pattern("layers.0.attn", ["layers.*"], recursive_star=False)

    def test_standalone_star_matches_single_segment(self):
        """Test that standalone * matches single segment when recursive_star=False."""
        assert matches_any_pattern("layers", ["*"], recursive_star=False)
        assert matches_any_pattern("norm", ["*"], recursive_star=False)

    def test_standalone_star_no_nested(self):
        """Test that standalone * does not match nested paths when recursive_star=False."""
        assert not matches_any_pattern("layers.0", ["*"], recursive_star=False)

    def test_intra_segment_wildcard(self):
        """Test wildcard within segment (e.g., layer_*)."""
        assert matches_any_pattern("layer_0", ["layer_*"], recursive_star=False)
        assert matches_any_pattern("layer_abc", ["layer_*"], recursive_star=False)
        assert not matches_any_pattern("other", ["layer_*"], recursive_star=False)

    def test_question_mark_wildcard(self):
        """Test ? matches exactly one character."""
        assert matches_any_pattern("layer0", ["layer?"], recursive_star=False)
        assert not matches_any_pattern("layer01", ["layer?"], recursive_star=False)

    def test_no_match(self):
        """Test when no patterns match."""
        assert not matches_any_pattern("head", ["layers.*"], recursive_star=False)

    def test_multiple_patterns(self):
        """Test matching against multiple patterns."""
        patterns = ["layers.*", "head", "norm"]
        assert matches_any_pattern("layers.0", patterns, recursive_star=False)
        assert matches_any_pattern("head", patterns, recursive_star=False)
        assert matches_any_pattern("norm", patterns, recursive_star=False)
        assert not matches_any_pattern("embed", patterns, recursive_star=False)

    def test_empty_patterns_no_match(self):
        """Test that empty patterns list matches nothing."""
        assert not matches_any_pattern("anything", [])

    # --- recursive_star=True (exclude patterns) ---

    def test_recursive_star_matches_descendants(self):
        """Test that * matches 1+ segments when recursive_star=True."""
        assert matches_any_pattern("foo.bar", ["foo.*"])
        assert matches_any_pattern("foo.bar.baz", ["foo.*"])
        assert matches_any_pattern("foo.bar.baz.weight", ["foo.*"])

    def test_recursive_star_weight_suffix(self):
        """Test *.weight matches any path ending in weight."""
        assert matches_any_pattern("lm_head.weight", ["*.weight"])
        assert matches_any_pattern("layers.0.self_attn.q_proj.weight", ["*.weight"])

    def test_recursive_standalone_star_matches_anything(self):
        """Test that standalone * matches any path with recursive_star=True."""
        assert matches_any_pattern("layers", ["*"])
        assert matches_any_pattern("layers.0", ["*"])
        assert matches_any_pattern("a.b.c.d", ["*"])

    def test_recursive_star_must_match_at_least_one(self):
        """Test that recursive * matches at least 1 segment."""
        # foo.* requires at least foo + one more segment
        assert not matches_any_pattern("foo", ["foo.*"])

    def test_recursive_exact_match(self):
        """Test exact match with recursive_star=True."""
        assert matches_any_pattern("lm_head", ["lm_head"])
        assert not matches_any_pattern("lm_head.weight", ["lm_head"])

    def test_recursive_specific_layer(self):
        """Test specific layer match."""
        assert matches_any_pattern("layers.31", ["layers.31"])
        assert not matches_any_pattern("layers.30", ["layers.31"])

    def test_edge_case_empty_path(self):
        """Test empty path — split("") gives [""] which * matches as one segment."""
        # In practice, empty paths don't occur (root module is skipped in offload_modules)
        assert matches_any_pattern("", ["*"])

    def test_intra_segment_wildcard_in_multi_segment(self):
        """Test intra-segment wildcard in multi-segment pattern."""
        assert matches_any_pattern("layers.0.self_attn", ["layers.*.self_attn"], recursive_star=False)
        assert not matches_any_pattern("layers.0.ffn", ["layers.*.self_attn"], recursive_star=False)


class TestAnyPathMatchesPattern:
    """Test cases for any_path_matches_pattern (inverse of matches_any_pattern)."""

    def test_exact_match_found(self):
        paths = ["layers", "head", "norm"]
        assert any_path_matches_pattern(paths, "head", recursive_star=False)

    def test_exact_match_not_found(self):
        paths = ["layers", "head", "norm"]
        assert not any_path_matches_pattern(paths, "embed", recursive_star=False)

    def test_wildcard_segment(self):
        paths = ["layers.0", "layers.1", "norm"]
        assert any_path_matches_pattern(paths, "layers.*", recursive_star=False)

    def test_wildcard_no_match(self):
        paths = ["layers.0.attn", "norm"]
        assert not any_path_matches_pattern(paths, "head.*", recursive_star=False)

    def test_single_star_not_recursive(self):
        """Standalone * matches exactly one segment when recursive_star=False."""
        paths = ["layers.0.attn"]
        assert not any_path_matches_pattern(paths, "layers.*", recursive_star=False)

    def test_recursive_star(self):
        """Standalone * matches 1+ segments when recursive_star=True."""
        paths = ["layers.0.attn", "layers.1.ffn"]
        assert any_path_matches_pattern(paths, "layers.*", recursive_star=True)

    def test_empty_paths(self):
        assert not any_path_matches_pattern([], "layers", recursive_star=False)

    def test_intra_segment_wildcard(self):
        paths = ["layer_0", "layer_1", "norm"]
        assert any_path_matches_pattern(paths, "layer_*", recursive_star=False)
        assert not any_path_matches_pattern(paths, "block_*", recursive_star=False)

    def test_question_mark(self):
        paths = ["layer0", "layer1"]
        assert any_path_matches_pattern(paths, "layer?", recursive_star=False)
        assert not any_path_matches_pattern(paths, "layer??", recursive_star=False)

    def test_multi_segment_pattern(self):
        paths = ["layers.0.self_attn", "layers.1.ffn"]
        assert any_path_matches_pattern(paths, "layers.*.self_attn", recursive_star=False)
        assert not any_path_matches_pattern(paths, "layers.*.mlp", recursive_star=False)

    def test_equivalence_with_matches_any_pattern(self):
        """any_path_matches_pattern must agree with the any(matches_any_pattern(...)) loop."""
        paths = ["layers.0.self_attn", "layers.1.ffn", "norm", "lm_head"]
        patterns = ["layers.*", "norm", "embed", "layers.*.self_attn", "lm_head.weight"]
        for pattern in patterns:
            for rs in (True, False):
                expected = any(matches_any_pattern(p, [pattern], recursive_star=rs) for p in paths)
                actual = any_path_matches_pattern(paths, pattern, recursive_star=rs)
                assert actual == expected, f"Mismatch for pattern={pattern!r}, recursive_star={rs}"


class TestPartitionPatterns:
    """Tests for partition_patterns: prefix parsing of ``name:`` / ``class:``."""

    def test_bare_patterns_treated_as_name(self):
        names, classes = partition_patterns(["layers.*", "lm_head"])
        assert names == ["layers.*", "lm_head"]
        assert classes == []

    def test_explicit_name_prefix_stripped(self):
        names, classes = partition_patterns(["name:layers.*", "name:lm_head"])
        assert names == ["layers.*", "lm_head"]
        assert classes == []

    def test_class_prefix_collected(self):
        names, classes = partition_patterns(["class:MoELayer", "class:*Expert*"])
        assert names == []
        assert classes == ["MoELayer", "*Expert*"]

    def test_mixed_prefixes_preserved(self):
        names, classes = partition_patterns(["layers.*", "name:embed", "class:SharedExpertMLP", "class:Linear"])
        assert names == ["layers.*", "embed"]
        assert classes == ["SharedExpertMLP", "Linear"]

    def test_empty_class_body_raises(self):
        with pytest.raises(ValueError, match="empty body"):
            partition_patterns(["class:"])

    def test_empty_name_body_raises(self):
        with pytest.raises(ValueError, match="empty body"):
            partition_patterns(["name:"])

    @pytest.mark.parametrize("body", [" ", "  ", "\t", "\n"])
    def test_whitespace_only_class_body_raises(self, body):
        with pytest.raises(ValueError, match="empty body"):
            partition_patterns([f"class:{body}"])

    @pytest.mark.parametrize("body", [" ", "  ", "\t", "\n"])
    def test_whitespace_only_name_body_raises(self, body):
        with pytest.raises(ValueError, match="empty body"):
            partition_patterns([f"name:{body}"])

    def test_order_preserved(self):
        raw = ["a", "class:Foo", "b", "name:c", "class:Bar"]
        names, classes = partition_patterns(raw)
        assert names == ["a", "b", "c"]
        assert classes == ["Foo", "Bar"]

    def test_empty_input(self):
        names, classes = partition_patterns([])
        assert names == []
        assert classes == []

    def test_returns_partitioned_patterns_dataclass(self):
        # Public surface for callers that need raw → body mappings (e.g. for
        # diagnostics that echo the user's exact configuration).
        result = partition_patterns(["layers.*", "name:embed", "class:MoELayer"])
        assert isinstance(result, PartitionedPatterns)
        assert result.name_bodies == {"layers.*": "layers.*", "name:embed": "embed"}
        assert result.class_bodies == {"class:MoELayer": "MoELayer"}
        assert result.name_patterns == ["layers.*", "embed"]
        assert result.class_patterns == ["MoELayer"]

    def test_dataclass_unpacks_to_body_lists(self):
        # Back-compat: `names, classes = partition_patterns(...)` keeps working.
        names, classes = partition_patterns(["a", "class:Foo"])
        assert names == ["a"]
        assert classes == ["Foo"]

    def test_raw_keys_preserve_user_text(self):
        # Diagnostic warnings in offload_manager echo the raw pattern; the
        # raw-keyed dicts must keep the prefix the user wrote.
        result = partition_patterns(["class:MoELayer", "name:embed", "bare"])
        assert "class:MoELayer" in result.class_bodies
        assert "name:embed" in result.name_bodies
        assert "bare" in result.name_bodies

    def test_bodies_are_immutable(self):
        # ``frozen=True`` on the dataclass blocks field rebinding; the
        # ``MappingProxyType`` wrap extends that to dict contents so the
        # immutability promise isn't leaky. Without the wrap, a caller could
        # mutate ``result.name_bodies["x"] = "y"`` behind the dataclass's back.
        result = partition_patterns(["layers.*", "class:MoELayer"])
        with pytest.raises(TypeError):
            result.name_bodies["new"] = "value"  # type: ignore[index]
        with pytest.raises(TypeError):
            result.class_bodies["new"] = "value"  # type: ignore[index]

    def test_bodies_isolated_from_source_dict_mutation(self):
        # The dataclass defensively copies the source dict before wrapping,
        # so callers that retain a reference to the pre-construction dict
        # can't leak mutations through the MappingProxyType view.
        source = {"layers.*": "layers.*"}
        result = PartitionedPatterns(name_bodies=source)
        source["sneaky"] = "injected"
        assert "sneaky" not in result.name_bodies


def _make_synthetic_class(short_name: str, module: str = "tests.synthetic") -> type:
    """Build a class whose ``__name__`` and ``__module__`` are controlled.

    Used to exercise :func:`matches_any_class_pattern` which tests both the
    short class name and the fully-qualified class name.
    """
    cls = type(short_name, (), {})
    cls.__module__ = module
    return cls


class TestMatchesAnyClassPattern:
    """Tests for the class-name glob matcher (short name + FQCN)."""

    def test_exact_match(self):
        assert matches_any_class_pattern(_make_synthetic_class("Linear"), ["Linear"])
        assert not matches_any_class_pattern(_make_synthetic_class("Conv2d"), ["Linear"])

    def test_star_wildcard(self):
        assert matches_any_class_pattern(_make_synthetic_class("SharedExpertMLP"), ["*Expert*"])
        assert matches_any_class_pattern(_make_synthetic_class("GatedMLP"), ["*MLP"])
        assert not matches_any_class_pattern(_make_synthetic_class("LayerNorm"), ["*MLP"])

    def test_question_mark(self):
        assert matches_any_class_pattern(_make_synthetic_class("MoE"), ["Mo?"])
        assert not matches_any_class_pattern(_make_synthetic_class("MoELayer"), ["Mo?"])

    def test_multiple_patterns_or(self):
        assert matches_any_class_pattern(_make_synthetic_class("Linear"), ["MoELayer", "Linear"])
        assert not matches_any_class_pattern(_make_synthetic_class("LayerNorm"), ["MoELayer", "Linear"])

    def test_empty_patterns_is_false(self):
        assert not matches_any_class_pattern(_make_synthetic_class("Linear"), [])

    def test_fqcn_exact_match(self):
        """A dotted pattern can match the fully-qualified class name."""
        cls = _make_synthetic_class("Linear", module="torch.nn.modules.linear")
        assert matches_any_class_pattern(cls, ["torch.nn.modules.linear.Linear"])
        assert not matches_any_class_pattern(cls, ["other.pkg.Linear"])

    def test_fqcn_glob_disambiguates_short_name_collision(self):
        """Users can target a specific package when two classes share a short name."""
        torch_linear = _make_synthetic_class("Linear", module="torch.nn.modules.linear")
        user_linear = _make_synthetic_class("Linear", module="myapp.layers")

        patterns = ["torch.nn.*.Linear"]
        assert matches_any_class_pattern(torch_linear, patterns)
        # Short-name ``Linear`` does not match ``torch.nn.*.Linear`` (no dot-split
        # semantics), and the user's FQCN lives in a different namespace.
        assert not matches_any_class_pattern(user_linear, patterns)

    def test_fqcn_star_spans_dots(self):
        """In class patterns ``*`` is atomic over the whole string, so it spans dots."""
        cls = _make_synthetic_class("SharedExpertMLP", module="vllm.model_executor.layers.moe")
        assert matches_any_class_pattern(cls, ["*.SharedExpertMLP"])
        assert matches_any_class_pattern(cls, ["vllm.*"])

    def test_short_name_still_wins_when_pattern_is_unqualified(self):
        """A pattern without dots is matched against the short name (backward compat)."""
        cls = _make_synthetic_class("MoELayer", module="some.deep.nested.module")
        assert matches_any_class_pattern(cls, ["MoELayer"])
        assert matches_any_class_pattern(cls, ["*Layer"])


class _MockLinear(torch.nn.Linear):
    """Distinct subclass used to exercise class-name matching in the tests."""


class _MockMoELayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 4)


class TestGetClassMatchedModulePaths:
    """Tests for walking a model and collecting paths by class-name match."""

    def _make_model(self):
        model = torch.nn.Module()
        model.linear = _MockLinear(4, 4)
        model.moe = _MockMoELayer()
        model.norm = torch.nn.LayerNorm(4)
        return model

    def test_empty_patterns_returns_empty_set(self):
        model = self._make_model()
        assert get_class_matched_module_paths(model, []) == set()

    def test_exact_class_name_match(self):
        model = self._make_model()
        paths = get_class_matched_module_paths(model, ["_MockLinear"])
        assert paths == {"linear"}

    def test_glob_class_name_match(self):
        model = self._make_model()
        paths = get_class_matched_module_paths(model, ["*MoE*"])
        assert paths == {"moe"}

    def test_includes_nested_matches(self):
        model = self._make_model()
        # moe.gate is Linear (base class), not _MockLinear.
        paths = get_class_matched_module_paths(model, ["Linear"])
        assert paths == {"moe.gate"}

    def test_empty_patterns_on_non_module_short_circuits(self):
        # No patterns means no work to do; model walkability is irrelevant.
        assert get_class_matched_module_paths({"a": 1}, []) == set()

    def test_non_module_with_patterns_raises(self):
        # Class matching against a non-walkable model is a configuration bug,
        # not "no match"; the function must surface it instead of silently
        # returning an empty set.
        with pytest.raises(TypeError, match="named_modules"):
            get_class_matched_module_paths({"a": 1}, ["Foo"])

    def test_root_module_excluded(self):
        """Root module (empty path) is never returned even if its class matches."""
        model = self._make_model()
        # Match the root's class name (``Module``).
        paths = get_class_matched_module_paths(model, ["Module"])
        # Only submodules (non-empty path) may be returned.
        assert "" not in paths

    def test_fqcn_pattern_disambiguates_across_modules(self):
        """FQCN globs can target a specific package even when short names collide."""
        model = torch.nn.Module()
        model.torch_linear = torch.nn.Linear(4, 4)
        # Build a throw-away class with the same short name but a different module.
        synthetic_linear_cls = type("Linear", (torch.nn.Module,), {})
        synthetic_linear_cls.__module__ = "tests.synthetic"
        model.user_linear = synthetic_linear_cls()

        paths = get_class_matched_module_paths(model, ["torch.nn.*.Linear"])
        assert paths == {"torch_linear"}

        paths = get_class_matched_module_paths(model, ["tests.synthetic.Linear"])
        assert paths == {"user_linear"}

        # Bare short-name pattern still matches both.
        paths = get_class_matched_module_paths(model, ["Linear"])
        assert paths == {"torch_linear", "user_linear"}
