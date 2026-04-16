# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadConfig and environment variable loading.

This test suite validates:
1. OffloadConfig class behavior (defaults, validation, properties)
2. Environment variable loading with type conversion
3. File-based configuration loading (INI, JSON, YAML)
4. Edge cases (case sensitivity, invalid values, etc.)
"""

import logging
import os
import warnings
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from flextensor.config import (
    OffloadConfig,
    _get_field_types,
    _parse_bool,
    _parse_none,
    load_config,
    load_config_from_env,
    load_config_from_file,
)
from flextensor.strategy import GreedyStrategy, KnapsackStrategy, NthLayerStrategy


class TestOffloadConfig:
    """Test OffloadConfig class behavior."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = OffloadConfig()
        assert config.gpu_device == 0
        assert config.pinned_memory is True
        assert config.shm_enabled is False
        assert config.discovery_iters == 1
        assert config.profiling_iters == 10
        assert config.transfer_budget_scale == 1.0
        assert config.transfer_mode == "allocation_block_transfer"
        assert config.num_blocks == 4
        assert config.profile_storage_dir is None
        assert config.profile_read_only is False
        assert config.load_strategy is None
        assert config.min_blocks == 4
        assert config.max_gpu_mem_fraction == 0.9
        with pytest.warns(DeprecationWarning):
            assert config.max_gpu_mem_bytes is None

    def test_max_gpu_mem_fraction_default(self):
        """Default max_gpu_mem_fraction is 0.9."""
        config = OffloadConfig()
        assert config.max_gpu_mem_fraction == 0.9

    def test_max_gpu_mem_fraction_none_latency_mode(self):
        """Setting max_gpu_mem_fraction=None enables latency mode."""
        config = OffloadConfig(max_gpu_mem_fraction=None)
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_custom(self):
        """Custom fraction is accepted."""
        config = OffloadConfig(max_gpu_mem_fraction=0.75)
        assert config.max_gpu_mem_fraction == 0.75

    def test_max_gpu_mem_fraction_boundary_values(self):
        """Boundary values: just above 0 and exactly 1.0 are valid."""
        config_low = OffloadConfig(max_gpu_mem_fraction=0.01)
        config_high = OffloadConfig(max_gpu_mem_fraction=1.0)
        assert config_low.max_gpu_mem_fraction == 0.01
        assert config_high.max_gpu_mem_fraction == 1.0

    def test_max_gpu_mem_fraction_zero_invalid(self):
        """max_gpu_mem_fraction=0.0 is rejected (gt=0.0 constraint)."""
        with pytest.raises(ValidationError):
            OffloadConfig(max_gpu_mem_fraction=0.0)

    def test_max_gpu_mem_fraction_above_one_invalid(self):
        """max_gpu_mem_fraction > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            OffloadConfig(max_gpu_mem_fraction=1.1)

    def test_max_gpu_mem_bytes_deprecated_emits_warning(self):
        """Setting max_gpu_mem_bytes emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="max_gpu_mem_bytes"):
            config = OffloadConfig(max_gpu_mem_bytes=10 * 1024**3)
            assert config.max_gpu_mem_bytes == 10 * 1024**3
        assert config.max_gpu_mem_fraction is None  # default suppressed

    def test_max_gpu_mem_bytes_and_fraction_both_set_raises(self):
        """Setting both fields raises ValueError."""
        with pytest.raises(ValidationError, match="Cannot set both"):
            OffloadConfig(max_gpu_mem_bytes=10 * 1024**3, max_gpu_mem_fraction=0.8)

    def test_custom_values(self):
        """Test setting custom values."""
        config = OffloadConfig(
            gpu_device=2,
            pinned_memory=False,
            discovery_iters=5,
            profiling_iters=20,
            transfer_budget_scale=2.0,
            transfer_mode="custom_mode",
            num_blocks=8,
        )
        assert config.gpu_device == 2
        assert config.pinned_memory is False
        assert config.discovery_iters == 5
        assert config.profiling_iters == 20
        assert config.transfer_budget_scale == 2.0
        assert config.transfer_mode == "custom_mode"
        assert config.num_blocks == 8

    def test_gpu_device_validation_negative(self):
        """Test that negative gpu_device raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(gpu_device=-1)

    def test_discovery_iters_validation_negative(self):
        """Test that negative discovery_iters raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(discovery_iters=-1)

    def test_profiling_iters_validation_negative(self):
        """Test that negative profiling_iters raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(profiling_iters=-1)

    def test_transfer_budget_scale_validation_zero(self):
        """Test that zero transfer_budget_scale raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(transfer_budget_scale=0.0)

    def test_transfer_budget_scale_validation_negative(self):
        """Test that negative transfer_budget_scale raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(transfer_budget_scale=-1.0)

    def test_num_blocks_validation_one(self):
        """Test that num_blocks=1 raises validation error (ge=2 constraint)."""
        with pytest.raises(ValidationError):
            OffloadConfig(num_blocks=1)

    def test_num_blocks_validation_zero(self):
        """Test that zero num_blocks raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(num_blocks=0)

    def test_num_blocks_validation_negative(self):
        """Test that negative num_blocks raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(num_blocks=-1)

    def test_min_blocks_custom(self):
        """Test setting custom min_blocks value."""
        config = OffloadConfig(min_blocks=3)
        assert config.min_blocks == 3

    def test_min_blocks_validation_below_minimum(self):
        """Test that min_blocks < 2 raises validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(min_blocks=1)

    def test_min_blocks_validation_at_minimum(self):
        """Test that min_blocks=2 is accepted."""
        config = OffloadConfig(min_blocks=2)
        assert config.min_blocks == 2

    def test_num_blocks_less_than_min_blocks_raises(self):
        """Test that num_blocks < min_blocks raises validation error."""
        with pytest.raises(ValidationError, match=r"num_blocks.*must be >= min_blocks"):
            OffloadConfig(num_blocks=2, min_blocks=4)

    def test_num_blocks_equal_min_blocks_accepted(self):
        """Test that num_blocks == min_blocks is accepted."""
        config = OffloadConfig(num_blocks=3, min_blocks=3)
        assert config.num_blocks == 3
        assert config.min_blocks == 3

    def test_max_gpu_mem_bytes_custom(self):
        """Test setting custom max_gpu_mem_bytes value."""
        with pytest.warns(DeprecationWarning):
            config = OffloadConfig(max_gpu_mem_bytes=48 * 1024**3)
            assert config.max_gpu_mem_bytes == 48 * 1024**3

    def test_max_gpu_mem_bytes_zero(self):
        """Test that max_gpu_mem_bytes=0 is accepted."""
        with pytest.warns(DeprecationWarning):
            config = OffloadConfig(max_gpu_mem_bytes=0)
            assert config.max_gpu_mem_bytes == 0

    def test_max_gpu_mem_bytes_validation_negative(self):
        """Test that negative max_gpu_mem_bytes raises validation error."""
        with pytest.raises(ValidationError), pytest.warns(DeprecationWarning):
            OffloadConfig(max_gpu_mem_bytes=-1)

    def test_pre_inference_iters_property(self):
        """Test pre_inference_iters property calculation."""
        config = OffloadConfig(discovery_iters=3, profiling_iters=7)
        assert config.pre_inference_iters == 10

    def test_pre_inference_iters_property_default(self):
        """Test pre_inference_iters property with default values."""
        config = OffloadConfig()
        assert config.pre_inference_iters == 11  # 1 + 10

    def test_knapsack_strategy_assignment(self):
        """Test assigning KnapsackStrategy to load_strategy."""
        strategy = KnapsackStrategy(scale=1.5, cyclic=True, group_size=2)
        config = OffloadConfig(load_strategy=strategy)
        assert config.load_strategy is strategy
        assert isinstance(config.load_strategy, KnapsackStrategy)

    def test_greedy_strategy_assignment(self):
        """Test assigning GreedyStrategy to load_strategy."""
        strategy = GreedyStrategy(threshold_mb=0.2)
        config = OffloadConfig(load_strategy=strategy)
        assert config.load_strategy is strategy
        assert isinstance(config.load_strategy, GreedyStrategy)

    def test_nth_layer_strategy_assignment(self):
        """Test assigning NthLayerStrategy to load_strategy."""
        strategy = NthLayerStrategy(nth_layer=2, threshold_mb=0.2)
        config = OffloadConfig(load_strategy=strategy)
        assert config.load_strategy is strategy
        assert isinstance(config.load_strategy, NthLayerStrategy)

    def test_edge_values_discovery_iters_zero(self):
        """Test discovery_iters at boundary value 0."""
        config = OffloadConfig(discovery_iters=0)
        assert config.discovery_iters == 0

    def test_edge_values_profiling_iters_zero(self):
        """Test profiling_iters at boundary value 0."""
        config = OffloadConfig(profiling_iters=0)
        assert config.profiling_iters == 0

    def test_include_patterns_default(self):
        """Test include_patterns default value is ['*']."""
        config = OffloadConfig()
        assert config.include_patterns == ["*"]

    def test_include_patterns_custom(self):
        """Test setting custom include_patterns."""
        config = OffloadConfig(include_patterns=["layers.*", "head"])
        assert config.include_patterns == ["layers.*", "head"]

    def test_include_patterns_empty_list_accepted(self):
        """Empty include_patterns is accepted for manual offload_block() usage."""
        config = OffloadConfig(include_patterns=[])
        assert config.include_patterns == []

    def test_exclude_patterns_default(self):
        """Test exclude_patterns default value is []."""
        config = OffloadConfig()
        assert config.exclude_patterns == []

    def test_exclude_patterns_custom(self):
        """Test setting custom exclude_patterns."""
        config = OffloadConfig(exclude_patterns=["lm_head", "*.norm"])
        assert config.exclude_patterns == ["lm_head", "*.norm"]

    def test_exclude_patterns_empty_list(self):
        """Test that empty exclude_patterns list is allowed."""
        config = OffloadConfig(exclude_patterns=[])
        assert config.exclude_patterns == []

    def test_json_schema_has_field_descriptions(self):
        """All OffloadConfig fields must have descriptions in JSON Schema."""
        schema = OffloadConfig.model_json_schema()
        properties = schema["properties"]
        missing = [name for name, prop in properties.items() if "description" not in prop]
        assert missing == [], (
            f"Fields missing 'description' in JSON Schema: {missing}. "
            "Each field must have an attribute docstring and model_config must have "
            "use_attribute_docstrings=True."
        )


class TestParseBool:
    """Test _parse_bool helper function."""

    def test_parse_true_lowercase(self):
        """Test parsing 'true' (lowercase)."""
        assert _parse_bool("true") is True

    def test_parse_true_uppercase(self):
        """Test parsing 'TRUE' (uppercase)."""
        assert _parse_bool("TRUE") is True

    def test_parse_true_mixed_case(self):
        """Test parsing 'True' (mixed case)."""
        assert _parse_bool("True") is True

    def test_parse_one(self):
        """Test parsing '1'."""
        assert _parse_bool("1") is True

    def test_parse_yes_lowercase(self):
        """Test parsing 'yes' (lowercase)."""
        assert _parse_bool("yes") is True

    def test_parse_yes_uppercase(self):
        """Test parsing 'YES' (uppercase)."""
        assert _parse_bool("YES") is True

    def test_parse_yes_mixed_case(self):
        """Test parsing 'Yes' (mixed case)."""
        assert _parse_bool("Yes") is True

    def test_parse_y(self):
        """Test parsing 'y'."""
        assert _parse_bool("y") is True

    def test_parse_on(self):
        """Test parsing 'on'."""
        assert _parse_bool("on") is True

    def test_parse_false_lowercase(self):
        """Test parsing 'false' (lowercase)."""
        assert _parse_bool("false") is False

    def test_parse_false_uppercase(self):
        """Test parsing 'FALSE' (uppercase)."""
        assert _parse_bool("FALSE") is False

    def test_parse_false_mixed_case(self):
        """Test parsing 'False' (mixed case)."""
        assert _parse_bool("False") is False

    def test_parse_zero(self):
        """Test parsing '0'."""
        assert _parse_bool("0") is False

    def test_parse_no_lowercase(self):
        """Test parsing 'no' (lowercase)."""
        assert _parse_bool("no") is False

    def test_parse_no_uppercase(self):
        """Test parsing 'NO' (uppercase)."""
        assert _parse_bool("NO") is False

    def test_parse_no_mixed_case(self):
        """Test parsing 'No' (mixed case)."""
        assert _parse_bool("No") is False

    def test_parse_n(self):
        """Test parsing 'n'."""
        assert _parse_bool("n") is False

    def test_parse_off(self):
        """Test parsing 'off'."""
        assert _parse_bool("off") is False

    def test_parse_invalid_value(self):
        """Test parsing invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse 'invalid' as boolean"):
            _parse_bool("invalid")

    def test_parse_empty_string(self):
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_bool("")


class TestParseNone:
    """Test _parse_none helper function."""

    def test_parse_none_lowercase(self):
        """Test parsing 'none' (lowercase)."""
        assert _parse_none("none") is None

    def test_parse_none_uppercase(self):
        """Test parsing 'NONE' (uppercase)."""
        assert _parse_none("NONE") is None

    def test_parse_none_mixed_case(self):
        """Test parsing 'None' (mixed case)."""
        assert _parse_none("None") is None

    def test_parse_null_lowercase(self):
        """Test parsing 'null' (lowercase)."""
        assert _parse_none("null") is None

    def test_parse_null_uppercase(self):
        """Test parsing 'NULL' (uppercase)."""
        assert _parse_none("NULL") is None

    def test_parse_null_mixed_case(self):
        """Test parsing 'Null' (mixed case)."""
        assert _parse_none("Null") is None

    def test_parse_invalid_value(self):
        """Test parsing invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse 'invalid' as None"):
            _parse_none("invalid")


class TestLoadConfigFromEnv:
    """Test load_config_from_env function."""

    def setup_method(self):
        """Clear relevant environment variables before each test."""
        # Save original environment
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_load_with_no_env_vars(self):
        """Test loading config with no environment variables uses defaults."""
        config = load_config_from_env()
        assert config.gpu_device == 0
        assert config.discovery_iters == 1
        assert config.profiling_iters == 10

    def test_enabled_default_false(self):
        """Test that enabled defaults to False when loading from env."""
        config = load_config_from_env()
        assert config.enabled is False

    def test_enabled_explicitly_set_true(self):
        """Test that enabled can be explicitly set to True via env var."""
        os.environ["FT_ENABLED"] = "1"
        config = load_config_from_env()
        assert config.enabled is True

    def test_enabled_explicitly_set_false(self):
        """Test that enabled can be explicitly set to False via env var."""
        os.environ["FT_ENABLED"] = "0"
        config = load_config_from_env()
        assert config.enabled is False

    def test_enabled_override_with_kwargs(self):
        """Test that enabled can be overridden with kwargs."""
        config = load_config_from_env(enabled=True)
        assert config.enabled is True

    def test_load_int_field(self):
        """Test loading integer field from environment."""
        os.environ["FT_GPU_DEVICE"] = "2"
        config = load_config_from_env()
        assert config.gpu_device == 2

    def test_load_bool_field_true(self):
        """Test loading boolean field (True) from environment."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "true"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

    def test_load_bool_field_false(self):
        """Test loading boolean field (False) from environment."""
        os.environ["FT_PINNED_MEMORY"] = "false"
        config = load_config_from_env()
        assert config.pinned_memory is False

    def test_load_bool_field_case_insensitive(self):
        """Test loading boolean field with different cases."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "TRUE"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

        os.environ["FT_ENABLE_DIAGNOSTICS"] = "False"
        config = load_config_from_env()
        assert config.enable_diagnostics is False

    def test_load_float_field(self):
        """Test loading float field from environment."""
        os.environ["FT_TRANSFER_BUDGET_SCALE"] = "2.5"
        config = load_config_from_env()
        assert config.transfer_budget_scale == 2.5

    def test_load_string_field(self):
        """Test loading string field from environment."""
        os.environ["FT_TRANSFER_MODE"] = "custom_transfer"
        config = load_config_from_env()
        assert config.transfer_mode == "custom_transfer"

    def test_load_multiple_fields(self):
        """Test loading multiple fields from environment."""
        os.environ["FT_GPU_DEVICE"] = "1"
        os.environ["FT_DISCOVERY_ITERS"] = "5"
        os.environ["FT_PROFILING_ITERS"] = "15"
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "true"

        config = load_config_from_env()
        assert config.gpu_device == 1
        assert config.discovery_iters == 5
        assert config.profiling_iters == 15
        assert config.enable_diagnostics is True

    def test_load_with_custom_prefix(self):
        """Test loading config with custom prefix."""
        os.environ["CUSTOM_GPU_DEVICE"] = "3"
        os.environ["CUSTOM_WARMUP_ITERS"] = "7"

        config = load_config_from_env(prefix="CUSTOM_")
        assert config.gpu_device == 3
        assert config.discovery_iters == 7

    def test_load_partial_config(self):
        """Test loading config with only some environment variables set."""
        os.environ["FT_GPU_DEVICE"] = "2"
        # Other fields should use defaults
        config = load_config_from_env()
        assert config.gpu_device == 2
        assert config.discovery_iters == 1  # default
        assert config.profiling_iters == 10  # default

    def test_kwargs_override_env_vars(self):
        """Test that kwargs override environment variables."""
        os.environ["FT_GPU_DEVICE"] = "1"
        os.environ["FT_DISCOVERY_ITERS"] = "5"

        config = load_config_from_env(gpu_device=2, discovery_iters=10)
        assert config.gpu_device == 2  # kwargs override
        assert config.discovery_iters == 10  # kwargs override

    def test_invalid_int_raises_error(self):
        """Test that invalid integer value raises error."""
        os.environ["FT_GPU_DEVICE"] = "not_an_int"

        with pytest.raises(ValueError, match="Failed to convert FT_GPU_DEVICE"):
            load_config_from_env()

    def test_invalid_float_raises_error(self):
        """Test that invalid float value raises error."""
        os.environ["FT_TRANSFER_BUDGET_SCALE"] = "not_a_float"

        with pytest.raises(ValueError, match="Failed to convert FT_TRANSFER_BUDGET_SCALE"):
            load_config_from_env()

    def test_invalid_bool_raises_error(self):
        """Test that invalid boolean value raises error."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "not_a_bool"

        with pytest.raises(ValueError, match="Failed to convert FT_ENABLE_DIAGNOSTICS"):
            load_config_from_env()

    def test_validation_error_propagates(self):
        """Test that pydantic validation errors are raised."""
        os.environ["FT_GPU_DEVICE"] = "-1"  # Invalid: must be >= 0

        with pytest.raises(ValidationError):
            load_config_from_env()

    def test_all_bool_fields(self):
        """Test loading all boolean fields from environment."""
        os.environ["FT_PINNED_MEMORY"] = "false"
        os.environ["FT_ENABLE_INSTRUMENTATION"] = "true"
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "true"

        config = load_config_from_env()
        assert config.pinned_memory is False
        assert config.enable_instrumentation is True
        assert config.enable_diagnostics is True

    def test_min_blocks_from_env(self):
        """Test loading min_blocks from FT_MIN_BLOCKS env var."""
        os.environ["FT_MIN_BLOCKS"] = "3"
        config = load_config_from_env()
        assert config.min_blocks == 3

    def test_max_gpu_mem_bytes_from_env(self):
        """Test loading max_gpu_mem_bytes from FT_MAX_GPU_MEM_BYTES env var."""
        os.environ["FT_MAX_GPU_MEM_BYTES"] = str(48 * 1024**3)
        with pytest.warns(DeprecationWarning, match="max_gpu_mem_bytes"):
            config = load_config_from_env()
            assert config.max_gpu_mem_bytes == 48 * 1024**3
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_bytes_none_from_env(self):
        """Test setting max_gpu_mem_bytes to None via FT_MAX_GPU_MEM_BYTES=none."""
        os.environ["FT_MAX_GPU_MEM_BYTES"] = "none"
        with pytest.warns(DeprecationWarning, match="max_gpu_mem_bytes"):
            config = load_config_from_env()
            assert config.max_gpu_mem_bytes is None

    def test_max_gpu_mem_fraction_from_env(self):
        """Test loading max_gpu_mem_fraction from FT_MAX_GPU_MEM_FRACTION env var."""
        os.environ["FT_MAX_GPU_MEM_FRACTION"] = "0.75"
        config = load_config_from_env()
        assert config.max_gpu_mem_fraction == 0.75

    def test_max_gpu_mem_fraction_none_from_env(self):
        """Test setting max_gpu_mem_fraction to None via FT_MAX_GPU_MEM_FRACTION=none."""
        os.environ["FT_MAX_GPU_MEM_FRACTION"] = "none"
        config = load_config_from_env()
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_bytes_and_fraction_env_both_set_raises(self):
        """Setting both FT_MAX_GPU_MEM_BYTES and FT_MAX_GPU_MEM_FRACTION raises."""
        os.environ["FT_MAX_GPU_MEM_BYTES"] = str(20 * 1024**3)
        os.environ["FT_MAX_GPU_MEM_FRACTION"] = "0.8"
        with pytest.raises(ValidationError, match="Cannot set both"):
            load_config_from_env()

    def test_all_int_fields(self):
        """Test loading all integer fields from environment."""
        os.environ["FT_GPU_DEVICE"] = "3"
        os.environ["FT_DISCOVERY_ITERS"] = "8"
        os.environ["FT_PROFILING_ITERS"] = "25"
        os.environ["FT_NUM_BLOCKS"] = "10"
        os.environ["FT_MIN_BLOCKS"] = "3"

        config = load_config_from_env()
        assert config.gpu_device == 3
        assert config.discovery_iters == 8
        assert config.profiling_iters == 25
        assert config.num_blocks == 10
        assert config.min_blocks == 3

    def test_all_float_fields(self):
        """Test loading all float fields from environment."""
        os.environ["FT_TRANSFER_BUDGET_SCALE"] = "2.5"

        config = load_config_from_env()
        assert config.transfer_budget_scale == 2.5

    def test_bool_variations_yes_no(self):
        """Test boolean parsing with 'yes' and 'no'."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "yes"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

        os.environ["FT_ENABLE_DIAGNOSTICS"] = "no"
        config = load_config_from_env()
        assert config.enable_diagnostics is False

    def test_bool_variations_one_zero(self):
        """Test boolean parsing with '1' and '0'."""
        os.environ["FT_ENABLE_DIAGNOSTICS"] = "1"
        config = load_config_from_env()
        assert config.enable_diagnostics is True

        os.environ["FT_ENABLE_DIAGNOSTICS"] = "0"
        config = load_config_from_env()
        assert config.enable_diagnostics is False

    def test_load_strategy_field_skipped(self):
        """Test that load_strategy field is skipped (complex type)."""
        # load_strategy is a complex type that can't be set via env vars
        os.environ["FT_LOAD_STRATEGY"] = "something"
        config = load_config_from_env()
        # Should ignore the env var and use default (None)
        assert config.load_strategy is None

    def test_mixed_env_and_kwargs_with_validation(self):
        """Test mixed environment and kwargs with validation."""
        os.environ["FT_GPU_DEVICE"] = "1"

        config = load_config_from_env(discovery_iters=20, enable_diagnostics=True)
        assert config.gpu_device == 1  # from env
        assert config.discovery_iters == 20  # from kwargs
        assert config.enable_diagnostics is True  # from kwargs

    def test_include_patterns_from_env(self):
        """Test loading include_patterns from FT_INCLUDE_PATTERNS env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*,head"
        config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "head"]

    def test_include_patterns_from_env_with_spaces(self):
        """Test loading include_patterns with spaces in env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*, head , model.norm"
        config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "head", "model.norm"]

    def test_include_patterns_from_env_single(self):
        """Test loading single module pattern from env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "model.*"
        config = load_config_from_env()
        assert config.include_patterns == ["model.*"]

    def test_include_patterns_from_env_wildcard(self):
        """Test loading wildcard module pattern from env var."""
        os.environ["FT_INCLUDE_PATTERNS"] = "*"
        config = load_config_from_env()
        assert config.include_patterns == ["*"]

    def test_include_patterns_kwargs_override_env(self):
        """Test that kwargs override env var for include_patterns."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*"
        config = load_config_from_env(include_patterns=["custom.*"])
        assert config.include_patterns == ["custom.*"]

    def test_include_patterns_none_literal_kept_as_string(self):
        """Ensure 'none' in a list[str] env var stays a string, not Python None."""
        os.environ["FT_INCLUDE_PATTERNS"] = "layers.*,none,head"
        config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "none", "head"]

    def test_exclude_patterns_none_literal_kept_as_string(self):
        """Ensure 'none' in exclude_patterns env var stays a string."""
        os.environ["FT_EXCLUDE_PATTERNS"] = "none,lm_head"
        config = load_config_from_env()
        assert config.exclude_patterns == ["none", "lm_head"]

    def test_str_none_field_parses_none_as_python_none(self):
        """Ensure 'none' in a top-level str|None field becomes Python None."""
        os.environ["FT_SHM_NAMESPACE"] = "none"
        config = load_config_from_env()
        assert config.shm_namespace is None

    def test_exclude_patterns_from_env(self):
        """Test loading exclude_patterns from FT_EXCLUDE_PATTERNS env var."""
        os.environ["FT_EXCLUDE_PATTERNS"] = "lm_head,*.norm"
        config = load_config_from_env()
        assert config.exclude_patterns == ["lm_head", "*.norm"]

    def test_exclude_patterns_from_env_single(self):
        """Test loading single exclude pattern from env var."""
        os.environ["FT_EXCLUDE_PATTERNS"] = "lm_head"
        config = load_config_from_env()
        assert config.exclude_patterns == ["lm_head"]


class TestLoadConfigFromFile:
    """Test load_config_from_file function."""

    def setup_method(self):
        """Clear relevant environment variables before each test."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_load_ini_file(self, tmp_path):
        """Test loading config from INI file."""
        config_file = tmp_path / "test.conf"
        config_file.write_text("""[flextensor]
enabled = true
gpu_device = 2
discovery_iters = 5
enable_diagnostics = true
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 2
        assert config.discovery_iters == 5
        assert config.enable_diagnostics is True

    def test_load_ini_file_with_ini_extension(self, tmp_path):
        """Test loading config from .ini file."""
        config_file = tmp_path / "test.ini"
        config_file.write_text("""[flextensor]
enabled = true
gpu_device = 3
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 3

    def test_load_json_file(self, tmp_path):
        """Test loading config from JSON file."""
        config_file = tmp_path / "test.json"
        config_file.write_text("""{
    "enabled": true,
    "gpu_device": 4,
    "discovery_iters": 8,
    "enable_diagnostics": false
}""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 4
        assert config.discovery_iters == 8
        assert config.enable_diagnostics is False

    def test_load_json_file_with_patterns(self, tmp_path):
        """Test loading include_patterns and exclude_patterns from JSON file."""
        config_file = tmp_path / "test_patterns.json"
        config_file.write_text("""{
    "enabled": true,
    "include_patterns": ["layers.*", "head"],
    "exclude_patterns": ["layers.*.norm"]
}""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head"]
        assert config.exclude_patterns == ["layers.*.norm"]

    def test_load_yaml_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 5
discovery_iters: 10
enable_diagnostics: true
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 5
        assert config.discovery_iters == 10
        assert config.enable_diagnostics is True

    def test_load_yaml_file_with_patterns(self, tmp_path):
        """Test loading include_patterns and exclude_patterns from YAML file."""
        config_file = tmp_path / "test_patterns.yaml"
        config_file.write_text("""enabled: true
include_patterns:
  - "layers.*"
  - "embed_tokens"
exclude_patterns:
  - "layers.*.norm"
  - "lm_head"
""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "embed_tokens"]
        assert config.exclude_patterns == ["layers.*.norm", "lm_head"]

    def test_load_yml_file(self, tmp_path):
        """Test loading config from .yml file."""
        config_file = tmp_path / "test.yml"
        config_file.write_text("""enabled: false
gpu_device: 6
""")
        config = load_config_from_file(config_file)
        assert config.enabled is False
        assert config.gpu_device == 6

    def test_load_from_env_var(self, tmp_path):
        """Test loading config from FT_CONFIG_FILE env var."""
        config_file = tmp_path / "env_test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 7
""")
        os.environ["FT_CONFIG_FILE"] = str(config_file)
        config = load_config_from_file()
        assert config.enabled is True
        assert config.gpu_device == 7

    def test_load_from_custom_env_var(self, tmp_path):
        """Test loading config from custom env var."""
        config_file = tmp_path / "custom_env.yaml"
        config_file.write_text("""enabled: true
gpu_device: 8
""")
        os.environ["MY_CONFIG_FILE"] = str(config_file)
        config = load_config_from_file(env_var="MY_CONFIG_FILE")
        assert config.enabled is True
        assert config.gpu_device == 8

    def test_kwargs_override_file_values(self, tmp_path):
        """Test that kwargs override file values."""
        config_file = tmp_path / "override.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        config = load_config_from_file(config_file, gpu_device=10, discovery_iters=20)
        assert config.enabled is True  # from file
        assert config.gpu_device == 10  # from kwargs
        assert config.discovery_iters == 20  # from kwargs

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config_from_file("/nonexistent/path/config.yaml")

    def test_no_config_specified(self):
        """Test that FileNotFoundError is raised when no config specified."""
        with pytest.raises(FileNotFoundError, match="No config file specified"):
            load_config_from_file()

    def test_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported format."""
        config_file = tmp_path / "test.txt"
        config_file.write_text("some content")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config_from_file(config_file)

    def test_ini_missing_section(self, tmp_path):
        """Test that ValueError is raised for INI file without [flextensor] section."""
        config_file = tmp_path / "bad.conf"
        config_file.write_text("""[other_section]
gpu_device = 1
""")
        with pytest.raises(ValueError, match="must contain"):
            load_config_from_file(config_file)

    def test_partial_config_file(self, tmp_path):
        """Test loading file with only some fields uses defaults for rest."""
        config_file = tmp_path / "partial.yaml"
        config_file.write_text("""gpu_device: 3
""")
        config = load_config_from_file(config_file)
        assert config.gpu_device == 3
        assert config.discovery_iters == 1  # default
        assert config.enabled is True  # default (not False like env loading)

    def test_json_with_string_values(self, tmp_path):
        """Test loading JSON with string values that need conversion."""
        config_file = tmp_path / "string_vals.json"
        config_file.write_text("""{
    "enabled": "true",
    "gpu_device": "5",
    "transfer_budget_scale": "1.5"
}""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 5
        assert config.transfer_budget_scale == 1.5

    def test_yaml_with_string_values(self, tmp_path):
        """Test loading YAML with string values that need conversion."""
        config_file = tmp_path / "string_vals.yaml"
        config_file.write_text("""enabled: "yes"
gpu_device: "6"
transfer_budget_scale: "1.5"
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 6
        assert config.transfer_budget_scale == 1.5

    def test_empty_yaml_file(self, tmp_path):
        """Test loading empty YAML file uses all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        config = load_config_from_file(config_file)
        assert config.gpu_device == 0  # default
        assert config.enabled is True  # default

    def test_path_object(self, tmp_path):
        """Test that Path object works as config_path."""
        config_file = tmp_path / "path_test.yaml"
        config_file.write_text("""gpu_device: 9
""")
        config = load_config_from_file(Path(config_file))
        assert config.gpu_device == 9

    def test_string_path(self, tmp_path):
        """Test that string path works as config_path."""
        config_file = tmp_path / "string_test.yaml"
        config_file.write_text("""gpu_device: 11
""")
        config = load_config_from_file(str(config_file))
        assert config.gpu_device == 11

    def test_all_fields_ini(self, tmp_path):
        """Test loading all supported fields from INI file."""
        config_file = tmp_path / "all_fields.conf"
        config_file.write_text("""[flextensor]
enabled = true
gpu_device = 1
pinned_memory = false
discovery_iters = 3
profiling_iters = 15
transfer_budget_scale = 2.0
transfer_mode = custom_mode
num_blocks = 8
""")
        config = load_config_from_file(config_file)
        assert config.enabled is True
        assert config.gpu_device == 1
        assert config.pinned_memory is False
        assert config.discovery_iters == 3
        assert config.profiling_iters == 15
        assert config.transfer_budget_scale == 2.0
        assert config.transfer_mode == "custom_mode"
        assert config.num_blocks == 8

    def test_validation_error_from_file(self, tmp_path):
        """Test that pydantic validation errors are raised for invalid values in file."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""gpu_device: -1
""")
        with pytest.raises(ValidationError):
            load_config_from_file(config_file)

    def test_min_blocks_from_yaml_file(self, tmp_path):
        """Test loading min_blocks from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""min_blocks: 3
""")
        config = load_config_from_file(config_file)
        assert config.min_blocks == 3

    def test_max_gpu_mem_bytes_from_yaml_file(self, tmp_path):
        """Test loading max_gpu_mem_bytes from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""max_gpu_mem_bytes: 51539607552
""")
        with pytest.warns(DeprecationWarning, match="max_gpu_mem_bytes"):
            config = load_config_from_file(config_file)
            assert config.max_gpu_mem_bytes == 48 * 1024**3
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_bytes_null_from_yaml_file(self, tmp_path):
        """Test that max_gpu_mem_bytes defaults to None when absent from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""gpu_device: 0
""")
        config = load_config_from_file(config_file)
        with pytest.warns(DeprecationWarning):
            assert config.max_gpu_mem_bytes is None

    def test_min_blocks_and_max_gpu_mem_bytes_from_json_file(self, tmp_path):
        """Test loading min_blocks and max_gpu_mem_bytes from JSON file."""
        config_file = tmp_path / "test.json"
        config_file.write_text("""{
    "min_blocks": 2,
    "max_gpu_mem_bytes": 85899345920
}""")
        with pytest.warns(DeprecationWarning, match="max_gpu_mem_bytes"):
            config = load_config_from_file(config_file)
            assert config.max_gpu_mem_bytes == 80 * 1024**3
        assert config.min_blocks == 2
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_from_yaml_file(self, tmp_path):
        """Test loading max_gpu_mem_fraction from YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("max_gpu_mem_fraction: 0.8\n")
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction == 0.8

    def test_max_gpu_mem_fraction_none_from_yaml_file(self, tmp_path):
        """Test that max_gpu_mem_fraction: null in YAML file results in None."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("max_gpu_mem_fraction: null\n")
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction is None

    def test_max_gpu_mem_fraction_default_when_absent_from_yaml(self, tmp_path):
        """Test that max_gpu_mem_fraction defaults to 0.9 when not in YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("gpu_device: 0\n")
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction == 0.9

    def test_max_gpu_mem_fraction_from_json_file(self, tmp_path):
        """Test loading max_gpu_mem_fraction from JSON file."""
        config_file = tmp_path / "test.json"
        config_file.write_text('{"max_gpu_mem_fraction": 0.6}')
        config = load_config_from_file(config_file)
        assert config.max_gpu_mem_fraction == 0.6

    def test_include_patterns_from_yaml_file(self, tmp_path):
        """Test loading include_patterns from YAML file."""
        config_file = tmp_path / "patterns.yaml"
        config_file.write_text("""include_patterns:
  - "layers.*"
  - "head"
  - "norm"
""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head", "norm"]

    def test_include_patterns_from_json_file(self, tmp_path):
        """Test loading include_patterns from JSON file."""
        config_file = tmp_path / "patterns.json"
        config_file.write_text("""{
    "include_patterns": ["model.*", "lm_head"]
}""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["model.*", "lm_head"]

    def test_include_patterns_default_when_not_in_file(self, tmp_path):
        """Test that include_patterns uses default when not in config file."""
        config_file = tmp_path / "no_patterns.yaml"
        config_file.write_text("""gpu_device: 1
""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["*"]  # default

    def test_exclude_patterns_from_yaml_file(self, tmp_path):
        """Test loading exclude_patterns from YAML file."""
        config_file = tmp_path / "exclude.yaml"
        config_file.write_text("""exclude_patterns:
  - "lm_head"
  - "*.norm"
""")
        config = load_config_from_file(config_file)
        assert config.exclude_patterns == ["lm_head", "*.norm"]

    def test_exclude_patterns_from_json_file(self, tmp_path):
        """Test loading exclude_patterns from JSON file."""
        config_file = tmp_path / "exclude.json"
        config_file.write_text("""{
    "include_patterns": ["*"],
    "exclude_patterns": ["lm_head", "*.scale"]
}""")
        config = load_config_from_file(config_file)
        assert config.include_patterns == ["*"]
        assert config.exclude_patterns == ["lm_head", "*.scale"]


class TestGetFieldTypes:
    """Test _get_field_types helper function."""

    def test_returns_dict(self):
        """Test that _get_field_types returns a dictionary."""
        field_types = _get_field_types()
        assert isinstance(field_types, dict)

    def test_contains_all_fields(self):
        """Test that all OffloadConfig fields are present."""
        field_types = _get_field_types()
        for field_name in OffloadConfig.model_fields:
            assert field_name in field_types

    def test_bool_fields(self):
        """Test that boolean fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["enabled"] is bool
        assert field_types["pinned_memory"] is bool
        assert field_types["enable_instrumentation"] is bool

    def test_int_fields(self):
        """Test that integer fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["gpu_device"] is int
        assert field_types["discovery_iters"] is int
        assert field_types["num_blocks"] is int
        assert field_types["min_blocks"] is int

    def test_optional_int_fields(self):
        """Test that optional int fields (int | None) are correctly identified as int."""
        field_types = _get_field_types()
        assert field_types["max_gpu_mem_bytes"] is int

    def test_float_fields(self):
        """Test that float fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["transfer_budget_scale"] is float
        assert field_types["max_gpu_mem_fraction"] is float

    def test_str_fields(self):
        """Test that string fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["transfer_mode"] is str

    def test_complex_fields(self):
        """Test that complex fields (like strategies) are marked as object."""
        field_types = _get_field_types()
        assert field_types["load_strategy"] is object

    def test_list_fields(self):
        """Test that list fields are correctly identified."""
        field_types = _get_field_types()
        assert field_types["include_patterns"] is list
        assert field_types["exclude_patterns"] is list


class TestLoadConfig:
    """Test unified load_config function with file + env + kwargs precedence."""

    def setup_method(self):
        """Clear relevant environment variables before each test."""
        self.original_env = os.environ.copy()

    def teardown_method(self):
        """Restore original environment after each test."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_env_only_no_file(self):
        """Test loading from env only when no file is specified."""
        os.environ["FT_GPU_DEVICE"] = "5"
        config = load_config()
        assert config.gpu_device == 5
        assert config.enabled is False  # default when env-only

    def test_env_only_enabled_default_false(self):
        """Test that enabled defaults to False when loading from env only."""
        config = load_config()
        assert config.enabled is False

    def test_env_override_file(self, tmp_path):
        """Test that env vars override file values."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        os.environ["FT_GPU_DEVICE"] = "10"

        config = load_config(config_path=config_file)
        assert config.enabled is True  # from file
        assert config.gpu_device == 10  # from env (overrides file)
        assert config.discovery_iters == 5  # from file

    def test_kwargs_override_env_and_file(self, tmp_path):
        """Test that kwargs override both env and file values."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        os.environ["FT_GPU_DEVICE"] = "10"
        os.environ["FT_DISCOVERY_ITERS"] = "15"

        config = load_config(config_path=config_file, gpu_device=20, discovery_iters=25)
        assert config.enabled is True  # from file
        assert config.gpu_device == 20  # from kwargs (overrides env and file)
        assert config.discovery_iters == 25  # from kwargs (overrides env and file)

    def test_file_with_use_env_false(self, tmp_path):
        """Test file loading without env override when use_env=False."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
""")
        os.environ["FT_GPU_DEVICE"] = "10"

        config = load_config(config_path=config_file, use_env=False)
        assert config.enabled is True  # from file
        assert config.gpu_device == 1  # from file (env not used)

    def test_file_from_env_var(self, tmp_path):
        """Test loading file path from FT_CONFIG_FILE env var."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 7
""")
        os.environ["FT_CONFIG_FILE"] = str(config_file)

        config = load_config()
        assert config.enabled is True
        assert config.gpu_device == 7

    def test_file_from_env_var_with_env_override(self, tmp_path):
        """Test file from env var with additional env overrides."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: true
gpu_device: 1
discovery_iters: 5
""")
        os.environ["FT_CONFIG_FILE"] = str(config_file)
        os.environ["FT_GPU_DEVICE"] = "10"

        config = load_config()
        assert config.enabled is True  # from file
        assert config.gpu_device == 10  # from env (overrides file)
        assert config.discovery_iters == 5  # from file

    def test_custom_env_prefix(self, tmp_path):
        """Test using custom env prefix."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""gpu_device: 1
""")
        os.environ["CUSTOM_GPU_DEVICE"] = "20"

        config = load_config(config_path=config_file, env_prefix="CUSTOM_")
        assert config.gpu_device == 20

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_path="/nonexistent/path.yaml")

    def test_file_not_found_from_env_var(self, tmp_path):
        """Test FileNotFoundError when FT_CONFIG_FILE points to missing file."""
        os.environ["FT_CONFIG_FILE"] = "/nonexistent/path.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config()

    def test_full_precedence_chain(self, tmp_path):
        """Test full precedence: file < env < kwargs."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: false
gpu_device: 1
discovery_iters: 5
profiling_iters: 10
""")
        os.environ["FT_GPU_DEVICE"] = "2"
        os.environ["FT_DISCOVERY_ITERS"] = "10"
        os.environ["FT_PROFILING_ITERS"] = "20"

        config = load_config(
            config_path=config_file,
            discovery_iters=15,
            profiling_iters=25,
        )

        # enabled: from file (not overridden)
        assert config.enabled is False
        # gpu_device: from env (overrides file)
        assert config.gpu_device == 2
        # discovery_iters: from kwargs (overrides env and file)
        assert config.discovery_iters == 15
        # profiling_iters: from kwargs (overrides env and file)
        assert config.profiling_iters == 25

    def test_enabled_from_env_overrides_file(self, tmp_path):
        """Test that FT_ENABLED from env overrides file value."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""enabled: false
""")
        os.environ["FT_ENABLED"] = "1"

        config = load_config(config_path=config_file)
        assert config.enabled is True  # from env (overrides file)


class TestShmConfigFields:
    """Test new SHM config fields on OffloadConfig."""

    def test_shm_config_fields_exist(self):
        """New SHM config fields have correct defaults."""
        config = OffloadConfig()
        assert config.shm_enabled is False
        assert config.shm_namespace is None
        assert config.shm_wait_timeout == 0.0

    def test_shm_enabled_replaces_use_shared_memory(self):
        """shm_enabled is the canonical field; use_shared_memory is deprecated alias."""
        config = OffloadConfig(shm_enabled=True)
        assert config.shm_enabled is True
        # Backward compat: use_shared_memory still readable (access warns, that's expected)
        with pytest.warns(DeprecationWarning):
            assert config.use_shared_memory is True

    def test_use_shared_memory_sets_shm_enabled(self):
        """Deprecated use_shared_memory sets shm_enabled."""
        with pytest.warns(DeprecationWarning):
            config = OffloadConfig(use_shared_memory=True)
        assert config.shm_enabled is True

    def test_use_shared_memory_warns_on_construction(self):
        """Passing use_shared_memory emits DeprecationWarning at construction time."""
        with pytest.warns(DeprecationWarning, match="use_shared_memory"):
            OffloadConfig(use_shared_memory=True)

    def test_use_shared_memory_warns_on_access(self):
        """Reading use_shared_memory on an instance emits DeprecationWarning."""
        config = OffloadConfig(shm_enabled=True)
        with pytest.warns(DeprecationWarning, match="use_shared_memory"):
            _ = config.use_shared_memory

    def test_shm_enabled_construction_no_warning(self):
        """Constructing with shm_enabled (the new field) emits no deprecation warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            OffloadConfig(shm_enabled=True)  # must not raise

    def test_shm_env_vars_loaded(self, monkeypatch):
        """FT_SHM_* env vars populate config fields."""
        monkeypatch.setenv("FT_SHM_ENABLED", "1")
        monkeypatch.setenv("FT_SHM_NAMESPACE", "my_model")
        monkeypatch.setenv("FT_SHM_WAIT_TIMEOUT", "300")
        config = load_config_from_env()
        assert config.shm_enabled is True
        assert config.shm_namespace == "my_model"
        assert config.shm_wait_timeout == 300.0


class TestModulePatternsDeprecation:
    """Tests for deprecated module_patterns → include_patterns migration."""

    def test_module_patterns_maps_to_include_patterns(self):
        """Passing module_patterns sets include_patterns with a deprecation warning."""
        with pytest.warns(DeprecationWarning, match="module_patterns"):
            config = OffloadConfig(module_patterns=["layers.*", "head"])
        assert config.include_patterns == ["layers.*", "head"]
        assert config.module_patterns == ["layers.*", "head"]

    def test_both_patterns_raises_value_error(self):
        """Passing both module_patterns and include_patterns raises ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            OffloadConfig(include_patterns=["layers.*"], module_patterns=["head"])

    def test_include_patterns_no_warning(self):
        """Using include_patterns directly emits no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = OffloadConfig(include_patterns=["layers.*"])
        assert config.include_patterns == ["layers.*"]
        assert not any("module_patterns" in str(warning.message) for warning in w)

    def test_module_patterns_env_var(self, monkeypatch):
        """FT_MODULE_PATTERNS env var maps to include_patterns with a warning."""
        monkeypatch.setenv("FT_MODULE_PATTERNS", "layers.*,head")
        with pytest.warns(DeprecationWarning, match="module_patterns"):
            config = load_config_from_env()
        assert config.include_patterns == ["layers.*", "head"]

    def test_both_env_patterns_raises_value_error(self, monkeypatch):
        """Setting both FT_MODULE_PATTERNS and FT_INCLUDE_PATTERNS raises ValueError."""
        monkeypatch.setenv("FT_MODULE_PATTERNS", "old_pattern")
        monkeypatch.setenv("FT_INCLUDE_PATTERNS", "layers.*")
        with pytest.raises(ValueError, match="Cannot set both"):
            load_config_from_env()

    def test_module_patterns_none_no_warning(self):
        """Passing module_patterns=None should not emit a warning or break validation."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = OffloadConfig(module_patterns=None)
        assert config.include_patterns == ["*"]
        assert not any("module_patterns" in str(warning.message) for warning in w)

    def test_module_patterns_from_json_file(self, tmp_path):
        """module_patterns in a JSON config file maps to include_patterns.

        Regression: _get_field_types() classified list[str] | None as object,
        so _process_data_dict silently dropped module_patterns before the
        pydantic validator could migrate it to include_patterns.
        """
        import json

        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({"module_patterns": ["layers.*", "head"]}))
        with pytest.warns(DeprecationWarning, match="module_patterns"):
            config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head"]

    def test_module_patterns_from_yaml_file(self, tmp_path):
        """module_patterns in a YAML config file maps to include_patterns."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("module_patterns:\n  - layers.*\n  - head\n")
        with pytest.warns(DeprecationWarning, match="module_patterns"):
            config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head"]

    def test_module_patterns_from_ini_file(self, tmp_path):
        """module_patterns in an INI config file maps to include_patterns.

        INI files store lists as comma-separated strings, which need special
        handling in _load_ini_file. module_patterns must survive through to
        the pydantic validator.
        """
        config_file = tmp_path / "test.conf"
        config_file.write_text("[flextensor]\nmodule_patterns = layers.*,head\n")
        with pytest.warns(DeprecationWarning, match="module_patterns"):
            config = load_config_from_file(config_file)
        assert config.include_patterns == ["layers.*", "head"]

    def test_get_field_types_classifies_optional_list_as_list(self):
        """_get_field_types must classify list[str] | None as list, not object.

        Regression: the Optional wrapper made the union type fall through
        to object, silently dropping module_patterns from file configs.
        """
        ft = _get_field_types()
        assert ft["module_patterns"] is list


class TestKnapsackScaleDeprecation:
    """Tests for deprecated knapsack_scale → transfer_budget_scale migration."""

    def test_knapsack_scale_maps_to_transfer_budget_scale(self):
        """Passing knapsack_scale sets transfer_budget_scale with a deprecation warning."""
        with pytest.warns(DeprecationWarning, match="knapsack_scale"):
            config = OffloadConfig(knapsack_scale=2.0)
        assert config.transfer_budget_scale == 2.0

    def test_both_fields_raises_value_error(self):
        """Passing both knapsack_scale and transfer_budget_scale raises ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            OffloadConfig(transfer_budget_scale=1.5, knapsack_scale=2.0)

    def test_transfer_budget_scale_no_warning(self):
        """Using transfer_budget_scale directly emits no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = OffloadConfig(transfer_budget_scale=1.5)
        assert config.transfer_budget_scale == 1.5
        assert not any("knapsack_scale" in str(warning.message) for warning in w)

    def test_knapsack_scale_warns_on_access(self):
        """Reading knapsack_scale on an instance emits DeprecationWarning."""
        config = OffloadConfig(transfer_budget_scale=1.5)
        with pytest.warns(DeprecationWarning, match="knapsack_scale"):
            assert config.knapsack_scale == 1.5

    def test_knapsack_scale_env_var(self, monkeypatch):
        """FT_KNAPSACK_SCALE env var maps to transfer_budget_scale with a warning."""
        monkeypatch.setenv("FT_KNAPSACK_SCALE", "2.5")
        with pytest.warns(DeprecationWarning, match="knapsack_scale"):
            config = load_config_from_env()
        assert config.transfer_budget_scale == 2.5

    def test_both_env_vars_raises_value_error(self, monkeypatch):
        """Setting both FT_KNAPSACK_SCALE and FT_TRANSFER_BUDGET_SCALE raises ValueError."""
        monkeypatch.setenv("FT_KNAPSACK_SCALE", "1.5")
        monkeypatch.setenv("FT_TRANSFER_BUDGET_SCALE", "2.0")
        with pytest.raises(ValueError, match="Cannot set both"):
            load_config_from_env()

    def test_knapsack_scale_from_json_file(self, tmp_path):
        """knapsack_scale in a JSON config file maps to transfer_budget_scale."""
        import json

        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({"knapsack_scale": 2.0}))
        with pytest.warns(DeprecationWarning, match="knapsack_scale"):
            config = load_config_from_file(config_file)
        assert config.transfer_budget_scale == 2.0

    def test_knapsack_scale_from_yaml_file(self, tmp_path):
        """knapsack_scale in a YAML config file maps to transfer_budget_scale."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("knapsack_scale: 2.0\n")
        with pytest.warns(DeprecationWarning, match="knapsack_scale"):
            config = load_config_from_file(config_file)
        assert config.transfer_budget_scale == 2.0

    def test_knapsack_scale_from_ini_file(self, tmp_path):
        """knapsack_scale in an INI config file maps to transfer_budget_scale."""
        config_file = tmp_path / "test.conf"
        config_file.write_text("[flextensor]\nknapsack_scale = 2.0\n")
        with pytest.warns(DeprecationWarning, match="knapsack_scale"):
            config = load_config_from_file(config_file)
        assert config.transfer_budget_scale == 2.0


class TestRemovedFieldsRejection:
    """Verify that removed OffloadConfig fields are rejected with clear errors."""

    _REMOVED: ClassVar[list[str]] = [
        "release_tensors",
        "enable_direct_mode",
        "enable_tracing",
        "rearrange_transfers",
        "compute_transfer_gap",
        "enable_untraced_tensor_discovery",
        "enable_module_tracker",
    ]

    @pytest.mark.parametrize("field", _REMOVED)
    def test_constructor_rejects_removed_field(self, field):
        """Passing a removed field to OffloadConfig() raises ValueError."""
        with pytest.raises(ValidationError, match=f"'{field}' was removed in v0.2.0"):
            OffloadConfig(**{field: False})

    @pytest.mark.parametrize("field", _REMOVED)
    def test_env_var_warns_for_removed_field(self, field, monkeypatch, caplog):
        """Setting a removed FT_* env var logs a warning."""
        env_var = f"FT_{field.upper()}"
        monkeypatch.setenv(env_var, "false")
        with caplog.at_level(logging.WARNING, logger="flextensor.config"):
            load_config_from_env()
        assert f"'{env_var}' is ignored" in caplog.text

    @pytest.mark.parametrize("field", _REMOVED)
    def test_ini_file_rejects_removed_field(self, field, tmp_path):
        """Removed field in an INI config file raises ValueError."""
        config_file = tmp_path / "test.conf"
        config_file.write_text(f"[flextensor]\n{field} = false\n")
        with pytest.raises(ValidationError, match=f"'{field}' was removed in v0.2.0"):
            load_config_from_file(config_file)

    @pytest.mark.parametrize("field", _REMOVED)
    def test_json_file_rejects_removed_field(self, field, tmp_path):
        """Removed field in a JSON config file raises ValueError."""
        import json

        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({field: False}))
        with pytest.raises(ValidationError, match=f"'{field}' was removed in v0.2.0"):
            load_config_from_file(config_file)

    def test_pinned_memory_not_rejected(self):
        """pinned_memory is still a valid field and must not be rejected."""
        config = OffloadConfig(pinned_memory=False)
        assert config.pinned_memory is False


class TestIterFieldRename:
    """Tests for the warmup_iters → discovery_iters and profile_iters → profiling_iters rename."""

    def test_discovery_iters_is_primary_field(self):
        config = OffloadConfig(discovery_iters=3)
        assert config.discovery_iters == 3

    def test_profiling_iters_is_primary_field(self):
        config = OffloadConfig(profiling_iters=5)
        assert config.profiling_iters == 5

    def test_warmup_iters_deprecated_alias_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = OffloadConfig(warmup_iters=3)
        assert any("warmup_iters" in str(warning.message) for warning in w)
        assert config.discovery_iters == 3

    def test_profile_iters_deprecated_alias_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = OffloadConfig(profile_iters=5)
        assert any("profile_iters" in str(warning.message) for warning in w)
        assert config.profiling_iters == 5

    def test_pre_inference_iters_property(self):
        config = OffloadConfig(discovery_iters=2, profiling_iters=8)
        assert config.pre_inference_iters == 10

    def test_all_warmup_iters_deprecated_property_warns(self):
        config = OffloadConfig(discovery_iters=2, profiling_iters=8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = config.all_warmup_iters
        assert val == 10
        assert any("all_warmup_iters" in str(warning.message) for warning in w)

    def test_ft_discovery_iters_env_var(self, monkeypatch):
        monkeypatch.setenv("FT_DISCOVERY_ITERS", "7")
        config = load_config()
        assert config.discovery_iters == 7

    def test_ft_warmup_iters_env_var_deprecated(self, monkeypatch):
        monkeypatch.setenv("FT_WARMUP_ITERS", "4")
        with pytest.warns(DeprecationWarning, match="warmup_iters"):
            config = load_config()
        assert config.discovery_iters == 4

    def test_warmup_and_discovery_iters_different_raises(self):
        """Passing both warmup_iters and discovery_iters with different values raises ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            OffloadConfig(warmup_iters=5, discovery_iters=3)

    def test_profile_and_profiling_iters_different_raises(self):
        """Passing both profile_iters and profiling_iters with different values raises ValueError."""
        with pytest.raises(ValueError, match="Cannot set both"):
            OffloadConfig(profile_iters=5, profiling_iters=3)

    def test_warmup_and_discovery_iters_same_value_warns(self):
        """Passing both warmup_iters and discovery_iters with same value emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="warmup_iters"):
            config = OffloadConfig(warmup_iters=5, discovery_iters=5)
        assert config.discovery_iters == 5

    def test_profile_and_profiling_iters_same_value_warns(self):
        """Passing both profile_iters and profiling_iters with same value emits DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="profile_iters"):
            config = OffloadConfig(profile_iters=5, profiling_iters=5)
        assert config.profiling_iters == 5

    def test_ft_profiling_iters_env_var(self, monkeypatch):
        """FT_PROFILING_ITERS env var maps to profiling_iters."""
        monkeypatch.setenv("FT_PROFILING_ITERS", "15")
        config = load_config()
        assert config.profiling_iters == 15

    def test_ft_profile_iters_env_var_deprecated(self, monkeypatch):
        """FT_PROFILE_ITERS env var maps to profiling_iters with deprecation warning."""
        monkeypatch.setenv("FT_PROFILE_ITERS", "8")
        with pytest.warns(DeprecationWarning, match="profile_iters"):
            config = load_config()
        assert config.profiling_iters == 8

    def test_ft_warmup_and_discovery_iters_env_conflict_raises(self, monkeypatch):
        """Setting both FT_WARMUP_ITERS and FT_DISCOVERY_ITERS raises ValueError."""
        monkeypatch.setenv("FT_WARMUP_ITERS", "4")
        monkeypatch.setenv("FT_DISCOVERY_ITERS", "7")
        with pytest.raises(ValueError, match="Cannot set both"):
            load_config()

    def test_ft_profile_and_profiling_iters_env_conflict_raises(self, monkeypatch):
        """Setting both FT_PROFILE_ITERS and FT_PROFILING_ITERS raises ValueError."""
        monkeypatch.setenv("FT_PROFILE_ITERS", "4")
        monkeypatch.setenv("FT_PROFILING_ITERS", "7")
        with pytest.raises(ValueError, match="Cannot set both"):
            load_config()


class TestModelCopyDeprecatedFieldSync:
    """Tests that model_copy(update=...) keeps deprecated fields in sync.

    Pydantic v2's model_copy bypasses mode='before' validators.  The
    OffloadConfig.model_copy override mirrors updates to deprecated
    counterparts so fields never desync.
    """

    def test_model_copy_syncs_discovery_to_warmup(self):
        config = OffloadConfig(discovery_iters=1)
        copied = config.model_copy(update={"discovery_iters": 5})
        assert copied.discovery_iters == 5
        assert copied.warmup_iters == 5

    def test_model_copy_syncs_warmup_to_discovery(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = OffloadConfig(warmup_iters=1)
        copied = config.model_copy(update={"warmup_iters": 7})
        assert copied.warmup_iters == 7
        assert copied.discovery_iters == 7

    def test_model_copy_syncs_profiling_to_profile(self):
        config = OffloadConfig(profiling_iters=10)
        copied = config.model_copy(update={"profiling_iters": 3})
        assert copied.profiling_iters == 3
        assert copied.profile_iters == 3

    def test_model_copy_syncs_profile_to_profiling(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = OffloadConfig(profile_iters=10)
        copied = config.model_copy(update={"profile_iters": 4})
        assert copied.profile_iters == 4
        assert copied.profiling_iters == 4

    def test_model_copy_syncs_shm_enabled_to_use_shared_memory(self):
        config = OffloadConfig(shm_enabled=False)
        copied = config.model_copy(update={"shm_enabled": True})
        assert copied.shm_enabled is True
        assert copied.use_shared_memory is True

    def test_model_copy_syncs_use_shared_memory_to_shm_enabled(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = OffloadConfig(use_shared_memory=False)
        copied = config.model_copy(update={"use_shared_memory": True})
        assert copied.use_shared_memory is True
        assert copied.shm_enabled is True

    def test_model_copy_syncs_transfer_budget_scale_to_knapsack_scale(self):
        config = OffloadConfig(transfer_budget_scale=1.0)
        copied = config.model_copy(update={"transfer_budget_scale": 0.8})
        assert copied.transfer_budget_scale == 0.8
        assert copied.knapsack_scale == 0.8

    def test_model_copy_syncs_knapsack_scale_to_transfer_budget_scale(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = OffloadConfig(knapsack_scale=1.0)
        copied = config.model_copy(update={"knapsack_scale": 0.6})
        assert copied.knapsack_scale == 0.6
        assert copied.transfer_budget_scale == 0.6

    def test_model_copy_explicit_both_sides_no_mirror(self):
        """When both sides of a pair are in the update with the same value, no mirroring happens."""
        config = OffloadConfig(discovery_iters=1)
        copied = config.model_copy(update={"discovery_iters": 5, "warmup_iters": 5})
        assert copied.discovery_iters == 5
        assert copied.warmup_iters == 5

    def test_model_copy_conflicting_both_sides_raises(self):
        """When both sides of a pair are in the update with different values, raise ValueError."""
        config = OffloadConfig(discovery_iters=1)
        with pytest.raises(ValueError, match="Cannot set both"):
            config.model_copy(update={"discovery_iters": 5, "warmup_iters": 3})

    def test_model_copy_without_update_unchanged(self):
        config = OffloadConfig(discovery_iters=3, profiling_iters=7)
        copied = config.model_copy()
        assert copied.discovery_iters == 3
        assert copied.warmup_iters == 3
        assert copied.profiling_iters == 7
        assert copied.profile_iters == 7

    def test_model_copy_deep_preserves_sync(self):
        config = OffloadConfig(discovery_iters=2)
        copied = config.model_copy(update={"discovery_iters": 9}, deep=True)
        assert copied.discovery_iters == 9
        assert copied.warmup_iters == 9

    def test_model_copy_multiple_pairs_updated(self):
        config = OffloadConfig(discovery_iters=1, profiling_iters=10, shm_enabled=False)
        copied = config.model_copy(update={"discovery_iters": 3, "profiling_iters": 2, "shm_enabled": True})
        assert copied.discovery_iters == 3
        assert copied.warmup_iters == 3
        assert copied.profiling_iters == 2
        assert copied.profile_iters == 2
        assert copied.shm_enabled is True
        assert copied.use_shared_memory is True

    def test_model_copy_unrelated_field_no_side_effects(self):
        """Updating include_patterns does not alter iter fields."""
        config = OffloadConfig(discovery_iters=5, profiling_iters=10)
        copied = config.model_copy(update={"include_patterns": ["layers.*"]})
        assert copied.include_patterns == ["layers.*"]
        assert copied.discovery_iters == 5
        assert copied.warmup_iters == 5
        assert copied.profiling_iters == 10
        assert copied.profile_iters == 10
