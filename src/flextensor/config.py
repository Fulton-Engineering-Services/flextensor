# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes and utilities for FlexTensor.

This module provides configuration management for the offload manager,
including support for loading configuration from environment variables and files.
"""

import configparser
import json
import logging
import os
import types
import warnings
from pathlib import Path
from typing import Annotated, Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, WithJsonSchema, model_validator
from typing_extensions import Self
from typing_extensions import deprecated as _deprecated

from flextensor.strategy import Strategy

logger = logging.getLogger(__name__)

_USE_SHARED_MEMORY_MSG = "`use_shared_memory` is deprecated. Use `shm_enabled` instead. Will be removed in v0.5."
_MAX_GPU_MEM_BYTES_MSG = (
    "`max_gpu_mem_bytes` is deprecated. Use `max_gpu_mem_fraction` instead. Will be removed in v0.5."
)

_DEFAULT_MAX_GPU_MEM_FRACTION = 0.9

_REMOVED_FIELDS: dict[str, tuple[str, str]] = {
    "release_tensors": ("v0.1.1", "GPU tensors are now always released after layer execution."),
    "enable_direct_mode": ("v0.1.1", "Direct mode is now always enabled."),
    "enable_tracing": ("v0.1.1", "Use TensorManager(_use_trace_tensor=True) for tracing."),
    "rearrange_transfers": ("v0.1.1", "Transfer rearrangement is now auto-enabled when gap layers are detected."),
    "compute_transfer_gap": ("v0.1.1", "Use TensorManager(_compute_transfer_gap=N) to override."),
    "enable_untraced_tensor_discovery": (
        "v0.1.1",
        "Untraced tensor discovery is now always enabled."
        " Use TensorManager(_enable_untraced_tensor_discovery=False) to override.",
    ),
    "enable_module_tracker": ("v0.1.1", "ModuleTracker is now always enabled for manual traps."),
}


class OffloadConfig(BaseModel):
    """Configuration for offload manager."""

    enabled: bool = Field(default=True)
    """Whether to enable offload."""

    gpu_device: int = Field(default=0, ge=0)
    """GPU device index to use."""

    pinned_memory: bool = Field(default=True)
    """Whether to use pinned (page-locked) memory for CPU tensors, enabling non-blocking GPU transfers."""

    use_shared_memory: Annotated[bool, _deprecated(_USE_SHARED_MEMORY_MSG)] = Field(
        default=False, deprecated=_USE_SHARED_MEMORY_MSG
    )
    """Whether to use shared memory.

    .. deprecated:: v0.4
        Use :attr:`shm_enabled` instead. Will be removed in v0.5.
    """

    shm_enabled: bool = Field(default=False)
    """Enable cross-process weight sharing via POSIX shared memory.

    When True, model weights are stored in shared memory so multiple processes
    (e.g., vLLM data-parallel replicas) share a single copy in CPU RAM.
    Replaces the deprecated ``use_shared_memory`` field.
    Can also be set via the ``FT_SHM_ENABLED`` env var.
    """

    shm_namespace: str | None = Field(default=None)
    """Base namespace for shared memory blocks.

    If None, auto-derived from model path + config hash.
    Can also be set via FT_SHM_NAMESPACE env var.
    """

    shm_wait_timeout: float = Field(default=0.0, ge=0.0)
    """Hard wall-clock timeout (seconds) for followers waiting on creator.

    0 means no hard limit — rely on heartbeat liveness detection only.
    Can also be set via FT_SHM_WAIT_TIMEOUT env var.
    """

    warmup_iters: int = Field(default=1, ge=0)
    """Number of warmup iterations before profiling begins."""

    profile_iters: int = Field(default=10, ge=0)
    """Number of profiling iterations to collect statistics."""

    knapsack_scale: float = Field(default=1.0, gt=0.0)
    """Duration multiplier for transfer budget estimation in latency mode.

    Values below 1.0 add a safety margin to absorb profiling measurement noise
    (e.g., 0.95 = 5% margin). Only affects latency mode (``max_gpu_mem_fraction=None``);
    in memory mode the strategy auto-scales to meet the GPU memory target.
    """

    transfer_mode: str = Field(default="allocation_block_transfer")
    """Tensor loading mode. Block transfer loaders (``allocation_block_transfer``,
    ``raw_block_transfer``) do not support shared tensors between layers;
    use ``strategy`` for models with shared weights."""

    num_blocks: int = Field(default=4, ge=2)
    """Number of memory blocks for block transfer loaders.

    More blocks reserve more GPU memory but give the pipelining scheduler more
    room to overlap transfers with compute. Fewer blocks (e.g., 2) reduce GPU
    memory overhead and can work well for models whose layers have long compute
    times (such as diffusers), since the transfer easily finishes within a
    single layer's compute window.
    """

    min_blocks: int = Field(default=4, ge=2)
    """Minimum number of blocks for assignment optimization search range.

    The optimizer tries block counts from ``min_blocks`` to ``num_blocks`` and
    picks the best. Lower values give more freedom and reduce GPU memory, but
    may hurt pipelining throughput. Must be >= 2 (pipelined execution requires
    at least two blocks).
    Can also be set via the ``FT_MIN_BLOCKS`` env var.
    """

    max_gpu_mem_fraction: float | None = Field(default=_DEFAULT_MAX_GPU_MEM_FRACTION, gt=0.0, le=1.0)
    """Target maximum GPU memory as a fraction of total device memory (e.g. 0.9 = 90%).

    When set, the strategy switches to *memory mode* and keeps peak GPU usage within this
    budget. At runtime, the fractional budget is capped by actual available GPU memory so
    the strategy never targets more than what is free. When ``None``, the strategy operates
    in *latency mode* (no memory constraint).
    Defaults to 0.9. Can also be set via the ``FT_MAX_GPU_MEM_FRACTION`` env var.
    """

    max_gpu_mem_bytes: Annotated[int | None, _deprecated(_MAX_GPU_MEM_BYTES_MSG)] = Field(
        default=None, ge=0, deprecated=_MAX_GPU_MEM_BYTES_MSG
    )
    """Hard GPU memory limit in bytes (never exceed).

    .. deprecated:: v0.4
        Use :attr:`max_gpu_mem_fraction` instead. Will be removed in v0.5.
        Env var ``FT_MAX_GPU_MEM_BYTES`` is similarly deprecated; use ``FT_MAX_GPU_MEM_FRACTION``.
    """

    profile_storage_dir: str | None = Field(default=None)
    """Directory for profile persistence."""

    profile_read_only: bool = Field(default=False)
    """Only load profiles, saving is disabled."""

    load_strategy: Annotated[Strategy, WithJsonSchema({"type": "object"})] | None = Field(default=None)
    """Override automatic strategy selection."""

    enable_instrumentation: bool = Field(default=False)
    """Whether to capture component init args for debugging."""

    instrumentation_output_dir: str = Field(default=".flextensor/instrumentation")
    """Instrumentation output directory."""

    module_patterns: list[str] = Field(default_factory=lambda: ["*"])
    """Module path patterns to offload (supports ``*`` and ``?`` wildcards).

    Can also be set via the ``FT_MODULE_PATTERNS`` env var as a comma-separated list.
    """

    enable_diagnostics: bool = Field(default=False)
    """Whether to log diagnostic information (per-trap duration statistics, block assignment table)
       after strategy computation.
    """

    model_config = {"arbitrary_types_allowed": True, "use_attribute_docstrings": True}

    # Note: pydantic v2 runs mode="before" validators in reverse definition order.
    # _sync_shm_fields (defined after this) runs first, then this validator.
    # These two validators have no dependencies on each other, so ordering is harmless.
    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_max_gpu_mem_bytes(cls, data: Any) -> Any:
        """Handle deprecated max_gpu_mem_bytes field."""
        if isinstance(data, dict):
            bytes_set = "max_gpu_mem_bytes" in data
            fraction_set = "max_gpu_mem_fraction" in data
            if bytes_set and not fraction_set:
                warnings.warn(_MAX_GPU_MEM_BYTES_MSG, DeprecationWarning, stacklevel=2)
                data["max_gpu_mem_fraction"] = None  # suppress default; activate bytes path
            elif bytes_set and fraction_set:
                msg = "Cannot set both max_gpu_mem_bytes and max_gpu_mem_fraction"
                raise ValueError(msg)
        return data

    @model_validator(mode="before")
    @classmethod
    def _sync_shm_fields(cls, data: Any) -> Any:
        """Sync deprecated use_shared_memory with shm_enabled."""
        if isinstance(data, dict):
            if "use_shared_memory" in data and "shm_enabled" not in data:
                warnings.warn(_USE_SHARED_MEMORY_MSG, DeprecationWarning, stacklevel=2)
                data["shm_enabled"] = data["use_shared_memory"]
            elif "shm_enabled" in data and "use_shared_memory" not in data:
                data["use_shared_memory"] = data["shm_enabled"]
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field, (version, msg) in _REMOVED_FIELDS.items():
                if field in data:
                    raise ValueError(
                        f"'{field}' was removed in {version}. {msg} Remove '{field}' from your OffloadConfig() call."
                    )
        return data

    @model_validator(mode="after")
    def _validate_block_range(self) -> Self:
        if self.num_blocks < self.min_blocks:
            raise ValueError(f"num_blocks ({self.num_blocks}) must be >= min_blocks ({self.min_blocks})")
        return self

    @property
    def all_warmup_iters(self) -> int:
        """Total warmup iterations including profile iterations."""
        return self.warmup_iters + self.profile_iters


def _get_field_types() -> dict[str, type]:
    """Get field names and their base types from OffloadConfig.

    Returns:
        Dictionary mapping field names to their types.
        For Union/Optional types, returns the first non-None type.
        For complex types (like strategies), returns object.
    """
    field_types: dict[str, type] = {}
    for name, field_info in OffloadConfig.model_fields.items():
        annotation = field_info.annotation
        if annotation is None:
            field_types[name] = str
            continue

        # Handle Union types (typing.Union and PEP 604 `X | Y`)
        origin = get_origin(annotation)
        if origin is Union or origin is types.UnionType:
            # Get first non-None type from Union
            args = [a for a in get_args(annotation) if a is not type(None)]
            if args:
                first_type = args[0]
                # Check if it's a complex type (like strategies)
                if get_origin(first_type) is Union or not isinstance(first_type, type):
                    field_types[name] = object
                elif first_type in (bool, int, float, str):
                    field_types[name] = first_type
                else:
                    field_types[name] = object
            else:
                field_types[name] = str
        elif isinstance(annotation, type) and annotation in (bool, int, float, str):
            field_types[name] = annotation
        else:
            field_types[name] = object

    return field_types


def _parse_bool(value: str) -> bool:
    """Parse a string value to boolean with case-insensitive matching.

    Args:
        value: String value to parse

    Returns:
        Boolean value

    Raises:
        ValueError: If the value cannot be parsed as a boolean
    """
    lower_value = value.lower()
    if lower_value in ("true", "1", "yes", "y", "on"):
        return True
    if lower_value in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean")


def _parse_none(value: str) -> None:
    """Parse a string value to None with case-insensitive matching.

    Args:
        value: String value to parse

    Returns:
        None if the value represents null/none, otherwise raises ValueError

    Raises:
        ValueError: If the value cannot be parsed as None
    """
    lower_value = value.lower()
    if lower_value in ("none", "null"):
        return None
    raise ValueError(f"Cannot parse '{value}' as None")


def _convert_env_value(value: str, field_type: type) -> Any:
    """Convert environment variable string to the appropriate type.

    Args:
        value: String value from environment variable
        field_type: Target type to convert to

    Returns:
        Converted value

    Raises:
        ValueError: If conversion fails
    """
    # Handle None/null values
    try:
        return _parse_none(value)
    except ValueError:
        pass  # Not a None value, continue with other conversions

    # Handle empty strings
    if value == "":
        return None

    # Type-specific conversions
    if field_type is bool:
        return _parse_bool(value)
    elif field_type is int:
        return int(value)
    elif field_type is float:
        return float(value)
    elif field_type is str:
        return value
    else:
        # For complex types (like strategies), return as string and let pydantic handle it
        return value


def _load_ini_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from INI file.

    Args:
        config_path: Path to the INI file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If the file cannot be parsed or section is missing
    """
    parser = configparser.ConfigParser()
    parser.read(config_path)

    section = "flextensor"
    if section not in parser:
        raise ValueError(f"INI file must contain [{section}] section")

    config_dict: dict[str, Any] = {}
    for field_name, field_type in field_types.items():
        if field_name in parser[section]:
            value = parser[section][field_name]
            if field_type is object:
                continue
            config_dict[field_name] = _convert_env_value(value, field_type)

    for field in _REMOVED_FIELDS:
        if field in parser[section]:
            config_dict[field] = parser[section][field]

    return config_dict


def _process_data_dict(data: dict[str, Any], field_types: dict[str, type]) -> dict[str, Any]:
    """Process a data dictionary into configuration values.

    Common logic for JSON and YAML file loading that handles field filtering,
    type conversion, and special cases like module_patterns.

    Args:
        data: Raw data dictionary from file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    config_dict: dict[str, Any] = {}
    for field_name in data:
        if field_name in _REMOVED_FIELDS:
            config_dict[field_name] = data[field_name]
            continue

        # Skip complex types unless it's module_patterns (a list field we support)
        if field_name in field_types and field_types[field_name] is object and field_name != "module_patterns":
            continue

        value = data[field_name]
        # Handle special case for module_patterns (list field)
        if field_name == "module_patterns":
            config_dict[field_name] = value
        elif field_name in field_types:
            field_type = field_types[field_name]
            # JSON/YAML already have proper types, but handle string values
            if isinstance(value, str) and field_type is not str and field_type is not object:
                config_dict[field_name] = _convert_env_value(value, field_type)
            elif field_type is not object:
                config_dict[field_name] = value

    return config_dict


def _load_json_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to the JSON file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    with config_path.open() as f:
        data = json.load(f)

    return _process_data_dict(data, field_types)


def _load_yaml_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    with config_path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return _process_data_dict(data, field_types)


def _load_from_file(config_path: Path, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from file based on extension.

    Args:
        config_path: Path to the configuration file
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values

    Raises:
        ValueError: If the file format is not supported
    """
    suffix = config_path.suffix.lower()
    if suffix in (".conf", ".ini"):
        return _load_ini_file(config_path, field_types)
    elif suffix == ".json":
        return _load_json_file(config_path, field_types)
    elif suffix in (".yaml", ".yml"):
        return _load_yaml_file(config_path, field_types)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. Use .conf, .ini, .json, .yaml, or .yml")


def _load_from_env(env_prefix: str, field_types: dict[str, type]) -> dict[str, Any]:
    """Load configuration from environment variables.

    Args:
        env_prefix: Prefix for environment variable names
        field_types: Dictionary mapping field names to types

    Returns:
        Dictionary of configuration values
    """
    config_dict: dict[str, Any] = {}

    for field, (version, msg) in _REMOVED_FIELDS.items():
        env_var_name = f"{env_prefix}{field.upper()}"
        if os.environ.get(env_var_name) is not None:
            logger.warning(
                "Environment variable '%s' is ignored: '%s' was removed in %s. %s",
                env_var_name,
                field,
                version,
                msg,
            )

    # Special handling for list field: module_patterns
    env_var_name = f"{env_prefix}MODULE_PATTERNS"
    env_value = os.environ.get(env_var_name)
    if env_value:
        config_dict["module_patterns"] = [p.strip() for p in env_value.split(",") if p.strip()]

    for field_name, field_type in field_types.items():
        # Skip module_patterns as it's handled above
        if field_name == "module_patterns":
            continue

        env_var_name = f"{env_prefix}{field_name.upper()}"
        env_value = os.environ.get(env_var_name)

        if env_value is not None:
            # Skip complex types like load_strategy
            if field_type is object:
                continue

            try:
                config_dict[field_name] = _convert_env_value(env_value, field_type)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert {env_var_name}={env_value} to {field_type.__name__}: {e}") from e

    return config_dict


def load_config(
    config_path: str | Path | None = None,
    env_prefix: str = "FT_",
    use_env: bool = True,
    config_file_env_var: str = "FT_CONFIG_FILE",
    **kwargs,
) -> OffloadConfig:
    """Load OffloadConfig with precedence: file < env vars < kwargs.

    This is the unified configuration loading function that combines file and
    environment variable loading with proper precedence.

    Precedence order (highest wins):

    1. kwargs (explicit arguments)
    2. Environment variables (if use_env=True)
    3. Config file values
    4. OffloadConfig defaults

    When loading from environment only (no file), `enabled` defaults to False.
    You must explicitly set FT_ENABLED=1 to enable offloading.

    Supported file formats (auto-detected by extension):

    - INI (.conf, .ini) — requires [flextensor] section
    - JSON (.json)
    - YAML (.yaml, .yml)

    Args:
        config_path: Path to configuration file. If None, checks config_file_env_var.
        env_prefix: Prefix for environment variable names (default: "FT_")
        use_env: Whether to load and override with environment variables (default: True)
        config_file_env_var: Environment variable for config file path (default: "FT_CONFIG_FILE")
        **kwargs: Additional keyword arguments to override all other values

    Returns:
        OffloadConfig instance

    Examples:
        >>> # Full integration: file + env + kwargs
        >>> config = load_config("flextensor.conf")
        >>>
        >>> # With env override: FT_GPU_DEVICE=2 overrides file value
        >>> config = load_config("flextensor.conf")
        >>>
        >>> # Env only (no file, enabled defaults to False)
        >>> config = load_config()
    """
    field_types = _get_field_types()
    config_dict: dict[str, Any] = {}

    # Load from file if path provided
    path_str = config_path or os.environ.get(config_file_env_var)
    if path_str is not None:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        config_dict = _load_from_file(path, field_types)
    elif use_env:
        # No file loaded and using env: default enabled to False
        config_dict["enabled"] = False

    # Override with environment variables
    if use_env:
        config_dict.update(_load_from_env(env_prefix, field_types))

    # Override with kwargs (highest precedence)
    config_dict.update(kwargs)

    return OffloadConfig(**config_dict)


def load_config_from_file(
    config_path: str | Path | None = None,
    env_var: str = "FT_CONFIG_FILE",
    **kwargs,
) -> OffloadConfig:
    """Load OffloadConfig from a configuration file (no env var override).

    This is a convenience wrapper around load_config() that loads from file only,
    without environment variable overrides.

    Supported file formats (auto-detected by extension):

    - INI (.conf, .ini) — requires [flextensor] section
    - JSON (.json)
    - YAML (.yaml, .yml)

    Args:
        config_path: Path to configuration file. If None, uses env_var.
        env_var: Environment variable name for config file path (default: "FT_CONFIG_FILE")
        **kwargs: Additional keyword arguments to override file values

    Returns:
        OffloadConfig instance with values from file and kwargs

    Raises:
        FileNotFoundError: If the config file does not exist
        ValueError: If the file format is not supported or file is invalid

    Examples:
        >>> config = load_config_from_file("flextensor.conf")
        >>> config.enabled
        True
    """
    # If no config_path and no env var set, raise helpful error
    if config_path is None and os.environ.get(env_var) is None:
        raise FileNotFoundError(f"No config file specified. Provide config_path or set {env_var} environment variable.")

    return load_config(config_path=config_path, use_env=False, config_file_env_var=env_var, **kwargs)


def load_config_from_env(prefix: str = "FT_", **kwargs) -> OffloadConfig:
    """Load OffloadConfig from environment variables (no file).

    This is a convenience wrapper around load_config() that loads from
    environment variables only, without file loading.

    Environment variables should be named as {prefix}{FIELD_NAME} in uppercase.
    For example, with the default prefix "FT_":

    - FT_ENABLED=1 (required to enable offloading)
    - FT_GPU_DEVICE=1
    - FT_WARMUP_ITERS=5

    Note: When loading from environment, `enabled` defaults to False.
    You must explicitly set FT_ENABLED=1 to enable offloading.

    Type conversions:

    - bool: Case-insensitive parsing of "true"/"false", "1"/"0", "yes"/"no", etc.
    - int: Direct integer conversion
    - float: Direct float conversion
    - str: Used as-is
    - None: Case-insensitive "none" or "null"

    Args:
        prefix: Prefix for environment variable names (default: "FT_")
        **kwargs: Additional keyword arguments to override environment variables

    Returns:
        OffloadConfig instance with values from environment variables and kwargs

    Examples:
        >>> import os
        >>> os.environ["FT_ENABLED"] = "1"
        >>> os.environ["FT_GPU_DEVICE"] = "1"
        >>> config = load_config_from_env()
        >>> config.enabled
        True
        >>> config.gpu_device
        1
    """
    return load_config(
        config_path=None,
        env_prefix=prefix,
        use_env=True,
        config_file_env_var="",  # Empty to prevent file loading
        **kwargs,
    )
