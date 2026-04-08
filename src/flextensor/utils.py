# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import pathlib
import re
import tempfile
from typing import Any

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "any_path_matches_pattern",
    "atomic_write_json",
    "calculate_tensor_size",
    "clear_and_delete_tensor",
    "delete_tensor",
    "get_module_paths",
    "is_dense_layout",
    "matches_any_pattern",
    "resolve_gpu_mem_bytes",
]


def _compile_segment_regex(segment: str) -> re.Pattern[str]:
    """Compile a single wildcard pattern segment into a regex.

    ``*`` is mapped to ``.*`` and ``?`` to ``.`` so that they behave as
    shell-style glob wildcards within one dot-separated segment.
    """
    return re.compile(re.escape(segment).replace(r"\*", ".*").replace(r"\?", "."))


def get_module_paths(model: torch.nn.Module) -> list[str]:
    """Return dot-separated paths for all non-root modules in *model*."""
    return [p for p, _ in model.named_modules() if p]


def matches_any_pattern(path: str, patterns: list[str], *, recursive_star: bool = True) -> bool:
    """Check if a dot-separated path matches any wildcard pattern.

    Args:
        path: Dot-separated path (e.g., "layers.0.self_attn").
        patterns: List of patterns to match against.
        recursive_star: If True, standalone ``*`` matches 1+ segments.
            If False, standalone ``*`` matches exactly 1 segment.

    Returns:
        True if the path matches any of the patterns.
    """
    path_parts = path.split(".")
    regex_cache: dict[str, re.Pattern[str]] = {}
    for pattern in patterns:
        pattern_parts = pattern.split(".")
        for pp in pattern_parts:
            if pp not in regex_cache:
                regex_cache[pp] = _compile_segment_regex(pp)
        if _match_parts(path_parts, pattern_parts, regex_cache, recursive_star):
            return True
    return False


def any_path_matches_pattern(paths: list[str], pattern: str, *, recursive_star: bool = True) -> bool:
    """Check if *any* path in *paths* matches a single wildcard *pattern*.

    Unlike :func:`matches_any_pattern` (one path, many patterns), this function
    is optimised for the inverse: one pattern tested against many paths.  The
    pattern is compiled once and reused for every path.

    Args:
        paths: Dot-separated paths (e.g., module paths from ``named_modules``).
        pattern: Wildcard pattern to match against.
        recursive_star: If True, standalone ``*`` matches 1+ segments.
            If False, standalone ``*`` matches exactly 1 segment.

    Returns:
        ``True`` if at least one path matches the pattern.
    """
    pattern_parts = pattern.split(".")
    regex_cache: dict[str, re.Pattern[str]] = {}
    for pp in pattern_parts:
        if pp not in regex_cache:
            regex_cache[pp] = _compile_segment_regex(pp)
    return any(_match_parts(path.split("."), pattern_parts, regex_cache, recursive_star) for path in paths)


def _match_parts(
    path_parts: list[str],
    pattern_parts: list[str],
    regex_cache: dict[str, re.Pattern[str]],
    recursive_star: bool,
) -> bool:
    """Recursive segment matcher.

    Args:
        path_parts: Remaining path segments.
        pattern_parts: Remaining pattern segments.
        regex_cache: Map from pattern segment to pre-compiled regex.
        recursive_star: If True, standalone ``*`` matches 1+ segments.
    """
    if not pattern_parts:
        return not path_parts
    if not path_parts:
        return False

    pp = pattern_parts[0]
    if pp == "*" and recursive_star:
        # Standalone * matches 1 or more path segments
        return any(
            _match_parts(path_parts[i + 1 :], pattern_parts[1:], regex_cache, recursive_star)
            for i in range(len(path_parts))
        )
    else:
        if regex_cache[pp].fullmatch(path_parts[0]):
            return _match_parts(path_parts[1:], pattern_parts[1:], regex_cache, recursive_star)
        return False


def resolve_gpu_mem_bytes(config, *, context: str = "") -> int | None:
    """Resolve GPU memory limit from config to an absolute byte count.

    If ``max_gpu_mem_fraction`` is set, queries GPU device properties and returns
    ``int(total_memory * fraction)``.  If ``None``, falls back to the deprecated
    ``max_gpu_mem_bytes`` value (which may itself be ``None`` for latency mode).

    Args:
        config (OffloadConfig): FlexTensor offload configuration.
        context: Description of calling context for error messages
            (e.g. ``"computing SHM namespace"``).

    Returns:
        Resolved byte count, or ``None`` for latency mode.

    Raises:
        RuntimeError: If the CUDA device query fails.
    """
    if config.max_gpu_mem_fraction is not None:
        try:
            props = torch.cuda.get_device_properties(config.gpu_device)
        except (RuntimeError, AssertionError) as e:
            ctx = f" while {context}" if context else ""
            raise RuntimeError(
                f"Failed to query GPU device {config.gpu_device}{ctx} "
                f"for max_gpu_mem_fraction={config.max_gpu_mem_fraction}. "
                f"Ensure CUDA is available and gpu_device={config.gpu_device} is valid."
            ) from e
        return int(props.total_memory * config.max_gpu_mem_fraction)

    # Fraction is None: either explicit latency mode or deprecated max_gpu_mem_bytes path.
    # Read via model_dump to avoid triggering the deprecation accessor warning.
    return config.model_dump(warnings=False).get("max_gpu_mem_bytes")


def atomic_write_json(file_path: str | pathlib.Path, data: dict[str, Any]) -> None:
    """
    Write JSON data to a file atomically.

    The data is first written to a temporary file in the same directory,
    then atomically renamed to the target path. This ensures the file is
    never partially written in case of a crash.

    Args:
        file_path: Path to the file to write to.
        data: Dictionary to serialize as JSON.

    Example:
        >>> data = {"model": "gpt2", "layers": 12}
        >>> atomic_write_json("/tmp/config.json", data)
    """
    target_path = pathlib.Path(file_path)

    # Create parent directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in the same directory (ensures same filesystem for atomic rename)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=target_path.parent, prefix=f".{target_path.name}.", suffix=".tmp", delete=False
        ) as tmp_file:
            tmp_path = pathlib.Path(tmp_file.name)  # Capture path before potential json.dump error
            json.dump(data, tmp_file, indent=2)

        # Atomic rename (replaces existing file if present)
        tmp_path.replace(target_path)
        tmp_path = None  # Successfully renamed, no cleanup needed
    finally:
        # Clean up temp file if rename failed or exception occurred
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)


def is_dense_layout(tensor: torch.Tensor) -> bool:
    """Check if tensor elements occupy a contiguous memory block with no gaps.

    True for C-contiguous, Fortran-contiguous, and any permutation-contiguous
    layout where all ``numel`` elements are packed without holes.
    False for sliced/strided views with gaps between elements.

    Args:
        tensor: The tensor to check.

    Returns:
        True if the tensor's elements are densely packed in memory.

    Example:
        >>> import torch
        >>> is_dense_layout(torch.randn(3, 4))
        True
        >>> is_dense_layout(torch.randn(4, 3).t())
        True
        >>> is_dense_layout(torch.randn(4, 5)[:, ::2])
        False
    """
    if tensor.is_contiguous() or tensor.numel() <= 1:
        return True
    max_offset = sum((s - 1) * abs(st) for s, st in zip(tensor.shape, tensor.stride(), strict=False))
    return max_offset + 1 == tensor.numel()


def calculate_tensor_size(tensor: torch.Tensor) -> int:
    """
    Calculate the size of a tensor in bytes.

    Args:
        tensor: The tensor to calculate the size of

    Returns:
        The size of the tensor in bytes
    """
    numel = tensor.numel()
    element_size = tensor.element_size()
    return numel * element_size


def delete_tensor(tensor: torch.Tensor) -> None:
    """
    Delete a tensor from memory.

    Args:
        tensor: The tensor to be deleted from memory.
    """
    del tensor


def clear_and_delete_tensor(tensor: torch.Tensor) -> None:
    """
    Clear a tensor's storage and then delete it from memory.

    This function first resizes the tensor's underlying storage to 0,
    effectively clearing all data, then deletes the tensor reference.
    This can be useful for more thorough memory cleanup.

    Note: This provides more aggressive memory cleanup for cases where standard deletion
    is insufficient, particularly useful during memory profiling or debugging.

    Args:
        tensor: The tensor to be cleared and deleted from memory.
    """
    tensor.untyped_storage().resize_(0)
    del tensor
