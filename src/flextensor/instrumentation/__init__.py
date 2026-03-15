# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instrumentation package for capturing component initialization arguments.

This package provides tools for debugging and analyzing FlexTensor component
configurations by capturing init args when instrumentation is enabled.

Example:
    >>> from flextensor.instrumentation import instrumentable, get_registry, dump_instrumentation
    >>>
    >>> # Enable instrumentation
    >>> registry = get_registry()
    >>> registry.enabled = True
    >>>
    >>> # Components decorated with @instrumentable will be tracked
    >>> # ... use FlexTensor ...
    >>>
    >>> # Dump to file
    >>> dump_instrumentation("/tmp/debug/instrumentation")
"""

from typing import Any

from flextensor.instrumentation.decorator import instrumentable
from flextensor.instrumentation.dumper import dump_to_directory, dump_to_file
from flextensor.instrumentation.host_resources import capture_host_resources
from flextensor.instrumentation.registry import ComponentRecord, InstrumentationRegistry

__all__ = [
    "ComponentRecord",
    "InstrumentationRegistry",
    "capture_host_resources",
    "dump_instrumentation",
    "dump_to_directory",
    "dump_to_file",
    "get_registry",
    "instrumentable",
]


def get_registry() -> InstrumentationRegistry:
    """Get the global instrumentation registry instance.

    Returns:
        The singleton InstrumentationRegistry instance.
    """
    return InstrumentationRegistry.get_instance()


def dump_instrumentation(output_dir: str, *, extra: dict[str, Any] | None = None) -> None:
    """Convenience function to dump instrumentation data to a file.

    Args:
        output_dir: Path to the output directory.
        extra: Additional key-value pairs to merge into the top-level JSON output.
            Forwarded to :func:`dump_to_directory`.
    """
    dump_to_directory(output_dir, extra=extra)
