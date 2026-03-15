# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File output logic for instrumentation data.

This module handles JSON formatting, timestamps, and file I/O for dumping
instrumentation records to disk.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import flextensor
from flextensor.instrumentation.host_resources import capture_host_resources
from flextensor.instrumentation.registry import ComponentRecord, InstrumentationRegistry

LOGGER = logging.getLogger(__name__)


def _record_to_dict(record: ComponentRecord) -> dict[str, Any]:
    """Convert a ComponentRecord to a JSON-serializable dictionary.

    Args:
        record: The component record to convert.

    Returns:
        Dictionary representation of the record.
    """
    return {
        "class_name": record.class_name,
        "module_path": record.module_path,
        "init_timestamp": record.init_timestamp,
        "args": record.args,
    }


def dump_to_file(output_path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
    """Dump all instrumentation records to a JSON file.

    The output file contains:
    - timestamp: When the dump was created
    - flextensor_version: Version of the flextensor library
    - components: List of component initialization records
    - host_memory: Host memory snapshot at dump time

    Additional top-level keys can be included via the ``extra`` parameter.

    Args:
        output_path: Path to the output JSON file.
        extra: Additional key-value pairs to merge into the top-level JSON output.
            Keys must not collide with built-in keys (timestamp, flextensor_version,
            components, host_memory).

    Raises:
        OSError: If the file cannot be written.
        ValueError: If any key in ``extra`` collides with a built-in key.
    """
    registry = InstrumentationRegistry.get_instance()
    records = registry.get_records()

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "flextensor_version": flextensor.__version__,
        "components": [_record_to_dict(record) for record in records],
        "host_memory": capture_host_resources(),
    }

    if extra is not None:
        collisions = set(extra) & set(output_data)
        if collisions:
            msg = f"Extra keys collide with built-in keys: {collisions}"
            raise ValueError(msg)
        output_data.update(extra)

    output_path = Path(output_path)

    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    LOGGER.info("Instrumentation data dumped to %s (%d components)", output_path, len(records))


def dump_to_directory(output_dir: str | Path, *, extra: dict[str, Any] | None = None) -> Path:
    """Dump instrumentation records to a timestamped file in a directory.

    Creates a file with a timestamp in the filename for uniqueness.

    Args:
        output_dir: Directory to write the output file.
        extra: Additional key-value pairs to merge into the top-level JSON output.
            Forwarded to :func:`dump_to_file`.

    Returns:
        Path to the created file.

    Raises:
        OSError: If the directory cannot be created or file cannot be written.
        ValueError: If any key in ``extra`` collides with a built-in key.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()

    output_dir = Path(output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"components.{timestamp}_pid{pid}.json"
    output_path = output_dir / filename
    dump_to_file(output_path, extra=extra)
    return output_path
