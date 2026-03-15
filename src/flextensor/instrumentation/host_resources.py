# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host resource snapshot capture for instrumentation.

This module provides a function to capture a point-in-time snapshot of host
memory using psutil.
"""

import psutil


def capture_host_resources() -> dict[str, int]:
    """Capture a snapshot of host memory.

    Uses psutil to read current host memory statistics.
    Memory values are reported in bytes.

    Returns:
        Dictionary containing:
            - host_memory_total: Total physical memory in bytes.
            - host_memory_used: Used physical memory in bytes.
            - host_memory_available: Available physical memory in bytes.
            - swap_total: Total swap space in bytes.
            - swap_used: Used swap space in bytes.
            - swap_free: Free swap space in bytes.

    Example:
        >>> snapshot = capture_host_resources()
        >>> snapshot["host_memory_total"]
        33572093952
    """
    vmem = psutil.virtual_memory()
    smem = psutil.swap_memory()

    return {
        "host_memory_total": vmem.total,
        "host_memory_used": vmem.used,
        "host_memory_available": vmem.available,
        "swap_total": smem.total,
        "swap_used": smem.used,
        "swap_free": smem.free,
    }
