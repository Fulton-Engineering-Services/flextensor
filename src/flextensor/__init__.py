# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FlexTensor: Tensor offloading and management library.

This module provides classes and functions for managing tensor offloading
strategies, collectors, loaders, and tensor management operations.
"""

import os
from importlib.metadata import PackageNotFoundError, version

# Package-wide beartype integration with environment variable control
try:
    import beartype.claw

    if os.environ.get("DISABLE_BEARTYPE") not in ["1", "true", "True", "TRUE"]:
        beartype.claw.beartype_this_package()
except ImportError:
    # beartype not available, continue without runtime type checking
    pass

# New simplified public API
# Internal implementation (kept for backward compatibility and advanced usage)
from flextensor.benchmark_tensor_mode import BenchmarkReplace, PreloadToDevice, TensorBenchmarkMode
from flextensor.config import OffloadConfig, load_config, load_config_from_env, load_config_from_file
from flextensor.lazy_model_init import load_model_from_profile
from flextensor.offload_manager import (
    DEFAULT_MANAGER_NAME,
    OffloadManager,
    get_gpu_memory_usage,
    get_offload_manager,
    init,
    load_profile,
    offload,
    offload_block,
    release,
    save_profile,
    set_config,
)
from flextensor.strategy import (
    AdaptiveKnapsackStrategy,
    AdaptiveStrategy,
    GlobalOffloadStrategy,
    GlobalTensorSelectionStrategy,
    GreedyStrategy,
    KnapsackBlockStrategy,
    KnapsackStrategy,
    NthLayerStrategy,
    Strategy,
)
from flextensor.tensor_manager import TensorManager
from flextensor.types import GPUMemoryUsage

try:
    __version__ = version("flextensor")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"  # Fallback for editable installs without version

__all__ = [  # noqa: RUF022
    # Simplified API (recommended)
    "DEFAULT_MANAGER_NAME",
    "GPUMemoryUsage",
    "OffloadConfig",
    "OffloadManager",
    "get_gpu_memory_usage",
    "get_offload_manager",
    "init",
    "load_config",
    "load_config_from_env",
    "load_config_from_file",
    "offload",
    "offload_block",
    "set_config",
    "save_profile",
    "load_profile",
    "release",
    # Internal/Advanced API (for backward compatibility)
    "AdaptiveKnapsackStrategy",
    "AdaptiveStrategy",
    "BenchmarkReplace",
    "GlobalOffloadStrategy",
    "GlobalTensorSelectionStrategy",
    "GreedyStrategy",
    "KnapsackBlockStrategy",
    "KnapsackStrategy",
    "load_model_from_profile",
    "NthLayerStrategy",
    "PreloadToDevice",
    "Strategy",
    "TensorBenchmarkMode",
    "TensorManager",
]
