# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the FlexTensor vLLM worker while preserving the public class path."""

import os

from flextensor.config import _register_env_var

_SELECTOR = "FT_VLLM_USE_V2_WORKER"
_register_env_var(_SELECTOR)

_selection = os.environ.get(_SELECTOR, "1")
if _selection == "1":
    from flextensor.contrib.vllm.v2.worker import FlexTensorOffloadWorker
elif _selection == "0":
    from flextensor.contrib.vllm._legacy_worker import FlexTensorOffloadWorker
else:
    raise ValueError(f"{_SELECTOR} must be '0' (legacy) or '1' (v2), got {_selection!r}")

__all__ = ["FlexTensorOffloadWorker"]
