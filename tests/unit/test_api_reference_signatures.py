# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for public types rendered in the API reference."""

from typing import Any, get_type_hints

import torch

from flextensor.config import load_config, load_config_from_env, load_config_from_file
from flextensor.state_handler import TensorManagerState
from flextensor.strategy import Strategy
from flextensor.tensor_manager import TensorManager


def test_api_reference_types_live_in_function_signatures() -> None:
    """Types needed by the API reference should be available from signatures."""
    for loader in (load_config, load_config_from_file, load_config_from_env):
        assert get_type_hints(loader)["kwargs"] is Any

    init_hints = get_type_hints(TensorManager.__init__)
    assert init_hints["device_gpu"] == str | torch.device
    assert init_hints["tensor_manager_load_strategy"] is Strategy
    assert init_hints["loader_type"] is str
    assert init_hints["remove_layers_operations"] == list[dict[str, Any]] | None
    assert init_hints["blocks"] is int
    assert init_hints["move_top_level_buffers_to_gpu"] is bool

    assert get_type_hints(TensorManager.load_state)["return"] is TensorManagerState
    assert get_type_hints(TensorManager.load_profile)["model"] is torch.nn.Module

    assert not hasattr(TensorManager, "run_profile_suite")
