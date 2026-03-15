# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for unit tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_cuda_device_properties():
    """Mock torch.cuda.get_device_properties to avoid requiring a real GPU.

    OffloadConfig now includes max_gpu_mem_fraction=0.9 as its default, which causes
    _initialize_tensor_manager to call torch.cuda.get_device_properties when resolving
    the fraction to bytes. This fixture prevents that call from failing in CPU-only
    test environments by supplying a fake 40 GB device.

    Tests that need to assert specific get_device_properties behaviour (e.g.
    TestMaxGpuMemResolution) override this fixture locally via their own patch context.
    """
    fake_props = MagicMock()
    fake_props.total_memory = 40 * 1024**3  # 40 GB
    fake_props.major = 8  # Ampere compute capability — prevents DeferredCudaCallError from
    fake_props.minor = 0  # _check_capability comparing (props.major, props.minor) < (3, 5)
    fake_props.name = "Mock GPU"
    with patch("flextensor.utils.torch.cuda.get_device_properties", return_value=fake_props):
        yield
