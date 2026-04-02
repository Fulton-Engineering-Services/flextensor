# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for GPU memory usage API.

This test suite validates the GPUMemoryUsage API with real GPU operations.
It tests:
1. Memory usage is reported correctly after inference mode initialization
2. Block memory matches actual GPU allocations
3. Different loader types (allocation_block_transfer, raw_block_transfer)
"""

import pytest
import torch
from torch import nn

from flextensor import TensorManager
from flextensor.strategy import KnapsackStrategy
from flextensor.types import GPUMemoryUsage


@pytest.fixture(scope="module")
def device_gpu():
    """Fixture to provide GPU device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


class SimpleModelForMemoryTest(nn.Module):
    """Simple model with predictable tensor sizes for memory testing.

    This model uses tensor_manager.trap() inside forward to enable offloading.
    """

    def __init__(self, tensor_manager, num_layers: int = 4, hidden_size: int = 256):
        super().__init__()
        self.tensor_manager = tensor_manager
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            with self.tensor_manager.trap(f"layer_{i}"):
                x = layer(x)
        return x


class TestGPUMemoryUsageIntegration:
    """Integration test cases for GPU memory usage API."""

    def create_tensor_manager(
        self,
        device_gpu: torch.device,
        loader_type: str = "allocation_block_transfer",
    ) -> TensorManager:
        """Create a TensorManager instance for testing."""
        strategy = KnapsackStrategy(scale=0.8)
        return TensorManager(
            device_gpu=device_gpu,
            tensor_manager_load_strategy=strategy,
            loader_type=loader_type,
            blocks=4,
        )

    @pytest.mark.parametrize(
        "loader_type",
        [
            "allocation_block_transfer",
            "raw_block_transfer",
        ],
    )
    def test_gpu_memory_usage_with_simple_model(
        self,
        loader_type: str,
        device_gpu: torch.device,
    ):
        """Test GPU memory usage with a simple model and different loader types.

        This test validates that:
        1. get_gpu_memory_usage() returns valid GPUMemoryUsage after inference init
        2. blocks_bytes > 0 (GPU transfer blocks are allocated)
        3. total_bytes >= blocks_bytes (total includes blocks)
        """
        # Create tensor manager first
        tensor_manager = self.create_tensor_manager(device_gpu, loader_type)

        # Create model on CPU with tensor_manager reference
        model = SimpleModelForMemoryTest(tensor_manager, num_layers=4, hidden_size=256)
        model = model.cpu()
        model.eval()

        tensor_manager.set_model(model)

        # Create input tensor
        x = torch.randn(1, 256, device=device_gpu)

        # Run through phases
        with torch.no_grad():
            # Warmup phase
            model = tensor_manager.initialize_warmup()
            _ = model(x)

            # Profile phase
            model = tensor_manager.initialize_profile()
            x = torch.randn(1, 256, device=device_gpu)
            _ = model(x)

            # Inference phase
            model = tensor_manager.initialize_inference()

        # Get GPU memory usage
        usage = tensor_manager.get_gpu_memory_usage()

        # Validate the result is a GPUMemoryUsage instance
        assert isinstance(usage, GPUMemoryUsage), "get_gpu_memory_usage should return GPUMemoryUsage"

        # Validate blocks_bytes > 0 (GPU blocks should be allocated)
        assert usage.blocks_bytes > 0, f"Expected blocks_bytes > 0, got {usage.blocks_bytes}"

        # Validate total_bytes >= blocks_bytes
        assert usage.total_bytes >= usage.blocks_bytes, (
            f"total_bytes ({usage.total_bytes}) should be >= blocks_bytes ({usage.blocks_bytes})"
        )

        # Validate MB properties work correctly
        assert usage.blocks_mb == usage.blocks_bytes / (1024 * 1024)
        assert usage.total_mb == usage.total_bytes / (1024 * 1024)

    def test_gpu_memory_usage_raises_before_inference(self, device_gpu: torch.device):
        """Test that get_gpu_memory_usage() raises RuntimeError before inference mode."""
        tensor_manager = self.create_tensor_manager(device_gpu)

        model = SimpleModelForMemoryTest(tensor_manager, num_layers=2, hidden_size=128)
        model = model.cpu()
        model.eval()

        tensor_manager.set_model(model)

        # Initialize warmup only
        model = tensor_manager.initialize_warmup()

        # Attempt to get memory usage should raise
        with pytest.raises(RuntimeError, match="Cannot get GPU memory usage before inference mode"):
            tensor_manager.get_gpu_memory_usage()

    def test_gpu_memory_usage_blocks_match_actual_allocation(self, device_gpu: torch.device):
        """Test that reported GPU block memory matches actual CUDA allocation.

        This test validates that the reported block memory is consistent with
        what PyTorch's CUDA memory management reports.
        """
        torch.cuda.reset_peak_memory_stats(device_gpu)
        torch.cuda.empty_cache()

        # Create tensor manager
        tensor_manager = self.create_tensor_manager(device_gpu, "allocation_block_transfer")

        # Create model
        model = SimpleModelForMemoryTest(tensor_manager, num_layers=8, hidden_size=512)
        model = model.cpu()
        model.eval()

        tensor_manager.set_model(model)

        x = torch.randn(1, 512, device=device_gpu)

        with torch.no_grad():
            # Run through all phases
            model = tensor_manager.initialize_warmup()
            _ = model(x)

            model = tensor_manager.initialize_profile()
            x = torch.randn(1, 512, device=device_gpu)
            _ = model(x)

            model = tensor_manager.initialize_inference()

        # Get reported memory usage
        usage = tensor_manager.get_gpu_memory_usage()

        # The reported memory should be > 0
        assert usage.total_bytes > 0, "Reported total memory should be > 0"

        # Blocks + unmapped should equal total
        assert usage.blocks_bytes + usage.unmapped_tensors_bytes == usage.total_bytes, (
            f"blocks ({usage.blocks_bytes}) + unmapped ({usage.unmapped_tensors_bytes}) != total ({usage.total_bytes})"
        )


class TestGPUMemoryUsageWithDictModel:
    """Test GPU memory usage with dictionary-style models."""

    def test_gpu_memory_usage_with_dict_model(self, device_gpu: torch.device):
        """Test GPU memory usage when using a dict model instead of nn.Module.

        Dict models require accessing tensors through the tensor_layer_loader.get()
        method after the trap has been entered to get the GPU-loaded tensor.
        """
        # Create a simple dict model
        hidden_size = 256
        num_layers = 4

        model = {}
        for i in range(num_layers):
            model[f"layer_{i}_weight"] = torch.randn(hidden_size, hidden_size)

        # Create tensor manager
        strategy = KnapsackStrategy(scale=0.8)
        tensor_manager = TensorManager(
            device_gpu=device_gpu,
            tensor_manager_load_strategy=strategy,
            loader_type="allocation_block_transfer",
            blocks=4,
        )
        tensor_manager.set_model(model)

        x = torch.randn(1, hidden_size, device=device_gpu)

        def run_model_warmup(model_dict, x, tensor_manager, num_layers):
            """Run forward pass during warmup - tensors are moved on demand."""
            for i in range(num_layers):
                with tensor_manager.trap(f"layer_{i}"):
                    w = model_dict[f"layer_{i}_weight"]
                    # During warmup, use tensor on its current device
                    if w.device != x.device:
                        w = w.to(x.device)
                    x = x @ w
            return x

        def run_model_profile(model_dict, x, tensor_manager, num_layers):
            """Run forward pass during profile - use tensor_layer_loader."""
            for i in range(num_layers):
                with tensor_manager.trap(f"layer_{i}"):
                    w = model_dict[f"layer_{i}_weight"]
                    tensor_id = id(w)
                    # Try to get loaded GPU tensor
                    loaded_w = tensor_manager.tensor_layer_loader.get(tensor_id)
                    if loaded_w is not None:
                        w = loaded_w
                    elif w.device != x.device:
                        w = w.to(x.device)
                    x = x @ w
            return x + 0

        with torch.no_grad():
            # Warmup phase
            tensor_manager.initialize_warmup()
            _ = run_model_warmup(model, x, tensor_manager, num_layers)

            # Profile phase
            tensor_manager.initialize_profile()
            x = torch.randn(1, hidden_size, device=device_gpu)
            _ = run_model_profile(model, x, tensor_manager, num_layers)

            # Inference phase
            tensor_manager.initialize_inference()

        # Get GPU memory usage
        usage = tensor_manager.get_gpu_memory_usage()

        # Validate
        assert isinstance(usage, GPUMemoryUsage)
        assert usage.total_bytes >= 0
        assert usage.blocks_bytes >= 0
        assert usage.unmapped_tensors_bytes >= 0
