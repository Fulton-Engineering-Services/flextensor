# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for handling tensor inner fields (tensor attributes on tensors)."""

import pytest
import torch
import torch.nn as nn

import flextensor


class LinearWithScale(nn.Module):
    """
    A linear layer that mimics the pattern used in DeepSeek-V3 model.

    The pattern is: self.weight.scale = self.scale = nn.Parameter(...)
    This creates a tensor attribute on weight that points to a module parameter.
    """

    def __init__(self, in_features: int, out_features: int, block_size: int = 128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Create weight parameter
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        # Create scale parameter and attach it as a tensor attribute to weight
        # This is the problematic pattern
        scale_out = (out_features + block_size - 1) // block_size
        scale_in = (in_features + block_size - 1) // block_size
        self.weight.scale = self.scale = nn.Parameter(torch.randn(scale_out, scale_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that uses both weight and weight.scale."""
        # Simulate operation that needs both weight and weight.scale on same device
        # This will fail if weight.scale is on CPU but weight is on GPU
        output = torch.nn.functional.linear(x, self.weight)

        # Access weight.scale to trigger the bug
        # In real DeepSeek-V3, this is used in dequantization kernels
        scale_device = self.weight.scale.device
        weight_device = self.weight.device

        # Verify both are on the same device
        assert scale_device == weight_device, f"weight.scale is on {scale_device} but weight is on {weight_device}"

        return output


class ModelWithInnerFields(nn.Module):
    """Model with multiple layers that have inner tensor fields."""

    def __init__(self):
        super().__init__()
        self.layer1 = LinearWithScale(64, 128)
        self.layer2 = LinearWithScale(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
def test_tensor_inner_fields_moved_to_gpu():
    """Test that tensor inner fields (like weight.scale) are properly moved to GPU."""
    # Create model and move to CPU
    model = ModelWithInnerFields()
    model = model.cpu()

    # Verify initial state
    assert model.layer1.weight.device.type == "cpu"
    assert model.layer1.scale.device.type == "cpu"
    assert model.layer1.weight.scale.device.type == "cpu"
    assert id(model.layer1.scale) == id(model.layer1.weight.scale), "scale and weight.scale should be same object"

    # Configure FlexTensor
    om = flextensor.get_offload_manager("test_inner_fields")
    config = flextensor.OffloadConfig(
        load_strategy=flextensor.NthLayerStrategy(nth_layer=1),
        discovery_iters=1,
        profiling_iters=1,
    )

    # Offload model
    config = config.model_copy(update={"include_patterns": ["layer1", "layer2"]})
    model = om.offload(model, config=config)

    # Run discovery + profiling iterations
    input_tensor = torch.randn(2, 64, device="cuda")
    for _ in range(config.discovery_iters + config.profiling_iters):
        output = model(input_tensor)

    # The model should complete without errors
    # If weight.scale was None or on CPU, the forward pass would fail
    assert output.shape == (2, 64)
    assert output.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
def test_tensor_inner_fields_in_offload_block():
    """Test that tensor inner fields work correctly with explicit offload blocks.

    Uses manual offload_block() calls instead of automatic forward patching
    to exercise the explicit block API path.
    """
    # Create model
    model = ModelWithInnerFields()
    model = model.cpu()

    # Configure FlexTensor — use include_patterns=[] so offload() sets up the
    # tensor manager but does NOT patch layer forwards (we wrap manually below).
    om = flextensor.get_offload_manager("test_inner_fields_block")
    config = flextensor.OffloadConfig(
        load_strategy=flextensor.NthLayerStrategy(nth_layer=1),
        discovery_iters=1,
        profiling_iters=1,
        include_patterns=[],
    )

    model = om.offload(model, config=config)

    input_tensor = torch.randn(2, 64, device="cuda")
    for _ in range(config.discovery_iters + config.profiling_iters):
        with om.offload_block("layer1"):
            x = model.layer1(input_tensor)
        x = torch.relu(x)
        with om.offload_block("layer2"):
            output = model.layer2(x)

    assert output.shape == (2, 64)
    assert output.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA support")
def test_tensor_inner_fields_none_check():
    """
    Test that inner fields are properly handled even when not in tensor_id_mapping.

    This is a more direct test of the bug: when weight.scale is accessed,
    tensor_layer_loader.get(field_id) might return None, and we should handle
    that case by moving the tensor to GPU.
    """
    """
    Test using lower-level tensor_manager API directly.
    """
    from flextensor.tensor_processors import MoveUnmappedTensorsToGPUProcessor

    # Create a simple module
    module = LinearWithScale(32, 64)
    module = module.cpu()

    device_gpu = torch.device("cuda")

    # Create a mapping of tensors
    tensor_id_mapping = {}
    weight_id = id(module.weight)
    scale_id = id(module.scale)

    # Move weight to GPU
    gpu_weight = module.weight.to(device=device_gpu, copy=True)
    tensor_id_mapping[weight_id] = gpu_weight

    # Move scale to GPU
    gpu_scale = module.scale.to(device=device_gpu, copy=True)
    tensor_id_mapping[scale_id] = gpu_scale

    # Create processor that should handle unmapped tensors (like weight.scale)
    processor = MoveUnmappedTensorsToGPUProcessor(device_gpu, tensor_id_mapping)

    # Process the weight - this should also handle weight.scale
    processed_weight = processor.process(module.weight)

    # The critical check: weight.scale should exist and be on GPU
    assert hasattr(processed_weight, "scale"), "weight should have scale attribute"
    assert processed_weight.scale is not None, "weight.scale should not be None"
    assert processed_weight.scale.device.type == "cuda", (
        f"weight.scale should be on cuda, but is on {processed_weight.scale.device}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
