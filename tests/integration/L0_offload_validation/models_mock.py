# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        res = F.silu(self.w1(x)) * self.w3(x)
        res2 = self.w2(res)
        x = res2
        res = F.silu(self.w1(x)) * self.w3(x)
        res2 = self.w2(res)
        x = res2
        res = F.silu(self.w1(x)) * self.w3(x)
        res2 = self.w2(res)
        x = res2

        return res2


class ExpertLayer(nn.Module):
    """
    Expert Layer containing n experts for Mixture-of-Experts (MoE) models.

    Attributes:
        num_experts (int): Number of experts in the layer.
        dim (int): Input and output dimensionality.
        inter_dim (int): Hidden layer dimensionality.
        experts (nn.ModuleList): List of expert modules.
    """

    def __init__(self, num_experts: int, dim: int, inter_dim: int):
        """
        Initializes the ExpertLayer.

        Args:
            num_experts (int): Number of experts in the layer.
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.inter_dim = inter_dim

        # Create the expert modules
        self.experts = nn.ModuleList([Expert(dim, inter_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ExpertLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        # Process input through all experts and add their outputs
        combined_output = torch.zeros_like(x)
        for expert in self.experts:
            expert_output = expert(x)
            combined_output += expert_output

        return combined_output


class Model(nn.Module):
    """
    Model containing n ExpertLayers that run sequentially.

    Attributes:
        num_layers (int): Number of ExpertLayer instances.
        dim (int): Input and output dimensionality.
        inter_dim (int): Hidden layer dimensionality.
        num_experts (int): Number of experts per ExpertLayer.
        layers (nn.ModuleList): List of ExpertLayer modules.
        input_projection (nn.Linear): Linear layer for input projection.
        output_projection (nn.Linear): Linear layer for output projection.
    """

    def __init__(self, num_layers: int, dim: int, inter_dim: int, num_experts: int, tensor_manager, iterations: int):
        """
        Initializes the Model.

        Args:
            num_layers (int): Number of ExpertLayer instances.
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
            num_experts (int): Number of experts per ExpertLayer.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.inter_dim = inter_dim
        self.num_experts = num_experts
        self.tensor_manager = tensor_manager

        # Input projection layer
        self.input_projection = nn.Linear(dim, dim)

        # Create the ExpertLayer modules
        self.layers = nn.ModuleList([ExpertLayer(num_experts, dim, inter_dim) for _ in range(num_layers)])

        # Output projection layer
        self.output_projection = nn.Linear(dim, dim)
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after processing through all layers.
        """
        # Input projection
        with self.tensor_manager.trap("input") as _trap:
            for _i in range(self.iterations):
                x = self.input_projection(x)

        # Process through all ExpertLayers sequentially
        idx = 0
        for layer in self.layers:
            with self.tensor_manager.trap("layer_" + str(idx)) as _trap:
                for _i in range(self.iterations):
                    x = layer(x)
                idx += 1

        # Output projection
        with self.tensor_manager.trap("output") as _trap:
            for _i in range(self.iterations):
                x = self.output_projection(x)

        return x


class NonUniformModel(nn.Module):
    """
    Model containing n ExpertLayers that run sequentially.

    """

    def __init__(self, num_layers: int, dim: int, inter_dim: int, num_experts_list, tensor_manager, iterations: int):
        """
        Initializes the Model.

        Args:
            num_layers (int): Number of ExpertLayer instances.
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
            num_experts_list (list): List of number of experts per ExpertLayer.
            tensor_manager: Tensor manager for offloading.
            iterations (int): Number of iterations.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.inter_dim = inter_dim
        self.num_experts_list = num_experts_list
        self.tensor_manager = tensor_manager
        self.iterations = iterations

        # Input projection layer
        self.input_projection = nn.Linear(dim, dim)

        # Create the ExpertLayer modules
        idx = 0
        experts_layer = []
        for _i in range(num_layers):
            experts_layer.append(ExpertLayer(num_experts_list[idx], dim, inter_dim))
            idx += 1
            if idx >= len(num_experts_list):
                idx = 0

        self.layers = nn.ModuleList(experts_layer)

        # Output projection layer
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after processing through all layers.
        """
        # Input projection
        with self.tensor_manager.trap("input") as _trap:
            for _i in range(self.iterations):
                x = self.input_projection(x)

        # Process through all ExpertLayers sequentially
        idx = 0
        for layer in self.layers:
            with self.tensor_manager.trap("layer_" + str(idx)) as _trap:
                for _i in range(self.iterations):
                    x = layer(x)
                idx += 1

        # Output projection
        with self.tensor_manager.trap("output") as _trap:
            for _i in range(self.iterations):
                x = self.output_projection(x)

        return x


class RepeatingProjection(nn.Module):
    """Linear projection that repeats its computation N times within a single forward call.

    When OffloadManager auto-traps this module, the iteration loop runs inside the
    trap block -- matching the behavior of manual traps in the old Model class.

    Attributes:
        linear (nn.Linear): The underlying linear layer.
        iterations (int): Number of times to apply the projection per forward call.
    """

    def __init__(self, dim: int, iterations: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iterations):
            x = self.linear(x)
        return x


class RepeatingExpertLayer(nn.Module):
    """ExpertLayer wrapper that repeats its computation N times within a single forward call.

    When OffloadManager auto-traps this module, the iteration loop runs inside the
    trap block -- matching the behavior of manual traps in the old Model class.

    Attributes:
        layer (ExpertLayer): The underlying expert layer.
        iterations (int): Number of times to apply the layer per forward call.
    """

    def __init__(self, num_experts: int, dim: int, inter_dim: int, iterations: int):
        super().__init__()
        self.layer = ExpertLayer(num_experts, dim, inter_dim)
        self.iterations = iterations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.iterations):
            x = self.layer(x)
        return x


class AutoTrapModel(nn.Module):
    """Expert model for OffloadManager auto-trap testing.

    Structured as a clean nn.Module hierarchy without any tensor_manager dependency.
    OffloadManager patches sub-module forwards with trap blocks automatically via
    module_patterns (e.g. ``["input_projection", "layers.*", "output_projection"]``).

    Uses RepeatingProjection and RepeatingExpertLayer so the iteration loop runs
    inside the auto-trap block, producing the same workload as the manual-trap Model.

    Attributes:
        input_projection (RepeatingProjection): Projection with iteration loop.
        layers (nn.ModuleList): ModuleList of RepeatingExpertLayer modules.
        output_projection (RepeatingProjection): Projection with iteration loop.
    """

    def __init__(self, num_layers: int, dim: int, inter_dim: int, num_experts: int, iterations: int = 1):
        """Initialize the AutoTrapModel.

        Args:
            num_layers: Number of ExpertLayer instances.
            dim: Input and output dimensionality.
            inter_dim: Hidden layer dimensionality.
            num_experts: Number of experts per ExpertLayer.
            iterations: Number of compute iterations per trap block (matches old Model behavior).
        """
        super().__init__()
        self.input_projection = RepeatingProjection(dim, iterations)
        self.layers = nn.ModuleList([
            RepeatingExpertLayer(num_experts, dim, inter_dim, iterations) for _ in range(num_layers)
        ])
        self.output_projection = RepeatingProjection(dim, iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through input projection, expert layers, and output projection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Output tensor after processing through all layers.
        """
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_projection(x)
        return x


class AutoTrapNonUniformModel(nn.Module):
    """Non-uniform expert model for OffloadManager auto-trap testing.

    Like AutoTrapModel but with variable expert counts per layer. Does not take
    a tensor_manager parameter; OffloadManager handles trap injection automatically.

    Attributes:
        input_projection (RepeatingProjection): Projection with iteration loop.
        layers (nn.ModuleList): ModuleList of RepeatingExpertLayer modules with varying expert counts.
        output_projection (RepeatingProjection): Projection with iteration loop.
    """

    def __init__(self, num_layers: int, dim: int, inter_dim: int, num_experts_list: list[int], iterations: int = 1):
        """Initialize the AutoTrapNonUniformModel.

        Args:
            num_layers: Number of ExpertLayer instances.
            dim: Input and output dimensionality.
            inter_dim: Hidden layer dimensionality.
            num_experts_list: List of expert counts per layer (cycled if shorter than num_layers).
            iterations: Number of compute iterations per trap block (matches old Model behavior).
        """
        super().__init__()
        self.input_projection = RepeatingProjection(dim, iterations)
        experts = []
        idx = 0
        for _ in range(num_layers):
            experts.append(RepeatingExpertLayer(num_experts_list[idx], dim, inter_dim, iterations))
            idx = (idx + 1) % len(num_experts_list)
        self.layers = nn.ModuleList(experts)
        self.output_projection = RepeatingProjection(dim, iterations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through input projection, expert layers, and output projection.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Output tensor after processing through all layers.
        """
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_projection(x)
        return x
