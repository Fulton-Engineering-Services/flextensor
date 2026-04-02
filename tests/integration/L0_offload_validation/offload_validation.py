# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation of tensor offloading functionality."""

import argparse
import hashlib
import random
import time
import uuid
from typing import ClassVar

import numpy as np
import torch
from config_manager import ConfigManager, ExperimentConfig
from models_mock import AutoTrapModel, AutoTrapNonUniformModel, Model, NonUniformModel
from torch import nn

from flextensor import (
    AdaptiveStrategy,
    GlobalOffloadStrategy,
    GlobalTensorSelectionStrategy,
    KnapsackBlockStrategy,
    KnapsackStrategy,
    OffloadConfig,
    TensorManager,
    get_offload_manager,
)
from flextensor.benchmark_tensor_mode import calculate_tensor_size
from flextensor.helpers import NoOpTensorManager
from flextensor.offload_manager import OffloadState


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_tensor_checksum(tensor: torch.Tensor) -> str:
    """
    Calculate MD5 checksum of a tensor's data.

    Args:
        tensor: Input tensor

    Returns:
        MD5 checksum as hex string

    Note:
        For bfloat16 and float16, converts to float32 first since numpy doesn't support these types.
    """
    # Move to CPU if needed
    tensor_cpu = tensor.detach().cpu().contiguous()

    # For bfloat16 and float16, convert to float32 first (numpy doesn't support these types)
    if tensor_cpu.dtype in (torch.bfloat16, torch.float16):
        tensor_cpu = tensor_cpu.to(torch.float32)

    # Convert to numpy and get bytes
    tensor_bytes = tensor_cpu.numpy().tobytes()

    # Calculate MD5 hash of the bytes
    md5_hash = hashlib.md5(tensor_bytes, usedforsecurity=False)
    return md5_hash.hexdigest()


class ModelUtils:
    """Utilities for model preparation and computation."""

    @staticmethod
    def prepare_basic_model_weights(layers: int, shape: torch.Size) -> dict[str, torch.Tensor]:
        """Prepare basic model weights on CPU."""
        device_cpu = torch.device("cpu")
        model = {}

        for i in range(layers):
            model[f"layers.{i}.feed_forward.w1.weight"] = torch.rand(shape, device=device_cpu)
            model[f"layers.{i}.feed_forward.w3.weight"] = torch.rand(shape, device=device_cpu)

        return model

    @staticmethod
    def prepare_expert_model(config: ExperimentConfig, tensor_manager: TensorManager) -> nn.Module:
        """Prepare expert model with tensor manager integration."""
        # Handle non-uniform models
        if config.model_type == "non_uniform":
            if config.num_experts_list is None:
                config.num_experts_list = [config.num_experts] * config.layers

            model = NonUniformModel(
                config.layers,
                config.dim,
                config.inter_dim,
                config.num_experts_list,
                tensor_manager,
                config.iterations,
            )
        else:  # "expert"
            model = Model(
                config.layers,
                config.dim,
                config.inter_dim,
                config.num_experts,
                tensor_manager,
                config.iterations,
            )

        return model

    @staticmethod
    def compute_model_size(model: dict[str, torch.Tensor] | nn.Module) -> float:
        """Calculate total model size in MB."""
        if isinstance(model, dict):
            return sum(calculate_tensor_size(tensor) for tensor in model.values()) / 1024 / 1024
        total_params = sum(p.numel() for p in model.parameters())
        # Assuming bfloat16 (2 bytes per parameter)
        return (total_params * 2) / (1024 * 1024)

    @staticmethod
    def create_input_tensor(config: ExperimentConfig, device: torch.device) -> torch.Tensor:
        """Create appropriate input tensor based on model type."""
        if config.model_type == "basic":
            # For basic model, create tensor matching the first layer shape
            shape = torch.Size(config.tensor_shape)
            return torch.ones(shape, device=device, dtype=config.tensor_dtype)
        # For expert models
        return torch.randn(
            config.batch_size,
            config.seq_len,
            config.dim,
            device=device,
            dtype=config.tensor_dtype,
        )


class ComputeEngine:
    """Handles computations for different model types."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def compute_basic_model(
        self,
        layers: int,
        input_tensor: torch.Tensor,
        model: dict[str, torch.Tensor],
        tensor_manager: TensorManager,
    ) -> tuple[torch.Tensor, float]:
        """Execute basic model computation across layers with tensor management."""
        res = input_tensor
        start_ns = time.time_ns()
        res = res * 1.5

        for i in range(layers):
            layer_name = f"layer.{i}"
            with tensor_manager.trap(layer_name):
                w1_name = f"layers.{i}.feed_forward.w1.weight"
                w3_name = f"layers.{i}.feed_forward.w3.weight"
                w1 = model[w1_name]
                w3 = model[w3_name]

                # Forward pass
                for _ in range(self.config.iterations):
                    res = res + w1
                    res = res + w3

                # Backward pass
                for _ in range(self.config.iterations):
                    res = res - w1
                    res = res - w3

        torch.cuda.synchronize()
        end_ns = time.time_ns()
        time_ms = (end_ns - start_ns) / 1e6

        return res, time_ms

    def compute_expert_model(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        tensor_manager: TensorManager,
    ) -> tuple[torch.Tensor, float]:
        """Execute expert model computation with tensor management.

        Runs model forward in a feedback loop of config.feedback_iters passes
        where each iteration's output becomes the next iteration's input. This
        tests that offloading does not corrupt tensor data across consecutive
        passes. The per-module computation volume is controlled by
        config.iterations inside the model's trap blocks.
        """
        start_ns = time.time_ns()

        result = input_tensor
        for _ in range(self.config.feedback_iters):
            result = model(result)

        torch.cuda.synchronize()
        end_ns = time.time_ns()
        time_ms = (end_ns - start_ns) / 1e6

        return result, time_ms

    def compute(
        self,
        input_tensor: torch.Tensor,
        model: dict[str, torch.Tensor] | nn.Module,
        tensor_manager: TensorManager,
    ) -> tuple[torch.Tensor, float]:
        """Execute computation based on model type."""
        if self.config.model_type == "basic":
            return self.compute_basic_model(
                self.config.layers,
                input_tensor,
                model,
                tensor_manager,
            )
        return self.compute_expert_model(
            input_tensor,
            model,
            tensor_manager,
        )


class MemoryTracker:
    """Tracks CUDA memory usage."""

    @staticmethod
    def cuda_usage_mb() -> float:
        """Get peak allocated CUDA memory in MB."""
        stats = torch.cuda.memory_stats()
        return stats["allocated_bytes.all.peak"] / (1024 * 1024)

    @staticmethod
    def reset_memory_stats():
        """Reset CUDA memory statistics."""
        torch.cuda.reset_peak_memory_stats()


class ExperimentResults:
    """Container for experiment results."""

    def __init__(self):
        self.warmup_time_ms: float = 0
        self.warmup_memory_mb: float = 0
        self.profile_time_ms: float = 0
        self.profile_memory_mb: float = 0
        self.profile_load_time_ms: float = 0
        self.infer_time_ms: float = 0
        self.infer_memory_mb: float = 0
        self.model_size_mb: float = 0
        self.output_checksum: str = ""
        self.warmup_vs_profile_match: bool = True
        self.profile_vs_inference_match: bool = True

    def print_summary(self):
        """Print comprehensive experiment results."""
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)

        print(f"Model size: {self.model_size_mb:.2f} MB")
        print(f"Warmup - Time: {self.warmup_time_ms:.2f}ms, Memory: {self.warmup_memory_mb:.2f}MB")
        print(f"Profile - Time: {self.profile_time_ms:.2f}ms, Memory: {self.profile_memory_mb:.2f}MB")
        print(f"Inference - Time: {self.infer_time_ms:.2f}ms, Memory: {self.infer_memory_mb:.2f}MB")
        if self.output_checksum:
            print(f"Output checksum (MD5): {self.output_checksum}")


class ModelFactory:
    """Factory for creating model instances based on configuration."""

    @staticmethod
    def create_basic_model(
        config: ExperimentConfig,
        _tensor_manager: TensorManager,
        device_gpu: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Create a basic model with weights dictionary.

        Args:
            config: Experiment configuration
            tensor_manager: TensorManager instance for benchmarking
            device_gpu: GPU device

        Returns:
            Tuple of (model weights dict, input tensor)
        """
        # Always create model on CPU first to ensure consistent RNG across baseline and offload
        # PyTorch has separate RNG states for CPU and CUDA, so creating on different devices
        # would result in different weight initialization even with the same seed
        shape = torch.Size(config.tensor_shape)
        with torch.device("cpu"):
            model = ModelUtils.prepare_basic_model_weights(config.layers, shape)

        # Move to appropriate device and convert dtype
        # Baseline: move to GPU, Offload: stay on CPU (will be managed by TensorManager)
        target_device = device_gpu if config.baseline_mode else torch.device("cpu")
        model = {name: layer.to(dtype=config.tensor_dtype, device=target_device) for name, layer in model.items()}

        # Create input tensor
        input_tensor = ModelUtils.create_input_tensor(config, device_gpu)

        return model, input_tensor

    @staticmethod
    def create_expert_model(
        config: ExperimentConfig,
        tensor_manager: TensorManager,
        device_gpu: torch.device,
    ) -> tuple[nn.Module, torch.Tensor]:
        """
        Create an expert model (uniform or non-uniform).

        Args:
            config: Experiment configuration
            tensor_manager: TensorManager instance for preprocessing
            device_gpu: GPU device

        Returns:
            Tuple of (model, input tensor)
        """
        # Always create model on CPU first to ensure consistent RNG across baseline and offload
        # PyTorch has separate RNG states for CPU and CUDA, so creating on different devices
        # would result in different weight initialization even with the same seed
        with torch.device("cpu"):
            model = ModelUtils.prepare_expert_model(config, tensor_manager)

        # Move to appropriate device and convert dtype
        # Baseline: move to GPU, Offload: stay on CPU (will be managed by TensorManager)
        target_device = device_gpu if config.baseline_mode else torch.device("cpu")
        model = model.to(dtype=config.tensor_dtype, device=target_device)

        model = model.eval()

        # Create input tensor
        input_tensor = ModelUtils.create_input_tensor(config, device_gpu)

        return model, input_tensor

    @staticmethod
    def create_model(
        config: ExperimentConfig,
        tensor_manager: TensorManager,
        device_gpu: torch.device,
    ) -> tuple[dict[str, torch.Tensor] | nn.Module, torch.Tensor]:
        """
        Create a model based on configuration.

        Args:
            config: Experiment configuration
            tensor_manager: TensorManager instance
            device_gpu: GPU device

        Returns:
            Tuple of (model, input tensor)
        """
        if config.model_type == "basic":
            return ModelFactory.create_basic_model(config, tensor_manager, device_gpu)
        return ModelFactory.create_expert_model(config, tensor_manager, device_gpu)

    @staticmethod
    def create_auto_trap_model(
        config: ExperimentConfig,
        device_gpu: torch.device,
    ) -> tuple[nn.Module, torch.Tensor]:
        """Create an auto-trap model for OffloadManager (no manual trap calls).

        Args:
            config: Experiment configuration
            device_gpu: GPU device

        Returns:
            Tuple of (model on CPU, input tensor on GPU)

        Raises:
            ValueError: If model_type is "basic" (not supported for high-level API)
        """
        if config.model_type == "basic":
            raise ValueError("basic model type is not supported for high-level API (OffloadManager)")

        with torch.device("cpu"):
            if config.model_type == "non_uniform":
                num_experts_list = config.num_experts_list or [config.num_experts] * config.layers
                model = AutoTrapNonUniformModel(
                    config.layers, config.dim, config.inter_dim, num_experts_list, config.iterations
                )
            else:
                model = AutoTrapModel(
                    config.layers, config.dim, config.inter_dim, config.num_experts, config.iterations
                )

        target_device = device_gpu if config.baseline_mode else torch.device("cpu")
        model = model.to(dtype=config.tensor_dtype, device=target_device).eval()

        input_tensor = ModelUtils.create_input_tensor(config, device_gpu)
        return model, input_tensor


class TensorManagerFactory:
    """Factory for creating TensorManager instances based on configuration."""

    @staticmethod
    def create_tensor_manager(
        config: ExperimentConfig,
        device_gpu: torch.device,
    ) -> TensorManager | NoOpTensorManager:
        """
        Create a TensorManager from configuration.

        Args:
            config: Experiment configuration
            device_gpu: GPU device

        Returns:
            Configured TensorManager instance or NoOpTensorManager for baseline mode
        """
        # If baseline mode is enabled, return NoOpTensorManager for GPU-only execution
        if config.baseline_mode:
            return NoOpTensorManager(device_gpu)

        # Initialize tensor manager strategy based on strategy_type
        strategy_type = getattr(config, "strategy_type", "knapsack")
        n_blocks = getattr(config, "n_blocks", 4)
        max_gpu_mem_gb = getattr(config, "max_gpu_mem_gb", 48.0)
        _, total_gpu_mem = torch.cuda.mem_get_info(device_gpu)
        max_gpu_mem_fraction = min((max_gpu_mem_gb * 1024**3) / total_gpu_mem, 1.0)

        if strategy_type == "global_offload":
            tensor_manager_load_strategy = GlobalOffloadStrategy(
                n_blocks=n_blocks,
                threshold_mb=1.0,
            )
        elif strategy_type == "global_tensor_selection":
            tensor_manager_load_strategy = GlobalTensorSelectionStrategy(
                n_blocks=n_blocks,
                threshold_mb=1.0,
                pop_size=30,
                epoch=50,
                max_early_stop=25,
                scale=0.9,
            )
        elif strategy_type == "adaptive":
            tensor_manager_load_strategy = AdaptiveStrategy(
                scale=config.knapsack_scale,
                loader_type=config.transfer_mode,
                n_blocks=n_blocks,
            )
        elif strategy_type == "knapsack_block":
            tensor_manager_load_strategy = KnapsackBlockStrategy(
                scale=config.knapsack_scale,
                threshold_mb=1.0,
                n_blocks=n_blocks,
            )
        elif config.model_type == "basic":
            tensor_manager_load_strategy = KnapsackStrategy(
                scale=config.knapsack_scale,
                n_blocks=n_blocks,
            )
        else:
            tensor_manager_load_strategy = KnapsackStrategy(
                scale=config.knapsack_scale,
                cyclic=False,
                group_size=1,
                n_blocks=n_blocks,
            )

        remove_layers_operations = []
        use_trace_tensor = False if config.model_type != "basic" else None

        tensor_manager_kwargs = {
            "remove_layers_operations": remove_layers_operations,
            "_rearrange_transfers": config.rearrange_transfers,
            "_compute_transfer_gap": config.compute_transfer_gap,
            "loader_type": config.transfer_mode,
            "max_gpu_mem_fraction": max_gpu_mem_fraction,
        }

        if use_trace_tensor is not None:
            tensor_manager_kwargs["_use_trace_tensor"] = use_trace_tensor

        return TensorManager(
            device_gpu,
            tensor_manager_load_strategy,
            pinned_memory=config.pinned_memory,
            **tensor_manager_kwargs,
        )


class CombinedTensorOffloadExperiment:
    """
    Main experiment orchestrator for combined tensor offloading tests.

    This class requires a pre-configured TensorManager and model to be provided externally,
    allowing for flexible configuration and separation of concerns.

    Example usage:
        ```python
        config = ExperimentConfig(model_type="basic", layers=8, iterations=10)
        device_gpu = torch.device("cuda")

        # Create tensor manager externally for full control
        tensor_manager = TensorManagerFactory.create_tensor_manager(config, device_gpu)

        # Create model externally
        model, input_tensor = ModelFactory.create_model(config, tensor_manager, device_gpu)

        # Run experiment with the provided tensor manager and model
        experiment = CombinedTensorOffloadExperiment(config, tensor_manager, model, input_tensor)
        experiment.run_experiment()
        ```
    """

    PROFILE_ITERS = 3

    def __init__(
        self,
        config: ExperimentConfig,
        tensor_manager: TensorManager,
        model: dict[str, torch.Tensor] | nn.Module,
        input_tensor: torch.Tensor,
    ):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration
            tensor_manager: Pre-configured TensorManager instance
            model: Pre-created model (either dict of tensors or nn.Module)
            input_tensor: Pre-created input tensor for the model
        """
        self.config = config
        self.tensor_manager = tensor_manager
        self.model = model
        self.input_tensor = input_tensor
        self.compute_engine = ComputeEngine(config)
        self.memory_tracker = MemoryTracker()
        self.results = ExperimentResults()

        # Initialize devices
        self.device_gpu = torch.device("cuda")

        # Calculate and store model size
        self.results.model_size_mb = ModelUtils.compute_model_size(model)

    def run_warmup_phase(
        self,
        model: dict[str, torch.Tensor] | nn.Module,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Run warmup phase with lazy loading."""
        print("Running warmup phase (lazy loading)...")

        self.memory_tracker.reset_memory_stats()

        res, time_ms = self.compute_engine.compute(
            input_tensor,
            model,
            self.tensor_manager,
        )

        self.results.warmup_time_ms = time_ms
        self.results.warmup_memory_mb = self.memory_tracker.cuda_usage_mb()

        print(f"Warmup - Time: {time_ms:.2f}ms, Memory: {self.results.warmup_memory_mb:.2f}MB")
        return res

    def run_profile_phase(
        self,
        model: dict[str, torch.Tensor] | nn.Module,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Run profile phase with preloaded tensors."""
        print("Running profile phase (tensor preloading)...")

        self.memory_tracker.reset_memory_stats()

        res, time_ms = self.compute_engine.compute(
            input_tensor,
            model,
            self.tensor_manager,
        )

        self.results.profile_time_ms = time_ms
        self.results.profile_memory_mb = self.memory_tracker.cuda_usage_mb()

        print(f"Profile - Time: {time_ms:.2f}ms, Memory: {self.results.profile_memory_mb:.2f}MB")

        return res

    def run_inference_phase(
        self,
        model: dict[str, torch.Tensor] | nn.Module,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference phase with managed tensors."""
        print("Running inference phase (managed tensors)...")

        # Reset loader and prepare inference mode
        self.memory_tracker.reset_memory_stats()
        res, time_ms = self.compute_engine.compute(
            input_tensor,
            model,
            self.tensor_manager,
        )

        self.tensor_manager.release_memory()

        self.results.infer_time_ms = time_ms
        self.results.infer_memory_mb = self.memory_tracker.cuda_usage_mb()

        # Calculate and store output checksum
        self.results.output_checksum = calculate_tensor_checksum(res)

        print(f"Inference - Time: {time_ms:.2f}ms, Memory: {self.results.infer_memory_mb:.2f}MB")
        print(f"Output checksum: {self.results.output_checksum[:16]}...")

        return res

    def validate_results(self, result1: torch.Tensor, result2: torch.Tensor, message: str = "Results") -> bool:
        """Validate that two computation results are equivalent.

        Returns:
            True if results match, False otherwise.
        """
        print(f"Validating {message}...")

        results_match = torch.equal(result1, result2)
        print(f"{message} match: {results_match}")

        if results_match:
            torch.testing.assert_close(result1, result2)
            print(f"✓ {message} validated successfully!")
        else:
            print(f"✗ {message} validation failed - results don't match!")

        return results_match

    def run_experiment(self):
        """Run the complete combined tensor offloading experiment."""
        print("Starting Combined Tensor Offloading Experiment")
        print("=" * 50)
        print("Configuration:")
        print(f"  Model type: {self.config.model_type}")
        print(f"  Layers: {self.config.layers}")
        if self.config.model_type == "basic":
            print(f"  Tensor shape: {self.config.tensor_shape}")
        else:
            print(f"  Dimensions: {self.config.dim} -> {self.config.inter_dim}")
            print(f"  Experts: {self.config.num_experts}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Sequence length: {self.config.seq_len}")
        print(f"  Iterations: {self.config.iterations}")
        print(f"  Pinned memory: {self.config.pinned_memory}")
        print(f"  Data type: {self.config.tensor_dtype}")
        print(f"  Rearrange transfers: {self.config.rearrange_transfers}")
        print(f"  Transfer mode: {self.config.transfer_mode}")
        print(f"  Baseline mode (GPU only): {self.config.baseline_mode}")
        print("-" * 50)

        with torch.no_grad():
            # Use pre-created model and input tensor
            model = self.model
            input_tensor = self.input_tensor

            if self.config.model_type == "basic":
                print(f"Pinned memory active: {all(tensor.is_pinned() for tensor in model.values())}")
            print("-" * 50)

            # Run stepwise profile mode
            self.tensor_manager.set_model(self.model)
            model = self.tensor_manager.initialize_warmup()
            res_warmup = self.run_warmup_phase(model, input_tensor)
            model = self.tensor_manager.initialize_profile()
            print(f"Running {self.PROFILE_ITERS} profile iterations...")
            for i in range(self.PROFILE_ITERS):
                print(f"  Profile iteration {i + 1}/{self.PROFILE_ITERS}")
                res_profile = self.run_profile_phase(model, input_tensor)

            model = self.tensor_manager.initialize_inference()
            res_infer = self.run_inference_phase(model, input_tensor)

            # Validate results
            self.results.warmup_vs_profile_match = self.validate_results(res_warmup, res_profile, "warmup vs profile")

            # Validate inference results
            self.results.profile_vs_inference_match = self.validate_results(
                res_profile, res_infer, "profile vs inference"
            )
            self.results.print_summary()

            # Raise error if any validation failed
            if not self.results.warmup_vs_profile_match or not self.results.profile_vs_inference_match:
                validation_errors = []
                if not self.results.warmup_vs_profile_match:
                    validation_errors.append("warmup vs profile")
                if not self.results.profile_vs_inference_match:
                    validation_errors.append("profile vs inference")
                raise ValueError(f"Validation failed: {', '.join(validation_errors)}")


class OffloadManagerExperiment:
    """Experiment orchestrator using the high-level OffloadManager API.

    Uses OffloadManager's auto-trap module patching and automatic state transitions
    (warmup -> profile -> inference) instead of manual TensorManager calls.
    Produces the same ExperimentResults as CombinedTensorOffloadExperiment for
    unified summary reporting.
    """

    MODULE_PATTERNS: ClassVar[list[str]] = ["input_projection", "layers.*", "output_projection"]
    WARMUP_ITERS = 1
    PROFILE_ITERS = 3

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        input_tensor: torch.Tensor,
    ):
        self.config = config
        self.model = model
        self.input_tensor = input_tensor
        self.results = ExperimentResults()
        self.device_gpu = torch.device("cuda")
        self.memory_tracker = MemoryTracker()
        self.results.model_size_mb = ModelUtils.compute_model_size(model)

    def run_experiment(self):
        """Run the complete OffloadManager experiment."""
        print("Starting OffloadManager Experiment (High-Level API)")
        print("=" * 50)
        print("  API type: high_level (OffloadManager)")
        print(f"  Model type: {self.config.model_type}")
        print(f"  Layers: {self.config.layers}")
        print(f"  Dimensions: {self.config.dim} -> {self.config.inter_dim}")
        print(f"  Experts: {self.config.num_experts}")
        print(f"  Feedback iters: {self.config.feedback_iters}")
        print(f"  Baseline mode: {self.config.baseline_mode}")
        if not self.config.baseline_mode:
            print(f"  Transfer mode: {self.config.transfer_mode}")
            print(f"  Module patterns: {self.MODULE_PATTERNS}")
            print(f"  Warmup iters: {self.WARMUP_ITERS}")
            print(f"  Profile iters: {self.PROFILE_ITERS}")
        print("-" * 50)

        with torch.no_grad():
            if self.config.baseline_mode:
                self._run_baseline()
            else:
                self._run_offloaded()

        self.results.print_summary()

        if not self.results.warmup_vs_profile_match or not self.results.profile_vs_inference_match:
            validation_errors = []
            if not self.results.warmup_vs_profile_match:
                validation_errors.append("warmup vs profile")
            if not self.results.profile_vs_inference_match:
                validation_errors.append("profile vs inference")
            raise ValueError(f"Validation failed: {', '.join(validation_errors)}")

    def _run_baseline(self):
        """Run model directly on GPU without offloading."""
        model = self.model
        x = self.input_tensor

        feedback_iters = self.config.feedback_iters

        # Warmup
        print(f"Running warmup phase ({feedback_iters} feedback iterations, GPU baseline)...")
        self.memory_tracker.reset_memory_stats()
        start_ns = time.time_ns()
        res_warmup = x
        for _ in range(feedback_iters):
            res_warmup = model(res_warmup)
        torch.cuda.synchronize()
        self.results.warmup_time_ms = (time.time_ns() - start_ns) / 1e6
        self.results.warmup_memory_mb = self.memory_tracker.cuda_usage_mb()
        print(f"Warmup - Time: {self.results.warmup_time_ms:.2f}ms, Memory: {self.results.warmup_memory_mb:.2f}MB")

        # Profile
        print(f"Running profile phase ({feedback_iters} feedback iterations, GPU baseline)...")
        self.memory_tracker.reset_memory_stats()
        start_ns = time.time_ns()
        res_profile = x
        for _ in range(feedback_iters):
            res_profile = model(res_profile)
        torch.cuda.synchronize()
        self.results.profile_time_ms = (time.time_ns() - start_ns) / 1e6
        self.results.profile_memory_mb = self.memory_tracker.cuda_usage_mb()
        print(f"Profile - Time: {self.results.profile_time_ms:.2f}ms, Memory: {self.results.profile_memory_mb:.2f}MB")

        # Inference
        print(f"Running inference phase ({feedback_iters} feedback iterations, GPU baseline)...")
        self.memory_tracker.reset_memory_stats()
        start_ns = time.time_ns()
        res_infer = x
        for _ in range(feedback_iters):
            res_infer = model(res_infer)
        torch.cuda.synchronize()
        self.results.infer_time_ms = (time.time_ns() - start_ns) / 1e6
        self.results.infer_memory_mb = self.memory_tracker.cuda_usage_mb()
        self.results.output_checksum = calculate_tensor_checksum(res_infer)
        print(f"Inference - Time: {self.results.infer_time_ms:.2f}ms, Memory: {self.results.infer_memory_mb:.2f}MB")
        print(f"Output checksum: {self.results.output_checksum[:16]}...")

        # Validate
        self.results.warmup_vs_profile_match = self._validate(res_warmup, res_profile, "warmup vs profile")
        self.results.profile_vs_inference_match = self._validate(res_profile, res_infer, "profile vs inference")

    def _run_offloaded(self):
        """Run model with OffloadManager auto-trap and AdaptiveStrategy."""
        feedback_iters = self.config.feedback_iters

        # OffloadManager counts each proxy() call as one iteration for state
        # transitions, so multiply by feedback_iters to account for the inner
        # feedback loop within each logical iteration.
        offload_config = OffloadConfig(
            module_patterns=self.MODULE_PATTERNS,
            warmup_iters=self.WARMUP_ITERS * feedback_iters,
            profile_iters=self.PROFILE_ITERS * feedback_iters,
            transfer_mode=self.config.transfer_mode,
            knapsack_scale=self.config.knapsack_scale,
            num_blocks=self.config.n_blocks,
            pinned_memory=self.config.pinned_memory,
        )

        manager_name = f"test_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        proxy = om.offload(self.model, offload_config)

        x = self.input_tensor

        try:
            # Warmup phase
            print(f"Running warmup phase ({self.WARMUP_ITERS}x{feedback_iters} iterations)...")
            self.memory_tracker.reset_memory_stats()
            start_ns = time.time_ns()
            for _ in range(self.WARMUP_ITERS):
                res_warmup = x
                for _ in range(feedback_iters):
                    res_warmup = proxy(res_warmup)
            torch.cuda.synchronize()
            self.results.warmup_time_ms = (time.time_ns() - start_ns) / 1e6
            self.results.warmup_memory_mb = self.memory_tracker.cuda_usage_mb()
            warmup_ms = self.results.warmup_time_ms
            warmup_mem = self.results.warmup_memory_mb
            print(f"Warmup - Time: {warmup_ms:.2f}ms, Memory: {warmup_mem:.2f}MB")

            # Profile phase
            print(f"Running profile phase ({self.PROFILE_ITERS}x{feedback_iters} iterations)...")
            self.memory_tracker.reset_memory_stats()
            start_ns = time.time_ns()
            for i in range(self.PROFILE_ITERS):
                res = x
                for _ in range(feedback_iters):
                    res = proxy(res)
                if i == 0:
                    res_profile = res
            torch.cuda.synchronize()
            self.results.profile_time_ms = (time.time_ns() - start_ns) / 1e6
            self.results.profile_memory_mb = self.memory_tracker.cuda_usage_mb()
            profile_ms = self.results.profile_time_ms
            profile_mem = self.results.profile_memory_mb
            print(f"Profile - Time: {profile_ms:.2f}ms, Memory: {profile_mem:.2f}MB")

            # Verify OffloadManager reached INFERENCE state after warmup + profile
            assert om._current_state == OffloadState.INFERENCE, (
                f"Expected INFERENCE state, got {om._current_state.value}"
            )

            # Inference phase
            print(f"Running inference phase ({feedback_iters} feedback iterations)...")
            self.memory_tracker.reset_memory_stats()
            start_ns = time.time_ns()
            res_infer = x
            for _ in range(feedback_iters):
                res_infer = proxy(res_infer)
            torch.cuda.synchronize()
            self.results.infer_time_ms = (time.time_ns() - start_ns) / 1e6
            self.results.infer_memory_mb = self.memory_tracker.cuda_usage_mb()
            self.results.output_checksum = calculate_tensor_checksum(res_infer)
            print(f"Inference - Time: {self.results.infer_time_ms:.2f}ms, Memory: {self.results.infer_memory_mb:.2f}MB")
            print(f"Output checksum: {self.results.output_checksum[:16]}...")

            # Validate
            self.results.warmup_vs_profile_match = self._validate(res_warmup, res_profile, "warmup vs profile")
            self.results.profile_vs_inference_match = self._validate(res_profile, res_infer, "profile vs inference")
        finally:
            om.release()

    def _validate(self, result1: torch.Tensor, result2: torch.Tensor, message: str) -> bool:
        """Validate that two computation results are equivalent."""
        print(f"Validating {message}...")
        results_match = torch.equal(result1, result2)
        if results_match:
            torch.testing.assert_close(result1, result2)
            print(f"✓ {message} validated successfully!")
        else:
            print(f"✗ {message} validation failed - results don't match!")
        return results_match


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tensor Offloading Validation",
        epilog="Use --list-presets to see available configurations, then --config <preset-name> to run a specific test.",  # noqa: E501
    )

    # Config file loading (primary mode of operation)
    parser.add_argument(
        "--config",
        type=str,
        help="Load configuration from preset name or config file (required unless using --list-presets)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available preset configurations and exit",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = no seed set)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    manager = ConfigManager()

    # Handle --list-presets
    if args.list_presets:
        presets = manager.list_presets()
        print("\nAvailable Preset Configurations:")
        print("=" * 60)
        for preset in presets:
            print(f"  • {preset}")
        print("\nUsage: --config <preset-name>")
        print("=" * 60)
        return

    # Require --config argument
    if not args.config:
        print("✗ Error: --config argument is required")
        print("\nUse --list-presets to see available configurations")
        print("Usage: python offload_validation.py --config <preset-name>")
        return

    # Load config from preset or file
    try:
        # Try as preset first
        config = manager.get_preset(args.config)
        print(f"✓ Loaded preset configuration: {args.config}")
    except KeyError:
        # Try as saved config file
        try:
            config = manager.load_config(args.config)
            print(f"✓ Loaded configuration from file: {args.config}")
        except FileNotFoundError:
            print(f"✗ Error: Config '{args.config}' not found as preset or file")
            print("\nAvailable presets:")
            for p in manager.list_presets():
                print(f"  • {p}")
            return

    # Override seed if provided via command line
    if args.seed is not None:
        config.seed = args.seed
        print(f"✓ Overriding seed with command-line value: {args.seed}")

    # Set random seed if specified
    if config.seed is not None:
        set_seed(config.seed)
        print(f"✓ Random seed set to: {config.seed}")

    # Initialize device
    device_gpu = torch.device("cuda")

    # Create tensor manager externally
    tensor_manager = TensorManagerFactory.create_tensor_manager(config, device_gpu)

    # Create model externally
    model, input_tensor = ModelFactory.create_model(config, tensor_manager, device_gpu)

    # Create and run experiment
    experiment = CombinedTensorOffloadExperiment(config, tensor_manager, model, input_tensor)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
