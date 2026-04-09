#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Configuration Manager for Tensor Offloading Experiments

Provides utilities to:
- Load configurations from JSON/YAML files
- Save configurations for reproducibility
- Access preset configurations
- Validate configurations
"""

import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field


class ModelPreset(BaseModel):
    """Configuration for model structure (separate from test/tensor manager config)."""

    model_type: Literal["basic", "expert", "non_uniform"] = Field(
        default="basic",
        description="Type of model to test",
    )
    layers: int = Field(default=64, description="Number of layers in the model", gt=0)
    tensor_shape: tuple[int, int] = Field(
        default=(14336, 4096),
        description="Shape of tensors for basic model",
    )
    tensor_dtype: torch.dtype = Field(
        default=torch.bfloat16,
        description="Data type for tensors",
    )
    iterations: int = Field(
        default=50,
        description="Number of compute iterations (simulates model workload)",
        gt=0,
    )
    feedback_iters: int = Field(
        default=2,
        description="Number of feedback iterations (output of pass N feeds into pass N+1)",
        gt=0,
    )

    # Expert model specific configuration
    dim: int = Field(default=4096, description="Input/output dimension for expert models", gt=0)
    inter_dim: int = Field(default=14336, description="Intermediate dimension for expert models", gt=0)
    num_experts: int = Field(default=8, description="Number of experts in expert models", gt=0)
    batch_size: int = Field(default=1, description="Batch size for expert models", gt=0)
    seq_len: int = Field(default=1024, description="Sequence length for expert models", gt=0)
    use_non_uniform: bool = Field(
        default=False,
        description="Whether to use non-uniform expert distribution",
    )
    num_experts_list: list | None = Field(
        default=None,
        description="List of expert counts per layer for non-uniform models",
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (None = no seed set)",
    )

    model_config = {"arbitrary_types_allowed": True}


class TestPreset(BaseModel):
    """Configuration for tensor manager and test execution (separate from model structure)."""

    api_type: Literal["low_level", "high_level"] = Field(
        default="low_level",
        description="API type: low_level (TensorManager manual traps) or high_level (OffloadManager auto-trap)",
    )
    transfer_mode: str = Field(
        default="strategy",
        description='Type of tensor transfer mode: "strategy", "raw_block_transfer", "allocation_block_transfer"',
    )
    pinned_memory: bool = Field(
        default=True,
        description="Whether to use pinned memory for transfers",
    )
    transfer_budget_scale: float = Field(
        default=1.0,
        description="Transfer budget scale factor for tensor manager",
        gt=0.0,
    )
    strategy_type: Literal["knapsack", "knapsack_block", "global_offload", "global_tensor_selection", "adaptive"] = (
        Field(
            default="knapsack",
            description="Type of offloading strategy",
        )
    )
    n_blocks: int = Field(
        default=4,
        description="Number of memory blocks for block-based strategies",
        gt=1,
    )
    max_gpu_mem_gb: float = Field(
        default=48.0,
        description="Maximum GPU memory in GB for GlobalOffloadStrategy",
        gt=0.0,
    )

    # Transfer optimization configuration
    rearrange_transfers: bool = Field(
        default=False,
        description="Whether to enable transfer rearrangement optimization",
    )
    compute_transfer_gap: int = Field(
        default=1,
        description="Minimum gap between compute and transfer layers",
        ge=0,
    )

    # Baseline mode (GPU only, no offloading)
    baseline_mode: bool = Field(
        default=False,
        description="Use NoOpTensorManager for GPU-only baseline",
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (None = no seed set)",
    )


class ExperimentConfig(BaseModel):
    """Configuration for combined tensor offloading experiments."""

    # API type
    api_type: Literal["low_level", "high_level"] = Field(
        default="low_level",
        description="API type: low_level (TensorManager manual traps) or high_level (OffloadManager auto-trap)",
    )

    # Model configuration
    model_type: Literal["basic", "expert", "non_uniform"] = Field(
        default="basic",
        description="Type of model to test",
    )
    layers: int = Field(default=64, description="Number of layers in the model", gt=0)
    tensor_shape: tuple[int, int] = Field(
        default=(14336, 4096),
        description="Shape of tensors for basic model",
    )

    # Expert model specific configuration
    dim: int = Field(default=4096, description="Input/output dimension for expert models", gt=0)
    inter_dim: int = Field(default=14336, description="Intermediate dimension for expert models", gt=0)
    num_experts: int = Field(default=8, description="Number of experts in expert models", gt=0)
    batch_size: int = Field(default=1, description="Batch size for expert models", gt=0)
    seq_len: int = Field(default=1024, description="Sequence length for expert models", gt=0)
    use_non_uniform: bool = Field(
        default=False,
        description="Whether to use non-uniform expert distribution",
    )
    num_experts_list: list | None = Field(
        default=None,
        description="List of expert counts per layer for non-uniform models",
    )

    # Execution configuration
    iterations: int = Field(
        default=50,
        description="Number of compute iterations (simulates model workload)",
        gt=0,
    )
    feedback_iters: int = Field(
        default=2,
        description="Number of feedback iterations (output of pass N feeds into pass N+1)",
        gt=0,
    )
    pinned_memory: bool = Field(
        default=True,
        description="Whether to use pinned memory for transfers",
    )
    transfer_budget_scale: float = Field(
        default=1.0, description="Transfer budget scale factor for tensor manager", gt=0.0
    )
    strategy_type: Literal["knapsack", "knapsack_block", "global_offload", "global_tensor_selection", "adaptive"] = (
        Field(
            default="knapsack",
            description="Type of offloading strategy",
        )
    )
    n_blocks: int = Field(
        default=4,
        description="Number of memory blocks for block-based strategies",
        gt=1,
    )
    max_gpu_mem_gb: float = Field(
        default=48.0,
        description="Maximum GPU memory in GB for GlobalOffloadStrategy",
        gt=0.0,
    )
    tensor_dtype: torch.dtype = Field(
        default=torch.bfloat16,
        description="Data type for tensors",
    )

    # Transfer optimization configuration
    rearrange_transfers: bool = Field(
        default=False,
        description="Whether to enable transfer rearrangement optimization",
    )
    compute_transfer_gap: int = Field(
        default=1,
        description="Minimum gap between compute and transfer layers",
        ge=0,
    )

    # TensorManager configuration
    transfer_mode: str = Field(
        default="strategy",
        description='Type of tensor transfer mode: "strategy", "raw_block_transfer", "allocation_block_transfer"',
    )

    # Baseline mode (GPU only, no offloading)
    baseline_mode: bool = Field(
        default=False,
        description="Use NoOpTensorManager for GPU-only baseline",
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility (None = no seed set)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_presets(cls, model: ModelPreset, test: TestPreset) -> "ExperimentConfig":
        """Create ExperimentConfig by combining ModelPreset and TestPreset."""
        # Prefer test.seed if provided, otherwise use model.seed
        seed = test.seed if test.seed is not None else model.seed

        return cls(
            # API type
            api_type=test.api_type,
            # Model configuration
            model_type=model.model_type,
            layers=model.layers,
            tensor_shape=model.tensor_shape,
            tensor_dtype=model.tensor_dtype,
            iterations=model.iterations,
            feedback_iters=model.feedback_iters,
            dim=model.dim,
            inter_dim=model.inter_dim,
            num_experts=model.num_experts,
            batch_size=model.batch_size,
            seq_len=model.seq_len,
            use_non_uniform=model.use_non_uniform,
            num_experts_list=model.num_experts_list,
            # Test configuration
            pinned_memory=test.pinned_memory,
            transfer_budget_scale=test.transfer_budget_scale,
            strategy_type=test.strategy_type,
            n_blocks=test.n_blocks,
            max_gpu_mem_gb=test.max_gpu_mem_gb,
            rearrange_transfers=test.rearrange_transfers,
            compute_transfer_gap=test.compute_transfer_gap,
            transfer_mode=test.transfer_mode,
            baseline_mode=test.baseline_mode,
            # Reproducibility
            seed=seed,
        )

    def to_model_preset(self) -> ModelPreset:
        """Extract model preset from this config."""
        return ModelPreset(
            model_type=self.model_type,
            layers=self.layers,
            tensor_shape=self.tensor_shape,
            tensor_dtype=self.tensor_dtype,
            iterations=self.iterations,
            feedback_iters=self.feedback_iters,
            dim=self.dim,
            inter_dim=self.inter_dim,
            num_experts=self.num_experts,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            use_non_uniform=self.use_non_uniform,
            num_experts_list=self.num_experts_list,
            seed=self.seed,
        )

    def to_test_preset(self) -> TestPreset:
        """Extract test preset from this config."""
        return TestPreset(
            api_type=self.api_type,
            transfer_mode=self.transfer_mode,
            pinned_memory=self.pinned_memory,
            transfer_budget_scale=self.transfer_budget_scale,
            strategy_type=self.strategy_type,
            n_blocks=self.n_blocks,
            max_gpu_mem_gb=self.max_gpu_mem_gb,
            rearrange_transfers=self.rearrange_transfers,
            compute_transfer_gap=self.compute_transfer_gap,
            baseline_mode=self.baseline_mode,
            seed=self.seed,
        )


class ConfigManager:
    """Manages experiment configurations - loading, saving, and presets."""

    def __init__(self, config_dir: Path | None = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing config files. Defaults to ./configs/
        """
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        # Initialize separate preset libraries
        self._model_presets = self._create_model_presets()
        self._test_presets = self._create_test_presets()

        # Initialize combined preset configs (backward compatibility)
        self._presets = self._create_presets()

    def _create_model_presets(self) -> dict[str, ModelPreset]:
        """Create a library of model presets."""
        models = {}

        # ========== Basic Model Presets ==========

        models["basic-small"] = ModelPreset(
            model_type="basic",
            layers=4,
            tensor_shape=(8192, 4096),
            iterations=5,
        )

        models["basic-medium"] = ModelPreset(
            model_type="basic",
            layers=8,
            tensor_shape=(14336, 4096),
            iterations=5,
        )

        models["basic-large"] = ModelPreset(
            model_type="basic",
            layers=16,
            tensor_shape=(14336, 4096),
            iterations=10,
        )

        models["basic-full"] = ModelPreset(
            model_type="basic",
            layers=16,
            tensor_shape=(14336, 4096),
            iterations=10,
        )

        # ========== Expert Model Presets ==========

        models["expert-small"] = ModelPreset(
            model_type="expert",
            layers=3,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            iterations=5,
        )

        models["expert-medium"] = ModelPreset(
            model_type="expert",
            layers=4,
            dim=4096,
            inter_dim=14336,
            num_experts=8,
            iterations=5,
        )

        models["expert-large"] = ModelPreset(
            model_type="expert",
            layers=8,
            dim=4096,
            inter_dim=14336,
            num_experts=16,
            iterations=10,
        )

        models["expert-8layer-8expert"] = ModelPreset(
            model_type="expert",
            layers=8,
            dim=1024,
            inter_dim=4096,
            num_experts=8,
            batch_size=1,
            seq_len=1024,
            iterations=4,
        )

        # ========== Non-Uniform Expert Model Presets ==========

        models["non-uniform-small"] = ModelPreset(
            model_type="non_uniform",
            use_non_uniform=True,
            layers=3,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            iterations=2,
        )

        models["non-uniform-16layer"] = ModelPreset(
            model_type="non_uniform",
            use_non_uniform=True,
            layers=16,
            dim=1024,
            inter_dim=4096,
            num_experts=5,
            iterations=4,
            num_experts_list=[
                4,
                3,
                1,
                4,
                5,
                1,
                4,
                5,
                4,
                4,
                3,
                4,
                3,
                4,
                5,
                2,
            ],
        )

        return models

    def _create_test_presets(self) -> dict[str, TestPreset]:
        """Create a library of test/tensor manager presets."""
        tests = {}

        # ========== Baseline (GPU-only) Presets ==========

        tests["baseline"] = TestPreset(
            baseline_mode=True,
        )

        # ========== Offloading Presets - Strategy Loader ==========

        tests["offload-strategy"] = TestPreset(
            transfer_mode="strategy",
            baseline_mode=False,
        )

        # ========== Global Offload Strategy Presets ==========

        tests["offload-global"] = TestPreset(
            transfer_mode="strategy",
            strategy_type="global_offload",
            n_blocks=4,
            max_gpu_mem_gb=48.0,
            baseline_mode=False,
        )

        tests["offload-knapsack-block"] = TestPreset(
            transfer_mode="strategy",
            strategy_type="knapsack_block",
            n_blocks=4,
            baseline_mode=False,
        )

        # ========== Global Tensor Selection Strategy Presets ==========

        tests["offload-tensor-selection"] = TestPreset(
            transfer_mode="strategy",
            strategy_type="global_tensor_selection",
            n_blocks=4,
            max_gpu_mem_gb=48.0,
            baseline_mode=False,
        )

        # ========== Offloading Presets - Raw Block Transfer ==========

        tests["offload-raw"] = TestPreset(
            transfer_mode="raw_block_transfer",
            baseline_mode=False,
        )

        tests["offload-raw-block-transfer"] = TestPreset(
            transfer_mode="raw_block_transfer",
            baseline_mode=False,
        )

        # ========== Offloading Presets - Allocation Block Transfer ==========

        tests["offload-allocation-block-transfer"] = TestPreset(
            transfer_mode="allocation_block_transfer",
            baseline_mode=False,
        )

        # ========== Offloading Presets - Strategy ==========

        tests["offload-batch"] = TestPreset(
            transfer_mode="strategy",
            baseline_mode=False,
        )

        tests["offload-batch-transfer"] = TestPreset(
            transfer_mode="strategy",
            baseline_mode=False,
        )

        # ========== Memory Optimization Presets ==========

        tests["memory-optimized"] = TestPreset(
            loader_type="strategy",
            scale=0.5,
            baseline_mode=False,
        )

        # ========== Rearranged Transfer Presets ==========

        tests["rearranged-transfers"] = TestPreset(
            loader_type="strategy",
            rearrange_transfers=True,
            min_compute_transfer_gap=2,
            baseline_mode=False,
        )

        # ========== Adaptive Strategy Preset ==========

        tests["offload-adaptive"] = TestPreset(
            transfer_mode="allocation_block_transfer",
            strategy_type="adaptive",
            n_blocks=4,
            max_gpu_mem_gb=48.0,
            baseline_mode=False,
        )

        # ========== OffloadManager (High-Level API) Presets ==========

        tests["om-baseline"] = TestPreset(
            api_type="high_level",
            baseline_mode=True,
        )

        tests["om-offload-adaptive"] = TestPreset(
            api_type="high_level",
            transfer_mode="allocation_block_transfer",
            strategy_type="adaptive",
            n_blocks=4,
            max_gpu_mem_gb=48.0,
            baseline_mode=False,
        )

        return tests

    def _create_presets(self) -> dict[str, ExperimentConfig]:
        """Create a library of preset configurations."""
        presets = {}

        # ========== Baseline Presets ==========

        presets["baseline-basic-small"] = ExperimentConfig(
            model_type="basic",
            layers=4,
            iterations=5,
            tensor_shape=(8192, 4096),
            baseline_mode=True,
        )

        presets["baseline-basic-medium"] = ExperimentConfig(
            model_type="basic",
            layers=8,
            iterations=10,
            tensor_shape=(14336, 4096),
            baseline_mode=True,
        )

        presets["baseline-basic-large"] = ExperimentConfig(
            model_type="basic",
            layers=16,
            iterations=20,
            tensor_shape=(14336, 4096),
            baseline_mode=True,
        )

        presets["baseline-expert-small"] = ExperimentConfig(
            model_type="expert",
            layers=3,
            iterations=2,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            baseline_mode=True,
        )

        presets["baseline-expert-medium"] = ExperimentConfig(
            model_type="expert",
            layers=4,
            iterations=5,
            dim=4096,
            inter_dim=14336,
            num_experts=8,
            baseline_mode=True,
        )

        presets["baseline-expert-large"] = ExperimentConfig(
            model_type="expert",
            layers=8,
            iterations=10,
            dim=4096,
            inter_dim=14336,
            num_experts=16,
            baseline_mode=True,
        )

        # ========== Offloading Presets - Strategy Loader ==========

        presets["offload-basic-small-strategy"] = ExperimentConfig(
            model_type="basic",
            layers=4,
            iterations=5,
            tensor_shape=(8192, 4096),
            loader_type="strategy",
            baseline_mode=False,
        )

        presets["offload-basic-medium-strategy"] = ExperimentConfig(
            model_type="basic",
            layers=8,
            iterations=10,
            tensor_shape=(14336, 4096),
            loader_type="strategy",
            baseline_mode=False,
        )

        presets["offload-expert-small-strategy"] = ExperimentConfig(
            model_type="expert",
            layers=3,
            iterations=2,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            loader_type="strategy",
            baseline_mode=False,
        )

        presets["offload-expert-medium-strategy"] = ExperimentConfig(
            model_type="expert",
            layers=4,
            iterations=5,
            dim=4096,
            inter_dim=14336,
            num_experts=8,
            loader_type="strategy",
            baseline_mode=False,
        )

        # ========== Offloading Presets - Raw Block Transfer ==========

        presets["offload-basic-small-raw"] = ExperimentConfig(
            model_type="basic",
            layers=4,
            iterations=5,
            tensor_shape=(8192, 4096),
            loader_type="raw_block_transfer",
            baseline_mode=False,
        )

        presets["offload-basic-medium-raw"] = ExperimentConfig(
            model_type="basic",
            layers=8,
            iterations=10,
            tensor_shape=(14336, 4096),
            loader_type="raw_block_transfer",
            baseline_mode=False,
        )

        presets["offload-expert-small-raw"] = ExperimentConfig(
            model_type="expert",
            layers=3,
            iterations=2,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            loader_type="raw_block_transfer",
            baseline_mode=False,
        )

        # ========== Offloading Presets - Strategy ==========

        presets["offload-basic-small-batch"] = ExperimentConfig(
            model_type="basic",
            layers=4,
            iterations=5,
            tensor_shape=(8192, 4096),
            loader_type="strategy",
            baseline_mode=False,
        )

        presets["offload-expert-small-batch"] = ExperimentConfig(
            model_type="expert",
            layers=3,
            iterations=2,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            loader_type="strategy",
            baseline_mode=False,
        )

        # ========== Non-Uniform Presets ==========

        presets["non-uniform-small-baseline"] = ExperimentConfig(
            model_type="non_uniform",
            use_non_uniform=True,
            layers=3,
            iterations=2,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            baseline_mode=True,
        )

        presets["non-uniform-small-strategy"] = ExperimentConfig(
            model_type="non_uniform",
            use_non_uniform=True,
            layers=3,
            iterations=2,
            dim=2048,
            inter_dim=8192,
            num_experts=4,
            loader_type="strategy",
            baseline_mode=False,
        )

        # ========== Memory Optimization Presets ==========

        presets["memory-optimized-basic"] = ExperimentConfig(
            model_type="basic",
            layers=16,
            iterations=10,
            tensor_shape=(14336, 4096),
            loader_type="strategy",
            scale=0.5,  # Reduce memory usage
            baseline_mode=False,
        )

        presets["memory-optimized-expert"] = ExperimentConfig(
            model_type="expert",
            layers=8,
            iterations=5,
            dim=4096,
            inter_dim=14336,
            num_experts=16,
            loader_type="strategy",
            scale=0.6,
            baseline_mode=False,
        )

        # ========== Rearranged Transfer Presets ==========

        presets["rearranged-basic"] = ExperimentConfig(
            model_type="basic",
            layers=8,
            iterations=10,
            tensor_shape=(14336, 4096),
            loader_type="strategy",
            rearrange_transfers=True,
            min_compute_transfer_gap=2,
            baseline_mode=False,
        )

        return presets

    def get_preset(self, name: str) -> ExperimentConfig:
        """
        Get a preset configuration by name.

        Args:
            name: Name of the preset configuration

        Returns:
            ExperimentConfig instance

        Raises:
            KeyError: If preset name not found
        """
        if name not in self._presets:
            available = ", ".join(self._presets.keys())
            msg = f"Preset '{name}' not found. Available presets: {available}"
            raise KeyError(msg)
        return self._presets[name]

    def get_model_preset(self, name: str) -> ModelPreset:
        """
        Get a model preset by name.

        Args:
            name: Name of the model preset

        Returns:
            ModelPreset instance

        Raises:
            KeyError: If model preset name not found
        """
        if name not in self._model_presets:
            available = ", ".join(self._model_presets.keys())
            msg = f"Model preset '{name}' not found. Available model presets: {available}"
            raise KeyError(msg)
        return self._model_presets[name]

    def get_test_preset(self, name: str) -> TestPreset:
        """
        Get a test preset by name.

        Args:
            name: Name of the test preset

        Returns:
            TestPreset instance

        Raises:
            KeyError: If test preset name not found
        """
        if name not in self._test_presets:
            available = ", ".join(self._test_presets.keys())
            msg = f"Test preset '{name}' not found. Available test presets: {available}"
            raise KeyError(msg)
        return self._test_presets[name]

    def compose_presets(self, model_name: str, test_name: str) -> ExperimentConfig:
        """
        Compose a model preset and test preset into an ExperimentConfig.

        Args:
            model_name: Name of the model preset
            test_name: Name of the test preset

        Returns:
            ExperimentConfig instance

        Example:
            config = manager.compose_presets("basic-small", "offload-strategy-quick")
        """
        model = self.get_model_preset(model_name)
        test = self.get_test_preset(test_name)
        return ExperimentConfig.from_presets(model, test)

    def list_presets(self, filter_by: str | None = None) -> list[str]:
        """
        List available preset configurations.

        Args:
            filter_by: Optional string to filter preset names (e.g., "baseline", "expert")

        Returns:
            List of preset names
        """
        presets = list(self._presets.keys())
        if filter_by:
            presets = [p for p in presets if filter_by.lower() in p.lower()]
        return sorted(presets)

    def list_model_presets(self, filter_by: str | None = None) -> list[str]:
        """
        List available model presets.

        Args:
            filter_by: Optional string to filter preset names (e.g., "basic", "expert")

        Returns:
            List of model preset names
        """
        presets = list(self._model_presets.keys())
        if filter_by:
            presets = [p for p in presets if filter_by.lower() in p.lower()]
        return sorted(presets)

    def list_test_presets(self, filter_by: str | None = None) -> list[str]:
        """
        List available test presets.

        Args:
            filter_by: Optional string to filter preset names (e.g., "baseline", "offload", "strategy")

        Returns:
            List of test preset names
        """
        presets = list(self._test_presets.keys())
        if filter_by:
            presets = [p for p in presets if filter_by.lower() in p.lower()]
        return sorted(presets)

    def save_config(self, config: ExperimentConfig, name: str, overwrite: bool = False) -> Path:
        """
        Save a configuration to a JSON file.

        Args:
            config: ExperimentConfig to save
            name: Name for the config file (without extension)
            overwrite: Whether to overwrite existing file

        Returns:
            Path to the saved config file

        Raises:
            FileExistsError: If file exists and overwrite=False
        """
        filepath = self.config_dir / f"{name}.json"

        if filepath.exists() and not overwrite:
            msg = f"Config file already exists: {filepath}"
            raise FileExistsError(msg)

        # Convert config to dict
        config_dict = config.model_dump()

        # Handle torch.dtype serialization
        if "tensor_dtype" in config_dict:
            config_dict["tensor_dtype"] = str(config_dict["tensor_dtype"])

        # Save to JSON
        with filepath.open("w") as f:
            json.dump(config_dict, f, indent=2)

        return filepath

    def load_config(self, name: str) -> ExperimentConfig:
        """
        Load a configuration from a JSON file.

        Args:
            name: Name of the config file (with or without .json extension)

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: If config file not found
        """
        # Handle with or without .json extension
        if not name.endswith(".json"):
            name = f"{name}.json"

        filepath = self.config_dir / name

        if not filepath.exists():
            msg = f"Config file not found: {filepath}"
            raise FileNotFoundError(msg)

        # Load from JSON
        with filepath.open("r") as f:
            config_dict = json.load(f)

        # Handle torch.dtype deserialization
        if "tensor_dtype" in config_dict:
            dtype_str = config_dict["tensor_dtype"]
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            for key, dtype in dtype_map.items():
                if key in dtype_str:
                    config_dict["tensor_dtype"] = dtype
                    break

        # Convert tuple fields
        if "tensor_shape" in config_dict and isinstance(config_dict["tensor_shape"], list):
            config_dict["tensor_shape"] = tuple(config_dict["tensor_shape"])

        return ExperimentConfig(**config_dict)

    def list_saved_configs(self) -> list[str]:
        """
        List all saved configuration files.

        Returns:
            List of config file names (without .json extension)
        """
        configs = []
        for filepath in self.config_dir.glob("*.json"):
            configs.append(filepath.stem)
        return sorted(configs)

    def export_preset_configs(self, preset_names: list[str] | None = None):
        """
        Export preset configurations to JSON files.

        Args:
            preset_names: List of preset names to export. If None, exports all.
        """
        if preset_names is None:
            preset_names = list(self._presets.keys())

        for name in preset_names:
            if name in self._presets:
                config = self._presets[name]
                self.save_config(config, name, overwrite=True)
                print(f"✓ Exported preset: {name}")

    def create_config_suite(self, suite_name: str, config_names: list[str]) -> Path:
        """
        Create a test suite file with multiple configurations.

        Args:
            suite_name: Name for the suite file
            config_names: List of config names (presets or saved configs)

        Returns:
            Path to the suite file
        """
        suite_filepath = self.config_dir / f"suite_{suite_name}.json"

        suite_data = {
            "suite_name": suite_name,
            "configs": config_names,
        }

        with suite_filepath.open("w") as f:
            json.dump(suite_data, f, indent=2)

        return suite_filepath

    def load_config_suite(self, suite_name: str) -> list[ExperimentConfig]:
        """
        Load all configurations from a test suite.

        Args:
            suite_name: Name of the suite file (with or without suite_ prefix and .json)

        Returns:
            List of ExperimentConfig instances
        """
        # Handle suite file naming
        if not suite_name.startswith("suite_"):
            suite_name = f"suite_{suite_name}"
        if not suite_name.endswith(".json"):
            suite_name = f"{suite_name}.json"

        suite_filepath = self.config_dir / suite_name

        if not suite_filepath.exists():
            msg = f"Suite file not found: {suite_filepath}"
            raise FileNotFoundError(msg)

        with suite_filepath.open("r") as f:
            suite_data = json.load(f)

        configs = []
        for config_name in suite_data["configs"]:
            # Try to load as preset first, then as saved config
            try:
                config = self.get_preset(config_name)
            except KeyError:
                config = self.load_config(config_name)
            configs.append(config)

        return configs

    def print_config_summary(self, config: ExperimentConfig, name: str = "Config"):
        """Print a human-readable summary of a configuration."""
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print(f"{'=' * 60}")
        print(f"Model Type:      {config.model_type}")
        print(f"Layers:          {config.layers}")
        print(f"Iterations:      {config.iterations}")
        print(f"Baseline Mode:   {config.baseline_mode}")

        if config.model_type == "basic":
            print(f"Tensor Shape:    {config.tensor_shape}")
        else:
            print(f"Dimensions:      {config.dim} -> {config.inter_dim}")
            print(f"Num Experts:     {config.num_experts}")
            print(f"Batch Size:      {config.batch_size}")
            print(f"Seq Length:      {config.seq_len}")

        if not config.baseline_mode:
            print(f"Transfer Mode:   {config.transfer_mode}")
            print(f"Pinned Memory:   {config.pinned_memory}")
            print(f"Budget Scale:  {config.transfer_budget_scale}")
            print(f"Rearrange:       {config.rearrange_transfers}")

        print(f"{'=' * 60}\n")

    def print_model_preset_summary(self, model: ModelPreset, name: str = "Model Preset"):
        """Print a human-readable summary of a model preset."""
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print(f"{'=' * 60}")
        print(f"Model Type:      {model.model_type}")
        print(f"Layers:          {model.layers}")
        print(f"Iterations:      {model.iterations}")
        print(f"Tensor Dtype:    {model.tensor_dtype}")

        if model.model_type == "basic":
            print(f"Tensor Shape:    {model.tensor_shape}")
        else:
            print(f"Dimensions:      {model.dim} -> {model.inter_dim}")
            print(f"Num Experts:     {model.num_experts}")
            print(f"Batch Size:      {model.batch_size}")
            print(f"Seq Length:      {model.seq_len}")
            if model.use_non_uniform:
                print("Non-Uniform:     Yes")
                if model.num_experts_list:
                    print(f"Experts List:    {model.num_experts_list}")

        print(f"{'=' * 60}\n")

    def print_test_preset_summary(self, test: TestPreset, name: str = "Test Preset"):
        """Print a human-readable summary of a test preset."""
        print(f"\n{'=' * 60}")
        print(f"{name}")
        print(f"{'=' * 60}")
        print(f"Baseline Mode:   {test.baseline_mode}")

        if not test.baseline_mode:
            print(f"Transfer Mode:   {test.transfer_mode}")
            print(f"Pinned Memory:   {test.pinned_memory}")
            print(f"Budget Scale:  {test.transfer_budget_scale}")
            if test.rearrange_transfers:
                print("Rearrange:       Yes")
                print(f"Transfer Gap:    {test.compute_transfer_gap}")

        print(f"{'=' * 60}\n")
