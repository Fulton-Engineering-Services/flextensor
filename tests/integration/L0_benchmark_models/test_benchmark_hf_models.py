# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for HuggingFace model benchmarking with bermuda TensorManager."""

import psutil
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flextensor import TensorManager
from flextensor.benchmark_tensor_mode import BenchmarkReplace
from flextensor.strategy import KnapsackStrategy


class TestHuggingFaceModelBenchmarking:
    """Test cases for benchmarking HuggingFace models with TensorManager."""

    @pytest.fixture(scope="class")
    def device_gpu(self):
        """Fixture to provide GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda:0")

    def create_tensor_manager(self, device_gpu):
        """Create a fresh TensorManager instance."""
        strategy = KnapsackStrategy(scale=0.8)
        return TensorManager(
            device_gpu=device_gpu,
            pinned_memory=True,
            tensor_manager_load_strategy=strategy,
            benchmark_cls=BenchmarkReplace,
        )

    def load_hf_model(self, model_name: str):
        """
        Load HuggingFace model with error handling.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Tuple of (model, tokenizer, model_size_mb)
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
        assert model.device.type == "cpu", "Model should be loaded on CPU"
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        return model, tokenizer, model_size_mb

    @pytest.mark.parametrize(
        "model_name",
        [
            "katuni4ka/tiny-random-deepseek-v3",  # Smallest, fastest to download, MoE
            "TinyLlama/TinyLlama_v1.1",  # Small real model
            "Qwen/Qwen3-0.6B",  # Medium size
        ],
    )
    def test_model_benchmarking(self, model_name: str, device_gpu: torch.device):
        """
        Test benchmarking functionality with HuggingFace models.

        Args:
            model_name: HuggingFace model identifier
            device_gpu: GPU device for testing
        """
        tensor_manager = self.create_tensor_manager(device_gpu)

        # First load the model normally (without benchmark context)
        # as benchmarking this line causes appearance of meta tensors and other non-standard objects
        model, tokenizer, model_size_mb = self.load_hf_model(model_name)

        # Benchmark moving the model to GPU
        with tensor_manager.benchmark_context(iterations=5):
            model = model.to("cpu")

        assert len(tensor_manager.tensor_statistics_map) > 0, (
            f"No tensor statistics collected during GPU transfer for {model_name}"
        )
        assert len(tensor_manager.tensors_map) > 0, f"No tensors mapped during GPU transfer for {model_name}"

        # Validate tensor statistics
        for tensor_id, stats in tensor_manager.tensor_statistics_map.items():
            assert stats.tensor_id == tensor_id, f"Tensor ID mismatch for {model_name}"
            assert stats.size_bytes > 0, f"Invalid tensor size for {model_name}: {stats.size_bytes}"
            assert stats.load_time_ms >= 0, f"Invalid load time for {model_name}: {stats.load_time_ms}"

        # Check that tensors are properly mapped
        for tensor_id, tensor in tensor_manager.tensors_map.items():
            assert tensor_id in tensor_manager.tensor_statistics_map, f"Unmapped tensor statistics for {model_name}"
            assert hasattr(tensor, "size"), f"Invalid tensor object for {model_name}"

        # Test basic model functionality without benchmarking
        test_prompt = "The capital of France is"
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device_gpu)

        model = model.to(device_gpu)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits

        # Check output validity
        assert logits is not None, f"No logits generated for {model_name}"
        assert logits.shape[0] == input_ids.shape[0], f"Batch size mismatch for {model_name}"
        assert logits.shape[-1] > 0, f"Invalid vocabulary size for {model_name}"

        total_tensors = len(tensor_manager.tensor_statistics_map)
        total_size_bytes = sum(stats.size_bytes for stats in tensor_manager.tensor_statistics_map.values())
        avg_load_time = (
            sum(stats.load_time_ms for stats in tensor_manager.tensor_statistics_map.values()) / total_tensors
        )

        print(f"\n{model_name} Model GPU Transfer Benchmark Results:")
        print(f"  Model size: {model_size_mb:.2f} MB")
        print(f"  Tensors tracked during GPU transfer: {total_tensors}")
        print(f"  Total tracked size: {total_size_bytes / 1024 / 1024:.2f} MB")
        print(f"  Average transfer time: {avg_load_time:.2f} ms")
        print(f"  Output shape: {logits.shape}")

    def test_benchmark_cleanup(self, device_gpu: torch.device):
        """
        Test that benchmarking properly cleans up resources.

        Args:
            device_gpu: GPU device for testing
        """
        model_name = "katuni4ka/tiny-random-deepseek-v3"

        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated(device_gpu)
        initial_cpu_memory = psutil.Process().memory_info().rss

        model, tokenizer, _ = self.load_hf_model(model_name)

        tensor_manager = self.create_tensor_manager(device_gpu)
        with tensor_manager.benchmark_context(iterations=2):
            model = model.to("cpu")

        test_prompt = "Cleanup test"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device_gpu)

        model = model.to(device_gpu)
        with torch.no_grad():
            outputs = model(input_ids)

        del model
        del outputs
        del input_ids
        torch.cuda.empty_cache()

        final_gpu_memory = torch.cuda.memory_allocated(device_gpu)
        final_cpu_memory = psutil.Process().memory_info().rss
        gpu_memory_diff = final_gpu_memory - initial_gpu_memory
        cpu_memory_diff = final_cpu_memory - initial_cpu_memory

        print(f"\nMemory usage for {model_name}:")
        print(f"  Initial GPU: {initial_gpu_memory / 1024 / 1024:.2f} MB")
        print(f"  Final GPU: {final_gpu_memory / 1024 / 1024:.2f} MB")
        print(f"  GPU Difference: {gpu_memory_diff / 1024 / 1024:.2f} MB")
        print(f"  Initial CPU: {initial_cpu_memory / 1024 / 1024:.2f} MB")
        print(f"  Final CPU: {final_cpu_memory / 1024 / 1024:.2f} MB")
        print(f"  CPU Difference: {cpu_memory_diff / 1024 / 1024:.2f} MB")

        assert gpu_memory_diff < 8 * 1024 * 1024, f"Excessive GPU memory usage: {gpu_memory_diff / 1024 / 1024:.2f} MB"
        assert cpu_memory_diff < 50 * 1024 * 1024, f"Excessive CPU memory usage: {cpu_memory_diff / 1024 / 1024:.2f} MB"
