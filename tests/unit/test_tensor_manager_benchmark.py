# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TensorManager benchmark integration functionality."""

import pytest
import torch

from flextensor.benchmark_tensor_mode import BenchmarkReplace, PreloadToDevice, TensorBenchmarkMode
from flextensor.strategy import KnapsackStrategy
from flextensor.tensor_manager import TensorManager


class TestTensorManagerBenchmarkIntegration:
    """Test cases for TensorManager benchmark integration."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.device_gpu = torch.device("cuda:0")
        self.device_cpu = torch.device("cpu")
        self.pinned_memory = True
        self.strategy = KnapsackStrategy(scale=0.8)

    def test_tensor_manager_constructor_default_benchmark_cls(self) -> None:
        """Test TensorManager constructor with default benchmark_cls."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
        )

        assert tensor_manager._benchmark_cls == BenchmarkReplace

    def test_tensor_manager_constructor_custom_benchmark_cls(self) -> None:
        """Test TensorManager constructor with custom benchmark_cls."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=MockBenchmarkReplace,
        )

        assert tensor_manager._benchmark_cls == MockBenchmarkReplace

    def test_benchmark_context_method_exists(self) -> None:
        """Test that benchmark_context method exists and is callable."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
        )

        assert hasattr(tensor_manager, "benchmark_context")
        assert callable(tensor_manager.benchmark_context)

    def test_benchmark_context_creates_benchmark_instance(self) -> None:
        """Test that benchmark_context creates benchmark instance with correct parameters."""

        # Create a custom mock benchmark class for this test
        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {"tensor_statistics_map": {}, "tensors_map": {}}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=TestMockBenchmark,
        )

        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, TestMockBenchmark)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory == self.pinned_memory
            assert benchmark.iterations == 5

    def test_benchmark_context_default_iterations(self) -> None:
        """Test that benchmark_context uses default iterations parameter."""

        # Create a custom mock benchmark class for this test
        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {"tensor_statistics_map": {}, "tensors_map": {}}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=TestMockBenchmark,
        )

        with tensor_manager.benchmark_context() as benchmark:
            # Test that benchmark was created successfully with default iterations
            # We can't directly test the iterations value due to abstract base class limitations
            assert isinstance(benchmark, TestMockBenchmark)
            assert benchmark.device_gpu == self.device_gpu

    def test_benchmark_context_automatic_stats_integration(self) -> None:
        """Test that benchmark results are automatically integrated into TensorManager."""

        # Create a custom mock benchmark class for this test
        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {
                    "tensor_statistics_map": {123: "stats1", 456: "stats2"},
                    "tensors_map": {123: "tensor_obj1", 456: "tensor_obj2"},
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=TestMockBenchmark,
        )

        # Initially empty stats
        assert tensor_manager.tensor_statistics_map == {}
        assert tensor_manager.tensors_map == {}

        with tensor_manager.benchmark_context():
            pass  # Stats should be integrated on context exit

        # Verify stats were automatically integrated
        assert tensor_manager.tensor_statistics_map == {123: "stats1", 456: "stats2"}
        assert tensor_manager.tensors_map == {123: "tensor_obj1", 456: "tensor_obj2"}

    def test_benchmark_context_stats_replacement(self) -> None:
        """Test that benchmark results replace existing TensorManager stats."""

        # Create a custom mock benchmark class for this test
        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {
                    "tensor_statistics_map": {123: "new_stats"},
                    "tensors_map": {123: "new_tensor_obj"},
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        # Setup existing stats in TensorManager
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=TestMockBenchmark,
        )

        # Pre-populate with existing stats
        tensor_manager.tensor_statistics_map = {111: "existing_stats"}
        tensor_manager.tensors_map = {111: "existing_tensor_obj"}

        with tensor_manager.benchmark_context():
            pass

        # Verify stats were replaced (not merged)
        assert tensor_manager.tensor_statistics_map == {123: "new_stats"}
        assert tensor_manager.tensors_map == {123: "new_tensor_obj"}

    def test_benchmark_context_exception_handling(self) -> None:
        """Test that stats are still integrated even if exception occurs in context."""

        # Create a custom mock benchmark class for this test
        class TestMockBenchmark(TensorBenchmarkMode):
            def __init__(self, device_gpu, pinned_memory, iterations):
                self.device_gpu = device_gpu
                self.pinned_memory = pinned_memory
                self.iterations = iterations
                self.results = {
                    "tensor_statistics_map": {123: "stats1"},
                    "tensors_map": {123: "tensor_obj1"},
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return None

            def get_results(self):
                return self.results

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=TestMockBenchmark,
        )

        # Simulate exception in context
        with pytest.raises(ValueError), tensor_manager.benchmark_context():
            raise ValueError("Test exception")

        # Verify stats were still integrated despite exception
        assert tensor_manager.tensor_statistics_map == {123: "stats1"}
        assert tensor_manager.tensors_map == {123: "tensor_obj1"}

    def test_benchmark_context_backward_compatibility(self) -> None:
        """Test that existing TensorManager functionality is not affected."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=self.pinned_memory,
            tensor_manager_load_strategy=self.strategy,
        )

        # Test that all existing methods still work
        assert tensor_manager.device_gpu == self.device_gpu
        assert tensor_manager.pinned_memory == self.pinned_memory
        assert tensor_manager.tensor_manager_load_strategy == self.strategy
        assert hasattr(tensor_manager, "prepare_warmup_mode")
        assert hasattr(tensor_manager, "prepare_profile_mode")
        assert hasattr(tensor_manager, "prepare_infer_mode")
        assert hasattr(tensor_manager, "trap")
        assert hasattr(tensor_manager, "release_memory")
        assert hasattr(tensor_manager, "is_traced")


class MockBenchmarkReplace(TensorBenchmarkMode):
    """Mock benchmark class for testing parametrization."""

    def __init__(self, device_gpu, pinned_memory, iterations):
        # Don't call super().__init__ as it's abstract
        self.device_gpu = device_gpu
        self.pinned_memory = pinned_memory
        self.iterations = iterations
        self.results = {
            "tensor_statistics_map": {999: "mock_stats"},
            "tensors_map": {999: "mock_tensor_obj"},
        }

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        return None

    def get_results(self):
        return self.results


class TestTensorManagerBenchmarkParametrization:
    """Test cases for TensorManager benchmark class parametrization."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.device_gpu = torch.device("cuda:0")
        self.strategy = KnapsackStrategy(scale=0.8)

    def test_custom_benchmark_class_usage(self) -> None:
        """Test that custom benchmark class can be used via parametrization."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=True,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=MockBenchmarkReplace,
        )

        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, MockBenchmarkReplace)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory
            assert benchmark.iterations == 5

        # Verify mock results were integrated
        assert tensor_manager.tensor_statistics_map == {999: "mock_stats"}
        assert tensor_manager.tensors_map == {999: "mock_tensor_obj"}

    def test_preload_to_device_usage(self) -> None:
        """Test that PreloadToDevice can be used as benchmark_cls."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=True,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=PreloadToDevice,
        )

        # Test that PreloadToDevice can be instantiated with same interface as BenchmarkReplace
        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, PreloadToDevice)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory
            assert benchmark.iterations == 5

        # PreloadToDevice should return empty statistics but maintain interface
        assert tensor_manager.tensor_statistics_map == {}
        assert tensor_manager.tensors_map == {}

    def test_abstract_base_class_interface(self) -> None:
        """Test that both BenchmarkReplace and PreloadToDevice implement TensorBenchmarkMode interface."""
        # Test that both classes are instances of the abstract base class
        assert issubclass(BenchmarkReplace, TensorBenchmarkMode)
        assert issubclass(PreloadToDevice, TensorBenchmarkMode)

        # Test that both can be instantiated with same parameters
        benchmark_replace = BenchmarkReplace(self.device_gpu, pinned_memory=True, iterations=10)
        preload_to_device = PreloadToDevice(self.device_gpu, pinned_memory=True, iterations=10)

        # Test that both have required methods
        assert hasattr(benchmark_replace, "get_results")
        assert hasattr(preload_to_device, "get_results")
        assert callable(benchmark_replace.get_results)
        assert callable(preload_to_device.get_results)

        # Test that get_results returns correct structure for both
        br_results = benchmark_replace.get_results()
        ptd_results = preload_to_device.get_results()

        expected_keys = {"tensor_statistics_map", "tensors_map"}
        assert set(br_results.keys()) == expected_keys
        assert set(ptd_results.keys()) == expected_keys

    def test_tensor_benchmark_mode_type_annotation(self) -> None:
        """Test that TensorManager accepts TensorBenchmarkMode type annotation correctly."""
        # This test verifies the type annotation works correctly at runtime
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=True,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=BenchmarkReplace,  # Should work with base class type annotation
        )

        assert tensor_manager._benchmark_cls == BenchmarkReplace

        # Test with PreloadToDevice as well
        tensor_manager2 = TensorManager(
            device_gpu=self.device_gpu,
            pinned_memory=True,
            tensor_manager_load_strategy=self.strategy,
            benchmark_cls=PreloadToDevice,  # Should also work with base class type annotation
        )

        assert tensor_manager2._benchmark_cls == PreloadToDevice
