# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU memory usage API.

This test suite validates the GPUMemoryUsage dataclass and the get_gpu_memory_usage()
method in OffloadManager. Tests use mocking to avoid GPU dependencies.

Key behaviors tested:
- GPUMemoryUsage dataclass properties (bytes and MB conversions)
- OffloadManager.get_gpu_memory_usage() delegation to TensorManager
- Error handling when called in wrong state (before inference mode)
- Module-level convenience function get_gpu_memory_usage()
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.offload_manager import OffloadConfig, OffloadManager, OffloadPhase
from flextensor.types import GPUMemoryUsage

# Note: TensorStatistics and LayerStatistics imports are done locally
# in tests that need them to avoid circular import issues


class TestGPUMemoryUsageDataclass:
    """Test cases for GPUMemoryUsage dataclass."""

    def test_dataclass_initialization(self):
        """Test that GPUMemoryUsage can be initialized with required fields."""
        usage = GPUMemoryUsage(
            blocks_bytes=1024 * 1024 * 100,  # 100 MB
            unmapped_tensors_bytes=1024 * 1024 * 50,  # 50 MB
            total_bytes=1024 * 1024 * 150,  # 150 MB
        )

        assert usage.blocks_bytes == 1024 * 1024 * 100
        assert usage.unmapped_tensors_bytes == 1024 * 1024 * 50
        assert usage.total_bytes == 1024 * 1024 * 150

    def test_blocks_mb_property(self):
        """Test blocks_mb property converts bytes to megabytes correctly."""
        usage = GPUMemoryUsage(
            blocks_bytes=1024 * 1024 * 100,
            unmapped_tensors_bytes=0,
            total_bytes=1024 * 1024 * 100,
        )

        assert usage.blocks_mb == 100.0

    def test_unmapped_tensors_mb_property(self):
        """Test unmapped_tensors_mb property converts bytes to megabytes correctly."""
        usage = GPUMemoryUsage(
            blocks_bytes=0,
            unmapped_tensors_bytes=1024 * 1024 * 50,
            total_bytes=1024 * 1024 * 50,
        )

        assert usage.unmapped_tensors_mb == 50.0

    def test_total_mb_property(self):
        """Test total_mb property converts bytes to megabytes correctly."""
        usage = GPUMemoryUsage(
            blocks_bytes=1024 * 1024 * 100,
            unmapped_tensors_bytes=1024 * 1024 * 50,
            total_bytes=1024 * 1024 * 150,
        )

        assert usage.total_mb == 150.0

    def test_fractional_mb_values(self):
        """Test MB properties with non-round byte values."""
        # 1.5 MB = 1572864 bytes
        usage = GPUMemoryUsage(
            blocks_bytes=1572864,
            unmapped_tensors_bytes=524288,  # 0.5 MB
            total_bytes=2097152,  # 2 MB
        )

        assert usage.blocks_mb == 1.5
        assert usage.unmapped_tensors_mb == 0.5
        assert usage.total_mb == 2.0

    def test_zero_memory_usage(self):
        """Test GPUMemoryUsage with zero values."""
        usage = GPUMemoryUsage(
            blocks_bytes=0,
            unmapped_tensors_bytes=0,
            total_bytes=0,
        )

        assert usage.blocks_bytes == 0
        assert usage.unmapped_tensors_bytes == 0
        assert usage.total_bytes == 0
        assert usage.blocks_mb == 0.0
        assert usage.unmapped_tensors_mb == 0.0
        assert usage.total_mb == 0.0


class MockTrap:
    """Mock trap context manager for testing."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestOffloadManagerGPUMemoryUsage:
    """Test cases for OffloadManager.get_gpu_memory_usage()."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()
        self.x = torch.randn(4, 10)

    @pytest.mark.parametrize("skip_discovery", [False, True])
    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_get_gpu_memory_usage_after_driving_iters_before_inference(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
        skip_discovery,
    ):
        """Driving ``iters_before_inference`` forwards must reach INFERENCE.

        ``skip_discovery`` changes when INFERENCE is reached (the discovery
        component drops out), and ``get_gpu_memory_usage()`` raises outside
        INFERENCE — so this pins the two together on both paths. Uses the
        path-aware ``om.iters_before_inference``.
        """
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        expected_usage = GPUMemoryUsage(
            blocks_bytes=1024,
            unmapped_tensors_bytes=512,
            total_bytes=1536,
        )
        mock_tensor_manager.get_gpu_memory_usage.return_value = expected_usage

        om = OffloadManager(f"test_memory_skip_{skip_discovery}")
        config = OffloadConfig(
            enabled=True,
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["linear"],
            skip_discovery=skip_discovery,
        )
        model = om.offload(self.model, config=config)

        for _ in range(om.iters_before_inference):
            with torch.no_grad():
                _ = model(self.x)

        assert om._current_phase == OffloadPhase.INFERENCE
        assert om.get_gpu_memory_usage().total_bytes == expected_usage.total_bytes

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_get_gpu_memory_usage_in_inference_state(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test get_gpu_memory_usage() returns correct values in inference state."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        # Mock the get_gpu_memory_usage method
        expected_usage = GPUMemoryUsage(
            blocks_bytes=1024 * 1024 * 100,
            unmapped_tensors_bytes=1024 * 1024 * 50,
            total_bytes=1024 * 1024 * 150,
        )
        mock_tensor_manager.get_gpu_memory_usage.return_value = expected_usage

        # Create offload manager and transition to inference
        om = OffloadManager("test_memory")
        config = OffloadConfig(
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["linear"], skip_discovery=False
        )
        model = om.offload(self.model, config=config)

        # Run through discovery and profiling to reach inference
        for _ in range(om.iters_before_inference):
            with torch.no_grad():
                _ = model(self.x)

        # Verify we're in inference phase
        assert om._current_phase == OffloadPhase.INFERENCE

        # Get memory usage
        usage = om.get_gpu_memory_usage()

        # Verify the result
        assert usage.blocks_bytes == expected_usage.blocks_bytes
        assert usage.unmapped_tensors_bytes == expected_usage.unmapped_tensors_bytes
        assert usage.total_bytes == expected_usage.total_bytes
        mock_tensor_manager.get_gpu_memory_usage.assert_called_once()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_get_gpu_memory_usage_raises_in_discovery_phase(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test get_gpu_memory_usage() raises RuntimeError in discovery phase."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model

        # Create offload manager (stays in discovery)
        om = OffloadManager("test_warmup_error")
        config = OffloadConfig(
            enabled=True, discovery_iters=5, profiling_iters=5, include_patterns=["linear"], skip_discovery=False
        )
        om.offload(self.model, config=config)

        # Verify we're in discovery phase
        assert om._current_phase == OffloadPhase.DISCOVERY

        # Attempt to get memory usage should raise
        with pytest.raises(RuntimeError, match="Cannot get GPU memory usage in phase discovery"):
            om.get_gpu_memory_usage()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_get_gpu_memory_usage_raises_in_profiling_phase(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test get_gpu_memory_usage() raises RuntimeError in profiling phase."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model

        # Create offload manager and transition to profiling
        om = OffloadManager("test_profile_error")
        config = OffloadConfig(
            enabled=True, discovery_iters=1, profiling_iters=5, include_patterns=["linear"], skip_discovery=False
        )
        model = om.offload(self.model, config=config)

        # Run discovery to transition to profiling
        with torch.no_grad():
            _ = model(self.x)

        # Verify we're in profiling phase
        assert om._current_phase == OffloadPhase.PROFILING

        # Attempt to get memory usage should raise
        with pytest.raises(RuntimeError, match="Cannot get GPU memory usage in phase profiling"):
            om.get_gpu_memory_usage()

    def test_get_gpu_memory_usage_raises_in_not_initialized_state(self):
        """Test get_gpu_memory_usage() raises RuntimeError when not initialized."""
        om = OffloadManager("test_not_init_error")

        # Verify we're in not_initialized state
        assert om._current_phase == OffloadPhase.NOT_INITIALIZED

        # Attempt to get memory usage should raise
        with pytest.raises(RuntimeError, match="Cannot get GPU memory usage in phase not_initialized"):
            om.get_gpu_memory_usage()


class TestModuleLevelGetGPUMemoryUsage:
    """Test cases for module-level get_gpu_memory_usage() convenience function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.model = SimpleModel()
        self.model.cpu()
        self.model.eval()
        self.x = torch.randn(4, 10)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_module_level_get_gpu_memory_usage(
        self,
        _mock_strategy_cls,
        mock_tensor_manager_cls,
    ):
        """Test module-level get_gpu_memory_usage() function."""
        import flextensor

        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = lambda name, args=(), kwargs=None: MockTrap(name)
        mock_tensor_manager.initialize_warmup.return_value = self.model
        mock_tensor_manager.initialize_profile.return_value = self.model
        mock_tensor_manager.initialize_inference.return_value = self.model

        # Mock the get_gpu_memory_usage method
        expected_usage = GPUMemoryUsage(
            blocks_bytes=1024 * 1024 * 200,
            unmapped_tensors_bytes=1024 * 1024 * 100,
            total_bytes=1024 * 1024 * 300,
        )
        mock_tensor_manager.get_gpu_memory_usage.return_value = expected_usage

        # Use the simplified API with default manager
        config = flextensor.OffloadConfig(
            enabled=True, discovery_iters=1, profiling_iters=1, include_patterns=["linear"], skip_discovery=False
        )
        model = flextensor.offload(self.model, config=config)
        om = flextensor.get_offload_manager()

        # Run through discovery and profiling to reach inference
        for _ in range(om.iters_before_inference):
            with torch.no_grad():
                _ = model(self.x)

        # Get memory usage via module-level function (uses default manager)
        usage = flextensor.get_gpu_memory_usage()

        # Verify result
        assert usage.total_mb == 300.0
        assert usage.blocks_mb == 200.0
        assert usage.unmapped_tensors_mb == 100.0


def _create_mock_cuda_stream():
    """Create a mock CUDA stream with all required methods."""
    mock_stream = MagicMock()
    mock_stream.synchronize = MagicMock()
    mock_stream.wait_event = MagicMock()
    mock_stream.wait_stream = MagicMock()
    mock_stream.record_event = MagicMock(return_value=MagicMock())
    return mock_stream


def _create_mock_cuda_event():
    """Create a mock CUDA event with all required methods."""
    mock_event = MagicMock()
    mock_event.synchronize = MagicMock()
    mock_event.record = MagicMock()
    mock_event.query = MagicMock(return_value=True)
    return mock_event


class TestTensorStrategyLoaderGPUMemory:
    """Test cases for TensorStrategyLoader.get_gpu_memory_bytes().

    Note: get_gpu_memory_bytes() now returns PEAK memory (maximum memory at any
    point during execution) rather than current memory. This is calculated by
    simulating the sliding window execution pattern.
    """

    def test_strategy_loader_peak_memory_calculation(self):
        """Test that TensorStrategyLoader correctly calculates peak GPU memory.

        This test doesn't require GPU - peak memory is calculated from the strategy
        maps without actually allocating GPU memory.
        """
        from flextensor.collectors import LayerStatistics, TensorStatistics
        from flextensor.loaders import TensorStrategyLoader

        # Create test tensors on CPU (pinned for the loader)
        tensors_map = {
            1: torch.randn(100, 100),  # 40KB (100*100*4 bytes)
            2: torch.randn(200, 200),  # 160KB (200*200*4 bytes)
            3: torch.randn(50, 50),  # 10KB (50*50*4 bytes)
        }

        # Create TensorStatistics with size_bytes
        tensor_info_1 = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=100 * 100 * 4,  # 40KB
            load_time_ms=0.1,
        )
        tensor_info_2 = TensorStatistics(
            tensor_id=2,
            name="tensor_2",
            size_bytes=200 * 200 * 4,  # 160KB
            load_time_ms=0.1,
        )
        tensor_info_3 = TensorStatistics(
            tensor_id=3,
            name="tensor_3",
            size_bytes=50 * 50 * 4,  # 10KB
            load_time_ms=0.1,
        )

        # Strategy: load tensor_1,2 at layer1, tensor_3 at layer2
        # Release: tensor_1,2 at layer2, tensor_3 at layer3
        # Peak occurs at layer2 when all 3 tensors are loaded
        strategy_map = {
            "layer1": [tensor_info_1, tensor_info_2],
            "layer2": [tensor_info_3],
        }
        release_strategy_map = {
            "layer2": [tensor_info_1, tensor_info_2],
            "layer3": [tensor_info_3],
        }

        layer_stats = [
            LayerStatistics(label="layer1", tensors=[tensor_info_1, tensor_info_2], duration=1.0),
            LayerStatistics(label="layer2", tensors=[tensor_info_3], duration=1.0),
            LayerStatistics(label="layer3", tensors=[], duration=1.0),
        ]

        # Use mock device to avoid GPU requirement
        device_gpu = MagicMock()
        device_gpu.type = "cuda"

        # Mock all CUDA operations to avoid needing actual GPU
        mock_stream = _create_mock_cuda_stream()
        mock_current_stream = _create_mock_cuda_stream()
        with (
            patch.object(torch.cuda, "Stream", return_value=mock_stream),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "Event", side_effect=_create_mock_cuda_event),
            patch.object(torch.cuda, "stream"),
            patch.object(torch.cuda, "current_stream", return_value=mock_current_stream),
        ):
            loader = TensorStrategyLoader(
                layer_stats=layer_stats,
                strategy_map=strategy_map,
                release_strategy_map=release_strategy_map,
                tensors_map=tensors_map,
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
            )

        # Get peak GPU memory
        peak_bytes = loader.get_gpu_memory_bytes()

        # Peak is at layer2: tensor_1 (40KB) + tensor_2 (160KB) + tensor_3 (10KB) = 210KB
        # But tensors 1,2 are released at layer2 after tensor_3 is loaded
        # So peak is all 3 tensors: 40KB + 160KB + 10KB = 210KB
        expected_peak = (100 * 100 * 4) + (200 * 200 * 4) + (50 * 50 * 4)

        assert peak_bytes == expected_peak
        assert peak_bytes > 0

    def test_strategy_loader_peak_memory_sliding_window_no_overlap(self):
        """Test peak memory with sliding window pattern (no overlap).

        Pattern: load and release in same layer (release happens at end of layer after load)
        - layer1: load tensor_1 (40KB), release tensor_1
        - layer2: load tensor_2 (160KB), release tensor_2

        Peak should be 160KB (largest single tensor, they don't overlap since
        tensor_1 is released before tensor_2 is loaded).
        """
        from flextensor.collectors import LayerStatistics, TensorStatistics
        from flextensor.loaders import TensorStrategyLoader

        tensors_map = {
            1: torch.randn(100, 100),
            2: torch.randn(200, 200),
        }

        tensor_info_1 = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=100 * 100 * 4,
            load_time_ms=0.1,
        )
        tensor_info_2 = TensorStatistics(
            tensor_id=2,
            name="tensor_2",
            size_bytes=200 * 200 * 4,
            load_time_ms=0.1,
        )

        # Sequential pattern: load and release in same layer
        strategy_map = {
            "layer1": [tensor_info_1],
            "layer2": [tensor_info_2],
        }
        release_strategy_map = {
            "layer1": [tensor_info_1],
            "layer2": [tensor_info_2],
        }

        layer_stats = [
            LayerStatistics(label="layer1", tensors=[tensor_info_1], duration=1.0),
            LayerStatistics(label="layer2", tensors=[tensor_info_2], duration=1.0),
        ]

        device_gpu = MagicMock()
        device_gpu.type = "cuda"

        # Mock all CUDA operations to avoid needing actual GPU
        mock_stream = _create_mock_cuda_stream()
        mock_current_stream = _create_mock_cuda_stream()
        with (
            patch.object(torch.cuda, "Stream", return_value=mock_stream),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "Event", side_effect=_create_mock_cuda_event),
            patch.object(torch.cuda, "stream"),
            patch.object(torch.cuda, "current_stream", return_value=mock_current_stream),
        ):
            loader = TensorStrategyLoader(
                layer_stats=layer_stats,
                strategy_map=strategy_map,
                release_strategy_map=release_strategy_map,
                tensors_map=tensors_map,
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
            )

        peak_bytes = loader.get_gpu_memory_bytes()

        # Peak is 160KB (tensor_2), since tensors don't overlap
        # (tensor_1 released at end of layer1, before tensor_2 loaded at layer2)
        expected_peak = 200 * 200 * 4  # 160KB

        assert peak_bytes == expected_peak

    def test_strategy_loader_peak_memory_with_overlap(self):
        """Test peak memory with overlapping tensors.

        Pattern: tensor_1 loaded at layer1, released at layer2 (after tensor_2 is loaded)
        - layer1: load tensor_1 (40KB)
        - layer2: load tensor_2 (160KB), release tensor_1
        - layer3: release tensor_2

        Peak should be 200KB at layer2 when both tensors are simultaneously loaded.
        """
        from flextensor.collectors import LayerStatistics, TensorStatistics
        from flextensor.loaders import TensorStrategyLoader

        tensors_map = {
            1: torch.randn(100, 100),
            2: torch.randn(200, 200),
        }

        tensor_info_1 = TensorStatistics(
            tensor_id=1,
            name="tensor_1",
            size_bytes=100 * 100 * 4,  # 40KB
            load_time_ms=0.1,
        )
        tensor_info_2 = TensorStatistics(
            tensor_id=2,
            name="tensor_2",
            size_bytes=200 * 200 * 4,  # 160KB
            load_time_ms=0.1,
        )

        # Overlapping pattern: tensor_1 released AFTER tensor_2 is loaded
        strategy_map = {
            "layer1": [tensor_info_1],
            "layer2": [tensor_info_2],
        }
        release_strategy_map = {
            "layer2": [tensor_info_1],  # tensor_1 released at layer2 (after tensor_2 loaded)
            "layer3": [tensor_info_2],
        }

        layer_stats = [
            LayerStatistics(label="layer1", tensors=[tensor_info_1], duration=1.0),
            LayerStatistics(label="layer2", tensors=[tensor_info_2], duration=1.0),
            LayerStatistics(label="layer3", tensors=[], duration=1.0),
        ]

        device_gpu = MagicMock()
        device_gpu.type = "cuda"

        # Mock all CUDA operations to avoid needing actual GPU
        mock_stream = _create_mock_cuda_stream()
        mock_current_stream = _create_mock_cuda_stream()
        with (
            patch.object(torch.cuda, "Stream", return_value=mock_stream),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "Event", side_effect=_create_mock_cuda_event),
            patch.object(torch.cuda, "stream"),
            patch.object(torch.cuda, "current_stream", return_value=mock_current_stream),
        ):
            loader = TensorStrategyLoader(
                layer_stats=layer_stats,
                strategy_map=strategy_map,
                release_strategy_map=release_strategy_map,
                tensors_map=tensors_map,
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
            )

        peak_bytes = loader.get_gpu_memory_bytes()

        # Peak is at layer2 when both tensors are loaded: 40KB + 160KB = 200KB
        expected_peak = (100 * 100 * 4) + (200 * 200 * 4)

        assert peak_bytes == expected_peak

    def test_strategy_loader_gpu_memory_empty(self):
        """Test that TensorStrategyLoader returns 0 for empty strategy."""
        from flextensor.loaders import TensorStrategyLoader

        device_gpu = MagicMock()
        device_gpu.type = "cuda"

        # Mock all CUDA operations to avoid needing actual GPU
        mock_stream = _create_mock_cuda_stream()
        mock_current_stream = _create_mock_cuda_stream()
        with (
            patch.object(torch.cuda, "Stream", return_value=mock_stream),
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "Event", side_effect=_create_mock_cuda_event),
            patch.object(torch.cuda, "stream"),
            patch.object(torch.cuda, "current_stream", return_value=mock_current_stream),
        ):
            loader = TensorStrategyLoader(
                layer_stats=[],
                strategy_map={},
                release_strategy_map={},
                tensors_map={},
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
            )

        # Get peak GPU memory
        gpu_bytes = loader.get_gpu_memory_bytes()

        # Should be 0 since no tensors in strategy
        assert gpu_bytes == 0
