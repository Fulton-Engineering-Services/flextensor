# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TensorManager benchmark integration functionality."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.benchmark_tensor_mode import BenchmarkReplace, NoOpBenchmark, PreloadToDevice, TensorBenchmarkMode
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.strategy import KnapsackStrategy
from flextensor.tensor import TraceTensor
from flextensor.tensor_manager import TensorManager


class TestTensorManagerBenchmarkIntegration:
    """Test cases for TensorManager benchmark integration."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.device_gpu = torch.device("cuda:0")
        self.device_cpu = torch.device("cpu")
        self.strategy = KnapsackStrategy(scale=0.8)

    def test_tensor_manager_default_benchmark_cls_is_noop(self) -> None:
        """Default (non-tracing) TensorManager uses NoOpBenchmark."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )

        assert tensor_manager._benchmark_cls == NoOpBenchmark

    def test_tensor_manager_tracing_uses_benchmark_replace(self) -> None:
        """With _use_trace_tensor=True, TensorManager uses BenchmarkReplace."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            _use_trace_tensor=True,
        )

        assert tensor_manager._benchmark_cls == BenchmarkReplace

    def test_trace_tensor_rebinds_is_traced(self) -> None:
        """_use_trace_tensor=True rebinds is_traced to is_traced_trace_tensor."""
        tm_default = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tm_tracing = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            _use_trace_tensor=True,
        )

        assert tm_default.is_traced == tm_default.__class__.is_traced.__get__(tm_default)
        assert tm_tracing.is_traced == tm_tracing.is_traced_trace_tensor

    def test_default_is_traced_uses_id_set(self) -> None:
        """Default is_traced returns True only for tensors whose id is in traced_tensors."""
        tm = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor = torch.zeros(4)
        assert tm.is_traced(tensor) is False

        tm.traced_tensors.add(id(tensor))
        assert tm.is_traced(tensor) is True

        assert tm.is_traced("not a tensor") is False

    def test_trace_tensor_is_traced_uses_isinstance(self) -> None:
        """With _use_trace_tensor=True, is_traced checks isinstance(TraceTensor)."""
        tm = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            _use_trace_tensor=True,
        )
        plain = torch.zeros(4)
        traced = TraceTensor(torch.zeros(4))

        assert tm.is_traced(plain) is False
        assert tm.is_traced(traced) is True

        tm.traced_tensors.add(id(plain))
        assert tm.is_traced(plain) is False

    def test_benchmark_context_method_exists(self) -> None:
        """Test that benchmark_context method exists and is callable."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )

        assert hasattr(tensor_manager, "benchmark_context")
        assert callable(tensor_manager.benchmark_context)

    def test_benchmark_context_creates_benchmark_instance(self) -> None:
        """Test that benchmark_context creates benchmark instance with correct parameters."""

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
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, TestMockBenchmark)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory is True
            assert benchmark.iterations == 5

    def test_benchmark_context_default_iterations(self) -> None:
        """Test that benchmark_context uses default iterations parameter."""

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
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        with tensor_manager.benchmark_context() as benchmark:
            assert isinstance(benchmark, TestMockBenchmark)
            assert benchmark.device_gpu == self.device_gpu

    def test_benchmark_context_automatic_stats_integration(self) -> None:
        """Test that benchmark results are automatically integrated into TensorManager."""

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
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        assert tensor_manager.tensor_statistics_map == {}
        assert tensor_manager.tensors_map == {}

        with tensor_manager.benchmark_context():
            pass

        assert tensor_manager.tensor_statistics_map == {123: "stats1", 456: "stats2"}
        assert tensor_manager.tensors_map == {123: "tensor_obj1", 456: "tensor_obj2"}

    def test_benchmark_context_stats_replacement(self) -> None:
        """Test that benchmark results replace existing TensorManager stats."""

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

        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        tensor_manager.tensor_statistics_map = {111: "existing_stats"}
        tensor_manager.tensors_map = {111: "existing_tensor_obj"}

        with tensor_manager.benchmark_context():
            pass

        assert tensor_manager.tensor_statistics_map == {123: "new_stats"}
        assert tensor_manager.tensors_map == {123: "new_tensor_obj"}

    def test_benchmark_context_exception_handling(self) -> None:
        """Test that stats are still integrated even if exception occurs in context."""

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
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = TestMockBenchmark

        with pytest.raises(ValueError), tensor_manager.benchmark_context():
            raise ValueError("Test exception")

        assert tensor_manager.tensor_statistics_map == {123: "stats1"}
        assert tensor_manager.tensors_map == {123: "tensor_obj1"}

    def test_hardcoded_internal_fields(self) -> None:
        """Test that release_tensors and direct_enabled are hardcoded, pinned_memory is configurable."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )

        assert tensor_manager.pinned_memory is True
        assert tensor_manager.release_tensors is True
        assert tensor_manager.direct_enabled is True
        assert tensor_manager.tensor_manager_load_strategy == self.strategy
        assert hasattr(tensor_manager, "prepare_warmup_mode")
        assert hasattr(tensor_manager, "prepare_profile_mode")
        assert hasattr(tensor_manager, "prepare_infer_mode")
        assert hasattr(tensor_manager, "trap")
        assert hasattr(tensor_manager, "release_memory")
        assert hasattr(tensor_manager, "is_traced")

    def test_pinned_memory_configurable(self) -> None:
        """Test that pinned_memory can be set to False."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            pinned_memory=False,
        )

        assert tensor_manager.pinned_memory is False

    @pytest.mark.parametrize("loader_type", ["allocation_block_transfer", "raw_block_transfer"])
    def test_block_transfer_requires_direct_mode(self, loader_type: str) -> None:
        """Block transfer loaders must reject _direct_mode=False."""
        with pytest.raises(ValueError, match="_direct_mode=False is incompatible"):
            TensorManager(
                device_gpu=self.device_gpu,
                tensor_manager_load_strategy=self.strategy,
                loader_type=loader_type,
                _direct_mode=False,
            )

    def test_strategy_loader_allows_direct_mode_false(self) -> None:
        """Strategy loader accepts _direct_mode=False."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
            loader_type="strategy",
            _direct_mode=False,
        )

        assert tensor_manager.direct_enabled is False


class MockBenchmarkReplace(TensorBenchmarkMode):
    """Mock benchmark class for testing."""

    def __init__(self, device_gpu, pinned_memory, iterations):
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
        """Test that a custom benchmark class can be injected and used."""
        tensor_manager = TensorManager(
            device_gpu=self.device_gpu,
            tensor_manager_load_strategy=self.strategy,
        )
        tensor_manager._benchmark_cls = MockBenchmarkReplace

        with tensor_manager.benchmark_context(iterations=5) as benchmark:
            assert isinstance(benchmark, MockBenchmarkReplace)
            assert benchmark.device_gpu == self.device_gpu
            assert benchmark.pinned_memory
            assert benchmark.iterations == 5

        assert tensor_manager.tensor_statistics_map == {999: "mock_stats"}
        assert tensor_manager.tensors_map == {999: "mock_tensor_obj"}

    def test_abstract_base_class_interface(self) -> None:
        """Test that both BenchmarkReplace and PreloadToDevice implement TensorBenchmarkMode interface."""
        assert issubclass(BenchmarkReplace, TensorBenchmarkMode)
        assert issubclass(PreloadToDevice, TensorBenchmarkMode)

        benchmark_replace = BenchmarkReplace(self.device_gpu, pinned_memory=True, iterations=10)
        preload_to_device = PreloadToDevice(self.device_gpu, pinned_memory=True, iterations=10)

        assert hasattr(benchmark_replace, "get_results")
        assert hasattr(preload_to_device, "get_results")
        assert callable(benchmark_replace.get_results)
        assert callable(preload_to_device.get_results)

        br_results = benchmark_replace.get_results()
        ptd_results = preload_to_device.get_results()

        expected_keys = {"tensor_statistics_map", "tensors_map"}
        assert set(br_results.keys()) == expected_keys
        assert set(ptd_results.keys()) == expected_keys


def _make_layer(label: str, n_tensors: int = 1, duration: float = 10.0) -> LayerStatistics:
    """Helper to create a LayerStatistics with dummy tensors."""
    tensors = [
        TensorStatistics(tensor_id=i, name=f"{label}_t{i}", size_bytes=1024, load_time_ms=0.1) for i in range(n_tensors)
    ]
    return LayerStatistics(label=label, tensors=tensors, duration=duration)


def _make_gap_layer(label: str) -> LayerStatistics:
    """Helper to create a gap layer (no tensors)."""
    return LayerStatistics(label=label, tensors=[], duration=5.0)


def _make_tm(**kwargs) -> TensorManager:
    """Create a TensorManager with minimal required args."""
    defaults = {
        "device_gpu": torch.device("cuda:0"),
        "tensor_manager_load_strategy": KnapsackStrategy(scale=0.8),
    }
    defaults.update(kwargs)
    return TensorManager(**defaults)


class TestUntracedTensorDiscoveryBranch:
    """Verify enable_untraced_tensor_discovery gates discover_untraced_tensors_for_layers."""

    def _prepare_tm(self, *, discovery_enabled: bool) -> TensorManager:
        tm = _make_tm(_enable_untraced_tensor_discovery=discovery_enabled)
        tm.layer_statistics_collector = MagicMock()
        fake_stats = [MagicMock()]
        tm.layer_statistics_collector.get_layer_stats.return_value = fake_stats
        tm.model = MagicMock()
        tm.tensor_id_to_name_map = {}
        tm.module_tracker = None
        return tm

    @patch("flextensor.tensor_manager.TensorLayerLoader")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.discover_untraced_tensors_for_layers")
    def test_discovery_called_when_enabled(self, mock_discover, mock_filter, _mock_loader) -> None:
        mock_filter.return_value.filter_by_tensor_ids.return_value = [_make_layer("layer_0")]
        mock_discover.return_value = [_make_layer("layer_0")]
        tm = self._prepare_tm(discovery_enabled=True)
        tm.prepare_profile_direct_mode()
        mock_discover.assert_called_once()

    @patch("flextensor.tensor_manager.TensorLayerLoader")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.discover_untraced_tensors_for_layers")
    def test_discovery_skipped_when_disabled(self, mock_discover, mock_filter, _mock_loader) -> None:
        mock_filter.return_value.filter_by_tensor_ids.return_value = [_make_layer("layer_0")]
        tm = self._prepare_tm(discovery_enabled=False)
        tm.prepare_profile_direct_mode()
        mock_discover.assert_not_called()


class TestAutoEnableRearrangeTransfers:
    """Verify rearrange_transfers is auto-enabled when gap layers are detected.

    Tests exercise prepare_infer_mode with heavy patching to isolate the
    auto-enable branch that checks for permanent gap layers.
    """

    def _prepare_tm_for_infer(self, *, rearrange: bool = False) -> TensorManager:
        tm = _make_tm(loader_type="strategy", _rearrange_transfers=rearrange)
        tm.layer_statistics_collector = MagicMock()
        tm.layer_statistics_collector.get_layer_stats.return_value = []
        tm.model = MagicMock(spec=[])
        tm.tensor_id_to_name_map = {}
        tm.tensors_map = {}
        return tm

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps", return_value=True)
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch.object(TensorManager, "_resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality")
    def test_auto_enables_on_gap_layers(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_with_gaps = [_make_layer("layer_0"), _make_gap_layer("gap"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_with_gaps

        tm = self._prepare_tm_for_infer(rearrange=False)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "gap": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is True
        mock_has_gaps.assert_called_once()

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps", return_value=False)
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch.object(TensorManager, "_resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality")
    def test_no_auto_enable_without_transfer_gaps(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_with_gaps = [_make_layer("layer_0"), _make_gap_layer("gap"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_with_gaps

        tm = self._prepare_tm_for_infer(rearrange=False)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "gap": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is False

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps")
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch.object(TensorManager, "_resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality")
    def test_no_auto_enable_without_gap_layers(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_no_gaps = [_make_layer("layer_0"), _make_layer("layer_1"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_no_gaps

        tm = self._prepare_tm_for_infer(rearrange=False)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "layer_1": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is False
        mock_has_gaps.assert_not_called()

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps")
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch.object(TensorManager, "_resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics")
    @patch("flextensor.tensor_manager.IterativeLayerStatisticsFilter")
    @patch("flextensor.tensor_manager.report_profiling_quality")
    def test_already_enabled_skips_auto_detection(
        self, _report_quality, _filter, mock_compute_stats, _bench, _mem, _budget, _remove, mock_has_gaps, _loader
    ) -> None:
        stats_with_gaps = [_make_layer("layer_0"), _make_gap_layer("gap"), _make_layer("layer_2")]
        mock_compute_stats.return_value = stats_with_gaps

        tm = self._prepare_tm_for_infer(rearrange=True)
        result = MagicMock()
        result.strategy_map = {"layer_0": [], "gap": [], "layer_2": []}
        result.block_data = None
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = result

        tm.prepare_infer_mode()

        assert tm.rearrange_transfers is True
        mock_has_gaps.assert_not_called()
