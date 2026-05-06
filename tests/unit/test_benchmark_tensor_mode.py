# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BenchmarkReplace class."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from flextensor.benchmark_tensor_mode import BenchmarkReplace, NoOpBenchmark, PreloadToDevice
from flextensor.collectors import TensorStatistics
from flextensor.host_pinning import HostPinner


class TestBenchmarkReplace:
    """Test cases for BenchmarkReplace class."""

    @pytest.fixture(autouse=True)
    def mock_cuda_sync(self):
        """Fixture to mock torch.cuda.synchronize.

        This fixture is automatically applied to all tests in this class.
        """
        with patch("torch.cuda.synchronize"):
            yield

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.device_gpu = torch.device("cuda:0")
        self.device_cpu = torch.device("cpu")

    def test_init_default_parameters(self) -> None:
        """Test BenchmarkReplace initialization with default parameters."""
        benchmark = BenchmarkReplace(self.device_gpu, host_pinner=HostPinner())

        assert benchmark.device_gpu == self.device_gpu
        assert benchmark.pinned_memory is True
        assert benchmark.iterations == 10
        assert benchmark.warmup_iterations == 5
        assert benchmark.tensor_statistics_map == {}
        assert benchmark.tensors_map == {}
        assert benchmark.pinned_memory_limit_mb is None

    def test_init_custom_parameters(self):
        """Test BenchmarkReplace initialization with custom parameters."""
        benchmark = BenchmarkReplace(
            self.device_gpu,
            pinned_memory=False,
            iterations=5,
            pinned_memory_limit_mb=128.0,
            host_pinner=HostPinner(),
        )

        assert benchmark.device_gpu == self.device_gpu
        assert benchmark.pinned_memory is False
        assert benchmark.iterations == 5
        assert benchmark.warmup_iterations == 5
        assert benchmark.pinned_memory_limit_mb == 128.0

    def test_context_manager_enter_exit(self):
        """Test BenchmarkReplace as context manager."""
        benchmark = BenchmarkReplace(self.device_gpu, host_pinner=HostPinner())

        with benchmark as ctx:
            assert ctx is benchmark

        # Should complete without error
        assert True

    def test_get_results_empty(self):
        """Test get_results returns empty results initially."""
        benchmark = BenchmarkReplace(self.device_gpu, host_pinner=HostPinner())
        results = benchmark.get_results()

        expected = {"tensor_statistics_map": {}, "tensors_map": {}}

        assert results == expected

    def test_get_results_structure(self):
        """Test get_results returns correct structure."""
        benchmark = BenchmarkReplace(self.device_gpu, host_pinner=HostPinner())

        # Add some mock data
        benchmark.tensor_statistics_map[123] = TensorStatistics(
            tensor_id=123,
            name="test",
            size_bytes=1 * 1024 * 1024,
            load_time_ms=1.5,
        )
        benchmark.tensors_map[123] = "mock_tensor"

        results = benchmark.get_results()

        assert "tensor_statistics_map" in results
        assert "tensors_map" in results

        assert results["tensor_statistics_map"][123].tensor_id == 123
        assert results["tensors_map"][123] == "mock_tensor"

    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_basic(self, mock_wrap_trace):
        """Test __torch_function__ basic functionality."""
        iterations = 2
        benchmark = BenchmarkReplace(self.device_gpu, iterations=iterations, host_pinner=HostPinner())

        # Create mock tensor
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False
        mock_tensor.numel.return_value = 1024
        mock_tensor.element_size.return_value = 4
        mock_tensor.pin_memory.return_value = mock_tensor
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False

        # Mock wrapped tensor - must spec torch.Tensor for beartype
        mock_wrapped_tensor = Mock(spec=torch.Tensor)
        mock_wrapped_tensor.numel.return_value = 1024
        mock_wrapped_tensor.element_size.return_value = 4
        mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
        mock_wrap_trace.return_value = mock_wrapped_tensor

        # Mock function call
        mock_func = Mock(return_value=mock_tensor)

        # Mock CUDA events
        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_start_event.elapsed_time.return_value = 1.5  # 1.5ms

        with patch("torch.cuda.Event") as mock_event_class:
            mock_event_class.side_effect = [mock_start_event, mock_end_event] * iterations

            result = benchmark.__torch_function__(mock_func, (), ())

        # Verify function was called
        mock_func.assert_called_once_with()

        # Verify tensor operations
        assert mock_tensor.to.call_count >= 2  # warmup + iterations

        # Verify wrapping was called
        mock_wrap_trace.assert_called_once_with(mock_tensor)

        # Verify result
        assert result == mock_wrapped_tensor

        # Verify results are stored
        results = benchmark.get_results()
        assert len(results["tensor_statistics_map"]) == 1
        assert len(results["tensors_map"]) == 1

    @patch("torch.cuda.is_available", return_value=True)
    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_pinned_memory_enabled(self, mock_wrap_trace, _mock_cuda_available):
        """Test __torch_function__ with pinned memory enabled.

        ``torch.cuda.is_available`` is patched to ``True`` because the
        default ``HostPinner()`` short-circuits ``tensor.pin_memory()`` on
        CPU-only hosts (per the documented unit-test environment) — without
        this patch the assertion below would only pass on dev machines
        that happen to have CUDA installed.
        """
        benchmark = BenchmarkReplace(self.device_gpu, pinned_memory=True, iterations=1, host_pinner=HostPinner())

        # Create mock tensor - small size, not pinned
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False
        mock_tensor.numel.return_value = 1024  # Small tensor
        mock_tensor.element_size.return_value = 4
        mock_tensor.pin_memory.return_value = mock_tensor
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False

        mock_wrapped_tensor = Mock(spec=torch.Tensor)
        mock_wrapped_tensor.numel.return_value = 1024
        mock_wrapped_tensor.element_size.return_value = 4
        mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
        mock_wrap_trace.return_value = mock_wrapped_tensor

        mock_func = Mock(return_value=mock_tensor)

        # Mock CUDA events
        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_start_event.elapsed_time.return_value = 1.0  # 1.0ms

        with patch("torch.cuda.Event") as mock_event_class:
            mock_event_class.side_effect = [mock_start_event, mock_end_event]

            benchmark.__torch_function__(mock_func, (), ())

        # Verify pin_memory was called for small tensor
        mock_tensor.pin_memory.assert_called_once()

    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_routes_pin_through_host_pinner(self, mock_wrap_trace):
        """Wiring guard: ``BenchmarkReplace.__torch_function__`` must dispatch
        pinning through the configured ``host_pinner``. The other tests in
        this class only assert ``tensor.pin_memory()`` was called — which
        still passes if a regression drops ``host_pinner=self.host_pinner``
        from the ``BenchmarkReplace(...)`` constructor inside
        ``TensorManager.benchmark_context`` (the default ``HostPinner()``
        falls back to ``tensor.pin_memory()`` too). This test fails in
        that scenario.

        Asserts on ``pin`` (not ``try_pin``) because the benchmark uses the
        strict variant: a pin failure must surface as a RuntimeError instead
        of silently producing pageable transfer measurements that would
        contaminate strategy decisions downstream.
        """
        mock_pinner = MagicMock(spec=HostPinner)
        # Gate uses host_pinner.is_pinned(...) (centralised registry-aware check),
        # not tensor.is_pinned() — see C2(a) regression test below.
        mock_pinner.is_pinned.return_value = False
        benchmark = BenchmarkReplace(self.device_gpu, pinned_memory=True, iterations=1, host_pinner=mock_pinner)

        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False
        mock_tensor.numel.return_value = 1024
        mock_tensor.element_size.return_value = 4
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False
        mock_pinner.pin.return_value = mock_tensor  # behave like a successful in-place pin

        mock_wrapped_tensor = Mock(spec=torch.Tensor)
        mock_wrapped_tensor.numel.return_value = 1024
        mock_wrapped_tensor.element_size.return_value = 4
        mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
        mock_wrap_trace.return_value = mock_wrapped_tensor

        mock_func = Mock(return_value=mock_tensor)

        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_start_event.elapsed_time.return_value = 1.0

        with patch("torch.cuda.Event") as mock_event_class:
            mock_event_class.side_effect = [mock_start_event, mock_end_event]
            benchmark.__torch_function__(mock_func, (), ())

        mock_pinner.pin.assert_called_once()
        # First positional arg must be the source CPU tensor.
        called_with_tensor = mock_pinner.pin.call_args.args[0]
        assert called_with_tensor is mock_tensor
        # Native tensor.pin_memory() must NOT have been called — the pinner owns the policy.
        mock_tensor.pin_memory.assert_not_called()

    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_aborts_on_pin_failure(self, mock_wrap_trace):
        """A host_register pin failure must propagate as a RuntimeError, not
        silently degrade to a pageable transfer measurement that would
        contaminate ``TensorStatistics.load_time_ms`` and feed corrupt data
        into downstream strategy decisions."""
        mock_pinner = MagicMock(spec=HostPinner)
        mock_pinner.is_pinned.return_value = False
        mock_pinner.pin.side_effect = RuntimeError("simulated cudaHostRegister failure")
        benchmark = BenchmarkReplace(self.device_gpu, pinned_memory=True, iterations=1, host_pinner=mock_pinner)

        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False
        mock_tensor.numel.return_value = 1024
        mock_tensor.element_size.return_value = 4
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False

        mock_func = Mock(return_value=mock_tensor)

        with pytest.raises(RuntimeError, match="cudaHostRegister"):
            benchmark.__torch_function__(mock_func, (), ())

        # Nothing must have been recorded — partial / contaminated stats are
        # worse than no stats.
        assert benchmark.tensor_statistics_map == {}
        assert benchmark.tensors_map == {}

    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_short_circuits_via_host_pinner_is_pinned(self, mock_wrap_trace):
        """The "already pinned, skip pin()" gate must consult
        ``host_pinner.is_pinned(...)``, not ``tensor.is_pinned()``. In
        host_register mode ``tensor.is_pinned()`` always reports False
        because PyTorch is unaware of ``cudaHostRegister`` registrations,
        so a ``tensor.is_pinned()`` gate would re-pin every storage on
        every intercept. Only the registry-aware ``HostPinner.is_pinned``
        knows the truth.

        Asserts that when the pinner reports True (registry hit),
        ``pin()`` is not called.
        """
        mock_pinner = MagicMock(spec=HostPinner)
        # Pinner sees the registration even though tensor.is_pinned() lies.
        mock_pinner.is_pinned.return_value = True
        benchmark = BenchmarkReplace(self.device_gpu, pinned_memory=True, iterations=1, host_pinner=mock_pinner)

        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False  # would be True if the gate were correct
        mock_tensor.numel.return_value = 1024
        mock_tensor.element_size.return_value = 4
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False

        mock_wrapped_tensor = Mock(spec=torch.Tensor)
        mock_wrapped_tensor.numel.return_value = 1024
        mock_wrapped_tensor.element_size.return_value = 4
        mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
        mock_wrap_trace.return_value = mock_wrapped_tensor

        mock_func = Mock(return_value=mock_tensor)

        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_start_event.elapsed_time.return_value = 1.0

        with patch("torch.cuda.Event") as mock_event_class:
            mock_event_class.side_effect = [mock_start_event, mock_end_event]
            benchmark.__torch_function__(mock_func, (), ())

        # Gate must have asked the pinner.
        mock_pinner.is_pinned.assert_called()
        # And because the pinner reported True, no pin call should have happened.
        mock_pinner.pin.assert_not_called()

    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_pinned_memory_disabled(self, mock_wrap_trace):
        """Test __torch_function__ with pinned memory disabled."""
        benchmark = BenchmarkReplace(self.device_gpu, pinned_memory=False, iterations=1, host_pinner=HostPinner())

        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False
        mock_tensor.numel.return_value = 1024
        mock_tensor.element_size.return_value = 4
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False

        mock_wrapped_tensor = Mock(spec=torch.Tensor)
        mock_wrapped_tensor.numel.return_value = 1024
        mock_wrapped_tensor.element_size.return_value = 4
        mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
        mock_wrap_trace.return_value = mock_wrapped_tensor

        mock_func = Mock(return_value=mock_tensor)

        # Mock CUDA events
        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_start_event.elapsed_time.return_value = 1.0  # 1.0ms

        with patch("torch.cuda.Event") as mock_event_class:
            mock_event_class.side_effect = [mock_start_event, mock_end_event]

            benchmark.__torch_function__(mock_func, (), ())

        # Verify pin_memory was NOT called
        assert not hasattr(mock_tensor, "pin_memory") or not mock_tensor.pin_memory.called

    @patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor")
    def test_torch_function_large_tensor_no_pinning(self, mock_wrap_trace):
        """Test __torch_function__ doesn't pin large tensors."""
        benchmark = BenchmarkReplace(
            self.device_gpu,
            pinned_memory=True,
            iterations=1,
            pinned_memory_limit_mb=512.0,
            host_pinner=HostPinner(),
        )

        # Create mock tensor - large size
        mock_tensor = Mock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.is_pinned.return_value = False
        mock_tensor.numel.return_value = 1024 * 1024 * 600  # > 512MB
        mock_tensor.element_size.return_value = 4
        mock_tensor.pin_memory = Mock(return_value=mock_tensor)
        mock_tensor.device.type = "cpu"
        mock_tensor.is_meta = False

        mock_wrapped_tensor = Mock(spec=torch.Tensor)
        mock_wrapped_tensor.numel.return_value = 1024 * 1024 * 600
        mock_wrapped_tensor.element_size.return_value = 4
        mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
        mock_wrap_trace.return_value = mock_wrapped_tensor

        mock_func = Mock(return_value=mock_tensor)

        # Mock CUDA events
        mock_start_event = Mock()
        mock_end_event = Mock()
        mock_start_event.elapsed_time.return_value = 1.0  # 1.0ms

        with patch("torch.cuda.Event") as mock_event_class:
            mock_event_class.side_effect = [mock_start_event, mock_end_event]

            benchmark.__torch_function__(mock_func, (), ())

        # Verify pin_memory was NOT called for large tensor
        mock_tensor.pin_memory.assert_not_called()

    def test_multiple_iterations_warmup(self):
        """Test that warmup and benchmark iterations are respected."""
        iterations = 3
        benchmark = BenchmarkReplace(self.device_gpu, iterations=iterations, host_pinner=HostPinner())

        with patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor") as mock_wrap:
            mock_tensor = Mock(spec=torch.Tensor)
            mock_tensor.to.return_value = mock_tensor
            mock_tensor.is_pinned.return_value = True  # Already pinned
            mock_tensor.numel.return_value = 1024
            mock_tensor.element_size.return_value = 4
            mock_tensor.device.type = "cpu"
            mock_tensor.is_meta = False

            mock_wrapped_tensor = Mock(spec=torch.Tensor)
            mock_wrapped_tensor.numel.return_value = 1024
            mock_wrapped_tensor.element_size.return_value = 4
            mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
            mock_wrap.return_value = mock_wrapped_tensor

            mock_func = Mock(return_value=mock_tensor)

            # Mock CUDA events
            mock_start_event = Mock()
            mock_end_event = Mock()
            mock_start_event.elapsed_time.return_value = 1.0  # 1.0ms

            with patch("torch.cuda.Event") as mock_event_class:
                mock_event_class.side_effect = [mock_start_event, mock_end_event] * iterations

                benchmark.__torch_function__(mock_func, (), ())

            # Should be called: warmup_iterations + iterations times
            # warmup (5) + benchmark (3) = 8 times
            assert mock_tensor.to.call_count == 8

    def test_tensor_statistics_creation(self):
        """Test that TensorStatistics are created correctly."""
        with patch("flextensor.benchmark_tensor_mode.wrap_trace_tensor") as mock_wrap:
            benchmark = BenchmarkReplace(self.device_gpu, iterations=1, host_pinner=HostPinner())

            mock_tensor = Mock(spec=torch.Tensor)
            mock_tensor.to.return_value = mock_tensor
            mock_tensor.is_pinned.return_value = True
            mock_tensor.numel.return_value = 1024
            mock_tensor.element_size.return_value = 4
            mock_tensor.device.type = "cpu"
            mock_tensor.is_meta = False

            mock_wrapped_tensor = Mock(spec=torch.Tensor)
            mock_wrapped_tensor.numel.return_value = 1024
            mock_wrapped_tensor.element_size.return_value = 4
            mock_wrapped_tensor.requires_grad_.return_value = mock_wrapped_tensor
            mock_wrap.return_value = mock_wrapped_tensor

            mock_func = Mock(return_value=mock_tensor)

            # Mock CUDA events
            mock_start_event = Mock()
            mock_end_event = Mock()
            mock_start_event.elapsed_time.return_value = 1.0  # 1.0ms

            with patch("torch.cuda.Event") as mock_event_class:
                mock_event_class.side_effect = [mock_start_event, mock_end_event]

                benchmark.__torch_function__(mock_func, (), ())

            results = benchmark.get_results()
            stats_map = results["tensor_statistics_map"]

            assert len(stats_map) == 1
            tensor_id = id(mock_wrapped_tensor)
            stats = stats_map[tensor_id]

            assert isinstance(stats, TensorStatistics)
            assert stats.tensor_id == tensor_id
            assert stats.name == ""
            assert stats.size_bytes == 1024 * 4  # 4kb
            assert stats.load_time_ms == 1.0  # From mocked elapsed_time


@pytest.mark.parametrize("benchmark_cls", [BenchmarkReplace, PreloadToDevice, NoOpBenchmark])
class TestBenchmarkModeSignatureContract:
    """All :class:`TensorBenchmarkMode` subclasses must accept the abstract
    ``__init__`` signature, including ``host_pinner``. ``TensorManager.benchmark_context``
    always passes ``host_pinner=self.host_pinner``, so a subclass missing the
    kwarg raises ``TypeError`` the moment ``_benchmark_cls`` swaps to it —
    silent until that swap, hard to attribute then. Pin the contract
    parametrically so any new subclass is covered automatically."""

    def test_accepts_host_pinner_kwarg(self, benchmark_cls):
        # MagicMock device avoids real CUDA on CPU CI.
        benchmark_cls(
            device_gpu=MagicMock(spec=torch.device),
            pinned_memory=True,
            iterations=1,
            pinned_memory_limit_mb=None,
            host_pinner=MagicMock(spec=HostPinner),
        )


class TestBenchmarkReplaceRequiresHostPinner:
    """``BenchmarkReplace`` is the only benchmark mode that actually pins
    tensors — silently substituting a default torch-mode :class:`HostPinner`
    when the kwarg is dropped would let
    ``pinned_memory_mode='host_register'`` run profiling under
    ``"torch"``-mode latencies and contaminate strategy decisions. The
    required-arg contract turns the misuse into a ``TypeError`` at
    construction instead of a silent mode mismatch at runtime."""

    def test_construction_without_pinner_raises(self):
        with pytest.raises(TypeError, match="host_pinner"):
            BenchmarkReplace(  # type: ignore[call-arg]
                device_gpu=MagicMock(spec=torch.device),
            )


class TestTensorStatistics:
    """Test cases for TensorStatistics dataclass."""

    def test_tensor_statistics_creation(self):
        """Test TensorStatistics creation."""
        stats = TensorStatistics(tensor_id=123, name="test_tensor", size_bytes=int(2.5 * 1024 * 1024), load_time_ms=1.5)

        assert stats.tensor_id == 123
        assert stats.name == "test_tensor"
        assert stats.size_bytes == 2.5 * 1024 * 1024
        assert stats.load_time_ms == 1.5
