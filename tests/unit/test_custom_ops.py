# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the compiled-offload rolling-block custom ops."""

import pytest
import torch
from torch import nn

from flextensor import custom_ops
from flextensor.loaders import PreallocatedLoader


class _RecordingLoader(PreallocatedLoader):
    def __init__(self) -> None:
        self.entered: list[str] = []
        self.exited: list[str] = []

    def enter(self, label: str) -> None:
        self.entered.append(label)

    def exit(self, label: str) -> None:
        self.exited.append(label)

    def preload(self) -> None:
        return None

    def prepare(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_singleton() -> None:
    custom_ops.clear_active_loader()
    yield
    custom_ops.clear_active_loader()


class TestOpRegistration:
    def test_ops_are_addressable(self) -> None:
        assert hasattr(torch.ops.flextensor, "pre_compute")
        assert hasattr(torch.ops.flextensor, "post_compute")


class TestFakeImplementationsRequired:
    """``Library.define`` + ``CompositeExplicitAutograd`` needs explicit fakes.

    ``@torch.library.custom_op`` supplies a trivial FakeTensor path for mutable
    zero-return ops; ``Library.define`` does not. Without ``register_fake``,
    FakeTensor tracing falls through to the real kernel and would run the loader
    during Dynamo tracing.
    """

    def test_library_define_without_fake_runs_composite_under_fake_tensor(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.library import Library

        lib = Library("flextensor_fake_probe", "FRAGMENT")
        lib.define("probe(Tensor(a!) x) -> ()")
        calls: list[str] = []

        def _impl(x: torch.Tensor) -> None:
            calls.append("impl")

        lib.impl("probe", _impl, "CompositeExplicitAutograd")

        with FakeTensorMode():
            torch.ops.flextensor_fake_probe.probe(torch.zeros(2))

        assert calls == ["impl"], (
            "Library.define without register_fake must fall through to Composite "
            "under FakeTensorMode — that is why flextensor pre/post_compute register fakes."
        )

    def test_flextensor_fakes_skip_loader_under_fake_tensor(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)

        with FakeTensorMode():
            t = torch.zeros(2)
            torch.ops.flextensor.pre_compute(t, "layer", 0)
            torch.ops.flextensor.post_compute(t, "layer", 0)

        assert loader.entered == []
        assert loader.exited == []

        # Sanity: eager path still dispatches to the loader.
        t = torch.zeros(2)
        torch.ops.flextensor.pre_compute(t, "layer", 0)
        torch.ops.flextensor.post_compute(t, "layer", 0)
        assert loader.entered == ["layer"]
        assert loader.exited == ["layer"]


class TestNoopWhenLoaderUnset:
    def test_enter_is_noop_without_loader(self) -> None:
        """Pre-install (unarmed) phase must remain a hard no-op."""
        t = torch.zeros(2)
        torch.ops.flextensor.pre_compute(t, "layer0", 0)
        torch.ops.flextensor.post_compute(t, "layer99", 0)
        assert custom_ops.get_active_loader(0) is None


class TestArmedMissingLoaderRaises:
    def test_cleared_armed_manager_raises_on_enter_and_exit(self) -> None:
        """After install+clear, runtime must not silently skip transfers."""
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        custom_ops.clear_active_loader(0)
        assert custom_ops.get_active_loader(0) is None
        assert custom_ops._STATES[0].require_loader is True

        t = torch.zeros(1)
        with pytest.raises(RuntimeError, match="no loader is registered"):
            torch.ops.flextensor.pre_compute(t, "layer0", 0)
        with pytest.raises(RuntimeError, match="no loader is registered"):
            torch.ops.flextensor.post_compute(t, "layer0", 0)
        assert loader.entered == []
        assert loader.exited == []

    def test_full_clear_disarms_and_restores_noop(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        custom_ops.clear_active_loader()
        t = torch.zeros(1)
        torch.ops.flextensor.pre_compute(t, "x", 0)
        torch.ops.flextensor.post_compute(t, "x", 0)
        assert loader.entered == []
        assert custom_ops._STATES == {}


class TestInstallAndDispatch:
    def test_install_sets_loader(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        assert custom_ops.get_active_loader(0) is loader

    def test_enter_exit_dispatch_to_loader_by_name(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.zeros(3)

        torch.ops.flextensor.pre_compute(t, "layer1", 0)
        torch.ops.flextensor.post_compute(t, "layer2", 0)

        assert loader.entered == ["layer1"]
        assert loader.exited == ["layer2"]

    def test_multi_manager_dispatch_is_independent(self) -> None:
        loader_a = _RecordingLoader()
        loader_b = _RecordingLoader()
        custom_ops.install_active_loader(loader_a, manager_id=1)
        custom_ops.install_active_loader(loader_b, manager_id=2)
        t = torch.zeros(1)

        torch.ops.flextensor.pre_compute(t, "a1", 1)
        torch.ops.flextensor.post_compute(t, "a0", 1)
        torch.ops.flextensor.pre_compute(t, "b2", 2)
        torch.ops.flextensor.post_compute(t, "b1", 2)

        assert loader_a.entered == ["a1"]
        assert loader_a.exited == ["a0"]
        assert loader_b.entered == ["b2"]
        assert loader_b.exited == ["b1"]

    def test_clear_one_manager_leaves_other_installed(self) -> None:
        loader_a = _RecordingLoader()
        loader_b = _RecordingLoader()
        custom_ops.install_active_loader(loader_a, manager_id=1)
        custom_ops.install_active_loader(loader_b, manager_id=2)

        custom_ops.clear_active_loader(1)

        assert custom_ops.get_active_loader(1) is None
        assert custom_ops.get_active_loader(2) is loader_b
        t = torch.zeros(1)
        with pytest.raises(RuntimeError, match="manager_id=1"):
            torch.ops.flextensor.pre_compute(t, "a0", 1)
        torch.ops.flextensor.pre_compute(t, "b0", 2)
        assert loader_b.entered == ["b0"]

    def test_clear_restores_noop(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        custom_ops.enable_compiled_profiling(0)
        custom_ops.clear_active_loader()

        assert custom_ops.get_active_loader(0) is None
        assert custom_ops._STATES == {}

        t = torch.zeros(1)
        torch.ops.flextensor.pre_compute(t, "x", 0)
        assert loader.entered == []

    def test_reinstall_replaces_loader_for_same_manager(self) -> None:
        first = _RecordingLoader()
        second = _RecordingLoader()
        custom_ops.install_active_loader(first, manager_id=0)
        custom_ops.install_active_loader(second, manager_id=0)
        assert custom_ops.get_active_loader(0) is second

    def test_carrier_values_unchanged_despite_mutation_schema(self) -> None:
        """Schema uses ``Tensor(a!)`` for compile ordering; eager kernels leave values intact."""
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.tensor([1.0, 2.0, 3.0])
        before = t.clone()
        torch.ops.flextensor.pre_compute(t, "only", 0)
        torch.ops.flextensor.post_compute(t, "only", 0)
        assert torch.equal(t, before)

    def test_opcheck_accepts_compute_boundary_ops(self) -> None:
        """``torch.library.opcheck`` validates schema/fake/autograd registration (torch>=2.5)."""
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.zeros(2)
        torch.library.opcheck(torch.ops.flextensor.pre_compute.default, (t, "only", 0))
        torch.library.opcheck(torch.ops.flextensor.post_compute.default, (t, "only", 0))
        assert loader.entered  # opcheck exercises the real impl
        assert loader.exited


class TestCompiledProfiling:
    @pytest.fixture(autouse=True)
    def _reset_profiling(self) -> None:
        custom_ops.clear_active_loader()
        yield
        custom_ops.clear_active_loader()

    def test_toggle_sets_flag_and_clears_state(self) -> None:
        custom_ops.enable_compiled_profiling(0)
        assert custom_ops._STATES[0].profiling is True
        custom_ops.disable_compiled_profiling(0)
        assert custom_ops._STATES[0].profiling is False

    def test_collect_is_empty_without_events(self) -> None:
        custom_ops.enable_compiled_profiling(0)
        assert custom_ops.collect_compiled_layer_durations(0) == {}

    def test_finish_combines_disable_and_collect(self) -> None:
        custom_ops.enable_compiled_profiling(0)
        assert custom_ops.finish_compiled_profiling(0) == {}
        assert custom_ops._STATES[0].profiling is False

    def test_disabled_profiling_records_nothing(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.zeros(2)
        torch.ops.flextensor.pre_compute(t, "a", 0)
        torch.ops.flextensor.post_compute(t, "a", 0)
        assert custom_ops.collect_compiled_layer_durations(0) == {}

    def test_cpu_carrier_does_not_record_events(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.zeros(2)  # CPU
        custom_ops.enable_compiled_profiling(0)
        torch.ops.flextensor.pre_compute(t, "a", 0)
        torch.ops.flextensor.post_compute(t, "a", 0)
        assert custom_ops._STATES[0].pending_start == {}
        assert custom_ops._STATES[0].duration_events == []
        assert custom_ops.finish_compiled_profiling(0) == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA event timing requires a GPU")
    def test_collect_syncs_end_events_not_current_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.zeros(1, device="cuda")
        custom_ops.enable_compiled_profiling(0)
        torch.ops.flextensor.pre_compute(t, "l0", 0)
        torch.ops.flextensor.post_compute(t, "l0", 0)

        def _forbidden_synchronize(*_args, **_kwargs):
            raise AssertionError("collect must not call torch.cuda.synchronize()")

        monkeypatch.setattr(torch.cuda, "synchronize", _forbidden_synchronize)
        durations = custom_ops.collect_compiled_layer_durations(0)
        assert "l0" in durations
        assert durations["l0"][0] >= 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA event timing requires a GPU")
    def test_records_per_unit_durations_on_cuda(self) -> None:
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        t = torch.zeros(2, device="cuda")
        custom_ops.enable_compiled_profiling(0)
        for _ in range(3):
            for name in ("l0", "l1"):
                torch.ops.flextensor.pre_compute(t, name, 0)
                torch.ops.flextensor.post_compute(t, name, 0)
        durations = custom_ops.finish_compiled_profiling(0)
        assert sorted(durations) == ["l0", "l1"]
        assert all(len(samples) == 3 for samples in durations.values())
        assert all(d >= 0.0 for samples in durations.values() for d in samples)
        assert custom_ops._STATES[0].duration_events == []

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Needs two CUDA devices so current device can differ from the carrier",
    )
    def test_records_on_carrier_device_when_current_device_differs(self) -> None:
        """Profiling must use the carrier GPU, not torch.cuda.current_device()."""
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)
        carrier_device = torch.device("cuda:1")
        t = torch.zeros(1, device=carrier_device)
        previous = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            assert torch.cuda.current_device() == 0
            custom_ops.enable_compiled_profiling(0)
            torch.ops.flextensor.pre_compute(t, "l0", 0)
            # Keep current device on 0 while the carrier work is on cuda:1.
            torch.ops.flextensor.post_compute(t, "l0", 0)
            durations = custom_ops.finish_compiled_profiling(0)
        finally:
            torch.cuda.set_device(previous)
        assert "l0" in durations
        assert durations["l0"][0] >= 0.0
        assert loader.entered == ["l0"]
        assert loader.exited == ["l0"]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA event timing requires a GPU")
    def test_collect_clears_events_for_manager_only(self) -> None:
        loader_a = _RecordingLoader()
        loader_b = _RecordingLoader()
        custom_ops.install_active_loader(loader_a, manager_id=1)
        custom_ops.install_active_loader(loader_b, manager_id=2)
        t = torch.zeros(1, device="cuda")
        custom_ops.enable_compiled_profiling(1)
        custom_ops.enable_compiled_profiling(2)
        torch.ops.flextensor.pre_compute(t, "a0", 1)
        torch.ops.flextensor.post_compute(t, "a0", 1)
        torch.ops.flextensor.pre_compute(t, "b0", 2)
        torch.ops.flextensor.post_compute(t, "b0", 2)
        custom_ops.disable_compiled_profiling(1)
        custom_ops.disable_compiled_profiling(2)
        assert len(custom_ops._STATES[1].duration_events) == 1
        assert len(custom_ops._STATES[2].duration_events) == 1

        custom_ops.collect_compiled_layer_durations(1)
        assert custom_ops._STATES[1].duration_events == []
        assert len(custom_ops._STATES[2].duration_events) == 1

        custom_ops.collect_compiled_layer_durations(2)
        assert custom_ops._STATES[2].duration_events == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA event timing requires a GPU")
    def test_profiling_is_scoped_per_manager(self) -> None:
        loader_a = _RecordingLoader()
        loader_b = _RecordingLoader()
        custom_ops.install_active_loader(loader_a, manager_id=1)
        custom_ops.install_active_loader(loader_b, manager_id=2)
        t = torch.zeros(1, device="cuda")
        custom_ops.enable_compiled_profiling(1)
        custom_ops.enable_compiled_profiling(2)
        torch.ops.flextensor.pre_compute(t, "a0", 1)
        torch.ops.flextensor.post_compute(t, "a0", 1)
        torch.ops.flextensor.pre_compute(t, "b0", 2)
        torch.ops.flextensor.post_compute(t, "b0", 2)
        durations_a = custom_ops.finish_compiled_profiling(1)
        durations_b = custom_ops.collect_compiled_layer_durations(2)
        assert "a0" in durations_a
        assert "b0" in durations_b
        assert custom_ops._STATES[2].profiling is True
        assert custom_ops._STATES[1].duration_events == []
        assert custom_ops._STATES[2].duration_events == []

    def test_concurrent_clear_does_not_drop_other_manager_events(self) -> None:
        """Clearing manager A must not rewrite manager B's event list.

        Uses synthetic duration entries (no CUDA) so the ownership race is
        exercised on CPU CI. Separate managers are documented as usable on
        separate threads.
        """
        import threading

        loader_a = _RecordingLoader()
        loader_b = _RecordingLoader()
        custom_ops.install_active_loader(loader_a, manager_id=1)
        custom_ops.install_active_loader(loader_b, manager_id=2)
        state_a = custom_ops._STATES[1]
        state_b = custom_ops._STATES[2]

        sentinel = object()
        stop = threading.Event()
        errors: list[BaseException] = []

        def clearer() -> None:
            try:
                while not stop.is_set():
                    # Public APIs that previously filter/slice-replaced a shared list.
                    state_a.duration_events.append(("a0", sentinel, sentinel))  # type: ignore[arg-type]
                    custom_ops.enable_compiled_profiling(1)
                    custom_ops.disable_compiled_profiling(1)
                    state_a.duration_events.clear()
            except BaseException as exc:  # noqa: BLE001 - surface to main thread
                errors.append(exc)

        def writer() -> None:
            try:
                for _ in range(2000):
                    state_b.duration_events.append(("b0", sentinel, sentinel))  # type: ignore[arg-type]
            except BaseException as exc:  # noqa: BLE001 - surface to main thread
                errors.append(exc)

        t_clear = threading.Thread(target=clearer)
        t_write = threading.Thread(target=writer)
        t_clear.start()
        t_write.start()
        t_write.join()
        stop.set()
        t_clear.join()

        assert errors == []
        assert len(state_b.duration_events) == 2000


class TestDynamoTraceability:
    @pytest.fixture(autouse=True)
    def _reset_state(self) -> None:
        custom_ops.clear_active_loader()
        yield
        custom_ops.clear_active_loader()
        torch._dynamo.reset()

    def test_pre_post_compute_ops_trace_as_single_subgraph_cpu(self) -> None:
        """Custom ops mark residency without splitting the unit into multiple subgraphs (CPU)."""
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, manager_id=0)

        class _Block(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                torch.ops.flextensor.pre_compute(x, "layer", 0)
                y = x * 2.0 + 1.0
                torch.ops.flextensor.post_compute(y, "layer", 0)
                return y

        mod = _Block().eval()
        x = torch.randn(2, 4)
        graph_count = [0]

        def counting_backend(gm: torch.fx.GraphModule, example_inputs: list) -> object:
            graph_count[0] += 1
            return gm.forward

        compiled = torch.compile(mod, backend=counting_backend, fullgraph=True)
        with torch.no_grad():
            out = compiled(x)

        assert graph_count[0] == 1, f"expected 1 Dynamo subgraph, got {graph_count[0]}"
        assert loader.entered == ["layer"]
        assert loader.exited == ["layer"]
        torch.testing.assert_close(out, x * 2.0 + 1.0)
