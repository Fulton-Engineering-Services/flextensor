# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for compiled-offload wiring (custom ops, INFERENCE setup).

These pin contracts for loader install, compiled-forward dispatch, and the
INFERENCE transition ordering. Compiled inference drives ``pre_compute/post_compute``
from auto-patched forwards (not via a trap class).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor import custom_ops
from flextensor.compile.warmup_tail import CompiledOffloadTailState
from flextensor.loaders import PreallocatedLoader
from flextensor.offload_manager import OffloadManager, OffloadPhase


class SimpleLayer(nn.Module):
    def __init__(self, features: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ModelWithLayers(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = SimpleLayer()
        self.layer2 = SimpleLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class _RecordingLoader(PreallocatedLoader):
    """Stand-in loader that records ``enter`` / ``exit`` labels."""

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
def _clean_loader_singleton() -> None:
    custom_ops.clear_active_loader()
    yield
    custom_ops.clear_active_loader()


class TestCompiledForwardLoaderDispatch:
    def test_happy_path_dispatches_enter_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FT_EXTERNAL_COMPILE", "1")
        model = ModelWithLayers()
        om = OffloadManager("test_co_happy")
        om._compiled.active = True
        om._patch_module_forward(model.layer1, "layer1")
        om._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
        om._install_compiled_forwards()  # noqa: SLF001

        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, om.compiled_offload_manager_id)

        x = torch.randn(2, 10)
        out = model.layer1(x)

        assert loader.entered == ["layer1"]
        assert loader.exited == ["layer1"]
        assert out.shape == (2, 10)
        om.release()


class TestRequireCompiledLoader:
    def test_wires_tensor_manager_loader(self) -> None:
        om = OffloadManager("test_install_loader")
        model = ModelWithLayers()
        om._compiled.active = True
        om._patch_module_forward(model.layer1, "layer1")
        om._patch_module_forward(model.layer2, "layer2")

        loader = _RecordingLoader()
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=loader)  # noqa: SLF001

        om._compiled.require_compiled_loader()

        mid = om.compiled_offload_manager_id
        assert custom_ops.get_active_loader(mid) is loader
        custom_ops.clear_active_loader(mid)

    def test_raises_when_no_loader(self) -> None:
        om = OffloadManager("test_install_loader_missing")
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=None)  # noqa: SLF001
        with pytest.raises(RuntimeError, match="no inference loader"):
            om._compiled.require_compiled_loader()
        assert custom_ops.get_active_loader(om.compiled_offload_manager_id) is None

    def test_clears_stale_loader_when_no_loader(self) -> None:
        om = OffloadManager("test_install_loader_stale")
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=None)  # noqa: SLF001
        stale_loader = _RecordingLoader()
        mid = om.compiled_offload_manager_id
        custom_ops.install_active_loader(stale_loader, mid)

        with pytest.raises(RuntimeError, match="no inference loader"):
            om._compiled.require_compiled_loader()
        assert custom_ops.get_active_loader(mid) is None

    def test_raises_when_loader_is_not_preallocated(self) -> None:
        om = OffloadManager("test_install_loader_wrong_type")
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=object())  # noqa: SLF001
        with pytest.raises(RuntimeError, match="is not a PreallocatedLoader"):
            om._compiled.require_compiled_loader()
        assert custom_ops.get_active_loader(om.compiled_offload_manager_id) is None


class TestCompiledOffloadTransferMode:
    def test_resolve_activation_rejects_strategy_with_compile_fn(self) -> None:
        from flextensor.config import OffloadConfig

        om = OffloadManager("test_compile_fn_strategy")
        config = OffloadConfig(transfer_mode="strategy")
        with pytest.raises(ValueError, match="requires a block transfer_mode"):
            om._compiled.resolve_activation(config, compile_fn=lambda m: m)
        assert om._compiled.active is False

    def test_invalid_reoffload_preserves_active_compiled_state(self) -> None:
        """Rejecting a bad re-offload must not tear down the working session."""
        from flextensor.config import OffloadConfig

        om = OffloadManager("test_invalid_reoffload_preserves")
        co = om._compiled
        mid = om.compiled_offload_manager_id
        compile_fn = lambda m: m  # noqa: E731

        assert co.resolve_activation(OffloadConfig(enabled=True, profile_mode="view"), compile_fn=compile_fn) is True
        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, mid)
        co.profile_compile_warm_remaining = 2
        co.substitutions.append((lambda _m: None, object()))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="requires a block transfer_mode"):
            co.resolve_activation(OffloadConfig(transfer_mode="strategy"), compile_fn=lambda m: m)

        assert co.active is True
        assert co.compile_fn is compile_fn
        assert co.profile_active is True
        assert co.replan_active is False
        assert custom_ops.get_active_loader(mid) is loader
        assert co.profile_compile_warm_remaining == 2
        assert len(co.substitutions) == 1

        with pytest.raises(ValueError, match="requires a block transfer_mode"):
            # Bypass OffloadConfig's own validator so we exercise resolve_activation's check.
            co.resolve_activation(
                OffloadConfig.model_construct(enabled=True, external_compile=True, transfer_mode="strategy"),
                compile_fn=None,
            )

        assert co.active is True
        assert co.compile_fn is compile_fn
        assert custom_ops.get_active_loader(mid) is loader


class TestCompiledToEagerReoffload:
    def test_resolve_activation_clears_stale_compile_fn_and_loader(self) -> None:
        """Compiled → eager re-offload must drop compile_fn and loader ownership."""
        from flextensor.config import OffloadConfig

        om = OffloadManager("test_compiled_to_eager_reoffload")
        co = om._compiled
        mid = om.compiled_offload_manager_id

        compiled_config = OffloadConfig(enabled=True, profile_mode="view")
        assert co.resolve_activation(compiled_config, compile_fn=lambda m: m) is True
        assert co.compile_fn is not None
        assert co.active is True

        loader = _RecordingLoader()
        custom_ops.install_active_loader(loader, mid)
        assert custom_ops.get_active_loader(mid) is loader

        eager_config = OffloadConfig(enabled=True)
        assert co.resolve_activation(eager_config, compile_fn=None) is False
        assert co.compile_fn is None
        assert co.active is False
        assert co.profile_active is False
        assert co.replan_active is False
        assert custom_ops.get_active_loader(mid) is None

        setup_calls: list[str] = []
        co.setup_inference_no_replan = lambda: setup_calls.append("compile")  # type: ignore[method-assign]
        co.setup_external_compiled_offload = lambda: setup_calls.append("external")  # type: ignore[method-assign]
        co.on_enter_inference()
        assert setup_calls == []

        om.release()
        assert co.compile_fn is None
        assert custom_ops.get_active_loader(mid) is None

    def test_resolve_activation_clears_tensor_manager_replan_state(self) -> None:
        """Compiled → eager/view re-offload must drop TM replan arming and snapshot."""
        from flextensor.config import OffloadConfig

        class _StubTM:
            def __init__(self) -> None:
                self._first_loader_non_destructive = False
                self._replan_source_data: dict = {}

            def arm_non_destructive_first_loader(self) -> None:
                self._first_loader_non_destructive = True

            def clear_replan_state(self) -> None:
                self._first_loader_non_destructive = False
                self._replan_source_data = {}

        om = OffloadManager("test_compiled_to_view_replan_clear")
        co = om._compiled
        tm = _StubTM()
        om._tensor_manager = tm

        external_config = OffloadConfig(enabled=True, external_compile=True)
        assert co.resolve_activation(external_config, compile_fn=None) is True
        co.arm_non_destructive_first_loader()
        assert tm._first_loader_non_destructive is True

        # Stale snapshot from a prior failed/no-op replan.
        tm._replan_source_data = {42: object()}

        view_config = OffloadConfig(enabled=True, profile_mode="view")
        assert co.resolve_activation(view_config, compile_fn=lambda m: m) is True
        assert tm._first_loader_non_destructive is False
        assert tm._replan_source_data == {}
        assert co.replan_active is False

        # View path must not re-arm non-destructive retention.
        co.arm_non_destructive_first_loader()
        assert tm._first_loader_non_destructive is False

        eager_config = OffloadConfig(enabled=True)
        tm._first_loader_non_destructive = True
        tm._replan_source_data = {7: object()}
        assert co.resolve_activation(eager_config, compile_fn=None) is False
        assert tm._first_loader_non_destructive is False
        assert tm._replan_source_data == {}


class TestTrapInferDirectEager:
    @patch("flextensor.trap_tensor_mode._graph_break")
    def test_trap_calls_loader_enter_exit(self, _mock_graph_break: MagicMock) -> None:
        from flextensor.trap_tensor_mode import TrapInferDirect

        loader = _RecordingLoader()
        tm = SimpleNamespace(tensor_layer_loader=loader)
        trap = TrapInferDirect(tm, "layer0", torch.device("cpu"))

        with trap:
            pass

        assert loader.entered == ["layer0"]
        assert loader.exited == ["layer0"]


class TestMultiManagerCompiledOffload:
    def test_two_managers_dispatch_to_independent_loaders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FT_EXTERNAL_COMPILE", "1")
        model_a = SimpleLayer(4)
        model_b = SimpleLayer(4)
        om_a = OffloadManager("transformer")
        om_b = OffloadManager("transformer2")
        for om, model, label in (
            (om_a, model_a, "blocks.0"),
            (om_b, model_b, "blocks.0"),
        ):
            om._compiled.active = True
            om._current_phase = OffloadPhase.INFERENCE  # noqa: SLF001
            om._patch_module_forward(model, label)
            om._install_compiled_forwards()  # noqa: SLF001

        loader_a = _RecordingLoader()
        loader_b = _RecordingLoader()
        custom_ops.install_active_loader(loader_a, om_a.compiled_offload_manager_id)
        custom_ops.install_active_loader(loader_b, om_b.compiled_offload_manager_id)

        x = torch.randn(2, 4)
        model_a(x)
        model_b(x)

        assert loader_a.entered == ["blocks.0"]
        assert loader_a.exited == ["blocks.0"]
        assert loader_b.entered == ["blocks.0"]
        assert loader_b.exited == ["blocks.0"]
        om_a.release()
        om_b.release()


class TestInferenceTransitionCompiledSetup:
    def test_transition_calls_forwards_then_external_setup_in_order(self) -> None:
        om = OffloadManager("test_inference_order")
        om._compiled.active = True
        om._compiled.compile_fn = None
        om._tensor_manager = MagicMock()  # noqa: SLF001
        om._tensor_manager.initialize_inference.return_value = nn.Identity()

        calls: list[str] = []

        def _forwards() -> None:
            calls.append("forwards")

        def _external() -> None:
            calls.append("external")

        om._install_compiled_forwards = _forwards  # type: ignore[method-assign]  # noqa: SLF001
        om._compiled.setup_external_compiled_offload = _external  # type: ignore[method-assign]
        om.config.enable_instrumentation = False

        with patch.object(om, "_swap_to_new_model"):
            om._transition_to_inference()  # noqa: SLF001

        assert calls == ["forwards", "external"]

    def test_setup_compiled_tail_installs_loader_and_applies_compile_fn(self) -> None:
        om = OffloadManager("test_compiled_tail")
        om._compiled.active = True
        om._compiled.replan_active = False
        om._compiled.compile_fn = lambda m: m

        install_calls: list[str] = []
        apply_calls: list[str] = []

        om._compiled.require_compiled_loader = lambda: install_calls.append("loader")  # type: ignore[method-assign]
        om._compiled.apply_compile_fn = lambda: apply_calls.append("compile")  # type: ignore[method-assign]

        om._compiled.setup_compiled_tail()

        assert install_calls == ["loader"]
        assert apply_calls == ["compile"]
        assert om._compiled.tail_state == CompiledOffloadTailState.DONE
        om.release()


class TestReinstallCompiledLoader:
    def test_replaces_active_loader(self) -> None:
        om = OffloadManager("test_reinstall_loader")
        model = ModelWithLayers()
        om._patch_module_forward(model.layer1, "layer1")
        om._patch_module_forward(model.layer2, "layer2")

        new_loader = _RecordingLoader()
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=new_loader)  # noqa: SLF001

        stale_loader = _RecordingLoader()
        mid = om.compiled_offload_manager_id
        custom_ops.install_active_loader(stale_loader, mid)

        om._compiled.reinstall_compiled_loader()

        assert custom_ops.get_active_loader(mid) is new_loader
        custom_ops.clear_active_loader(mid)

    def test_raises_when_rebuilt_loader_missing(self) -> None:
        om = OffloadManager("test_reinstall_loader_missing")
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=None)  # noqa: SLF001
        with pytest.raises(RuntimeError, match="no loader after re-plan"):
            om._compiled.reinstall_compiled_loader()
        assert custom_ops.get_active_loader(om.compiled_offload_manager_id) is None

    def test_clears_stale_loader_when_rebuilt_loader_missing(self) -> None:
        om = OffloadManager("test_reinstall_loader_stale")
        om._tensor_manager = SimpleNamespace(tensor_layer_loader=None)  # noqa: SLF001
        mid = om.compiled_offload_manager_id
        stale_loader = _RecordingLoader()
        custom_ops.install_active_loader(stale_loader, mid)

        with pytest.raises(RuntimeError, match="no loader after re-plan"):
            om._compiled.reinstall_compiled_loader()
        assert custom_ops.get_active_loader(mid) is None
