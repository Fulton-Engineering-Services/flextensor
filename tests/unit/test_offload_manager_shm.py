# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OffloadManager SHM follower initialization path and block naming."""

import inspect
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from flextensor.config import OffloadConfig
from flextensor.host_pinning import HostPinner
from flextensor.loaders import AllocationBlockController
from flextensor.offload_manager import OffloadManager, OffloadModelProxy, OffloadPhase


class _FakeCoordinator:
    """Minimal stub satisfying _ShmCoordinatorLike protocol for unit tests.

    Uses concrete instance attributes so beartype's runtime protocol check passes
    even on Python 3.13, where plain MagicMock is rejected.
    """

    def __init__(self, *, is_creator: bool = False, namespace: str = "") -> None:
        self.is_creator = is_creator
        self.namespace = namespace
        self.wait_for_ready = MagicMock()
        self.read_profile = MagicMock(return_value=MagicMock())


class TestInitializeFromShm:
    """Tests for _initialize_from_shm method on OffloadManager."""

    def test_method_exists(self):
        """OffloadManager has _initialize_from_shm method."""
        assert hasattr(OffloadManager, "_initialize_from_shm")
        assert callable(OffloadManager._initialize_from_shm)

    def test_method_signature(self):
        """_initialize_from_shm accepts coordinator and model parameters."""
        sig = inspect.signature(OffloadManager._initialize_from_shm)
        params = list(sig.parameters.keys())
        assert "coordinator" in params
        assert "model" in params

    def test_rejects_creator_coordinator(self):
        """_initialize_from_shm raises RuntimeError if coordinator is creator."""
        manager = OffloadManager("test_shm")
        coordinator = _FakeCoordinator(is_creator=True)
        model = torch.nn.Linear(4, 4)

        with pytest.raises(RuntimeError, match="creator"):
            manager._initialize_from_shm(coordinator, model)

    def test_sets_shm_namespace_on_tensor_manager(self):
        """Follower path propagates coordinator namespace to tensor manager."""
        manager = OffloadManager("test_shm_ns")
        coordinator = _FakeCoordinator(is_creator=False, namespace="ft_abc123_tp0_pp0")
        model = torch.nn.Linear(4, 4)

        mock_tm = MagicMock()
        mock_tm.shm_namespace = None

        def fake_init_tm():
            manager._tensor_manager = mock_tm

        manager._initialize_tensor_manager = fake_init_tm
        mock_tm.restore_state.side_effect = RuntimeError("stop here")

        with pytest.raises(RuntimeError, match="stop here"):
            manager._initialize_from_shm(coordinator, model)

        assert mock_tm.shm_namespace == "ft_abc123_tp0_pp0"

    def test_happy_path_returns_proxy_in_inference(self):
        """Full follower init sequence returns OffloadModelProxy in INFERENCE state."""
        manager = OffloadManager("test_shm_happy")
        coordinator = _FakeCoordinator(is_creator=False, namespace="ft_ns_happy")
        model = torch.nn.Linear(4, 4)

        mock_tm = MagicMock()
        mock_tm.shm_namespace = None
        mock_tm.initialize_warmup.return_value = model
        mock_tm.initialize_profile.return_value = model
        mock_tm.initialize_inference.return_value = model

        def fake_init_tm():
            manager._tensor_manager = mock_tm

        manager._initialize_tensor_manager = fake_init_tm

        # Patch _offload_modules and _exclude_modules to no-op (avoids real module patching)
        with (
            patch.object(manager, "_offload_modules") as mock_offload,
            patch.object(manager, "_exclude_modules") as mock_exclude,
        ):
            result = manager._initialize_from_shm(coordinator, model)

        assert isinstance(result, OffloadModelProxy)
        assert manager._current_phase == OffloadPhase.INFERENCE
        coordinator.wait_for_ready.assert_called_once()
        coordinator.read_profile.assert_called_once()
        mock_tm.restore_state.assert_called_once()
        mock_tm.initialize_warmup.assert_called_once()
        mock_tm.initialize_profile.assert_called_once()
        mock_tm.initialize_inference.assert_called_once()
        mock_offload.assert_called_once_with(model, manager.config.include_patterns)
        mock_exclude.assert_called_once_with(model, manager.config.exclude_patterns)
        # Follower installs the state-update hook for symmetry with offload();
        # update_state is a no-op in INFERENCE, but release() needs a handle
        # to uniformly clean up regardless of entry path.
        assert manager._model is model, "hook must be installed on the same model the test owns"
        assert manager._state_hook_handle is not None
        tagged = [h for h in model._forward_hooks.values() if getattr(h, "_ft_state_update_hook", False)]
        assert len(tagged) == 1, f"expected exactly one tagged state-update hook on the model, found {len(tagged)}"

    def test_follower_installs_compiled_loader_when_compiled_offload_active(self):
        """SHM followers must register the rolling loader, not just compile forwards."""
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

        manager = OffloadManager("test_shm_compiled_loader")
        manager.set_config(OffloadConfig(external_compile=True))
        coordinator = _FakeCoordinator(is_creator=False, namespace="ft_ns_compiled")
        model = torch.nn.Linear(4, 4)

        loader = _RecordingLoader()
        mock_tm = MagicMock()
        mock_tm.shm_namespace = None
        mock_tm.tensor_layer_loader = loader
        mock_tm.initialize_warmup.return_value = model
        mock_tm.initialize_profile.return_value = model
        mock_tm.initialize_inference.return_value = model

        def fake_init_tm():
            manager._tensor_manager = mock_tm

        manager._initialize_tensor_manager = fake_init_tm

        custom_ops.clear_active_loader()
        try:
            with (
                patch.object(manager, "_offload_modules"),
                patch.object(manager, "_exclude_modules"),
                patch.object(manager, "_patch_module_forward"),
            ):
                manager._initialize_from_shm(coordinator, model)

            mid = manager.compiled_offload_manager_id
            assert custom_ops.get_active_loader(mid) is loader
        finally:
            custom_ops.clear_active_loader()

    def test_follower_resolves_compiled_offload_from_config(self):
        """Follower init must read compiled_offload from config, not require manual flags."""
        manager = OffloadManager("test_shm_config_resolve")
        manager.set_config(OffloadConfig(external_compile=True))
        coordinator = _FakeCoordinator(is_creator=False, namespace="ft_ns_resolve")
        model = torch.nn.Linear(4, 4)

        mock_tm = MagicMock()
        mock_tm.shm_namespace = None
        mock_tm.initialize_warmup.return_value = model
        mock_tm.initialize_profile.return_value = model
        mock_tm.initialize_inference.return_value = model

        def fake_init_tm():
            manager._tensor_manager = mock_tm

        manager._initialize_tensor_manager = fake_init_tm

        with (
            patch.object(manager, "_offload_modules"),
            patch.object(manager, "_exclude_modules"),
            patch.object(manager, "_patch_module_forward"),
            patch.object(manager._compiled, "setup_external_compiled_offload") as mock_setup,
        ):
            manager._initialize_from_shm(coordinator, model)

        assert manager.compiled_offload_active is True
        mock_setup.assert_called_once()

    def test_follower_external_compile_arms_non_destructive_first_loader(self):
        """SHM + external_compile must preserve sources for a later replan."""
        manager = OffloadManager("test_shm_arm_replan_sources")
        manager.set_config(OffloadConfig(external_compile=True))
        coordinator = _FakeCoordinator(is_creator=False, namespace="ft_ns_arm")
        model = torch.nn.Linear(4, 4)

        mock_tm = MagicMock()
        mock_tm.shm_namespace = None
        mock_tm._first_loader_non_destructive = False
        mock_tm.arm_non_destructive_first_loader.side_effect = lambda: setattr(
            mock_tm, "_first_loader_non_destructive", True
        )
        mock_tm.initialize_warmup.return_value = model
        mock_tm.initialize_profile.return_value = model
        mock_tm.initialize_inference.return_value = model

        def fake_init_tm():
            manager._tensor_manager = mock_tm

        manager._initialize_tensor_manager = fake_init_tm

        with (
            patch.object(manager, "_offload_modules"),
            patch.object(manager, "_exclude_modules"),
            patch.object(manager, "_patch_module_forward"),
            patch.object(manager._compiled, "setup_external_compiled_offload"),
        ):
            manager._initialize_from_shm(coordinator, model)

        assert mock_tm._first_loader_non_destructive is True
        assert manager.compiled_replan_active is True


class TestBlockNameFn:
    """Tests for block_name_fn callable injection in AllocationBlockController."""

    def test_default_uses_pid_based_names(self):
        """Without block_name_fn, defaults to PID-based naming."""
        ctrl = AllocationBlockController(
            allocation_ordered={},
            device_gpu=torch.device("cpu"),
            tensors_map={},
            strategy_map={},
            label_to_block_id={},
            use_shm=True,
            host_pinner=HostPinner(),
        )
        expected = f"ft_{os.getpid()}_0"
        assert ctrl._block_name_fn(0) == expected

    def test_custom_callable_used(self):
        """Custom block_name_fn is called for block naming."""

        def custom_fn(index: int) -> str:
            return f"custom_block_{index}"

        ctrl = AllocationBlockController(
            allocation_ordered={},
            device_gpu=torch.device("cpu"),
            tensors_map={},
            strategy_map={},
            label_to_block_id={},
            use_shm=True,
            host_pinner=HostPinner(),
            block_name_fn=custom_fn,
        )
        assert ctrl._block_name_fn(0) == "custom_block_0"
        assert ctrl._block_name_fn(5) == "custom_block_5"

    def test_namespace_callable_produces_deterministic_names(self):
        """Namespace-derived callable produces deterministic, non-PID names."""
        from flextensor.shm.namespace import weight_block_name

        ns = "ft_abc123_tp0_pp0"

        def ns_fn(index: int) -> str:
            return weight_block_name(ns, index)

        ctrl = AllocationBlockController(
            allocation_ordered={},
            device_gpu=torch.device("cpu"),
            tensors_map={},
            strategy_map={},
            label_to_block_id={},
            use_shm=True,
            host_pinner=HostPinner(),
            block_name_fn=ns_fn,
        )
        name = ctrl._block_name_fn(0)
        assert name == f"{ns}_w0"
