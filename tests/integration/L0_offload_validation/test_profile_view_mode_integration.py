# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation of ``profile_mode='view'``.

The view-mode profile path replaces the property-getter profile model with
a model patched to use views into a single rotating GPU block (plus a fixed
prefix for tensors shared across labels). The tests in this file validate
two end-to-end properties:

1. **Functional equivalence with the legacy ``getter`` profile mode.**
   Running the same model and inputs through warmup → profile → inference
   under ``profile_mode='view'`` produces the same final output (within
   floating-point tolerance) as the same flow under ``profile_mode='getter'``.

2. **Profile teardown actually frees its blocks before inference.** The
   profile-time GPU footprint is bounded at ``shared_size + max_layer_size``
   and the controller's blocks are released before inference allocates its
   own per-strategy blocks. We assert the controller is dropped and that no
   model parameter still points at the (now-empty) profile block storages.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from flextensor import TensorManager
from flextensor.profile_block_controller import ProfileBlockController
from flextensor.strategy import KnapsackStrategy

# CI runner gating per ``tests/CLAUDE.md``. The test models are small (a 4-input,
# 8-hidden MLP), so any CUDA runner is sufficient — ``min_24g`` is the smallest
# documented bracket and lets CI dispatch onto any of the GPU runner pools.
pytestmark = pytest.mark.gpu_vram_min_24g


@pytest.fixture
def device_gpu() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


class _SmallModel(nn.Module):
    """Tiny model with explicit ``tensor_manager.trap`` per layer.

    Each layer has its own weights, so there are no tensors shared across
    labels — the rotating block does all the work. We add a tied buffer used
    by every layer to also exercise the shared-prefix path.
    """

    def __init__(self, tensor_manager: TensorManager, num_layers: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.tensor_manager = tensor_manager
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(num_layers)])
        # Buffer referenced by every label — this is what makes it ``shared``.
        self.register_buffer("scale", torch.full((hidden,), 1.0 / hidden), persistent=True)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        for i, layer in enumerate(self.layers):
            with self.tensor_manager.trap(f"layer_{i}"):
                x = layer(x) * self.scale
        return x


def _make_manager(
    device_gpu: torch.device,
    *,
    profile_mode: str,
    loader_type: str = "allocation_block_transfer",
) -> TensorManager:
    return TensorManager(
        device_gpu=device_gpu,
        tensor_manager_load_strategy=KnapsackStrategy(scale=0.8),
        loader_type=loader_type,
        blocks=2,
        profile_mode=profile_mode,
    )


def _run_full_pipeline(
    tensor_manager: TensorManager,
    model: nn.Module,
    x: torch.Tensor,
    *,
    profiling_iters: int = 2,
) -> torch.Tensor:
    """Run warmup → profile → inference and return the inference output."""
    tensor_manager.set_model(model)

    with torch.no_grad():
        live = tensor_manager.initialize_warmup()
        _ = live(x)

        live = tensor_manager.initialize_profile()
        for _ in range(profiling_iters):
            _ = live(x)

        live = tensor_manager.initialize_inference()
        result = live(x)

    return result


class TestProfileViewModeEquivalence:
    """``profile_mode='view'`` must produce the same inference output as ``'getter'``."""

    @pytest.mark.parametrize("num_layers", [2, 4])
    @pytest.mark.parametrize(
        "loader_type",
        ["allocation_block_transfer", "raw_block_transfer", "strategy"],
    )
    def test_view_matches_direct_inference_output(
        self,
        device_gpu: torch.device,
        num_layers: int,
        loader_type: str,
    ) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 128, device=device_gpu)

        # Direct (getter) profile path.
        torch.manual_seed(0)
        tm_direct = _make_manager(device_gpu, profile_mode="getter", loader_type=loader_type)
        model_direct = _SmallModel(tm_direct, num_layers=num_layers, hidden=128).cpu().eval()
        out_direct = _run_full_pipeline(tm_direct, model_direct, x)

        # View profile path on a freshly-seeded model with identical weights.
        torch.manual_seed(0)
        tm_view = _make_manager(device_gpu, profile_mode="view", loader_type=loader_type)
        model_view = _SmallModel(tm_view, num_layers=num_layers, hidden=128).cpu().eval()
        out_view = _run_full_pipeline(tm_view, model_view, x)

        assert out_direct.shape == out_view.shape
        torch.testing.assert_close(out_direct, out_view, rtol=1e-5, atol=1e-5)


class TestProfileViewModeTeardown:
    """View-mode profile blocks must be released before inference allocates its own."""

    def test_profile_block_controller_dropped_after_inference_init(
        self,
        device_gpu: torch.device,
    ) -> None:
        torch.manual_seed(0)
        tm = _make_manager(device_gpu, profile_mode="view")
        model = _SmallModel(tm, num_layers=3, hidden=64).cpu().eval()
        x = torch.randn(2, 64, device=device_gpu)

        tm.set_model(model)

        with torch.no_grad():
            live = tm.initialize_warmup()
            _ = live(x)

            live = tm.initialize_profile()
            assert isinstance(tm.tensor_layer_loader, ProfileBlockController)
            controller_ref = tm.tensor_layer_loader
            assert controller_ref.block_size > 0
            _ = live(x)

            live = tm.initialize_inference()

        # After inference setup the controller has been torn down, its blocks
        # released back to the caching allocator, and the manager no longer
        # holds a reference.
        assert not isinstance(tm.tensor_layer_loader, ProfileBlockController)
        assert controller_ref.get_gpu_memory_bytes() == 0
        assert controller_ref.get_tensor_id_to_view_mapping() == {}

    def test_inference_runs_after_view_profile_teardown(
        self,
        device_gpu: torch.device,
    ) -> None:
        torch.manual_seed(0)
        tm = _make_manager(device_gpu, profile_mode="view")
        model = _SmallModel(tm, num_layers=2, hidden=64).cpu().eval()
        x = torch.randn(2, 64, device=device_gpu)

        out = _run_full_pipeline(tm, model, x, profiling_iters=2)

        assert out.is_cuda
        assert out.shape == (2, 64)


def _make_dict_model(num_layers: int, hidden: int) -> dict[str, torch.Tensor]:
    """A ``dict[str, torch.Tensor]`` "model": just weights, no Modules. The
    forward callable below threads the dict through manual traps. Plain
    ``Tensor`` (not ``nn.Parameter``) on purpose -- this is the shape that
    exercises ``TensorReplacementProcessor``'s non-Parameter branch.
    """
    return {f"layer_{i}_weight": torch.randn(hidden, hidden) for i in range(num_layers)}


def _forward_dict_model(
    model_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    tensor_manager: TensorManager,
    num_layers: int,
) -> torch.Tensor:
    """Forward pass that reads weights through dict keys directly.

    No explicit ``tensor_layer_loader.get(...)`` here -- view mode patches the
    profile-phase dict in place with views, and inference patches the
    original dict via ``prepare_view_model`` for block-transfer loaders. The
    point of this fixture is to prove plain dict-key access works through
    the full lifecycle under ``profile_mode='view'``.
    """
    for i in range(num_layers):
        with tensor_manager.trap(f"layer_{i}"):
            w = model_dict[f"layer_{i}_weight"]
            if w.device != x.device:
                w = w.to(x.device)
            x = x @ w
    return x


def _run_dict_pipeline(
    tensor_manager: TensorManager,
    model_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    num_layers: int,
    profiling_iters: int = 2,
) -> torch.Tensor:
    tensor_manager.set_model(model_dict)

    with torch.no_grad():
        live = tensor_manager.initialize_warmup()
        _ = _forward_dict_model(live, x, tensor_manager, num_layers)

        live = tensor_manager.initialize_profile()
        for _ in range(profiling_iters):
            _ = _forward_dict_model(live, x, tensor_manager, num_layers)

        live = tensor_manager.initialize_inference()
        return _forward_dict_model(live, x, tensor_manager, num_layers)


class TestProfileViewModeDictModel:
    """End-to-end coverage of ``profile_mode='view'`` with a
    ``dict[str, torch.Tensor]`` model. The dispatch unit test mocks
    ``_prepare_view_profile_model``; this test exercises the real path:
    dict shallow copy, ``MoveUnmappedTensorsToGPUProcessor`` +
    ``TensorReplacementProcessor`` over the dict, view-patched
    profile-time access, and teardown back into the original dict.
    """

    def test_view_matches_direct_inference_output_dict_model(
        self,
        device_gpu: torch.device,
    ) -> None:
        num_layers = 3
        hidden = 64
        x = torch.randn(2, hidden, device=device_gpu)

        torch.manual_seed(0)
        tm_direct = _make_manager(device_gpu, profile_mode="getter")
        torch.manual_seed(1)
        model_direct = _make_dict_model(num_layers, hidden)
        out_direct = _run_dict_pipeline(tm_direct, model_direct, x, num_layers=num_layers)

        torch.manual_seed(0)
        tm_view = _make_manager(device_gpu, profile_mode="view")
        torch.manual_seed(1)
        model_view = _make_dict_model(num_layers, hidden)
        out_view = _run_dict_pipeline(tm_view, model_view, x, num_layers=num_layers)

        assert out_view.is_cuda
        assert out_direct.shape == out_view.shape
        torch.testing.assert_close(out_direct, out_view, rtol=1e-5, atol=1e-5)

    def test_view_mode_dict_drops_controller_after_inference(
        self,
        device_gpu: torch.device,
    ) -> None:
        torch.manual_seed(2)
        tm = _make_manager(device_gpu, profile_mode="view")
        model_dict = _make_dict_model(num_layers=3, hidden=64)
        x = torch.randn(2, 64, device=device_gpu)

        out = _run_dict_pipeline(tm, model_dict, x, num_layers=3)

        # Controller must be torn down before inference begins; without this
        # the rotating block stays alive across inference and ``set_model``
        # ownership semantics break for dict-shaped reuse.
        assert not isinstance(tm.tensor_layer_loader, ProfileBlockController)
        assert out.shape == (2, 64)

    def test_view_mode_dict_does_not_pin_block_after_teardown(
        self,
        device_gpu: torch.device,
    ) -> None:
        """Regression: the patched profile dict must not keep block views alive.

        For dict-shaped models the profile dict is a shallow copy whose entries
        are replaced wholesale with controller views (plain ``Tensor``, not
        ``nn.Parameter``). Teardown only restores ``.data`` on Parameters, so
        without an explicit container rewrite the returned dict keeps aliasing
        the GPU block and pins its storage past the profile phase.
        """
        torch.manual_seed(3)
        tm = _make_manager(device_gpu, profile_mode="view")
        model_dict = _make_dict_model(num_layers=3, hidden=64)
        keys = [f"layer_{i}_weight" for i in range(3)]
        x = torch.randn(2, 64, device=device_gpu)

        tm.set_model(model_dict)
        with torch.no_grad():
            live = tm.initialize_warmup()
            _ = _forward_dict_model(live, x, tm, num_layers=3)

            profile_dict = tm.initialize_profile()
            # After patching, each entry is a profile-block view.
            profile_views = {key: profile_dict[key] for key in keys}
            for _ in range(2):
                _ = _forward_dict_model(profile_dict, x, tm, num_layers=3)

            # Teardown happens as part of inference setup.
            tm.initialize_inference()

        # Each entry must be rewritten away from its block view and back to a
        # tensor tracked in ``tensors_map`` (block views are never tracked there).
        tracked = {id(t) for t in tm.tensors_map.values()}
        for key in keys:
            assert profile_dict[key] is not profile_views[key]
            assert id(profile_dict[key]) in tracked
