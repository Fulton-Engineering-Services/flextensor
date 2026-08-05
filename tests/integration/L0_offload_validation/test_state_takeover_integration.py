# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic end-to-end coverage for saved-state takeover."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional

import flextensor as ft
from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.state_handler import TensorManagerState


class _SplitTakeoverModel(nn.Module):
    def __init__(self, promote_device: torch.device, demote_device: torch.device, constant_device: torch.device):
        super().__init__()
        self.promote = nn.Linear(8, 12, bias=False, device=promote_device)
        self.demote = nn.Linear(12, 4, bias=False, device=demote_device)
        self.permanent = nn.Parameter(torch.linspace(0.75, 1.25, 4, device=constant_device), requires_grad=False)
        self.register_buffer("constant", torch.linspace(-0.2, 0.2, 4, device=constant_device))
        with torch.no_grad():
            self.promote.weight.copy_(self._values(self.promote.weight, 11))
            self.demote.weight.copy_(self._values(self.demote.weight, 7))

    @staticmethod
    def _values(tensor: torch.Tensor, modulus: int) -> torch.Tensor:
        values = torch.arange(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        return ((values.remainder(modulus) - modulus // 2) / tensor.shape[-1]).view_as(tensor)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.demote(self.promote(inputs)) * self.permanent + self.constant


def _named_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(model.named_buffers(remove_duplicate=False))
    return tensors


def _strategy_state(model: nn.Module) -> TensorManagerState:
    tensors = _named_tensors(model)
    managed = tensors["demote.weight"]
    stat = TensorStatistics(
        tensor_id=id(managed),
        name="demote.weight",
        size_bytes=managed.numel() * managed.element_size(),
        load_time_ms=0.1,
    )
    return TensorManagerState(
        loader_type="strategy",
        tensor_id_to_name_map={id(tensor): name for name, tensor in tensors.items()},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"demote": [stat]},
        release_strategy={"demote": [stat]},
        label_to_block_id={},
        stats=[LayerStatistics(label="demote", tensors=[stat], duration=1.0)],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=["promote.weight", "permanent", "constant"],
        shm_block_name_map=None,
    )


def _allocation_block_state(model: nn.Module) -> TensorManagerState:
    state = _strategy_state(model)
    managed = _named_tensors(model)["demote.weight"]
    size_bytes = managed.numel() * managed.element_size()
    state.loader_type = "allocation_block_transfer"
    state.allocation_ordered = {0: ["demote"]}
    state.label_to_size_map = {"demote": size_bytes}
    state.block_sizes = {0: managed.untyped_storage().nbytes()}
    state.label_to_block_id = {"demote": 0}
    state.transfer_to_compute_map = {"demote": "demote"}
    state.release_strategy = {}
    state.view_tensors_ids = [id(managed)]
    state.view_tensors_names = ["demote.weight"]
    return state


def _raw_block_state(model: nn.Module) -> TensorManagerState:
    state = _allocation_block_state(model)
    state.loader_type = "raw_block_transfer"
    return state


def test_disabled_takeover_uses_public_proxy_in_place_on_cpu() -> None:
    manager_name = "cpu-noop-state-takeover"
    model = _SplitTakeoverModel(torch.device("cpu"), torch.device("cpu"), torch.device("cpu")).eval()
    state = _strategy_state(model)
    module_ids = {name: id(module) for name, module in model.named_modules()}
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    inputs = torch.linspace(-1.0, 1.0, 16).view(2, 8)
    with torch.no_grad():
        expected = model(inputs).clone()

    try:
        final_model = ft.offload_from_state(
            model,
            state,
            config=ft.OffloadConfig(enabled=False, include_patterns=["demote"]),
            name=manager_name,
        )

        assert final_model.__subject__ is model
        assert {name: id(module) for name, module in final_model.named_modules()} == module_ids
        assert {name: id(parameter) for name, parameter in final_model.named_parameters()} == parameter_ids
        with torch.no_grad():
            output = final_model(inputs)
        torch.testing.assert_close(output, expected)
    finally:
        ft.release(name=manager_name)


@pytest.mark.gpu_vram_min_24g
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("loader_type", "state_factory"),
    [
        pytest.param("strategy", _strategy_state, id="strategy"),
        pytest.param("allocation_block_transfer", _allocation_block_state, id="allocation-block"),
        pytest.param("raw_block_transfer", _raw_block_state, id="raw-block"),
    ],
)
def test_takeover_moves_split_homes_and_finalizes_existing_model(loader_type, state_factory) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    manager_name = f"split-state-takeover-{loader_type}"
    model = _SplitTakeoverModel(torch.device("cpu"), device, device).eval()
    state = state_factory(model)
    module_ids = {name: id(module) for name, module in model.named_modules()}
    module_classes = {id(module): type(module) for module in model.modules()}
    parameter_ids = {name: id(parameter) for name, parameter in model.named_parameters()}
    inputs = torch.linspace(-1.0, 1.0, 16, device=device).view(2, 8)
    tensors_before = {name: tensor.detach().cpu().clone() for name, tensor in _named_tensors(model).items()}
    loader = None
    with torch.no_grad():
        expected = functional.linear(
            functional.linear(inputs, tensors_before["promote.weight"].to(device)),
            tensors_before["demote.weight"].to(device),
        )
        expected = expected * tensors_before["permanent"].to(device) + tensors_before["constant"].to(device)

    try:
        final_model = ft.offload_from_state(
            model,
            state,
            config=ft.OffloadConfig(
                transfer_mode=loader_type,
                include_patterns=["demote"],
                pinned_memory=False,
                profile_mode="getter",
                gpu_device=device.index or 0,
            ),
            name=manager_name,
        )

        assert final_model.__subject__ is model
        assert {name: id(module) for name, module in final_model.named_modules()} == module_ids
        assert {name: id(parameter) for name, parameter in final_model.named_parameters()} == parameter_ids
        named_parameters = dict(final_model.named_parameters())
        assert named_parameters["promote.weight"].is_cuda
        assert named_parameters["permanent"].is_cuda
        assert dict(final_model.named_buffers())["constant"].is_cuda
        manager = ft.get_offload_manager(manager_name)
        tensor_manager = manager.get_tensor_manager()
        assert tensor_manager is not None
        loader = tensor_manager.tensor_layer_loader
        if loader_type == "strategy":
            assert named_parameters["demote.weight"].device.type == "cpu"
        else:
            assert named_parameters["demote.weight"].is_cuda
            assert tensor_manager.tensor_layer_loader.allocation_controller.block_map_cpu
        with torch.no_grad():
            output = final_model(inputs)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output, expected)
        for name, tensor in _named_tensors(final_model).items():
            torch.testing.assert_close(tensor.detach().cpu(), tensors_before[name])

        with pytest.raises(RuntimeError, match="already active"):
            ft.offload_from_state(model, state, name=manager_name)
    finally:
        ft.release(name=manager_name)

    assert all(type(module) is module_classes[id(module)] for module in model.modules())
    assert not any("_ft_original_forward_func" in module.__dict__ for module in model.modules())
    assert not any(
        getattr(hook, "_ft_state_update_hook", False)
        for module in model.modules()
        for hook in module._forward_hooks.values()
    )
    assert loader is not None
    if loader_type == "strategy":
        assert loader.cpu_to_gpu_map == {}
        assert loader.transfer_events == {}
    else:
        assert loader.allocation_controller.block_map_cpu == {}
        assert loader.allocation_controller.block_map_gpu == {}
