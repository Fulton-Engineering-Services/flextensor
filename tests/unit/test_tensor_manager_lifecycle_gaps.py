# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for TensorManager lifecycle seams that no test previously pinned.

Each of these guards a behaviour that could be deleted outright without any
existing test noticing:

* ``prepare_infer_mode`` shutting the profile-time loader down before the
  inference loader allocates — the only release point for
  ``UntimedTrapRescuer``'s owned GPU copies.
* ``_move_non_offloaded_tensors_to_gpu`` tolerating ``extra_pin_ids`` that are
  absent from ``tensors_map`` (documented as "silently skipped").
* ``device_gpu`` normalization from ``str`` to ``torch.device``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch import nn

from flextensor.state_handler import LoaderInputData
from flextensor.tensor_manager import TensorManager


def _tensor_manager(device_gpu: str | torch.device | None = None) -> TensorManager:
    if device_gpu is None:
        device_gpu = torch.device("cpu")
    return TensorManager(device_gpu=device_gpu, tensor_manager_load_strategy=MagicMock(), pinned_memory=False)


class TestDeviceGpuNormalization:
    """``device_gpu`` is typed ``str | torch.device``; downstream wants a device.

    ``loaders`` and ``gpu_budget`` read ``.type`` and hand it to ``torch`` APIs,
    so the ``str`` form would crash there. Normalizing at the boundary means
    every consumer sees exactly one type.
    """

    def test_string_device_is_normalized(self) -> None:
        tm = _tensor_manager("cpu")

        assert isinstance(tm.device_gpu, torch.device)
        assert tm.device_gpu == torch.device("cpu")

    def test_device_instance_is_passed_through(self) -> None:
        device = torch.device("cpu")
        tm = _tensor_manager(device)

        assert tm.device_gpu is device


class TestCreateLoaderShutsDownProfileLoader:
    """The outgoing loader must shut down before the inference loader builds.

    ``_create_loader`` is the seam: it is reached from every
    ``prepare_infer_mode`` path and holds the only release point for the
    rescuer's owned GPU copies. Deleting those three lines leaks the full
    rescue set for the process lifetime, silently.
    """

    def test_outgoing_loader_is_shut_down_and_cleared(self) -> None:
        tm = _tensor_manager()
        outgoing = MagicMock()
        tm.tensor_layer_loader = outgoing
        tm.loader_type = "strategy"
        tm._setup_strategy_loader = MagicMock()  # isolate the release from loader construction

        tm._create_loader(LoaderInputData())

        outgoing.shutdown.assert_called_once()
        assert tm.tensor_layer_loader is None, "the released loader must not stay installed"

    def test_shutdown_happens_before_the_new_loader_is_built(self) -> None:
        """Ordering matters: the new loader allocates against the freed budget."""
        tm = _tensor_manager()
        outgoing = MagicMock()
        tm.tensor_layer_loader = outgoing
        tm.loader_type = "strategy"
        order: list[str] = []
        outgoing.shutdown.side_effect = lambda: order.append("shutdown")
        tm._setup_strategy_loader = MagicMock(side_effect=lambda *a, **k: order.append("build"))

        tm._create_loader(LoaderInputData())

        assert order == ["shutdown", "build"]

    def test_absent_loader_is_tolerated(self) -> None:
        tm = _tensor_manager()
        tm.tensor_layer_loader = None
        tm.loader_type = "strategy"
        tm._setup_strategy_loader = MagicMock()

        tm._create_loader(LoaderInputData())

        assert tm.tensor_layer_loader is None


class TestExtraPinIdsToleratesUnknownIds:
    """``extra_pin_ids`` is sourced from a map that may have already shrunk.

    The docstring promises absent ids are "silently skipped"; without that a
    stale id would raise ``KeyError`` from the pop loop and abort offload setup.
    """

    def _tensor_manager_with_model(self) -> TensorManager:
        tm = _tensor_manager()
        model = nn.Linear(4, 4)
        tm.model = model
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map)
        return tm

    def test_unknown_id_is_skipped_not_raised(self) -> None:
        tm = self._tensor_manager_with_model()
        bogus_id = 424242

        tm._move_non_offloaded_tensors_to_gpu(extra_pin_ids={bogus_id})

        assert bogus_id not in tm.tensors_map

    def test_known_id_is_still_pinned(self) -> None:
        tm = self._tensor_manager_with_model()
        pinned_id = next(iter(tm.tensors_map))

        tm._move_non_offloaded_tensors_to_gpu(extra_pin_ids={pinned_id, 424242})

        assert pinned_id not in tm.tensors_map, "a pinned id must leave offload tracking"
        assert pinned_id not in tm.traced_tensors


class _TwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class TestModelScopedPreprocessingResetsPerCycle:
    """include/exclude placement must be applied to *every* model offloaded.

    ``_non_offloaded_tensors_moved`` guards ``_move_non_offloaded_tensors_to_gpu``
    so it runs once per model. It was set on the first cycle and never reset, so
    a second ``offload()`` returned immediately and never moved the second
    model's non-matching parameters to GPU — with discovery enabled those stay
    on CPU and fail the first GPU forward.
    """

    def _manager(self) -> TensorManager:
        return TensorManager(
            device_gpu=torch.device("cpu"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            include_patterns=["layer1"],
        )

    def _run_cycle(self, tm: TensorManager, model: nn.Module) -> set[int]:
        """Drive one model through the per-cycle preprocessing seam."""
        tm.set_model(model)
        tm.tensors_map = {id(p): p for p in model.parameters()}
        tm.traced_tensors = set(tm.tensors_map)
        tm._move_non_offloaded_tensors_to_gpu()
        return set(tm.tensors_map)

    def test_first_cycle_retains_only_included_layer(self) -> None:
        tm = self._manager()
        model = _TwoLayerModel()

        retained = self._run_cycle(tm, model)

        assert retained == {id(p) for p in model.layer1.parameters()}

    def test_second_cycle_applies_placement_to_the_new_model(self) -> None:
        tm = self._manager()
        first, second = _TwoLayerModel(), _TwoLayerModel()

        self._run_cycle(tm, first)
        retained = self._run_cycle(tm, second)

        assert retained == {id(p) for p in second.layer1.parameters()}, (
            "the second model's non-matching parameters were never moved to GPU — "
            "the once-per-model guard was not reset for the new model"
        )
