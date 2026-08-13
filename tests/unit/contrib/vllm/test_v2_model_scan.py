# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from flextensor.config import OffloadConfig

from ._v2_test_utils import (
    RecordingStateStrategy,
    SelectTensorStrategy,
    _assert_value_only,
    _scan_loaded_model,
    _set_cuda_snapshot,
    _state_offloader,
    _state_root,
)


class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 2, bias=False)


def _decoder_root(layer_count: int = 3) -> nn.Module:
    root = nn.Module()
    root.model = nn.Module()
    root.model.embed_tokens = nn.Embedding(8, 2)
    root.model.layers = nn.ModuleList(DecoderLayer() for _index in range(layer_count))
    root.model.norm = nn.LayerNorm(2)
    root.lm_head = nn.Linear(2, 8, bias=False)
    root.logits_processor = nn.Linear(2, 8, bias=False)
    return root


def test_build_state_uses_final_loaded_tensors(bootstrap_module, monkeypatch) -> None:
    root = _state_root()
    offloader = bootstrap_module.VllmBootstrapOffloader()
    offloader.wrap_modules(iter([root.first, root.second]))
    replacement = nn.Parameter(torch.ones(3, 3))
    root.first.weight = replacement
    offloader.post_init()
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=1024, total=2048)

    state = offloader.build_state(
        root,
        OffloadConfig(
            enabled=True,
            pinned_memory=False,
            load_strategy=SelectTensorStrategy({id(replacement)}),
            transfer_mode="strategy",
        ),
        "cuda:0",
    )

    assert id(replacement) in state.tensor_id_to_name_map
    replacement_stat = next(
        statistic for layer in state.stats for statistic in layer.tensors if statistic.tensor_id == id(replacement)
    )
    assert replacement_stat.size_bytes == 36
    assert offloader._state_built
    assert offloader._live_units == []


def test_build_state_does_not_move_or_pin_tensors(bootstrap_module, monkeypatch) -> None:
    offloader, root = _state_offloader(bootstrap_module, monkeypatch)
    strategy = RecordingStateStrategy("strategy")
    config = OffloadConfig(
        load_strategy=strategy,
        transfer_mode="strategy",
        pinned_memory=False,
    )
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    before = {
        name: (parameter.device, parameter.untyped_storage()._cdata) for name, parameter in root.named_parameters()
    }
    state = offloader.build_state(root, config, "cuda:0")
    after = {
        name: (parameter.device, parameter.untyped_storage()._cdata) for name, parameter in root.named_parameters()
    }

    assert after == before
    assert state.loader_type == config.transfer_mode


def test_final_scan_uses_post_transform_tensor_ids_and_keeps_unsafe_storage_resident(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)

    scan = _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda:0")

    assert [layer.label for layer in scan.layer_stats] == ["first", "second"]
    assert {stat.tensor_id for layer in scan.layer_stats for stat in layer.tensors} == {
        id(transformed.first.weight),
        id(transformed.second.weight),
    }
    assert {stat.load_time_ms for layer in scan.layer_stats for stat in layer.tensors} == {0.0}
    assert scan.name_by_tensor_id[id(transformed.first.weight)] == "first.weight"
    assert scan.name_by_tensor_id[id(transformed.second.weight)] == "second.weight"
    assert {scan.name_by_tensor_id[tensor_id] for tensor_id in scan.gpu_constant_ids} == {
        "root_only",
        "first.cache",
        "first.cross_unit",
        "first.left_view",
        "first.right_view",
    }
    storage_key = scan.storage_by_tensor_id[id(transformed.first.weight)].key
    assert isinstance(storage_key.device, torch.device)
    assert storage_key.device in {torch.device("cpu"), torch.device("cuda:0")}
    assert (
        scan.storage_by_tensor_id[id(transformed.first.left_view)].key.storage_impl_id
        == scan.storage_by_tensor_id[id(transformed.first.right_view)].key.storage_impl_id
    )
    with pytest.raises(FrozenInstanceError):
        scan.layer_stats = []  # type: ignore[misc]
    assert not hasattr(scan, "__dict__")
    _assert_value_only(scan)


def test_final_scan_keeps_vllm_sidecars_gpu_resident(bootstrap_module) -> None:
    class GateLinear(nn.Linear):
        pass

    root = nn.Module()
    root.language_model = nn.Module()
    root.language_model.model = nn.Module()
    root.language_model.model.layers = nn.ModuleList([nn.Module()])
    layer = root.language_model.model.layers[0]
    layer.gate_linear = GateLinear(2, 2, bias=False)
    layer.linear_attn = nn.Module()
    layer.linear_attn.A_log = nn.Parameter(torch.ones(2))
    layer.safe = nn.Linear(2, 2, bias=False)

    scan = bootstrap_module.model_scan.scan_loaded_model(
        root,
        ((0, 0, layer),),
        torch.device("cuda:0"),
    )

    eligible_names = {stat.name for stats in scan.layer_stats for stat in stats.tensors}
    assert "language_model.model.layers.0.safe.weight" in eligible_names
    assert "language_model.model.layers.0.gate_linear.weight" not in eligible_names
    assert "language_model.model.layers.0.linear_attn.A_log" not in eligible_names
    assert id(layer.gate_linear.weight) in scan.gpu_constant_ids
    assert id(layer.linear_attn.A_log) in scan.gpu_constant_ids


def test_final_scan_keeps_parameter_with_unregistered_storage_alias_resident(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    transformed.first.hidden_view = transformed.first.weight[:1]

    scan = _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda:0")

    assert [layer.label for layer in scan.layer_stats] == ["second"]
    assert [[stat.name for stat in layer.tensors] for layer in scan.layer_stats] == [["second.weight"]]
    assert id(transformed.first.weight) in scan.gpu_constant_ids
    assert id(transformed.first.hidden_view) not in scan.name_by_tensor_id
    assert id(transformed.first.hidden_view) not in scan.storage_by_tensor_id


@pytest.mark.parametrize("same_object", [True, False], ids=["same-object", "shared-storage"])
def test_final_scan_keeps_parameter_buffer_aliases_resident(
    bootstrap_module,
    monkeypatch,
    same_object,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    parameter = transformed.first.weight
    buffer = parameter if same_object else parameter.detach()
    transformed.first.register_buffer("weight_buffer", buffer)

    scan = _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda:0")

    assert id(parameter) in scan.gpu_constant_ids
    assert all(stat.tensor_id != id(parameter) for layer in scan.layer_stats for stat in layer.tensors)


def test_final_scan_reports_actionable_wrong_root(bootstrap_module, monkeypatch) -> None:
    offloader, _model = _state_offloader(bootstrap_module, monkeypatch)
    wrong_root = _state_root()

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error) as caught:
        _scan_loaded_model(bootstrap_module, offloader, wrong_root, "cuda:0")

    message = str(caught.value)
    assert "root_type=torch.nn.modules.module.Module" in message
    assert "separately loaded model or drafter" in message
    assert "wrong root" in message
    assert "registered nn.Module tree" in message


def test_scan_uses_observed_units_and_keeps_selected_edges_resident(bootstrap_module) -> None:
    root = _decoder_root()
    live_units = tuple((0, index, layer) for index, layer in enumerate(root.model.layers))

    scan = bootstrap_module.model_scan.scan_loaded_model(
        root,
        live_units,
        torch.device("cuda:0"),
        include_patterns=[
            "class:DecoderLayer",
            "model.embed_tokens",
            "model.norm",
            "lm_head",
            "logits_processor",
        ],
        exclude_patterns=[],
    )

    assert [unit.label for unit in scan.layer_stats] == [
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
    ]
    assert {
        id(root.model.embed_tokens.weight),
        id(root.model.norm.weight),
        id(root.model.norm.bias),
        id(root.lm_head.weight),
        id(root.logits_processor.weight),
    } <= scan.gpu_constant_ids


def test_scan_schedules_only_observed_units_with_selected_tensors(bootstrap_module) -> None:
    root = _decoder_root()
    live_units = tuple((0, index, layer) for index, layer in enumerate(root.model.layers))

    scan = bootstrap_module.model_scan.scan_loaded_model(
        root,
        live_units,
        torch.device("cuda:0"),
        include_patterns=["model.layers.1"],
        exclude_patterns=[],
    )

    assert [unit.label for unit in scan.layer_stats] == ["model.layers.1"]


def test_scan_rejects_no_runtime_observed_offloadable_tensors(bootstrap_module) -> None:
    root = _decoder_root(layer_count=1)

    with pytest.raises(
        bootstrap_module.VllmFlexTensorV2Error,
        match="no runtime-observed offloadable tensors",
    ):
        bootstrap_module.model_scan.scan_loaded_model(
            root,
            ((0, 0, root.model.layers[0]),),
            torch.device("cuda:0"),
            include_patterns=["model.embed_tokens"],
            exclude_patterns=[],
        )


def test_observed_unit_keeps_all_registered_module_aliases(bootstrap_module) -> None:
    root = _decoder_root(layer_count=1)
    root.layer_alias = root.model.layers[0]

    resolved = bootstrap_module.model_scan._resolve_units(root, ((0, 0, root.model.layers[0]),))

    assert [path for path, _module in resolved[0].qualified_modules] == ["model.layers.0", "layer_alias"]


def test_duplicate_observed_runtime_label_is_rejected(bootstrap_module) -> None:
    root = _decoder_root(layer_count=1)
    layer = root.model.layers[0]

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match="runtime label is duplicate"):
        bootstrap_module.model_scan._resolve_units(root, ((0, 0, layer), (1, 0, layer)))


def test_scan_applies_effective_patterns_to_tensor_selection(bootstrap_module, monkeypatch) -> None:
    root = _decoder_root(layer_count=1)
    include_patterns = ["model.layers.0"]
    exclude_patterns = ["model.layers.0.mlp"]
    calls = []
    monkeypatch.setattr(
        bootstrap_module.model_scan,
        "get_non_offloaded_tensor_ids",
        lambda _model, _tensors, include_patterns, exclude_patterns: (
            calls.append((
                "tensors",
                include_patterns,
                exclude_patterns,
            ))
            or set()
        ),
    )

    bootstrap_module.model_scan.scan_loaded_model(
        root,
        ((0, 0, root.model.layers[0]),),
        torch.device("cuda:0"),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    assert calls == [
        ("tensors", include_patterns, exclude_patterns),
    ]


def test_build_state_rejects_uninspectable_hidden_reachable_tensor_before_strategy(
    bootstrap_module,
    monkeypatch,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    transformed.first.hidden_sparse = torch.sparse_coo_tensor(
        torch.tensor([[0], [1]]),
        torch.tensor([1.0]),
        (2, 2),
    )
    strategy = RecordingStateStrategy("strategy")
    _set_cuda_snapshot(monkeypatch, bootstrap_module, available=100, total=200)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"reachable.*storage"):
        offloader.build_state(
            transformed,
            OffloadConfig(load_strategy=strategy, transfer_mode="strategy"),
            "cuda:0",
        )

    assert strategy.calls == []


@pytest.mark.parametrize("failure", ["meta", "layout"])
def test_final_scan_rejects_unsafe_registered_tensor_with_qualified_names(
    bootstrap_module,
    monkeypatch,
    failure,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    if failure == "meta":
        invalid = nn.Parameter(torch.empty(2, device="meta"))
        transformed.first.bad = invalid
        transformed.bad_alias = invalid
    else:
        invalid = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2))
        transformed.first.register_buffer("bad", invalid)
        transformed.register_buffer("bad_alias", invalid)

    with pytest.raises(bootstrap_module.VllmFlexTensorV2Error) as caught:
        _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda:0")

    assert "first.bad" in str(caught.value)
    assert "bad_alias" in str(caught.value)
    assert failure in str(caught.value)


@pytest.mark.parametrize(("tensor_device", "accepted"), [("cuda:0", True), ("cuda:1", False)])
@pytest.mark.filterwarnings("ignore:CUDA initialization.*")
def test_final_scan_uses_normalized_unindexed_cuda_device(
    bootstrap_module,
    monkeypatch,
    tensor_device,
    accepted,
) -> None:
    offloader, transformed = _state_offloader(bootstrap_module, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    with FakeTensorMode():
        transformed.second.weight = nn.Parameter(torch.empty(3, device=tensor_device))

    if accepted:
        scan = _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda")
        assert [stat.name for stat in scan.layer_stats[1].tensors] == ["second.weight"]
        assert scan.storage_by_tensor_id[id(transformed.second.weight)].key.device == torch.device("cuda:0")
    else:
        with pytest.raises(bootstrap_module.VllmFlexTensorV2Error, match=r"device=cuda:1 target=cuda:0"):
            _scan_loaded_model(bootstrap_module, offloader, transformed, "cuda")
