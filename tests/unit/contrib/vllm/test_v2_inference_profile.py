# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from flextensor.collectors import LayerStatistics, TensorStatistics
from flextensor.config import OffloadConfig
from flextensor.offload_timing import (
    OFFLOAD_TIMING_MEASURE_MAX_PASSES,
    OffloadTimingReport,
    OffloadTimingSnapshot,
    TrapTimingStats,
)
from flextensor.state_handler import TensorManagerState, TensorManagerStateHandler

from ._v2_worker_test_utils import _install_bootstrap_fakes, _worker


@pytest.fixture
def profile_module(worker_module):
    return worker_module.inference_profile


def _scheduled(*, context: tuple[int, ...] = (), generation: tuple[int, ...] = (), replay: bool = True):
    context_ids = tuple(f"context-{index}" for index in range(len(context)))
    generation_ids = tuple(f"generation-{index}" for index in range(len(generation)))
    return SimpleNamespace(
        num_scheduled_tokens=dict(zip((*context_ids, *generation_ids), (*context, *generation), strict=True)),
        scheduled_new_reqs=[SimpleNamespace(req_id=request_id) for request_id in context_ids],
        scheduled_cached_reqs=SimpleNamespace(is_context_phase=lambda _request_id: False),
        replay=replay,
        result=object(),
    )


def _runtime_worker(worker_module):
    worker = _worker(worker_module, [])
    worker.model_runner.uniform_decode_query_len = 1
    worker._flextensor_profile_refresh_enabled = True
    worker._flextensor_timing_batch = "decode"
    worker._flextensor_profile_sample_count = 0
    worker._flextensor_profile_sample_target = 2
    worker._flextensor_replay_patch = None
    return worker


def _install_fake_cudagraph(profile_module, monkeypatch):
    class FakeCUDAGraph:
        def __init__(self) -> None:
            self.failure = None

        def replay(self):
            if self.failure is not None:
                raise self.failure
            return "replayed"

    monkeypatch.setattr(profile_module.torch.cuda, "CUDAGraph", FakeCUDAGraph)
    return FakeCUDAGraph()


def _fresh_profile_module(profile_module):
    spec = importlib.util.spec_from_file_location("_fresh_v2_inference_profile", profile_module.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.mark.parametrize(
    ("scheduler_output", "expected"),
    [
        (_scheduled(context=(1,)), "prefill"),
        (_scheduled(context=(128,) * 8), "prefill"),
        (_scheduled(generation=(1,)), "decode"),
        (_scheduled(generation=(4,) * 16), "decode"),
        (_scheduled(context=(16,), generation=(1,)), None),
        (_scheduled(), None),
    ],
)
def test_classify_timing_batch(profile_module, scheduler_output, expected) -> None:
    assert profile_module.classify_timing_batch(scheduler_output) == expected


def test_replay_counter_advances_only_after_success(profile_module, monkeypatch) -> None:
    graph = _install_fake_cudagraph(profile_module, monkeypatch)
    assert profile_module.patch_cudagraph_replay_counter()

    graph.failure = RuntimeError("replay failed")
    with pytest.raises(RuntimeError, match="replay failed"):
        graph.replay()
    assert profile_module.current_cudagraph_replay_generation() == 0

    graph.failure = None
    assert graph.replay() == "replayed"
    assert profile_module.current_cudagraph_replay_generation() == 1


def test_replay_counter_reuses_installed_wrapper_owned_generation(profile_module, monkeypatch) -> None:
    graph = _install_fake_cudagraph(profile_module, monkeypatch)
    assert profile_module.patch_cudagraph_replay_counter()
    assert not hasattr(profile_module.torch.cuda.CUDAGraph.replay, "_ft_vllm_timing_wrapped")
    graph.replay()

    fresh_profile_module = _fresh_profile_module(profile_module)
    assert fresh_profile_module.patch_cudagraph_replay_counter()
    graph.replay()

    assert profile_module.current_cudagraph_replay_generation() == 2
    assert fresh_profile_module.current_cudagraph_replay_generation() == 2


def test_restore_replay_patch_does_not_overwrite_later_replacement(profile_module, monkeypatch) -> None:
    _install_fake_cudagraph(profile_module, monkeypatch)
    original = profile_module.torch.cuda.CUDAGraph.replay
    patch = profile_module.patch_cudagraph_replay_counter()
    assert patch is not None

    profile_module.restore_cudagraph_replay(patch)
    assert profile_module.torch.cuda.CUDAGraph.replay is original

    second_patch = profile_module.patch_cudagraph_replay_counter()
    assert second_patch is not None

    def replacement(_graph):
        return None

    profile_module.torch.cuda.CUDAGraph.replay = replacement
    profile_module.restore_cudagraph_replay(second_patch)
    assert profile_module.torch.cuda.CUDAGraph.replay is replacement


def test_execute_model_keeps_only_matching_real_graph_batches(worker_module, profile_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    timing_calls = []
    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        begin_offload_timing_sample=lambda: timing_calls.append("begin"),
        finish_offload_timing_sample=lambda replay_generation: timing_calls.append(replay_generation) or True,
    )
    base_worker = worker_module.FlexTensorOffloadWorker.__mro__[1]
    graph = _install_fake_cudagraph(profile_module, monkeypatch)
    assert profile_module.patch_cudagraph_replay_counter()

    def execute(_worker, scheduler_output):
        if scheduler_output.replay:
            graph.replay()
        return scheduler_output.result

    monkeypatch.setattr(base_worker, "execute_model", execute)
    saves = []
    monkeypatch.setattr(worker, "_save_refreshed_profile", lambda: saves.append("save"))

    prefill = _scheduled(context=(4, 1))
    first_decode = _scheduled(generation=(1, 1), replay=False)
    second_decode = _scheduled(generation=(1,))
    worker.execute_model(prefill)
    worker.execute_model(first_decode)
    worker.execute_model(second_decode)

    assert timing_calls == ["begin", None, "begin", 2]
    assert worker._flextensor_profile_sample_count == 2
    assert saves == ["save"]


def test_execute_model_skips_replayed_empty_batch_without_disabling_refresh(
    worker_module, profile_module, monkeypatch
) -> None:
    worker = _runtime_worker(worker_module)
    base_worker = worker_module.FlexTensorOffloadWorker.__mro__[1]
    graph = _install_fake_cudagraph(profile_module, monkeypatch)
    assert profile_module.patch_cudagraph_replay_counter()

    def execute(_worker, scheduler_output):
        graph.replay()
        return scheduler_output.result

    monkeypatch.setattr(base_worker, "execute_model", execute)
    updates = []

    empty = _scheduled()
    assert worker.execute_model(empty) is empty.result
    assert worker._flextensor_profile_refresh_enabled

    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        begin_offload_timing_sample=lambda: None,
        finish_offload_timing_sample=lambda replay_generation: updates.append(replay_generation) or True,
    )
    worker.execute_model(_scheduled(generation=(1,)))

    assert len(updates) == 1
    assert worker._flextensor_profile_sample_count == 1
    assert worker._flextensor_profile_refresh_enabled


def test_compile_starts_fresh_production_sampling_after_vllm_capture(worker_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    worker._flextensor_profile_sample_count = 1
    events = worker._events
    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        reset_offload_timing_sampling=lambda: events.append("reset-offload-timing"),
    )
    patch = object()
    monkeypatch.setattr(worker_module.inference_profile, "patch_cudagraph_replay_counter", lambda: patch)

    result = worker.compile_or_warm_up_model()

    assert result == "compilation-times"
    assert events == ["vllm-compile-or-warm-up", "reset-offload-timing"]
    assert worker._flextensor_replay_patch is patch
    assert worker._flextensor_profile_sample_count == 0


def test_compile_excludes_warmup_execute_from_production_sampling(worker_module, profile_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    worker._flextensor_profile_sample_target = 1
    worker._offload_config = OffloadConfig(pinned_memory=False)
    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        begin_offload_timing_sample=lambda: None,
        finish_offload_timing_sample=lambda replay_generation: True,
        cancel_offload_timing_sample=lambda: None,
        reset_offload_timing_sampling=lambda: None,
        runtime_state=object(),
    )
    base_worker = worker_module.FlexTensorOffloadWorker.__mro__[1]
    warmup = _scheduled(generation=(1,), replay=False)

    def compile_or_warm_up_model(_worker):
        worker.execute_model(warmup)
        return "compilation-times"

    monkeypatch.setattr(base_worker, "compile_or_warm_up_model", compile_or_warm_up_model)
    monkeypatch.setattr(base_worker, "execute_model", lambda _worker, scheduler_output: scheduler_output.result)
    monkeypatch.setattr(profile_module, "patch_cudagraph_replay_counter", lambda: None)
    monkeypatch.setattr(profile_module, "save_refreshed_profile", lambda **_kwargs: None)

    assert worker.compile_or_warm_up_model() == "compilation-times"

    assert worker._flextensor_profile_refresh_enabled
    assert worker._flextensor_profile_sample_count == 0


def test_first_timing_failure_disables_refresh(worker_module, profile_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    restores = []
    worker._flextensor_replay_patch = object()
    base_worker = worker_module.FlexTensorOffloadWorker.__mro__[1]
    graph = _install_fake_cudagraph(profile_module, monkeypatch)
    assert profile_module.patch_cudagraph_replay_counter()

    def execute(_worker, scheduler_output):
        graph.replay()
        return scheduler_output.result

    monkeypatch.setattr(base_worker, "execute_model", execute)
    updates = []
    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        begin_offload_timing_sample=lambda: None,
        finish_offload_timing_sample=lambda replay_generation: updates.append(replay_generation) or False,
    )
    monkeypatch.setattr(
        worker_module.inference_profile,
        "restore_cudagraph_replay",
        lambda patch: restores.append(patch),
    )

    worker.execute_model(_scheduled(generation=(1,)))
    worker.execute_model(_scheduled(generation=(1,)))

    assert len(updates) == 1
    assert worker._flextensor_profile_sample_count == 0
    assert not worker._flextensor_profile_refresh_enabled
    assert restores


def test_multiple_graph_replays_in_one_selected_call_publish_once(worker_module, profile_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    graph = _install_fake_cudagraph(profile_module, monkeypatch)
    assert profile_module.patch_cudagraph_replay_counter()
    base_worker = worker_module.FlexTensorOffloadWorker.__mro__[1]
    monkeypatch.setattr(
        base_worker,
        "execute_model",
        lambda _worker, scheduler_output: (graph.replay(), graph.replay(), scheduler_output.result)[-1],
    )
    finishes = []
    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        begin_offload_timing_sample=lambda: None,
        finish_offload_timing_sample=lambda replay_generation: finishes.append(replay_generation) or True,
    )

    worker.execute_model(_scheduled(generation=(1,)))

    assert finishes == [2]
    assert worker._flextensor_profile_sample_count == 1


def test_mixed_batch_serves_without_consuming_sample(worker_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    timing_calls = []
    worker._flextensor_bootstrap_offloader = SimpleNamespace(
        begin_offload_timing_sample=lambda: timing_calls.append("begin"),
        finish_offload_timing_sample=lambda replay_generation: timing_calls.append(replay_generation) or True,
    )

    result = worker.execute_model(_scheduled(context=(8,), generation=(1,)))

    assert result is not None
    assert timing_calls == []
    assert worker._flextensor_profile_refresh_enabled


def test_shutdown_restores_replay_patch(worker_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    patch = object()
    worker._flextensor_replay_patch = patch
    restored = []
    monkeypatch.setattr(worker_module.inference_profile, "restore_cudagraph_replay", restored.append)

    worker.shutdown()

    assert restored == [patch]


def test_load_model_rejects_refresh_target_above_retention(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir="/profiles",
        offload_timing="cuda_graph",
        profiling_iters=OFFLOAD_TIMING_MEASURE_MAX_PASSES + 1,
        pinned_memory=False,
    )
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "decode")
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    with pytest.raises(
        RuntimeError,
        match=rf"profiling_iters.*{OFFLOAD_TIMING_MEASURE_MAX_PASSES}",
    ):
        worker.load_model()

    assert events == []


def _runtime_state(duration: float = 1.0) -> TensorManagerState:
    tensor = TensorStatistics(
        tensor_id=7,
        name="layer.weight",
        size_bytes=16,
        load_time_ms=0.75,
    )
    return TensorManagerState(
        loader_type="strategy",
        tensor_id_to_name_map={7: "layer.weight"},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={"layer": [tensor]},
        release_strategy={"layer": [tensor]},
        label_to_block_id={},
        stats=[LayerStatistics(label="layer", tensors=[tensor], duration=duration)],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map=None,
    )


def test_successful_profile_load_logs_path(worker_module, profile_module, monkeypatch, tmp_path) -> None:
    state = _runtime_state()
    monkeypatch.setattr(profile_module.TensorManagerStateHandler, "load_from_file", lambda _path: state)

    assert (
        profile_module.load_saved_profile(OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False))
        is state
    )
    assert ("info", f"saved profile loaded path={tmp_path / 'profile.json'}") in worker_module._test_logger_records


def test_save_without_profile_directory_does_not_collect(profile_module, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(profile_module.flextensor, "collect_offload_timing", lambda: calls.append("collect"))

    profile_module.save_refreshed_profile(
        config=OffloadConfig(profile_storage_dir=None, pinned_memory=False),
        state=_runtime_state(),
    )

    assert calls == []


def test_save_logs_report_pass_count(profile_module, monkeypatch, tmp_path) -> None:
    snapshot = OffloadTimingSnapshot()
    report = OffloadTimingReport(
        per_trap=(TrapTimingStats(label="layer", compute_min=2.5),),
        passes=(snapshot, snapshot),
    )
    info_calls = []
    monkeypatch.setattr(profile_module.flextensor, "collect_offload_timing", lambda: report)
    monkeypatch.setattr(profile_module.LOGGER, "info", lambda *args: info_calls.append(args))

    profile_module.save_refreshed_profile(
        config=OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False),
        state=_runtime_state(),
    )

    assert info_calls[-1][-1] == 2


def test_incomplete_report_logs_bounded_missing_label_sample(profile_module, monkeypatch, tmp_path) -> None:
    state = _runtime_state()
    state.stats = [LayerStatistics(label=f"unit.{index}", tensors=[], duration=1.0) for index in range(12)]
    report = OffloadTimingReport(per_trap=(TrapTimingStats(label="captured", compute_min=2.5),))
    warning_calls = []
    monkeypatch.setattr(profile_module.flextensor, "collect_offload_timing", lambda: report)
    monkeypatch.setattr(profile_module.LOGGER, "warning", lambda *args: warning_calls.append(args))

    profile_module.save_refreshed_profile(
        config=OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False),
        state=state,
    )

    message = " ".join(str(part) for part in warning_calls[-1])
    assert "11/12" in message
    missing_labels = [f"unit.{index}" for index in range(1, 12)]
    assert f"sample={sorted(missing_labels)[:10]}" in message


def test_save_refreshed_profile_replaces_only_copied_durations(profile_module, monkeypatch, tmp_path) -> None:
    state = _runtime_state()
    report = OffloadTimingReport(
        per_trap=(TrapTimingStats(label="captured", compute_min=2.5),),
    )
    monkeypatch.setattr(profile_module.flextensor, "collect_offload_timing", lambda: report)

    profile_module.save_refreshed_profile(
        config=OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False),
        state=state,
    )

    saved = TensorManagerStateHandler.load_from_file(tmp_path / "profile.json")
    assert state.stats[0].duration == pytest.approx(1.0)
    assert state.stats[0].tensors[0].load_time_ms == pytest.approx(0.75)
    assert saved.stats[0].duration == pytest.approx(2.5)
    assert saved.stats[0].tensors[0].load_time_ms == pytest.approx(0.75)
    assert saved.load_strategy["layer"][0].load_time_ms == pytest.approx(0.75)
    assert saved.tensor_id_to_name_map == {7: "layer.weight"}


def test_incomplete_report_preserves_previous_profile(profile_module, monkeypatch, tmp_path) -> None:
    previous = _runtime_state(duration=7.0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    TensorManagerStateHandler.save_to_file(tmp_path / "profile.json", previous)
    state = _runtime_state()
    monkeypatch.setattr(profile_module.flextensor, "collect_offload_timing", lambda: OffloadTimingReport())

    profile_module.save_refreshed_profile(
        config=OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False),
        state=state,
    )

    saved = TensorManagerStateHandler.load_from_file(tmp_path / "profile.json")
    assert saved.stats[0].duration == pytest.approx(7.0)


def test_worker_disables_refresh_before_profile_save(worker_module, monkeypatch) -> None:
    worker = _runtime_worker(worker_module)
    state = _runtime_state()
    worker._flextensor_bootstrap_offloader = SimpleNamespace(runtime_state=state)
    worker._offload_config = OffloadConfig(profile_storage_dir="/profiles", pinned_memory=False)
    calls = []
    monkeypatch.setattr(
        worker_module.inference_profile,
        "save_refreshed_profile",
        lambda **kwargs: calls.append(kwargs),
    )

    worker._save_refreshed_profile()

    assert not worker._flextensor_profile_refresh_enabled
    assert calls == [
        {
            "config": worker._offload_config,
            "state": state,
        }
    ]
