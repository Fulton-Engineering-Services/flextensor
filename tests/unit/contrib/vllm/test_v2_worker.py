# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flextensor.config import OffloadConfig
from flextensor.contrib.vllm._patterns import VLLM_DEFAULT_EXCLUDE_PATTERNS, VLLM_DEFAULT_INCLUDE_PATTERNS

from ._v2_worker_test_utils import _install_bootstrap_fakes, _worker


def test_load_model_passes_explicit_logical_device_index(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.device = torch.device("cuda:3")
    captured = {}

    def load_config(**kwargs):
        captured.update(kwargs)
        return OffloadConfig(enabled=False, pinned_memory=False)

    monkeypatch.setattr(worker_module, "load_config", load_config)
    worker.load_model()

    assert captured == {"gpu_device": 3}
    assert events == ["vllm-load-model"]


def test_load_model_rejects_unindexed_worker_device(worker_module) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    worker.device = torch.device("cuda")

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match="explicit index"):
        worker.load_model()

    assert events == []


def test_load_saved_profile_uses_existing_profile_json(worker_module, monkeypatch, tmp_path) -> None:
    profile = object.__new__(worker_module.inference_profile.TensorManagerState)
    captured = []
    monkeypatch.setattr(
        worker_module.inference_profile.TensorManagerStateHandler,
        "load_from_file",
        lambda path: captured.append(path) or profile,
    )

    result = worker_module.inference_profile.load_saved_profile(
        OffloadConfig(profile_storage_dir=str(tmp_path), profile_read_only=True, pinned_memory=False)
    )

    assert result is profile
    assert captured == [tmp_path / "profile.json"]


def test_missing_saved_profile_uses_conservative_fallback(worker_module, tmp_path) -> None:
    result = worker_module.inference_profile.load_saved_profile(
        OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False)
    )

    assert result is None
    assert any(
        level == "warning" and message.startswith("saved profile not found")
        for level, message in worker_module._test_logger_records
    )


def test_malformed_saved_profile_shape_uses_conservative_fallback(worker_module, tmp_path) -> None:
    (tmp_path / "profile.json").write_text(
        '{"version": 3, "tensor_id_to_name_map": []}',
        encoding="utf-8",
    )

    result = worker_module.inference_profile.load_saved_profile(
        OffloadConfig(profile_storage_dir=str(tmp_path), pinned_memory=False)
    )

    assert result is None
    assert any(
        level == "warning" and message.startswith("saved profile at")
        for level, message in worker_module._test_logger_records
    )


def test_load_model_passes_saved_profile_to_takeover(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir="/profiles",
        profile_read_only=True,
        pinned_memory=False,
    )
    profile = object()
    loaded_configs = []
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(
        worker_module.inference_profile,
        "load_saved_profile",
        lambda actual_config: loaded_configs.append(actual_config) or profile,
    )
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert loaded_configs == [worker._offload_config]
    assert worker._offload_config.profile_storage_dir == config.profile_storage_dir
    assert worker_module._test_takeover_profile is profile
    assert worker._flextensor_bootstrap_offloader is worker_module._test_bootstrap


@pytest.mark.parametrize("offloader_name", ["UVAOffloader", "PrefetchOffloader", "BaseOffloader"])
def test_load_model_rejects_active_native_offloader_before_model_construction(
    worker_module,
    monkeypatch,
    offloader_name,
) -> None:
    from vllm.model_executor.offloader import base as offloader_base

    events: list[str] = []
    worker = _worker(worker_module, events)
    active_offloader = getattr(offloader_base, offloader_name)()
    set_calls = []
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: OffloadConfig(pinned_memory=False))
    monkeypatch.setattr(worker_module, "_vllm_version", lambda: "0.23.0")
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    monkeypatch.setattr(worker_module, "_offloader_api", lambda: (lambda: active_offloader, set_calls.append))
    monkeypatch.setattr(
        worker_module,
        "VllmBootstrapOffloader",
        lambda: pytest.fail("bootstrap construction must not run for a native offloader conflict"),
    )

    with pytest.raises(worker_module.VllmFlexTensorV2Error, match=offloader_name):
        worker.load_model()

    assert set_calls == []
    assert events == []


def test_load_model_enables_writable_refresh_with_derived_timing(worker_module, monkeypatch) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir="/profiles",
        offload_timing="off",
        transfer_mode="allocation_block_transfer",
        profiling_iters=3,
        pinned_memory=False,
    )
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    monkeypatch.setenv("FT_VLLM_TIMING_BATCH", "decode")
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert worker._flextensor_profile_refresh_enabled
    assert worker._flextensor_timing_batch == "decode"
    assert worker._flextensor_profile_sample_target == 3
    assert worker._flextensor_profile_sample_count == 0
    assert worker._offload_config.offload_timing == "eager"


@pytest.mark.parametrize(
    ("profile_storage_dir", "profile_read_only", "timing_batch"),
    [
        (None, False, "decode"),
        ("/profiles", True, "decode"),
        ("/profiles", False, ""),
    ],
)
def test_load_model_disables_refresh_without_every_required_setting(
    worker_module,
    monkeypatch,
    profile_storage_dir,
    profile_read_only,
    timing_batch,
) -> None:
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    config = OffloadConfig(
        profile_storage_dir=profile_storage_dir,
        profile_read_only=profile_read_only,
        transfer_mode="allocation_block_transfer",
        pinned_memory=False,
    )
    monkeypatch.setattr(worker_module, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(worker_module.inference_profile, "load_saved_profile", lambda _config: None)
    if timing_batch is None:
        monkeypatch.delenv("FT_VLLM_TIMING_BATCH", raising=False)
    else:
        monkeypatch.setenv("FT_VLLM_TIMING_BATCH", timing_batch)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    worker.load_model()

    assert not worker._flextensor_profile_refresh_enabled


def test_load_model_constructs_once_and_adopts_state_before_return(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)

    def publish(model_runner, raw_model, actual_proxy):
        assert model_runner is worker.model_runner
        assert raw_model is model_runner.model
        assert actual_proxy is worker_module._test_proxy
        events.append("publish-proxy")
        model_runner.model = actual_proxy

    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(release=lambda: events.append("release-flextensor")),
        raising=False,
    )
    monkeypatch.setattr(worker_module, "__version__", "test-version")
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", publish, raising=False)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None), raising=False)

    worker.load_model()

    assert events == [
        "get-previous-offloader",
        "set-bootstrap-offloader",
        "vllm-load-model",
        "get-raw-model",
        "start-takeover-memory-profile",
        "bootstrap-post-init",
        "build-state",
        "offload-from-state",
        "stop-takeover-memory-profile",
        "publish-proxy",
    ]
    assert events.count("set-bootstrap-offloader") == 1
    assert worker_module._test_singleton.value is worker_module._test_bootstrap
    assert worker.model_runner.model is worker_module._test_proxy
    assert worker.model_runner.model_memory_usage == 323
    assert worker._offload_config.include_patterns == VLLM_DEFAULT_INCLUDE_PATTERNS
    assert worker._offload_config.exclude_patterns == VLLM_DEFAULT_EXCLUDE_PATTERNS
    assert worker._offload_config.external_compile is False
    assert (
        "info",
        f"FlexTensor test-version offloading enabled with config: {worker._offload_config}",
    ) in worker_module._test_logger_records
    assert ("info", "FlexTensor vLLM integration v2 state takeover complete") in worker_module._test_logger_records


RAW = nn.Module()
PROXY = nn.Module()


def _contains_identity(root: object, target: object) -> bool:
    queue = [root]
    seen: set[int] = set()
    while queue:
        owner = queue.pop(0)
        if owner is target:
            return True
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        for attribute in ("model", "runnable"):
            if hasattr(owner, attribute):
                queue.append(getattr(owner, attribute))
    return False


@pytest.mark.parametrize("shape", ["direct", "nested-model", "nested-runnable"])
def test_publish_replaces_known_runner_reference(worker_module, shape):
    nested = {
        "direct": RAW,
        "nested-model": SimpleNamespace(model=RAW),
        "nested-runnable": SimpleNamespace(runnable=RAW, model=RAW),
    }[shape]
    runner = SimpleNamespace(model=nested)
    worker_module._publish_model_to_runner(runner, RAW, PROXY)

    assert _contains_identity(runner, PROXY)
    assert not _contains_identity(runner, RAW)


def test_publish_accepts_vllm_v2_model_runner(worker_module):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = GPUModelRunner(model=RAW)

    worker_module._publish_model_to_runner(runner, RAW, PROXY)

    assert runner.model is PROXY


def test_publish_rejects_missing_raw_model_without_following_unknown_attributes(worker_module):
    runner = SimpleNamespace(model=SimpleNamespace(other=RAW))

    with pytest.raises(RuntimeError, match="could not publish OffloadModelProxy"):
        worker_module._publish_model_to_runner(runner, RAW, PROXY)

    assert runner.model.other is RAW


def test_publish_rejects_cycle_without_raw_model(worker_module):
    runner = SimpleNamespace()
    runner.model = runner

    with pytest.raises(RuntimeError, match="could not publish OffloadModelProxy"):
        worker_module._publish_model_to_runner(runner, RAW, PROXY)


def _fail_at(events: list[str], boundary: str):
    events.append(boundary)
    raise RuntimeError(boundary)


@pytest.mark.parametrize(
    "boundary",
    [
        "bootstrap-constructor",
        "set-bootstrap-offloader",
        "vllm-load-model",
        "get-raw-model",
        "bootstrap-post-init",
        "build-state",
        "offload-from-state",
    ],
)
def test_pre_takeover_failure_restores_singleton_without_releasing_manager(
    worker_module,
    monkeypatch,
    boundary,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    worker._failure = boundary
    worker_module._test_failure = boundary

    fake_flextensor = SimpleNamespace(release=lambda: events.append("release-flextensor"))
    monkeypatch.setattr(worker_module, "flextensor", fake_flextensor)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=lambda _callback: None))

    with pytest.raises(RuntimeError, match=boundary):
        worker.load_model()

    assert events[-1] == "restore-previous-offloader"
    assert "release-flextensor" not in events


@pytest.mark.parametrize(
    "boundary",
    ["publish-proxy", "register-atexit"],
)
def test_post_takeover_failure_releases_manager_before_restoring_singleton(
    worker_module,
    monkeypatch,
    boundary,
):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    active_owner = SimpleNamespace(value=None)
    real_publish = worker_module._publish_model_to_runner
    worker_module._test_failure = boundary

    def activate(_model, _config):
        active_owner.value = worker_module._test_proxy
        return worker_module._test_proxy

    worker_module._test_takeover_callback = activate

    def release():
        assert active_owner.value is worker_module._test_proxy
        events.append("release-flextensor")
        active_owner.value = None

    def publish(model_runner, raw_model, actual_proxy):
        events.append("publish-proxy")
        if boundary == "publish-proxy":
            raise RuntimeError(boundary)
        real_publish(model_runner, raw_model, actual_proxy)

    def register(_callback):
        events.append("register-atexit")
        if boundary == "register-atexit":
            raise RuntimeError(boundary)

    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(release=release),
    )
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", publish)
    monkeypatch.setattr(worker_module, "atexit", SimpleNamespace(register=register))

    with pytest.raises(RuntimeError, match=boundary):
        worker.load_model()

    assert events[-2:] == ["release-flextensor", "restore-previous-offloader"]
    assert active_owner.value is None


def test_profiler_exit_failure_releases_takeover_before_restoring_singleton(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    active_owner = SimpleNamespace(value=None)

    class FailingDeviceMemoryProfiler:
        consumed_memory = 123

        def __enter__(self):
            events.append("start-takeover-memory-profile")
            return self

        def __exit__(self, _exc_type, _exc_val, _exc_tb):
            events.append("stop-takeover-memory-profile")
            raise RuntimeError("profiler-exit")

    def activate(_model, _config):
        active_owner.value = worker_module._test_proxy
        return worker_module._test_proxy

    worker_module._test_takeover_callback = activate

    def release():
        assert active_owner.value is worker_module._test_proxy
        events.append("release-flextensor")
        active_owner.value = None

    monkeypatch.setattr(worker_module, "DeviceMemoryProfiler", lambda _device: FailingDeviceMemoryProfiler())
    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(release=release),
    )

    with pytest.raises(RuntimeError, match="profiler-exit"):
        worker.load_model()

    assert events[-2:] == ["release-flextensor", "restore-previous-offloader"]
    assert active_owner.value is None


def test_release_failure_still_restores_previous_singleton(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    _install_bootstrap_fakes(worker_module, monkeypatch, events)
    worker_module._test_takeover_callback = lambda _model, _config: worker_module._test_proxy
    monkeypatch.setattr(
        worker_module,
        "flextensor",
        SimpleNamespace(
            release=lambda: _fail_at(events, "release-failed"),
        ),
    )
    monkeypatch.setattr(worker_module, "_publish_model_to_runner", lambda *args: _fail_at(events, "publish-failed"))

    with pytest.raises(RuntimeError, match="release-failed"):
        worker.load_model()

    assert events[-1] == "restore-previous-offloader"


def test_shutdown_restores_previous_singleton_before_release_and_vllm_shutdown(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    previous = object()
    worker._flextensor_previous_offloader = previous
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: (
            lambda: pytest.fail("shutdown must not read the current singleton"),
            lambda offloader: (
                events.append("restore-previous-offloader")
                if offloader is previous
                else pytest.fail("wrong singleton restored")
            ),
        ),
    )
    monkeypatch.setattr(worker_module.flextensor, "release", lambda: events.append("release-flextensor"))

    worker.shutdown()

    assert events == ["restore-previous-offloader", "release-flextensor", "vllm-shutdown"]


def test_shutdown_keeps_previous_offloader_for_retry_when_release_fails(worker_module, monkeypatch):
    events: list[str] = []
    worker = _worker(worker_module, events)
    previous = object()
    worker._flextensor_previous_offloader = previous
    monkeypatch.setattr(
        worker_module,
        "_offloader_api",
        lambda: (
            lambda: pytest.fail("shutdown must not read the current singleton"),
            lambda offloader: (
                events.append("restore-previous-offloader")
                if offloader is previous
                else pytest.fail("wrong singleton restored")
            ),
        ),
    )
    release_attempts = 0

    def release():
        nonlocal release_attempts
        release_attempts += 1
        events.append("release-flextensor")
        if release_attempts == 1:
            raise RuntimeError("release failed")

    monkeypatch.setattr(worker_module.flextensor, "release", release)

    with pytest.raises(RuntimeError, match="release failed"):
        worker.shutdown()

    assert worker._flextensor_previous_offloader is previous

    worker.shutdown()

    assert not hasattr(worker, "_flextensor_previous_offloader")
    assert events == [
        "restore-previous-offloader",
        "release-flextensor",
        "vllm-shutdown",
        "restore-previous-offloader",
        "release-flextensor",
        "vllm-shutdown",
    ]


def test_shared_integration_helper_defaults_to_v2():
    from tests.integration._vllm_server import (
        FLEXTENSOR_OFFLOAD_WORKER_CLS,
        VllmOffloadSmokeCase,
    )

    case = VllmOffloadSmokeCase("model", "output").with_flextensor_offload()
    assert FLEXTENSOR_OFFLOAD_WORKER_CLS == ("flextensor.contrib.vllm.worker.FlexTensorOffloadWorker")
    assert FLEXTENSOR_OFFLOAD_WORKER_CLS in case.cli_args
