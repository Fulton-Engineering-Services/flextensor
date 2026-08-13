# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types

import pytest

SELECTOR = "FT_VLLM_USE_V2_WORKER"
PUBLIC_MODULE = "flextensor.contrib.vllm.worker"
LEGACY_MODULE = "flextensor.contrib.vllm._legacy_worker"
V2_MODULE = "flextensor.contrib.vllm.v2.worker"


class LegacyWorker:
    pass


class V2Worker:
    pass


def _target_module(name: str, worker: type) -> types.ModuleType:
    module = types.ModuleType(name)
    module.FlexTensorOffloadWorker = worker
    return module


@pytest.mark.parametrize("selector", [None, "1"])
def test_public_worker_defaults_to_v2_without_importing_legacy(monkeypatch, selector) -> None:
    monkeypatch.delitem(sys.modules, PUBLIC_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, LEGACY_MODULE, raising=False)
    monkeypatch.setitem(sys.modules, V2_MODULE, _target_module(V2_MODULE, V2Worker))
    if selector is None:
        monkeypatch.delenv(SELECTOR, raising=False)
    else:
        monkeypatch.setenv(SELECTOR, selector)
    module = importlib.import_module(PUBLIC_MODULE)
    assert module.FlexTensorOffloadWorker is V2Worker
    assert LEGACY_MODULE not in sys.modules


def test_public_worker_selects_legacy(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, PUBLIC_MODULE, raising=False)
    monkeypatch.setitem(sys.modules, LEGACY_MODULE, _target_module(LEGACY_MODULE, LegacyWorker))
    monkeypatch.setenv(SELECTOR, "0")
    module = importlib.import_module(PUBLIC_MODULE)
    assert module.FlexTensorOffloadWorker is LegacyWorker


def test_public_worker_rejects_invalid_selector(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, PUBLIC_MODULE, raising=False)
    monkeypatch.setenv(SELECTOR, "yes")
    with pytest.raises(ValueError, match=r"FT_VLLM_USE_V2_WORKER.*0.*1"):
        importlib.import_module(PUBLIC_MODULE)
