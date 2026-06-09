# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM worker default include/exclude patterns."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _find_repo_root(path: Path) -> Path:
    """Find the repository root from a path inside the checkout."""
    for parent in path.resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"Could not find repository root from {path}")


@pytest.fixture()
def worker_module():
    """Load the real worker.py with lightweight module stubs."""
    stubs: dict[str, types.ModuleType] = {}

    def _stub(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        stubs[name] = mod
        return mod

    class _Worker:  # noqa: N801 - mimic vLLM's class name
        pass

    class _PsutilError(Exception):
        pass

    _stub("psutil", Error=_PsutilError, virtual_memory=lambda: SimpleNamespace(available=0))

    flextensor = _stub("flextensor")
    flextensor.__path__ = []
    contrib = _stub("flextensor.contrib")
    contrib.__path__ = []
    vllm_pkg = _stub("flextensor.contrib.vllm")
    vllm_pkg.__path__ = []

    _stub("flextensor.config", load_config=lambda **_: None)
    _stub("flextensor.contrib.vllm.loader")
    _stub("flextensor.contrib.vllm._drafter_device", ensure_drafter_on_device=lambda *_: None)
    _stub("flextensor.contrib.vllm._logging", safely_install_flextensor_logging_bridge=lambda: None)
    _stub(
        "flextensor.utils",
        config_field_was_set=lambda config, field_name: field_name in getattr(config, "model_fields_set", set()),
    )

    _stub("vllm")
    _stub("vllm.logger", init_logger=logging.getLogger)
    _stub("vllm.utils")
    _stub("vllm.utils.mem_constants", GiB_bytes=1 << 30)
    _stub("vllm.v1")
    _stub("vllm.v1.worker")
    _stub("vllm.v1.worker.gpu_worker", Worker=_Worker)

    module_name = "flextensor.contrib.vllm.worker"
    previous = {name: sys.modules.get(name) for name in [*stubs, module_name]}
    sys.modules.update(stubs)
    sys.modules.pop(module_name, None)

    worker_path = _find_repo_root(Path(__file__)) / "src" / "flextensor" / "contrib" / "vllm" / "worker.py"
    spec = importlib.util.spec_from_file_location(module_name, worker_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


class TestVllmDefaultPatternUpdates:
    """vLLM worker defaults must be layer-focused and MoE-safe."""

    def test_wildcard_include_gets_vllm_layer_defaults(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
        )

        updates = worker_module._vllm_config_updates(config)

        assert updates["include_patterns"] == worker_module.VLLM_DEFAULT_INCLUDE_PATTERNS
        assert updates["exclude_patterns"] == worker_module.VLLM_DEFAULT_EXCLUDE_PATTERNS
        assert updates["include_patterns"] == [
            "class:*DecoderLayer",
            "class:*DecoderBlock",
            "class:*TransformerBlock",
            "model.embed_tokens",
            "model.norm",
            "model.norm_f",
            "lm_head",
            "logits_processor",
            "language_model.model.embed_tokens",
            "language_model.model.norm",
            "language_model.lm_head",
            "language_model.logits_processor",
        ]
        assert updates["exclude_patterns"] == [
            "class:GateLinear",
            "model.layers.*.mixer.shared_experts",
            "model.layers.*.mixer.fc1_latent_proj",
            "model.layers.*.mixer.fc2_latent_proj",
            "model.layers.*.mlp.shared_expert",
            "language_model.model.layers.*.mlp.shared_expert",
            "language_model.model.layers.*.mlp.gate",
            "language_model.model.layers.*.mlp.shared_expert_gate",
            "language_model.model.layers.*.linear_attn.A_log",
            "language_model.model.layers.*.linear_attn.dt_bias",
        ]
        assert "*" not in updates["include_patterns"]
        assert "class:*Block" not in updates["include_patterns"]

    def test_explicit_empty_exclude_patterns_are_preserved(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=1,
            profiling_iters=1,
            include_patterns=["*"],
            exclude_patterns=[],
            model_fields_set={"exclude_patterns"},
        )

        updates = worker_module._vllm_config_updates(config)

        assert "exclude_patterns" not in updates

    def test_custom_include_and_exclude_patterns_are_preserved(self, worker_module) -> None:
        config = SimpleNamespace(
            discovery_iters=5,
            profiling_iters=4,
            include_patterns=["custom.layers.*"],
            exclude_patterns=["custom.layers.*.keep_on_gpu"],
        )

        updates = worker_module._vllm_config_updates(config)

        assert "include_patterns" not in updates
        assert "exclude_patterns" not in updates
        assert updates["discovery_iters"] == 5
        assert updates["profiling_iters"] == 4
