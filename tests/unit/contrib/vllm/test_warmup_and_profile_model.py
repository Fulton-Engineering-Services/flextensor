# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``FlexTensorOffloadWorker.warmup_and_profile_model``.

These tests pin down the contract the worker has with FlexTensor's profiling-
control API and the exact sequence of ``_dummy_run`` calls:

* ``compile_or_warm_up_model()`` is called twice; the **second** invocation
  must be wrapped in ``flextensor.pause_profiling()`` so vLLM's mixed-batch
  warmup does not pollute profiling statistics.
* Discovery loop issues ``_dummy_run(1, skip_eplb=True)`` exactly
  ``discovery_iters - 2`` times (the ``-2`` accounts for the first
  ``compile_or_warm_up_model`` which vLLM has already executed and which
  contributes warmup iterations).
* Profiling loop issues ``_dummy_run(max_num_tokens, skip_eplb=True)`` exactly
  ``profiling_iters`` times, where ``max_num_tokens`` is clamped to
  ``min(max_model_len, max_num_batched_tokens)``.
* Two trailing ``_dummy_run(max_num_tokens, skip_eplb=True)`` calls drive the
  transition to inference mode.

vLLM is not importable in the unit-test environment, so we stub the minimum
set of ``vllm.*`` submodules in ``sys.modules`` before importing the worker.
This gives the tests direct coverage of the real method (not a copy), which
is the whole point of the exercise.
"""

from __future__ import annotations

import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import psutil
import pytest

# ---------------------------------------------------------------------------
# Module import fixture: stub vllm + flextensor.contrib.vllm.loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def worker_module():
    """Import ``flextensor.contrib.vllm._legacy_worker`` with vllm stubbed out.

    The worker module unconditionally imports a handful of ``vllm.*`` symbols
    at the top level, plus ``flextensor.contrib.vllm.loader`` (which itself
    pulls in a much larger slice of vllm).  We register minimal stand-ins so
    the real worker source file can be loaded and exercised without a vllm
    installation.
    """
    v2_runner_env = os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    stubs: dict[str, types.ModuleType] = {}

    def _stub(name: str, **attrs) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        stubs[name] = mod
        return mod

    # Stand-in base class so Worker subclassing works.
    class _Worker:  # noqa: N801 — mimic vllm's class name
        def __init__(self, *args, **kwargs):
            pass

    _stub("vllm")
    _stub("vllm.logger", init_logger=logging.getLogger)
    _stub("vllm.utils")
    _stub("vllm.utils.mem_constants", GiB_bytes=1 << 30)
    _stub("vllm.v1")
    _stub("vllm.v1.worker")
    _stub("vllm.v1.worker.gpu_worker", Worker=_Worker)
    # Shadow the loader module so the real one (which pulls in vllm) isn't touched.
    _stub("flextensor.contrib.vllm.loader")

    previous = {name: sys.modules.get(name) for name in stubs}
    # Also remove any cached real worker module so our stubs take effect.
    previous["flextensor.contrib.vllm._legacy_worker"] = sys.modules.pop("flextensor.contrib.vllm._legacy_worker", None)

    sys.modules.update(stubs)

    # Importing the worker triggers ``safely_install_flextensor_logging_bridge()``,
    # which (via our stubbed ``init_logger``) copies pytest's root handlers onto
    # the ``flextensor`` logger. Snapshot its state so we can restore it.
    from flextensor._logging import _BRIDGE_MARKER  # noqa: PLC0415

    ft_logger = logging.getLogger("flextensor")
    ft_snapshot = (
        list(ft_logger.handlers),
        ft_logger.level,
        ft_logger.propagate,
        getattr(ft_logger, _BRIDGE_MARKER, None),
    )

    try:
        import flextensor.contrib.vllm._legacy_worker as worker  # noqa: PLC0415

        yield worker
    finally:
        if v2_runner_env is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = v2_runner_env

        # Restore prior module state so other tests see the real / absent
        # vllm modules, not our stubs.
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

        # Undo the bridge's side effects on the ``flextensor`` logger.
        ft_logger.handlers[:] = ft_snapshot[0]
        ft_logger.setLevel(ft_snapshot[1])
        ft_logger.propagate = ft_snapshot[2]
        if ft_snapshot[3] is None:
            if hasattr(ft_logger, _BRIDGE_MARKER):
                delattr(ft_logger, _BRIDGE_MARKER)
        else:
            setattr(ft_logger, _BRIDGE_MARKER, ft_snapshot[3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(
    worker_module,
    *,
    discovery_iters: int,
    profiling_iters: int,
    max_model_len: int,
    max_num_batched_tokens: int,
):
    """Build a ``FlexTensorOffloadWorker`` instance wired with mocks.

    We bypass ``__init__`` (which requires a real vLLM config) and set only
    the attributes ``warmup_and_profile_model`` actually touches.
    """
    cls = worker_module.FlexTensorOffloadWorker
    w = cls.__new__(cls)

    w._offload_config = SimpleNamespace(
        discovery_iters=discovery_iters,
        profiling_iters=profiling_iters,
    )

    # model_runner.model_runner._dummy_run is the call we assert on.
    w.model_runner = MagicMock()
    w.model_runner.max_model_len = max_model_len

    w.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
    )

    # compile_or_warm_up_model is invoked twice; the second must be inside
    # pause_profiling.  Using a MagicMock lets us verify call order and count.
    w.compile_or_warm_up_model = MagicMock()

    return w


def _compiled_offload_manager(
    *,
    replan_active: bool = False,
    replan_iters: int = 0,
    eager_profiling_iters: int | None = None,
    profiling_iters: int = 2,
) -> MagicMock:
    from flextensor.compile import COMPILED_EAGER_PROFILE_FORWARDS
    from flextensor.offload_manager import OffloadPhase

    om = MagicMock()
    om.compiled_offload_active = True
    om.compiled_replan_active = replan_active
    om.request_strategy_replan.return_value = replan_iters
    if eager_profiling_iters is None:
        eager_profiling_iters = COMPILED_EAGER_PROFILE_FORWARDS if replan_active else profiling_iters
    om.eager_profiling_iters = eager_profiling_iters
    om.phase = OffloadPhase.INFERENCE
    return om


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_worker_defaults_v2_model_runner_off(worker_module):
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0"


class TestWarmupAndProfileModelCallContract:
    """Pins down the exact call sequence of ``warmup_and_profile_model``."""

    def test_dummy_run_call_counts_and_args(self, worker_module):
        """Discovery + profiling + 2 trailing inference-transition calls."""
        discovery_iters = 5  # -> 5 - 2 = 3 discovery _dummy_run calls
        profiling_iters = 4  # -> 4 profiling _dummy_run calls
        max_model_len = 8192
        max_num_batched_tokens = 2048  # -> clamp to 2048

        w = _make_worker(
            worker_module,
            discovery_iters=discovery_iters,
            profiling_iters=profiling_iters,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        with (
            patch.object(worker_module.flextensor, "pause_profiling") as mock_pause,
            patch.object(
                worker_module.flextensor,
                "get_offload_manager",
                return_value=_compiled_offload_manager(profiling_iters=profiling_iters),
            ),
        ):
            mock_pause.return_value.__enter__ = MagicMock(return_value=None)
            mock_pause.return_value.__exit__ = MagicMock(return_value=False)

            w.warmup_and_profile_model()

        expected_max_num_tokens = min(max_model_len, max_num_batched_tokens)
        expected_discovery = discovery_iters - 2

        expected_calls = (
            [call(1, skip_eplb=True)] * expected_discovery
            + [call(expected_max_num_tokens, skip_eplb=True)] * profiling_iters
            + [call(expected_max_num_tokens, skip_eplb=True)] * 2  # inference transition
        )

        actual = w.model_runner._dummy_run.call_args_list
        assert actual == expected_calls, (
            f"_dummy_run call sequence mismatch.\nExpected {len(expected_calls)} calls, "
            f"got {len(actual)}.\nExpected: {expected_calls}\nActual:   {actual}"
        )

    def test_pause_profiling_wraps_second_compile_or_warm_up(self, worker_module):
        """The second ``compile_or_warm_up_model()`` must run inside pause_profiling."""
        w = _make_worker(
            worker_module,
            discovery_iters=3,
            profiling_iters=2,
            max_model_len=1024,
            max_num_batched_tokens=1024,
        )

        events: list[str] = []

        class _Ctx:
            def __enter__(self):
                events.append("pause_enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                events.append("pause_exit")
                return False

        def _compile_side_effect():
            events.append("compile")

        w.compile_or_warm_up_model.side_effect = _compile_side_effect

        with (
            patch.object(worker_module.flextensor, "pause_profiling", return_value=_Ctx()) as mock_pause,
            patch.object(
                worker_module.flextensor,
                "get_offload_manager",
                return_value=_compiled_offload_manager(profiling_iters=2, eager_profiling_iters=2),
            ),
        ):
            w.warmup_and_profile_model()

        # pause_profiling is called exactly once and wraps the second compile call.
        mock_pause.assert_called_once_with()
        assert events == ["compile", "pause_enter", "compile", "pause_exit"], (
            f"Unexpected ordering of compile/pause events: {events}. "
            "The second compile_or_warm_up_model() must execute inside pause_profiling()."
        )

    def test_pause_profiling_exits_when_second_compile_raises(self, worker_module):
        """If the second compile fails, pause context must still exit.

        This protects against leaving profiling suspended when warmup fails.
        """
        w = _make_worker(
            worker_module,
            discovery_iters=3,
            profiling_iters=2,
            max_model_len=1024,
            max_num_batched_tokens=1024,
        )

        events: list[str] = []

        class _Ctx:
            def __enter__(self):
                events.append("pause_enter")
                return self

            def __exit__(self, exc_type, exc, tb):
                events.append(f"pause_exit:{exc_type.__name__ if exc_type else 'None'}")
                return False

        call_index = {"value": 0}

        def _compile_side_effect():
            call_index["value"] += 1
            events.append(f"compile_{call_index['value']}")
            if call_index["value"] == 2:
                raise RuntimeError("warmup failed")

        w.compile_or_warm_up_model.side_effect = _compile_side_effect

        with (
            patch.object(worker_module.flextensor, "pause_profiling", return_value=_Ctx()) as mock_pause,
            pytest.raises(RuntimeError, match="warmup failed"),
        ):
            w.warmup_and_profile_model()

        mock_pause.assert_called_once_with()
        assert events == [
            "compile_1",
            "pause_enter",
            "compile_2",
            "pause_exit:RuntimeError",
        ], f"pause_profiling context did not exit as expected on failure: {events}"

    def test_max_num_tokens_is_clamped_by_max_model_len(self, worker_module):
        """``max_num_tokens`` = min(max_model_len, max_num_batched_tokens)."""
        w = _make_worker(
            worker_module,
            discovery_iters=3,
            profiling_iters=1,
            max_model_len=512,  # <-- the smaller of the two
            max_num_batched_tokens=4096,
        )

        with (
            patch.object(worker_module.flextensor, "pause_profiling", return_value=MagicMock()),
            patch.object(
                worker_module.flextensor,
                "get_offload_manager",
                return_value=_compiled_offload_manager(profiling_iters=1, eager_profiling_iters=1),
            ),
        ):
            w.warmup_and_profile_model()

        profiling_and_inference_calls = [
            c for c in w.model_runner._dummy_run.call_args_list if c != call(1, skip_eplb=True)
        ]
        for c in profiling_and_inference_calls:
            assert c == call(512, skip_eplb=True), (
                f"Non-discovery _dummy_run used {c.args[0]} tokens; expected clamped value 512."
            )

    def test_discovery_iters_floor_of_three_applied_by_config_loading(self, worker_module):
        """If ``discovery_iters == 3`` (the floor enforced in ``load_model``),
        the discovery loop issues exactly one ``_dummy_run(1, ...)`` call
        (``3 - 2 = 1``)."""
        w = _make_worker(
            worker_module,
            discovery_iters=3,
            profiling_iters=1,
            max_model_len=1024,
            max_num_batched_tokens=1024,
        )

        with (
            patch.object(worker_module.flextensor, "pause_profiling", return_value=MagicMock()),
            patch.object(
                worker_module.flextensor,
                "get_offload_manager",
                return_value=_compiled_offload_manager(profiling_iters=1, eager_profiling_iters=1),
            ),
        ):
            w.warmup_and_profile_model()

        discovery_calls = [c for c in w.model_runner._dummy_run.call_args_list if c == call(1, skip_eplb=True)]
        assert len(discovery_calls) == 1

    @pytest.mark.parametrize(
        "exc",
        [
            OSError("denied"),
            AttributeError("svmem missing 'available'"),
            psutil.AccessDenied("restricted /proc"),
        ],
        ids=["OSError", "AttributeError", "psutil.AccessDenied"],
    )
    def test_host_memory_probe_failure_does_not_abort_inference_transition(self, worker_module, exc):
        """The pre-flight host-mem hint must tolerate ``psutil.virtual_memory()``
        failures: log at DEBUG and continue. A propagated exception here would
        crash worker bring-up on hardened containers (gVisor / distroless /
        restricted ``/proc``), the exact failure mode the original
        ``except Exception: pass`` was guarding against. ``psutil.AccessDenied``
        is the load-bearing case — it subclasses ``psutil.Error``, **not**
        ``OSError``, so a clause that lists only ``OSError`` would let it
        through.
        """
        discovery_iters = 3  # -> 1 discovery _dummy_run call
        profiling_iters = 1
        max_num_tokens = 1024
        w = _make_worker(
            worker_module,
            discovery_iters=discovery_iters,
            profiling_iters=profiling_iters,
            max_model_len=max_num_tokens,
            max_num_batched_tokens=max_num_tokens,
        )

        with (
            patch.object(worker_module.flextensor, "pause_profiling", return_value=MagicMock()),
            patch.object(worker_module.psutil, "virtual_memory", side_effect=exc),
            patch.object(
                worker_module.flextensor,
                "get_offload_manager",
                return_value=_compiled_offload_manager(profiling_iters=profiling_iters),
            ),
        ):
            w.warmup_and_profile_model()

        actual = w.model_runner._dummy_run.call_args_list
        expected = (
            [call(1, skip_eplb=True)] * (discovery_iters - 2)
            + [call(max_num_tokens, skip_eplb=True)] * profiling_iters
            + [call(max_num_tokens, skip_eplb=True)] * 2  # inference-transition pair
        )
        assert actual == expected, (
            f"warmup_and_profile_model must complete despite {type(exc).__name__} from psutil.virtual_memory; "
            f"expected trailing two _dummy_run({max_num_tokens}, skip_eplb=True) calls but got {actual}"
        )

    def test_eager_profiling_seed_uses_three_forwards_when_replan_active(self, worker_module):
        """Compiled-offload + replan needs the fixed eager seed, not profiling_iters."""
        from flextensor.compile import COMPILED_EAGER_PROFILE_FORWARDS

        w = _make_worker(
            worker_module,
            discovery_iters=3,
            profiling_iters=2,
            max_model_len=1024,
            max_num_batched_tokens=1024,
        )

        with (
            patch.object(worker_module.flextensor, "pause_profiling", return_value=MagicMock()),
            patch.object(
                worker_module.flextensor,
                "get_offload_manager",
                return_value=_compiled_offload_manager(replan_active=True, replan_iters=0),
            ),
        ):
            w.warmup_and_profile_model()

        max_calls = [c for c in w.model_runner._dummy_run.call_args_list if c == call(1024, skip_eplb=True)]
        assert len(max_calls) == COMPILED_EAGER_PROFILE_FORWARDS


class TestVllmModelContext:
    """Pins the model update behavior of ``vllm_model_context``."""

    def test_updates_model_runner_when_body_raises_after_replacement(self, worker_module):
        original_model = object()
        replacement_model = object()
        model_runner = SimpleNamespace(model=original_model)

        with (
            pytest.raises(RuntimeError, match="offload failed after replacement"),
            worker_module.vllm_model_context(model_runner) as model_container,
        ):
            model_container[0] = replacement_model
            raise RuntimeError("offload failed after replacement")

        assert model_runner.model is replacement_model


class TestResolveCudaGraphWrapper:
    """Pins the import-fallback contract for ``_resolve_cuda_graph_wrapper``."""

    @staticmethod
    def _stub_modules(stubs: dict[str, types.ModuleType]) -> dict:
        previous = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        return previous

    @staticmethod
    def _restore_modules(previous: dict) -> None:
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    def test_returns_primary_path_class_and_logs_info(self, worker_module, caplog):
        worker_module._CUDA_GRAPH_WRAPPER_LOGGED = False

        class _PrimaryWrapper:
            pass

        prim = types.ModuleType("vllm.compilation.wrapper")
        prim.CUDAGraphWrapper = _PrimaryWrapper
        comp = types.ModuleType("vllm.compilation")
        previous = self._stub_modules({"vllm.compilation": comp, "vllm.compilation.wrapper": prim})

        try:
            with caplog.at_level("INFO", logger="flextensor.contrib.vllm.worker"):
                resolved = worker_module._resolve_cuda_graph_wrapper()
        finally:
            self._restore_modules(previous)

        assert resolved is _PrimaryWrapper
        assert any("primary path" in r.getMessage() for r in caplog.records), (
            f"expected INFO logging the primary path; got: {[r.getMessage() for r in caplog.records]}"
        )

    def test_falls_back_when_primary_path_missing_and_chains_primary_error(self, worker_module, caplog):
        worker_module._CUDA_GRAPH_WRAPPER_LOGGED = False

        class _FallbackWrapper:
            pass

        # Primary path must raise ImportError. Stub the package without the
        # ``wrapper`` submodule so ``from vllm.compilation.wrapper import ...`` fails.
        comp = types.ModuleType("vllm.compilation")
        cg = types.ModuleType("vllm.compilation.cuda_graph")
        cg.CUDAGraphWrapper = _FallbackWrapper
        previous = self._stub_modules({"vllm.compilation": comp, "vllm.compilation.cuda_graph": cg})
        # Belt-and-braces: forget any cached wrapper module so the import really fails.
        sys.modules.pop("vllm.compilation.wrapper", None)

        try:
            with caplog.at_level("DEBUG", logger="flextensor.contrib.vllm.worker"):
                resolved = worker_module._resolve_cuda_graph_wrapper()
        finally:
            self._restore_modules(previous)

        assert resolved is _FallbackWrapper
        assert any("fallback path" in r.getMessage() for r in caplog.records), (
            f"expected DEBUG logging the fallback path; got: {[r.getMessage() for r in caplog.records]}"
        )
        assert any("primary import failed" in r.getMessage() for r in caplog.records), (
            "fallback log must reference the primary failure for triage"
        )

    def test_returns_none_and_warns_when_both_paths_missing(self, worker_module, caplog):
        worker_module._CUDA_GRAPH_WRAPPER_LOGGED = False

        comp = types.ModuleType("vllm.compilation")
        previous = self._stub_modules({"vllm.compilation": comp})
        sys.modules.pop("vllm.compilation.wrapper", None)
        sys.modules.pop("vllm.compilation.cuda_graph", None)

        try:
            with caplog.at_level("WARNING", logger="flextensor.contrib.vllm.worker"):
                resolved = worker_module._resolve_cuda_graph_wrapper()
        finally:
            self._restore_modules(previous)

        assert resolved is None
        assert any("CUDAGraphWrapper not importable" in r.getMessage() for r in caplog.records), (
            "double-failure must surface a WARNING citing both failed paths"
        )
