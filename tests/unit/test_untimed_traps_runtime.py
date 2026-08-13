# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empirical tests for the runtime impact of untimed (dropped) traps.

These tests verify what *actually* happens at inference when a label is
dropped by ``compute_layer_statistics`` — i.e. the trap had tensor IDs but
no duration samples (a profile-coverage gap, e.g. vLLM's
``logits_processor`` firing at decode-time but missed by prefill-shaped
profile iterations).

Two outcomes are validated — both loader families are now safe at
runtime, but the safety nets live in different places:

* **Strategy loader (``loader_type='strategy'``).** The fall-through to
  CPU was reproduced (it would have flowed through
  :func:`flextensor.tensor_manager._make_tensor_getter` and
  :meth:`flextensor.trap_tensor_mode.TrapInfer.__torch_function__`), but
  the loader now closes that gap itself via
  :func:`flextensor.loaders._compute_untimed_traced_preload`: any
  tensor in ``tensors_map`` that appears in no ``layer_stats`` row is
  added to the preload set at ``__init__`` time, so
  ``TensorStrategyLoader.get(...)`` returns a GPU copy instead of
  ``None``. The getter tests below pin both edges: unconfigured mocks
  keep the legacy fall-through used by older unit tests, while a
  production-like manager with a real device fails loudly on misses so
  profiling cannot hide an unplanned copy.

* **Block loaders (``allocation_block_transfer``, ``raw_block_transfer``).**
  ``prepare_view_model`` runs
  :class:`flextensor.tensor_processors.MoveUnmappedTensorsToGPUProcessor`,
  whose contract is "if not in ``tensor_id_to_view_map``, move to GPU
  permanently." So the same dropped tensor lands on GPU before
  inference begins. That permanent-GPU path is budgeted before strategy
  computation and guarded again immediately before each CUDA move.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from flextensor.collectors import (
    IterativeLayerStatistics,
    IterativeLayerStatisticsCollector,
    LayerStatistics,
    TensorStatistics,
)
from flextensor.helpers import ProfilingSuspender, TrapNestingGuard
from flextensor.loaders import (
    TensorStrategyLoader,
    UntimedTrapRescuer,
    _compute_untimed_traced_preload,
)
from flextensor.tensor_manager import (
    TensorManager,
    _make_tensor_getter,
    compute_layer_statistics,
)
from flextensor.tensor_processors import (
    MoveUnmappedTensorsToGPUProcessor,
    compute_reachable_tensor_ids,
)
from flextensor.trap_tensor_mode import TrapInfer

# ---------------------------------------------------------------------------
# CUDA mocking helpers — let TensorStrategyLoader.__init__ run on a CPU box.
# Same pattern used in tests/unit/test_gpu_memory_usage.py.
# ---------------------------------------------------------------------------


def _mock_cuda_stream() -> MagicMock:
    s = MagicMock()
    s.synchronize = MagicMock()
    s.wait_event = MagicMock()
    s.wait_stream = MagicMock()
    s.record_event = MagicMock(return_value=MagicMock())
    return s


def _mock_cuda_event() -> MagicMock:
    e = MagicMock()
    e.synchronize = MagicMock()
    e.record = MagicMock()
    e.query = MagicMock(return_value=True)
    return e


def _patched_cuda() -> list[Any]:
    """Return a list of started patchers; caller is responsible for stopping them."""
    patchers = [
        patch.object(torch.cuda, "Stream", return_value=_mock_cuda_stream()),
        patch.object(torch.cuda, "synchronize"),
        patch.object(torch.cuda, "Event", side_effect=_mock_cuda_event),
        patch.object(torch.cuda, "stream"),
        patch.object(torch.cuda, "current_stream", return_value=_mock_cuda_stream()),
    ]
    for p in patchers:
        p.start()
    return patchers


def _stop_patches(patchers: list[Any]) -> None:
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# Anchor the upstream filter: untimed labels never reach LayerStatistics.
# ---------------------------------------------------------------------------


class TestComputeLayerStatisticsDropsUntimed:
    """``compute_layer_statistics`` silently drops ``duration is None`` rows.

    This is the root cause that both downstream paths inherit. If this
    invariant ever changes (e.g. raises or logs), the downstream tests
    below need to be revisited too.
    """

    def test_label_without_duration_is_silently_dropped(self) -> None:
        timed = IterativeLayerStatistics(label="timed", tensor_ids={1}, duration=10.0)
        untimed = IterativeLayerStatistics(label="untimed", tensor_ids={2}, duration=None)

        tensor_statistics_map = {
            1: TensorStatistics(tensor_id=1, name="t1", size_bytes=1024, load_time_ms=0.1),
            2: TensorStatistics(tensor_id=2, name="t2", size_bytes=2048, load_time_ms=0.1),
        }

        result = compute_layer_statistics([timed, untimed], tensor_statistics_map)

        assert [r.label for r in result] == ["timed"]
        all_tensor_ids = {ti.tensor_id for r in result for ti in r.tensors}
        assert 1 in all_tensor_ids, "timed tensor must reach LayerStatistics"
        assert 2 not in all_tensor_ids, (
            "untimed tensor must be silently dropped — this is the contract that "
            "leaves dropped tensors invisible to the strategy and to all loaders"
        )


# ---------------------------------------------------------------------------
# Direct unit tests for the rescue helper.
# ---------------------------------------------------------------------------


class TestComputeUntimedTracedPreload:
    """Direct contract for ``_compute_untimed_traced_preload``.

    Tested standalone so the loader scaffolding doesn't muddy the
    semantics. The helper's job is narrow: return tensor IDs that are
    in ``tensors_map`` but appear in no layer's ``tensors`` list.
    """

    def _make_layer(self, label: str, tensor_ids: list[int]) -> LayerStatistics:
        tensors = [
            TensorStatistics(tensor_id=tid, name=f"t{tid}", size_bytes=4, load_time_ms=0.1) for tid in tensor_ids
        ]
        return LayerStatistics(label=label, tensors=tensors, duration=1.0)

    def test_returns_ids_in_tensors_map_not_in_any_layer(self) -> None:
        layers = [self._make_layer("L1", [1, 2])]
        # IDs 1, 2 are timed; ID 3 is in tensors_map but never appeared in a layer.
        tensors_map = {1: torch.zeros(1), 2: torch.zeros(1), 3: torch.zeros(1)}

        result = _compute_untimed_traced_preload(layers, tensors_map)

        assert result == {3}

    def test_returns_empty_when_every_id_appears_in_some_layer(self) -> None:
        """No rescue needed when profile coverage is complete."""
        layers = [self._make_layer("L1", [1]), self._make_layer("L2", [2])]
        tensors_map = {1: torch.zeros(1), 2: torch.zeros(1)}

        assert _compute_untimed_traced_preload(layers, tensors_map) == set()

    def test_returns_full_tensors_map_when_layer_stats_empty(self) -> None:
        """Pathological coverage failure: every traced tensor must be rescued."""
        tensors_map = {1: torch.zeros(1), 2: torch.zeros(1), 3: torch.zeros(1)}

        assert _compute_untimed_traced_preload([], tensors_map) == {1, 2, 3}

    def test_does_not_rescue_ids_outside_tensors_map(self) -> None:
        """A layer mentioning an untracked ID does not pollute the rescue set.

        The rescue is scoped to ``tensors_map`` membership — only
        intentionally-managed tensors are eligible. An ID that appears
        in ``layer_stats`` but isn't in ``tensors_map`` is a
        discovery/registration mismatch the rescue is not responsible
        for.
        """
        layers = [self._make_layer("L1", [99])]  # 99 is not in tensors_map
        tensors_map = {1: torch.zeros(1)}

        result = _compute_untimed_traced_preload(layers, tensors_map)

        assert result == {1}, "only tensors_map members are eligible for rescue"


# ---------------------------------------------------------------------------
# Direct contract tests for ``compute_reachable_tensor_ids`` live next to the
# helper itself, in ``tests/unit/test_tensor_processors.py``
# (``TestReachableTensorMapProcessor``). The two narrowing-via-loader tests
# below stay here because they exercise the rescue's *use* of that input.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Direct unit tests for the rescuer class.
# ---------------------------------------------------------------------------


class TestUntimedTrapRescuer:
    """Direct contract for :class:`UntimedTrapRescuer`.

    The class is also exercised indirectly via
    :class:`TestStrategyLoaderUntimedRescue` (preload path) and the trap-path
    tests, but its own surface — activation, ``reachable_tensor_ids``
    narrowing, owned vs passthrough routing, ``pin`` / ``shutdown`` ownership
    semantics, and the WARNING content (count, MiB, truncated id-hint past
    8) — needs direct coverage so refactors of the class itself surface in
    tests focused on the class.
    """

    GPU_DEVICE = torch.device("cuda:0")
    CPU_DEVICE = torch.device("cpu")

    @staticmethod
    def _layer(label: str, tensor_ids: list[int]) -> IterativeLayerStatistics:
        return IterativeLayerStatistics(label=label, tensor_ids=set(tensor_ids), duration=1.0)

    def _fake_tensor(self, device: torch.device, *, numel: int = 4, element_size: int = 4) -> MagicMock:
        """Tensor double exposing only what :class:`UntimedTrapRescuer` touches.

        ``.to(...)`` returns a *distinct* GPU-side mock so the owned-copy
        branch can be told apart from passthrough by identity.
        """
        t = MagicMock(spec=torch.Tensor)
        t.device = device
        t.numel.return_value = numel
        t.element_size.return_value = element_size
        moved = MagicMock(spec=torch.Tensor)
        moved.device = self.GPU_DEVICE
        t.to.return_value = moved
        return t

    def _build(
        self,
        *,
        layers: list[IterativeLayerStatistics],
        tensors_map: dict[int, MagicMock],
        reachable_tensor_ids: set[int] | None = None,
        del_tensor_func: MagicMock | None = None,
        id_to_name_map: dict[int, str] | None = None,
    ) -> UntimedTrapRescuer:
        """Construct a rescuer with ``torch.cuda.synchronize`` patched out."""
        kwargs: dict[str, object] = {}
        if reachable_tensor_ids is not None:
            kwargs["reachable_tensor_ids"] = reachable_tensor_ids
        if del_tensor_func is not None:
            kwargs["del_tensor_func"] = del_tensor_func
        if id_to_name_map is not None:
            kwargs["id_to_name_map"] = id_to_name_map
        with patch.object(torch.cuda, "synchronize"):
            return UntimedTrapRescuer(layers, tensors_map, self.GPU_DEVICE, **kwargs)

    # ------------------------------------------------------------------ ctor

    def test_no_untimed_ids_means_no_pin_no_warning(self, caplog) -> None:
        """Complete profile coverage → rescuer is a no-op and silent."""
        layers = [self._layer("L1", [1])]
        tensors_map = {1: self._fake_tensor(self.CPU_DEVICE)}

        with caplog.at_level("WARNING", logger="flextensor.loaders"):
            rescuer = self._build(layers=layers, tensors_map=tensors_map)

        assert rescuer.get(1) is None, "timed ids must not appear in the rescuer's map"
        assert caplog.records == [], "no rescue → no WARNING; otherwise the log becomes noise"

    def test_untimed_id_on_host_routes_through_owned_copy(self) -> None:
        """CPU-resident → ``.to(device=gpu, copy=True)`` + ownership.

        Ownership is the precondition for ``del_tensor_func`` to fire on
        shutdown; this test verifies both the copy call and the resulting
        del-func behavior (asserted via ``shutdown()`` in the shutdown tests).
        """
        cpu_tensor = self._fake_tensor(self.CPU_DEVICE)
        tensors_map = {1: cpu_tensor}
        layers: list[IterativeLayerStatistics] = []  # nothing timed → 1 is untimed

        rescuer = self._build(layers=layers, tensors_map=tensors_map)

        assert rescuer.get(1) is cpu_tensor.to.return_value
        cpu_tensor.to.assert_called_once_with(device=self.GPU_DEVICE, copy=True)

    def test_untimed_id_already_on_gpu_takes_passthrough(self) -> None:
        """GPU-resident → no copy, storage shared with the model.

        Avoids the 2x footprint issue: if the canonical tensor already
        lives on the rescuer's device, the rescuer must NOT allocate a
        second buffer. Passthrough (no del-func on shutdown) is asserted
        by the corresponding shutdown test.
        """
        gpu_tensor = self._fake_tensor(self.GPU_DEVICE)
        tensors_map = {1: gpu_tensor}

        rescuer = self._build(layers=[], tensors_map=tensors_map)

        assert rescuer.get(1) is gpu_tensor
        gpu_tensor.to.assert_not_called()

    def test_reachable_tensor_ids_narrows_rescue_scope(self, caplog) -> None:
        """Untimed ids outside ``reachable_tensor_ids`` are excluded from rescue.

        This is the gate that keeps transient (non-model) tensors from
        getting force-pinned to GPU as a side effect of being in
        ``tensors_map``.
        """
        tensors_map = {
            1: self._fake_tensor(self.CPU_DEVICE),  # untimed AND reachable → rescued
            2: self._fake_tensor(self.CPU_DEVICE),  # untimed but NOT reachable → excluded
        }

        with caplog.at_level("WARNING", logger="flextensor.loaders"):
            rescuer = self._build(
                layers=[],
                tensors_map=tensors_map,
                reachable_tensor_ids={1},
            )

        assert rescuer.get(1) is tensors_map[1].to.return_value, "id 1 is reachable — must be rescued"
        assert rescuer.get(2) is None, (
            "id 2 is untimed but unreachable from the model graph; the rescue scope must respect that intersection"
        )

    def test_activation_warning_includes_count_bytes_and_full_id_list(self, caplog) -> None:
        """≤8 untimed ids and no name map → all ids appear as ``id=N``, no '... more' suffix."""
        tensors_map = {tid: self._fake_tensor(self.CPU_DEVICE, numel=256, element_size=4) for tid in range(1, 4)}

        with caplog.at_level("WARNING", logger="flextensor.loaders"):
            self._build(layers=[], tensors_map=tensors_map)

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "3 tensor(s)" in msg, "count must be reported so coverage-gap size is visible"
        expected_mib = (3 * 256 * 4) / (1024 * 1024)
        assert f"{expected_mib:.2f} MiB" in msg, "byte budget drives the cost-of-rescue signal"
        assert "tensors: id=1, id=2, id=3" in msg
        assert "more" not in msg, "no truncation suffix when the id count fits the inline window"

    def test_activation_warning_truncates_id_hint_past_eight(self, caplog) -> None:
        """>8 untimed ids → first 8 are listed inline, rest collapsed to '... (N more)'."""
        tensors_map = {tid: self._fake_tensor(self.CPU_DEVICE) for tid in range(1, 12)}

        with caplog.at_level("WARNING", logger="flextensor.loaders"):
            self._build(layers=[], tensors_map=tensors_map)

        msg = caplog.records[0].getMessage()
        assert "tensors: id=1, id=2, id=3, id=4, id=5, id=6, id=7, id=8, ... (3 more)" in msg, (
            "truncation must show the first 8 in sorted order and collapse the tail "
            "so the WARNING stays readable on large coverage gaps"
        )

    def test_activation_warning_resolves_names_when_id_to_name_map_supplied(self, caplog) -> None:
        """With ``id_to_name_map`` → hint becomes ``name (id=N)`` so operators can grep the model.

        This is the actionability fix: a raw integer like ``140234517123920``
        is meaningless in a log aggregator; ``lm_head.weight (id=...)`` is
        immediately greppable in the source.
        """
        tensors_map = {
            1: self._fake_tensor(self.CPU_DEVICE, numel=256, element_size=4),
            2: self._fake_tensor(self.CPU_DEVICE, numel=256, element_size=4),
            3: self._fake_tensor(self.CPU_DEVICE, numel=256, element_size=4),
        }
        id_to_name_map = {1: "lm_head.weight", 3: "model.embed_tokens.weight"}  # id 2 deliberately unmapped

        with caplog.at_level("WARNING", logger="flextensor.loaders"):
            self._build(layers=[], tensors_map=tensors_map, id_to_name_map=id_to_name_map)

        msg = caplog.records[0].getMessage()
        assert "lm_head.weight (id=1)" in msg, "named ids must surface the symbolic name for actionability"
        assert "model.embed_tokens.weight (id=3)" in msg
        assert "id=2" in msg, "ids without a mapping must still be reported, not silently dropped"
        assert "None (id=2)" not in msg, "unmapped ids must not stringify the missing name as 'None'"

    # ------------------------------------------------------------------- pin

    def test_pin_adopts_passthrough_tensor(self) -> None:
        """``pin()`` adopts the given tensor as passthrough (shared storage).

        After construction, the only supported ``pin()`` semantics is
        passthrough — the tensor is owned by the model, so shutdown must
        not run ``del_tensor_func`` on it. Ownership can only be set at
        ``__init__`` time.
        """
        del_func = MagicMock()
        rescuer = self._build(
            layers=[self._layer("L1", [1])],
            tensors_map={1: self._fake_tensor(self.GPU_DEVICE)},
            del_tensor_func=del_func,
        )

        new_tensor = self._fake_tensor(self.GPU_DEVICE)
        rescuer.pin(99, new_tensor)
        assert rescuer.get(99) is new_tensor

        rescuer.shutdown()
        del_func.assert_not_called()

    # ----------------------------------------------------------- get / state

    def test_get_returns_none_for_unknown_id(self) -> None:
        rescuer = self._build(layers=[self._layer("L1", [1])], tensors_map={1: self._fake_tensor(self.GPU_DEVICE)})

        assert rescuer.get(424242) is None, "miss is the loader's signal to fall through to the next path"

    # -------------------------------------------------------------- shutdown

    def test_shutdown_runs_del_func_on_owned_entries(self) -> None:
        del_func = MagicMock()
        cpu_tensor = self._fake_tensor(self.CPU_DEVICE)
        gpu_copy = cpu_tensor.to.return_value
        rescuer = self._build(layers=[], tensors_map={1: cpu_tensor}, del_tensor_func=del_func)

        rescuer.shutdown()

        del_func.assert_called_once_with(gpu_copy)
        assert rescuer.get(1) is None, "shutdown must drop the local reference"

    def test_shutdown_does_not_run_del_func_on_passthrough_entries(self) -> None:
        """Passthrough storage is shared with the model — shutdown must not aggressively free it."""
        del_func = MagicMock()
        gpu_tensor = self._fake_tensor(self.GPU_DEVICE)
        rescuer = self._build(layers=[], tensors_map={1: gpu_tensor}, del_tensor_func=del_func)

        rescuer.shutdown()

        del_func.assert_not_called()
        assert rescuer.get(1) is None, "local reference must still be dropped"

    def test_shutdown_is_idempotent(self) -> None:
        """Second ``shutdown`` is a no-op — no re-iteration, no double-free."""
        del_func = MagicMock()
        rescuer = self._build(layers=[], tensors_map={1: self._fake_tensor(self.CPU_DEVICE)}, del_tensor_func=del_func)

        rescuer.shutdown()
        rescuer.shutdown()

        assert del_func.call_count == 1, "second shutdown must not re-call del_func on already-released entries"

    # -------------------------------------------------------------- OOM path

    def test_oom_mid_rescue_releases_partial_copies_and_re_raises_with_context(self) -> None:
        """OOM on iteration N must run ``del_func`` on N-1 already-allocated copies.

        Without the try/except + ``shutdown()`` cleanup in ``__init__``, the
        partially-built rescuer goes out of scope and the caching allocator
        keeps the N-1 storages until GC; the follow-up OOM error message
        can itself OOM. The re-raise must stay typed as ``OutOfMemoryError``
        so downstream OOM handlers still catch it (a refactor that bottles
        it into ``RuntimeError`` would silently slip past those handlers).
        """
        del_func = MagicMock()

        first = self._fake_tensor(self.CPU_DEVICE)
        second = self._fake_tensor(self.CPU_DEVICE)
        second.to.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        tensors_map = {1: first, 2: second}

        with pytest.raises(torch.cuda.OutOfMemoryError, match=r"pinning 2 tensor"):
            self._build(layers=[], tensors_map=tensors_map, del_tensor_func=del_func)

        (
            del_func.assert_called_once_with(first.to.return_value),
            (
                "the one already-allocated owned copy must be routed through del_func; "
                "otherwise OOM recovery is impossible and the storage stays in the "
                "caching allocator until GC"
            ),
        )


# ---------------------------------------------------------------------------
# loader_type='strategy': the runtime fall-through bug, now closed at the loader.
# ---------------------------------------------------------------------------


class TestStrategyLoaderUntimedRescue:
    """``loader_type='strategy'``: dropped tensor IDs are rescued by preload.

    Codex iter3's diagnosis was correct in spirit — without this rescue,
    the strategy loader would have no entry for an untimed tensor, would
    return ``None`` from ``.get(...)``, and both the property-getter
    path and the ``TrapInfer`` path would fall through to the original
    CPU tensor. The fix in
    :func:`flextensor.loaders._compute_untimed_traced_preload` adds
    those tensor IDs to the preload set at ``__init__`` time so the
    loader has a GPU copy ready. The narrowing via
    ``reachable_tensor_ids`` keeps the rescue scoped to model-owned
    tensors so future dynamic-discovery code paths can't push transient
    tensors into the rescue set. The trap-path tests stay as contract
    tests on the fall-through behaviour itself.
    """

    def _make_loader_without_untimed(
        self,
        untimed_tensor: torch.Tensor,
        *,
        reachable_tensor_ids: set[int] | None = None,
    ) -> TensorStrategyLoader:
        """Build a TensorStrategyLoader that knows ``untimed_tensor`` (in
        ``tensors_map``) but has no strategy entry referencing it — exactly
        what ``compute_layer_statistics`` produces upstream when its label
        was dropped.

        ``reachable_tensor_ids`` defaults to "the timed tensor and the untimed
        tensor" — i.e. both belong to the live model — so the rescue fires.
        Tests that exercise the narrowing pass an explicit set.

        Uses a real CUDA device when available so the rescue's
        ``tensor.to(device=device_gpu, copy=True)`` actually moves to
        GPU; falls back to CPU otherwise (CPU→CPU ``.to()`` is identity
        and exercises the routing logic without GPU placement).
        """
        timed_tensor = torch.zeros(4, dtype=torch.float32)
        tensors_map = {
            id(timed_tensor): timed_tensor,
            id(untimed_tensor): untimed_tensor,
        }
        timed_info = TensorStatistics(
            tensor_id=id(timed_tensor),
            name="timed",
            size_bytes=timed_tensor.numel() * timed_tensor.element_size(),
            load_time_ms=0.1,
        )
        layer_stats = [LayerStatistics(label="L1", tensors=[timed_info], duration=1.0)]
        strategy_map = {"L1": [timed_info]}
        release_strategy_map = {"L1": [timed_info]}

        device_gpu = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        if reachable_tensor_ids is None:
            reachable_tensor_ids = {id(timed_tensor), id(untimed_tensor)}

        patchers = _patched_cuda()
        try:
            return TensorStrategyLoader(
                layer_stats=layer_stats,
                strategy_map=strategy_map,
                release_strategy_map=release_strategy_map,
                tensors_map=tensors_map,
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
                reachable_tensor_ids=reachable_tensor_ids,
            )
        finally:
            _stop_patches(patchers)

    def test_loader_preloads_untimed_traced_tensor(self) -> None:
        """The fix: untimed tensor in ``tensors_map`` is preloaded at init.

        Without ``_compute_untimed_traced_preload``, this tensor would be
        absent from ``cpu_to_gpu_map`` (``_compute_preload`` only iterates
        ``layer_stats``-listed tensors, and ``enter()`` only transfers
        tensors listed in ``strategy_map[label]`` — both post-filter, both
        miss the dropped row). The rescue puts the tensor in the preload
        set so ``__init__`` copies it to GPU eagerly.
        """
        untimed = torch.zeros(8, dtype=torch.float32)
        loader = self._make_loader_without_untimed(untimed)

        rescued = loader.get(id(untimed))

        assert rescued is not None, (
            "TensorStrategyLoader must preload untimed traced tensors — otherwise "
            "the trap path falls through to the original CPU tensor"
        )
        if torch.cuda.is_available():
            assert rescued.device.type == "cuda", "rescued tensor must live on the GPU device, not on CPU"

    def test_loader_returns_none_for_unknown_tensor_id(self) -> None:
        """Negative control: tensors that were never registered stay unknown.

        The rescue is scoped to ``tensors_map`` membership — a tensor ID
        the manager never knew about must still produce ``None`` from
        ``.get(...)`` so the trap path's ``is_traced`` check (which
        gates the loader lookup) remains the authoritative gate.
        """
        untimed = torch.zeros(8, dtype=torch.float32)
        loader = self._make_loader_without_untimed(untimed)

        # An arbitrary integer that's not the id of any tensor we registered.
        spurious_id = id(self) ^ 0xDEADBEEF
        assert loader.get(spurious_id) is None

    def test_reachable_tensor_ids_narrows_rescue(self) -> None:
        """Defence-in-depth: tensors not reachable from the model are excluded.

        When ``reachable_tensor_ids`` does not include the untimed
        tensor's id, the rescue must skip it — even though the tensor
        is still in ``tensors_map``. This protects against any future
        code path that registers a transient or non-model tensor into
        ``tensors_map``: the strategy loader will not force-pin it to
        GPU.
        """
        untimed = torch.zeros(8, dtype=torch.float32)
        # Construct a reachable set that DOES NOT include the untimed tensor —
        # simulates a future scenario where something added it to
        # tensors_map but the live model doesn't reach it.
        reachable = {id(untimed) ^ 0xDEADBEEF}  # arbitrary id, not the untimed one
        loader = self._make_loader_without_untimed(untimed, reachable_tensor_ids=reachable)

        assert loader.get(id(untimed)) is None, (
            "tensor outside reachable_tensor_ids must be excluded from the rescue — "
            "this is the defence-in-depth that prevents force-pinning of non-model tensors"
        )

    def test_reachable_tensor_ids_none_preserves_broad_rescue(self) -> None:
        """Backward-compat: ``reachable_tensor_ids=None`` rescues everything.

        Construction paths that do not pass ``reachable_tensor_ids``
        (e.g. external callers, older tests) keep the broad rescue
        semantics — every tensor in ``tensors_map`` not appearing in
        any layer is preloaded.
        """
        untimed = torch.zeros(8, dtype=torch.float32)
        # Bypass the fixture's auto-populated reachable set by passing None
        # explicitly via _make_loader_without_untimed's keyword.
        timed_tensor = torch.zeros(4, dtype=torch.float32)
        tensors_map = {id(timed_tensor): timed_tensor, id(untimed): untimed}
        timed_info = TensorStatistics(tensor_id=id(timed_tensor), name="timed", size_bytes=16, load_time_ms=0.1)
        layer_stats = [LayerStatistics(label="L1", tensors=[timed_info], duration=1.0)]

        device_gpu = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        patchers = _patched_cuda()
        try:
            loader = TensorStrategyLoader(
                layer_stats=layer_stats,
                strategy_map={"L1": [timed_info]},
                release_strategy_map={"L1": [timed_info]},
                tensors_map=tensors_map,
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
                reachable_tensor_ids=None,
            )
        finally:
            _stop_patches(patchers)

        assert loader.get(id(untimed)) is not None, (
            "with reachable_tensor_ids=None the rescue must keep its broad scope — "
            "this preserves the contract for callers that opt out of the narrowing"
        )

    def test_rescue_emits_warning_with_count_bytes_and_ids(self, caplog: pytest.LogCaptureFixture) -> None:
        """The rescue's WARNING is the ONLY signal that profile coverage is
        incomplete — pin its content so a future refactor that drops the
        warning, lowers the level, or omits the byte/ID hint can't ship green.
        """
        untimed = torch.zeros(8, dtype=torch.float32)

        with caplog.at_level("WARNING", logger="flextensor.loaders"):
            self._make_loader_without_untimed(untimed)

        rescue_records = [r for r in caplog.records if "Untimed-trap rescue activated" in r.getMessage()]
        assert rescue_records, f"rescue must emit a WARNING; got: {[r.getMessage() for r in caplog.records]}"
        msg = rescue_records[0].getMessage()
        assert "1 tensor(s)" in msg, f"warning must report rescued count: {msg}"
        assert "MiB" in msg, f"warning must report rescued bytes: {msg}"
        assert str(id(untimed)) in msg, (
            f"warning must include the rescued tensor id ({id(untimed)}) for cross-reference: {msg}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to exercise the device equality branch")
    def test_preload_passthrough_when_canonical_already_on_gpu(self) -> None:
        """Cross-ref tensors mutated to GPU by ``_make_tensor_getter`` land in
        the rescue set after the cross-ref filter in ``prepare_infer_mode``
        drops them from ``layer_stats``. Their ``.data`` is already on the
        target device, so a ``tensor.to(device_gpu, copy=True)`` here would
        permanently double their footprint. The preload loop must share the
        canonical storage instead, mirroring :class:`UntimedTrapRescuer`'s
        passthrough branch.
        """
        untimed = torch.zeros(8, dtype=torch.float32, device="cuda:0")
        loader = self._make_loader_without_untimed(untimed)

        rescued = loader.get(id(untimed))

        assert rescued is untimed, (
            "preload must share storage with the canonical when it's already on the target "
            "device — otherwise cross-ref tensors leak a permanent 2x GPU footprint"
        )

    def test_release_never_frees_a_passthrough_preloaded_tensor(self) -> None:
        """A passthrough entry is the canonical model tensor, not a loader copy.

        ``release_for_label`` calls ``del_tensor`` on ``cpu_to_gpu_map`` entries.
        For a passthrough id that is ``clear_and_delete_tensor`` on live model
        weights — silent corruption, not a crash. The disjointness of
        ``preload_ids`` and ``release_strategy_map`` was previously an untested
        invariant maintained by two separate callers.
        """
        # Must already sit on the loader's target device for the passthrough
        # branch to fire; the helper uses CUDA when it is available.
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        untimed = torch.zeros(8, dtype=torch.float32, device=device)
        loader = self._make_loader_without_untimed(untimed)
        assert id(untimed) in loader._passthrough_preload_ids, "precondition: entry must be passthrough"

        loader.del_tensor = MagicMock()
        # Force the hazardous overlap the invariant is supposed to prevent.
        loader.release_strategy_map["overlap"] = [
            TensorStatistics(tensor_id=id(untimed), name="untimed", size_bytes=untimed.nbytes, load_time_ms=0.1)
        ]

        loader.release_for_label("overlap")

        loader.del_tensor.assert_not_called()
        assert loader.get(id(untimed)) is untimed, "the canonical tensor must survive the release"

    def test_release_still_frees_owned_copies(self) -> None:
        """The guard must not turn release into a no-op for real copies."""
        untimed = torch.zeros(8, dtype=torch.float32)
        loader = self._make_loader_without_untimed(untimed)

        owned_id = 987654321
        owned = torch.zeros(4, dtype=torch.float32)
        loader.cpu_to_gpu_map[owned_id] = owned
        loader.del_tensor = MagicMock()
        loader.release_strategy_map["owned"] = [
            TensorStatistics(tensor_id=owned_id, name="owned", size_bytes=owned.nbytes, load_time_ms=0.1)
        ]

        loader.release_for_label("owned")

        loader.del_tensor.assert_called_once()
        assert owned_id not in loader.cpu_to_gpu_map

    def test_property_getter_falls_through_when_loader_returns_none(self) -> None:
        """Contract: direct-mode getter falls through to ``tensor_ref`` on ``None``.

        The strategy loader's rescue keeps this path from being reached for
        traced tensors in production, but the precondition has to hold:
        if ``loader.get(...)`` ever returned ``None`` for a traced tensor
        on a manager constructed without a ``device_gpu`` (only really
        possible in test harnesses — production goes through
        :meth:`OffloadManager._initialize_tensor_manager` which always
        passes a ``torch.device``), the patched module property surfaces
        the original (CPU) ``tensor_ref``. The cross-layer copy in
        :func:`_make_tensor_getter` only fires when ``device_gpu`` is
        configured; this test pins the unconfigured-manager fall-through.
        """
        untimed_cpu = torch.zeros(8, dtype=torch.float32)
        assert untimed_cpu.device.type == "cpu", "fixture sanity"

        tm = MagicMock()
        tm.is_traced_by_id.return_value = True
        tm.tensor_layer_loader = MagicMock()
        tm.tensor_layer_loader.get.return_value = None
        tm.device_gpu = None  # unconfigured manager — exercises the fall-through branch

        getter = _make_tensor_getter(untimed_cpu, tm, missing_fields=[])
        result = getter(MagicMock())

        assert result is untimed_cpu, "the patched module property still returns the original (CPU) tensor"
        assert result.device.type == "cpu", "and that tensor is on CPU — what flows into the next forward op"

    def test_trap_infer_falls_through_when_loader_returns_none(self) -> None:
        """Contract: TorchFunctionMode path also falls through on ``None``.

        Same contract as the property-getter test above, applied to the
        non-direct-mode path (``trap_tensor_mode.py:150-172``). Whether
        the resulting op crashes (heterogeneous device args) or silently
        runs on CPU (homogeneous CPU args) is a property of PyTorch's
        dispatcher — what flextensor controls is whether the CPU tensor
        reaches the dispatcher in the first place. The loader's rescue
        ensures this branch isn't taken for traced tensors in
        production.
        """
        untimed_cpu = torch.zeros(8, dtype=torch.float32)

        tm = MagicMock()
        tm.is_traced.return_value = True
        tm.tensor_layer_loader = MagicMock()
        tm.tensor_layer_loader.get.return_value = None

        trap = TrapInfer(tm, "untimed_layer", torch.device("cpu"))

        seen_args: list[tuple] = []

        def echo(*args: object, **kwargs: object) -> tuple[object, ...]:
            del kwargs
            seen_args.append(args)
            return args

        trap.__torch_function__(echo, [], (untimed_cpu,))

        assert len(seen_args) == 1
        forwarded = seen_args[0]
        assert forwarded[0] is untimed_cpu, (
            "TrapInfer forwarded the original CPU tensor unchanged — the runtime "
            "fall-through bug for loader_type='strategy' is reproduced"
        )
        assert forwarded[0].device.type == "cpu"

    def test_trap_infer_uses_loader_tensor_when_loader_has_entry(self) -> None:
        """Negative control: when the loader DOES have a GPU entry, that's what flows.

        Pins the contract that the fall-through is conditioned on
        ``loader.get(...) is None`` — if the upstream fix were ever to
        synthesize an entry for untimed-but-parameterized tensors, this
        is the path it would route through.
        """
        original_cpu = torch.zeros(8, dtype=torch.float32)
        replacement = torch.ones(8, dtype=torch.float32)

        tm = MagicMock()
        tm.is_traced.return_value = True
        tm.tensor_layer_loader = MagicMock()
        tm.tensor_layer_loader.get.return_value = replacement

        trap = TrapInfer(tm, "timed_layer", torch.device("cpu"))

        seen_args: list[tuple] = []

        def echo(*args: object, **kwargs: object) -> tuple[object, ...]:
            del kwargs
            seen_args.append(args)
            return args

        trap.__torch_function__(echo, [], (original_cpu,))

        forwarded = seen_args[0]
        assert forwarded[0] is replacement, "loader's tensor wins when present"
        assert forwarded[0] is not original_cpu


# ---------------------------------------------------------------------------
# Block loaders: the safety net rescues untimed tensors at finalization.
# ---------------------------------------------------------------------------


class TestBlockLoaderUntimedSafetyNet:
    """Block loaders move untimed tensors to GPU via ``MoveUnmappedTensorsToGPUProcessor``.

    Codex iter3's framing implied silent CPU compute / device-mismatch
    at runtime. For block loaders that's wrong: the runtime is safe by
    construction because ``_prepare_view_model_from_id_to_view_map``
    moves anything not in ``block_controller.tensor_id_to_view_map``
    to GPU before inference begins.

    The safety net also participates in memory protection: strategy
    planning reserves bytes for reachable tensors missing from layer
    statistics, and the processor checks CUDA free memory before an
    unmapped move.
    """

    def test_unmapped_processor_routes_untimed_to_move_to_gpu(self) -> None:
        """Untimed tensor (not in view map) goes through ``move_to_gpu.process``.

        This is the empirical answer to Codex's "silent CPU compute"
        claim for block loaders: there is no path by which an untimed
        tensor reaches inference still on CPU — the per-tensor
        traversal in ``MoveUnmappedTensorsToGPUProcessor`` catches it
        first.
        """
        timed = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        untimed = nn.Parameter(torch.zeros(8, dtype=torch.float32))

        gpu_view_for_timed = torch.zeros(4, dtype=torch.float32)
        tensor_id_to_view_map = {id(timed): gpu_view_for_timed}

        device_gpu = torch.device("cpu")
        proc = MoveUnmappedTensorsToGPUProcessor(device_gpu, tensor_id_to_view_map)

        with patch.object(proc.move_to_gpu, "process", wraps=proc.move_to_gpu.process) as spy:
            result_timed = proc.process(timed)
            result_untimed = proc.process(untimed)

        moved_tensor_ids = [id(c.args[0]) for c in spy.call_args_list]
        assert id(untimed) in moved_tensor_ids, (
            "untimed tensor must be routed through move_to_gpu — this is the safety "
            "net that prevents the strategy-loader-style fall-through for block loaders"
        )
        assert id(timed) not in moved_tensor_ids, (
            "timed (view-mapped) tensor must NOT be redundantly moved — "
            "TensorReplacementProcessor will swap it for the view later"
        )
        assert result_timed is gpu_view_for_timed, "timed tensor returns its view"
        assert result_untimed is not None

    def test_unmapped_processor_checks_cuda_free_memory_before_move(self) -> None:
        """The auto-pin path has a move-time guard before touching CUDA memory."""

        tensor = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        tensor_bytes = tensor.numel() * tensor.element_size()
        proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cuda:0"), tensor_id_mapping={})

        with (
            patch.object(torch.cuda, "mem_get_info", return_value=(tensor_bytes - 1, tensor_bytes * 2)),
            patch.object(proc.move_to_gpu, "process", return_value=tensor) as mock_move,
            pytest.raises(RuntimeError, match=r"Insufficient GPU memory.*unmapped tensor"),
        ):
            proc.process(tensor)

        mock_move.assert_not_called()

    def test_unmapped_processor_moves_when_cuda_free_memory_is_sufficient(self) -> None:
        """The auto-pin path proceeds when the move-time CUDA guard has room."""

        tensor = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        tensor_bytes = tensor.numel() * tensor.element_size()
        proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cuda:0"), tensor_id_mapping={})

        with (
            patch.object(torch.cuda, "mem_get_info", return_value=(tensor_bytes + 32 * 1024**2, tensor_bytes * 4)),
            patch.object(proc.move_to_gpu, "process", return_value=tensor) as mock_move,
        ):
            result = proc.process(tensor)

        assert result is tensor
        mock_move.assert_called_once()
        assert mock_move.call_args.args[0] is tensor

    def test_unmapped_bytes_accounting_for_real_gpu_target(self) -> None:
        """When the move target is a real CUDA device, bytes are accounted.

        Skipped on CPU-only runners; the byte counter only increments
        when ``new_tensor.device.type == 'cuda'`` (see
        ``MoveUnmappedTensorsToGPUProcessor.process``). This test
        guards the attribution that ``get_gpu_memory_usage().unmapped_tensors_mb``
        relies on.
        """
        if not torch.cuda.is_available():
            import pytest

            pytest.skip("requires CUDA for byte accounting on the move-to-gpu path")

        device_gpu = torch.device("cuda:0")
        untimed = nn.Parameter(torch.zeros(8, dtype=torch.float32))
        proc = MoveUnmappedTensorsToGPUProcessor(device_gpu, tensor_id_mapping={})

        proc.process(untimed)

        assert proc.unmapped_gpu_bytes == untimed.numel() * untimed.element_size(), (
            "unmapped bytes must reflect the auto-pinned untimed tensor — "
            "this is the only signal a user has that profile coverage regressed"
        )

    def test_unmapped_bytes_exclude_view_mapped_tensors(self) -> None:
        """View-mapped tensors are not counted as unmapped.

        The view-map branch never enters the ``device.type == 'cuda'``
        accounting path — there's no move — so the byte counter must
        stay at zero. Runs on CPU.
        """
        view_target = torch.zeros(4, dtype=torch.float32)
        view_mapped = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cpu"), tensor_id_mapping={id(view_mapped): view_target})

        proc.process(view_mapped)

        assert proc.unmapped_gpu_bytes == 0, "view-mapped tensors must NOT increment unmapped bytes"

    def test_apply_resets_unmapped_bytes(self) -> None:
        """``apply()`` must zero the bytes counter so reuse doesn't accumulate.

        The processor is reused across phases; stale bytes from a prior
        run would inflate ``get_gpu_memory_usage().unmapped_tensors_bytes``
        and break the ``blocks + unmapped == total`` contract documented
        in ``test_gpu_memory_usage_integration.py``.
        """
        proc = MoveUnmappedTensorsToGPUProcessor(torch.device("cpu"), tensor_id_mapping={})
        proc.unmapped_gpu_bytes = 1234

        proc.apply(nn.Linear(2, 2))

        assert proc.unmapped_gpu_bytes == 0, "apply() must reset unmapped_gpu_bytes"


# ---------------------------------------------------------------------------
# Codex iter3 symmetric flaw: paused-only-observed labels are still rescued.
# ---------------------------------------------------------------------------


class TestPauseSuppressedRescue:
    """Paused-only-observed labels collapse into the same rescue set.

    Codex iter3 framed a separate concern from F1: when ``record_all``
    no-ops during a paused warmup pass (the fix from 4bc0c98 that
    prevents pollution of per-layer tensor sets), a label whose
    *only* sighting happens during that paused pass becomes invisible
    to ``layer_stats``. Combined with the dummy-input limitation of
    DISCOVERY / real profiling, the claim is that for data-dependent
    models (MoE / conditional branches) the dropped tensor flows
    through the trap path on CPU at inference time.

    Empirically, for **statically-registered** experts the runtime
    chain is already broken by the F1 rescue:

    * ``preprocess_model``'s :class:`TensorMappingProcessor` walks the
      module structure and registers every reachable tensor in
      ``tensors_map`` regardless of forward-pass routing.
    * The strategy loader's rescue
      (:func:`flextensor.loaders._compute_untimed_traced_preload`)
      returns ``tensors_map - layer_tensor_ids``. It does not
      distinguish *why* a label is missing — silent-drop, paused
      suppression, dummy-input miss all collapse into the same set.
    * The reachability narrowing
      (:func:`flextensor.tensor_processors.compute_reachable_tensor_ids`)
      sees the expert's params (they're structural ``nn.Parameter`` s),
      so they survive the intersection.

    This test plugs that argument empirically end-to-end: real
    two-expert MoE, real ``compute_reachable_tensor_ids`` walk,
    real ``TensorStrategyLoader`` rescue. Expert A is the
    paused-only-observed one (no ``layer_stats`` row); expert B is
    timed normally.

    What this does NOT cover: dynamically-instantiated experts whose
    weights only materialise inside a paused forward (lazy weights,
    runtime-grown ``ModuleDict``). Those aren't in ``tensors_map``
    from preprocess and the rescue can't see them — tracked
    separately as a design-level concern.
    """

    class _TwoExpertMoE(nn.Module):
        """Minimal stand-in for an MoE — two statically-registered experts."""

        def __init__(self) -> None:
            super().__init__()
            self.expert_a = nn.Linear(4, 4, bias=False)
            self.expert_b = nn.Linear(4, 4, bias=False)

    def test_paused_only_observed_expert_is_rescued(self) -> None:
        model = self._TwoExpertMoE()

        tensors_map = {
            id(model.expert_a.weight): model.expert_a.weight,
            id(model.expert_b.weight): model.expert_b.weight,
        }

        b_weight = model.expert_b.weight
        b_info = TensorStatistics(
            tensor_id=id(b_weight),
            name="expert_b.weight",
            size_bytes=b_weight.numel() * b_weight.element_size(),
            load_time_ms=0.1,
        )
        layer_stats = [LayerStatistics(label="expert_b", tensors=[b_info], duration=1.0)]
        strategy_map = {"expert_b": [b_info]}
        release_strategy_map = {"expert_b": [b_info]}

        device_gpu = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        reachable_ids = compute_reachable_tensor_ids(model)

        patchers = _patched_cuda()
        try:
            loader = TensorStrategyLoader(
                layer_stats=layer_stats,
                strategy_map=strategy_map,
                release_strategy_map=release_strategy_map,
                tensors_map=tensors_map,
                device_gpu=device_gpu,
                release_tensors=False,
                stream_priority=0,
                reachable_tensor_ids=reachable_ids,
            )
        finally:
            _stop_patches(patchers)

        rescued = loader.get(id(model.expert_a.weight))
        assert rescued is not None, (
            "expert A's params are in tensors_map (preprocess_model's "
            "structural walk) and must be rescued by the strategy loader "
            "even though their only sighting was during a paused warmup — "
            "this is the empirical refutation of Codex iter3's runtime "
            "claim for static MoE / conditional-branch models"
        )
        if torch.cuda.is_available():
            assert rescued.device.type == "cuda"

        # Expert B is *not* in the preload set on purpose: it has a
        # layer_stats row, so it gets loaded mid-strategy via
        # ``enter("expert_b")`` rather than at __init__.  The point of
        # this test is the rescue, not the per-layer load path.
        assert id(b_weight) not in loader.preload_ids, "timed tensors don't need rescue — they're loaded by enter()"


# ---------------------------------------------------------------------------
# Cross-module reference pin (vLLM-shaped: child module's forward never runs;
# parent reads child.weight directly via attribute access).
# ---------------------------------------------------------------------------


def _mark_module_patched(module: nn.Module, name: str) -> None:
    """Mirror what ``OffloadManager._patch_module_forward`` writes."""
    module._ft_original_forward_func = type(module).forward  # noqa: SLF001
    module._ft_offload_name = name  # noqa: SLF001


class _Inner(nn.Module):
    """Stand-in for ``lm_head``: owns the weight, ``forward`` never runs."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        raise AssertionError("_Inner.forward must not be called in this test")


class _Outer(nn.Module):
    """Stand-in for ``logits_processor``: reads ``inner.weight`` directly."""

    def __init__(self, inner: _Inner) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.inner.weight)


class _Root(nn.Module):
    """vLLM-shaped topology: ``id(inner)`` reachable as both ``inner`` and ``outer.inner``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inner = _Inner(dim)
        self.outer = _Outer(self.inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(x)


class TestCrossModuleReferenceIsPinnedAtPreprocess:
    """Cross-referenced tensors are pinned to GPU at preprocess time and
    dropped from offload tracking, not handled by a runtime fallback."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_inner_weight_pinned_and_removed_from_offload_tracking(self) -> None:
        device_gpu = torch.device("cuda:0")
        root = _Root(dim=8)
        _mark_module_patched(root.inner, "inner")
        _mark_module_patched(root.outer, "outer")

        inner_weight_id = id(root.inner.weight)

        tm = TensorManager(
            device_gpu=device_gpu,
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
        )
        tm.model = root
        tm.tensors_map = {id(p): p for p in root.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())
        tm.set_skip_discovery(True)

        with patch("flextensor.tensor_manager.preprocess_model"):
            tm.initialize_warmup()

        assert root.inner.weight.device.type == "cuda"
        assert inner_weight_id not in tm.tensors_map
        assert inner_weight_id not in tm.traced_tensors

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_getter_does_not_invoke_runtime_copy_after_pin(self) -> None:
        device_gpu = torch.device("cuda:0")
        root = _Root(dim=8)
        _mark_module_patched(root.inner, "inner")
        _mark_module_patched(root.outer, "outer")

        tm = TensorManager(
            device_gpu=device_gpu,
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
        )
        tm.model = root
        tm.tensors_map = {id(p): p for p in root.parameters()}
        tm.traced_tensors = set(tm.tensors_map.keys())
        tm.set_skip_discovery(True)

        with patch("flextensor.tensor_manager.preprocess_model"):
            tm.initialize_warmup()

        pinned_weight = root.inner.weight
        spy = MagicMock(wraps=pinned_weight.to)
        with patch.object(pinned_weight, "to", spy):
            getter = _make_tensor_getter(pinned_weight, tm, missing_fields=[])
            result = getter(MagicMock())

        assert result is pinned_weight
        spy.assert_not_called()


# ---------------------------------------------------------------------------
# Self-healing runtime fallback for cross-module refs the preprocess-time
# detector misses (vLLM-shaped: lm_head passed as positional arg, not stored
# on the caller, so the multi-parent detector cannot see it).
# ---------------------------------------------------------------------------


def _wire_taint_state(tm: MagicMock) -> None:
    """Back the taint API methods with a single bool so tests can read/write
    via ``is_current_trap_tainted()`` instead of reaching for a private attr.
    """
    tm._active_trap_tainted = False  # noqa: SLF001

    def _mark() -> None:
        tm._active_trap_tainted = True  # noqa: SLF001

    def _reset() -> None:
        tm._active_trap_tainted = False  # noqa: SLF001

    def _is() -> bool:
        return tm._active_trap_tainted  # noqa: SLF001

    tm.mark_current_trap_tainted.side_effect = _mark
    tm.reset_current_trap_taint.side_effect = _reset
    tm.is_current_trap_tainted.side_effect = _is


def _fake_tensor_manager_for_getter(device_gpu: torch.device) -> MagicMock:
    """Build a manager stub exposing only the surface ``_make_tensor_getter`` reads.

    The real getter does not need the full ``TensorManager``; it only touches
    ``device_gpu``, ``is_traced_by_id``, ``tensor_layer_loader.get``,
    ``observed_cross_refs``, ``traced_tensors``, and ``mark_current_trap_tainted``.
    Mocking keeps the test focused on the fallback branch without dragging in
    CUDA streams or loaders.

    ``traced_tensors`` is a real ``set`` rather than a ``MagicMock`` so the
    fallback's ``discard(tensor_id)`` cleanup — which prunes the id from the
    offload-scheduling pool the moment it self-heals — is observable in tests.

    ``is_traced_by_id`` consults that same set rather than returning a constant
    ``True``. Pinning it ``True`` unconditionally made the ``discard`` untestable:
    the second-access test passed on the device check alone, so deleting the
    ``discard`` left every test green.
    """
    tm = MagicMock()
    tm.device_gpu = device_gpu
    tm.tensor_layer_loader.get.return_value = None
    tm.observed_cross_refs = set()
    tm.traced_tensors = set()
    tm.is_traced_by_id.side_effect = lambda tensor_id: tensor_id in tm.traced_tensors
    # A real Mapping: the cross-layer warning resolves ids through
    # ``format_tensor_id_hint``, whose ``id_to_name`` is beartype-checked.
    tm.tensor_id_to_name_map = {}
    _wire_taint_state(tm)
    return tm


class TestSelfHealingCrossModuleFallback:
    """Runtime fallback for cross-module refs the preprocess detector misses.

    The vLLM shape — ``logits_processor.forward(lm_head, …)`` with ``lm_head``
    passed positionally rather than stored — has no static signal: ``lm_head``
    has a single parent (the root) so the multi-parent detector returns it as
    "not cross-referenced". The getter catches the access at runtime and
    self-heals so subsequent reads are zero-overhead.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_first_access_mutates_data_records_id_and_taints_trap(self) -> None:
        device_gpu = torch.device("cuda:0")
        tm = _fake_tensor_manager_for_getter(device_gpu)
        tensor_ref = nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
        tm.traced_tensors.add(id(tensor_ref))
        assert tensor_ref.device.type == "cpu"

        getter = _make_tensor_getter(tensor_ref, tm, missing_fields=[])
        result = getter(MagicMock())

        assert result is tensor_ref, (
            "fallback must return the same Python object (downstream nn.Module "
            "machinery holds references to it); only the underlying .data is swapped"
        )
        assert tensor_ref.device.type == "cuda", (
            "tensor_ref.data must be GPU-resident after the rescue so subsequent "
            "reads short-circuit before re-entering the fallback branch"
        )
        assert id(tensor_ref) in tm.observed_cross_refs
        assert id(tensor_ref) not in tm.traced_tensors, (
            "fallback must drop the id from traced_tensors so subsequent reads "
            "skip the loader .get() round-trip and short-circuit on the "
            "is_traced_by_id guard"
        )
        assert tm.is_current_trap_tainted() is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_second_access_skips_copy_and_does_not_retaint(self) -> None:
        device_gpu = torch.device("cuda:0")
        tm = _fake_tensor_manager_for_getter(device_gpu)
        tensor_ref = nn.Parameter(torch.randn(4, 4, dtype=torch.float32))

        getter = _make_tensor_getter(tensor_ref, tm, missing_fields=[])
        getter(MagicMock())

        tm.reset_current_trap_taint()  # simulate next trap entry
        tm.mark_current_trap_tainted.reset_mock()

        spy = MagicMock(wraps=tensor_ref.data.to)
        with patch.object(tensor_ref.data, "to", spy):
            getter(MagicMock())

        spy.assert_not_called()
        tm.mark_current_trap_tainted.assert_not_called()
        assert tm.is_current_trap_tainted() is False

    def test_fallback_inert_when_device_gpu_is_unconfigured(self) -> None:
        """Mock-based tests where ``device_gpu`` is a ``MagicMock`` must not trip the fallback.

        The ``isinstance(device_gpu, torch.device)`` guard exists for exactly
        this contract — unit tests that don't set up a real device should
        keep the existing CPU fall-through behaviour.
        """
        tm = _fake_tensor_manager_for_getter(MagicMock())
        tensor_ref = nn.Parameter(torch.zeros(2, 2))

        getter = _make_tensor_getter(tensor_ref, tm, missing_fields=[])
        result = getter(MagicMock())

        assert result is tensor_ref
        assert tensor_ref.device.type == "cpu"
        assert tm.observed_cross_refs == set()
        assert tm.is_current_trap_tainted() is False


class TestPrepareInferModeFiltersObservedCrossRefs:
    """``prepare_infer_mode`` must drop runtime-detected cross-module refs
    from ``layer_stats`` before strategy compute.

    Why: the getter fallback mutates ``tensor_ref.data`` to GPU on first hit,
    so the tensor is permanently resident from that moment on. Leaving its id
    in ``layer_stats`` would let ``TensorStrategyLoader.enter()`` schedule a
    fresh ``tensor.to(device_gpu, copy=True)`` (a D2D allocation) and a
    matching ``release_for_label`` free per enter/exit cycle for an
    already-resident tensor, plus an extra entry in ``preload_ids``. The
    ``tensors_map``-level cleanup the preprocess pin does is not available
    here because ``_move_non_offloaded_tensors_to_gpu`` freezes
    ``tensors_map`` as ``MappingProxyType`` before the getter is installed.
    """

    @staticmethod
    def _prepare_tm(known_ids: set[int]) -> TensorManager:
        tm = TensorManager(
            device_gpu=torch.device("cuda:0"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            loader_type="strategy",
        )
        tm.layer_statistics_collector = MagicMock()
        tm.model = MagicMock(spec=[])
        # ``prepare_infer_mode`` calls ``build_parameters_mapping(self.model)``
        # and then drops any tensor id missing from the resulting name map. The
        # model here is a bare mock, so we stub the rebuild to keep the
        # "not a parameter" filter from masking the behaviour under test.
        tm.build_parameters_mapping = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda _m: tm.__dict__.__setitem__(
                "tensor_id_to_name_map", {tid: f"t{tid}" for tid in known_ids}
            )
        )
        return tm

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps", return_value=False)
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics", return_value=[])
    @patch("flextensor.tensor_manager.report_profiling_quality", return_value=None)
    def test_cross_ref_ids_dropped_before_strategy_compute(
        self, _report_quality, mock_compute_stats, _bench, _mem, _budget, _remove, _gaps, _loader
    ) -> None:
        cross_ref_id, normal_id = 7777, 8888
        tm = self._prepare_tm(known_ids={cross_ref_id, normal_id})
        # ``tensors_map`` keeps BOTH ids — the freeze contract guarantees the
        # entry survives; the filter must work without mutating the map.
        tm.tensors_map = {cross_ref_id: torch.zeros(2), normal_id: torch.zeros(2)}
        tm.layer_statistics_collector.get_layer_stats.return_value = [
            IterativeLayerStatistics(label="layer_0", tensor_ids=[cross_ref_id, normal_id], duration=10.0),
        ]
        tm.observed_cross_refs.add(cross_ref_id)
        strategy_result = MagicMock(strategy_map={"layer_0": []}, block_data=None)
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = strategy_result

        tm.prepare_infer_mode()

        (filtered_layer_stats, _stats_map), _ = mock_compute_stats.call_args
        all_ids = {tid for layer in filtered_layer_stats for tid in layer.tensor_ids}
        assert cross_ref_id not in all_ids, (
            "observed_cross_refs id leaked into the strategy input; the loader "
            "would schedule a per-cycle D2D copy for an already-resident tensor"
        )
        assert normal_id in all_ids, "filter must not drop unrelated tensor ids"

    @patch.object(TensorManager, "_create_loader")
    @patch("flextensor.tensor_manager.strategy_has_transfer_gaps", return_value=False)
    @patch("flextensor.tensor_manager.remove_layers_compound", side_effect=lambda s, *a: s)
    @patch("flextensor.tensor_manager.resolve_gpu_budget", return_value=1024**3)
    @patch.object(TensorManager, "_get_memory_transfer_stats", return_value={})
    @patch.object(TensorManager, "_benchmark_tensor_statistics", return_value={})
    @patch("flextensor.tensor_manager.compute_layer_statistics", return_value=[])
    @patch("flextensor.tensor_manager.report_profiling_quality", return_value=None)
    def test_no_cross_refs_leaves_layer_stats_unchanged(
        self, _report_quality, mock_compute_stats, _bench, _mem, _budget, _remove, _gaps, _loader
    ) -> None:
        normal_id = 4242
        tm = self._prepare_tm(known_ids={normal_id})
        tm.tensors_map = {normal_id: torch.zeros(2)}
        tm.layer_statistics_collector.get_layer_stats.return_value = [
            IterativeLayerStatistics(label="layer_0", tensor_ids=[normal_id], duration=10.0),
        ]
        assert tm.observed_cross_refs == set()
        strategy_result = MagicMock(strategy_map={"layer_0": []}, block_data=None)
        tm.tensor_manager_load_strategy = MagicMock()
        tm.tensor_manager_load_strategy.compute.return_value = strategy_result

        tm.prepare_infer_mode()

        (filtered_layer_stats, _stats_map), _ = mock_compute_stats.call_args
        all_ids = {tid for layer in filtered_layer_stats for tid in layer.tensor_ids}
        assert all_ids == {normal_id}, "no-op filter must not drop or add ids"


class TestTaintedTrapSkipsDurationRecording:
    """The active trap's ``__exit__`` drops the duration sample when tainted.

    Per-trap-call granularity: the first invocation that triggers the rescue
    is dropped from the duration aggregate; subsequent invocations on the
    same trap (and other trap labels) are recorded normally because the flag
    is reset at every ``__enter__``.
    """

    def _make_manager_with_collector(self) -> MagicMock:
        tm = MagicMock()
        _wire_taint_state(tm)
        tm.module_tracker = None
        tm.is_profiling_suspended.return_value = False
        tm.layer_statistics_collector = MagicMock()
        tm.record_duration = MagicMock()
        tm.record_all = MagicMock()
        tm.record_tensors = MagicMock()
        tm.trap_nesting_guard = MagicMock()
        return tm

    def test_trap_direct_skips_record_duration_when_tainted(self) -> None:
        from flextensor.trap_tensor_mode import TrapDirect

        tm = self._make_manager_with_collector()
        tm.tensor_layer_loader = MagicMock()
        tm.trap_start_event = MagicMock()
        tm.trap_end_event = MagicMock()
        tm.trap_start_event.elapsed_time.return_value = 42.0

        trap = TrapDirect(tm, trace_id="L1", device_gpu=torch.device("cpu"))
        trap.__enter__()
        tm.mark_current_trap_tainted()  # rescue fired inside the window
        trap.__exit__(None, None, None)

        tm.record_duration.assert_not_called()

    def test_trap_direct_records_duration_when_not_tainted(self) -> None:
        from flextensor.trap_tensor_mode import TrapDirect

        tm = self._make_manager_with_collector()
        tm.tensor_layer_loader = MagicMock()
        tm.trap_start_event = MagicMock()
        tm.trap_end_event = MagicMock()
        tm.trap_start_event.elapsed_time.return_value = 42.0

        trap = TrapDirect(tm, trace_id="L1", device_gpu=torch.device("cpu"))
        trap.__enter__()
        trap.__exit__(None, None, None)

        tm.record_duration.assert_called_once_with("L1", 42.0)

    def test_trap_records_tensors_only_when_tainted(self) -> None:
        """Tainted PROFILING exit routes through ``record_tensors`` with the
        PROFILING-side suspension opt-in (``respect_suspension=True``).

        The DISCOVERY-side default (``WarmupTrap``) is the other call to
        ``record_tensors`` and uses ``respect_suspension=False`` (the
        bypass-suspension default).
        """
        from flextensor.trap_tensor_mode import Trap

        tm = self._make_manager_with_collector()
        tm.tensor_layer_loader = MagicMock()
        tm.trap_start_event = MagicMock()
        tm.trap_end_event = MagicMock()
        tm.trap_start_event.elapsed_time.return_value = 42.0

        trap = Trap(tm, trace_id="L1", device_gpu=torch.device("cpu"))
        trap.__enter__()
        trap.tensors_ids = {123, 456}
        tm.mark_current_trap_tainted()
        trap.__exit__(None, None, None)

        tm.record_all.assert_not_called()
        tm.record_tensors.assert_called_once_with("L1", {123, 456}, respect_suspension=True)


class TestTaintedTrapWithRealCollectorEndToEnd:
    """End-to-end: ``Trap.__exit__`` + real collector + suspension state.

    The two existing tests (``..._passes_respect_suspension_so_paused_pass_is_dropped``
    and ``test_record_tensors_respect_suspension_does_not_widen_tensor_set``)
    pin the seam from each side (trap forwards the kwarg; collector honours
    the kwarg) but not the full chain. This class wires both halves together
    so a refactor that re-routes ``Trap.__exit__``'s tainted branch around
    ``record_tensors`` — or otherwise breaks the end-to-end invariant
    without touching either seam test — gets caught.

    Invariant under test: a tainted PROFILING trap exiting inside a
    ``suspend_profiling()`` window must NOT widen the per-label tensor
    manifest with that iteration's ``tensors_ids``. This is the
    data-dependent-widening defence on MoE / conditional-branch /
    mixed-batch profiles.
    """

    @staticmethod
    def _make_tm_with_real_collector() -> TensorManager:
        """``TensorManager`` built field-by-field with the real collector + suspender.

        Mirrors :func:`tests.unit.test_profiling_control._make_tm` (bypassing
        ``__init__`` and populating only what ``Trap.__exit__`` reaches) and
        adds the trap-side fixtures (events, nesting guard, loader stub) so
        a real ``Trap`` can be entered and exited end-to-end on a CPU host.
        """
        with patch.object(TensorManager, "__init__", lambda self, *a, **kw: None):
            tm = TensorManager.__new__(TensorManager)
        tm.layer_statistics_collector = IterativeLayerStatisticsCollector()
        tm._profiling_suspender = ProfilingSuspender()  # noqa: SLF001
        tm.trap_nesting_guard = TrapNestingGuard()
        tm._active_trap_tainted = False  # noqa: SLF001
        tm.module_tracker = None
        tm.tensor_layer_loader = MagicMock()
        tm.trap_start_event = MagicMock()
        tm.trap_end_event = MagicMock()
        tm.trap_start_event.elapsed_time.return_value = 1.0
        return tm

    def _run_tainted_trap(
        self,
        tm: TensorManager,
        label: str,
        tensor_ids: set[int],
    ) -> None:
        """Drive a single ``Trap`` window that marks taint mid-flight."""
        from flextensor.trap_tensor_mode import Trap

        trap = Trap(tm, trace_id=label, device_gpu=torch.device("cpu"))
        trap.__enter__()
        try:
            trap.tensors_ids = set(tensor_ids)
            tm.mark_current_trap_tainted()
        finally:
            trap.__exit__(None, None, None)

    def test_tainted_trap_during_suspension_does_not_widen_tensor_set(self) -> None:
        tm = self._make_tm_with_real_collector()

        self._run_tainted_trap(tm, "L1", {1, 2})

        tm.suspend_profiling()
        try:
            self._run_tainted_trap(tm, "L1", {3, 4})
        finally:
            tm.resume_profiling()

        union = tm.layer_statistics_collector.get_union_tensor_ids()
        assert union["L1"] == {1, 2}, (
            "tainted trap inside suspend_profiling() must not widen the per-label set — "
            "this is the MoE / conditional-branch defence; a refactor that re-routes "
            "the tainted branch around respect_suspension semantics would silently "
            "regress this"
        )


class TestMarkTaintIsScopedToActiveTrapWindow:
    """``mark_current_trap_tainted`` must no-op when no trap is active.

    The property getter installed by ``extend_nn_module`` can fire from
    *anywhere* Python touches a module attribute — between traps, during
    setup, in user hooks, during inference. Without this guard, any such
    fallback would set ``_active_trap_tainted=True`` and the next profile
    trap's ``__exit__`` would drop an innocent duration sample.
    """

    @staticmethod
    def _make_tm() -> TensorManager:
        return TensorManager(
            device_gpu=torch.device("cuda:0"),
            tensor_manager_load_strategy=MagicMock(),
            pinned_memory=False,
            loader_type="strategy",
        )

    def test_mark_is_noop_when_no_trap_active(self) -> None:
        tm = self._make_tm()
        assert tm.trap_nesting_guard.is_active() is False, "fixture sanity"

        tm.mark_current_trap_tainted()

        assert tm.is_current_trap_tainted() is False, (
            "getter fallbacks firing outside any trap window must not leak taint "
            "into the next profile trap's duration sample"
        )

    def test_mark_sets_flag_when_trap_active(self) -> None:
        tm = self._make_tm()
        tm.trap_nesting_guard.acquire("L1")
        try:
            tm.mark_current_trap_tainted()
            assert tm.is_current_trap_tainted() is True, (
                "rescue firing inside an active trap window must mark taint so "
                "the matching __exit__ drops the corrupted duration"
            )
        finally:
            tm.trap_nesting_guard.release()

    def test_mark_is_noop_during_inference_trap(self) -> None:
        """Inference traps don't acquire the nesting guard, so taint marking
        from a runtime getter fallback during inference must be a no-op.
        This is the contract — ``TrapInfer`` / ``TrapInferDirect`` have no
        consumer for the flag, so marking would be dead state that could
        later poison a re-profiling run if mode transitions are reused.
        """
        from flextensor.trap_tensor_mode import TrapInferDirect

        tm = self._make_tm()
        tm.tensor_layer_loader = MagicMock()

        trap = TrapInferDirect(tm, trace_id="L1", device_gpu=torch.device("cpu"))
        trap.__enter__()
        try:
            tm.mark_current_trap_tainted()
            assert tm.is_current_trap_tainted() is False, (
                "mark must no-op inside an inference trap — the inference trap "
                "classes don't consume this flag, so marking it would be a "
                "cross-mode leak waiting to happen"
            )
        finally:
            trap.__exit__(None, None, None)


class TestTrapEnterResetsStaleTaint:
    """Each trap ``__enter__`` clears stale taint left by an earlier window.

    Belt-and-suspenders with :class:`TestMarkTaintIsScopedToActiveTrapWindow`:
    the guard prevents the leak in the first place, but a manual
    ``__enter__``/``__exit__`` caller that hits an exception inside the
    window without going through ``with`` would still leave the flag set.
    Resetting at enter matches the contract documented in
    :class:`TestTaintedTrapSkipsDurationRecording`'s docstring.
    """

    @staticmethod
    def _make_tm_stub() -> MagicMock:
        tm = MagicMock()
        _wire_taint_state(tm)
        tm.module_tracker = None
        tm.is_profiling_suspended.return_value = False
        tm.layer_statistics_collector = MagicMock()
        tm.tensor_layer_loader = MagicMock()
        tm.trap_start_event = MagicMock()
        tm.trap_end_event = MagicMock()
        tm.trap_start_event.elapsed_time.return_value = 1.0
        tm.trap_nesting_guard = MagicMock()
        return tm

    def test_trap_direct_enter_clears_stale_taint(self) -> None:
        from flextensor.trap_tensor_mode import TrapDirect

        tm = self._make_tm_stub()
        tm.mark_current_trap_tainted()
        assert tm.is_current_trap_tainted() is True, "fixture sanity: pre-set stale taint"

        trap = TrapDirect(tm, trace_id="L1", device_gpu=torch.device("cpu"))
        trap.__enter__()
        try:
            assert tm.is_current_trap_tainted() is False
        finally:
            trap.__exit__(None, None, None)

    def test_trap_enter_clears_stale_taint(self) -> None:
        from flextensor.trap_tensor_mode import Trap

        tm = self._make_tm_stub()
        tm.mark_current_trap_tainted()
        assert tm.is_current_trap_tainted() is True, "fixture sanity: pre-set stale taint"

        trap = Trap(tm, trace_id="L1", device_gpu=torch.device("cpu"))
        trap.__enter__()
        try:
            assert tm.is_current_trap_tainted() is False
        finally:
            # Must call __exit__ to unregister the TorchFunctionMode globally;
            # leaving it registered would invoke Trap.__torch_function__ on the
            # next torch op anywhere in the test process.
            trap.__exit__(None, None, None)

    # WarmupTrap intentionally does NOT reset taint: the getter that sets the
    # flag (``_make_tensor_getter``) is only installed on profile/inference
    # paths, never during warmup, so the flag is structurally unsettable
    # inside a warmup window. Asserting reset there would pin a contract that
    # the class does not (and need not) enforce.


class TestCrossLayerAccessIsReported:
    """The cross-layer promotion must be observable at the moment it happens.

    ``_report_cross_layer_access`` runs only from ``prepare_infer_mode``, so
    during INFERENCE (and on the restored-profile path) it never fires. Without
    a per-tensor warning here the promotion is silent, and each one permanently
    pins a CPU master weight to GPU — visible later only as unexplained memory
    growth.

    The promotion itself needs a real second device, so the tensor is stubbed:
    these tests pin the *reporting* contract (fires once, names the tensor),
    which is what was missing, and they stay runnable on CPU-only hosts. The
    promotion's own behaviour is covered by the CUDA-gated tests above.
    """

    def _manager(self, name: str = "lm_head.weight") -> MagicMock:
        tm = _fake_tensor_manager_for_getter(torch.device("cuda:0"))
        tm.tensor_id_to_name_map = {}
        self._name = name
        return tm

    def _stub_tensor(self, tm: MagicMock) -> MagicMock:
        """A tensor stub on a device that differs from ``tm.device_gpu``."""
        tensor_ref = MagicMock()
        tensor_ref.device = torch.device("cpu")
        tensor_ref.nbytes = 4 * 1024 * 1024
        tm.tensor_id_to_name_map[id(tensor_ref)] = self._name
        tm.traced_tensors.add(id(tensor_ref))
        return tensor_ref

    def test_first_access_logs_a_warning_naming_the_tensor(self, caplog) -> None:
        tm = self._manager()
        tensor_ref = self._stub_tensor(tm)
        getter = _make_tensor_getter(tensor_ref, tm, missing_fields=[])

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_manager"):
            getter(MagicMock())

        assert "Cross-layer tensor access" in caplog.text
        assert "lm_head.weight" in caplog.text, "the warning must name the tensor, not just its id"
        assert id(tensor_ref) in tm.observed_cross_refs

    def test_repeat_access_does_not_re_log(self, caplog) -> None:
        """Dedup via ``observed_cross_refs`` — one log per tensor, not per read."""
        tm = self._manager()
        tensor_ref = self._stub_tensor(tm)
        getter = _make_tensor_getter(tensor_ref, tm, missing_fields=[])

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_manager"):
            getter(MagicMock())
            caplog.clear()
            getter(MagicMock())

        assert "Cross-layer tensor access" not in caplog.text

    def test_oom_during_promotion_is_reraised_with_context(self) -> None:
        """A bare OOM from inside a property getter is maximally opaque.

        The promotion is the one H2D copy in this module that was previously
        unwrapped; its siblings all re-raise with size and remediation.
        """
        tm = self._manager()
        tensor_ref = self._stub_tensor(tm)
        tensor_ref.data.to.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
        getter = _make_tensor_getter(tensor_ref, tm, missing_fields=[])

        with pytest.raises(torch.cuda.OutOfMemoryError) as excinfo:
            getter(MagicMock())

        message = str(excinfo.value)
        assert "lm_head.weight" in message, "must name the tensor that could not be promoted"
        assert "4.00 MiB" in message, "must state the size so the operator can act on it"
        assert "exclude_patterns" in message, "must offer remediation"
        assert isinstance(excinfo.value.__cause__, torch.cuda.OutOfMemoryError), "must chain the original"
