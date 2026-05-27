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
  ``None``. The trap-path tests below stay as *contract tests* on the
  fall-through, documenting the precondition the loader is now
  responsible for upholding.

* **Block loaders (``allocation_block_transfer``, ``raw_block_transfer``).**
  ``prepare_view_model`` runs
  :class:`flextensor.tensor_processors.MoveUnmappedTensorsToGPUProcessor`,
  whose contract is "if not in ``tensor_id_to_view_map``, move to GPU
  permanently." So the same dropped tensor lands on GPU before
  inference begins. That permanent-GPU path is budgeted before strategy
  computation and guarded again immediately before each CUDA move.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.collectors import (
    IterativeLayerStatistics,
    LayerStatistics,
    TensorStatistics,
)
from flextensor.loaders import (
    TensorStrategyLoader,
    _compute_untimed_traced_preload,
)
from flextensor.tensor_manager import (
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


def _patched_cuda():
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


def _stop_patches(patchers) -> None:
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

    def test_rescue_emits_warning_with_count_bytes_and_ids(self, caplog) -> None:
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

    def test_property_getter_falls_through_when_loader_returns_none(self) -> None:
        """Contract: direct-mode getter falls through to ``tensor_ref`` on ``None``.

        The strategy loader's rescue keeps this path from being reached for
        traced tensors in production, but the precondition has to hold:
        if ``loader.get(...)`` ever returned ``None`` for a traced tensor,
        the patched module property would surface the original (CPU)
        ``tensor_ref``. This test pins the contract the loader is
        responsible for upholding.
        """
        untimed_cpu = torch.zeros(8, dtype=torch.float32)
        assert untimed_cpu.device.type == "cpu", "fixture sanity"

        tm = MagicMock()
        tm.is_traced_by_id.return_value = True
        tm.tensor_layer_loader = MagicMock()
        tm.tensor_layer_loader.get.return_value = None

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

        def echo(*args, **kwargs):
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

        def echo(*args, **kwargs):
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
