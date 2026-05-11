# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: torch.compile + CUDA graphs + FlexTensor profile round-trip.

This test suite answers the following research questions:

1. Can a profile be saved and restored, then used with torch.compile and/or
   CUDA graphs for subsequent inference — without re-running warmup/profile?
   (Group A)

2. Does wrapping the offloaded proxy with ``torch.compile`` after the eager
   lifecycle produce correct inference output that matches the eager path?
   (Group B)

3. How does Dynamo graph structure (subgraph count) differ between a pure
   GPU model and a FlexTensor-offloaded model?  Can fullgraph=True work?
   (Group C)

4. What is the relative inference wall time for:
   (a) CPU eager, (b) CPU compiled, (c) CPU-compiled then moved to GPU,
   (d) GPU compiled — all without FlexTensor offloading?
   This isolates the "compile on CPU then run on GPU" question.
   (Group D)

Model
-----
SimpleModel: 20-layer MoE-style expert network, dim=256, bfloat16.
Fully static topology — no data-dependent control flow — so it is capturable
by CUDA graphs when all weights reside on GPU.

Single-process design
---------------------
All tests run in one process.  Profile save + restore are simulated by
creating a fresh model instance with identical weights (fixed seed) and
calling offload_from_profile().  Multi-process variant is left to a future
L1 test suite.
"""

import subprocess  # noqa: S404 - subprocess isolates a torch.compile call that can segfault
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from flextensor import OffloadConfig, get_offload_manager, offload_from_profile
from flextensor.offload_manager import OffloadPhase
from tests.integration._compile_helpers import (
    SimpleModel,
    capture_cuda_graph,
    make_offload_config,
    make_simple_model,
    run_offload_lifecycle,
    set_seed,
    tensor_checksum,
)

# Models in this suite are small (20-layer dim=256 bfloat16) — 24g tier is ample.
pytestmark = pytest.mark.gpu_mem_24g

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

RTOL = 1e-2
ATOL = 1e-2

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 20
DIM = 256
INTER_DIM = 512
NUM_EXPERTS = 2
SEED = 42
BATCH = 1
SEQ_LEN = 64

# All layers are candidates for offloading; strategy decides which to keep on GPU.
MODULE_PATTERNS = ["input_projection", "layers.*", "output_projection"]

# Lifecycle iteration counts (consistent with existing integration tests).
WARMUP_ITERS = 1
PROFILE_ITERS = 3
FEEDBACK_ITERS = 2


# ---------------------------------------------------------------------------
# Suite-local thin wrappers around the shared compile helpers
# ---------------------------------------------------------------------------


def _make_offload_config(feedback_iters: int = FEEDBACK_ITERS) -> OffloadConfig:
    return make_offload_config(
        warmup_iters=WARMUP_ITERS,
        profile_iters=PROFILE_ITERS,
        feedback_iters=feedback_iters,
        module_patterns=MODULE_PATTERNS,
    )


def _create_model(device: torch.device, on_cpu: bool = True) -> SimpleModel:
    return make_simple_model(
        num_layers=NUM_LAYERS,
        dim=DIM,
        inter_dim=INTER_DIM,
        num_experts=NUM_EXPERTS,
        dtype=torch.bfloat16,
        device=torch.device("cpu") if on_cpu else device,
        seed=SEED,
    )


def _make_input(device: torch.device) -> torch.Tensor:
    set_seed(SEED)
    return torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.bfloat16)


def _run_offload_lifecycle(
    proxy: nn.Module,
    x: torch.Tensor,
    feedback_iters: int = FEEDBACK_ITERS,
) -> torch.Tensor:
    """Drive warmup → profile → inference; return the inference-phase output."""
    _, _, res_inference = run_offload_lifecycle(
        proxy,
        x,
        warmup_iters=WARMUP_ITERS,
        profile_iters=PROFILE_ITERS,
        feedback_iters=feedback_iters,
    )
    return res_inference


def _capture_cuda_graph(
    model: nn.Module,
    static_input: torch.Tensor,
    feedback_iters: int = FEEDBACK_ITERS,
    warmup_runs: int = 3,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    return capture_cuda_graph(
        model,
        static_input,
        feedback_iters=feedback_iters,
        warmup_runs=warmup_runs,
    )


def _has_torch_2_10_weakref_bug() -> bool:
    """Whether the running torch is the 2.10 build that trips the
    ``graph_bytecode_inputs`` external-object weakref-eviction bug.

    Canonical reference for the workaround applied at every call site of this
    function — skip reasons should cite this docstring rather than restate it.

    Bug
    ---
    On torch 2.10, the compiled artifact for FlexTensor's loader methods
    holds an index into ``torch._dynamo.graph_bytecode_inputs.index_to_external_object_weakref``
    whose entry is evicted before the bytecode runs. The eviction surfaces
    in two ways depending on which torch path observes it:

    - ``AssertionError("Index not registered in index_to_user_object_weakref")``
      raised from ``get_external_object_by_index`` when the compiled bytecode
      runs (the assertion message is a torch-internal misnomer — the dict is
      actually named ``..._external_object_weakref``).
    - A process-level segfault inside ``reset_user_object_tracking`` when
      Dynamo subsequently invalidates frames that referenced the evicted entry.

    The segfault path can't be probed in-process (the probe itself would crash
    the worker), so we gate on the torch version that ships the affected
    Dynamo build. Fixed upstream in torch 2.11.

    Cleanup checklist when minimum torch is bumped past 2.10
    --------------------------------------------------------
    Raise the minimum torch in ``pyproject.toml`` to ``>= 2.11``; then:

    - delete this helper.
    - drop every ``skipif(_has_torch_2_10_weakref_bug(), ...)`` site
      (``TestGraphStructureInspection``, ``TestPerformanceComparison``).
    - drop the ``except AssertionError`` workaround in
      ``_count_dynamo_subgraphs`` that tolerates the assertion-path manifestation.
    - revisit ``_fullgraph_subprocess.py`` and the ``noqa: S404`` rationale on
      the ``import subprocess`` line: subprocess isolation was motivated by
      the segfault path; with the bug gone, the test could move back in-process.

    Tracked in an NV/torch-internal issue.
    """
    return torch.__version__.startswith("2.10")


def _count_dynamo_subgraphs(model: nn.Module, x: torch.Tensor) -> int:
    """Count distinct Dynamo subgraphs compiled for a single forward pass.

    Uses a counting backend: each call to the backend means Dynamo produced
    one subgraph.  A model with no graph breaks produces 1 subgraph.
    FlexTensor traps produce N+1 subgraphs (one per break boundary).

    Args:
        model: Model to trace (should be in final state, e.g. INFERENCE).
        x: Input tensor used for tracing.

    Returns:
        Number of Dynamo subgraphs produced.
    """
    graph_count: list[int] = [0]

    def counting_backend(gm: torch.fx.GraphModule, example_inputs: list) -> Callable:
        """Dynamo backend that counts subgraphs by incrementing a closure-captured
        counter on each call; returns ``gm.forward`` unmodified so the traced
        graph runs as-if compiled by ``"eager"``.
        """
        graph_count[0] += 1
        return gm.forward

    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, backend=counting_backend)
        with torch.no_grad():
            try:
                compiled(x)
            except AssertionError as exc:
                # Tolerate the assertion-path manifestation of the torch 2.10
                # weakref-eviction bug — see ``_has_torch_2_10_weakref_bug``
                # for the full description. The counting_backend has already
                # fired once per subgraph during tracing, so graph_count is
                # accurate by the time this runtime assertion surfaces; only
                # swallow on 2.10, only for that specific assertion, and only
                # if tracing actually produced subgraphs — otherwise the
                # exception indicates a real tracing failure we must not hide.
                if not (_has_torch_2_10_weakref_bug() and "Index not registered" in str(exc) and graph_count[0] > 0):
                    raise
    finally:
        torch._dynamo.reset()

    return graph_count[0]


def _measure_inference_time(
    model: nn.Module,
    x: torch.Tensor,
    n_warmup: int = 3,
    n_measure: int = 10,
) -> float:
    """Return median wall-time (seconds) for one forward pass.

    Args:
        model: Model to benchmark.
        x: Input tensor.
        n_warmup: Runs discarded before measurement.
        n_measure: Runs included in the median.

    Returns:
        Median wall-time in seconds.
    """
    is_cuda = x.device.type == "cuda"
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()

        times = []
        for _ in range(n_measure):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture()
def saved_profile(device: torch.device, tmp_path: Path) -> tuple[Path, torch.Tensor]:
    """Run offload lifecycle, save profile to tmp_path, release manager.

    Returns:
        (profile_dir, reference_output) where reference_output is the
        inference-phase output that subsequent restore tests should match.
    """
    manager_name = f"fixture_save_{uuid.uuid4().hex[:8]}"
    om = get_offload_manager(manager_name)
    config = _make_offload_config()
    model = _create_model(device, on_cpu=True)
    x = _make_input(device)
    proxy = om.offload(model, config)
    try:
        ref_out = _run_offload_lifecycle(proxy, x)
        assert om._current_phase == OffloadPhase.INFERENCE
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        om.save_profile(str(profile_dir))
    finally:
        om.release()

    return profile_dir, ref_out


# ===========================================================================
# Group A — Profile round-trip correctness
# ===========================================================================


class TestProfileRoundtripBaseline:
    """After save/restore, eager inference must match the pre-save reference output.

    This is the baseline for all Group A tests: it confirms that
    offload_from_profile() correctly restores the offload strategy and that
    output is numerically consistent with the original run after the state
    machine completes warmup and profile phases.

    Note: offload_from_profile() restores the saved strategy (which layers to
    offload and their profiled transfer costs) but still requires the warmup and
    profile state-machine iterations to set up the tensor loaders.  Tests must
    call _run_offload_lifecycle() after offload_from_profile() before asserting
    INFERENCE state.
    """

    def test_restore_eager_matches_reference(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """offload_from_profile proxy must produce output matching the pre-save reference."""
        profile_dir, ref_out = saved_profile
        x = _make_input(device)
        config = _make_offload_config()
        manager_name = f"test_restore_eager_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE, (
                f"Expected INFERENCE after lifecycle, got {om._current_phase}"
            )
            with torch.no_grad():
                out = x
                for _ in range(FEEDBACK_ITERS):
                    out = proxy(out)
            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="Restored proxy output diverges from pre-save reference",
            )
        finally:
            om.release()

    def test_restore_inference_is_deterministic(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """Multiple inference calls on a restored proxy must produce identical outputs."""
        profile_dir, _ = saved_profile
        x = _make_input(device)
        config = _make_offload_config()
        manager_name = f"test_restore_det_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            checksums = []
            with torch.no_grad():
                for _ in range(5):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = proxy(out)
                    checksums.append(tensor_checksum(out))

            assert len(set(checksums)) == 1, f"Inconsistent inference after restore: {checksums}"
        finally:
            om.release()


class TestProfileRoundtripCudaGraph:
    """Profile round-trip + manual CUDA graph capture.

    After restoring from a saved profile and completing the warmup/profile
    lifecycle, the proxy is in INFERENCE state with stable GPU addresses —
    satisfying the requirements for CUDA graph capture.
    """

    def test_restore_then_cuda_graph_matches_reference(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """CUDA graph captured on a restored proxy must match the pre-save reference."""
        profile_dir, ref_out = saved_profile
        x = _make_input(device)
        config = _make_offload_config()
        manager_name = f"test_cg_restore_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            static_input = x.clone()
            graph, static_out = _capture_cuda_graph(proxy, static_input)
            graph.replay()
            torch.cuda.synchronize()

            torch.testing.assert_close(
                static_out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="CUDA graph (post-restore) output diverges from pre-save reference",
            )
        finally:
            om.release()

    def test_restore_then_cuda_graph_multiple_replays(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """CUDA graph replays on a restored proxy must be deterministic."""
        profile_dir, _ = saved_profile
        config = _make_offload_config()
        manager_name = f"test_cg_replay_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, _make_input(device))
            assert om._current_phase == OffloadPhase.INFERENCE

            static_input = _make_input(device)
            graph, static_out = _capture_cuda_graph(proxy, static_input)

            checksums = []
            for _ in range(5):
                graph.replay()
                torch.cuda.synchronize()
                checksums.append(tensor_checksum(static_out))

            assert len(set(checksums)) == 1, f"Inconsistent replays after restore: {checksums}"
        finally:
            om.release()

    def test_restore_then_cuda_graph_state_unchanged(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """CUDA graph capture and replay must not disturb OffloadManager INFERENCE state."""
        profile_dir, _ = saved_profile
        config = _make_offload_config()
        manager_name = f"test_cg_state_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, _make_input(device))
            assert om._current_phase == OffloadPhase.INFERENCE

            static_input = _make_input(device)
            graph, _ = _capture_cuda_graph(proxy, static_input)
            assert om._current_phase == OffloadPhase.INFERENCE, "State changed during capture"

            graph.replay()
            torch.cuda.synchronize()
            assert om._current_phase == OffloadPhase.INFERENCE, "State changed after replay"
        finally:
            om.release()


class TestProfileRoundtripTorchCompile:
    """Profile round-trip + torch.compile on the restored proxy (no CUDA graph).

    The offload-then-compile order is applied after restore: the model is
    first restored to INFERENCE state via offload_from_profile(), then
    torch.compile() wraps the proxy returned by offload_from_profile.

    Compiling the proxy (not the underlying model) is what exercises the
    state-update forward hook: OptimizedModule bypasses
    OffloadModelProxy.__call__, so the @torch._dynamo.disable hook is
    needed for phase transitions to fire under compile.

    Note: combining torch.compile(inductor) with manual torch.cuda.CUDAGraph()
    capture is tested separately in TestCompileAndCudaGraphIncompatibility at
    the end of this file, because that combination corrupts CUDA allocator
    state on failure and must run last.
    """

    def test_restore_then_compile_matches_reference(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """torch.compile(proxy) on a restored model must produce output close to the reference."""
        profile_dir, ref_out = saved_profile
        x = _make_input(device)
        config = _make_offload_config()
        manager_name = f"test_compile_restore_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            torch._dynamo.reset()
            compiled = torch.compile(proxy, backend="inductor")

            with torch.no_grad():
                out = x
                for _ in range(FEEDBACK_ITERS):
                    out = compiled(out)

            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="Compiled (post-restore) output diverges from reference",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    def test_restore_then_compile_state_unchanged(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """Invoking torch.compile(proxy) after restore must not disturb OffloadManager state."""
        profile_dir, _ = saved_profile
        config = _make_offload_config()
        manager_name = f"test_compile_state_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, _make_input(device))
            assert om._current_phase == OffloadPhase.INFERENCE

            torch._dynamo.reset()
            compiled = torch.compile(proxy, backend="inductor")

            with torch.no_grad():
                compiled(_make_input(device))

            assert om._current_phase == OffloadPhase.INFERENCE, (
                "torch.compile(proxy) must not change OffloadManager state"
            )
        finally:
            om.release()
            torch._dynamo.reset()


class TestProfileRoundtripReduceOverhead:
    """Profile round-trip + torch.compile(mode='reduce-overhead').

    reduce-overhead tells Inductor to attempt automatic CUDA graph capture
    for each subgraph between Dynamo graph breaks.  FlexTensor's explicit
    graph breaks at trap boundaries fragment the graph, so Inductor will
    try to capture each fragment separately.

    Whether this succeeds depends on whether the transfer side-effects at
    trap boundaries are visible inside a captured fragment.  This test
    documents the current behavior and is marked xfail(strict=False) to
    allow both outcomes.
    """

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "Inductor's automatic CUDA graph capture (reduce-overhead) may fail "
            "when FlexTensor trap boundaries fragment the forward graph.  Each "
            "fragment must have stable tensor addresses on replay; transfer "
            "side-effects at boundaries can violate this constraint.  "
            "Manual torch.cuda.CUDAGraph() capture at the proxy level is the "
            "recommended alternative (see TestProfileRoundtripCudaGraph)."
        ),
    )
    def test_reduce_overhead_matches_reference(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """torch.compile(proxy, mode=reduce-overhead) on a restored model must match reference."""
        profile_dir, ref_out = saved_profile
        x = _make_input(device)
        config = _make_offload_config()
        manager_name = f"test_ro_restore_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, x)

            torch._dynamo.reset()
            compiled = torch.compile(proxy, mode="reduce-overhead")

            # Extra warmup passes so Inductor can attempt graph capture.
            with torch.no_grad():
                for _ in range(5):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = compiled(out)

            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="reduce-overhead compiled output diverges from reference",
            )
        finally:
            om.release()
            torch._dynamo.reset()


# ===========================================================================
# Group B — torch.compile wrapping the offloaded proxy
# ===========================================================================


class TestCompileWrappedProxy:
    """Wrap the offloaded proxy with ``torch.compile``.

    Two supported flows:

    1. **Discovery eager, profile+inference compiled** (two-process scenario)::

           proxy = ft.offload(model, cfg)
           for _ in range(discovery_iters): proxy(x)      # discovery eager
           compiled = torch.compile(proxy)
           for _ in range(profiling_iters): compiled(x)   # profile under compile
           ft.save_profile(...)

           # new process
           proxy = ft.offload_from_profile(model, ...)
           compiled = torch.compile(proxy)
           compiled(x)                                    # inference under compile

    2. **Compile after the full eager lifecycle**::

           proxy = ft.offload(model, cfg)
           for _ in range(total_iters): proxy(x)          # full lifecycle eager
           compiled = torch.compile(proxy)
           compiled(x)                                    # inference under compile

    Discovery cannot run under ``torch.compile``: ``WarmupTrap`` is a
    ``TorchFunctionMode`` whose ``__torch_function__`` body stages tensors via
    ``id(arg)`` lookups and explicit device copies — both are opaque to
    Dynamo's FakeTensor tracer, and a CPU-weight / GPU-input mix fails
    ``FakeTensorProp`` device propagation before the mode can act.

    Profile+inference are compile-compatible because direct-mode traps
    (``TrapDirect`` / ``TrapInferDirect``) are plain context managers whose
    ``__enter__`` / ``__exit__`` begin and end with ``_graph_break()``.  Dynamo
    compiles the layer's tensor ops between the breaks; the loader's tensor
    movement runs eagerly in the resume functions between subgraphs.

    State transitions fire via a ``@torch._dynamo.disable``-decorated forward
    hook installed on the wrapped model, so ``update_state`` runs reliably even
    when ``OptimizedModule`` bypasses ``OffloadModelProxy.__call__``.
    """

    def test_compile_after_lifecycle_matches_eager(self, device: torch.device) -> None:
        """torch.compile(proxy) after eager lifecycle must match eager output."""
        config = _make_offload_config()
        x = _make_input(device)

        # Reference: eager offload, no compile.
        ref_name = f"test_ctp_ref_{uuid.uuid4().hex[:8]}"
        om_ref = get_offload_manager(ref_name)
        model_ref = _create_model(device, on_cpu=True)
        proxy_ref = om_ref.offload(model_ref, config)
        try:
            ref_out = _run_offload_lifecycle(proxy_ref, x)
        finally:
            om_ref.release()

        # Test: eager lifecycle, then wrap proxy with torch.compile.
        torch._dynamo.reset()
        test_name = f"test_ctp_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(test_name)
        model = _create_model(device, on_cpu=True)
        proxy = om.offload(model, config)
        try:
            _ = _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            compiled = torch.compile(proxy, backend="inductor")
            with torch.no_grad():
                out = x
                for _ in range(FEEDBACK_ITERS):
                    out = compiled(out)

            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="torch.compile(proxy) output diverges from eager offload reference",
            )
        finally:
            om.release()
            torch._dynamo.reset()

    def test_compile_on_restored_profile_matches_reference(
        self,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Save profile after eager run; restore; wrap with torch.compile; verify.

        Covers the MR's "two-process" scenario in a single process:
          Process 1 equivalent: offload -> eager lifecycle -> save_profile
          Process 2 equivalent: offload_from_profile -> torch.compile -> infer
        """
        config = _make_offload_config()
        x = _make_input(device)

        # Phase 1: eager offload, run lifecycle, save profile.
        torch._dynamo.reset()
        name1 = f"test_cop_save_{uuid.uuid4().hex[:8]}"
        om1 = get_offload_manager(name1)
        model1 = _create_model(device, on_cpu=True)
        proxy1 = om1.offload(model1, config)
        profile_dir = tmp_path / "roundtrip_profile"
        profile_dir.mkdir()
        try:
            ref_out = _run_offload_lifecycle(proxy1, x)
            assert om1._current_phase == OffloadPhase.INFERENCE
            om1.save_profile(str(profile_dir))
        finally:
            om1.release()
            torch._dynamo.reset()

        # Phase 2: restore profile, wrap with torch.compile, run inference.
        name2 = f"test_cop_restore_{uuid.uuid4().hex[:8]}"
        model2 = _create_model(device, on_cpu=True)
        proxy2 = offload_from_profile(model2, str(profile_dir), config, name=name2)
        om2 = get_offload_manager(name2)
        try:
            _run_offload_lifecycle(proxy2, x)
            assert om2._current_phase == OffloadPhase.INFERENCE

            compiled = torch.compile(proxy2, backend="inductor")
            with torch.no_grad():
                out = x
                for _ in range(FEEDBACK_ITERS):
                    out = compiled(out)

            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="torch.compile on restored profile diverges from eager reference",
            )
        finally:
            om2.release()
            torch._dynamo.reset()

    def test_compile_preserves_inference_phase(self, device: torch.device) -> None:
        """torch.compile(proxy) calls must not knock the manager out of INFERENCE."""
        config = _make_offload_config()
        manager_name = f"test_ctp_phase_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(device, on_cpu=True)

        torch._dynamo.reset()
        proxy = om.offload(model, config)
        try:
            _run_offload_lifecycle(proxy, _make_input(device))
            assert om._current_phase == OffloadPhase.INFERENCE

            compiled = torch.compile(proxy, backend="inductor")
            with torch.no_grad():
                for _ in range(3):
                    _ = compiled(_make_input(device))
            assert om._current_phase == OffloadPhase.INFERENCE
        finally:
            om.release()
            torch._dynamo.reset()

    def test_user_two_phase_single_process_flow(
        self,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """End-to-end user-visible flow: eager discovery -> compiled profile -> save;
        then restore -> compiled infer, simulated within a single Python process
        (no fork / subprocess).

        Phase A (writer):
            proxy = ft.offload(m, cfg)
            for _ in range(discovery_iters): proxy(x)   # discovery eager
            compiled = torch.compile(proxy)
            for _ in range(profiling_iters): compiled(x)  # profile under compile
            ft.save_profile(...)

        Phase B (reader, after om.release() + torch._dynamo.reset()):
            proxy = ft.offload_from_profile(m, ...)
            compiled = torch.compile(proxy)
            compiled(x)                                    # inference under compile

        Limitations: this runs in one interpreter, so it does NOT exercise the
        true cross-process boundary (Dynamo compile cache, CUDA-allocator
        capture flag, the FT singleton registry all live in process globals).
        A real subprocess flow is left to a follow-up L1 test.

        Discovery must run eagerly: ``WarmupTrap`` is a ``TorchFunctionMode`` that
        does not compose with Dynamo's FakeTensor mode (device propagation fails
        when weights are on CPU and inputs on GPU).  Once the manager reaches
        ``PROFILING``, direct-mode traps are compile-compatible and profile
        timing reflects the compiled kernels.
        """
        config = _make_offload_config()
        x = _make_input(device)

        # Eager reference: full lifecycle, no compile.
        ref_name = f"test_uf_ref_{uuid.uuid4().hex[:8]}"
        om_ref = get_offload_manager(ref_name)
        model_ref = _create_model(device, on_cpu=True)
        proxy_ref = om_ref.offload(model_ref, config)
        try:
            ref_out = _run_offload_lifecycle(proxy_ref, x)
        finally:
            om_ref.release()

        # Process 1: discovery eager, compile, profile under compile, save.
        torch._dynamo.reset()
        name1 = f"test_uf_p1_{uuid.uuid4().hex[:8]}"
        om1 = get_offload_manager(name1)
        model1 = _create_model(device, on_cpu=True)
        proxy1 = om1.offload(model1, config)
        profile_dir = tmp_path / "user_flow_profile"
        profile_dir.mkdir()
        try:
            with torch.no_grad():
                for _ in range(WARMUP_ITERS):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = proxy1(out)
            assert om1._current_phase == OffloadPhase.PROFILING, (
                f"Expected PROFILING after discovery, got {om1._current_phase}"
            )

            compiled1 = torch.compile(proxy1, backend="inductor")
            with torch.no_grad():
                for _ in range(PROFILE_ITERS):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = compiled1(out)
            assert om1._current_phase == OffloadPhase.INFERENCE, (
                f"Expected INFERENCE after compiled profile, got {om1._current_phase}"
            )

            om1.save_profile(str(profile_dir))
        finally:
            om1.release()
            torch._dynamo.reset()

        # Process 2: restore, compile immediately, infer.
        name2 = f"test_uf_p2_{uuid.uuid4().hex[:8]}"
        model2 = _create_model(device, on_cpu=True)
        proxy2 = offload_from_profile(model2, str(profile_dir), config, name=name2)
        om2 = get_offload_manager(name2)
        try:
            compiled2 = torch.compile(proxy2, backend="inductor")
            with torch.no_grad():
                out = x
                for _ in range(FEEDBACK_ITERS):
                    out = compiled2(out)

            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="Two-process compile flow output diverges from eager reference",
            )
        finally:
            om2.release()
            torch._dynamo.reset()


# ===========================================================================
# Group C — Graph structure inspection
# ===========================================================================


@pytest.mark.skipif(
    _has_torch_2_10_weakref_bug(),
    reason=(
        "torch 2.10 reset_user_object_tracking bug — see _has_torch_2_10_weakref_bug. "
        "User-facing graph-break correctness is covered by TestCompileWrappedProxy "
        "(numerical match against eager reference)."
    ),
)
class TestGraphStructureInspection:
    """Compare Dynamo subgraph count for GPU-only vs FlexTensor-offloaded models.

    A GPU-only model with no graph breaks compiles to a single Dynamo subgraph.
    FlexTensor inserts dynamo.graph_break() at every TrapInfer enter/exit,
    fragmenting the forward into N+1 subgraphs for N offloaded modules.

    The counting backend approach (_count_dynamo_subgraphs) is version-stable:
    it counts how many times the backend is called during the first traced
    forward pass, which equals the number of distinct subgraphs Dynamo produces.
    """

    def test_gpu_only_model_produces_single_subgraph(self, device: torch.device) -> None:
        """A fully GPU-resident model with no graph breaks compiles to a small
        number of subgraphs.

        Range rather than equality: future torch versions may emit a synthetic
        preamble (input setup, autograd plumbing) as a separate subgraph.  The
        contract this test pins is "no FlexTensor-induced fragmentation",
        which is what the offloaded comparison test below verifies more
        strictly.
        """
        model = _create_model(device, on_cpu=False)
        x = _make_input(device)

        num_graphs = _count_dynamo_subgraphs(model, x)
        print(f"\n[C1] GPU-only model: {num_graphs} Dynamo subgraph(s)")

        assert 1 <= num_graphs <= 2, (
            f"Expected 1-2 subgraphs for a GPU-only model with no graph breaks, got {num_graphs}"
        )

    def test_offloaded_model_produces_multiple_subgraphs(self, device: torch.device) -> None:
        """FlexTensor TrapInfer graph breaks fragment the forward into multiple subgraphs."""
        config = _make_offload_config()
        manager_name = f"test_gs_offload_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(device, on_cpu=True)
        proxy = om.offload(model, config)
        x = _make_input(device)
        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            num_graphs = _count_dynamo_subgraphs(model, x)
            print(
                f"\n[C1] Offloaded model ({NUM_LAYERS} layers): {num_graphs} Dynamo subgraph(s)\n"
                f"     Each trap boundary (enter + exit) contributes at least one break.\n"
                f"     Expected: > 1 subgraph."
            )

            assert num_graphs > 1, (
                f"Expected multiple Dynamo subgraphs due to FlexTensor trap graph breaks, "
                f"got {num_graphs}. Check that TrapInfer.__enter__ calls dynamo.graph_break()."
            )
        finally:
            om.release()

    def test_subgraph_count_exceeds_gpu_only_baseline(self, device: torch.device) -> None:
        """Offloaded model must produce strictly more subgraphs than a GPU-only model.

        This directly quantifies the compilation cost of FlexTensor's graph
        fragmentation: more subgraphs means more independent compilation units,
        less cross-layer fusion, and more Dynamo overhead per compilation pass.
        """
        config = _make_offload_config()
        manager_name = f"test_gs_cmp_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model_offload = _create_model(device, on_cpu=True)
        proxy = om.offload(model_offload, config)
        x = _make_input(device)
        try:
            _run_offload_lifecycle(proxy, x)

            gpu_graphs = _count_dynamo_subgraphs(_create_model(device, on_cpu=False), x)
            offload_graphs = _count_dynamo_subgraphs(model_offload, x)

            print(
                f"\n[C1] Subgraph comparison:\n"
                f"     GPU-only:  {gpu_graphs}\n"
                f"     Offloaded: {offload_graphs}\n"
                f"     Ratio:     {offload_graphs / max(gpu_graphs, 1):.1f}x"
            )

            assert offload_graphs > gpu_graphs, (
                f"Offloaded model ({offload_graphs} graphs) must fragment more than "
                f"GPU-only baseline ({gpu_graphs} graphs)"
            )
        finally:
            om.release()


class TestFullgraphMode:
    """torch.compile(fullgraph=True) must fail on a FlexTensor-offloaded model.

    FlexTensor explicitly calls dynamo.graph_break() at trap boundaries.
    fullgraph=True instructs Dynamo to raise an error on any graph break,
    so this combination is fundamentally incompatible.

    The compile attempt runs in a subprocess so a fatal signal cannot crash
    the pytest worker — see ``_has_torch_2_10_weakref_bug`` for the
    underlying issue. Any non-zero subprocess exit — exception or fatal
    signal — satisfies the contract that fullgraph=True must not silently
    succeed.
    """

    def test_fullgraph_fails_with_offloaded_model(self, device: torch.device) -> None:
        """fullgraph=True must raise on a model with FlexTensor trap graph breaks."""
        helper = Path(__file__).parent / "_fullgraph_subprocess.py"
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(helper)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert "REACHED_COMPILE" in result.stderr, (
            "Subprocess stderr lacks the REACHED_COMPILE sentinel — either it "
            "failed before reaching torch.compile, or it crashed before flushing "
            f"stderr. This run cannot verify fullgraph behavior. exit={result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        # The decisive assertion: a non-zero exit alone is not sufficient because
        # a post-compile crash (e.g. inside ``om.release()`` in ``finally``) would
        # mask a silent fullgraph success that printed UNEXPECTED_SUCCESS just
        # before crashing.
        assert "UNEXPECTED_SUCCESS" not in result.stderr, (
            "torch.compile(fullgraph=True) silently succeeded on an offloaded model — "
            "FlexTensor's dynamo.graph_break() calls must force Dynamo to raise.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
        assert result.returncode != 0, (
            "Subprocess exited 0 without printing UNEXPECTED_SUCCESS — likely a "
            "harness bug: neither the compile-failed nor the silent-success path "
            f"was taken. exit={result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


# ===========================================================================
# Group D — Four-way performance comparison (informational)
# ===========================================================================


@pytest.mark.skipif(
    _has_torch_2_10_weakref_bug(),
    reason=(
        "torch 2.10 reset_user_object_tracking bug — see _has_torch_2_10_weakref_bug. "
        "Informational performance test with no FlexTensor-specific assertions, so "
        "no FlexTensor coverage is lost on 2.10."
    ),
)
class TestPerformanceComparison:
    """Wall-time comparison for four compile/device scenarios (no timing assertions).

    Scenarios
    ---------
    (a) CPU eager        — model on CPU, no torch.compile
    (b) CPU compiled     — model on CPU, torch.compile(backend='inductor')
    (c) CPU-compiled     — compile on CPU, then model.to('cuda'):
        → first GPU call triggers Dynamo guard failure + full GPU recompile
        → steady-state GPU throughput after recompile
    (d) GPU compiled     — model on GPU from the start, torch.compile(backend='inductor')

    Research questions answered
    ---------------------------
    - Is CPU compilation work reused when the model moves to GPU?  (No: guard fails.)
    - After recompile, does (c) steady-state match (d)?  (Should, within noise.)
    - How much extra latency does the first-call recompile add to (c)?

    No pass/fail timing thresholds are used — results are printed for
    manual inspection and analysis.
    """

    def test_four_way_performance_comparison(self, device: torch.device) -> None:
        """Measure and print inference times for four compile/device combinations."""
        results: dict[str, float] = {}

        # (a) CPU eager
        model_a = _create_model(device, on_cpu=True)
        x_cpu = _make_input(torch.device("cpu"))
        results["(a) CPU eager"] = _measure_inference_time(model_a, x_cpu)

        # (b) CPU compiled (compile + warm up on CPU, measure on CPU)
        torch._dynamo.reset()
        model_b = _create_model(device, on_cpu=True)
        compiled_b = torch.compile(model_b, backend="inductor")
        with torch.no_grad():
            compiled_b(x_cpu)  # Trigger CPU compilation
        results["(b) CPU compiled"] = _measure_inference_time(compiled_b, x_cpu)
        torch._dynamo.reset()

        # (c) CPU-compiled then moved to GPU
        torch._dynamo.reset()
        model_c = _create_model(device, on_cpu=True)
        compiled_c = torch.compile(model_c, backend="inductor")
        with torch.no_grad():
            compiled_c(x_cpu)  # Trigger CPU compilation (guard: device=cpu)
        model_c.to(device)

        x_gpu = _make_input(device)

        # First GPU call: guard fails → full GPU recompile included in measurement.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            compiled_c(x_gpu)
        torch.cuda.synchronize()
        results["(c) CPU→GPU first call (incl. recompile)"] = time.perf_counter() - t0

        # Steady-state: guard now passes for GPU device, reusing compiled artifact.
        results["(c) CPU→GPU steady-state"] = _measure_inference_time(compiled_c, x_gpu)
        torch._dynamo.reset()

        # (d) GPU compiled from the start
        torch._dynamo.reset()
        model_d = _create_model(device, on_cpu=False)
        compiled_d = torch.compile(model_d, backend="inductor")
        with torch.no_grad():
            compiled_d(x_gpu)  # Trigger GPU compilation
        results["(d) GPU compiled"] = _measure_inference_time(compiled_d, x_gpu)
        torch._dynamo.reset()

        # Print summary table
        print("\n" + "=" * 72)
        print("  Four-way compile/device performance comparison")
        print(f"  Model: SimpleModel({NUM_LAYERS} layers, dim={DIM}), bfloat16")
        print("=" * 72)
        for label, t_sec in results.items():
            print(f"  {label:<45s}  {t_sec * 1000:8.2f} ms")
        print("=" * 72)

        # Informational observations (not pass/fail thresholds).
        steady_c = results["(c) CPU→GPU steady-state"]
        first_c = results["(c) CPU→GPU first call (incl. recompile)"]
        gpu_d = results["(d) GPU compiled"]
        recompile_overhead_ms = (first_c - steady_c) * 1000

        print(f"\n  Recompile overhead in (c): {recompile_overhead_ms:.1f} ms")
        print(f"  (c) steady-state vs (d):   {abs(steady_c - gpu_d) * 1000:.1f} ms difference")

        if first_c > steady_c * 2:
            print("  [OBSERVED] First GPU call in (c) includes significant recompile latency.")
        if abs(steady_c - gpu_d) / max(gpu_d, 1e-9) < 0.20:
            print("  [OBSERVED] (c) steady-state matches (d) within 20% — GPU recompile is effective.")
        else:
            print("  [OBSERVED] (c) steady-state differs from (d) by >20% — investigate.")

        # Sanity assertions — guard against silently-broken measurements.
        assert all(t > 0 for t in results.values()), f"All timings must be positive, got {results}"
        # Cross-device guard recompile must be slower than steady state on the same
        # device — first_c includes a full Inductor recompile, steady_c does not.
        assert first_c > steady_c, (
            f"First GPU call ({first_c * 1000:.2f} ms) should include recompile overhead and "
            f"exceed steady-state ({steady_c * 1000:.2f} ms); negative overhead indicates the "
            f"measurement is broken or Dynamo skipped recompilation."
        )


# ===========================================================================
# Isolation class — MUST remain last in the file
# ===========================================================================


class TestCompileAndCudaGraphIncompatibility:
    """torch.compile(inductor) + manual CUDA graph capture is incompatible with FlexTensor.

    This class MUST be the last test class in the file.

    Root cause
    ----------
    When torch.compile(backend="inductor") is applied to a FlexTensor-patched
    model, Dynamo traces through the trap enter/exit boundaries.  The graph
    breaks inserted by FlexTensor prevent Dynamo from fusing across layers, but
    Dynamo still traces INTO the loader's enter() and exit() methods (because
    the graph break fires after enter() is called, not before).  Inductor
    compiles these methods into kernels that contain CPU→GPU tensor copies
    (buf0.copy_(arg1_1)).

    When a manual CUDA graph capture is subsequently started with
    torch.cuda.graph(graph, stream=s), those Inductor-compiled copy operations
    run on stream 0 (the default stream), not on the capture stream s.  CUDA
    rejects this with cudaErrorStreamCaptureUnsupported.

    When the capture context manager then tries to call capture_end() to clean
    up, that call also fails (cudaErrorStreamCaptureInvalidated), and PyTorch's
    C++ CUDACachingAllocator never receives notifyCaptureAbort().  The allocator
    is left with its internal capture-active flag set, causing all subsequent
    torch.randn(..., device='cuda') calls to raise:
      "Offset increment outside graph capture encountered unexpectedly."

    This CUDA allocator state corruption is a PyTorch bug (capture_end failure
    should call notifyCaptureAbort, not just propagate the error).  There is no
    Python-level API to reset the flag.  The test is therefore isolated as the
    very last test in the suite so that the corruption does not cascade.

    Alternatives that DO work
    -------------------------
    - torch.compile + eager inference (no CUDA graph): TestProfileRoundtripTorchCompile
    - Manual CUDA graph without torch.compile: TestProfileRoundtripCudaGraph
    - Automatic CUDA graphs via reduce-overhead: TestProfileRoundtripReduceOverhead
      (xfail — different failure mode, but no allocator corruption)
    """

    @pytest.mark.xfail(
        strict=False,
        raises=Exception,
        reason=(
            "torch.compile(backend='inductor') causes Dynamo to trace and compile "
            "FlexTensor loader enter()/exit() methods.  Inductor emits CPU→GPU copy "
            "kernels (buf0.copy_) that run on the default stream, which CUDA rejects "
            "during graph capture (cudaErrorStreamCaptureUnsupported).  The subsequent "
            "capture_end() failure corrupts the CUDA allocator's capture-active flag "
            "(PyTorch bug: notifyCaptureAbort is not called on capture_end failure). "
            "Use manual CUDA graph WITHOUT torch.compile (TestProfileRoundtripCudaGraph) "
            "or torch.compile WITHOUT manual CUDA graph (TestProfileRoundtripTorchCompile)."
        ),
    )
    def test_compile_and_cuda_graph_incompatible(
        self,
        device: torch.device,
        saved_profile: tuple[Path, torch.Tensor],
    ) -> None:
        """torch.compile(inductor) + manual CUDA graph capture must fail with a clear error.

        This test documents the incompatibility and the exact failure mode.
        It is expected to raise an AcceleratorError or RuntimeError.
        """
        profile_dir, _ = saved_profile
        x = _make_input(device)
        config = _make_offload_config()
        manager_name = f"test_compile_cg_incompat_{uuid.uuid4().hex[:8]}"
        model = _create_model(device, on_cpu=True)
        proxy = offload_from_profile(model, str(profile_dir), config, name=manager_name)
        om = get_offload_manager(manager_name)
        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            torch._dynamo.reset()
            compiled_model = torch.compile(model, backend="inductor")

            # Trigger Inductor compilation of the patched forward (including loaders).
            with torch.no_grad():
                for _ in range(2):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = compiled_model(out)

            # Attempt CUDA graph capture — expected to fail because Inductor-compiled
            # loader kernels use the default stream, violating capture constraints.
            static_input = x.clone()
            graph, _static_out = _capture_cuda_graph(compiled_model, static_input)
            graph.replay()
            torch.cuda.synchronize()
        finally:
            om.release()
            torch._dynamo.reset()
