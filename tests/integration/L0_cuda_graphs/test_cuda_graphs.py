# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that FlexTensor offloading works correctly with CUDA Graphs.

CUDA Graphs capture a sequence of GPU operations into a graph that can be
replayed with minimal CPU overhead.  FlexTensor's tensor loaders support
CUDA graph capture via fork/join event synchronisation on the transfer stream
(see ``PreallocatedBatchTransferTensorLoader``).

Key constraints of CUDA graph capture:
- All tensors referenced during capture must have stable GPU addresses on replay.
- CPU-GPU transfers must happen on streams that are *forked* into the graph
  capture scope; no new CPU-GPU transfers can be initiated inside a captured
  region unless the transfer stream was forked at the start.
- No dynamic control flow (if/else on tensor values, dynamic shapes).

Tests cover:
1. **GPU-only baseline**: CUDA graph capture works on the model architecture
   itself (no FlexTensor involvement).
2. **Post-release capture**: After FlexTensor lifecycle + release, the model
   can be moved to GPU and captured as a CUDA graph.
3. **Capture after offload**: FlexTensor offload -> warmup/profile -> CUDA
   graph capture during INFERENCE -> replay and validate.
4. **Multi-replay consistency**: Replaying the captured graph many times must
   produce identical results every time.
5. **Graph-vs-eager correctness**: Output from graph replay must match eager
   (non-graph) inference under the same offloading setup.
"""

import uuid

import pytest
import torch
from torch import nn

from flextensor import OffloadConfig, get_offload_manager
from flextensor.offload_manager import OffloadPhase
from tests.integration._compile_helpers import (
    capture_cuda_graph,
    make_offload_config,
    make_simple_model,
    run_offload_lifecycle,
    tensor_checksum,
)

# Small MoE-style models; 24g tier is ample for CUDA graph capture.
pytestmark = pytest.mark.gpu_vram_24g

# ---------------------------------------------------------------------------
# Suite constants
# ---------------------------------------------------------------------------

MODULE_PATTERNS = ["input_projection", "layers.*", "output_projection"]
WARMUP_ITERS = 1
PROFILE_ITERS = 3
FEEDBACK_ITERS = 2
SEED = 42
NUM_LAYERS = 4
DIM = 512
INTER_DIM = 1024
NUM_EXPERTS = 2
BATCH = 1
SEQ_LEN = 128


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


def _create_model_and_input(
    device: torch.device,
    on_cpu: bool = True,
) -> tuple[nn.Module, torch.Tensor]:
    model = make_simple_model(
        num_layers=NUM_LAYERS,
        dim=DIM,
        inter_dim=INTER_DIM,
        num_experts=NUM_EXPERTS,
        dtype=torch.bfloat16,
        device=torch.device("cpu") if on_cpu else device,
        seed=SEED,
    )
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.bfloat16)
    return model, x


def _run_offload_lifecycle(
    proxy: nn.Module,
    x: torch.Tensor,
    feedback_iters: int = FEEDBACK_ITERS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return run_offload_lifecycle(
        proxy,
        x,
        warmup_iters=WARMUP_ITERS,
        profile_iters=PROFILE_ITERS,
        feedback_iters=feedback_iters,
    )


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


# ---------------------------------------------------------------------------
# Tests -- GPU-only baseline (no FlexTensor)
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestGPUOnlyBaseline:
    """Verify the model architecture is CUDA-graph-capturable without FlexTensor."""

    def test_gpu_model_capture_and_replay(self, device: torch.device) -> None:
        """A fully GPU-resident model can be captured and replayed as a CUDA graph."""
        model, x = _create_model_and_input(device, on_cpu=False)

        with torch.no_grad():
            res_eager = x.clone()
            for _ in range(FEEDBACK_ITERS):
                res_eager = model(res_eager)
        checksum_eager = tensor_checksum(res_eager)

        static_input = x.clone()
        graph, static_out = _capture_cuda_graph(model, static_input)
        graph.replay()
        torch.cuda.synchronize()
        checksum_graph = tensor_checksum(static_out)

        assert checksum_eager == checksum_graph, f"Eager vs graph mismatch: {checksum_eager} vs {checksum_graph}"

    def test_gpu_model_multiple_replays(self, device: torch.device) -> None:
        """Multiple replays of a GPU-only graph produce identical results."""
        model, x = _create_model_and_input(device, on_cpu=False)

        static_input = x.clone()
        graph, static_out = _capture_cuda_graph(model, static_input)

        checksums = []
        for _ in range(5):
            graph.replay()
            torch.cuda.synchronize()
            checksums.append(tensor_checksum(static_out))

        assert len(set(checksums)) == 1, f"Inconsistent replays: {checksums}"

    def test_gpu_model_different_inputs(self, device: torch.device) -> None:
        """Changing static input data before replay produces correct new output."""
        model, x = _create_model_and_input(device, on_cpu=False)

        static_input = x.clone()
        graph, static_out = _capture_cuda_graph(model, static_input)

        graph.replay()
        torch.cuda.synchronize()
        checksum_a = tensor_checksum(static_out)

        static_input.copy_(torch.randn_like(static_input))
        graph.replay()
        torch.cuda.synchronize()
        checksum_b = tensor_checksum(static_out)

        assert checksum_a != checksum_b, "Different inputs should produce different outputs"

        with torch.no_grad():
            res_eager = static_input.clone()
            for _ in range(FEEDBACK_ITERS):
                res_eager = model(res_eager)
        checksum_eager = tensor_checksum(res_eager)

        assert checksum_b == checksum_eager, f"Graph vs eager mismatch with new input: {checksum_b} vs {checksum_eager}"


# ---------------------------------------------------------------------------
# Tests -- Eager offloaded inference (sanity check, no graph capture)
# ---------------------------------------------------------------------------


class TestEagerOffloadedInference:
    """Sanity check that offloading without graph capture works correctly."""

    def test_offloaded_inference_matches_gpu_baseline(self, device: torch.device) -> None:
        """Offloaded eager inference must match pure GPU baseline."""
        # GPU baseline
        model_gpu, x = _create_model_and_input(device, on_cpu=False)
        with torch.no_grad():
            res_baseline = x.clone()
            for _ in range(FEEDBACK_ITERS):
                res_baseline = model_gpu(res_baseline)
        checksum_baseline = tensor_checksum(res_baseline)
        del model_gpu

        # Offloaded
        manager_name = f"test_eager_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()
        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _, _, res_offload = _run_offload_lifecycle(proxy, x)
            checksum_offload = tensor_checksum(res_offload)

            assert om._current_phase == OffloadPhase.INFERENCE
            assert checksum_baseline == checksum_offload, (
                f"GPU baseline vs offloaded mismatch: {checksum_baseline} vs {checksum_offload}"
            )
        finally:
            om.release()

    def test_offloaded_inference_is_deterministic(self, device: torch.device) -> None:
        """Multiple offloaded eager inferences produce identical results."""
        manager_name = f"test_eager_det_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()
        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)

            checksums = []
            with torch.no_grad():
                for _ in range(5):
                    res = x
                    for _ in range(FEEDBACK_ITERS):
                        res = proxy(res)
                    checksums.append(tensor_checksum(res))

            assert len(set(checksums)) == 1, f"Inconsistent eager inference: {checksums}"
        finally:
            om.release()


# ---------------------------------------------------------------------------
# Tests -- CUDA graph capture after FlexTensor release
# ---------------------------------------------------------------------------


class TestCaptureAfterRelease:
    """Capture CUDA graph after FlexTensor offload lifecycle is completed and released."""

    def test_capture_after_release_matches_baseline(self, device: torch.device) -> None:
        """After release + move to GPU, the model can be CUDA-graph captured."""
        # GPU baseline
        model_gpu, x = _create_model_and_input(device, on_cpu=False)
        with torch.no_grad():
            res_baseline = x.clone()
            for _ in range(FEEDBACK_ITERS):
                res_baseline = model_gpu(res_baseline)
        checksum_baseline = tensor_checksum(res_baseline)
        del model_gpu

        # Offload lifecycle then release
        manager_name = f"test_rel_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()
        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)
        _run_offload_lifecycle(proxy, x)
        om.release()

        # Move model to GPU (weights are restored, forward is unpatched)
        model = model.to(device)

        # Capture as CUDA graph
        static_input = x.clone()
        graph, static_out = _capture_cuda_graph(model, static_input)
        graph.replay()
        torch.cuda.synchronize()
        checksum_graph = tensor_checksum(static_out)

        assert checksum_baseline == checksum_graph, (
            f"Baseline vs post-release graph mismatch: {checksum_baseline} vs {checksum_graph}"
        )

    def test_capture_after_release_multiple_replays(self, device: torch.device) -> None:
        """Post-release CUDA graph replays are deterministic."""
        manager_name = f"test_rel_multi_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()
        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)
        _run_offload_lifecycle(proxy, x)
        om.release()

        model = model.to(device)
        static_input = x.clone()
        graph, static_out = _capture_cuda_graph(model, static_input)

        checksums = []
        for _ in range(5):
            graph.replay()
            torch.cuda.synchronize()
            checksums.append(tensor_checksum(static_out))

        assert len(set(checksums)) == 1, f"Inconsistent post-release replays: {checksums}"


# ---------------------------------------------------------------------------
# Tests -- CUDA graph capture with ACTIVE offloading
# ---------------------------------------------------------------------------


class TestCaptureWithActiveOffload:
    """Capture a CUDA graph while FlexTensor offloading is active in INFERENCE state.

    The loaders clear their cross-iteration event maps at the join point
    (last layer exit), so the next iteration starts with no stale events.
    This allows CUDA graph capture to succeed without stream-capture
    isolation violations.
    """

    def test_graph_capture_succeeds(self, device: torch.device) -> None:
        """CUDA graph capture must not raise during INFERENCE with offloaded model."""
        manager_name = f"test_cg_cap_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            graph, static_out = _capture_cuda_graph(proxy, x.clone())
            assert graph is not None
            assert static_out.shape == x.shape
        finally:
            om.release()

    def test_graph_replay_matches_eager(self, device: torch.device) -> None:
        """Graph replay output must match eager offloaded inference."""
        manager_name = f"test_cg_eager_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _, _, res_eager = _run_offload_lifecycle(proxy, x)
            checksum_eager = tensor_checksum(res_eager)

            static_input = x.clone()
            graph, static_out = _capture_cuda_graph(proxy, static_input)
            graph.replay()
            torch.cuda.synchronize()
            checksum_graph = tensor_checksum(static_out)

            assert checksum_eager == checksum_graph, f"Eager vs graph mismatch: {checksum_eager} vs {checksum_graph}"
        finally:
            om.release()

    def test_graph_state_machine_unchanged(self, device: torch.device) -> None:
        """CUDA graph capture and replay must not disturb OffloadManager state."""
        manager_name = f"test_cg_state_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            static_input = x.clone()
            graph, _ = _capture_cuda_graph(proxy, static_input)
            assert om._current_phase == OffloadPhase.INFERENCE

            graph.replay()
            torch.cuda.synchronize()
            assert om._current_phase == OffloadPhase.INFERENCE
        finally:
            om.release()


class TestMultiReplayConsistency:
    """Replaying a captured CUDA graph multiple times must produce identical results."""

    def test_replays_are_deterministic(self, device: torch.device) -> None:
        """N replays of the same graph must produce the exact same output."""
        manager_name = f"test_cg_det_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)

            static_input = x.clone()
            graph, static_out = _capture_cuda_graph(proxy, static_input)

            checksums = []
            for _ in range(5):
                graph.replay()
                torch.cuda.synchronize()
                checksums.append(tensor_checksum(static_out))

            assert len(set(checksums)) == 1, f"Inconsistent replays: {checksums}"
        finally:
            om.release()

    def test_replay_with_different_input_data(self, device: torch.device) -> None:
        """Changing the static input data before replay must produce a new (correct) output."""
        manager_name = f"test_cg_diff_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)

            static_input = x.clone()
            graph, static_out = _capture_cuda_graph(proxy, static_input)

            # Replay with original input
            graph.replay()
            torch.cuda.synchronize()
            checksum_a = tensor_checksum(static_out)

            # Overwrite static_input in-place and replay
            static_input.copy_(torch.randn_like(static_input))
            graph.replay()
            torch.cuda.synchronize()
            checksum_b = tensor_checksum(static_out)

            assert checksum_a != checksum_b, "Different inputs should produce different outputs"

            # Verify the new result matches eager execution with the same input
            with torch.no_grad():
                res_eager = static_input.clone()
                for _ in range(FEEDBACK_ITERS):
                    res_eager = proxy(res_eager)
            checksum_eager = tensor_checksum(res_eager)

            assert checksum_b == checksum_eager, (
                f"Graph replay vs eager mismatch with new input: {checksum_b} vs {checksum_eager}"
            )
        finally:
            om.release()


class TestBaselineConsistency:
    """Compare graph-captured offloaded results against a pure GPU baseline."""

    def test_graph_offload_matches_gpu_baseline(self, device: torch.device) -> None:
        """CUDA-graph-captured offloaded model must match pure GPU (no offload) baseline."""
        # GPU baseline (no offload, no graph)
        model_gpu, x = _create_model_and_input(device, on_cpu=False)
        with torch.no_grad():
            res_baseline = x.clone()
            for _ in range(FEEDBACK_ITERS):
                res_baseline = model_gpu(res_baseline)
        checksum_baseline = tensor_checksum(res_baseline)
        del model_gpu

        # Offload + CUDA graph
        manager_name = f"test_cg_bl_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()
        model_offload, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model_offload, config)

        try:
            _run_offload_lifecycle(proxy, x)

            static_input = x.clone()
            graph, static_out = _capture_cuda_graph(proxy, static_input)
            graph.replay()
            torch.cuda.synchronize()
            checksum_graph = tensor_checksum(static_out)

            assert checksum_baseline == checksum_graph, (
                f"GPU baseline vs offload+graph mismatch: {checksum_baseline} vs {checksum_graph}"
            )
        finally:
            om.release()
