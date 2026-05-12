# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate that FlexTensor offloading works correctly with Torch-TensorRT.

**Supported order**: offload first, then ``torch_tensorrt.compile``.
FlexTensor patches module forwards, runs warmup/profile to reach INFERENCE,
then TRT compilation is applied.

**Unsupported order**: ``torch_tensorrt.compile`` first, then offload.
``torch_tensorrt`` uses ``torch.compile`` internally, wrapping the model in
``OptimizedModule`` whose recursive structure causes ``RecursionError`` in
FlexTensor's module preprocessing (same limitation as plain ``torch.compile``).

**Numerical note**: TensorRT compilation has a tolerance for numerical
differences due to FP16/FP32 kernel selection, so we use relaxed tolerances
(``rtol``/``atol``) rather than exact checksum matching.
"""

import uuid
from typing import ClassVar

import pytest
import torch
from torch import nn

try:
    import torch_tensorrt  # noqa: F401

    HAS_TORCH_TRT = True
except ImportError:
    HAS_TORCH_TRT = False

from flextensor import OffloadConfig, get_offload_manager
from flextensor.offload_manager import OffloadPhase
from tests.integration._compile_helpers import (
    make_offload_config,
    make_simple_model,
    run_offload_lifecycle,
)

pytestmark = [
    pytest.mark.gpu_vram_24g,
    pytest.mark.skipif(not HAS_TORCH_TRT, reason="torch_tensorrt not installed"),
]


# ---------------------------------------------------------------------------
# Suite constants
# ---------------------------------------------------------------------------

MODULE_PATTERNS = ["input_projection", "layers.*", "output_projection"]
WARMUP_ITERS = 1
PROFILE_ITERS = 3
FEEDBACK_ITERS = 2
SEED = 42
NUM_LAYERS = 3
DIM = 256
INTER_DIM = 512
NUM_EXPERTS = 2
BATCH = 1
SEQ_LEN = 64

RTOL = 1e-2
ATOL = 1e-2


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
    dtype: torch.dtype = torch.float16,
) -> tuple[nn.Module, torch.Tensor]:
    model = make_simple_model(
        num_layers=NUM_LAYERS,
        dim=DIM,
        inter_dim=INTER_DIM,
        num_experts=NUM_EXPERTS,
        dtype=dtype,
        device=torch.device("cpu") if on_cpu else device,
        seed=SEED,
    )
    x = torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=dtype)
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


def _trt_compile_model(model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
    """Compile a model with Torch-TensorRT using dynamo backend.

    Uses torch.compile with the torch_tensorrt backend, which is the
    recommended approach for PyTorch 2.x integration.
    """
    return torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.float16},
            "truncate_long_and_double": True,
            "min_block_size": 1,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


class TestOffloadThenTRT:
    """FlexTensor offload() first, then TRT-compile sub-modules."""

    def test_offload_then_trt_numerical_correctness(self, device: torch.device) -> None:
        """TRT-compiled+offloaded model must produce results close to non-TRT offloaded model."""
        # Reference: offload only (no TRT)
        ref_name = f"test_otrt_ref_{uuid.uuid4().hex[:8]}"
        om_ref = get_offload_manager(ref_name)
        config = _make_offload_config()

        model_ref, x = _create_model_and_input(device, on_cpu=True)
        proxy_ref = om_ref.offload(model_ref, config)

        try:
            _, _, res_ref = _run_offload_lifecycle(proxy_ref, x)
        finally:
            om_ref.release()

        # Test: offload, reach INFERENCE, then TRT-compile the model
        test_name = f"test_otrt_{uuid.uuid4().hex[:8]}"
        om_test = get_offload_manager(test_name)

        model_test, x = _create_model_and_input(device, on_cpu=True)
        proxy_test = om_test.offload(model_test, config)

        try:
            _run_offload_lifecycle(proxy_test, x)
            assert om_test._current_phase == OffloadPhase.INFERENCE

            compiled = _trt_compile_model(model_test, x)

            with torch.no_grad():
                res_trt = x
                for _ in range(FEEDBACK_ITERS):
                    res_trt = compiled(res_trt)

            torch.testing.assert_close(
                res_ref.float(),
                res_trt.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="Offload+TRT output diverges from offload-only reference",
            )
        finally:
            om_test.release()

    def test_offload_then_trt_state_machine(self, device: torch.device) -> None:
        """OffloadManager must remain in INFERENCE after TRT compilation."""
        manager_name = f"test_otrt_sm_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            _trt_compile_model(model, x)
            assert om._current_phase == OffloadPhase.INFERENCE
        finally:
            om.release()

    def test_offload_then_trt_multiple_inferences(self, device: torch.device) -> None:
        """Multiple TRT inference passes must produce consistent results."""
        manager_name = f"test_otrt_multi_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)
            compiled = _trt_compile_model(model, x)

            results = []
            with torch.no_grad():
                for _ in range(3):
                    res = x
                    for _ in range(FEEDBACK_ITERS):
                        res = compiled(res)
                    results.append(res.clone())

            for i in range(1, len(results)):
                torch.testing.assert_close(
                    results[0],
                    results[i],
                    rtol=0,
                    atol=0,
                    msg=f"Inconsistent TRT results between run 0 and run {i}",
                )
        finally:
            om.release()


class TestTRTThenOffload:
    """TRT-compile first, then FlexTensor offload().

    This order is NOT supported: ``torch_tensorrt.compile`` uses ``torch.compile``
    internally, wrapping the model in ``OptimizedModule`` whose custom
    ``__setattr__`` proxy corrupts the underlying model's ``_modules`` dict
    during FlexTensor's preprocessing.  Same limitation as plain
    ``torch.compile`` -- always offload first, then compile.
    """

    @pytest.mark.xfail(
        strict=True,
        raises=AttributeError,
        reason=(
            "OptimizedModule.__setattr__ proxies attribute writes to _orig_mod, "
            "corrupting the model's _modules dict during preprocessing. "
            "Requires unwrapping _orig_mod in offload() entry point."
        ),
    )
    def test_trt_then_offload_raises(self, device: torch.device) -> None:
        """Offloading a TRT-compiled model raises AttributeError."""
        manager_name = f"test_trt_off_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        compiled = _trt_compile_model(model, x)
        try:
            proxy = om.offload(compiled, config)
            _run_offload_lifecycle(proxy, x)
        finally:
            om.release()


class TestBaselineConsistency:
    """Compare offload+TRT results against a pure GPU baseline (no offloading, no TRT)."""

    def test_offload_trt_matches_gpu_baseline(self, device: torch.device) -> None:
        """Offloaded+TRT output must be close to a pure GPU run (within TRT tolerance)."""
        # GPU baseline (no offload, no TRT)
        model_gpu, x = _create_model_and_input(device, on_cpu=False)
        with torch.no_grad():
            res_baseline = x.clone()
            for _ in range(FEEDBACK_ITERS):
                res_baseline = model_gpu(res_baseline)
        del model_gpu

        # Offload + TRT
        manager_name = f"test_bc_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()
        model_offload, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model_offload, config)

        try:
            _run_offload_lifecycle(proxy, x)
            compiled = _trt_compile_model(model_offload, x)

            with torch.no_grad():
                res_offload_trt = x.clone()
                for _ in range(FEEDBACK_ITERS):
                    res_offload_trt = compiled(res_offload_trt)

            torch.testing.assert_close(
                res_baseline.float(),
                res_offload_trt.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="GPU baseline vs offload+TRT mismatch (beyond TRT tolerance)",
            )
        finally:
            om.release()


class TestRegionalTRTCompilation:
    """TRT-compile sub-modules within offloaded layers rather than the whole model.

    FlexTensor patches each offloaded layer's ``forward`` with trap context
    managers.  Compiling the *patched* forward directly with TRT causes
    segfaults because dynamo traces through CUDA stream operations in the trap.

    The safe approach is to compile **sub-modules** (``Expert``)
    within each offloaded layer — these are below the offload boundary and
    contain pure compute::

        for expert in layer.experts:
            expert.forward = torch.compile(
                expert.forward, backend="torch_tensorrt", options={...}
            )

    The trap logic stays in eager mode; only the inner compute is TRT-compiled.
    """

    TRT_OPTIONS: ClassVar[dict] = {
        "enabled_precisions": {torch.float16},
        "truncate_long_and_double": True,
        "min_block_size": 1,
    }

    @classmethod
    def _compile_sub_modules(cls, model: nn.Module) -> None:
        """TRT-compile every ``Expert`` forward in-place."""
        for layer in model.layers:
            for expert in layer.experts:
                expert.forward = torch.compile(
                    expert.forward,
                    backend="torch_tensorrt",
                    options=cls.TRT_OPTIONS,
                )

    def test_regional_trt_numerical_correctness(self, device: torch.device) -> None:
        """Sub-module TRT-compiled output must be close to non-compiled offloaded output."""
        manager_name = f"test_rtrt_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _, _, res_eager = _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            self._compile_sub_modules(model)
            with torch.no_grad():
                res_compiled = x
                for _ in range(FEEDBACK_ITERS):
                    res_compiled = proxy(res_compiled)

            torch.testing.assert_close(
                res_eager.float(),
                res_compiled.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="Regional TRT compile vs eager offloaded output diverges",
            )
        finally:
            om.release()

    def test_regional_trt_multiple_runs_consistent(self, device: torch.device) -> None:
        """Multiple inference runs with sub-module TRT compilation must be deterministic."""
        manager_name = f"test_rtrt_det_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)
            self._compile_sub_modules(model)

            results = []
            with torch.no_grad():
                for _ in range(3):
                    res = x
                    for _ in range(FEEDBACK_ITERS):
                        res = proxy(res)
                    results.append(res.clone())

            for i in range(1, len(results)):
                torch.testing.assert_close(
                    results[0],
                    results[i],
                    rtol=0,
                    atol=0,
                    msg=f"Inconsistent regional TRT results between run 0 and run {i}",
                )
        finally:
            om.release()

    def test_regional_trt_state_unchanged(self, device: torch.device) -> None:
        """Sub-module TRT compilation must not alter the OffloadManager state."""
        manager_name = f"test_rtrt_state_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _run_offload_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            self._compile_sub_modules(model)
            assert om._current_phase == OffloadPhase.INFERENCE

            with torch.no_grad():
                res = x
                for _ in range(FEEDBACK_ITERS):
                    res = proxy(res)

            assert om._current_phase == OffloadPhase.INFERENCE
            assert res.shape == x.shape
        finally:
            om.release()

    def test_partial_regional_trt(self, device: torch.device) -> None:
        """TRT-compiling only some sub-modules must still produce valid results."""
        manager_name = f"test_rtrt_partial_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        config = _make_offload_config()

        model, x = _create_model_and_input(device, on_cpu=True)
        proxy = om.offload(model, config)

        try:
            _, _, res_eager = _run_offload_lifecycle(proxy, x)

            # TRT-compile experts in only the first layer
            for expert in model.layers[0].experts:
                expert.forward = torch.compile(
                    expert.forward,
                    backend="torch_tensorrt",
                    options=self.TRT_OPTIONS,
                )

            with torch.no_grad():
                res_partial = x
                for _ in range(FEEDBACK_ITERS):
                    res_partial = proxy(res_partial)

            torch.testing.assert_close(
                res_eager.float(),
                res_partial.float(),
                rtol=RTOL,
                atol=ATOL,
                msg="Partial regional TRT compile vs eager output diverges",
            )
        finally:
            om.release()
