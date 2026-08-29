# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for NVMe disk offload.

Tests the full NVMe weight-eviction → GPU-read-back pipeline through the
FlexTensor ``OffloadManager`` lifecycle, plus direct backend roundtrips.

External dependencies:
- NVIDIA GPU (CUDA required — the session-level ``conftest.py`` enforces this).
- A writable local filesystem path for NVMe block files. By default pytest's
  ``tmp_path`` is used; set ``FT_NVME_TEST_PATH`` to a directory on a real
  NVMe filesystem when running cuFile (GDS) tests, which require
  block-device-backed files.
- ``nvidia-fs`` kernel module loaded for cuFile tests (optional). On GB10
  unified memory, cuFile's pre-flight check fails and the backend
  auto-falls back to ``PosixBackend``; cuFile-specific tests are skipped.

Test groups:
1. Backend GPU roundtrips — PosixBackend and CuFileBackend directly on GPU memory.
2. NVMe offload lifecycle — full OffloadManager lifecycle with NVMe eviction,
   parametrized across both block controller types.
3. CUDA graph capture — graph capture and replay with NVMe-backed offload.
4. Profile roundtrip — save/restore profile with NVMe backing.
"""

import uuid
from pathlib import Path

import pytest
import torch
from torch import nn

from flextensor import get_offload_manager, offload_from_profile
from flextensor.nvme_transfer import PosixBackend, make_nvme_backend
from flextensor.offload_manager import OffloadPhase
from tests.integration._compile_helpers import (
    make_simple_model,
    run_offload_lifecycle,
    tensor_checksum,
)
from tests.integration.L0_nvme_offload._nvme_helpers import (
    assert_nvme_files_exist,
    cufile_available,
    make_non_nvme_config,
    make_nvme_offload_config,
    posix_backend_gpu_roundtrip,
)

# Small MoE-style models; 24g tier is ample.
pytestmark = pytest.mark.gpu_vram_24g

# ---------------------------------------------------------------------------
# Suite constants
# ---------------------------------------------------------------------------

MODULE_PATTERNS = ["input_projection", "layers.*", "output_projection"]
DISCOVERY_ITERS = 1
PROFILING_ITERS = 3
FEEDBACK_ITERS = 2
SEED = 42
NUM_LAYERS = 4
DIM = 512
INTER_DIM = 1024
NUM_EXPERTS = 2
BATCH = 1
SEQ_LEN = 128

BLOCK_TRANSFER_MODES = ["raw_block_transfer", "allocation_block_transfer"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def device() -> torch.device:
    """GPU device fixture; skips when CUDA is unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


def _create_model(on_cpu: bool = True, device: torch.device | None = None) -> nn.Module:
    """Create a seeded ``SimpleModel`` on CPU or GPU."""
    return make_simple_model(
        num_layers=NUM_LAYERS,
        dim=DIM,
        inter_dim=INTER_DIM,
        num_experts=NUM_EXPERTS,
        dtype=torch.bfloat16,
        device=torch.device("cpu") if on_cpu else (device or torch.device("cuda")),
        seed=SEED,
    )


def _make_input(device: torch.device) -> torch.Tensor:
    """Create a deterministic input tensor for the model."""
    return torch.randn(BATCH, SEQ_LEN, DIM, device=device, dtype=torch.bfloat16)


def _run_lifecycle(proxy: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Drive warmup → profile → inference; return the inference-phase output."""
    _, _, res_inference = run_offload_lifecycle(
        proxy,
        x,
        discovery_iters=DISCOVERY_ITERS,
        profiling_iters=PROFILING_ITERS,
        feedback_iters=FEEDBACK_ITERS,
    )
    return res_inference


# ===========================================================================
# Group 1 — Backend GPU roundtrips
# ===========================================================================


class TestNvmeBackendGpuRoundtrip:
    """Direct backend roundtrips on GPU memory (no FlexTensor lifecycle).

    These tests exercise the lowest layer of the NVMe stack — writing
    weight blocks to disk and reading them back into GPU memory — without
    the block controller or OffloadManager machinery.
    """

    def test_posix_backend_roundtrip_gpu(self, device: torch.device, tmp_path: Path) -> None:
        """PosixBackend write→read roundtrip into GPU memory must preserve data."""
        original, read_gpu = posix_backend_gpu_roundtrip(tmp_path, (256, 256), torch.float32)
        assert torch.equal(read_gpu.cpu(), original)

    def test_posix_backend_multiple_blocks_gpu(self, device: torch.device, tmp_path: Path) -> None:
        """Multiple blocks at aligned offsets in one file must roundtrip correctly to GPU."""
        backend = PosixBackend(alignment=4096)
        file_path = str(tmp_path / "multi_gpu.bin")
        fd = backend.open_file(file_path)

        data1 = torch.randn(500, dtype=torch.float32)
        data2 = torch.arange(300, dtype=torch.int64) * 2
        data3 = torch.randn(200, dtype=torch.float16)
        bytes1 = data1.contiguous().view(torch.uint8).reshape(-1)
        bytes2 = data2.contiguous().view(torch.uint8).reshape(-1)
        bytes3 = data3.contiguous().view(torch.uint8).reshape(-1)

        ref1 = backend.write_block(fd, bytes1, offset=0)
        ref2 = backend.write_block(fd, bytes2, offset=ref1.aligned_nbytes)
        ref3 = backend.write_block(fd, bytes3, offset=ref1.aligned_nbytes + ref2.aligned_nbytes)

        buf1 = torch.zeros(ref1.aligned_nbytes, dtype=torch.uint8, device=device)
        buf2 = torch.zeros(ref2.aligned_nbytes, dtype=torch.uint8, device=device)
        buf3 = torch.zeros(ref3.aligned_nbytes, dtype=torch.uint8, device=device)
        backend.read_block(fd, buf1, ref1.file_offset, ref1.logical_nbytes)
        backend.read_block(fd, buf2, ref2.file_offset, ref2.logical_nbytes)
        backend.read_block(fd, buf3, ref3.file_offset, ref3.logical_nbytes)

        assert torch.equal(
            buf1[: ref1.logical_nbytes].view(torch.float32).reshape(500),
            data1.to(device),
        )
        assert torch.equal(
            buf2[: ref2.logical_nbytes].view(torch.int64).reshape(300),
            data2.to(device),
        )
        assert torch.equal(
            buf3[: ref3.logical_nbytes].view(torch.float16).reshape(200),
            data3.to(device),
        )

        backend.close_file(fd)
        backend.close()

    def test_posix_backend_alignment_padding_gpu(self, device: torch.device, tmp_path: Path) -> None:
        """Sub-alignment data must be padded on disk and read back correctly to GPU."""
        backend = PosixBackend(alignment=4096)
        file_path = str(tmp_path / "aligned_gpu.bin")
        fd = backend.open_file(file_path)

        data = torch.arange(25, dtype=torch.float32)  # 100 bytes
        data_bytes = data.contiguous().view(torch.uint8).reshape(-1)
        ref = backend.write_block(fd, data_bytes, offset=0)

        assert ref.logical_nbytes == 100
        assert ref.aligned_nbytes == 4096
        assert Path(file_path).stat().st_size >= 4096

        gpu_buf = torch.zeros(ref.aligned_nbytes, dtype=torch.uint8, device=device)
        backend.read_block(fd, gpu_buf, ref.file_offset, ref.logical_nbytes)
        result = gpu_buf[: ref.logical_nbytes].view(torch.float32).reshape(25)
        assert torch.equal(result, data.to(device))

        backend.close_file(fd)
        backend.close()

    @pytest.mark.skipif(not cufile_available(), reason="cuFile (GDS) not available on this GPU")
    def test_cufile_backend_roundtrip_gpu(self, device: torch.device, tmp_path: Path) -> None:
        """CuFileBackend write→read roundtrip into GPU memory must preserve data.

        This test runs only on GDS-capable GPUs where ``nvidia-fs`` is loaded
        and the GPU model supports GDS P2P DMA. On GB10 unified memory, the
        pre-flight check fails and this test is skipped.
        """
        from flextensor.nvme_transfer import CuFileBackend

        backend = CuFileBackend(alignment=4096)
        file_path = str(tmp_path / "cufile_gpu.bin")
        fd = backend.open_file(file_path)

        data = torch.randn(256, dtype=torch.float32)
        data_bytes = data.contiguous().view(torch.uint8).reshape(-1)
        block_ref = backend.write_block(fd, data_bytes, offset=0)

        gpu_buf = torch.zeros(block_ref.aligned_nbytes, dtype=torch.uint8, device=device)
        backend.read_block(fd, gpu_buf, block_ref.file_offset, block_ref.logical_nbytes)
        read_gpu = gpu_buf[: block_ref.logical_nbytes].view(torch.float32).reshape(256)

        assert torch.equal(read_gpu, data)

        backend.close_file(fd)
        backend.close()

    def test_make_nvme_backend_posix(self) -> None:
        """make_nvme_backend('posix') must return a PosixBackend."""
        backend = make_nvme_backend("posix")
        assert isinstance(backend, PosixBackend)
        backend.close()

    def test_make_nvme_backend_cufile_fallback(self) -> None:
        """make_nvme_backend('cufile') falls back to PosixBackend when GDS is unavailable.

        On GB10 (and any GPU where the pre-flight check fails), the factory
        returns a PosixBackend. On GDS-capable GPUs, it returns a CuFileBackend.
        Either outcome is valid — the contract is that the factory never raises.
        """
        from flextensor.nvme_transfer import CuFileBackend

        backend = make_nvme_backend("cufile")
        if not cufile_available():
            assert isinstance(backend, PosixBackend)
        else:
            assert isinstance(backend, CuFileBackend)
        backend.close()


# ===========================================================================
# Group 2 — NVMe offload lifecycle (parametrized across block controllers)
# ===========================================================================


class TestNvmeOffloadLifecycle:
    """Full OffloadManager lifecycle with NVMe weight eviction.

    Parametrized across both block controller types (raw_block_transfer and
    allocation_block_transfer) to exercise both eviction paths. Verifies
    inference output equivalence against the non-NVMe baseline, determinism,
    and that NVMe block files are written to disk.
    """

    @pytest.mark.parametrize("transfer_mode", BLOCK_TRANSFER_MODES)
    def test_nvme_offload_matches_non_nvme_baseline(
        self,
        device: torch.device,
        tmp_path: Path,
        transfer_mode: str,
    ) -> None:
        """NVMe-backed inference output must match the non-NVMe baseline.

        Runs the same model with identical seed and config (except NVMe fields)
        through the full OffloadManager lifecycle. The offload strategy should
        produce the same GPU-resident weights regardless of whether the source
        is CPU-block-backed or NVMe-backed.
        """
        # Baseline: no NVMe offload.
        baseline_config = make_non_nvme_config(
            transfer_mode=transfer_mode,
            module_patterns=MODULE_PATTERNS,
        )
        baseline_name = f"test_nvme_baseline_{uuid.uuid4().hex[:8]}"
        om_baseline = get_offload_manager(baseline_name)
        model_baseline = _create_model(on_cpu=True)
        proxy_baseline = om_baseline.offload(model_baseline, baseline_config)
        x = _make_input(device)
        try:
            ref_out = _run_lifecycle(proxy_baseline, x)
            ref_checksum = tensor_checksum(ref_out)
        finally:
            om_baseline.release()

        # NVMe offload.
        nvme_config, _nvme_dir = make_nvme_offload_config(
            tmp_path,
            transfer_mode=transfer_mode,
            module_patterns=MODULE_PATTERNS,
        )
        nvme_name = f"test_nvme_offload_{uuid.uuid4().hex[:8]}"
        om_nvme = get_offload_manager(nvme_name)
        model_nvme = _create_model(on_cpu=True)
        proxy_nvme = om_nvme.offload(model_nvme, nvme_config)
        try:
            nvme_out = _run_lifecycle(proxy_nvme, x)
            nvme_checksum = tensor_checksum(nvme_out)
        finally:
            om_nvme.release()

        assert ref_checksum == nvme_checksum, (
            f"Baseline ({ref_checksum}) vs NVMe offload ({nvme_checksum}) output mismatch "
            f"with transfer_mode={transfer_mode}"
        )

    @pytest.mark.parametrize("transfer_mode", BLOCK_TRANSFER_MODES)
    def test_nvme_offload_is_deterministic(
        self,
        device: torch.device,
        tmp_path: Path,
        transfer_mode: str,
    ) -> None:
        """Multiple NVMe-backed inference iterations must produce identical outputs."""
        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode=transfer_mode,
            module_patterns=MODULE_PATTERNS,
        )
        manager_name = f"test_nvme_det_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(on_cpu=True)
        proxy = om.offload(model, nvme_config)
        x = _make_input(device)
        try:
            _run_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            checksums = []
            with torch.no_grad():
                for _ in range(5):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = proxy(out)
                    checksums.append(tensor_checksum(out))

            assert len(set(checksums)) == 1, f"Inconsistent NVMe inference: {checksums}"
        finally:
            om.release()

    @pytest.mark.parametrize("transfer_mode", BLOCK_TRANSFER_MODES)
    def test_nvme_blocks_written_to_disk(
        self,
        device: torch.device,
        tmp_path: Path,
        transfer_mode: str,
    ) -> None:
        """After lifecycle, blocks.bin must exist and be non-empty in the NVMe path."""
        nvme_config, nvme_dir = make_nvme_offload_config(
            tmp_path,
            transfer_mode=transfer_mode,
            module_patterns=MODULE_PATTERNS,
        )
        manager_name = f"test_nvme_files_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(on_cpu=True)
        proxy = om.offload(model, nvme_config)
        x = _make_input(device)
        try:
            _run_lifecycle(proxy, x)
        finally:
            om.release()

        assert_nvme_files_exist(nvme_dir)

    @pytest.mark.parametrize("transfer_mode", BLOCK_TRANSFER_MODES)
    def test_nvme_offload_state_machine(
        self,
        device: torch.device,
        tmp_path: Path,
        transfer_mode: str,
    ) -> None:
        """NVMe-backed offload must reach INFERENCE without phase-transition errors."""
        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode=transfer_mode,
            module_patterns=MODULE_PATTERNS,
        )
        manager_name = f"test_nvme_state_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(on_cpu=True)
        proxy = om.offload(model, nvme_config)
        x = _make_input(device)
        try:
            _run_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE, (
                f"Expected INFERENCE after lifecycle, got {om._current_phase}"
            )
        finally:
            om.release()


# ===========================================================================
# Group 3 — CUDA graph capture with NVMe-backed offload
# ===========================================================================


class TestNvmeOffloadCudaGraph:
    """CUDA graph capture and replay with NVMe-backed offload.

    Follows the L0_cuda_graphs pattern: lifecycle → INFERENCE → graph capture
    → replay. Verifies that the NVMe read path in ``schedule_transfer``
    composes with CUDA graph stream-fork synchronization.
    """

    def test_nvme_graph_capture_succeeds(self, device: torch.device, tmp_path: Path) -> None:
        """CUDA graph capture must not raise during INFERENCE with NVMe-backed offload."""
        from tests.integration._compile_helpers import capture_cuda_graph

        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode="allocation_block_transfer",
            module_patterns=MODULE_PATTERNS,
        )
        manager_name = f"test_nvme_cg_cap_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(on_cpu=True)
        proxy = om.offload(model, nvme_config)
        x = _make_input(device)
        try:
            _run_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            graph, static_out = capture_cuda_graph(proxy, x.clone(), feedback_iters=FEEDBACK_ITERS)
            assert graph is not None
            assert static_out.shape == x.shape
        finally:
            om.release()

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "CUDA graph replay with POSIX NVMe backend is not supported: "
            "os.pread is CPU-side and not captured in the graph, so on replay "
            "the shared pinned buffer contains stale data from the last "
            "capture-time read. cuFile (GDS) would work because it writes "
            "directly to GPU memory (capturable), but GDS is fundamentally "
            "unsupported on GB10 unified memory — cuFileHandleRegister fails "
            "with CU_FILE_IO_NOT_SUPPORTED (rc=5008) regardless of cuFile "
            "version (tested 1.15.1 and 1.18.1), nvidia-fs load state, or "
            "cuFileSetParameterBool flags (FORCE_COMPAT_MODE, "
            "SKIP_TOPOLOGY_DETECTION, USE_PCIP2PDMA). The error is a "
            "platform architecture limitation: no PCIe P2P path exists from "
            "the NVMe controller to the GPU's unified memory. Graph capture "
            "itself succeeds (see test_nvme_graph_capture_succeeds) — only "
            "the replay correctness is affected when using the POSIX fallback."
        ),
    )
    def test_nvme_graph_replay_matches_eager(self, device: torch.device, tmp_path: Path) -> None:
        """Graph replay output must match eager NVMe-backed inference."""
        from tests.integration._compile_helpers import capture_cuda_graph

        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode="allocation_block_transfer",
            module_patterns=MODULE_PATTERNS,
        )
        manager_name = f"test_nvme_cg_eager_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(on_cpu=True)
        proxy = om.offload(model, nvme_config)
        x = _make_input(device)
        try:
            res_eager = _run_lifecycle(proxy, x)
            checksum_eager = tensor_checksum(res_eager)

            static_input = x.clone()
            graph, static_out = capture_cuda_graph(proxy, static_input, feedback_iters=FEEDBACK_ITERS)
            graph.replay()
            torch.cuda.synchronize()
            checksum_graph = tensor_checksum(static_out)

            assert checksum_eager == checksum_graph, (
                f"Eager ({checksum_eager}) vs graph ({checksum_graph}) mismatch with NVMe offload"
            )
        finally:
            om.release()

    def test_nvme_graph_state_machine_unchanged(self, device: torch.device, tmp_path: Path) -> None:
        """CUDA graph capture and replay must not disturb OffloadManager INFERENCE state."""
        from tests.integration._compile_helpers import capture_cuda_graph

        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode="allocation_block_transfer",
            module_patterns=MODULE_PATTERNS,
        )
        manager_name = f"test_nvme_cg_state_{uuid.uuid4().hex[:8]}"
        om = get_offload_manager(manager_name)
        model = _create_model(on_cpu=True)
        proxy = om.offload(model, nvme_config)
        x = _make_input(device)
        try:
            _run_lifecycle(proxy, x)
            assert om._current_phase == OffloadPhase.INFERENCE

            static_input = x.clone()
            graph, _ = capture_cuda_graph(proxy, static_input, feedback_iters=FEEDBACK_ITERS)
            assert om._current_phase == OffloadPhase.INFERENCE, "State changed during capture"

            graph.replay()
            torch.cuda.synchronize()
            assert om._current_phase == OffloadPhase.INFERENCE, "State changed after replay"
        finally:
            om.release()


# ===========================================================================
# Group 4 — Profile save/restore roundtrip with NVMe backing
# ===========================================================================


class TestNvmeOffloadProfileRoundtrip:
    """Profile save and restore with NVMe disk offload.

    Follows the L0_compile_profile_roundtrip pattern: run NVMe-backed
    lifecycle, save profile, restore in a new manager, run lifecycle again,
    and verify inference output matches the pre-save reference.
    """

    def test_nvme_profile_save_restore_matches_reference(
        self,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Saved NVMe-backed profile must restore and produce matching inference output."""
        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode="allocation_block_transfer",
            module_patterns=MODULE_PATTERNS,
        )
        x = _make_input(device)

        # Phase 1: eager NVMe offload, run lifecycle, save profile.
        name1 = f"test_nvme_prof_save_{uuid.uuid4().hex[:8]}"
        om1 = get_offload_manager(name1)
        model1 = _create_model(on_cpu=True)
        proxy1 = om1.offload(model1, nvme_config)
        profile_dir = tmp_path / "nvme_profile"
        profile_dir.mkdir()
        try:
            ref_out = _run_lifecycle(proxy1, x)
            assert om1._current_phase == OffloadPhase.INFERENCE
            om1.save_profile(str(profile_dir))
        finally:
            om1.release()

        # Phase 2: restore profile in a new manager, run lifecycle, verify.
        name2 = f"test_nvme_prof_restore_{uuid.uuid4().hex[:8]}"
        model2 = _create_model(on_cpu=True)
        proxy2 = offload_from_profile(model2, str(profile_dir), nvme_config, name=name2)
        om2 = get_offload_manager(name2)
        try:
            _run_lifecycle(proxy2, x)
            assert om2._current_phase == OffloadPhase.INFERENCE

            with torch.no_grad():
                out = x
                for _ in range(FEEDBACK_ITERS):
                    out = proxy2(out)

            torch.testing.assert_close(
                out.float(),
                ref_out.float(),
                rtol=1e-2,
                atol=1e-2,
                msg="Restored NVMe profile output diverges from pre-save reference",
            )
        finally:
            om2.release()

    def test_nvme_profile_restore_deterministic(
        self,
        device: torch.device,
        tmp_path: Path,
    ) -> None:
        """Multiple inference calls on a restored NVMe profile must be deterministic."""
        nvme_config, _ = make_nvme_offload_config(
            tmp_path,
            transfer_mode="allocation_block_transfer",
            module_patterns=MODULE_PATTERNS,
        )
        x = _make_input(device)

        # Save phase.
        name1 = f"test_nvme_det_save_{uuid.uuid4().hex[:8]}"
        om1 = get_offload_manager(name1)
        model1 = _create_model(on_cpu=True)
        proxy1 = om1.offload(model1, nvme_config)
        profile_dir = tmp_path / "nvme_det_profile"
        profile_dir.mkdir()
        try:
            _run_lifecycle(proxy1, x)
            om1.save_profile(str(profile_dir))
        finally:
            om1.release()

        # Restore + verify determinism.
        name2 = f"test_nvme_det_restore_{uuid.uuid4().hex[:8]}"
        model2 = _create_model(on_cpu=True)
        proxy2 = offload_from_profile(model2, str(profile_dir), nvme_config, name=name2)
        om2 = get_offload_manager(name2)
        try:
            _run_lifecycle(proxy2, x)

            checksums = []
            with torch.no_grad():
                for _ in range(5):
                    out = x
                    for _ in range(FEEDBACK_ITERS):
                        out = proxy2(out)
                    checksums.append(tensor_checksum(out))

            assert len(set(checksums)) == 1, f"Inconsistent restored NVMe inference: {checksums}"
        finally:
            om2.release()
