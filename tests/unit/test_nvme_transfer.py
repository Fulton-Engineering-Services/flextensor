# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVMe transfer backends.

Exercises ``_align_up``, :class:`PosixBackend` roundtrips (CPU and GPU),
the ``use_odirect`` flag, the ``nvidia-fs`` probe, the
:func:`make_nvme_backend` factory fallback logic (including
``use_odirect`` propagation), and :class:`NvmeBlockRef` dataclass fields.
No GPU is required for POSIX and alignment tests; GPU-tensor write tests
require CUDA and are skipped otherwise.
"""

from pathlib import Path

import pytest
import torch
from pytest import MonkeyPatch

from flextensor.nvme_transfer import (
    NvmeBlockRef,
    PosixBackend,
    _align_up,
    is_nvidia_fs_available,
    make_nvme_backend,
)


class TestAlignUp:
    """Tests for the ``_align_up`` alignment helper."""

    def test_exact_multiple(self) -> None:
        """A size already aligned must return unchanged."""
        assert _align_up(4096, 4096) == 4096

    def test_round_up(self) -> None:
        """Non-aligned sizes must round up to the next alignment boundary."""
        assert _align_up(1, 4096) == 4096
        assert _align_up(4097, 4096) == 8192
        assert _align_up(8191, 4096) == 8192

    def test_zero(self) -> None:
        """Zero must remain zero (no padding for empty blocks)."""
        assert _align_up(0, 4096) == 0

    def test_small_alignment(self) -> None:
        """Alignment must work with non-4096 values (e.g. 512-byte sectors)."""
        assert _align_up(5, 512) == 512
        assert _align_up(512, 512) == 512


class TestPosixBackend:
    """Tests for the POSIX fallback backend (no GPU or kernel module required)."""

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        """A single block written to NVMe must read back with identical bytes."""
        backend = PosixBackend(alignment=4096)
        file_path = str(tmp_path / "test_block.bin")
        fd = backend.open_file(file_path)

        data = torch.arange(1000, dtype=torch.float32)
        data_bytes = data.view(torch.uint8).reshape(-1).contiguous()
        block_ref = backend.write_block(fd, data_bytes, offset=0)
        assert block_ref.file_offset == 0
        assert block_ref.logical_nbytes == data_bytes.numel()
        assert block_ref.aligned_nbytes == _align_up(data_bytes.numel(), 4096)

        # Read back into a GPU-like buffer (use CPU for unit tests).
        gpu_buf = torch.zeros(block_ref.aligned_nbytes, dtype=torch.uint8)
        backend.read_block(fd, gpu_buf, block_ref.file_offset, block_ref.logical_nbytes)

        result_bytes = gpu_buf[: block_ref.logical_nbytes]
        result = result_bytes.view(torch.float32)[:1000]
        assert torch.equal(result, data)

        backend.close_file(fd)
        backend.close()

    def test_multiple_blocks_in_one_file(self, tmp_path: Path) -> None:
        """Multiple blocks at sequential aligned offsets must roundtrip independently."""
        backend = PosixBackend(alignment=4096)
        file_path = str(tmp_path / "multi_block.bin")
        fd = backend.open_file(file_path)

        data1 = torch.arange(500, dtype=torch.float32)
        data2 = torch.arange(300, dtype=torch.int64) * 2
        bytes1 = data1.view(torch.uint8).reshape(-1).contiguous()
        bytes2 = data2.view(torch.uint8).reshape(-1).contiguous()

        ref1 = backend.write_block(fd, bytes1, offset=0)
        ref2 = backend.write_block(fd, bytes2, offset=ref1.aligned_nbytes)

        assert ref2.file_offset == ref1.aligned_nbytes

        buf1 = torch.zeros(ref1.aligned_nbytes, dtype=torch.uint8)
        buf2 = torch.zeros(ref2.aligned_nbytes, dtype=torch.uint8)
        backend.read_block(fd, buf1, ref1.file_offset, ref1.logical_nbytes)
        backend.read_block(fd, buf2, ref2.file_offset, ref2.logical_nbytes)

        result1 = buf1[: ref1.logical_nbytes].view(torch.float32)[:500]
        result2 = buf2[: ref2.logical_nbytes].view(torch.int64)[:300]
        assert torch.equal(result1, data1)
        assert torch.equal(result2, data2)

        backend.close_file(fd)
        backend.close()

    def test_alignment_padding(self, tmp_path: Path) -> None:
        """Sub-alignment data must be padded to the alignment boundary on disk."""
        backend = PosixBackend(alignment=4096)
        file_path = str(tmp_path / "aligned.bin")
        fd = backend.open_file(file_path)

        # 100 bytes of data should be padded to 4096.
        data = torch.arange(25, dtype=torch.float32)  # 25 * 4 = 100 bytes
        data_bytes = data.view(torch.uint8).reshape(-1).contiguous()
        ref = backend.write_block(fd, data_bytes, offset=0)

        assert ref.logical_nbytes == 100
        assert ref.aligned_nbytes == 4096

        # File should be at least 4096 bytes.
        file_size = Path(file_path).stat().st_size
        assert file_size >= 4096

        backend.close_file(fd)
        backend.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to create a GPU tensor")
    def test_writes_gpu_tensor_and_roundtrips(self, tmp_path: Path) -> None:
        """``write_block`` must accept GPU tensors and roundtrip data correctly.

        On unified-memory platforms (GB10), GPU memory is host-accessible so
        ``os.pwrite`` can read directly from the GPU pointer. This test uses
        ``use_odirect=False`` (the unified-memory setting) so the write goes
        through buffered I/O.
        """
        backend = PosixBackend(alignment=4096, use_odirect=False)
        file_path = str(tmp_path / "gpu_write.bin")
        fd = backend.open_file(file_path)

        data = torch.arange(1000, dtype=torch.float32, device="cuda")
        data_bytes = data.view(torch.uint8).reshape(-1).contiguous()
        ref = backend.write_block(fd, data_bytes, offset=0)
        assert ref.logical_nbytes == data_bytes.numel()
        assert ref.aligned_nbytes == _align_up(data_bytes.numel(), 4096)

        # Read back into CPU buffer and verify.
        gpu_buf = torch.zeros(ref.aligned_nbytes, dtype=torch.uint8)
        backend.read_block(fd, gpu_buf, ref.file_offset, ref.logical_nbytes)
        result = gpu_buf[: ref.logical_nbytes].view(torch.float32)[:1000]
        assert torch.equal(result, data.cpu())

        backend.close_file(fd)
        backend.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to create a GPU tensor")
    def test_ctypes_buffer_from_gpu_tensor(self) -> None:
        """``_ctypes_buffer_from_tensor`` must create a readable ctypes buffer from a GPU pointer.

        On GB10 unified memory, the GPU pointer from ``data_ptr()`` should be
        host-accessible. This test verifies the buffer object is created with
        the correct size; reading its contents requires a working unified-memory
        path (tested in ``test_writes_gpu_tensor_and_roundtrips``).
        """
        import ctypes

        tensor = torch.arange(64, dtype=torch.uint8, device="cuda")
        buf = PosixBackend._ctypes_buffer_from_tensor(tensor)
        assert isinstance(buf, ctypes.Array)
        assert len(buf) == 64


class TestPosixBackendUseOdirect:
    """Tests for the ``use_odirect`` flag on :class:`PosixBackend`."""

    def test_default_use_odirect_true(self) -> None:
        """``PosixBackend()`` must default to ``use_odirect=True``."""
        backend = PosixBackend(alignment=4096)
        assert backend.use_odirect is True
        backend.close()

    def test_use_odirect_false_opens_buffered(self, tmp_path: Path) -> None:
        """``use_odirect=False`` must open files without ``O_DIRECT``.

        On macOS (no ``O_DIRECT``) and on unified-memory platforms, the
        flag forces buffered I/O so GPU pointers that may not meet
        ``O_DIRECT`` alignment requirements still work.
        """
        backend = PosixBackend(alignment=4096, use_odirect=False)
        file_path = str(tmp_path / "buffered.bin")
        fd = backend.open_file(file_path)

        data = torch.arange(100, dtype=torch.float32)
        data_bytes = data.view(torch.uint8).reshape(-1).contiguous()
        ref = backend.write_block(fd, data_bytes, offset=0)
        assert ref.logical_nbytes == data_bytes.numel()

        buf = torch.zeros(ref.aligned_nbytes, dtype=torch.uint8)
        backend.read_block(fd, buf, ref.file_offset, ref.logical_nbytes)
        result = buf[: ref.logical_nbytes].view(torch.float32)[:100]
        assert torch.equal(result, data)

        backend.close_file(fd)
        backend.close()


class TestNvidiaFsProbe:
    """Tests for the ``nvidia-fs`` kernel module probe."""

    def test_probe_returns_bool(self) -> None:
        """``is_nvidia_fs_available`` must return a boolean (not raise)."""
        result = is_nvidia_fs_available()
        assert isinstance(result, bool)


class TestMakeNvmeBackend:
    """Tests for the ``make_nvme_backend`` factory and fallback logic."""

    def test_posix_mode_returns_posix_backend(self) -> None:
        """Explicit ``'posix'`` mode must always return a :class:`PosixBackend`."""
        backend = make_nvme_backend("posix", alignment=4096)
        assert isinstance(backend, PosixBackend)
        backend.close()

    def test_cufile_falls_back_to_posix_without_nvidia_fs(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """``'cufile'`` mode must fall back to POSIX when ``nvidia-fs`` is absent."""
        monkeypatch.setattr("flextensor.nvme_transfer.is_nvidia_fs_available", lambda: False)
        backend = make_nvme_backend("cufile", alignment=4096)
        assert isinstance(backend, PosixBackend)
        backend.close()

    def test_posix_mode_propagates_use_odirect_false(self) -> None:
        """``make_nvme_backend('posix', use_odirect=False)`` must set the flag."""
        backend = make_nvme_backend("posix", alignment=4096, use_odirect=False)
        assert isinstance(backend, PosixBackend)
        assert backend.use_odirect is False
        backend.close()

    def test_posix_mode_default_use_odirect_true(self) -> None:
        """``make_nvme_backend('posix')`` must default to ``use_odirect=True``."""
        backend = make_nvme_backend("posix", alignment=4096)
        assert isinstance(backend, PosixBackend)
        assert backend.use_odirect is True
        backend.close()

    def test_cufile_fallback_propagates_use_odirect_false(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """``make_nvme_backend('cufile', use_odirect=False)`` fallback must carry the flag."""
        monkeypatch.setattr("flextensor.nvme_transfer.is_nvidia_fs_available", lambda: False)
        backend = make_nvme_backend("cufile", alignment=4096, use_odirect=False)
        assert isinstance(backend, PosixBackend)
        assert backend.use_odirect is False
        backend.close()


class TestNvmeBlockRef:
    """Tests for the :class:`NvmeBlockRef` dataclass field storage."""

    def test_fields(self) -> None:
        """All constructor arguments must be stored as attributes."""
        ref = NvmeBlockRef(
            file_path="/mnt/nvme/test.bin",
            file_offset=4096,
            logical_nbytes=100,
            aligned_nbytes=4096,
        )
        assert ref.file_path == "/mnt/nvme/test.bin"
        assert ref.file_offset == 4096
        assert ref.logical_nbytes == 100
        assert ref.aligned_nbytes == 4096
