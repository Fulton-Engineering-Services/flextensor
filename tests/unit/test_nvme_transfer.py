# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVMe transfer backends (PosixBackend, alignment, fallback)."""


import pathlib

import pytest
import torch

from flextensor.nvme_transfer import (
    NvmeBlockRef,
    PosixBackend,
    _align_up,
    is_nvidia_fs_available,
    make_nvme_backend,
)


class TestAlignUp:
    def test_exact_multiple(self) -> None:
        assert _align_up(4096, 4096) == 4096

    def test_round_up(self) -> None:
        assert _align_up(1, 4096) == 4096
        assert _align_up(4097, 4096) == 8192
        assert _align_up(8191, 4096) == 8192

    def test_zero(self) -> None:
        assert _align_up(0, 4096) == 0

    def test_small_alignment(self) -> None:
        assert _align_up(5, 512) == 512
        assert _align_up(512, 512) == 512


class TestPosixBackend:
    """Tests for the POSIX fallback backend (no GPU or kernel module required)."""

    def test_write_read_roundtrip(self, tmp_path) -> None:
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

    def test_multiple_blocks_in_one_file(self, tmp_path) -> None:
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

    def test_alignment_padding(self, tmp_path) -> None:
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
        file_size = pathlib.Path(file_path).stat().st_size
        assert file_size >= 4096

        backend.close_file(fd)
        backend.close()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA to create a GPU tensor")
    def test_rejects_gpu_tensor_on_write(self, tmp_path) -> None:
        backend = PosixBackend(alignment=4096)
        file_path = str(tmp_path / "reject.bin")
        fd = backend.open_file(file_path)

        data = torch.zeros(10, dtype=torch.uint8, device="cuda")
        with pytest.raises(ValueError, match="expects a CPU tensor"):
            backend.write_block(fd, data, offset=0)

        backend.close_file(fd)
        backend.close()


class TestNvidiaFsProbe:
    def test_probe_returns_bool(self) -> None:
        result = is_nvidia_fs_available()
        assert isinstance(result, bool)


class TestMakeNvmeBackend:
    def test_posix_mode_returns_posix_backend(self) -> None:
        backend = make_nvme_backend("posix", alignment=4096)
        assert isinstance(backend, PosixBackend)
        backend.close()

    def test_cufile_falls_back_to_posix_without_nvidia_fs(self, monkeypatch) -> None:
        monkeypatch.setattr("flextensor.nvme_transfer.is_nvidia_fs_available", lambda: False)
        backend = make_nvme_backend("cufile", alignment=4096)
        assert isinstance(backend, PosixBackend)
        backend.close()


class TestNvmeBlockRef:
    def test_fields(self) -> None:
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
