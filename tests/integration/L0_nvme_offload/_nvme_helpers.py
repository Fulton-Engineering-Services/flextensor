# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for NVMe disk-offload integration tests.

External dependencies
---------------------
- A writable directory on a local filesystem (not NFS) for NVMe block files.
  By default pytest's ``tmp_path`` is used; set ``FT_NVME_TEST_PATH`` to point
  at a real NVMe-backed directory when testing cuFile (GDS), which requires
  block-device-backed files.
- ``nvidia-fs`` kernel module loaded for cuFile tests (optional — POSIX
  fallback tests run without it).
- ``O_DIRECT`` support in the target filesystem for optimal POSIX paths; the
  backend silently falls back to buffered I/O when ``O_DIRECT`` is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from flextensor.nvme_transfer import (
    CuFileBackend,
    PosixBackend,
    is_nvidia_fs_available,
    make_nvme_backend,
)
from tests.integration._compile_helpers import make_offload_config

if TYPE_CHECKING:
    from flextensor import OffloadConfig


def _nvme_test_base(tmp_path: Path) -> Path:
    """Return the base directory for NVMe test files.

    Honors the ``FT_NVME_TEST_PATH`` environment variable so tests can write
    to a real NVMe filesystem when the default ``tmp_path`` is tmpfs.

    Args:
        tmp_path: Pytest's per-test temporary directory.

    Returns:
        Directory path (created if it does not exist).
    """
    env_path = os.environ.get("FT_NVME_TEST_PATH")
    base = Path(env_path) if env_path else tmp_path / "nvme_blocks"
    base.mkdir(parents=True, exist_ok=True)
    return base


def make_nvme_offload_config(
    tmp_path: Path,
    *,
    transfer_mode: str = "allocation_block_transfer",
    nvme_transfer_mode: str = "posix",
    num_blocks: int = 4,
    discovery_iters: int = 1,
    profiling_iters: int = 3,
    feedback_iters: int = 2,
    module_patterns: list[str] | None = None,
) -> tuple[OffloadConfig, Path]:
    """Build an ``OffloadConfig`` with NVMe disk offload enabled.

    Wraps :func:`make_offload_config` from the shared compile helpers, then
    overrides the NVMe fields. The NVMe offload path is set to a unique
    subdirectory of ``tmp_path`` (or ``FT_NVME_TEST_PATH``).

    Args:
        tmp_path: Pytest temporary directory (or ``FT_NVME_TEST_PATH`` override).
        transfer_mode: Block controller type — ``"raw_block_transfer"`` or
            ``"allocation_block_transfer"``. NVMe offload requires a block
            transfer mode.
        nvme_transfer_mode: ``"posix"`` or ``"cufile"``. cuFile auto-falls back
            to POSIX when ``nvidia-fs`` is unavailable or GDS P2P is unsupported.
        num_blocks: Number of GPU transfer blocks.
        discovery_iters: Discovery-phase forward passes.
        profiling_iters: Profiling-phase forward passes.
        feedback_iters: Per-iteration feedback loop count.
        module_patterns: Module name glob patterns for offload candidates.

    Returns:
        ``(config, nvme_dir)`` where ``nvme_dir`` is the path the block
        controller will write ``blocks.bin`` to.
    """
    config = make_offload_config(
        discovery_iters=discovery_iters,
        profiling_iters=profiling_iters,
        feedback_iters=feedback_iters,
        module_patterns=module_patterns,
        transfer_mode=transfer_mode,
        num_blocks=num_blocks,
    )

    nvme_dir = _nvme_test_base(tmp_path) / f"nvme_{os.getpid()}_{id(config):x}"
    nvme_dir.mkdir(parents=True, exist_ok=True)

    config = config.model_copy(
        update={
            "nvme_offload_enabled": True,
            "nvme_offload_path": str(nvme_dir),
            "nvme_transfer_mode": nvme_transfer_mode,
        }
    )
    return config, nvme_dir


def cufile_available() -> bool:
    """Return ``True`` if a real ``CuFileBackend`` can be instantiated.

    Probes both ``nvidia-fs`` availability and the cuFile pre-flight check
    (which fails on GB10 unified memory where GDS P2P is unsupported). Tests
    that require cuFile should skip when this returns ``False``.

    Returns:
        ``True`` if ``make_nvme_backend("cufile")`` returns a
        :class:`CuFileBackend` instance.
    """
    if not is_nvidia_fs_available():
        return False
    backend = make_nvme_backend("cufile")
    is_cufile = isinstance(backend, CuFileBackend)
    backend.close()
    return is_cufile


def posix_backend_gpu_roundtrip(
    tmp_path: Path,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    alignment: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write a random tensor to NVMe via :class:`PosixBackend`, read back to GPU.

    Args:
        tmp_path: Directory for the NVMe file.
        shape: Shape of the test tensor.
        dtype: Data type of the test tensor.
        alignment: File alignment (default 4096).

    Returns:
        ``(original_cpu, read_gpu)`` — the original CPU tensor and the tensor
        read back from NVMe into GPU memory.
    """
    backend = PosixBackend(alignment=alignment)
    file_path = str(tmp_path / "roundtrip.bin")
    fd = backend.open_file(file_path)

    data = torch.randn(shape, dtype=dtype)
    data_bytes = data.contiguous().view(torch.uint8).reshape(-1)
    block_ref = backend.write_block(fd, data_bytes, offset=0)

    gpu_buf = torch.zeros(block_ref.aligned_nbytes, dtype=torch.uint8, device="cuda")
    backend.read_block(fd, gpu_buf, block_ref.file_offset, block_ref.logical_nbytes)
    read_gpu = gpu_buf[: block_ref.logical_nbytes].view(dtype).reshape(shape).clone()

    backend.close_file(fd)
    backend.close()
    return data, read_gpu


def assert_nvme_files_exist(nvme_dir: Path) -> None:
    """Assert that NVMe weight block files were written to ``nvme_dir``.

    Args:
        nvme_dir: Directory expected to contain ``blocks.bin``.

    Raises:
        AssertionError: If ``blocks.bin`` does not exist or is empty.
    """
    blocks_file = nvme_dir / "blocks.bin"
    assert blocks_file.exists(), f"NVMe block file not found: {blocks_file}"
    assert blocks_file.stat().st_size > 0, f"NVMe block file is empty: {blocks_file}"


def make_non_nvme_config(
    *,
    transfer_mode: str = "allocation_block_transfer",
    num_blocks: int = 4,
    discovery_iters: int = 1,
    profiling_iters: int = 3,
    feedback_iters: int = 2,
    module_patterns: list[str] | None = None,
) -> OffloadConfig:
    """Build an ``OffloadConfig`` without NVMe offload (the baseline).

    Used for comparison tests that run the same model with and without NVMe
    offload to verify output equivalence.

    Args:
        transfer_mode: Block controller type.
        num_blocks: Number of GPU transfer blocks.
        discovery_iters: Discovery-phase forward passes.
        profiling_iters: Profiling-phase forward passes.
        feedback_iters: Per-iteration feedback loop count.
        module_patterns: Module name glob patterns for offload candidates.

    Returns:
        ``OffloadConfig`` with ``nvme_offload_enabled=False`` (the default).
    """
    return make_offload_config(
        discovery_iters=discovery_iters,
        profiling_iters=profiling_iters,
        feedback_iters=feedback_iters,
        module_patterns=module_patterns,
        transfer_mode=transfer_mode,
        num_blocks=num_blocks,
    )
