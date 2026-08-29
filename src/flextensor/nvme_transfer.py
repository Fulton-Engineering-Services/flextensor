# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVMe transfer backends for disk-based weight offload.

Provides :class:`NvmeTransferBackend` implementations that write combined
weight blocks to NVMe files during construction and read them back to GPU
memory during inference via cuFile (GDS) or POSIX ``pread``.

The block controllers (:class:`~flextensor.loaders.RawBlockController`,
:class:`~flextensor.loaders.AllocationBlockController`) call
:meth:`NvmeTransferBackend.write_block` during construction (after weights are
packed into a contiguous uint8 block) and
:meth:`NvmeTransferBackend.read_block` from ``schedule_transfer`` during
inference (replacing the CPU→GPU ``copy_`` path).
"""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

from flextensor.config import NvmeTransferMode  # noqa: TC001 — beartype resolves at runtime

logger = logging.getLogger(__name__)

__all__ = [
    "CuFileBackend",
    "NvmeBlockRef",
    "NvmeTransferBackend",
    "PosixBackend",
    "is_nvidia_fs_available",
    "make_nvme_backend",
]

_CUFILE_LIB_PATHS = [
    "/usr/local/cuda/lib64/libcufile.so",
    "/usr/local/cuda-13.0/lib64/libcufile.so",
    "/usr/lib/aarch64-linux-gnu/libcufile.so",
]

_NVIDIA_FS_VERSION_PATH = "/proc/driver/nvidia-fs/version"


def is_nvidia_fs_available() -> bool:
    """Return ``True`` if the ``nvidia-fs`` kernel module is loaded.

    Probes ``/proc/driver/nvidia-fs/version`` — the GDS kernel module creates
    this proc entry when loaded (``modprobe nvidia-fs``).
    """
    return Path(_NVIDIA_FS_VERSION_PATH).exists()


def _align_up(size: int, alignment: int) -> int:
    """Round *size* up to the next multiple of *alignment*."""
    return (size + alignment - 1) // alignment * alignment


@dataclass
class NvmeBlockRef:
    """Metadata for a weight block stored on NVMe.

    Attributes:
        file_path: Absolute path to the NVMe file.
        file_offset: Byte offset within the file (always alignment-aligned).
        logical_nbytes: True unpadded size of the weight block.
        aligned_nbytes: Padded size (multiple of ``alignment``) written to disk.
    """

    file_path: str
    file_offset: int
    logical_nbytes: int
    aligned_nbytes: int


@runtime_checkable
class NvmeTransferBackend(Protocol):
    """Protocol for NVMe transfer backends.

    Implementations write weight blocks to NVMe files during construction and
    read them back to GPU memory during inference.
    """

    alignment: int

    def open_file(self, file_path: str) -> int:
        """Open an NVMe file for read/write and return the file descriptor."""
        ...

    def close_file(self, fd: int) -> None:
        """Close an NVMe file descriptor."""
        ...

    def write_block(self, fd: int, data: torch.Tensor, offset: int) -> NvmeBlockRef:
        """Write a uint8 weight block to an NVMe file.

        Args:
            fd: File descriptor from :meth:`open_file`.
            data: Contiguous uint8 tensor on CPU containing the packed weights.
            offset: Byte offset within the file (must be alignment-aligned).

        Returns:
            :class:`NvmeBlockRef` with the logical and aligned sizes.
        """
        ...

    def read_block(self, fd: int, gpu_buf: torch.Tensor, offset: int, nbytes: int) -> None:
        """Read a weight block from NVMe into a GPU buffer.

        Args:
            fd: File descriptor from :meth:`open_file`.
            gpu_buf: GPU tensor (uint8) to read into. Must be alignment-aligned
                for cuFile.
            offset: Byte offset within the file (must be alignment-aligned).
            nbytes: Number of bytes to read (will be rounded up to alignment).
        """
        ...

    def close(self) -> None:
        """Release any backend-level resources (cached descriptors, etc.)."""
        ...


class PosixBackend:
    """Fallback NVMe transfer backend using POSIX ``pread`` + ``copy_``.

    Reads weight blocks from NVMe into pinned CPU memory, then copies to the
    GPU block view. On GB10 unified memory, the pinned CPU buffer is in the
    same physical DRAM as the GPU, so no bounce-buffer copy is needed — but
    the ``copy_`` call is still required to place the data in the GPU block
    view's storage.

    No kernel modules or CUDA libraries required; works as a universal
    fallback.
    """

    def __init__(self, alignment: int = 4096) -> None:
        self.alignment = alignment
        self._pinned_buf: torch.Tensor | None = None
        self._pinned_buf_size: int = 0

    def open_file(self, file_path: str) -> int:
        o_direct = getattr(os, "O_DIRECT", 0)
        flags = os.O_RDWR | os.O_CREAT | o_direct
        try:
            fd = os.open(file_path, flags, 0o644)
        except OSError:
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT, 0o644)
        return fd

    def close_file(self, fd: int) -> None:
        os.close(fd)

    def write_block(self, fd: int, data: torch.Tensor, offset: int) -> NvmeBlockRef:
        """Write a uint8 block to the file, padding to alignment."""
        if data.device.type != "cpu":
            raise ValueError(f"PosixBackend.write_block expects a CPU tensor; got {data.device}")
        logical_nbytes = data.numel() * data.element_size()
        aligned_nbytes = _align_up(logical_nbytes, self.alignment)

        data_bytes = data.contiguous().view(torch.uint8).flatten()
        if aligned_nbytes > logical_nbytes:
            padded = torch.zeros(aligned_nbytes, dtype=torch.uint8)
            padded[:logical_nbytes] = data_bytes
            buf = padded.numpy()
        else:
            buf = data_bytes.numpy()

        written = os.pwrite(fd, buf.tobytes(), offset)
        if written != aligned_nbytes:
            raise OSError(f"Short write: expected {aligned_nbytes}, wrote {written}")

        return NvmeBlockRef(
            file_path="",
            file_offset=offset,
            logical_nbytes=logical_nbytes,
            aligned_nbytes=aligned_nbytes,
        )

    def _ensure_pinned_buf(self, nbytes: int) -> torch.Tensor:
        """Get or grow a pinned CPU buffer for staging reads.

        Falls back to a non-pinned tensor when ``pin_memory()`` is unavailable
        (e.g. CPU-only hosts without CUDA).
        """
        if self._pinned_buf is None or self._pinned_buf_size < nbytes:
            self._pinned_buf_size = _align_up(max(nbytes, self.alignment), self.alignment)
            buf = torch.zeros(self._pinned_buf_size, dtype=torch.uint8)
            with contextlib.suppress(RuntimeError):
                buf = buf.pin_memory()
            self._pinned_buf = buf
        return self._pinned_buf

    def read_block(self, fd: int, gpu_buf: torch.Tensor, offset: int, nbytes: int) -> None:
        """Read from NVMe into pinned CPU, then copy to GPU buffer."""
        aligned_nbytes = _align_up(nbytes, self.alignment)
        pinned = self._ensure_pinned_buf(aligned_nbytes)

        buf_view = pinned[:aligned_nbytes]
        buf_bytes = os.pread(fd, aligned_nbytes, offset)
        if len(buf_bytes) != aligned_nbytes:
            raise OSError(f"Short read: expected {aligned_nbytes}, got {len(buf_bytes)}")
        buf_view.copy_(torch.frombuffer(buf_bytes, dtype=torch.uint8))

        gpu_view = gpu_buf.view(torch.uint8).flatten()[:nbytes]
        gpu_view.copy_(buf_view[:nbytes], non_blocking=True)

    def close(self) -> None:
        self._pinned_buf = None
        self._pinned_buf_size = 0


class CuFileBackend:
    """cuFile (GDS) NVMe transfer backend — direct NVMe→GPU reads.

    Uses ``cuFileRead`` to transfer weight blocks directly from NVMe storage
    into GPU memory without staging through CPU buffers. Requires:

    * The ``nvidia-fs`` kernel module loaded (``modprobe nvidia-fs``).
    * ``libcufile.so`` available on the library path.
    * 4K-aligned file offsets, GPU buffer addresses, and read sizes.
    """

    def __init__(self, alignment: int = 4096) -> None:
        self.alignment = alignment
        self._cufile: Any = None
        self._lib: Any = None
        self._handles: dict[int, Any] = {}
        self._init_cufile()
        self._preflight_check()

    def _preflight_check(self) -> None:
        """Verify cuFileHandleRegister works by registering a temp file.

        On platforms where the GPU model is not supported for GDS (e.g. GB10
        unified memory), ``cuFileDriverOpen`` succeeds but
        ``cuFileHandleRegister`` fails with rc=5008. This pre-flight detects
        that and raises ``RuntimeError`` so :func:`make_nvme_backend` falls
        back to :class:`PosixBackend`.
        """
        import ctypes
        import tempfile

        fd, tmp_path = tempfile.mkstemp(prefix="cufile_preflight_", suffix=".bin")
        try:
            os.ftruncate(fd, self.alignment)
            handle = ctypes.c_void_p(None)
            c_fd = ctypes.c_int(fd)
            rc = self._cufile["cuFileHandleRegister"](ctypes.byref(handle), ctypes.byref(c_fd))
            if rc != 0:
                raise RuntimeError(
                    f"cuFileHandleRegister pre-flight failed with rc={rc}. The GPU "
                    f"may not support GDS P2P DMA (e.g. GB10 unified memory). "
                    f"Use nvme_transfer_mode='posix' for NVMe reads via pread."
                )
            self._cufile["cuFileHandleDeregister"](ctypes.c_void_p(handle.value))
        finally:
            os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _init_cufile(self) -> None:
        """Lazy-load libcufile via ctypes and call cuFileDriverInit."""
        import ctypes
        import ctypes.util

        lib_path = None
        for candidate in _CUFILE_LIB_PATHS:
            if Path(candidate).exists():
                lib_path = candidate
                break
        if lib_path is None:
            # Try LD_LIBRARY_PATH discovery
            try:
                lib_path = ctypes.util.find_library("cufile")
            except OSError:
                lib_path = None
        if lib_path is None:
            raise RuntimeError(
                "CuFileBackend: libcufile.so not found. Install the CUDA "
                "libcufile package or use nvme_transfer_mode='posix'."
            )

        self._lib = ctypes.CDLL(lib_path)

        driver_open = self._lib.cuFileDriverOpen
        driver_open.restype = ctypes.c_int
        rc = driver_open()
        if rc != 0:
            raise RuntimeError(
                f"cuFileDriverOpen failed with rc={rc}. Ensure nvidia-fs is loaded "
                f"(modprobe nvidia-fs) and libcufile.so is compatible with the "
                f"installed NVIDIA driver."
            )

        self._cufile = {
            "cuFileDriverOpen": self._lib.cuFileDriverOpen,
            "cuFileDriverClose": self._lib.cuFileDriverClose,
            "cuFileHandleRegister": self._lib.cuFileHandleRegister,
            "cuFileHandleDeregister": self._lib.cuFileHandleDeregister,
            "cuFileRead": self._lib.cuFileRead,
            "cuFileWrite": self._lib.cuFileWrite,
        }

        read_fn = self._cufile["cuFileRead"]
        read_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        read_fn.restype = ctypes.c_ssize_t

        write_fn = self._cufile["cuFileWrite"]
        write_fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        write_fn.restype = ctypes.c_ssize_t

        register_fn = self._cufile["cuFileHandleRegister"]
        register_fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        ]
        register_fn.restype = ctypes.c_int

        deregister_fn = self._cufile["cuFileHandleDeregister"]
        deregister_fn.argtypes = [ctypes.c_void_p]
        deregister_fn.restype = ctypes.c_int

    def open_file(self, file_path: str) -> int:
        """Open a file with O_DIRECT and register it with cuFile."""
        import ctypes

        o_direct = getattr(os, "O_DIRECT", 0)
        flags = os.O_RDWR | os.O_CREAT | o_direct
        try:
            fd = os.open(file_path, flags, 0o644)
        except OSError:
            fd = os.open(file_path, os.O_RDWR | os.O_CREAT, 0o644)

        handle = ctypes.c_void_p(None)
        c_fd = ctypes.c_int(fd)
        rc = self._cufile["cuFileHandleRegister"](ctypes.byref(handle), ctypes.byref(c_fd))
        if rc != 0:
            os.close(fd)
            raise RuntimeError(
                f"cuFileHandleRegister failed with rc={rc} for {file_path}. "
                f"The file may not be on a filesystem that supports O_DIRECT, "
                f"or nvidia-fs may not have a GDS peer mapping to this NVMe device."
            )

        # Pack (fd, cufile_handle) into a single int for storage.
        # We keep a dict mapping fd -> cufile handle for deregistration.
        self._handles: dict[int, Any] = getattr(self, "_handles", {})
        self._handles[fd] = handle.value
        return fd

    def close_file(self, fd: int) -> None:
        """Deregister the cuFile handle and close the file descriptor."""
        import ctypes

        handle = self._handles.pop(fd, None)
        if handle is not None:
            self._cufile["cuFileHandleDeregister"](ctypes.c_void_p(handle))
        os.close(fd)

    def write_block(self, fd: int, data: torch.Tensor, offset: int) -> NvmeBlockRef:
        """Write a uint8 block to NVMe via cuFile (CPU pointer source)."""
        if data.device.type != "cpu":
            raise ValueError(f"CuFileBackend.write_block expects a CPU tensor; got {data.device}")
        logical_nbytes = data.numel() * data.element_size()
        aligned_nbytes = _align_up(logical_nbytes, self.alignment)

        data_bytes = data.contiguous().view(torch.uint8).flatten()
        if aligned_nbytes > logical_nbytes:
            padded = torch.zeros(aligned_nbytes, dtype=torch.uint8)
            padded[:logical_nbytes] = data_bytes
            src_ptr = padded.data_ptr()
        else:
            src_ptr = data_bytes.data_ptr()

        handle = self._handles[fd]
        written = self._cufile["cuFileWrite"](
            handle,
            src_ptr,
            aligned_nbytes,
            offset,
        )
        if written < 0:
            raise OSError(f"cuFileWrite failed with rc={written}")
        if written != aligned_nbytes:
            raise OSError(f"Short cuFile write: expected {aligned_nbytes}, wrote {written}")

        return NvmeBlockRef(
            file_path="",
            file_offset=offset,
            logical_nbytes=logical_nbytes,
            aligned_nbytes=aligned_nbytes,
        )

    def read_block(self, fd: int, gpu_buf: torch.Tensor, offset: int, nbytes: int) -> None:
        """Read directly from NVMe into GPU memory via cuFileRead."""
        aligned_nbytes = _align_up(nbytes, self.alignment)
        gpu_ptr = gpu_buf.view(torch.uint8).flatten().data_ptr()

        handle = self._handles[fd]
        read = self._cufile["cuFileRead"](
            handle,
            gpu_ptr,
            aligned_nbytes,
            offset,
        )
        if read < 0:
            raise OSError(f"cuFileRead failed with rc={read}")
        if read != aligned_nbytes:
            raise OSError(f"Short cuFile read: expected {aligned_nbytes}, got {read}")

    def close(self) -> None:
        if self._cufile is not None:
            try:
                self._cufile["cuFileDriverClose"]()
            except Exception:
                logger.debug("cuFileDriverClose failed during shutdown", exc_info=True)
        self._cufile = None
        self._lib = None


def make_nvme_backend(
    mode: NvmeTransferMode,
    alignment: int = 4096,
) -> NvmeTransferBackend:
    """Create an NVMe transfer backend from the configured mode.

    Falls back to :class:`PosixBackend` with a ``WARNING`` when:

    * ``mode="cufile"`` but ``nvidia-fs`` is not loaded.
    * ``mode="cufile"`` but ``libcufile.so`` cannot be found or initialized.

    Args:
        mode: ``"cufile"`` for GDS direct-to-GPU, ``"posix"`` for pread+copy.
        alignment: File/device alignment (default 4096 for standard NVMe).

    Returns:
        A :class:`NvmeTransferBackend` instance.
    """
    if mode == "cufile":
        if not is_nvidia_fs_available():
            logger.warning(
                "nvme_transfer_mode='cufile' requested but nvidia-fs is not loaded "
                "(%s not found). Falling back to 'posix'. Run 'modprobe nvidia-fs' "
                "to enable cuFile (GDS) direct-to-GPU reads.",
                _NVIDIA_FS_VERSION_PATH,
            )
            return PosixBackend(alignment=alignment)

        try:
            return CuFileBackend(alignment=alignment)
        except RuntimeError as exc:
            logger.warning(
                "CuFileBackend initialization failed (%s). Falling back to 'posix'.",
                exc,
            )
            return PosixBackend(alignment=alignment)

    return PosixBackend(alignment=alignment)
