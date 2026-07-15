# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host memory pinning helpers: :class:`HostPinner` over PyTorch's pinned
allocator and ``cudaHostRegister`` in-place registration. Construct via
:func:`make_host_pinner`; see ``docs/explanation/configuration.md`` and
``docs/how-to/troubleshooting.md`` for user-facing semantics.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Final, Literal

import torch

__all__ = [
    "HostPinRegistry",
    "HostPinner",
    "NoOpHostPinner",
    "PinnedMemoryMode",
    "is_available",
    "make_host_pinner",
]

logger = logging.getLogger(__name__)

PinnedMemoryMode = Literal["torch", "host_register"]
"""The two supported pinning strategies for ``pinned_memory_mode``."""

# Flag values from ``driver_types.h`` in the CUDA toolkit: Default=0,
# Portable=1, Mapped=2. We pass Portable so the registration is visible across
# CUDA contexts.
_CUDA_HOST_REGISTER_PORTABLE: Final[int] = 1

_CUDART_PROBE_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    RuntimeError,
    OSError,
    AttributeError,
    ImportError,
)


def _probe_cudart() -> Any:
    """Return :func:`torch.cuda.cudart()` if it exposes the calls we need.

    Probes fresh on every call. The probe is intentionally **not cached**:
    ``torch.cuda.cudart()`` and ``torch.cuda.is_available()`` are both
    memoized inside PyTorch (``functools.lru_cache``), so the steady-state
    cost is two ``lru_cache`` lookups plus two ``hasattr`` checks —
    negligible next to the ``cudaHostRegister`` syscall the result is fed
    into. Skipping our own cache eliminates a class of foot-gun where a
    transient init-order or driver glitch (e.g. an early call before CUDA
    finished initialising) would have permanently disabled host_register
    pinning for the rest of the process.

    Silent on purpose: callers attach context (raise / log) appropriately.
    See :func:`is_available`, :meth:`HostPinRegistry.pin_in_place`, and
    :func:`make_host_pinner` for the public surfaces.

    Returns ``None`` when the runtime isn't usable (no CUDA, older PyTorch
    build, broken libcudart, etc.) — callers should fall back to
    :meth:`torch.Tensor.pin_memory` in that case.
    """
    if not torch.cuda.is_available():
        return None
    try:
        candidate = torch.cuda.cudart()
    except _CUDART_PROBE_EXCEPTIONS:
        return None
    if not hasattr(candidate, "cudaHostRegister") or not hasattr(candidate, "cudaHostUnregister"):
        return None
    return candidate


def is_available() -> bool:
    """Return True if manual host pinning via ``cudaHostRegister`` is usable."""
    return _probe_cudart() is not None


def _host_register_unavailability_reason() -> tuple[str | None, BaseException | None]:
    """Probe the CUDA runtime and report why host_register pinning is unavailable.

    Returns ``(None, None)`` when host_register pinning IS available.
    On failure, returns ``(reason_string, exc_or_None)``. Callers that log
    should pass the returned exception as ``exc_info=`` so the underlying
    traceback (driver mismatch, .so load failure, etc.) reaches the
    operator who has to act on it. Used by :func:`make_host_pinner` for
    its fallback WARNING.

    The probe runs fresh — same code path as :func:`_probe_cudart` — so
    the returned reason matches whatever ``torch.cuda.*`` reports right
    now, which is what the operator can act on.
    """
    if not torch.cuda.is_available():
        return ("torch.cuda.is_available() returned False", None)
    try:
        candidate = torch.cuda.cudart()
    except _CUDART_PROBE_EXCEPTIONS as exc:
        return (f"torch.cuda.cudart() raised: {exc!r}", exc)
    if not hasattr(candidate, "cudaHostRegister") or not hasattr(candidate, "cudaHostUnregister"):
        return ("torch.cuda.cudart() lacks cudaHostRegister/cudaHostUnregister", None)
    return (None, None)


def _rc_value(rc: Any) -> int:
    """Convert a cudart return code into an int regardless of its concrete type.

    PyTorch's cudart binding returns a ``cudaError`` enum (``.value`` is the int
    error code). Some older builds or tests may pass a plain int instead.
    """
    return rc.value if hasattr(rc, "value") else int(rc)


def _check_rc(rc: Any, op: str, *, ptr: int | None = None, nbytes: int | None = None) -> None:
    """Raise ``RuntimeError`` on non-zero cudart rc, including ptr/size when known."""
    if _rc_value(rc) == 0:
        return
    where = f" (ptr={ptr:#x}, size={nbytes})" if ptr is not None else ""
    raise RuntimeError(f"{op} failed with rc={_rc_value(rc)}{where}")


def _warn_cudart_unavailable(retained_ptrs: list[int]) -> None:
    """Warn that ``torch.cuda.cudart()`` is no longer reachable so the listed
    pinned regions could not be unregistered.

    The registry retains these entries (and therefore the backing tensors'
    storages), so PyTorch's allocator cannot recycle them while the kernel
    still considers the pages page-locked. The kernel-side pin persists until
    process exit.

    No-op if ``retained_ptrs`` is empty.
    """
    if not retained_ptrs:
        return
    if len(retained_ptrs) == 1:
        logger.warning(
            "HostPinRegistry: torch.cuda.cudart() became unavailable; "
            "cudaHostUnregister(%#x) was not called. The registry retains "
            "the entry so the storage can't be recycled, but the kernel-side "
            "pin is held until process exit.",
            retained_ptrs[0],
        )
    else:
        logger.warning(
            "HostPinRegistry: torch.cuda.cudart() became unavailable; "
            "%d pinned region(s) were retained in the registry. Their "
            "storages can't be recycled by PyTorch, but the kernel-side "
            "pins are held until process exit.",
            len(retained_ptrs),
        )


def _warn_unregister_failed(ptr: int, detail: str, *, exc_info: bool = False) -> None:
    """Warn that ``cudaHostUnregister`` did not succeed for ``ptr``.
    ``detail`` describes how it failed (e.g. ``"returned 712"`` or
    ``"raised RuntimeError"``). The registry entry is retained so the
    backing storage isn't freed under a still-page-locked page.
    """
    logger.warning(
        "cudaHostUnregister(%#x) %s; registry entry was retained to keep "
        "the backing storage alive. The kernel-side pin is held until "
        "process exit.",
        ptr,
        detail,
        exc_info=exc_info,
    )


class HostPinRegistry:
    """Track ``cudaHostRegister`` registrations so they can be released together.

    One registry instance is typically owned by a
    :class:`~flextensor.tensor_manager.TensorManager` and shared with the
    processors, loaders, and allocation blocks that it creates. Call
    :meth:`release_all` from the manager's shutdown path.

    All public methods are thread-safe; the lock is held across each
    cudart call so the same pointer can't be double-(un)registered.

    The registry retains a **strong reference** to each backing tensor, so
    its storage stays alive until :meth:`release` / :meth:`release_all`;
    callers don't need a separate keep-alive list.

    ``cudaHostRegister`` state is per-process. Don't share a registry
    across processes — children must build their own via
    :func:`make_host_pinner` after CUDA is initialised, and must not
    release pins inherited from the parent. Prefer ``multiprocessing``
    start method ``"spawn"`` over fork.
    """

    def __init__(self) -> None:
        # Key: data pointer (int). Value: (size_bytes, backing tensor keeping storage alive).
        self._entries: dict[int, tuple[int, torch.Tensor]] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def is_registered(self, ptr: int) -> bool:
        with self._lock:
            return ptr in self._entries

    def pin_in_place(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin ``tensor``'s storage in place via ``cudaHostRegister``.

        Entries are keyed by storage address, so a tensor and a view of
        the same tensor register only once. The first pin's size and
        tensor reference are the ones kept — re-pinning a view does not
        shrink the registered byte range.

        No-op for non-CPU tensors, meta tensors, empty tensors, tensors
        with a null storage pointer, tensors already pinned (by PyTorch's
        allocator or otherwise), and pointers already registered here.
        Returns the same tensor object for chaining.

        Raises:
            RuntimeError: If the CUDA runtime cannot be reached via
                :func:`torch.cuda.cudart` or the registration call itself
                returns a non-zero error code.
        """
        if tensor.device.type != "cpu":
            return tensor
        if tensor.is_meta or tensor.numel() == 0:
            return tensor
        if tensor.is_pinned():
            return tensor

        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if ptr == 0:
            logger.debug(
                "host_register skipped: tensor has null storage pointer (numel=%d, dtype=%s)",
                tensor.numel(),
                tensor.dtype,
            )
            return tensor

        nbytes = storage.nbytes()

        with self._lock:
            if ptr in self._entries:
                # Already registered — keep the first backing tensor reference.
                return tensor

            cudart = _probe_cudart()
            if cudart is None:
                reason, cause = _host_register_unavailability_reason()
                raise RuntimeError(
                    f"Manual host pinning requested but the CUDA runtime is not usable "
                    f"({reason or 'unspecified cause'}). Ensure CUDA is installed or "
                    f"set pinned_memory_mode='torch'."
                ) from cause

            rc = cudart.cudaHostRegister(ptr, nbytes, _CUDA_HOST_REGISTER_PORTABLE)
            _check_rc(rc, "cudaHostRegister", ptr=ptr, nbytes=nbytes)
            # Keep a reference to the tensor so the storage (and pointer)
            # can't be freed before we unregister.
            self._entries[ptr] = (nbytes, tensor)
            registry_size = len(self._entries)

        logger.debug("host-pinned %d bytes at %#x (registry size=%d)", nbytes, ptr, registry_size)
        return tensor

    def release(self, ptr: int) -> bool:
        """Unregister a single pointer.

        Returns ``True`` only when ``cudaHostUnregister`` ran and returned a
        success code. Returns ``False`` if the pointer wasn't tracked, the
        CUDA runtime is no longer reachable, or the call returned a non-zero
        error code. Use :meth:`is_registered` to query "was this pointer
        tracked" without attempting to unregister.

        The cudart-unavailable and non-zero-rc branches log at WARNING with
        ``ptr``, ``rc`` (where applicable), and the call name so operators
        running at the default log level can see when registry and kernel
        state have drifted apart. The "pointer wasn't tracked" branch returns
        ``False`` without a WARNING (DEBUG-level only) — callers can
        pre-check with :meth:`is_registered` if they need to distinguish
        that case.

        On failure the entry stays tracked so the tensor is not freed and
        :meth:`release` can be called again to retry.
        """
        with self._lock:
            if ptr not in self._entries:
                logger.debug(
                    "HostPinRegistry.release(%#x): pointer is not tracked by this registry; "
                    "no cudaHostUnregister call was made.",
                    ptr,
                )
                return False

            cudart = _probe_cudart()
            if cudart is None:
                _warn_cudart_unavailable([ptr])
                return False

            try:
                rc = cudart.cudaHostUnregister(ptr)
            except Exception as exc:
                _warn_unregister_failed(ptr, f"raised {type(exc).__name__}", exc_info=True)
                return False

            if _rc_value(rc) != 0:
                _warn_unregister_failed(ptr, f"returned {rc}")
                return False

            self._entries.pop(ptr, None)
            return True

    def release_all(self) -> None:
        """Unregister every pointer the registry is tracking.

        Logs at WARNING when the CUDA runtime is no longer reachable (so the
        unregister calls are skipped) and per ``cudaHostUnregister`` failure,
        so operators running at the default log level see when registry and
        kernel state have drifted apart at shutdown.

        Successfully released entries are removed; failed ones stay tracked
        so their tensors are not freed and :meth:`release_all` can be called
        again to retry.
        """
        with self._lock:
            if not self._entries:
                return

            cudart = _probe_cudart()
            if cudart is None:
                _warn_cudart_unavailable(list(self._entries))
                return

            for ptr in list(self._entries):
                try:
                    rc = cudart.cudaHostUnregister(ptr)
                except Exception as exc:
                    _warn_unregister_failed(ptr, f"raised {type(exc).__name__}", exc_info=True)
                    continue
                if _rc_value(rc) != 0:
                    _warn_unregister_failed(ptr, f"returned {rc}")
                    continue
                self._entries.pop(ptr, None)


class HostPinner:
    """Unified entry point for pinning a CPU tensor for fast GPU transfer.

    Encapsulates the ``pinned_memory_mode`` dispatch so call sites don't have
    to branch on ``"torch"`` vs ``"host_register"``. Always usable: an instance
    constructed without a registry maps :meth:`pin` to
    :meth:`torch.Tensor.pin_memory` so callers can hold a single object whether
    or not manual pinning is in effect.

    Construction
    ------------
    - ``HostPinner()`` — torch mode (no copy avoidance).
    - ``HostPinner(HostPinRegistry())`` — host_register mode; the registry's
      ``release_all`` must be invoked on shutdown.

    See :func:`is_available` for whether host_register mode is supported on the
    current platform.
    """

    def __init__(self, registry: HostPinRegistry | None = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> HostPinRegistry | None:
        """The bound :class:`HostPinRegistry`, or ``None`` in torch mode."""
        return self._registry

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin ``tensor`` for fast non-blocking GPU transfer.

        Strict in both modes — failures raise ``RuntimeError``:

        - **host_register mode** (registry attached): the same tensor with
          its existing storage pinned in place via ``cudaHostRegister``.
        - **torch mode** (no registry): a new tensor backed by a fresh
          pinned allocation from PyTorch's caching pinned allocator. A
          ``tensor.pin_memory()`` failure (e.g. ``RLIMIT_MEMLOCK``
          exhaustion) raises.

        Requires a CUDA host; constructing a :class:`HostPinner` on a
        CPU-only host is a misuse. :func:`make_host_pinner` enforces this
        at config time.

        Callers should always reassign: ``t = pinner.pin(t)``.

        No-op (returns the tensor unchanged) for non-CPU tensors, meta
        tensors, empty tensors, and tensors that are already pinned.
        """
        if self._should_skip_pin(tensor):
            return tensor
        if self._registry is not None:
            self._registry.pin_in_place(tensor)
            return tensor
        return tensor.pin_memory()

    @staticmethod
    def _should_skip_pin(tensor: torch.Tensor) -> bool:
        """Return True if ``tensor`` cannot/need-not be pinned: non-CPU,
        meta, empty, or already pinned.
        """
        if tensor.device.type != "cpu":
            return True
        if tensor.is_meta or tensor.numel() == 0:
            return True
        return tensor.is_pinned()

    def is_pinned(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` is currently pinned by this pinner.

        Detects both PyTorch's pinned allocator (torch mode) and this
        pinner's :class:`HostPinRegistry` (host_register mode). Returns
        ``False`` for tensors with no usable storage pointer (e.g. meta,
        lazy, or fake tensors whose ``untyped_storage()`` raises); the
        underlying error is logged at DEBUG.
        """
        if tensor.is_pinned():
            return True
        if self._registry is None:
            return False
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except RuntimeError:
            logger.debug(
                "HostPinner.is_pinned: tensor.untyped_storage().data_ptr() raised; treating as not pinned.",
                exc_info=True,
            )
            return False
        return self._registry.is_registered(ptr)

    def release_all(self) -> None:
        """Release every registration owned by the bound registry.

        No-op in torch mode.
        """
        if self._registry is not None:
            self._registry.release_all()


class NoOpHostPinner(HostPinner):
    """Pinner that performs no pinning — every ``pin`` call returns the tensor
    unchanged. Returned by :func:`make_host_pinner` when ``pinned_memory=False``
    so call sites can hold a typed :class:`HostPinner` without each having to
    branch on the public flag.
    """

    def __init__(self) -> None:
        super().__init__(registry=None)

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def make_host_pinner(pinned_memory: bool, mode: PinnedMemoryMode) -> HostPinner:
    """Build a :class:`HostPinner` from the public ``(pinned_memory, mode)`` config.

    Returns :class:`NoOpHostPinner` when ``pinned_memory=False`` (the
    legitimate opt-out). For ``pinned_memory=True``, requires a CUDA host:
    offloading without a GPU has no purpose, and silently degrading to
    pageable transfers would mask the misconfiguration as a perf regression.

    For ``mode="host_register"`` returns a :class:`HostPinner` backed by a
    fresh :class:`HostPinRegistry` when cudart is usable; on a CUDA host
    where cudart is broken or missing, falls back to torch mode and
    surfaces a ``WARNING`` log naming the cause.

    Args:
        pinned_memory: Whether pinning is requested at all. ``False``
            short-circuits to a :class:`NoOpHostPinner` regardless of
            ``mode``.
        mode: Pinning strategy when ``pinned_memory=True`` —
            :class:`PinnedMemoryMode` (``"torch"`` or ``"host_register"``).

    Returns:
        A :class:`HostPinner` (or :class:`NoOpHostPinner` when pinning is
        disabled) wired to the appropriate backend. Callers always get a
        usable :class:`HostPinner`; only the kind of pinning differs.

    Raises:
        RuntimeError: If ``pinned_memory=True`` but
            ``torch.cuda.is_available()`` is ``False``. Set
            ``pinned_memory=False`` on intentional CPU-only deployments.

    Note:
        Logs at ``WARNING`` level when ``mode="host_register"`` cannot use
        ``torch.cuda.cudart()`` and falls back to torch mode, or when a
        non-torch mode is ignored because ``pinned_memory=False``.
    """
    if not pinned_memory:
        if mode != "torch":
            logger.warning(
                "pinned_memory_mode=%r is ignored because pinned_memory=False; no pinning will be performed.",
                mode,
            )
        return NoOpHostPinner()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "pinned_memory=True requires a CUDA host but torch.cuda.is_available() "
            "is False. Set pinned_memory=False on CPU-only hosts."
        )

    if mode == "host_register":
        if is_available():
            return HostPinner(HostPinRegistry())
        reason, cause = _host_register_unavailability_reason()
        logger.warning(
            "pinned_memory_mode='host_register' requested but the CUDA runtime is not usable "
            "(%s); falling back to 'torch'.",
            reason or "no specific cause detected by re-probe",
            exc_info=cause,
        )

    return HostPinner()
