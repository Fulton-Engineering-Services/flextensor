# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the manual cudaHostRegister pinning path."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from flextensor import host_pinning
from flextensor.host_pinning import HostPinner, HostPinRegistry, NoOpHostPinner, make_host_pinner


@dataclass
class _FakeCudaError:
    """Stand-in for the ``cudaError`` enum returned by torch.cuda.cudart()."""

    value: int


class _FakeCudart:
    """In-memory stand-in for torch.cuda.cudart() exposing the two calls we use."""

    def __init__(self, fail_on_register: bool = False, fail_on_unregister: bool = False):
        self.fail_on_register = fail_on_register
        self.fail_on_unregister = fail_on_unregister
        self.registered: set[int] = set()
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []

    def cudaHostRegister(self, ptr, size, flags):  # noqa: N802 — mirrors the CUDA API name
        self.register_calls.append((int(ptr), int(size), int(flags)))
        if self.fail_on_register:
            return _FakeCudaError(1)
        self.registered.add(int(ptr))
        return _FakeCudaError(0)

    def cudaHostUnregister(self, ptr):  # noqa: N802 — mirrors the CUDA API name
        self.unregister_calls.append(int(ptr))
        if self.fail_on_unregister:
            return _FakeCudaError(1)
        self.registered.discard(int(ptr))
        return _FakeCudaError(0)


@pytest.fixture
def fake_cudart(monkeypatch):
    """Install a fake cudart stub so tests don't need a real CUDA runtime.

    Also pretends ``torch.cuda.is_available()`` is True — the fixture's
    semantic intent is "CUDA host with usable cudart", and ``make_host_pinner``
    now gates ``pinned_memory=True`` on ``is_available()``.
    """
    fake = _FakeCudart()
    monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return fake


class TestHostPinRegistry:
    """Behavior of :class:`HostPinRegistry`."""

    def test_pin_in_place_registers_ptr_and_size(self, fake_cudart):
        registry = HostPinRegistry()
        tensor = torch.zeros(64, dtype=torch.uint8)

        registry.pin_in_place(tensor)

        assert len(registry) == 1
        assert len(fake_cudart.register_calls) == 1
        ptr, size, flags = fake_cudart.register_calls[0]
        assert ptr == tensor.untyped_storage().data_ptr()
        assert size == tensor.untyped_storage().nbytes()
        # cudaHostRegisterPortable=1 — registration must be visible across
        # CUDA contexts (multi-context inference, vLLM workers). A future
        # refactor to cudaHostRegisterDefault=0 would silently break those
        # deployments; pin the documented invariant from
        # ``host_pinning.py``'s ``_CUDA_HOST_REGISTER_PORTABLE`` comment.
        assert flags == host_pinning._CUDA_HOST_REGISTER_PORTABLE == 1
        assert registry.is_registered(ptr)

    def test_pin_in_place_is_idempotent_for_same_storage(self, fake_cudart):
        registry = HostPinRegistry()
        tensor = torch.zeros(32, dtype=torch.uint8)

        registry.pin_in_place(tensor)
        registry.pin_in_place(tensor)

        assert len(fake_cudart.register_calls) == 1
        assert len(registry) == 1

    def test_pin_in_place_dedups_views_of_same_storage(self, fake_cudart):
        """Dedup contract: the registry is keyed on ``storage.data_ptr()``,
        not on ``id(tensor)``. A tensor and a narrower view of the same
        storage must collapse to a single ``cudaHostRegister`` call, and
        the first pin's ``nbytes`` (the full storage size) must stay
        registered — re-pinning a view does not re-register with the
        view's smaller byte range."""
        registry = HostPinRegistry()
        base = torch.zeros(64, dtype=torch.uint8)
        view = base[:16]

        assert id(base) != id(view)
        assert base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()

        registry.pin_in_place(base)
        first_nbytes = base.untyped_storage().nbytes()

        registry.pin_in_place(view)

        ptr = base.untyped_storage().data_ptr()
        assert len(fake_cudart.register_calls) == 1, (
            "view sharing storage with an already-pinned tensor must not re-register"
        )
        assert fake_cudart.register_calls[0][0] == ptr
        assert fake_cudart.register_calls[0][1] == first_nbytes
        assert len(registry) == 1
        assert registry.is_registered(ptr)

    def test_pin_in_place_skips_non_cpu_tensor(self, fake_cudart):
        registry = HostPinRegistry()
        cuda_tensor = MagicMock(spec=torch.Tensor)
        cuda_tensor.device = torch.device("cuda:0")

        registry.pin_in_place(cuda_tensor)

        assert fake_cudart.register_calls == []
        assert len(registry) == 0

    def test_pin_in_place_skips_empty_tensor(self, fake_cudart):
        registry = HostPinRegistry()
        empty = torch.empty(0, dtype=torch.uint8)

        registry.pin_in_place(empty)

        assert fake_cudart.register_calls == []

    def test_pin_in_place_skips_null_storage_pointer(self, fake_cudart):
        """Tensors with a null storage pointer (``data_ptr() == 0``) must
        skip ``cudaHostRegister`` — calling it with a NULL pointer surfaces
        as ``cudaErrorInvalidValue`` and would crash the loader.

        Reachable in production via :meth:`RawBlockController.combine_tensors`
        (``loaders.py:561``), which deliberately replaces consumed tensors'
        storage with ``torch.empty(0, device='cpu', ...)`` — those tensors
        then flow back into the pinning path. The skip-on-null guard at
        ``host_pinning.py``'s :meth:`HostPinRegistry.pin_in_place` is
        load-bearing for that interaction; this test pins it.
        """
        registry = HostPinRegistry()
        # torch.empty(0) is the canonical zero-storage tensor: numel()==0
        # is caught by the empty-tensor guard, but the storage pointer is
        # also 0. Force-test the null-ptr branch independently by using a
        # mock with non-zero numel but zero data_ptr.
        null_ptr_tensor = MagicMock(spec=torch.Tensor)
        null_ptr_tensor.device = torch.device("cpu")
        null_ptr_tensor.is_meta = False
        null_ptr_tensor.numel.return_value = 64
        null_ptr_tensor.element_size.return_value = 1
        null_ptr_tensor.is_pinned.return_value = False
        storage = MagicMock()
        storage.data_ptr.return_value = 0
        storage.nbytes.return_value = 64
        null_ptr_tensor.untyped_storage.return_value = storage

        registry.pin_in_place(null_ptr_tensor)

        # Critical: cudaHostRegister(0, ...) must NOT have been issued.
        assert fake_cudart.register_calls == [], (
            "cudaHostRegister was called with a null storage pointer — the pin_in_place null-ptr guard regressed"
        )
        # And no entry should have been recorded under ptr=0.
        assert len(registry) == 0

    def test_pin_in_place_raises_when_cudart_unavailable(self, monkeypatch):
        """When the probe fails, ``pin_in_place`` raises a ``RuntimeError``
        that names the actionable remediation (``pinned_memory_mode='torch'``).
        The exact reason text is appended via
        :func:`_host_register_unavailability_reason` and varies by host
        (CPU-only vs broken-cudart), so we anchor on stable substrings.
        """
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)
        registry = HostPinRegistry()
        tensor = torch.zeros(8, dtype=torch.uint8)

        with pytest.raises(RuntimeError, match=r"CUDA runtime is not usable"):
            registry.pin_in_place(tensor)

    def test_pin_in_place_chains_underlying_cudart_exception(self, monkeypatch):
        """When ``torch.cuda.cudart()`` itself raises, the ``RuntimeError``
        from :meth:`pin_in_place` must be **chained** (``raise ... from
        exc``) so the operator sees the original traceback in the
        triage log, not just our re-stringified message. This is the
        diagnostic upgrade that motivated dropping the negative cache.
        """
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        original = OSError("simulated libcudart.so load failure")

        def boom():
            raise original

        monkeypatch.setattr(torch.cuda, "cudart", boom)

        registry = HostPinRegistry()
        tensor = torch.zeros(8, dtype=torch.uint8)

        with pytest.raises(RuntimeError) as excinfo:
            registry.pin_in_place(tensor)

        # The reason string must surface the captured exception's repr,
        # AND ``__cause__`` must be the original exception object so
        # downstream loggers / debuggers can walk the chain.
        assert "torch.cuda.cudart() raised" in str(excinfo.value)
        assert "simulated libcudart.so load failure" in str(excinfo.value)
        assert excinfo.value.__cause__ is not None, (
            "pin_in_place must chain the cudart exception via 'raise ... "
            "from cause' so the original traceback survives in tracebacks"
        )
        assert isinstance(excinfo.value.__cause__, OSError)
        assert "libcudart" in str(excinfo.value.__cause__)

    def test_pin_in_place_raises_on_register_failure(self, monkeypatch):
        fake = _FakeCudart(fail_on_register=True)
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: fake)
        registry = HostPinRegistry()
        tensor = torch.zeros(8, dtype=torch.uint8)

        with pytest.raises(RuntimeError, match="cudaHostRegister"):
            registry.pin_in_place(tensor)

        # A failed registration must not leave a tracking entry behind.
        assert len(registry) == 0

    def test_pin_in_place_propagates_when_binding_raises(self, monkeypatch):
        """``cudaHostRegister`` *raising* (vs. returning non-zero rc) must
        propagate cleanly with no half-state. ``pin_in_place`` has no
        try/except around the binding call, so a binding-level exception
        (e.g. PyTorch surfacing a driver error as ``RuntimeError`` instead
        of an rc) reaches the caller. This is the right behaviour — but
        the registry must not have already added the entry under the lock,
        which would silently retain a backing tensor for a never-pinned
        storage and confuse ``release_all`` later.
        """

        class _RaisingCudart:
            def __init__(self) -> None:
                self.register_calls: list[int] = []

            def cudaHostRegister(self, ptr, _size, _flags):  # noqa: N802
                self.register_calls.append(int(ptr))
                raise RuntimeError("simulated driver error from cudaHostRegister binding")

            def cudaHostUnregister(self, ptr):  # noqa: N802
                return _FakeCudaError(0)

        raising = _RaisingCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: raising)
        registry = HostPinRegistry()
        tensor = torch.zeros(8, dtype=torch.uint8)

        with pytest.raises(RuntimeError, match="simulated driver error"):
            registry.pin_in_place(tensor)

        assert len(raising.register_calls) == 1, "test precondition: binding was called once"
        assert len(registry) == 0, (
            "pin_in_place must not record an entry when the binding raised — "
            "a stale entry would retain the backing tensor for an unpinned "
            "storage and confuse release_all"
        )
        assert not registry.is_registered(tensor.untyped_storage().data_ptr())

    def test_release_all_unregisters_every_tracked_pointer(self, fake_cudart):
        registry = HostPinRegistry()
        tensors = [torch.zeros(n, dtype=torch.uint8) for n in (16, 32, 48)]
        expected_ptrs = []
        for t in tensors:
            registry.pin_in_place(t)
            expected_ptrs.append(t.untyped_storage().data_ptr())

        registry.release_all()

        assert len(registry) == 0
        assert sorted(fake_cudart.unregister_calls) == sorted(expected_ptrs)
        # release_all should be idempotent.
        fake_cudart.unregister_calls.clear()
        registry.release_all()
        assert fake_cudart.unregister_calls == []

    def test_release_single_ptr(self, fake_cudart):
        registry = HostPinRegistry()
        tensor = torch.zeros(16, dtype=torch.uint8)
        registry.pin_in_place(tensor)
        ptr = tensor.untyped_storage().data_ptr()

        assert registry.release(ptr) is True
        assert registry.release(ptr) is False
        assert ptr in fake_cudart.unregister_calls

    def test_release_logs_debug_when_pointer_not_tracked(self, fake_cudart, caplog):
        """The ``release()`` API collapses three outcomes (untracked /
        cudart-gone / rc≠0) into a single ``False`` return. The cudart-gone
        and rc≠0 branches log at WARNING; the untracked branch logged
        nothing, leaving operators chasing "why didn't release fire?" with
        no diagnostic. Log at DEBUG so it's discoverable when the level is
        raised but doesn't spam release-on-best-effort callers."""
        import logging

        registry = HostPinRegistry()
        unknown_ptr = 0xDEADBEEF

        with caplog.at_level(logging.DEBUG, logger=host_pinning.__name__):
            assert registry.release(unknown_ptr) is False

        # The untracked branch must have produced a DEBUG record naming the ptr,
        # and must NOT have called cudaHostUnregister.
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("not tracked" in r.message for r in debug_records), (
            "release() with an untracked pointer must produce a DEBUG diagnostic"
        )
        assert fake_cudart.unregister_calls == [], "release() must short-circuit before cudart for untracked pointers"

    def test_does_not_swallow_reference_to_backing_tensor(self, fake_cudart):
        """The registry must hold a strong reference to the tensor so the
        storage it registered can't be freed before we unregister it."""
        registry = HostPinRegistry()
        tensor = torch.zeros(16, dtype=torch.uint8)
        ptr = tensor.untyped_storage().data_ptr()
        registry.pin_in_place(tensor)

        del tensor

        assert registry.is_registered(ptr)
        registry.release_all()
        assert ptr in fake_cudart.unregister_calls

    def test_release_returns_false_and_warns_when_cudart_disappears(self, monkeypatch, caplog):
        """If cudart becomes unreachable between pin_in_place and release, the
        method must report failure (False), log a WARNING, and **retain** the
        registry entry — dropping it would let PyTorch's allocator recycle a
        still-page-locked storage."""
        import logging

        fake = _FakeCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: fake)
        registry = HostPinRegistry()
        tensor = torch.zeros(16, dtype=torch.uint8)
        registry.pin_in_place(tensor)
        ptr = tensor.untyped_storage().data_ptr()

        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            result = registry.release(ptr)

        assert result is False
        assert fake.unregister_calls == []  # cudart was None — call must be skipped
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when cudart disappears mid-process"
        assert f"{ptr:#x}" in warnings[0].message
        assert "cudaHostUnregister" in warnings[0].message or "cudart" in warnings[0].message
        # Entry must be retained so the backing tensor's storage stays alive.
        assert registry.is_registered(ptr)

    def test_release_returns_false_and_warns_on_unregister_failure(self, monkeypatch, caplog):
        """A non-zero ``cudaHostUnregister`` rc must surface as a WARNING (not
        DEBUG) and the function must report failure. The entry must be
        retained so the still-page-locked storage isn't recycled."""
        import logging

        register_only = _FakeCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: register_only)
        registry = HostPinRegistry()
        tensor = torch.zeros(16, dtype=torch.uint8)
        registry.pin_in_place(tensor)
        ptr = tensor.untyped_storage().data_ptr()

        failing = _FakeCudart(fail_on_unregister=True)
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: failing)
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            result = registry.release(ptr)

        assert result is False
        assert failing.unregister_calls == [ptr]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when cudaHostUnregister returns non-zero"
        assert f"{ptr:#x}" in warnings[0].message
        assert "cudaHostUnregister" in warnings[0].message
        # Entry must be retained so the kernel-pinned storage stays alive.
        assert registry.is_registered(ptr)

    def test_release_all_warns_when_cudart_disappears(self, monkeypatch, caplog):
        """release_all must log at WARNING when cudart is gone and **retain**
        every entry — dropping them would let PyTorch's allocator recycle
        still-page-locked storages."""
        import logging

        fake = _FakeCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: fake)
        registry = HostPinRegistry()
        tensors = [torch.zeros(n, dtype=torch.uint8) for n in (16, 32, 48)]
        for t in tensors:
            registry.pin_in_place(t)
        expected_ptrs = [t.untyped_storage().data_ptr() for t in tensors]

        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            registry.release_all()

        # All entries must be retained so the kernel-pinned storages stay alive.
        assert len(registry) == 3
        for ptr in expected_ptrs:
            assert registry.is_registered(ptr)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when cudart is unavailable at release_all"
        # Mention the retained count so operators know the scale of the leak.
        assert "3" in warnings[0].message

    def test_release_all_warns_on_unregister_failure(self, monkeypatch, caplog):
        """Per-pointer unregister failures during release_all must log at
        WARNING and retain the failing entries; successful entries are
        removed from the registry."""
        import logging

        ok = _FakeCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: ok)
        registry = HostPinRegistry()
        tensors = [torch.zeros(n, dtype=torch.uint8) for n in (16, 32)]
        for t in tensors:
            registry.pin_in_place(t)
        expected_ptrs = [t.untyped_storage().data_ptr() for t in tensors]

        failing = _FakeCudart(fail_on_unregister=True)
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: failing)
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            registry.release_all()

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == len(expected_ptrs), f"expected one WARNING per failing unregister; got {len(warnings)}"
        warning_text = " ".join(r.message for r in warnings)
        for ptr in expected_ptrs:
            assert f"{ptr:#x}" in warning_text
        # All failing entries must be retained.
        assert len(registry) == len(expected_ptrs)
        for ptr in expected_ptrs:
            assert registry.is_registered(ptr)

    def test_release_returns_false_and_warns_when_unregister_raises(self, monkeypatch, caplog):
        """If ``cudart.cudaHostUnregister`` itself raises (RuntimeError /
        OSError / AttributeError — the last is the typical shape when a
        test rig swaps cudart out partway through teardown), :meth:`release`
        must catch the exception, log a WARNING with ``exc_info`` so the
        operator can identify the cause, retain the entry so the still-
        page-locked storage isn't recycled, and return ``False``."""
        import logging

        ok = _FakeCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: ok)
        registry = HostPinRegistry()
        tensor = torch.zeros(16, dtype=torch.uint8)
        registry.pin_in_place(tensor)
        ptr = tensor.untyped_storage().data_ptr()

        class _RaisingCudart:
            def cudaHostUnregister(self, ptr):  # noqa: N802
                raise OSError("simulated cudart shared-library teardown")

        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: _RaisingCudart())
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            result = registry.release(ptr)

        assert result is False
        assert registry.is_registered(ptr), "entry must be retained when unregister raises"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when cudaHostUnregister raises"
        assert "raised" in warnings[0].message
        assert "OSError" in warnings[0].message
        assert f"{ptr:#x}" in warnings[0].message
        assert warnings[0].exc_info is not None, "exc_info must be attached for triage"

    def test_release_all_continues_past_raising_entries(self, monkeypatch, caplog):
        """If one ``cudaHostUnregister`` call raises mid-loop, :meth:`release_all`
        must catch it per-entry, retain that entry, and keep going so
        subsequent pointers are still attempted. Without this, a single
        flaky cudart call would silently leak every remaining pin and —
        combined with ``TensorManager.shutdown()`` — mask the loader-
        teardown exception that triggered the cleanup in the first place.
        """
        import logging

        ok = _FakeCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: ok)
        registry = HostPinRegistry()
        tensors = [torch.zeros(n, dtype=torch.uint8) for n in (16, 32, 48)]
        for t in tensors:
            registry.pin_in_place(t)
        ptr_a, ptr_b, ptr_c = (t.untyped_storage().data_ptr() for t in tensors)

        class _RaiseOnMiddle:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def cudaHostUnregister(self, ptr):  # noqa: N802
                self.calls.append(int(ptr))
                if int(ptr) == ptr_b:
                    raise RuntimeError("simulated transient cudart failure")
                return _FakeCudaError(0)

        flaky = _RaiseOnMiddle()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: flaky)
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            registry.release_all()

        # The raising call must not have aborted the loop — every ptr must
        # have been attempted exactly once.
        assert sorted(flaky.calls) == sorted([ptr_a, ptr_b, ptr_c])
        # The two non-raising entries must be released; only the raising
        # one must be retained.
        assert not registry.is_registered(ptr_a)
        assert registry.is_registered(ptr_b)
        assert not registry.is_registered(ptr_c)
        assert len(registry) == 1

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING for the raising entry"
        assert any("RuntimeError" in r.message for r in warnings)
        assert any(f"{ptr_b:#x}" in r.message for r in warnings)

    def test_release_all_retains_only_failed_entries(self, monkeypatch, caplog):
        """Mixed success/failure at release_all: succeeding entries must be
        dropped and failing ones retained, so kernel-pinned storages stay
        alive but cleanly-released ones don't waste host memory.

        Internal-contract test: writes directly to ``registry._entries``
        because the public API (``pin_in_place``) keys on the allocator's
        actual storage pointers, which can't be controlled to make the
        fake cudart selectively fail by pointer. The deliberate access to
        the private dict is the cheapest way to drive the
        succeed-one / fail-one branch without coupling to allocation
        addresses; it does mean a future rename of ``_entries`` would
        require touching this test.
        """
        import logging

        good_ptr = 0xAAA0
        bad_ptr = 0xBBB0
        good_backing = torch.zeros(8, dtype=torch.uint8)
        bad_backing = torch.zeros(8, dtype=torch.uint8)

        registry = HostPinRegistry()
        registry._entries[good_ptr] = (8, good_backing)
        registry._entries[bad_ptr] = (8, bad_backing)

        class _SelectiveFail:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def cudaHostUnregister(self, ptr):  # noqa: N802
                self.calls.append(int(ptr))
                return _FakeCudaError(1 if int(ptr) == bad_ptr else 0)

        selective = _SelectiveFail()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: selective)
        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            registry.release_all()

        assert sorted(selective.calls) == sorted([good_ptr, bad_ptr])
        assert not registry.is_registered(good_ptr)
        assert registry.is_registered(bad_ptr)
        assert len(registry) == 1
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert f"{bad_ptr:#x}" in warnings[0].message

    def test_pin_in_place_skips_already_pinned_tensor(self, fake_cudart):
        """A tensor already pinned by PyTorch's caching pinned allocator
        must short-circuit pin_in_place — passing one to cudaHostRegister
        would fail with cudaErrorHostMemoryAlreadyRegistered (712) and
        surface as a confusing RuntimeError."""
        registry = HostPinRegistry()
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = torch.device("cpu")
        tensor.is_meta = False
        tensor.numel.return_value = 16
        tensor.is_pinned.return_value = True

        result = registry.pin_in_place(tensor)

        assert result is tensor
        assert fake_cudart.register_calls == []
        assert len(registry) == 0

    def test_release_serializes_concurrent_calls_for_same_pointer(self, monkeypatch):
        """Two threads racing on ``release(same_ptr)`` must not both invoke
        ``cudaHostUnregister`` — the loser must see the membership check
        fail and return ``False`` cleanly. Without the lock held across the
        syscall, both threads would pass the membership check and the loser
        would call cudart with a pointer the kernel no longer tracks,
        producing a spurious ``cudaErrorHostMemoryNotRegistered`` warning.
        """
        import threading
        import time

        in_unregister = threading.Event()
        proceed_unregister = threading.Event()

        class _BlockingCudart:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def cudaHostRegister(self, ptr, _size, _flags):  # noqa: N802
                return _FakeCudaError(0)

            def cudaHostUnregister(self, ptr):  # noqa: N802
                self.calls.append(int(ptr))
                in_unregister.set()
                # Block inside the syscall so the test can confirm thread B
                # cannot enter cudaHostUnregister while thread A is here.
                proceed_unregister.wait(timeout=2.0)
                return _FakeCudaError(0)

        cudart = _BlockingCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: cudart)
        registry = HostPinRegistry()
        # Inject a single entry so both threads target the same pointer.
        ptr = 0xDEADBEEF
        registry._entries[ptr] = (16, torch.zeros(16, dtype=torch.uint8))

        results: dict[str, bool] = {}

        def _release(name: str) -> None:
            results[name] = registry.release(ptr)

        thread_a = threading.Thread(target=_release, args=("a",))
        thread_b = threading.Thread(target=_release, args=("b",))

        thread_a.start()
        assert in_unregister.wait(timeout=2.0), "thread A must reach cudaHostUnregister"

        thread_b.start()
        # Give thread B a chance to attempt — it must be blocked on the lock,
        # not progressing into cudaHostUnregister.
        time.sleep(0.05)
        assert len(cudart.calls) == 1, (
            "thread B must not have entered cudaHostUnregister while A still holds the registry lock"
        )

        proceed_unregister.set()
        thread_a.join(timeout=2.0)
        thread_b.join(timeout=2.0)

        # Exactly one cudaHostUnregister call total — A's. B saw the entry
        # already removed and returned False without calling the kernel.
        assert len(cudart.calls) == 1
        assert sum(1 for ok in results.values() if ok) == 1, "exactly one thread must report success"
        assert sum(1 for ok in results.values() if not ok) == 1, (
            "exactly one thread must report False (lost the race cleanly)"
        )
        assert not registry.is_registered(ptr)

    def test_pin_in_place_serializes_concurrent_calls_for_same_pointer(self, monkeypatch):
        """Symmetric to ``test_release_serializes_concurrent_calls_for_same_pointer``
        for the register side. Two threads racing on
        ``pin_in_place(tensors_with_same_storage)`` must result in exactly
        one ``cudaHostRegister`` call — without the lock held across the
        membership check + kernel call, both threads would pass the
        ``ptr not in self._entries`` check and double-register the same
        storage, producing a spurious ``cudaErrorHostMemoryAlreadyRegistered``.

        We have a guard test for the release side (the riskier op because
        double-unregister can free pages still in use by the kernel), but
        the symmetric register-side guard was missing — pinning the lock
        contract here prevents a future "narrow the lock to just the dict
        write" refactor from regressing it.
        """
        import threading
        import time

        in_register = threading.Event()
        proceed_register = threading.Event()

        class _BlockingCudart:
            def __init__(self) -> None:
                self.register_calls: list[int] = []

            def cudaHostRegister(self, ptr, _size, _flags):  # noqa: N802
                self.register_calls.append(int(ptr))
                in_register.set()
                # Block inside the syscall so the test can confirm thread B
                # cannot enter cudaHostRegister while thread A is here.
                proceed_register.wait(timeout=2.0)
                return _FakeCudaError(0)

            def cudaHostUnregister(self, ptr):  # noqa: N802
                return _FakeCudaError(0)

        cudart = _BlockingCudart()
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: cudart)
        registry = HostPinRegistry()

        # Both threads pin the same underlying storage (a tensor + a view of it).
        # The dedup-by-data_ptr() contract means exactly one register call
        # should happen even if both threads pass the membership check
        # concurrently — but only if the lock is held across the syscall.
        base = torch.zeros(64, dtype=torch.uint8)
        view = base[16:48]  # shares storage with `base`
        assert base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr(), (
            "test precondition: the view must share storage with base"
        )

        def _pin(t: torch.Tensor) -> None:
            registry.pin_in_place(t)

        thread_a = threading.Thread(target=_pin, args=(base,))
        thread_b = threading.Thread(target=_pin, args=(view,))

        thread_a.start()
        assert in_register.wait(timeout=2.0), "thread A must reach cudaHostRegister"

        thread_b.start()
        # Give thread B a chance to attempt — it must be blocked on the lock,
        # not progressing into cudaHostRegister.
        time.sleep(0.05)
        assert len(cudart.register_calls) == 1, (
            "thread B must not have entered cudaHostRegister while A still holds the registry lock"
        )

        proceed_register.set()
        thread_a.join(timeout=2.0)
        thread_b.join(timeout=2.0)

        # Exactly one cudaHostRegister call total — A's. B saw the entry
        # already present and skipped the kernel call.
        assert len(cudart.register_calls) == 1, (
            "double-register: lock did not serialize concurrent pin_in_place "
            "for the same storage pointer — both threads called cudaHostRegister"
        )
        assert len(registry) == 1
        assert registry.is_registered(base.untyped_storage().data_ptr())


class TestIsAvailable:
    def test_returns_false_when_cudart_missing(self, monkeypatch):
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)
        assert host_pinning.is_available() is False

    def test_returns_true_when_cudart_present(self, monkeypatch):
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: _FakeCudart())
        assert host_pinning.is_available() is True


class TestHostPinner:
    """Behavior of the unified :class:`HostPinner` dispatch."""

    def test_default_construction_has_no_registry(self):
        # Bare HostPinner() = torch dispatch path: no registry attached, so
        # pin() will fall through to tensor.pin_memory().
        pinner = HostPinner()
        assert pinner.registry is None

    def test_construction_with_registry_attaches_it(self):
        # HostPinner(registry) = host_register dispatch path: pin() routes
        # through the supplied registry's cudaHostRegister call.
        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        assert pinner.registry is registry

    def test_pin_torch_mode_calls_pin_memory(self):
        # No real CUDA needed: we mock the tensor's pin_memory.
        pinner = HostPinner()
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = torch.device("cpu")
        tensor.is_meta = False
        tensor.numel.return_value = 16
        tensor.is_pinned.return_value = False
        pinned = MagicMock(spec=torch.Tensor)
        tensor.pin_memory.return_value = pinned

        result = pinner.pin(tensor)

        tensor.pin_memory.assert_called_once_with()
        assert result is pinned

    def test_pin_torch_mode_raises_on_pin_memory_failure(self, monkeypatch):
        """torch-mode :meth:`pin` must raise on ``tensor.pin_memory()``
        failure (e.g. RLIMIT_MEMLOCK exhaustion) so callers that need
        strict semantics — like :class:`BenchmarkReplace`, which would
        otherwise record pageable transfer times into
        ``TensorStatistics.load_time_ms`` — can abort instead of
        contaminating downstream strategy data. Best-effort callers
        should use :meth:`try_pin`."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        pinner = HostPinner()
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = torch.device("cpu")
        tensor.is_meta = False
        tensor.numel.return_value = 16
        tensor.is_pinned.return_value = False
        tensor.pin_memory.side_effect = RuntimeError("CUDA error: out of memory")

        with pytest.raises(RuntimeError, match="out of memory"):
            pinner.pin(tensor)

    def test_pin_host_register_mode_uses_registry_in_place(self, fake_cudart):
        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        tensor = torch.zeros(32, dtype=torch.uint8)

        result = pinner.pin(tensor)

        assert result is tensor  # same object — pinned in place
        assert len(fake_cudart.register_calls) == 1
        assert len(registry) == 1

    def test_pin_skips_non_cpu_tensor(self, fake_cudart):
        pinner = HostPinner(HostPinRegistry())
        cuda_tensor = MagicMock(spec=torch.Tensor)
        cuda_tensor.device = torch.device("cuda:0")
        cuda_tensor.is_meta = False
        cuda_tensor.numel.return_value = 16

        result = pinner.pin(cuda_tensor)

        assert result is cuda_tensor
        assert fake_cudart.register_calls == []

    def test_pin_skips_meta_tensor(self, fake_cudart):
        pinner = HostPinner(HostPinRegistry())
        meta = torch.empty(8, dtype=torch.uint8, device="meta")

        result = pinner.pin(meta)

        assert result is meta
        assert fake_cudart.register_calls == []

    def test_pin_skips_already_pinned_tensor(self):
        # Use a mock so we don't depend on a real pinning path being available.
        pinner = HostPinner()
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = torch.device("cpu")
        tensor.is_meta = False
        tensor.numel.return_value = 16
        tensor.is_pinned.return_value = True

        result = pinner.pin(tensor)

        tensor.pin_memory.assert_not_called()
        assert result is tensor

    def test_release_all_releases_registry(self, fake_cudart):
        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        tensors = [torch.zeros(n, dtype=torch.uint8) for n in (16, 32)]
        for t in tensors:
            pinner.pin(t)
        assert len(registry) == 2

        pinner.release_all()

        assert len(registry) == 0

    def test_release_all_torch_mode_is_noop(self):
        pinner = HostPinner()
        # Should not raise even though there's no registry to release.
        pinner.release_all()

    def test_is_pinned_torch_mode_delegates_to_tensor(self):
        """In torch mode, ``is_pinned`` reflects PyTorch's allocator state."""
        pinner = HostPinner()
        unpinned = MagicMock(spec=torch.Tensor)
        unpinned.is_pinned.return_value = False
        pinned = MagicMock(spec=torch.Tensor)
        pinned.is_pinned.return_value = True

        assert pinner.is_pinned(unpinned) is False
        assert pinner.is_pinned(pinned) is True

    def test_is_pinned_host_register_mode_consults_registry(self, fake_cudart):
        """In host_register mode, ``is_pinned`` returns True for tensors whose
        storage pointer the registry tracks — even though
        ``tensor.is_pinned()`` reports False (cudaHostRegister doesn't update
        PyTorch's pinned-allocator flag)."""
        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        registered = torch.zeros(16, dtype=torch.uint8)
        unregistered = torch.zeros(16, dtype=torch.uint8)
        pinner.pin(registered)

        assert pinner.is_pinned(registered) is True
        assert pinner.is_pinned(unregistered) is False
        assert not registered.is_pinned()  # sanity: torch's flag was never set

    def test_is_pinned_returns_false_when_storage_ptr_unavailable(self):
        """``is_pinned`` must not raise on tensors whose storage can't yield a
        data_ptr (e.g. partially-constructed mocks)."""
        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        broken = MagicMock(spec=torch.Tensor)
        broken.is_pinned.return_value = False
        broken.untyped_storage.side_effect = RuntimeError("no storage")

        assert pinner.is_pinned(broken) is False


class TestMakeHostPinner:
    """Behavior of the :func:`make_host_pinner` factory."""

    def test_pinned_memory_false_returns_noop(self, fake_cudart):
        # pinned_memory=False short-circuits to NoOpHostPinner regardless of
        # the requested mode — the requested mode is recorded in the WARNING
        # log but has no runtime effect.
        pinner = make_host_pinner(pinned_memory=False, mode="host_register")
        assert isinstance(pinner, NoOpHostPinner)
        assert pinner.registry is None

    def test_pinned_memory_false_returns_noop_pinner(self):
        """``pinned_memory=False`` must return a pinner whose ``pin`` is a
        no-op. The previous behavior returned a torch-mode
        :class:`HostPinner` whose ``pin`` still called
        :meth:`torch.Tensor.pin_memory`, silently violating the user's
        opt-out at every call site (e.g. RawBlockController)."""
        pinner = make_host_pinner(pinned_memory=False, mode="torch")
        tensor = MagicMock(spec=torch.Tensor)
        tensor.device = torch.device("cpu")
        tensor.is_meta = False
        tensor.numel.return_value = 16
        tensor.is_pinned.return_value = False

        assert pinner.pin(tensor) is tensor
        tensor.pin_memory.assert_not_called()

    def test_pinned_memory_false_with_non_default_mode_warns(self, fake_cudart, caplog):
        """``pinned_memory=False`` is the master switch — any non-default
        ``mode`` is silently ignored. Surface that at WARNING so an operator
        who explicitly opted into ``host_register`` notices their setting
        was overridden."""
        import logging

        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            make_host_pinner(pinned_memory=False, mode="host_register")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING when pinned_memory=False overrides mode='host_register'"
        msg = warnings[0].message
        assert "host_register" in msg
        assert "pinned_memory=False" in msg

    def test_pinned_memory_false_with_default_mode_does_not_log(self, fake_cudart, caplog):
        """The redundant-config WARNING must not fire on the common default
        (``mode='torch'``) — only when the user explicitly opted into a
        non-default mode and then disabled pinning."""
        import logging

        with caplog.at_level(logging.INFO, logger=host_pinning.__name__):
            make_host_pinner(pinned_memory=False, mode="torch")

        assert not any(r.levelno >= logging.INFO for r in caplog.records)

    def test_torch_mode_returns_bare_host_pinner(self, monkeypatch):
        # Patch cuda.is_available so we don't trip the no-CUDA WARNING on
        # CPU-only CI; this test only cares about the returned pinner.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        pinner = make_host_pinner(pinned_memory=True, mode="torch")
        # No registry → dispatches via tensor.pin_memory() (the torch path).
        assert pinner.registry is None
        assert not isinstance(pinner, NoOpHostPinner)

    def test_host_register_returns_fresh_registry_per_call(self, fake_cudart):
        """``make_host_pinner`` must construct an independent
        :class:`HostPinRegistry` on every call. Sharing a registry across
        managers would allow a child / second manager's ``release_all`` to
        free pointers the first manager still owns — a kernel-level
        use-after-unregister hazard described in the module docstring's
        fork/multiprocessing contract."""
        p1 = make_host_pinner(pinned_memory=True, mode="host_register")
        p2 = make_host_pinner(pinned_memory=True, mode="host_register")

        assert p1.registry is not None and p2.registry is not None
        assert p1.registry is not p2.registry, (
            "make_host_pinner must not return a shared registry — each call owns its own pin lifecycle"
        )

    def test_torch_mode_raises_when_cuda_unavailable(self, monkeypatch):
        """``pinned_memory=True`` on a host without CUDA is a misconfiguration:
        offloading has no purpose without a GPU, and silently degrading to
        pageable transfers would mask the misconfiguration as a perf
        regression. The factory raises so the operator must explicitly opt
        out via ``pinned_memory=False``."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="pinned_memory=True requires a CUDA host"):
            make_host_pinner(pinned_memory=True, mode="torch")

    def test_torch_mode_does_not_warn_when_cuda_available(self, monkeypatch, caplog):
        """The no-CUDA WARNING must not fire on a CUDA-enabled host —
        otherwise every production startup would emit a spurious warning."""
        import logging

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            make_host_pinner(pinned_memory=True, mode="torch")

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_host_register_mode_when_available(self, fake_cudart):
        # cudart usable → host_register path: HostPinner with a fresh registry.
        pinner = make_host_pinner(pinned_memory=True, mode="host_register")
        assert isinstance(pinner.registry, HostPinRegistry)
        assert not isinstance(pinner, NoOpHostPinner)

    def test_host_register_on_cpu_only_host_raises(self, monkeypatch):
        """``mode='host_register'`` on a CPU-only host hits the same no-CUDA
        gate as ``mode='torch'`` — neither can pin without a GPU. The factory
        raises rather than silently degrading."""
        monkeypatch.setattr(host_pinning, "_probe_cudart", lambda: None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="pinned_memory=True requires a CUDA host"):
            make_host_pinner(pinned_memory=True, mode="host_register")

    def test_host_register_fallback_reports_cudart_raised(self, monkeypatch, caplog):
        """When ``torch.cuda.cudart()`` raises (e.g. broken CUDA build), the
        fallback warning must surface the concrete exception **and attach the
        traceback via ``exc_info``**, not just a generic "libcudart could not
        be loaded" message.

        Drives the real :func:`_probe_cudart` / :func:`_host_register_unavailability_reason`
        path end-to-end so the test exercises the actual probe-failure capture
        ``make_host_pinner`` consumes.
        """
        import logging

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom():
            raise RuntimeError("simulated cudart import failure")

        monkeypatch.setattr(torch.cuda, "cudart", boom)

        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            pinner = make_host_pinner(pinned_memory=True, mode="host_register")

        # Fell back to torch path: bare HostPinner, no registry, not no-op.
        assert pinner.registry is None
        assert not isinstance(pinner, NoOpHostPinner)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        msg = warnings[0].message
        assert "torch.cuda.cudart() raised" in msg
        assert "simulated cudart import failure" in msg
        # The reviewer's ask: the underlying traceback must be attached so
        # operators see *which* call raised, not just our re-stringified
        # repr. exc_info is a (type, value, tb) tuple when set; bool checks
        # via ``warnings[0].exc_info is not None`` were ambiguous because
        # the formatter coerces it.
        assert warnings[0].exc_info is not None, (
            "make_host_pinner must attach exc_info=cause so the cudart traceback reaches the operator"
        )
        exc_type, exc_val, _tb = warnings[0].exc_info
        assert exc_type is RuntimeError
        assert "simulated cudart import failure" in str(exc_val)

    def test_host_register_fallback_reports_missing_symbols(self, monkeypatch, caplog):
        """When ``torch.cuda.cudart()`` returns an object lacking the required
        symbols (older PyTorch build), the fallback warning must say so —
        not blame "libcudart could not be loaded". This branch has no
        underlying exception, so ``exc_info`` must be falsy.
        """
        import logging

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "cudart", lambda: object())  # no register/unregister

        with caplog.at_level(logging.WARNING, logger=host_pinning.__name__):
            pinner = make_host_pinner(pinned_memory=True, mode="host_register")

        # Fell back to torch path: bare HostPinner, no registry, not no-op.
        assert pinner.registry is None
        assert not isinstance(pinner, NoOpHostPinner)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert "lacks cudaHostRegister/cudaHostUnregister" in warnings[0].message
        # No underlying exception in the missing-symbols branch — exc_info
        # must NOT be attached, otherwise the formatter would print a stale
        # / misleading traceback for an unrelated frame.
        assert warnings[0].exc_info is None, (
            "missing-symbols branch has no underlying exception; exc_info must not be attached"
        )

    def test_invalid_mode_raises(self):
        """An invalid ``mode`` is rejected at the function boundary by
        @beartype's runtime Literal check; the violation message must name
        the offending value and the allowed members so the caller can fix
        their config without reading the source."""
        from beartype.roar import BeartypeCallHintParamViolation

        with pytest.raises(BeartypeCallHintParamViolation) as excinfo:
            make_host_pinner(pinned_memory=True, mode="not_a_mode")  # type: ignore[arg-type]
        msg = str(excinfo.value)
        assert "not_a_mode" in msg
        assert "torch" in msg and "host_register" in msg


class TestProbeCudart:
    """Behaviour of :func:`_probe_cudart`.

    The probe is intentionally **stateless** — there is no module-level
    cache. The previous design memoized negative results for the lifetime
    of the process, which silently disabled host_register pinning forever
    on a transient init-order or driver hiccup. ``test_reprobes_each_call``
    pins the no-cache contract so a future "let's just memoize this" PR
    can't quietly reintroduce that foot-gun.
    """

    def test_returns_none_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert host_pinning._probe_cudart() is None

    def test_returns_none_when_cudart_raises(self, monkeypatch):
        """Each documented exception class from ``torch.cuda.cudart()`` must
        be swallowed silently so callers can fall back without a try/except.
        Anything outside the documented set must propagate.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        for exc_cls in (RuntimeError, OSError, AttributeError, ImportError):

            def boom(_e=exc_cls):
                raise _e("simulated probe failure")

            monkeypatch.setattr(torch.cuda, "cudart", boom)
            assert host_pinning._probe_cudart() is None, (
                f"_probe_cudart must swallow {exc_cls.__name__} from torch.cuda.cudart()"
            )

    def test_returns_none_when_cudart_missing_symbols(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "cudart", lambda: object())
        assert host_pinning._probe_cudart() is None

    def test_unexpected_exception_propagates(self, monkeypatch):
        """The narrowed ``except`` lets unexpected exception classes escape.

        Catching ``Exception`` previously masked programmer-error exceptions
        (e.g. a ``TypeError`` from a refactor) as a silent torch-mode
        fallback. Pin the narrow catch contract so a future "let's catch
        Exception again, just in case" change is rejected.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def boom():
            raise TypeError("unexpected — should propagate, not be swallowed")

        monkeypatch.setattr(torch.cuda, "cudart", boom)
        with pytest.raises(TypeError, match="should propagate"):
            host_pinning._probe_cudart()

    def test_reprobes_each_call(self, monkeypatch):
        """``_probe_cudart`` must hit ``torch.cuda.cudart()`` on every call —
        no module-level negative cache. A previous design cached failures
        for the process lifetime, so a transient init-order glitch
        permanently disabled host_register pinning. PyTorch already
        memoizes ``cudart()`` itself, so re-probing is cheap; this test
        guards the "no own cache" contract.
        """
        call_count = 0

        def counting_cudart():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("simulated transient cudart failure")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "cudart", counting_cudart)

        assert host_pinning._probe_cudart() is None
        assert host_pinning._probe_cudart() is None
        assert host_pinning._probe_cudart() is None
        assert call_count == 3, (
            "expected three probes (one per call); a value <3 means a negative-result cache crept back in"
        )

    def test_reprobing_recovers_from_transient_failure(self, monkeypatch):
        """Concretizes the *reason* re-probing matters: an early-call
        ``RuntimeError`` (e.g. CUDA still initialising) followed by a
        successful probe must resolve to host_register pinning, not the
        previous "permanently torch-mode" trap.
        """
        attempts: list[int] = []

        class _OkCudart:
            def cudaHostRegister(self, ptr, size, flags):  # noqa: N802
                pass

            def cudaHostUnregister(self, ptr):  # noqa: N802
                pass

        def flaky_cudart():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("CUDA not ready yet")
            return _OkCudart()

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "cudart", flaky_cudart)

        assert host_pinning._probe_cudart() is None  # transient failure
        result = host_pinning._probe_cudart()  # next probe succeeds
        assert isinstance(result, _OkCudart), (
            "after a transient cudart failure, the next probe must be "
            "able to succeed — caching a negative result would have made "
            "this branch unreachable"
        )


class TestHostRegisterRealCuda:
    """Smoke-test the real ``cudaHostRegister`` / ``cudaHostUnregister`` round-trip.

    Every other test in this file mocks :func:`torch.cuda.cudart` via the
    ``fake_cudart`` fixture. Mocks validate our return-code handling but
    not the wiring to the real driver — they cannot catch
    :func:`torch.cuda.cudart` API drift between PyTorch versions, the
    returned ``cudaError`` enum's ``.value`` semantics changing, or
    real-driver edge cases like ``cudaErrorHostMemoryAlreadyRegistered``.

    These tests run the same registry path against the real cudart binding
    when CUDA is available, providing a canary for those gaps. Skipped on
    CPU-only hosts (the common CI path) so the cost is zero when the test
    cannot meaningfully run.
    """

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs a real CUDA host to call cudaHostRegister",
    )
    def test_pinner_round_trip_register_and_release_all(self):
        """``HostPinner.pin`` followed by ``release_all`` round-trips against real cudart."""
        assert host_pinning.is_available(), (
            "CUDA is available but torch.cuda.cudart() is not exposing "
            "cudaHostRegister/cudaHostUnregister — pinned_memory_mode="
            "'host_register' wiring is broken on this PyTorch build"
        )

        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        tensor = torch.zeros(1024, dtype=torch.uint8)

        assert not tensor.is_pinned(), "test precondition: starts unpinned"

        returned = pinner.pin(tensor)

        assert returned is tensor, "host_register mode must return the same tensor object — no copy"
        assert pinner.is_pinned(tensor), (
            "after pin(), HostPinner.is_pinned must report True via the "
            "registry — tensor.is_pinned() returns False for "
            "cudaHostRegister-pinned storage because PyTorch's caching "
            "allocator did not allocate it"
        )
        assert len(registry) == 1
        assert registry.is_registered(tensor.untyped_storage().data_ptr())

        registry.release_all()

        assert len(registry) == 0
        assert not registry.is_registered(tensor.untyped_storage().data_ptr())

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs a real CUDA host to call cudaHostRegister",
    )
    def test_release_unregisters_individual_entry(self):
        """``release(ptr)`` removes one entry without disturbing others."""
        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        t1 = torch.zeros(64, dtype=torch.uint8)
        t2 = torch.zeros(64, dtype=torch.uint8)

        pinner.pin(t1)
        pinner.pin(t2)
        assert len(registry) == 2

        ptr1 = t1.untyped_storage().data_ptr()
        ptr2 = t2.untyped_storage().data_ptr()

        try:
            assert registry.release(ptr1) is True
            assert not registry.is_registered(ptr1)
            assert registry.is_registered(ptr2), "releasing one ptr must not disturb others"
            assert len(registry) == 1
        finally:
            registry.release_all()
        assert len(registry) == 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="needs a real CUDA host to call cudaHostRegister",
    )
    def test_make_host_pinner_returns_registry_backed_pinner(self):
        """The factory-built pinner uses the real registry path on a CUDA host.

        Locks in the contract that ``pinned_memory=True`` +
        ``mode="host_register"`` returns a registry-backed
        :class:`HostPinner` rather than silently falling back to torch
        mode (which would happen if :func:`is_available` returned False).
        """
        pinner = make_host_pinner(pinned_memory=True, mode="host_register")
        try:
            assert pinner.registry is not None, (
                "host_register mode on a real CUDA host must return a "
                "registry-backed pinner; a None registry means the cudart "
                "probe failed and we silently fell back to torch mode"
            )
            tensor = torch.zeros(64, dtype=torch.uint8)
            pinner.pin(tensor)
            assert len(pinner.registry) == 1
        finally:
            pinner.release_all()
        assert len(pinner.registry) == 0
