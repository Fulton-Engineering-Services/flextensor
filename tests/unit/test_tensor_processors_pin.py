# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :class:`MoveToPinMemoryTensorProcessor`.

These tests use a spy ``HostPinner`` so they run on any platform — no CUDA
runtime required. They lock down the processor's documented contract:

- Skips non-CPU and meta tensors.
- Caches by ``id(source)`` so re-processing the same tensor reuses the result
  and only invokes the pinner once.
- Returns whatever the pinner returns (a fresh tensor in torch mode; the same
  tensor in host_register mode).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from flextensor.host_pinning import HostPinner, HostPinRegistry
from flextensor.tensor_processors import MoveToPinMemoryTensorProcessor


class _SpyPinner(HostPinner):
    """HostPinner stand-in that records every ``pin`` call.

    Two flavours, controlled by ``in_place``:

    - ``in_place=False`` (torch mode): returns a fresh tensor for each call,
      mimicking ``tensor.pin_memory()`` which allocates a new pinned buffer.
    - ``in_place=True`` (host_register mode): returns the same tensor,
      mimicking ``cudaHostRegister`` which pins the existing storage.
    """

    def __init__(self, *, in_place: bool) -> None:
        super().__init__(registry=None)
        self.in_place = in_place
        self.calls: list[torch.Tensor] = []

    def pin(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls.append(tensor)
        return tensor if self.in_place else tensor.clone()


class TestSkipsTensorsThatCannotBePinned:
    def test_meta_tensor_returned_unchanged(self):
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        meta_tensor = torch.empty(4, device="meta")

        result = processor.process(meta_tensor)

        assert result is meta_tensor
        assert pinner.calls == [], "Pinner must not be invoked on meta tensors"

    def test_non_tensor_returned_unchanged(self):
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)

        result = processor.process("not a tensor")

        assert result == "not a tensor"
        assert pinner.calls == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to construct a non-CPU tensor")
    def test_non_cpu_tensor_returned_unchanged(self):
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        cuda_tensor = torch.zeros(4, device="cuda")

        result = processor.process(cuda_tensor)

        assert result is cuda_tensor
        assert pinner.calls == [], "Pinner must not be invoked on non-CPU tensors"


class TestCachingByIdentity:
    def test_same_source_pins_once_and_returns_cached_result(self):
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src = torch.zeros(8)

        first = processor.process(src)
        second = processor.process(src)

        assert len(pinner.calls) == 1, "Pinner must be invoked exactly once for the same source"
        assert second is first, "Second process(src) must return the cached pinned tensor"

    def test_different_sources_each_get_pinned(self):
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src_a = torch.zeros(4)
        src_b = torch.zeros(4)

        out_a = processor.process(src_a)
        out_b = processor.process(src_b)

        assert len(pinner.calls) == 2
        assert out_a is not out_b


def test_apply_leaves_tensor_nested_in_list_attribute_untouched() -> None:
    model = torch.nn.Module()
    nested = torch.arange(4.0)
    model.tensors = [nested]
    pinner = _SpyPinner(in_place=False)

    MoveToPinMemoryTensorProcessor(pinner).apply(model)

    assert model.tensors[0] is nested
    assert pinner.calls == []


class TestModeDispatch:
    def test_torch_mode_returns_fresh_tensor(self):
        """Torch mode: pinner returns a new tensor → processor returns it (not src)."""
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src = torch.zeros(4)

        result = processor.process(src)

        assert result is not src
        assert torch.equal(result, src)

    def test_host_register_mode_returns_same_tensor(self):
        """Host_register mode: pinner pins in place → processor returns src itself."""
        pinner = _SpyPinner(in_place=True)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src = torch.zeros(4)

        result = processor.process(src)

        assert result is src, "host_register mode must return the same tensor object"
        assert pinner.calls == [src]


class TestDefaultPinner:
    def test_construction_without_pinner_raises(self):
        """``host_pinner`` is required — silently substituting a default
        torch-mode :class:`HostPinner` would mean a future call site that
        forgets to plumb the manager's pinner would silently get
        torch-mode pinning even when ``pinned_memory_mode='host_register'``
        was configured. The required-arg contract makes the misuse a
        ``TypeError`` at construction time instead of a silent mode
        mismatch at runtime.
        """
        with pytest.raises(TypeError, match="host_pinner"):
            MoveToPinMemoryTensorProcessor()  # type: ignore[call-arg]


class TestInjectedPinnerIsTheOneUsed:
    """Wiring guard: a refactor that drops the injected ``host_pinner`` and
    silently substitutes a default ``HostPinner()`` (e.g. a regression of
    the form ``self.host_pinner = host_pinner or HostPinner()`` losing its
    truthy check, or the constructor accidentally swallowing the kwarg)
    would silently disable host_register-mode pinning for the most common
    discovery-phase processor — and every existing test would still pass
    because the substitute is a valid pinner with the same surface.

    Same regression class as ``test_torch_function_routes_pin_through_host_pinner``
    in ``test_benchmark_tensor_mode.py``: the spy verifies that the exact
    pinner instance passed in is the one whose ``pin`` was called.
    """

    def test_process_dispatches_through_injected_pinner_pin(self):
        spy = MagicMock(spec=HostPinner)
        spy.pin.return_value = torch.zeros(8)

        processor = MoveToPinMemoryTensorProcessor(spy)
        src = torch.zeros(8)

        processor.process(src)

        spy.pin.assert_called_once()
        assert spy.pin.call_args.args[0] is src, (
            "MoveToPinMemoryTensorProcessor invoked pin on a different tensor "
            "than the one passed to process() — pin signal is going to the wrong place"
        )
        assert processor.host_pinner is spy


class TestPinFailurePropagates:
    """A pin failure (RLIMIT_MEMLOCK exhaustion, cudaHostRegister rc≠0, etc.)
    must propagate out of ``process`` rather than silently substituting the
    unpinned source tensor.

    Regression target: a future ``try: pin(...) except RuntimeError: return src``
    refactor would re-introduce the silent-pageable downgrade we removed when
    ``HostPinner.try_pin`` was deleted, and CI would stay green because every
    other test in this module only exercises the success path.
    """

    def test_runtime_error_from_pin_propagates(self):
        spy = MagicMock(spec=HostPinner)
        spy.pin.side_effect = RuntimeError("simulated cudaHostRegister failure")

        processor = MoveToPinMemoryTensorProcessor(spy)
        src = torch.zeros(8)

        with pytest.raises(RuntimeError, match="simulated cudaHostRegister failure"):
            processor.process(src)

        assert id(src) not in processor.cache, (
            "a failed pin must not leave a cache entry — re-running process() "
            "must call the pinner again rather than returning a stale unpinned tensor"
        )


class TestHostRegisterModeCachesViaRegistry:
    """In host_register mode, ``pin`` returns the same tensor on success — the
    processor must still cache the entry so subsequent exposures of the same
    Python object skip the registry lookup."""

    def test_host_register_success_populates_cache(self, monkeypatch):
        from flextensor import host_pinning as host_pinning_module

        # Stub cudart so HostPinRegistry.pin_in_place can register the storage
        # without needing a real CUDA runtime.
        class _OkCudart:
            def __init__(self) -> None:
                self.registered: set[int] = set()

            def cudaHostRegister(self, ptr, _size, _flags):  # noqa: N802
                self.registered.add(int(ptr))

                class _Ok:
                    value = 0

                return _Ok()

            def cudaHostUnregister(self, ptr):  # noqa: N802
                self.registered.discard(int(ptr))

                class _Ok:
                    value = 0

                return _Ok()

        monkeypatch.setattr(host_pinning_module, "_probe_cudart", lambda: _OkCudart())

        registry = HostPinRegistry()
        pinner = HostPinner(registry)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src = torch.zeros(8, dtype=torch.uint8)

        first = processor.process(src)
        second = processor.process(src)

        assert first is src
        assert second is first, "host_register success must populate the cache"
        assert len(registry) == 1


class TestCleanupResetsCache:
    """``cleanup()`` is invoked at the end of every ``apply()`` and must reset
    the id-keyed cache. Without this override (now fixed), CPython's id reuse
    on GC'd tensors would let a stale entry be returned for a different
    storage in a later ``apply()`` — a silent correctness bug."""

    def test_cleanup_clears_cache(self):
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src = torch.zeros(8)

        processor.process(src)
        assert processor.cache, "precondition: cache populated by a successful pin"

        processor.cleanup()

        assert processor.cache == {}

    def test_apply_invokes_cleanup(self):
        """Sanity check that the base ``apply()`` actually drives cleanup,
        so the override above runs in production code paths."""
        pinner = _SpyPinner(in_place=False)
        processor = MoveToPinMemoryTensorProcessor(pinner)
        src = torch.zeros(4)
        processor.process(src)
        assert processor.cache

        # apply() with an empty model still triggers cleanup() at the end.
        processor.apply(torch.nn.Module())

        assert processor.cache == {}
