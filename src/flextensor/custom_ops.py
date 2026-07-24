# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque custom ops that mark compute boundaries for FlexTensor's block H2D scheduler.

Default offload keeps each unit's trap / loader work outside the compiled
graph, so discovery, profiling, and non-compiled inference run eagerly.
Compiled-offload inference instead marks residency with two
``torch.library``-registered custom ops:

* ``torch.ops.flextensor.pre_compute(input_tensor, offload_unit_name, manager_id)``
* ``torch.ops.flextensor.post_compute(output_tensor, offload_unit_name, manager_id)``

These name the *compute* boundaries (wait for residency / release staging), not a
particular loader shape. Storage behind them is a ring of reusable staging slots
owned by a :class:`~flextensor.loaders.PreallocatedLoader`.

Under ``torch.compile``, the custom ops are opaque residency markers: tracing
continues through them and the layer body between ``pre_compute`` and
``post_compute`` can compile as one subgraph. The registration recipe follows
the same pattern used elsewhere for compile-safe prefetch hooks:

* declare the activation carrier as mutated (``Tensor(a!)`` in the schema) —
  see **Mutation schema contract** below,
* register a no-op ``register_fake`` so FakeTensor tracing does not run the
  real ``CompositeExplicitAutograd`` kernel / loader (required for
  ``Library.define``; ``@torch.library.custom_op`` can omit this),
* run the actual stream / event ordering inside the implementation (it
  delegates to the installed preallocated loader).

**Mutation schema contract.** The carrier is a data-dependency token for
``torch.compile``: the layer body consumes the same activation tensor after
``pre_compute`` (and produces the tensor passed to ``post_compute``), so a
mutation annotation creates the FX / AOT edge that keeps H2D wait → compute →
slot release ordered. The eager kernels do **not** write carrier storage
(values are unchanged; profiling may read ``carrier.device`` / stream only).
The ``Tensor(a!)`` over-declaration is intentional. Covered by
``torch.library.opcheck`` in ``tests/unit/test_custom_ops.py``.

The ``offload_unit_name`` argument is the stable offload-unit label (the same
string closed over by compiled forwards). Dynamic expert selection stays
separate runtime data and is not encoded here.

The implementation depends on module-level registries keyed by ``manager_id``
(install via :func:`install_active_loader`, clear via
:func:`clear_active_loader`). Each :class:`~flextensor.OffloadManager` receives
a stable ``compiled_offload_manager_id`` at construction; patched modules carry
it as ``_ft_manager_id`` so compiled forwards dispatch to the correct loader
even when multiple named managers (e.g. Wan2.2 ``transformer`` /
``transformer2``) are active in one process.

Pre-INFERENCE discovery / profiling may execute the same baked-in custom ops
before a loader exists; that phase is explicitly unarmed and remains a hard
no-op. :func:`install_active_loader` arms the manager for inference: a missing
loader after that point raises rather than silently skipping H2D / sync / slot
release. Fake implementations stay no-ops for tracing.
"""

import logging
from dataclasses import dataclass, field

import torch
from torch.library import Library

from flextensor.loaders import PreallocatedLoader

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-manager state (loader registry + optional compiled-graph profiling).
# ---------------------------------------------------------------------------
#
# One :class:`_ManagerState` per :class:`~flextensor.OffloadManager`. Clearing
# or collecting for manager A only touches ``_STATES[A]`` — never a shared list
# that another manager might append to concurrently.


@dataclass
class _ManagerState:
    loader: PreallocatedLoader | None = None
    require_loader: bool = False  # armed by install_active_loader; missing loader then errors
    profiling: bool = False
    pending_start: dict[str, torch.cuda.Event] = field(default_factory=dict)
    duration_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)


_STATES: dict[int, _ManagerState] = {}


def _ensure_state(manager_id: int) -> _ManagerState:
    state = _STATES.get(manager_id)
    if state is None:
        state = _ManagerState()
        _STATES[manager_id] = state
    return state


def _clear_profiling(state: _ManagerState) -> None:
    state.profiling = False
    state.pending_start.clear()
    state.duration_events.clear()


def enable_compiled_profiling(manager_id: int = 0) -> None:
    """Start recording per-unit CUDA-event timings for ``manager_id``.

    Clears any previously collected events for that manager. Call immediately
    before the profiling forwards; pair with :func:`finish_compiled_profiling`
    (or :func:`disable_compiled_profiling` + :func:`collect_compiled_layer_durations`).
    """
    state = _ensure_state(manager_id)
    state.pending_start.clear()
    state.duration_events.clear()
    state.profiling = True


def disable_compiled_profiling(manager_id: int = 0) -> None:
    """Stop recording per-unit CUDA-event timings for ``manager_id``.

    Collected events remain until :func:`collect_compiled_layer_durations`
    or :func:`finish_compiled_profiling` consumes them.
    """
    state = _STATES.get(manager_id)
    if state is None:
        return
    state.profiling = False
    state.pending_start.clear()


def reset_compiled_profiling_state() -> None:
    """Drop compiled-graph profiling fields across every manager (loaders kept)."""
    for state in _STATES.values():
        _clear_profiling(state)


def collect_compiled_layer_durations(manager_id: int = 0) -> dict[str, list[float]]:
    """Return per-unit compiled durations in milliseconds for ``manager_id``.

    Keys are offload-unit names. Synchronizes each collected end event directly
    (not the process current device), then consumes the events.
    """
    state = _STATES.get(manager_id)
    if state is None:
        return {}
    manager_events = list(state.duration_events)
    durations: dict[str, list[float]] = {}
    for unit_name, start_event, end_event in manager_events:
        end_event.synchronize()
        durations.setdefault(unit_name, []).append(start_event.elapsed_time(end_event))
    state.duration_events.clear()
    return durations


def finish_compiled_profiling(manager_id: int = 0) -> dict[str, list[float]]:
    """Stop profiling for ``manager_id`` and return consumed per-unit durations.

    Production path helper: combines :func:`disable_compiled_profiling` and
    :func:`collect_compiled_layer_durations`.
    """
    disable_compiled_profiling(manager_id)
    return collect_compiled_layer_durations(manager_id)


def _record_unit_start(manager_id: int, offload_unit_name: str, carrier: torch.Tensor) -> None:
    state = _STATES.get(manager_id)
    if state is None or not state.profiling or not carrier.is_cuda:
        return
    stream = torch.cuda.current_stream(carrier.device)
    event = torch.cuda.Event(enable_timing=True)
    event.record(stream)
    state.pending_start[offload_unit_name] = event


def _record_unit_end(manager_id: int, offload_unit_name: str, carrier: torch.Tensor) -> None:
    state = _STATES.get(manager_id)
    if state is None or not state.profiling:
        return
    start_event = state.pending_start.pop(offload_unit_name, None)
    if start_event is None or not carrier.is_cuda:
        return
    stream = torch.cuda.current_stream(carrier.device)
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record(stream)
    state.duration_events.append((offload_unit_name, start_event, end_event))


def install_active_loader(
    loader: PreallocatedLoader,
    manager_id: int = 0,
) -> None:
    """Install the rolling-block loader for ``manager_id``; arms require_loader."""
    state = _ensure_state(manager_id)
    if state.loader is not None and state.loader is not loader:
        LOGGER.warning(
            "FlexTensor custom_ops: install_active_loader(manager_id=%d) replacing loader (id=0x%x -> id=0x%x).",
            manager_id,
            id(state.loader),
            id(loader),
        )
    state.loader = loader
    state.require_loader = True
    LOGGER.info(
        "FlexTensor custom_ops: rotating loader installed for manager_id=%d (loader id=0x%x)",
        manager_id,
        id(loader),
    )


def clear_active_loader(manager_id: int | None = None) -> None:
    """Remove loader registration(s).

    ``None`` clears all managers. A per-id clear drops the loader but keeps an
    armed slot so later ``pre_compute/post_compute`` calls still error.
    """
    if manager_id is None:
        _STATES.clear()
        return
    state = _STATES.get(manager_id)
    if state is None:
        return
    if state.require_loader:
        state.loader = None
        _clear_profiling(state)
        return
    _STATES.pop(manager_id, None)


def has_active_loader(manager_id: int) -> bool:
    """Return whether ``manager_id`` has a registered rolling-block loader."""
    state = _STATES.get(manager_id)
    return state is not None and state.loader is not None


def get_active_loader(manager_id: int = 0) -> PreallocatedLoader | None:
    """Return the loader registered for ``manager_id``, or ``None``."""
    state = _STATES.get(manager_id)
    return state.loader if state is not None else None


def _lookup_loader(manager_id: int) -> PreallocatedLoader | None:
    """Return the loader, or ``None`` only for the unarmed pre-install phase.

    Raises:
        RuntimeError: If this manager was armed by :func:`install_active_loader`
            but no loader is currently registered.
    """
    state = _STATES.get(manager_id)
    if state is None:
        return None
    if state.loader is not None:
        return state.loader
    if state.require_loader:
        raise RuntimeError(
            f"FlexTensor custom_ops: pre_compute/post_compute called for manager_id={manager_id} "
            "but no loader is registered. Skipping enter/exit would omit H2D transfer, "
            "synchronization, and slot release — offloaded blocks may read empty or stale "
            "weights. Reinstall via install_active_loader before inference continues."
        )
    return None


# ---------------------------------------------------------------------------
# Custom op implementations and fakes.
# ---------------------------------------------------------------------------


def _pre_compute_impl(
    input_tensor: torch.Tensor,  # compile-ordering carrier; values not written
    offload_unit_name: str,
    manager_id: int,
) -> None:
    loader = _lookup_loader(manager_id)
    if loader is not None:
        loader.enter(offload_unit_name)
    state = _STATES.get(manager_id)
    if state is not None and state.profiling:
        _record_unit_start(manager_id, offload_unit_name, input_tensor)


def _pre_compute_fake(
    input_tensor: torch.Tensor,
    offload_unit_name: str,
    manager_id: int,
) -> None:
    return


def _post_compute_impl(
    output_tensor: torch.Tensor,  # compile-ordering carrier; values not written
    offload_unit_name: str,
    manager_id: int,
) -> None:
    loader = _lookup_loader(manager_id)
    state = _STATES.get(manager_id)
    if state is not None and state.profiling:
        _record_unit_end(manager_id, offload_unit_name, output_tensor)
    if loader is None:
        return
    loader.exit(offload_unit_name)


def _post_compute_fake(
    output_tensor: torch.Tensor,
    offload_unit_name: str,
    manager_id: int,
) -> None:
    return


# ---------------------------------------------------------------------------
# torch.library registration.
# ---------------------------------------------------------------------------
_FT_LIB = Library("flextensor", "FRAGMENT")


def _register_compute_boundary_ops(lib: Library) -> None:
    # Tensor(a!): intentional compile-ordering contract; see module docstring.
    # Explicit register_fake required: Library.define + CompositeExplicitAutograd
    # does not get custom_op's trivial mutable zero-return FakeTensor path.
    lib.define("pre_compute(Tensor(a!) input_tensor, str offload_unit_name, int manager_id) -> ()")
    lib.define("post_compute(Tensor(a!) output_tensor, str offload_unit_name, int manager_id) -> ()")
    lib.impl("pre_compute", _pre_compute_impl, "CompositeExplicitAutograd")
    lib.impl("post_compute", _post_compute_impl, "CompositeExplicitAutograd")
    torch.library.register_fake("flextensor::pre_compute", _pre_compute_fake, lib=lib)
    torch.library.register_fake("flextensor::post_compute", _post_compute_fake, lib=lib)


_register_compute_boundary_ops(_FT_LIB)
