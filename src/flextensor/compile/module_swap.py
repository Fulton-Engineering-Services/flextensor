# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve outermost offloaded units and build in-place submodule setters."""

from collections.abc import Callable, Iterable

from torch import nn

from flextensor.tensor_discovery import has_patched_ancestor, is_offload_patched_module


def make_submodule_setter(model: nn.Module, qualified_name: str) -> Callable[[nn.Module], None]:
    """Build a setter that replaces the submodule at ``qualified_name`` in-place."""
    parent_path, _, child = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model

    def _setter(new_module: nn.Module) -> None:
        if isinstance(parent, nn.ModuleList) and child.isdigit():
            parent[int(child)] = new_module
        else:
            setattr(parent, child, new_module)

    return _setter


def resolve_compile_targets(
    model: nn.Module,
    patched_modules: Iterable[nn.Module],
) -> list[tuple[Callable[[nn.Module], None], nn.Module]]:
    """Return ``(setter, module)`` pairs — one per outermost offloaded unit.

    Only outermost patched modules are targeted: if a patched module has a patched
    ancestor, that ancestor is the offload/compile unit and the nested one is skipped.
    """
    if not patched_modules:
        return []
    name_by_module = {module: name for name, module in model.named_modules()}
    targets: list[tuple[Callable[[nn.Module], None], nn.Module]] = []
    for module in patched_modules:
        qualified_name = name_by_module.get(module)
        if qualified_name is None:
            continue
        if has_patched_ancestor(model, qualified_name):
            continue
        targets.append((make_submodule_setter(model, qualified_name), module))
    return targets


def resolve_compile_targets_on_model(
    model: nn.Module,
) -> list[tuple[Callable[[nn.Module], None], nn.Module]]:
    """Like :func:`resolve_compile_targets` but discovers patched units by walking ``model``."""
    targets: list[tuple[Callable[[nn.Module], None], nn.Module]] = []
    for qualified_name, module in model.named_modules():
        if not qualified_name:
            continue
        if not is_offload_patched_module(module):
            continue
        if has_patched_ancestor(model, qualified_name):
            continue
        targets.append((make_submodule_setter(model, qualified_name), module))
    return targets
