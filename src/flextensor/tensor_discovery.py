# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor discovery for finding untraced tensors.

Three discovery mechanisms (tried sequentially):
1. Auto trap (forward patching): Use get_offload_module_tensor_ids() to discover
   all tensors from patched modules via named_parameters().
2. Manual trap with ModuleTracker: Track which modules execute within each trap,
   then discover their parameters via named_parameters().
3. Prefix matching (fallback): Match tensors by name prefix inferred from traced tensors.

"""

from __future__ import annotations

from typing import Any

import torch

from flextensor.collectors import IterativeLayerStatistics

# =============================================================================
# Module Tracker - tracks modules executed within trap contexts
# =============================================================================


class ModuleTracker:
    """Tracks which modules are executed within each trap context.

    During warmup, forward hooks record which modules are called. Combined with
    the current trap name, this builds a mapping: trap_name → set of modules.

    After warmup, this mapping is used to discover untraced parameters by
    iterating each trap's modules and calling named_parameters().

    Usage:
        tracker = ModuleTracker()
        tracker.register(model)

        # During warmup - enter/exit trap context
        tracker.enter_trap("layer_0")
        model(x)  # Hooks record modules executed under "layer_0"
        tracker.exit_trap("layer_0")

        # After warmup - get tensor IDs per trap and unregister
        trap_to_tensors = tracker.get_trap_tensor_ids(tensors_map)
        tracker.unregister()
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._trap_to_modules: dict[str, set[torch.nn.Module]] = {}
        self._current_trap: str | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._registered = False

    def register(self, model: torch.nn.Module) -> None:
        """Register forward hooks on all modules in the model.

        Args:
            model: The model to track.
        """
        if self._registered:
            return

        self._register_hooks_recursive(model)
        self._registered = True

    def _register_hooks_recursive(self, module: torch.nn.Module) -> None:
        """Recursively register hooks on module and all children."""
        # Only register on modules that have parameters (leaf modules with weights)
        if any(True for _ in module.parameters(recurse=False)):
            handle = module.register_forward_hook(self._forward_hook)
            self._hooks.append(handle)

        for child in module.children():
            self._register_hooks_recursive(child)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        _inputs: Any,
        _output: Any,
    ) -> None:
        """Forward hook that records the module under current trap."""
        if self._current_trap is None:
            return

        if self._current_trap not in self._trap_to_modules:
            self._trap_to_modules[self._current_trap] = set()

        self._trap_to_modules[self._current_trap].add(module)

    def enter_trap(self, trap_name: str) -> None:
        """Enter a trap context for tracking.

        Args:
            trap_name: Name of the trap context.
        """
        self._current_trap = trap_name

    def exit_trap(self, trap_name: str) -> None:
        """Exit a trap context.

        Args:
            trap_name: Name of the trap context (for symmetry, not currently used).
        """
        del trap_name  # unused, kept for API symmetry
        self._current_trap = None

    @property
    def current_trap(self) -> str | None:
        """Get the current trap name."""
        return self._current_trap

    def get_trap_to_modules(self) -> dict[str, set[torch.nn.Module]]:
        """Get the mapping of trap names to modules executed within them.

        Returns:
            Dict mapping trap_name → set of modules.
        """
        return dict(self._trap_to_modules)

    def get_trap_tensor_ids(
        self,
        tensors_map: dict[int, torch.Tensor],
    ) -> dict[str, set[int]]:
        """Get tensor IDs for each trap from their executed modules.

        For each trap, iterates over modules that were executed within it
        and collects their parameter tensor IDs (including inner fields).

        Args:
            tensors_map: Map of known tensor IDs to tensors.

        Returns:
            Dict mapping trap_name → set of tensor IDs.
        """
        all_tensor_ids = set(tensors_map.keys())
        trap_to_tensor_ids: dict[str, set[int]] = {}

        for trap_name, modules in self._trap_to_modules.items():
            tensor_ids: set[int] = set()

            for module in modules:
                # Get all parameters from this module (non-recursive since we track all)
                for param in module.parameters(recurse=False):
                    param_id = id(param)
                    if param_id in all_tensor_ids:
                        tensor_ids.add(param_id)

                        # Also add inner tensor fields (e.g., weight.scale for FP8)
                        inner_ids = get_inner_tensor_field_ids(param)
                        for inner_id in inner_ids:
                            if inner_id in all_tensor_ids:
                                tensor_ids.add(inner_id)

            trap_to_tensor_ids[trap_name] = tensor_ids

        return trap_to_tensor_ids

    def clear(self) -> None:
        """Clear all tracked data but keep hooks registered."""
        self._trap_to_modules.clear()
        self._current_trap = None

    def unregister(self) -> None:
        """Remove all hooks and clear data."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._trap_to_modules.clear()
        self._current_trap = None
        self._registered = False


# =============================================================================
# Discovery strategy using ModuleTracker
# =============================================================================


def _discover_from_module_tracker(
    tracker: ModuleTracker | None,
    tensors_map: dict[int, torch.Tensor],
    untraced_tensor_ids: set[int],
    layer_labels_set: set[str],
    layer_to_new_tensors: dict[str, set[int]],
) -> set[int]:
    """Strategy: Discover untraced tensors from ModuleTracker.

    Uses the modules tracked during warmup to find their parameters
    that weren't traced via __torch_function__.

    Args:
        tracker: The ModuleTracker instance (or None if not used).
        tensors_map: Map of tensor IDs to tensors.
        untraced_tensor_ids: Set of tensor IDs not yet traced.
        layer_labels_set: Set of valid layer labels.
        layer_to_new_tensors: Mutable dict to populate with discovered tensors.

    Returns:
        Set of tensor IDs that were matched by this strategy.
    """
    if tracker is None:
        return set()

    matched_tensor_ids: set[int] = set()
    trap_to_tensor_ids = tracker.get_trap_tensor_ids(tensors_map)

    for trap_name, module_tensor_ids in trap_to_tensor_ids.items():
        if trap_name in layer_labels_set:
            untraced_in_trap = module_tensor_ids & untraced_tensor_ids
            layer_to_new_tensors[trap_name].update(untraced_in_trap)
            matched_tensor_ids.update(untraced_in_trap)

    return matched_tensor_ids


def get_inner_tensor_field_ids(tensor: torch.Tensor) -> list[int]:
    """Get IDs of inner tensor fields attached to a tensor.

    This detects additional tensor attributes like weight.scale for fp8 tensors.
    These inner fields are tensor attributes that aren't part of the base torch.Tensor class.

    Args:
        tensor: The tensor to inspect for inner tensor fields.

    Returns:
        List of tensor IDs for inner tensor fields.
    """
    inner_ids = []
    tensor_fields = set(dir(tensor))
    base_fields = set(dir(torch.Tensor))
    custom_fields = tensor_fields - base_fields

    for field_name in custom_fields:
        if field_name.startswith("_"):
            continue
        field = getattr(tensor, field_name, None)
        if isinstance(field, torch.Tensor):
            inner_ids.append(id(field))

    return inner_ids


def get_offload_module_tensor_ids(
    model: torch.nn.Module | dict | None,
    tensors_map: dict[int, torch.Tensor],
) -> dict[str, set[int]]:
    """Discover tensor IDs for each patched module in the model.

    Finds all modules with forward patching and collects the tensor IDs of their
    parameters. This directly maps tensors to their containing trap/layer without
    string matching.

    Args:
        model: The model to search for patched modules.
        tensors_map: Map of tensor IDs to tensors (used to filter to known tensors).

    Returns:
        Dict mapping offload_name (trap label) to set of tensor IDs.
    """
    label_to_tensor_ids: dict[str, set[int]] = {}
    all_tensor_ids = set(tensors_map.keys())

    if model is None or isinstance(model, dict):
        return label_to_tensor_ids

    for module in model.modules():
        # Check if module has forward patching applied
        if is_offload_patched_module(module):
            label = get_offload_name(module)
            if label is None:
                continue
            tensor_ids: set[int] = set()

            # Get all parameters from the patched module
            for param in module.parameters():
                param_id = id(param)
                if param_id in all_tensor_ids:
                    tensor_ids.add(param_id)

                    # Also add inner tensor fields (e.g., weight.scale)
                    inner_ids = get_inner_tensor_field_ids(param)
                    for inner_id in inner_ids:
                        if inner_id in all_tensor_ids:
                            tensor_ids.add(inner_id)

            label_to_tensor_ids[label] = tensor_ids

    return label_to_tensor_ids


def is_offload_patched_module(module: torch.nn.Module) -> bool:
    """Check if a module has forward patching applied.

    Args:
        module: The module to check.

    Returns:
        True if the module has forward patching applied (i.e., has _ft_original_forward_func).
    """
    return hasattr(module, "_ft_original_forward_func")


def get_offload_name(module: torch.nn.Module) -> str | None:
    """Get the offload name from a patched module.

    Args:
        module: The module to get the offload name from.

    Returns:
        The offload name if the module is patched, None otherwise.
    """
    return getattr(module, "_ft_offload_name", None)


def has_offload_modules(model: torch.nn.Module | dict | None) -> bool:
    """Check if model contains any modules with patched forward methods.

    Args:
        model: The model to check.

    Returns:
        True if model contains at least one module with forward patching.
    """
    if model is None or isinstance(model, dict):
        return False

    return any(is_offload_patched_module(module) for module in model.modules())


def _find_common_prefix(names: list[str]) -> str:
    """Find the common prefix of a list of dot-separated names.

    Finds the exact common prefix, then broadens it by stripping parts that
    diverge among the names. This avoids hardcoded submodule lists and instead
    relies on the actual structure of the tensor names within the trap.

    The algorithm stops at a "structural boundary" - typically after a numeric
    index that identifies the layer (e.g., "layers.0", "blocks.3").

    Args:
        names: List of tensor names (e.g., ["layers.0.attn.q.weight", "layers.0.ffn.w1.weight"]).

    Returns:
        Broad common prefix (e.g., "layers.0") or empty string if no common prefix.

    Example:
        ["layers.0.attn.q.weight", "layers.0.attn.k.weight"] → "layers.0"
        ["layers.0.attn.q.weight", "layers.0.ffn.w1.weight"] → "layers.0"
        ["model.encoder.layer.0.attention.self.query.weight",
         "model.encoder.layer.0.output.dense.weight"] → "model.encoder.layer.0"
    """
    if not names:
        return ""

    # Split all names into parts
    split_names = [name.split(".") for name in names]
    if not split_names:
        return ""

    # Find exact common prefix parts (where all names share the same part at position i)
    min_len = min(len(parts) for parts in split_names)
    common_parts = []

    for i in range(min_len):
        part = split_names[0][i]
        if all(parts[i] == part for parts in split_names):
            common_parts.append(part)
        else:
            break

    if not common_parts:
        return ""

    # Now broaden the prefix by finding a structural boundary.
    # Strategy: Keep the prefix up to and including the last numeric part,
    # as numeric indices typically identify layers (e.g., "layers.0", "blocks.12").
    # This works for most model architectures without hardcoding component names.

    # Find the last numeric part in the common prefix
    last_numeric_idx = -1
    for i, part in enumerate(common_parts):
        if part.isdigit():
            last_numeric_idx = i

    # If we found a numeric part, use everything up to and including it
    if last_numeric_idx >= 0:
        return ".".join(common_parts[: last_numeric_idx + 1])

    # Otherwise, use the full common prefix (all parts match exactly)
    return ".".join(common_parts)


def _get_layer_tensor_prefixes(
    layer_stats: list[IterativeLayerStatistics],
    tensor_id_to_name_map: dict[int, str],
) -> dict[str, str]:
    """Infer tensor name prefixes for each layer from traced tensors.

    For each layer, looks at the names of traced tensors and finds their common prefix.
    This allows matching untraced tensors based on the pattern of traced tensors.

    Example:
        Layer "0" has traced tensors: ["layers.0.attn.q.weight", "layers.0.attn.k.weight"]
        Inferred prefix: "layers.0"
        Untraced tensor "layers.0.ffn.w1.weight" matches this prefix → belongs to layer "0"

    Args:
        layer_stats: List of layer statistics with traced tensor IDs.
        tensor_id_to_name_map: Map of tensor IDs to names.

    Returns:
        Dict mapping layer label to inferred tensor name prefix.
    """
    label_to_prefix: dict[str, str] = {}

    for layer_stat in layer_stats:
        # Get names of traced tensors in this layer
        traced_names = []
        for tensor_id in layer_stat.tensor_ids:
            name = tensor_id_to_name_map.get(tensor_id, "")
            if name:
                traced_names.append(name)

        if traced_names:
            prefix = _find_common_prefix(traced_names)
            if prefix:
                label_to_prefix[layer_stat.label] = prefix

    return label_to_prefix


def _match_tensor_to_layer_by_prefix(
    tensor_name: str,
    label_to_prefix: dict[str, str],
) -> str | None:
    """Match a tensor name to a layer using inferred prefixes from traced tensors.

    Args:
        tensor_name: Name of the tensor (e.g., "layers.0.ffn.w1.weight").
        label_to_prefix: Map of layer labels to inferred prefixes.

    Returns:
        Best matching layer label, or None if no match found.
    """
    if not tensor_name:
        return None

    # Sort by prefix length (longest first) for best match
    sorted_labels = sorted(label_to_prefix.items(), key=lambda x: len(x[1]), reverse=True)

    for label, prefix in sorted_labels:
        if tensor_name.startswith(prefix + ".") or tensor_name == prefix:
            return label

    return None


def _discover_from_offload_modules(
    model: torch.nn.Module | dict | None,
    tensors_map: dict[int, torch.Tensor],
    untraced_tensor_ids: set[int],
    layer_labels_set: set[str],
    layer_to_new_tensors: dict[str, set[int]],
) -> set[int]:
    """Discover untraced tensors from patched modules.

    For models using forward patching, directly get tensor IDs from
    patched modules' parameters. This is the most accurate method.

    Args:
        model: The model to search for patched modules.
        tensors_map: Map of tensor IDs to tensors.
        untraced_tensor_ids: Set of tensor IDs not yet traced.
        layer_labels_set: Set of valid layer labels.
        layer_to_new_tensors: Mutable dict to populate with discovered tensors.

    Returns:
        Set of tensor IDs that were matched by this strategy.
    """
    matched_tensor_ids: set[int] = set()
    label_to_module_tensors = get_offload_module_tensor_ids(model, tensors_map)

    for label, module_tensor_ids in label_to_module_tensors.items():
        if label in layer_labels_set:
            untraced_in_module = module_tensor_ids & untraced_tensor_ids
            layer_to_new_tensors[label].update(untraced_in_module)
            matched_tensor_ids.update(untraced_in_module)

    return matched_tensor_ids


def _discover_by_prefix_matching(
    layer_stats: list[IterativeLayerStatistics],
    tensors_map: dict[int, torch.Tensor],
    tensor_id_to_name_map: dict[int, str],
    untraced_tensor_ids: set[int],
    already_matched: set[int],
    layer_to_new_tensors: dict[str, set[int]],
) -> set[int]:
    """Match untraced tensors to layers using name prefix matching.

    Infers tensor name prefixes from traced tensors in each layer, then matches
    untraced tensors based on those prefixes.

    Example:
        Layer "0" has traced tensors like "layers.0.attn.q.weight"
        → inferred prefix "layers.0"
        → matches untraced "layers.0.ffn.w1.weight"

    Args:
        layer_stats: List of layer statistics with traced tensor IDs.
        tensors_map: Map of tensor IDs to tensors.
        tensor_id_to_name_map: Map of tensor IDs to names.
        untraced_tensor_ids: Set of tensor IDs not yet traced.
        already_matched: Set of tensor IDs already matched by previous strategies.
        layer_to_new_tensors: Mutable dict to populate with discovered tensors.

    Returns:
        Set of tensor IDs that were matched by this strategy.
    """
    matched_tensor_ids: set[int] = set()
    remaining_untraced = untraced_tensor_ids - already_matched

    if not remaining_untraced or not tensor_id_to_name_map:
        return matched_tensor_ids

    all_tensor_ids = set(tensors_map.keys())
    label_to_prefix = _get_layer_tensor_prefixes(layer_stats, tensor_id_to_name_map)

    for tensor_id in remaining_untraced:
        tensor_name = tensor_id_to_name_map.get(tensor_id, "")
        best_match = _match_tensor_to_layer_by_prefix(tensor_name, label_to_prefix)

        if best_match is not None:
            layer_to_new_tensors[best_match].add(tensor_id)
            matched_tensor_ids.add(tensor_id)

            # Also add inner tensor fields (e.g., weight.scale)
            tensor = tensors_map[tensor_id]
            inner_ids = get_inner_tensor_field_ids(tensor)
            for inner_id in inner_ids:
                if inner_id in all_tensor_ids:
                    layer_to_new_tensors[best_match].add(inner_id)
                    matched_tensor_ids.add(inner_id)

    return matched_tensor_ids


def discover_untraced_tensors_for_layers(
    layer_stats: list[IterativeLayerStatistics],
    tensors_map: dict[int, torch.Tensor],
    model: torch.nn.Module | dict | None,
    tensor_id_to_name_map: dict[int, str],
    module_tracker: ModuleTracker | None = None,
) -> list[IterativeLayerStatistics]:
    """Add untraced tensors to layer statistics using a sequential strategy approach.

    This function discovers such tensors using strategies in order, stopping when
    one succeeds:

    1. Forward patching discovery (auto trap): If model uses forward patching,
       directly get tensor IDs from patched modules. This is the most accurate method.
    2. ModuleTracker (manual trap): If provided and forward patching didn't succeed,
       use modules tracked during warmup to discover parameters.
    3. Prefix matching (fallback): Only if both above fail, use name-based matching.

    Each strategy is tried only if the previous one didn't find all untraced tensors.

    Args:
        layer_stats: List of iterative layer statistics to augment.
        tensors_map: Map of tensor IDs to actual tensors.
        model: The model to search for patched modules.
        tensor_id_to_name_map: Map of tensor IDs to names (for prefix matching).
        module_tracker: Optional ModuleTracker that tracked modules during warmup.

    Returns:
        New list of IterativeLayerStatistics with untraced tensors added.
    """
    if not layer_stats or not tensors_map:
        return layer_stats

    # Collect all tensor IDs that are already traced (in any layer)
    traced_tensor_ids: set[int] = set()
    for layer_stat in layer_stats:
        traced_tensor_ids.update(layer_stat.tensor_ids)

    # Find untraced tensors (in tensors_map but not traced)
    all_tensor_ids = set(tensors_map.keys())
    untraced_tensor_ids = all_tensor_ids - traced_tensor_ids

    if not untraced_tensor_ids:
        return layer_stats

    # Initialize per-layer tensor mapping
    layer_labels = [stat.label for stat in layer_stats]
    layer_labels_set = set(layer_labels)
    layer_to_new_tensors: dict[str, set[int]] = {label: set() for label in layer_labels}

    # Strategy 1: Forward patching discovery (auto trap)
    # This is the most accurate - try first if model is available
    if isinstance(model, torch.nn.Module) and has_offload_modules(model):
        matched_from_patching = _discover_from_offload_modules(
            model, tensors_map, untraced_tensor_ids, layer_labels_set, layer_to_new_tensors
        )
        untraced_tensor_ids -= matched_from_patching

    # Strategy 2: ModuleTracker (manual trap)
    # Only try if we still have untraced tensors and tracker is available
    if untraced_tensor_ids and module_tracker is not None and isinstance(model, torch.nn.Module):
        matched_from_tracker = _discover_from_module_tracker(
            module_tracker, tensors_map, untraced_tensor_ids, layer_labels_set, layer_to_new_tensors
        )
        untraced_tensor_ids -= matched_from_tracker

    # Strategy 3: Prefix matching (fallback)
    # Only try if we still have untraced tensors
    if untraced_tensor_ids:
        matched_from_prefix = _discover_by_prefix_matching(
            layer_stats, tensors_map, tensor_id_to_name_map, untraced_tensor_ids, set(), layer_to_new_tensors
        )
        untraced_tensor_ids -= matched_from_prefix

    # Build enhanced layer stats
    enhanced_stats = []
    for layer_stat in layer_stats:
        new_tensor_ids = set(layer_stat.tensor_ids) | layer_to_new_tensors[layer_stat.label]
        enhanced_stats.append(
            IterativeLayerStatistics(
                label=layer_stat.label,
                tensor_ids=new_tensor_ids,
                duration=layer_stat.duration,
            )
        )

    return enhanced_stats
