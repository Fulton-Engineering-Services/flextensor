<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Pattern Matching

FlexTensor uses dot-separated wildcard patterns to select modules and parameters for offloading. This document explains the matching semantics.

## Pattern syntax

Patterns are dot-separated segments that match against module paths (e.g., `layers.0.self_attn`).

| Syntax | Meaning |
|--------|---------|
| `layers` | Exact match on the segment `layers` |
| `layer_*` | Intra-segment wildcard: matches `layer_0`, `layer_abc`, etc. |
| `layer?` | Single character wildcard: matches `layer0` but not `layer01` |
| `*` | Standalone wildcard: matches one or more segments (see below) |
| `layers.*` | `layers` followed by wildcard segment(s) |

## Include vs exclude wildcard behavior

The standalone `*` segment behaves differently in include and exclude patterns to match the most common use cases.

### Include patterns

In include patterns, a standalone `*` matches **exactly one** path segment. Combined with the ancestor guard (see [Offload units](#offload-units) below), this prevents accidentally nesting offload blocks inside each other.

| Pattern | Matches | Does not match |
|---------|---------|----------------|
| `layers.*` | `layers.0`, `layers.1` | `layers.0.attn` |
| `*` | `layers`, `norm`, `head` | `layers.0` |
| `layers.*.self_attn` | `layers.0.self_attn` | `layers.0.self_attn.q_proj` |

### Exclude patterns

In exclude patterns, a standalone `*` matches **one or more** path segments. This makes it easy to exclude all descendants of a module.

| Pattern | Matches |
|---------|---------|
| `foo.*` | `foo.bar`, `foo.bar.baz`, `foo.bar.baz.weight` |
| `*.weight` | `lm_head.weight`, `layers.0.self_attn.q_proj.weight` |
| `*` | Any path with one or more segments |

### Intra-segment wildcards

In both modes, `*` within other characters (e.g., `layer_*`) always matches within a single segment:

- `layer_*` matches `layer_0`, `layer_abc`, but not `layer_0.attn`

## Offload units

An **offload unit** is a patched module whose ancestors are not themselves patched. Offload units define the boundaries for tensor offloading -- all descendant modules and parameters belong to their ancestor's offload unit.

When include patterns overlap hierarchically (e.g., `["layers.*", "layers.*.attn"]`), the **ancestor guard** ensures only the outermost match (`layers.0`) is patched. The inner match (`layers.0.attn`) is skipped because it already belongs to `layers.0`'s offload unit.

## Evaluation order

1. **Include patterns** are evaluated to select candidate modules.
2. **Ancestor guard** skips any candidate whose ancestor is already patched, forming offload units.
3. **Exclude patterns** are applied to un-patch offload units matching the exclude. Exclude patterns targeting descendants within an offload unit are no-ops at the module level.
4. **Parameter-level excludes** filter individual parameters within offload units during tensor discovery.
5. A target matching both include and exclude is **not** offloaded.

## Module-level vs parameter-level matching

- **Module-level**: Include and exclude patterns match against module paths from `model.named_modules()`. Only offload units (modules with no patched ancestors) are independently patched; exclude patterns un-patch those offload units.
- **Parameter-level**: Exclude patterns also filter individual parameters during tensor discovery. A module-level exclude pattern (e.g., `layers.*.norm`) cascades to all parameters of the matching sub-module within the offload unit. A parameter-level pattern (e.g., `*.scale`) excludes specific tensors without excluding their parent module.
