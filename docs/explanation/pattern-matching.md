<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Pattern Matching

FlexTensor uses wildcard patterns to select modules and parameters for offloading. Each entry in `include_patterns` / `exclude_patterns` is one of three forms:

| Form | Selects on | Example |
|------|-----------|---------|
| `<glob>` | Module / parameter path (default) | `layers.*`, `*.weight` |
| `name:<glob>` | Module / parameter path (explicit) | `name:layers.*` |
| `class:<glob>` | Module's class (short name **and** fully-qualified class name) | `class:SharedExpertMLP`, `class:torch.nn.*.Linear` |

Bare patterns behave like `name:` — the prefix is only needed to disambiguate a literal path that starts with `class:` or `name:`, which is unusual.

This document explains the matching semantics for each form.

## Name patterns

Name patterns are dot-separated segments that match against module paths (e.g., `layers.0.self_attn`) or parameter paths (e.g., `layers.0.self_attn.q_proj.weight`).

| Syntax | Meaning |
|--------|---------|
| `layers` | Exact match on the segment `layers` |
| `layer_*` | Intra-segment wildcard: matches `layer_0`, `layer_abc`, etc. |
| `layer?` | Single character wildcard: matches `layer0` but not `layer01` |
| `*` | Standalone wildcard: matches one or more segments (see below) |
| `layers.*` | `layers` followed by wildcard segment(s) |

## Include vs exclude wildcard behavior (name patterns)

The standalone `*` segment behaves differently in include and exclude patterns to match the most common use cases.

### Include patterns

In include patterns, a standalone `*` matches **exactly one** path segment. Combined with the ancestor guard (see [Offload units](#offload-units) below), this prevents accidentally nesting offload blocks inside each other.

| Pattern | Matches | Does not match |
|---------|---------|----------------|
| `layers.*` | `layers.0`, `layers.1` | `layers.0.attn` |
| `*` | `layers`, `norm`, `head` | `layers.0` |
| `layers.*.self_attn` | `layers.0.self_attn` | `layers.0.self_attn.q_proj` |
| `*.weight` | `lm_head.weight`, `norm.weight` (parameter-level) | `layers.0.attn.q_proj.weight` (more than one `*` segment) |
| `layers.*.weight` | `layers.0.weight` (parameter-level) | `layers.0.attn.weight` |

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

## Class patterns

A `class:<glob>` pattern matches on the module's Python class rather than its path. Each pattern is tested against **both**:

- the short class name — `type(module).__name__`, e.g. `SharedExpertMLP`
- the fully-qualified class name (FQCN) — `f"{cls.__module__}.{cls.__qualname__}"`, e.g. `nemotron_h.layers.SharedExpertMLP`

A match on either haystack wins. Globs (`*`, `?`) are supported inside the body:

| Pattern | Matches |
|---------|---------|
| `class:SharedExpertMLP` | Any module whose short class name is `SharedExpertMLP` |
| `class:*Expert*` | Any class whose short name contains `Expert` (e.g. `SharedExpertMLP`, `RoutedExpert`) |
| `class:torch.nn.*.Linear` | `torch.nn`'s `Linear` (matched via FQCN) — disambiguates from a user-defined `Linear` |
| `class:*.SharedExpertMLP` | Any `SharedExpertMLP` regardless of which module it lives in |

Use a dotted pattern (e.g. `class:torch.nn.*.Linear`) when two classes share a short name and you need to disambiguate them.

**Why use class patterns.** One entry matches every module of a given class, regardless of its path. Useful when the same sub-module type appears in many places.

**Scope.** Class patterns are **module-level only** — every matching module is selected (or excluded) as a whole, along with all of its descendant parameters. A `class:` entry cannot target individual parameters; use a `name:` parameter pattern for that.

### Dict-typed models

Dict-typed models (used by the lazy-init / load-from-profile path) have no module hierarchy, so `class:` patterns cannot resolve against them:

- **`include_patterns` contains only `class:` entries** — raises `ValueError` at offload time, because no parameters would match. Use name-based patterns (e.g. `["layers.*.weight"]`) for dict models.
- **Otherwise** — any `class:` entries in `include_patterns` or `exclude_patterns` are ignored with a warning, and the surviving name patterns are applied as usual.

## Offload units

An **offload unit** is a patched module whose ancestors are not themselves patched. Offload units define the boundaries for tensor offloading -- all descendant modules and parameters belong to their ancestor's offload unit.

When include patterns overlap hierarchically (e.g., `["layers.*", "layers.*.attn"]`), the **ancestor guard** ensures only the outermost match (`layers.0`) is patched. The inner match (`layers.0.attn`) is skipped because it already belongs to `layers.0`'s offload unit.

## Evaluation order

1. **Include patterns** select candidate modules. Name and `class:` entries are combined — a module is a candidate if either kind matches.
2. **Ancestor guard** keeps only the outermost candidate in each subtree. The kept module becomes an offload unit; its descendants are offloaded as part of it.
3. **Exclude patterns** drop offload units that match an exclude entry (name or `class:`). Excludes that target a descendant inside an offload unit have no module-level effect — they are applied at step 4 instead.
4. **Parameter-level excludes** keep individual parameters of an offload unit on GPU during tensor discovery. A parameter matching both an include and an exclude is **not** offloaded.

## Module-level vs parameter-level matching

- **Module-level (name or `class:`)**: Include and exclude patterns match against module paths from `model.named_modules()` (name) or `type(module)` (class). Only offload units (modules with no patched ancestors) are independently patched; a matching exclude entry drops that offload unit from the offload set.
- **Parameter-level includes (name only)**: Include patterns can target individual parameters (e.g., `*.weight` or `layers.*.weight`). When a pattern's final segment matches a parameter name rather than a sub-module, only those specific tensors are selected for offloading within the matched module. This is useful for offloading only large weight tensors while keeping small biases or normalization scales on GPU. `class:` patterns do not have a parameter-level form — a `class:` match cascades to every parameter of the matched module.
- **Parameter-level excludes (name only)**: Exclude patterns also filter individual parameters during tensor discovery. A module-level exclude pattern (e.g., `layers.*.norm`) cascades to all parameters of the matching sub-module within the offload unit. A parameter-level pattern (e.g., `*.scale`) excludes specific tensors without excluding their parent module. As with includes, `class:` is module-level only.
