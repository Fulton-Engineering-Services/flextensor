<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# ADR-0004: Percentage-Based GPU Memory Limit Configuration

**Date**: 2026-03-02

**Status**: Accepted

## Context

The `max_gpu_mem_bytes` configuration field in `OffloadConfig` currently accepts only an absolute
byte value (`int | None`). When set, strategies switch from latency mode to memory mode and keep
peak GPU usage within that budget.

Specifying an absolute byte count is fragile in practice:

- Users must know the exact total memory of the target GPU and calculate the byte value themselves.
- The same configuration cannot be reused across GPUs with different memory sizes (e.g., A100 40GB
  vs A100 80GB) without manual adjustment.
- A percentage-based limit (e.g., 0.9 = "use up to 90% of GPU memory") is more intuitive, portable,
  and aligns with how users reason about memory budgets.

The field is consumed in a single place — `OffloadManager._initialize_tensor_manager()` — before being
passed to strategy constructors (`AdaptiveStrategy`, `KnapsackStrategy`, `GlobalOffloadStrategy`,
etc.) which all expect `max_gpu_mem_bytes: int | None`. The env var is `FT_MAX_GPU_MEM_BYTES`.

The resolution from fraction to bytes requires querying the GPU at runtime via
`torch.cuda.mem_get_info(device)` or `torch.cuda.get_device_properties(device).total_memory`.

## Decision

Option D (replace with fraction-only) implemented via a deprecate-first approach:

- Add `max_gpu_mem_fraction: float | None` (default `0.9`) as the primary field in `OffloadConfig`.
- Deprecate `max_gpu_mem_bytes` in v0.4; remove in v0.5.
- Resolution from fraction to bytes occurs in `OffloadManager._initialize_tensor_manager()` via
  `torch.cuda.get_device_properties(config.gpu_device).total_memory`.
- The SHM namespace hash uses resolved bytes (not the raw fraction), so different GPU SKUs produce
  different namespaces automatically — correct for homogeneous device fleets where all workers
  share the same GPU SKU.
- `max_gpu_mem_fraction` naming chosen for internal consistency with `max_gpu_mem_bytes` (direct
  suffix swap), taking precedence over ecosystem alignment with `gpu_memory_utilization` (vLLM).
- Default `0.9` aligns with vLLM and TensorRT-LLM conventions.

## Options

### Option A: Add a new `max_gpu_mem_fraction` field (additive, backward-compatible)

Add a second field alongside the existing one:

```python
max_gpu_mem_bytes: int | None = Field(default=None, ge=0)
max_gpu_mem_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
```

A pydantic `model_validator` ensures at most one is set. Resolution happens in
`OffloadManager._initialize_tensor_manager()`:

```python
if config.max_gpu_mem_fraction is not None:
    total = torch.cuda.get_device_properties(config.gpu_device).total_memory
    max_gpu_mem_bytes = int(total * config.max_gpu_mem_fraction)
```

Strategies continue to receive `max_gpu_mem_bytes: int` — zero downstream changes.

| | |
|---|---|
| **Pros** | Fully backward-compatible; explicit field names; easy pydantic validation; env vars are clear (`FT_MAX_GPU_MEM_FRACTION=0.9`); no changes to strategy code. |
| **Cons** | Two fields for the same concept; users must know which one to use; slightly larger config surface. |

### Option B: Union type — single field accepting `int` (bytes) or `float` (fraction)

Replace the existing field with a union:

```python
max_gpu_mem: int | float | None = Field(default=None)
```

Convention: `int` → absolute bytes, `float` in (0.0, 1.0] → fraction of total GPU memory.

| | |
|---|---|
| **Pros** | Single field; concise API; one env var (`FT_MAX_GPU_MEM`). |
| **Cons** | Breaking rename (`max_gpu_mem_bytes` → `max_gpu_mem`); implicit semantics based on Python type (`1` = 1 byte vs `1.0` = 100% — subtle); JSON/YAML cannot distinguish `int` vs `float` for values like `1` vs `1.0`; requires custom pydantic validator; harder to document. |

### Option C: String-based with units (e.g., `"90%"`, `"8GB"`)

Accept a string with an explicit unit suffix, plus raw `int` for backward compatibility:

```python
max_gpu_mem: str | int | None = Field(default=None)
```

Supports patterns like `"90%"`, `"8GB"`, `"4096MB"`, and bare `int` (bytes).

| | |
|---|---|
| **Pros** | Very user-friendly from config files and env vars (`FT_MAX_GPU_MEM=90%`); self-documenting values; extensible to other units. |
| **Cons** | Custom parser required; easy to mistype (`"90 %"`, `"8 GB"`); pydantic validation becomes entirely custom; breaking rename; string parsing adds complexity and error surface. |

### Option D: Replace with `max_gpu_mem_fraction` only (breaking)

Remove the bytes field entirely:

```python
max_gpu_mem_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
```

| | |
|---|---|
| **Pros** | Simplest API; no ambiguity; single field with clear semantics. |
| **Cons** | Breaking change — users with absolute byte limits lose that capability; percentage-based limits are less precise when exact byte control is needed; existing configs and env vars stop working. |

## Comparison Matrix

| Criteria | A (additive) | B (union) | C (string) | D (replace) |
|---|---|---|---|---|
| Backward compatible | Yes | No | No | No |
| Strategy code changes | None | None | None | None |
| Config surface | Two fields | One field | One field | One field |
| Env var clarity | Clear | Ambiguous | Clear | Clear |
| JSON/YAML safe | Yes | No (int/float) | Yes | Yes |
| Parsing complexity | Low | Medium | High | Low |
| Absolute byte support | Yes | Yes | Yes | No |

## Consequences

### Positive

- Users can specify GPU memory limits as a portable fraction, making configs reusable across
  different GPU SKUs.
- Aligns with the mental model of "use X% of my GPU" which is how most users think about memory
  budgets.
- Resolution to bytes is a single point of logic in `OffloadManager`, keeping strategies unchanged.
- Default `0.9` puts all configs in memory mode out of the box, reducing accidental VRAM exhaustion.

### Negative

- Fraction-based limits require a CUDA device to be available at config resolution time (not at
  config construction time), adding a runtime dependency.
- The default behavior change (from `None`/latency mode to `0.9`/memory mode) may surprise users
  upgrading from earlier versions. Documented explicitly in the CHANGELOG.

### Neutral

- The `target_gpu_mem_bytes` field may eventually need the same treatment (fraction support).
- The SHM namespace hash using resolved bytes (not raw fraction) means two processes targeting the
  same fraction on different GPU SKUs will produce different namespaces — correct behavior for
  fleet deployments with mixed hardware.

## References

- `flextensor/config.py` — `OffloadConfig.max_gpu_mem_bytes` field definition
- `flextensor/offload_manager.py` — strategy construction in `_initialize_tensor_manager()`
- `flextensor/strategy/utils.py` — `validate_memory_params()` shared validation
- `torch.cuda.mem_get_info()` / `torch.cuda.get_device_properties()` — runtime GPU memory query
