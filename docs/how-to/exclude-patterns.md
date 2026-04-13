<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# How to Exclude Modules and Parameters

FlexTensor supports both `include_patterns` and `exclude_patterns` for fine-grained control over which modules and parameters are offloaded.

## Offload everything except `lm_head`

```python
config = OffloadConfig(
    include_patterns=["*"],
    exclude_patterns=["lm_head"],
)
model = flextensor.offload(model, config=config)
```

## Exclude FP8 scale tensors

```python
config = OffloadConfig(
    include_patterns=["layers.*"],
    exclude_patterns=["*.scale"],
)
```

## Exclude specific layers

```python
config = OffloadConfig(
    include_patterns=["layers.*"],
    exclude_patterns=["layers.31"],
)
```

## Combine include and exclude patterns

Include patterns select which modules to offload. Exclude patterns then remove specific modules or parameters from that set.

```python
config = OffloadConfig(
    include_patterns=["embed", "layers.*", "head"],
    exclude_patterns=["head", "*.norm"],
)
# Result: offloads embed and layers.* but NOT head or any *.norm sub-modules
```

## Configure via environment variables

```bash
FT_INCLUDE_PATTERNS="*" FT_EXCLUDE_PATTERNS="lm_head,*.norm" python serve.py
```

In your code:

```python
config = flextensor.load_config()
model = flextensor.offload(model, config=config)
```

## Configure via YAML

```yaml
include_patterns:
  - "*"
exclude_patterns:
  - "lm_head"
  - "*.scale"
```

## Exclude a sub-module within an offload unit

When a layer is an offload unit, you can exclude specific sub-modules to keep their parameters on GPU:

```python
config = OffloadConfig(
    include_patterns=["layers.*"],
    exclude_patterns=["layers.*.norm"],
)
# Result: layers.0, layers.1, ... are offloaded but norm parameters stay on GPU
```

The exclude pattern `layers.*.norm` is not itself a patched module (it belongs to its parent's offload unit), so the exclusion happens at the parameter level during tensor discovery -- all parameters under `norm` are excluded from offloading.

## Evaluation order

1. `include_patterns` selects candidate modules to offload (default: `["*"]` = all).
2. **Ancestor guard** skips any candidate whose ancestor is already patched, forming offload units.
3. `exclude_patterns` removes offload units from the offloaded set (default: `[]` = exclude nothing). Excludes targeting sub-modules within an offload unit are handled at the parameter level.
4. A target matching both include and exclude is **not** offloaded.

## GPU residency behavior

Excluded modules and parameters are moved to GPU permanently during initialization. They stay on GPU throughout all phases (discovery, profiling, inference) and are never offloaded. This means:

- Excluded tensors consume GPU memory at all times
- They do not participate in the offload scheduling strategy
- Forward passes through excluded modules execute entirely on GPU without tensor transfers
