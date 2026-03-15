<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# ADR-0002: Configuration-based Module Patterns for Offloading

**Date**: 2026-01-23

**Status**: Accepted

## Context

FlexTensor needs to identify which modules in a model should be offloaded. The original API required users to explicitly pass a list of module paths to the `offload()` function:

```python
model = om.offload(model, ["embed", "layers.*", "head"])
```

This approach had several issues:

1. **Inconsistent configuration**: Module paths were specified via function parameter while all other settings (GPU device, warmup iterations, etc.) were configured via `OffloadConfig`. This created an inconsistent API where some configuration came from the config object and some from function arguments.

2. **No environment variable support**: Users couldn't configure module patterns via environment variables like other settings (`FT_ENABLED`, `FT_GPU_DEVICE`, etc.). This limited deployment flexibility, especially in containerized environments where environment variables are the preferred configuration method.

3. **vLLM integration complexity**: The `FlexTensorOffloadWorker` needed to determine appropriate module patterns for each model architecture. Without a unified configuration mechanism, this logic was scattered and hard to maintain.

4. **Config file support**: Module patterns couldn't be loaded from configuration files (YAML, JSON, INI), requiring code changes to modify which modules are offloaded.

The FlexTensor configuration system already supports loading from multiple sources with clear precedence: file < environment variables < kwargs. Module patterns should follow the same pattern.

## Decision

We will **move module patterns into `OffloadConfig`** as a first-class configuration field, removing the `module_paths` parameter from the `offload()` API.

Implementation approach:

1. **Add `module_patterns` field to `OffloadConfig`**: A list of module path patterns with wildcard support, defaulting to `["*"]` (offload all modules)

```python
class OffloadConfig(BaseModel):
    # ... existing fields ...
    module_patterns: list[str] = Field(
        default_factory=lambda: ["*"],
        description="List of module path patterns to offload (supports wildcards)",
    )
```

2. **Support environment variable configuration**: Add `FT_MODULE_PATTERNS` as a comma-separated list of patterns

```bash
FT_MODULE_PATTERNS="model.layers.*,lm_head,embed_tokens" vllm serve ...
```

3. **Support config file loading**: Module patterns can be specified in YAML, JSON, or INI files

```yaml
# flextensor.yaml
enabled: true
module_patterns:
  - model.layers.*
  - lm_head
  - embed_tokens
```

4. **Simplify `offload()` API**: Remove `module_paths` parameter; patterns come from `config.module_patterns`

```python
# Before
model = om.offload(model, ["layers.*", "head"], config)

# After
config = OffloadConfig(module_patterns=["layers.*", "head"])
model = om.offload(model, config)
```

5. **Use `*` as default**: When no patterns are specified, default to `["*"]` which matches all modules. This provides a reasonable default for quick experimentation.

## Alternatives Considered

### Alternative 1: Keep module_paths as Separate Parameter

**Approach**: Maintain the current API where module paths are passed as a separate argument to `offload()`.

**Pros**:
- No API change required
- Explicit separation between "what to offload" and "how to offload"
- Backward compatible

**Cons**:
- Inconsistent with other configuration (all settings in config except this one)
- No environment variable support for patterns
- Cannot load patterns from config files
- Users must modify code to change offload targets

**Why rejected**: The inconsistency with the rest of the configuration system creates a poor user experience. Users expect all FlexTensor settings to be configurable via environment variables and config files.

### Alternative 2: Auto-detect Module Patterns from Model Architecture

**Approach**: Automatically detect and select appropriate modules based on model architecture introspection (e.g., find transformer layers, embeddings, heads).

**Pros**:
- Zero-configuration experience for common models
- No need to understand model internals
- Adapts automatically to new architectures

**Cons**:
- Complex heuristics that may not work for all models
- Magic behavior that's hard to debug
- May select too many or too few modules
- Different model families use different naming conventions
- Adds maintenance burden for heuristics

**Why rejected**: While appealing in theory, automatic detection is fragile and hard to debug. When it works, it's convenient; when it doesn't, users have no recourse. Configuration-based approach gives users explicit control.

### Alternative 3: Decorator-based Module Marking

**Approach**: Users mark modules to offload using decorators or model attributes.

```python
class MyModel(nn.Module):
    @flextensor.offloadable
    def __init__(self):
        self.layers = OffloadableModuleList([...])
```

**Pros**:
- Explicit marking in model definition
- No configuration needed at runtime
- Self-documenting

**Cons**:
- Requires model code modification
- Cannot use with third-party models (vLLM, HuggingFace)
- Not compatible with pre-trained model loading
- Intrusive API

**Why rejected**: FlexTensor must work with existing models without modification. Most users work with pre-trained models from HuggingFace or other sources that cannot be modified.

### Alternative 4: Configuration-based Module Patterns (Selected)

**Approach**: Add `module_patterns` as a field in `OffloadConfig`, loadable from environment variables and config files.

**Pros**:
- Consistent with existing configuration system
- Environment variable support (`FT_MODULE_PATTERNS`)
- Config file support (YAML, JSON, INI)
- Explicit user control
- Works with any model without modification
- Easy to experiment (change env var, not code)
- Follows twelve-factor app principles for configuration

**Cons**:
- API change (removal of `module_paths` parameter)
- Users must learn pattern syntax
- Default `["*"]` may not be optimal for all cases

**Why selected**: This approach provides the best balance of flexibility, consistency, and usability. It integrates naturally with the existing configuration system and follows established patterns for application configuration.

## Consequences

### Positive

- **Unified configuration**: All FlexTensor settings are now in `OffloadConfig`, including module patterns
- **Environment variable support**: Deploy different configurations via `FT_MODULE_PATTERNS` without code changes
- **Config file support**: Define patterns in YAML/JSON/INI files for version control and reproducibility
- **Simplified API**: `offload()` takes only model and optional config, reducing argument complexity
- **Container-friendly**: Easily configure via environment variables in Kubernetes, Docker, etc.
- **Consistent precedence**: Module patterns follow the same file < env < kwargs precedence as other settings

### Negative

- **Breaking API change**: Existing code using `offload(model, module_paths)` must be updated to use `OffloadConfig(module_patterns=...)`. This is mitigated by:
  - The library is pre-1.0 and breaking changes are expected
  - The change is straightforward to migrate
  - Error messages guide users to the new API
- **Default behavior change**: Default `["*"]` offloads all modules, which may surprise users expecting no offloading by default. However, this is appropriate since:
  - `offload()` is explicitly called when offloading is desired
  - Users calling `offload()` expect something to be offloaded

### Neutral

- **Pattern syntax unchanged**: The wildcard pattern syntax (`layers.*`, `model.*.attention`) remains the same, just configured differently
- **vLLM integration unchanged**: `FlexTensorOffloadWorker` continues to work, now using `FT_MODULE_PATTERNS` for custom patterns

## References

### Internal Code References

- Configuration implementation: [`flextensor/config.py::OffloadConfig.module_patterns`](../../src/flextensor/config.py)
- Environment variable parsing: [`flextensor/config.py::_load_from_env`](../../src/flextensor/config.py)
- vLLM worker integration: [`flextensor/contrib/vllm/worker.py::FlexTensorOffloadWorker`](../../src/flextensor/contrib/vllm/worker.py)
- Unit tests: [`tests/unit/test_config.py`](../../tests/unit/test_config.py)

### External References

- Twelve-factor app configuration principles: https://12factor.net/config
- Pydantic Field documentation: https://docs.pydantic.dev/latest/concepts/fields/
