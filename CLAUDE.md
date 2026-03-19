<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# FlexTensor Development Guidelines

FlexTensor is a PyTorch tensor offloading library for running large models on limited GPU memory by offloading tensors between GPU and CPU.

**Quick Reference**: See `README.md` for API usage | [Dashboard](https://github.com/ai-dynamo/flextensor)

---

## Stack

Python 3.10+ | PyTorch >=2.5 (CUDA required) | beartype (runtime types) | pydantic | psutil | scipy | ruff | mypy strict | pytest

---

## Project Structure

```text
src/flextensor/
├── offload_manager.py        # High-level API (recommended)
├── tensor_manager.py         # Low-level API (advanced)
├── lazy_model_init.py        # Lazy model initialization from profiles
├── state_handler.py          # State persistence/serialization
├── strategy*.py              # Offloading strategies
├── collectors.py             # Statistics collector utilities
├── utils.py                  # Utility functions
├── benchmark_tensor_mode.py  # Profiling state machine
└── shm/                      # Shared memory subsystem (cross-process)
tests/
├── unit/                     # Fast tests, no GPU required
└── integration/L{0,1,2}_*/  # GPU tests by complexity level
```

**Architecture**: `OffloadManager` (singleton) → `TensorManager` → Strategies (`Knapsack`/`Greedy`/`NthLayer`/`Adaptive`/`Global`)

**State Machine**: `NOT_INITIALIZED` → `WARMUP` → `PROFILE` → `INFERENCE`

**Performance Target**: <5% overhead vs baseline

---

## Commands

```bash
# Setup
python3 -m venv --copies .venv && source .venv/bin/activate && pip3 install -e ".[dev]" && pre-commit install

# Testing
make test-unit                           # Fast unit tests (no GPU)
make test-integration                    # GPU integration tests
TEST_LEVEL=L0 make test-integration      # Specific level
make test-integration-isolated           # Isolated venvs per test

# Quality
pre-commit run -a
ruff check . && ruff format .
mypy src/flextensor/

# Debug
LOGLEVEL=DEBUG python script.py
DISABLE_BEARTYPE=1 python -m pdb script.py  # Disable runtime type checking

# Release
git tag -a v1.2.3 -m "Release version 1.2.3" && git push origin v1.2.3
```

### GitHub CLI (gh)

```bash
gh pr list / view <id> / view <id> --comments / diff <id>
gh pr create --fill
gh pr comment <id> --body "<note>"
gh api repos/:owner/:repo/pulls/:pr_number/comments               # inline comments
gh run list / watch <run_id> / view <run_id> --log
```

---

## Code Style

- **Line length**: 120 characters
- **Docstrings**: Google-style; required sections: Args, Returns, Raises, Example
- **License header**: SPDX format in all owned source files:
  ```
  # SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```
  For `.md`/`.html` wrap in `<!-- -->`, for `.css` wrap in `/* */`, for `.json` use `//`. Place after shebang if present.
- **Type hints**: Required everywhere; beartype for runtime (disable via `DISABLE_BEARTYPE=1`)
- **Coverage**: Minimum 80% for `flextensor/`

**Commits**: `feat(scope):` / `fix(scope):` / `docs:` / `test:` / `refactor:` / `chore:` — add `BREAKING CHANGE:` footer for major version bumps

**Branches**: `###-feature-name` or `username/feature-name`

---

## Deprecation

**Removal window**: Deprecated in `vX.Y` → remove in `vX.(Y+1)` minimum. State the removal version explicitly.

Use `@deprecated(msg)` from `typing_extensions` (PEP 702) — handles static analysis (mypy/pyright/IDEs) and runtime `DeprecationWarning`. Add `.. deprecated:: vX.Y` to the docstring (use `v`-prefix to match git tags).

```python
from typing_extensions import deprecated

@deprecated("Use `new_func()` instead. Will be removed in v0.5.")
def old_func(...):
    """Summary.

    .. deprecated:: v0.4
        Use :func:`new_func` instead. Will be removed in v0.5.
    """
    ...
```

- **Attributes**: stack `@deprecated` above `@property` (outermost)
- **Classes**: apply `@deprecated` to the class itself
- **No replacement**: omit "Use X instead." — e.g. "`old_func()` is deprecated and will be removed in v0.5."
- **Custom runtime message**: `@deprecated(msg, category=None)` + `warnings.warn("detail", DeprecationWarning, stacklevel=2)`

**Pydantic fields** (`OffloadConfig` and similar): combine `Annotated` + `Field(deprecated=...)` for both static analysis and runtime warning on access; add `warnings.warn` in the model validator for construction-time warning:

```python
import warnings
from typing import Annotated
from typing_extensions import deprecated as _deprecated

_MY_FIELD_MSG = "`old_field` is deprecated. Use `new_field` instead. Will be removed in vX.Y."

class MyConfig(BaseModel):
    old_field: Annotated[bool, _deprecated(_MY_FIELD_MSG)] = Field(default=False, deprecated=_MY_FIELD_MSG)
    """...\n\n    .. deprecated:: vX.W\n        Use :attr:`new_field` instead. Will be removed in vX.Y.\n    """

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data):
        if isinstance(data, dict) and "old_field" in data and "new_field" not in data:
            warnings.warn(_MY_FIELD_MSG, DeprecationWarning, stacklevel=2)
            data["new_field"] = data["old_field"]
        return data
```

**Process**: `chore(deprecate): mark <X> deprecated, remove in v<Y.Z>` → add to `### Deprecated` in `CHANGELOG.md`. At removal: commit with `BREAKING CHANGE:` footer → move to `### Removed`.

**Tooling**: `typing_extensions` is a transitive dep (no install needed); ruff `B028` and mypy strict already enforce correct usage.

---

## Design Principles

FlexTensor-specific:

- Prefer composition over inheritance for strategy implementations
- Use `Protocol` for type hints, not `ABC`, when runtime checking isn't needed
- Keep strategies focused on single algorithms
- Public API (`__all__`) clearly separated from internals
- Don't add config options until users request them

---

## Integration Tests

Structure: `tests/integration/L{0,1,2}_<name>/` with:
- `.dockerimage` — full image reference (e.g. `nvcr.io/nvidia/vllm:25.10-py3`)
- `requirements.txt` — additional deps beyond `flextensor[test]`
- `test.sh` — test runner script

**GPU memory markers**: `@pytest.mark.gpu_mem_{24g,40g,48g,80g,96g}` — CI auto-generates jobs per tier; unmarked tests run on highest-tier runner.

**test.sh pattern**:
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi >/dev/null 2>&1 || { echo "ERROR: No GPU."; exit 1; }
[ -z "${CI:-}" ] && [ -f "$SCRIPT_DIR/requirements.txt" ] && pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
PYTEST_ARGS=(-v -rA -s --tb=short --maxfail=3 --durations=10)
[ -n "${PYTEST_MARKER:-}" ] && PYTEST_ARGS+=(-m "$PYTEST_MARKER")
timeout "${TIMEOUT:-1800}" python -m pytest "$SCRIPT_DIR" "${PYTEST_ARGS[@]}"
```

---

## Performance

- **Target**: <5% overhead; `(offload_time - baseline_time) / baseline_time`
- Use pinned memory for CPU tensors (default: enabled); CUDA streams for async transfers
- Benchmarks must include: hardware specs, model details, warmup/iteration methodology, confidence intervals

---

## Recent Changes

> See `CHANGELOG.md` or GitHub release tags for full history.

**Key architectural decisions** (as of 2026-02):

- **`max_gpu_mem_fraction` in `OffloadConfig`** (`flextensor/config.py`): Replaces the
  deprecated `max_gpu_mem_bytes`. Accepts `float | None` in (0.0, 1.0] — e.g. `0.9` = 90% of GPU
  memory. Default `0.9` puts all configs in *memory mode* by default; set `None` for *latency mode*
  (no constraint). Resolution to bytes occurs in `OffloadManager._initialize_tensor_manager()` via
  `torch.cuda.get_device_properties()`. The SHM namespace hash uses resolved bytes (not raw
  fraction), so different GPU SKUs hash to different namespaces. Env var:
  `FT_MAX_GPU_MEM_FRACTION`. `max_gpu_mem_bytes` / `FT_MAX_GPU_MEM_BYTES` deprecated → removed
  in v0.5.

- **`load_model_from_profile()`** (`lazy_model_init.py`): Loads models from saved profile — creates params on meta device, buffers on GPU, loads only necessary safetensors weights. More memory-efficient than full model load.

- **State persistence** (`state_handler.py`): `TensorManagerStateHandler` serializes `TensorManager` state as JSON (`save_to_file`/`load_from_file`) or length-prefixed bytes (`save_state_to_bytes`/`load_state_from_bytes`) for SHM buffer sharing. Use `save_profile()` / `load_profile()` to skip warmup+profile on subsequent runs.

- **`init()` pre-initialization**: `flextensor.init()` prepares the tensor manager before `offload()`. Useful with `load_model_from_profile()`. `offload()` calls it internally.

- **Shared memory** (`shm/`): Cross-process tensor coordination via `FlexibleSharedMemory`, `MultiprocessCondition`, `ProcessFileLock`, `SemaphoreLock`, and `ShmCoordinator`. Opt-in via `shm_enabled=True` (default: `False`). SHM namespace computation (`shm/namespace.py`) derives deterministic block names from model path, config fields, and manager name; weight blocks use namespace-derived names (via `weight_block_name()`) rather than PID-based names. Version gating (`shm/coord_block.py`) prevents mismatched FlexTensor versions from sharing memory. `ShmCoordinator` (`shm/coordinator.py`) orchestrates the creator/follower pattern: the first process (creator) writes its profile to a coordination block in SHM and signals readiness; subsequent processes (followers) detect the existing block, validate version compatibility, and read the shared profile.

- **SHM follower initialization** (`offload_manager.py`): `OffloadManager._initialize_from_shm()` enables follower processes to skip warmup and profile, jumping directly from `NOT_INITIALIZED` to `INFERENCE` by loading the profile from shared memory via `ShmCoordinator`. This avoids redundant profiling when multiple processes share the same model.

- **Forward patching**: Patches module `forward()` directly — no proxy wrappers. Preserves model hierarchy, `isinstance` checks, and serialization.

- **`module_patterns` in `OffloadConfig`**: Wildcard patterns (`*`, `?`) specify which modules to offload. Env override: `FT_MODULE_PATTERNS=layers.*,encoder.*`.

- **`VLLM_DEFAULT_MODULE_PATTERNS` in `FlexTensorOffloadWorker`** (`contrib/vllm/worker.py`): When `OffloadConfig.module_patterns == ["*"]` (the global default), the vLLM worker substitutes decoder-only per-layer patterns (`model.embed_tokens`, `model.layers.*`, `model.norm`, `lm_head`, `logits_processor`) instead. This gives each transformer layer its own trap, enabling per-layer pipelining. Override via `FT_MODULE_PATTERNS` for non-standard model layouts.

- **GlobalStrategy optimizer**: Uses `scipy` (`differential_evolution` / `dual_annealing`).

- **`capture_host_resources()`** (`instrumentation/host_resources.py`): Point-in-time snapshot of host physical memory (`host_memory_total`, `host_memory_used`, `host_memory_available`) and swap space (`swap_total`, `swap_used`, `swap_free`) via `psutil`. All values are in bytes. Exported from `flextensor.instrumentation`. The `dump_to_file()` function includes a `host_memory` key in its JSON output alongside `components`.

- **`dump_to_file()` `extra` parameter** (`instrumentation/dumper.py`): `dump_to_file`, `dump_to_directory`, and `dump_instrumentation` accept `*, extra: dict[str, Any] | None = None`. The dict is merged flat into the top-level JSON (no nesting); a `ValueError` is raised if any key collides with built-in keys (`timestamp`, `flex_tensor_version`, `components`, `host_memory`). `OffloadManager` uses this to include `memory_transfer_stats` (tensor size → transfer time map from `TensorManager.get_memory_transfer_stats()`) in the instrumentation dump at inference transition.

- **GPU memory snapshot workers** (`contrib/vllm/snapshot.py`): Opt-in GPU memory snapshot collection during vLLM worker lifecycle. `MemorySnapshotMixin` provides `_take_snapshot(label)` and `_dump_snapshots()`. Each call to `_take_snapshot()` records GPU memory fields (bytes) plus a `host_memory` sub-object (bytes) from `capture_host_resources()`, giving a full system view at every lifecycle stage. `SnapshotWorker` applies the mixin to the standard vLLM `Worker`; `FlexTensorSnapshotWorker` applies it to `FlexTensorOffloadWorker`. All three are exported from `flextensor.contrib.vllm`. Set `FT_VLLM_SNAPSHOT_OUTPUT_DIR` to a directory path to enable JSON dump after the final warmup step; if unset, snapshots are collected in memory but not written to disk.

---

## Common Patterns

```python
# Named manager instances
om = flextensor.get_offload_manager()          # default singleton
om1 = flextensor.get_offload_manager("model1")

# Module patterns
config = flextensor.OffloadConfig(module_patterns=["layers.*", "attention.*"])
flextensor.offload(model, config=config)

# Profile caching
om = flextensor.get_offload_manager()
model = om.offload(model, config=config)
for _ in range(config.warmup_iters + config.profile_iters):
    model(input)
om.save_profile("/tmp/profiles/my_model")

# Load saved profile (skip warmup/profile on next run)
om.set_config(config)
om.load_profile("/tmp/profiles/my_model", model=model)
```

---

## Common Pitfalls

1. **Modify model before `offload()`**, not after — layers added post-patch won't be offloaded
2. **`get_gpu_memory_usage()` requires INFERENCE state** — complete warmup+profile iterations first
3. **Don't mix APIs** — use `get_offload_manager()` only, not direct `TensorManager()` alongside it
4. **Errors**: "No CUDA" → `torch.cuda.is_available()`; "Profile not found" → run warmup; "Module not found" → `model.named_modules()`
5. **Don't share a named manager across threads** — each `get_offload_manager(name)` instance is bound to the thread that created it; accessing the same name from another thread raises `RuntimeError`. Use a distinct name per thread instead.

<!-- MANUAL ADDITIONS START -->

## Running Unit Tests in Claude Code Sessions

The sandbox creates stub files (`HEAD`) that can interfere with `setuptools-scm` version
detection during editable installs. Use `SETUPTOOLS_SCM_PRETEND_VERSION` to bypass git:

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.4.0 UV_CACHE_DIR=$TMPDIR/.cache/uv uv run pytest tests/unit/ -v
```

Or use the venv directly (no version resolution needed):

```bash
SETUPTOOLS_SCM_PRETEND_VERSION=0.4.0 uv pip install -e ".[test]"
.venv/bin/python -m pytest tests/unit/ -v
```

`uv` tries to write to `~/.cache/uv` which is outside the sandbox write allowlist. Setting `UV_CACHE_DIR=$TMPDIR/.cache/uv` redirects the cache to a writable path.
<!-- MANUAL ADDITIONS END -->
