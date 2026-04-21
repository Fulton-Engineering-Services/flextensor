# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

import os

# Files that must all be present in the local HF cache before we trust it enough
# to enable offline mode. A partial cache left by an HTTP 429-interrupted
# download often contains ``config.json`` but not the tokenizer files, which
# later surfaces deep inside the tokenizer as ``vocab_file=None``. Probing all
# three avoids that false-positive. Safe for every LLM this suite currently
# exercises (all have a fast ``tokenizer.json``).
_REQUIRED_CACHE_FILES = (
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def enable_offline_if_cached(model_name: str) -> None:
    """Set ``HF_HUB_OFFLINE=1`` when *model_name* is fully present in the local HF cache.

    Prevents 429 burst rate-limit crashes from HF API calls for files not
    present in the repo (e.g. quantization configs).  Safe to call multiple
    times -- skips the check once offline mode is already active.

    Each file in :data:`_REQUIRED_CACHE_FILES` must resolve to a cached path;
    if any are missing, the cache is treated as partial and online mode stays
    on so the incomplete download can finish.
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return

    from huggingface_hub import try_to_load_from_cache

    missing = [name for name in _REQUIRED_CACHE_FILES if not isinstance(try_to_load_from_cache(model_name, name), str)]
    if missing:
        print(f"[HF cache] {model_name} not fully cached (missing: {', '.join(missing)}) — HF API calls enabled")
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[HF cache] {model_name} fully cached — set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")
