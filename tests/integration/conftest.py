# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

import os


def enable_offline_if_cached(model_name: str) -> None:
    """Set ``HF_HUB_OFFLINE=1`` when *model_name* is already in the local HF cache.

    Prevents 429 burst rate-limit crashes from HF API calls for files not
    present in the repo (e.g. quantization configs).  Safe to call multiple
    times — skips the check once offline mode is already active.
    """
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return

    from huggingface_hub import try_to_load_from_cache

    if isinstance(try_to_load_from_cache(model_name, "config.json"), str):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"[HF cache] {model_name} found in local cache — set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1")
    else:
        print(f"[HF cache] {model_name} not cached — HF API calls enabled")
