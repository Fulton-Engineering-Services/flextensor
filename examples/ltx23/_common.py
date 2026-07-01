# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the LTX 2.3 FlexTensor examples."""

from __future__ import annotations

import json
import subprocess  # noqa: S404 - used only for local nvidia-smi diagnostics in these examples.
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download

LTX_LICENSE_URL = "https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE"


def gpu_memory_snapshot() -> dict[str, Any]:
    """Collect lightweight GPU memory diagnostics for response metadata."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader,nounits"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception as exc:
        return {"error": repr(exc)}

    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "used_mib": int(parts[2]),
                "free_mib": int(parts[3]),
            })
    return {
        "gpus": gpus,
        "torch_allocated_mib": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0,
        "torch_reserved_mib": torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else 0.0,
    }


def resolve_file(path: str | None, repo_id: str, filename: str, cache_dir: str | None) -> str:
    """Return a local path or download a single file from Hugging Face Hub."""
    if path:
        return path
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def resolve_snapshot(path: str | None, repo_id: str, cache_dir: str | None) -> str:
    """Return a local snapshot path or download a repository snapshot from Hugging Face Hub."""
    if path:
        return path
    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir)


def require_external_license_ack(args: Any, repos_to_download: list[str]) -> None:
    """Require explicit acknowledgement before resolving external artifacts from Hugging Face."""
    if not repos_to_download or args.accept_external_licenses:
        return

    repos = ", ".join(sorted(set(repos_to_download)))
    raise SystemExit(
        "Automatic Hugging Face downloads require --accept-external-licenses. "
        "Review and comply with upstream model terms before downloading these artifacts. "
        f"LTX terms: {LTX_LICENSE_URL}. Repositories that would be downloaded: {repos}. "
        "To avoid automatic downloads, pass local paths with --distilled-checkpoint-path, "
        "--spatial-upsampler-path, --gemma-root, and --lora-path."
    )


def payload_value(payload: dict[str, Any], key: str, default: Any) -> Any:
    """Return a request value unless the key is missing or explicitly null."""
    if key not in payload or payload[key] is None:
        return default
    return payload[key]


def decode_json_payload(raw_body: bytes) -> dict[str, Any]:
    """Decode an HTTP request body as a JSON object payload."""
    payload = json.loads(raw_body or b"{}")
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object payload.")
    return payload
