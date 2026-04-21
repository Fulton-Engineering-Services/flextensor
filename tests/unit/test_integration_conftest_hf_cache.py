# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``enable_offline_if_cached`` in ``tests/integration/conftest.py``."""

from __future__ import annotations

import os
from unittest import mock

import pytest

pytest.importorskip("huggingface_hub")

from tests.integration.conftest import _REQUIRED_CACHE_FILES, enable_offline_if_cached  # noqa: E402

_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(autouse=True)
def _restore_hf_env() -> None:
    before = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    yield
    for k, v in before.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_enables_offline_when_all_required_files_are_cached(capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch("huggingface_hub.try_to_load_from_cache", return_value="/fake/path") as m:
        enable_offline_if_cached(_MODEL)

    assert m.call_count == len(_REQUIRED_CACHE_FILES)
    assert {c.args[1] for c in m.call_args_list} == set(_REQUIRED_CACHE_FILES)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert "fully cached" in capsys.readouterr().out


def test_stays_online_when_any_required_file_is_missing(capsys: pytest.CaptureFixture[str]) -> None:
    missing_file = "tokenizer.json"

    def fake_load(_model: str, filename: str) -> str | None:
        return None if filename == missing_file else "/fake/path"

    with mock.patch("huggingface_hub.try_to_load_from_cache", side_effect=fake_load):
        enable_offline_if_cached(_MODEL)

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    out = capsys.readouterr().out
    assert "not fully cached" in out
    assert missing_file in out


def test_noop_when_already_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    with mock.patch("huggingface_hub.try_to_load_from_cache") as m:
        enable_offline_if_cached(_MODEL)

    m.assert_not_called()
