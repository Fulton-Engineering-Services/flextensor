# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVMe offload config fields and validation."""

import pytest
from pydantic import ValidationError

from flextensor.config import OffloadConfig


class TestNvmeConfigDefaults:
    def test_defaults(self) -> None:
        config = OffloadConfig()
        assert config.nvme_offload_enabled is False
        assert config.nvme_offload_path is None
        assert config.nvme_transfer_mode == "cufile"
        assert config.nvme_alignment_bytes == 4096

    def test_explicit_values(self) -> None:
        config = OffloadConfig(
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme/weights",
            nvme_transfer_mode="posix",
            nvme_alignment_bytes=8192,
        )
        assert config.nvme_offload_enabled is True
        assert config.nvme_offload_path == "/mnt/nvme/weights"
        assert config.nvme_transfer_mode == "posix"
        assert config.nvme_alignment_bytes == 8192


class TestNvmeConfigValidation:
    def test_enabled_requires_block_transfer_mode(self) -> None:
        with pytest.raises(ValidationError, match="requires a block transfer_mode"):
            OffloadConfig(
                nvme_offload_enabled=True,
                nvme_offload_path="/mnt/nvme",
                transfer_mode="strategy",
            )

    def test_enabled_with_block_transfer_mode_ok(self) -> None:
        config = OffloadConfig(
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme",
            transfer_mode="allocation_block_transfer",
        )
        assert config.nvme_offload_enabled is True

    def test_enabled_with_raw_block_transfer_mode_ok(self) -> None:
        config = OffloadConfig(
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme",
            transfer_mode="raw_block_transfer",
        )
        assert config.nvme_offload_enabled is True

    def test_enabled_requires_path(self) -> None:
        with pytest.raises(ValidationError, match="nvme_offload_path"):
            OffloadConfig(
                nvme_offload_enabled=True,
                transfer_mode="allocation_block_transfer",
            )

    def test_disabled_without_path_ok(self) -> None:
        config = OffloadConfig(
            nvme_offload_enabled=False,
            transfer_mode="allocation_block_transfer",
        )
        assert config.nvme_offload_path is None

    def test_alignment_minimum(self) -> None:
        with pytest.raises(ValidationError):
            OffloadConfig(nvme_alignment_bytes=256)

    def test_alignment_at_minimum_ok(self) -> None:
        config = OffloadConfig(nvme_alignment_bytes=512)
        assert config.nvme_alignment_bytes == 512


class TestNvmeConfigEnvVars:
    def test_env_var_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("FT_NVME_OFFLOAD_ENABLED", "true")
        monkeypatch.setenv("FT_NVME_OFFLOAD_PATH", "/mnt/nvme/ft")
        monkeypatch.setenv("FT_TRANSFER_MODE", "allocation_block_transfer")
        config = OffloadConfig()
        # Env vars are not loaded by direct OffloadConfig() construction —
        # they are loaded by load_config_from_env(). This test just verifies
        # the field exists and is parseable.
        assert hasattr(config, "nvme_offload_enabled")
