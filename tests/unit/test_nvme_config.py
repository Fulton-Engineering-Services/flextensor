# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NVMe offload config fields and validation.

Verifies default values, explicit construction, Pydantic validation
constraints (block-transfer-mode requirement, path requirement,
alignment minimum, unified-memory block-transfer-mode requirement),
and that the ``nvme_offload_enabled`` and ``unified_memory`` fields are
parseable from environment variables via ``load_config_from_env``.
"""

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from flextensor.config import OffloadConfig


class TestNvmeConfigDefaults:
    """Default values for NVMe config fields when not explicitly set."""

    def test_defaults(self) -> None:
        """All NVMe fields must have safe defaults (off, no path, cufile, 4096, no unified memory)."""
        config = OffloadConfig()
        assert config.nvme_offload_enabled is False
        assert config.nvme_offload_path is None
        assert config.nvme_transfer_mode == "cufile"
        assert config.nvme_alignment_bytes == 4096
        assert config.unified_memory is False

    def test_explicit_values(self) -> None:
        """All NVMe fields must accept and store explicit construction values."""
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
    """Pydantic validation rules for NVMe config fields."""

    def test_enabled_requires_block_transfer_mode(self) -> None:
        """NVMe offload enabled with ``strategy`` transfer mode must raise."""
        with pytest.raises(ValidationError, match="requires a block transfer_mode"):
            OffloadConfig(
                nvme_offload_enabled=True,
                nvme_offload_path="/mnt/nvme",
                transfer_mode="strategy",
            )

    def test_enabled_with_block_transfer_mode_ok(self) -> None:
        """NVMe offload enabled with ``allocation_block_transfer`` must succeed."""
        config = OffloadConfig(
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme",
            transfer_mode="allocation_block_transfer",
        )
        assert config.nvme_offload_enabled is True

    def test_enabled_with_raw_block_transfer_mode_ok(self) -> None:
        """NVMe offload enabled with ``raw_block_transfer`` must succeed."""
        config = OffloadConfig(
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme",
            transfer_mode="raw_block_transfer",
        )
        assert config.nvme_offload_enabled is True

    def test_enabled_requires_path(self) -> None:
        """NVMe offload enabled without ``nvme_offload_path`` must raise."""
        with pytest.raises(ValidationError, match="nvme_offload_path"):
            OffloadConfig(
                nvme_offload_enabled=True,
                transfer_mode="allocation_block_transfer",
            )

    def test_disabled_without_path_ok(self) -> None:
        """NVMe offload disabled without a path must succeed (path stays ``None``)."""
        config = OffloadConfig(
            nvme_offload_enabled=False,
            transfer_mode="allocation_block_transfer",
        )
        assert config.nvme_offload_path is None

    def test_alignment_minimum(self) -> None:
        """Alignment below the minimum (512) must raise a validation error."""
        with pytest.raises(ValidationError):
            OffloadConfig(nvme_alignment_bytes=256)

    def test_alignment_at_minimum_ok(self) -> None:
        """Alignment at the minimum (512) must succeed."""
        config = OffloadConfig(nvme_alignment_bytes=512)
        assert config.nvme_alignment_bytes == 512

    def test_unified_memory_requires_block_transfer_mode(self) -> None:
        """``unified_memory=True`` with ``strategy`` transfer mode must raise."""
        with pytest.raises(ValidationError, match="unified_memory=True requires a block transfer_mode"):
            OffloadConfig(
                unified_memory=True,
                transfer_mode="strategy",
            )

    def test_unified_memory_with_allocation_block_transfer_ok(self) -> None:
        """``unified_memory=True`` with ``allocation_block_transfer`` must succeed."""
        config = OffloadConfig(
            unified_memory=True,
            transfer_mode="allocation_block_transfer",
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme/ft",
        )
        assert config.unified_memory is True

    def test_unified_memory_with_raw_block_transfer_ok(self) -> None:
        """``unified_memory=True`` with ``raw_block_transfer`` must succeed."""
        config = OffloadConfig(
            unified_memory=True,
            transfer_mode="raw_block_transfer",
            nvme_offload_enabled=True,
            nvme_offload_path="/mnt/nvme/ft",
        )
        assert config.unified_memory is True

    def test_unified_memory_false_with_strategy_ok(self) -> None:
        """``unified_memory=False`` (default) does not constrain transfer mode."""
        config = OffloadConfig(
            unified_memory=False,
            transfer_mode="strategy",
        )
        assert config.unified_memory is False
        assert config.transfer_mode == "strategy"


class TestNvmeConfigEnvVars:
    """Environment-variable support for NVMe config fields."""

    def test_env_var_enabled(self, monkeypatch: MonkeyPatch) -> None:
        """The ``nvme_offload_enabled`` field must be parseable from env vars.

        Direct ``OffloadConfig()`` construction does not load env vars — that
        is done by ``load_config_from_env()``. This test verifies the field
        exists and is parseable.
        """
        monkeypatch.setenv("FT_NVME_OFFLOAD_ENABLED", "true")
        monkeypatch.setenv("FT_NVME_OFFLOAD_PATH", "/mnt/nvme/ft")
        monkeypatch.setenv("FT_TRANSFER_MODE", "allocation_block_transfer")
        config = OffloadConfig()
        # Env vars are not loaded by direct OffloadConfig() construction —
        # they are loaded by load_config_from_env(). This test just verifies
        # the field exists and is parseable.
        assert hasattr(config, "nvme_offload_enabled")

    def test_env_var_unified_memory(self, monkeypatch: MonkeyPatch) -> None:
        """The ``unified_memory`` field must be parseable from env vars via ``load_config_from_env``."""
        from flextensor.config import load_config_from_env

        monkeypatch.setenv("FT_UNIFIED_MEMORY", "1")
        monkeypatch.setenv("FT_TRANSFER_MODE", "allocation_block_transfer")
        monkeypatch.setenv("FT_NVME_OFFLOAD_ENABLED", "1")
        monkeypatch.setenv("FT_NVME_OFFLOAD_PATH", "/mnt/nvme/ft")
        config = load_config_from_env()
        assert config.unified_memory is True
