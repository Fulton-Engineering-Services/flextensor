# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SHM namespace computation."""

from unittest.mock import MagicMock, patch

import pytest

from flextensor.config import OffloadConfig
from flextensor.shm.namespace import (
    SHM_PROTOCOL_VERSION,
    apply_rank_suffix,
    compute_shm_namespace,
    coord_block_name,
    profile_block_name,
    weight_block_name,
)


class TestComputeShmNamespace:
    """Tests for compute_shm_namespace()."""

    def test_deterministic_same_inputs(self):
        """Same inputs produce same namespace."""
        config = OffloadConfig(include_patterns=["layers.*"])
        ns1 = compute_shm_namespace("/models/qwen", config)
        ns2 = compute_shm_namespace("/models/qwen", config)
        assert ns1 == ns2

    def test_different_model_path_different_namespace(self):
        """Different model paths produce different namespaces."""
        config = OffloadConfig()
        ns1 = compute_shm_namespace("/models/qwen-7b", config)
        ns2 = compute_shm_namespace("/models/qwen-70b", config)
        assert ns1 != ns2

    def test_different_config_different_namespace(self):
        """Different config fields produce different namespaces."""
        c1 = OffloadConfig(include_patterns=["layers.*"])
        c2 = OffloadConfig(include_patterns=["attention.*"])
        ns1 = compute_shm_namespace("/models/qwen", c1)
        ns2 = compute_shm_namespace("/models/qwen", c2)
        assert ns1 != ns2

    def test_different_exclude_patterns_different_namespace(self):
        """Different exclude_patterns produce different namespaces."""
        c1 = OffloadConfig(include_patterns=["*"], exclude_patterns=["lm_head"])
        c2 = OffloadConfig(include_patterns=["*"], exclude_patterns=["lm_head", "*.norm"])
        ns1 = compute_shm_namespace("/models/qwen", c1)
        ns2 = compute_shm_namespace("/models/qwen", c2)
        assert ns1 != ns2

    def test_empty_exclude_patterns_differs_from_nonempty(self):
        """Empty exclude_patterns produces a different namespace than non-empty."""
        c1 = OffloadConfig(include_patterns=["*"], exclude_patterns=[])
        c2 = OffloadConfig(include_patterns=["*"], exclude_patterns=["lm_head"])
        ns1 = compute_shm_namespace("/models/qwen", c1)
        ns2 = compute_shm_namespace("/models/qwen", c2)
        assert ns1 != ns2

    def test_different_extra_keys_different_namespace(self):
        """Different vLLM config produces different namespace."""
        config = OffloadConfig()
        ns1 = compute_shm_namespace("/models/qwen", config, extra_keys={"quantization": "fp8"})
        ns2 = compute_shm_namespace("/models/qwen", config, extra_keys={"quantization": "awq"})
        assert ns1 != ns2

    def test_extra_keys_none_vs_empty(self):
        """None and empty extra_keys produce same result."""
        config = OffloadConfig()
        ns1 = compute_shm_namespace("/models/qwen", config, extra_keys=None)
        ns2 = compute_shm_namespace("/models/qwen", config, extra_keys={})
        assert ns1 == ns2

    def test_namespace_format(self):
        """Namespace starts with ft_ prefix and has 8 hex chars."""
        config = OffloadConfig()
        ns = compute_shm_namespace("/models/qwen", config)
        assert ns.startswith("ft_")
        assert len(ns) == 3 + 8  # "ft_" + 8 hex chars

    def test_shm_namespace_override(self):
        """Explicit shm_namespace in config overrides computed value."""
        config = OffloadConfig(shm_namespace="my_custom_ns")
        ns = compute_shm_namespace("/models/qwen", config)
        assert ns == "my_custom_ns"

    def test_different_manager_name_different_namespace(self):
        """Different manager names produce different namespaces."""
        config = OffloadConfig()
        ns1 = compute_shm_namespace("/models/qwen", config, manager_name="model_a")
        ns2 = compute_shm_namespace("/models/qwen", config, manager_name="model_b")
        assert ns1 != ns2

    def test_same_manager_name_deterministic(self):
        """Same manager name produces same namespace across calls."""
        config = OffloadConfig()
        ns1 = compute_shm_namespace("/models/qwen", config, manager_name="my_model")
        ns2 = compute_shm_namespace("/models/qwen", config, manager_name="my_model")
        assert ns1 == ns2

    def test_default_manager_name_matches_no_arg(self):
        """Omitting manager_name is equivalent to passing 'default'."""
        config = OffloadConfig()
        ns1 = compute_shm_namespace("/models/qwen", config)
        ns2 = compute_shm_namespace("/models/qwen", config, manager_name="default")
        assert ns1 == ns2

    def test_protocol_version_is_integer(self):
        """SHM_PROTOCOL_VERSION is a positive integer."""
        assert isinstance(SHM_PROTOCOL_VERSION, int)
        assert SHM_PROTOCOL_VERSION >= 1

    def test_same_fraction_different_total_mem_different_namespace(self):
        """Same fraction on different GPU SKUs produces different namespaces."""
        config = OffloadConfig(max_gpu_mem_fraction=0.9)

        props_40g = MagicMock()
        props_40g.total_memory = 40 * 1024**3
        props_80g = MagicMock()
        props_80g.total_memory = 80 * 1024**3

        with patch("flextensor.utils.torch.cuda.get_device_properties", return_value=props_40g):
            ns_40g = compute_shm_namespace("/models/qwen", config)

        with patch("flextensor.utils.torch.cuda.get_device_properties", return_value=props_80g):
            ns_80g = compute_shm_namespace("/models/qwen", config)

        assert ns_40g != ns_80g

    def test_same_fraction_same_total_mem_same_namespace(self):
        """Same fraction on identical GPUs produces the same namespace."""
        config = OffloadConfig(max_gpu_mem_fraction=0.9)
        props = MagicMock()
        props.total_memory = 80 * 1024**3

        with patch("flextensor.utils.torch.cuda.get_device_properties", return_value=props):
            ns1 = compute_shm_namespace("/models/qwen", config)
            ns2 = compute_shm_namespace("/models/qwen", config)

        assert ns1 == ns2

    def test_deprecated_bytes_path_no_gpu_query(self):
        """Deprecated max_gpu_mem_bytes path uses raw bytes in hash, no GPU query."""
        with pytest.warns(DeprecationWarning):
            config = OffloadConfig(max_gpu_mem_bytes=20 * 1024**3)

        with patch("flextensor.utils.torch.cuda.get_device_properties") as mock_props:
            ns = compute_shm_namespace("/models/qwen", config)

        mock_props.assert_not_called()
        assert ns.startswith("ft_")

    def test_latency_mode_no_gpu_query(self):
        """max_gpu_mem_fraction=None (latency mode) produces a valid namespace, no GPU query."""
        config = OffloadConfig(max_gpu_mem_fraction=None)

        with patch("flextensor.utils.torch.cuda.get_device_properties") as mock_props:
            ns = compute_shm_namespace("/models/qwen", config)

        mock_props.assert_not_called()
        assert ns.startswith("ft_")


class TestRankSuffix:
    """Tests for rank-scoped namespace suffix."""

    def test_tp_pp_suffix(self):
        """Rank suffix includes tp and pp."""
        ns = apply_rank_suffix("ft_abc12345", tp_rank=0, pp_rank=0)
        assert ns == "ft_abc12345_tp0_pp0"

    def test_tp_pp_ep_suffix(self):
        """Rank suffix includes ep when provided."""
        ns = apply_rank_suffix("ft_abc12345", tp_rank=1, pp_rank=0, ep_rank=2)
        assert ns == "ft_abc12345_tp1_pp0_ep2"

    def test_different_tp_rank_different_namespace(self):
        """Different TP ranks get different namespaces."""
        ns1 = apply_rank_suffix("ft_abc12345", tp_rank=0, pp_rank=0)
        ns2 = apply_rank_suffix("ft_abc12345", tp_rank=1, pp_rank=0)
        assert ns1 != ns2


class TestBlockNames:
    """Tests for SHM block name helpers."""

    def test_weight_block_name(self):
        """Weight block name includes index."""
        assert weight_block_name("ft_abc_tp0_pp0", 0) == "ft_abc_tp0_pp0_w0"
        assert weight_block_name("ft_abc_tp0_pp0", 3) == "ft_abc_tp0_pp0_w3"

    def test_profile_block_name(self):
        """Profile block name uses _prof suffix."""
        assert profile_block_name("ft_abc_tp0_pp0") == "ft_abc_tp0_pp0_prof"

    def test_coord_block_name(self):
        """Coord block name uses _crd suffix."""
        assert coord_block_name("ft_abc_tp0_pp0") == "ft_abc_tp0_pp0_crd"
