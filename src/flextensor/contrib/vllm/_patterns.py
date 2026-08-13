# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in tensor-selection policy shared by FlexTensor vLLM integrations."""

from typing import Any

from flextensor.utils import config_field_was_set

# Default patterns for per-layer offloading in decoder-only and text-only
# hybrid-wrapper vLLM models.
# Each transformer layer gets its own trap, enabling the prefetch pipeline to
# overlap CPU→GPU transfers with GPU compute for subsequent layers.
# Override via FT_INCLUDE_PATTERNS env var or OffloadConfig(include_patterns=[...]).
VLLM_DEFAULT_INCLUDE_PATTERNS: list[str] = [
    # Prefer class-based layer selection so model path changes across vLLM
    # versions do not collapse the worker back to one coarse model trap.
    "class:*DecoderLayer",
    "class:*DecoderBlock",
    "class:*TransformerBlock",
    "model.embed_tokens",
    "model.norm",
    # Nemotron-H uses ``model.norm_f`` for the final decoder norm.
    "model.norm_f",
    "lm_head",
    "logits_processor",
    # Qwen3.5/3.6 hybrid wrappers keep the causal LM under
    # ``language_model`` even in text-only serving mode.
    "language_model.model.embed_tokens",
    "language_model.model.norm",
    "language_model.lm_head",
    "language_model.logits_processor",
]

# Some MoE sidecars can contain CUDA-only kernels or small router/gating tensors
# that are safer kept GPU-resident while the routed expert bulk remains
# available to FlexTensor. Use class-based excludes when the class identifies
# only the sidecar; keep name patterns for shared classes and parameter-level
# exclusions. Non-matching patterns are harmless for other architectures.
VLLM_DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    "class:GateLinear",
    "model.layers.*.mixer.shared_experts",
    "model.layers.*.mixer.fc1_latent_proj",
    "model.layers.*.mixer.fc2_latent_proj",
    # Qwen3.5/3.6 shared experts are invoked by vLLM's FusedMoE runner
    # side path, including a separate CUDA stream, so keep them resident.
    "model.layers.*.mlp.shared_expert",
    "language_model.model.layers.*.mlp.shared_expert",
    # Qwen3.5/3.6 MoE router/gating tensors are tiny compared with expert
    # weights and are better kept resident while expert linears are offloaded.
    "language_model.model.layers.*.mlp.gate",
    "language_model.model.layers.*.mlp.shared_expert_gate",
    # Qwen3.5/3.6 GDN linear-attention kernels read these parameters inside
    # vLLM custom ops, outside the normal torch function argument rewrite path.
    "language_model.model.layers.*.linear_attn.A_log",
    "language_model.model.layers.*.linear_attn.dt_bias",
]


def resolve_vllm_patterns(config: Any) -> tuple[list[str], list[str]]:
    """Resolve public selectors while preserving an explicit empty exclude list."""
    includes = VLLM_DEFAULT_INCLUDE_PATTERNS if config.include_patterns == ["*"] else config.include_patterns
    excludes = config.exclude_patterns
    if not excludes and not config_field_was_set(config, "exclude_patterns"):
        excludes = VLLM_DEFAULT_EXCLUDE_PATTERNS
    return list(includes), list(excludes)
