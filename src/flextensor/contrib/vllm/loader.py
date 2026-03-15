# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlexTensor-enabled vLLM Model Loader.

Loads models on CPU, then processes weights on GPU layer-by-layer (for CUDA-only ops
like FP8 quantization or MLA attention) before moving them back to CPU for FlexTensor
management. Automatically used by FlexTensorOffloadWorker.
"""

from contextlib import suppress

import torch
from torch import nn
from vllm.attention.layer import Attention, MLAAttention
from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


@register_model_loader("flextensor")
class FlexTensorModelLoader(DefaultModelLoader):
    """Model loader for FlexTensor offloading with 2-phase loading strategy."""

    _UNDERLYING_FORMAT = "auto"

    def _prepare_weights(
        self,
        model_name_or_path: str,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights, mapping flextensor format to auto."""
        original_format = self.load_config.load_format
        if original_format == "flextensor":
            self.load_config.load_format = self._UNDERLYING_FORMAT

        try:
            return super()._prepare_weights(model_name_or_path, revision, fall_back_to_pt, allow_patterns_overrides)
        finally:
            self.load_config.load_format = original_format

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = "") -> nn.Module:
        """Load model with 2-phase strategy: CPU init/load -> GPU weight processing (if needed)."""
        cpu_device = torch.device("cpu")
        gpu_device = vllm_config.device_config.device

        logger.info("FlexTensorModelLoader: Phase 1 - Init and load weights on CPU")

        # Override default device to CPU since current_platform.device_type
        # is sometimes used as default device for tensor creation
        original_default_device = None
        original_platform_device_type = current_platform.device_type
        try:
            current_platform.device_type = "cpu"
            with suppress(Exception):
                original_default_device = torch.get_default_device()
                torch.set_default_device("cpu")
            with set_default_torch_dtype(model_config.dtype):
                model = initialize_model(vllm_config=vllm_config, model_config=model_config, prefix=prefix)
                self.load_weights(model, model_config)
        finally:
            current_platform.device_type = original_platform_device_type
            with suppress(Exception):
                torch.set_default_device(original_default_device)

        logger.info("FlexTensorModelLoader: Phase 2 - Process weights on GPU (if needed)")
        self._process_weights_layer_by_layer(model, model_config, gpu_device, cpu_device)

        return model.eval()

    def _find_layers_needing_gpu_processing(self, model: nn.Module) -> dict[str, nn.Module]:
        """Find decoder layers containing modules that need GPU weight processing."""
        layers_to_process: dict[str, nn.Module] = {}

        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            has_quant = isinstance(quant_method, QuantizeMethodBase)
            has_attn_processing = isinstance(module, (Attention, MLAAttention)) and hasattr(
                module, "process_weights_after_loading"
            )

            if not (has_quant or has_attn_processing):
                continue

            # Find the parent "decoder layer" to process as a unit
            # This ensures dependencies like kv_b_proj are included
            layer_name = self._find_processing_unit(name)
            if layer_name is None:
                continue
            if layer_name not in layers_to_process:
                parent = model
                for part in layer_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                layers_to_process[layer_name] = parent

        return layers_to_process

    def _process_weights_layer_by_layer(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        gpu_device: torch.device,
        cpu_device: torch.device,
    ) -> None:
        """Process weights layer-by-layer on GPU to avoid OOM."""
        layers_to_process = self._find_layers_needing_gpu_processing(model)

        if not layers_to_process:
            logger.info("FlexTensorModelLoader: No layers need GPU processing")
            return

        logger.info("FlexTensorModelLoader: Processing %d layers on GPU", len(layers_to_process))
        processed_count = 0
        for layer_name, layer_module in layers_to_process.items():
            try:
                layer_module.to(gpu_device)
                self._process_layer_weights(layer_module, model_config, gpu_device)
                layer_module.to(cpu_device)
                torch.cuda.empty_cache()
                processed_count += 1
            except Exception as e:
                logger.error("Failed to process layer %s: %s", layer_name, e)
                with suppress(Exception):
                    layer_module.to(cpu_device)
                raise

        logger.info("FlexTensorModelLoader: Processed %d layers", processed_count)

    def _find_processing_unit(self, module_name: str) -> str | None:
        """Find parent decoder layer for module (e.g., 'model.layers.0' for 'model.layers.0.attn')."""
        parts = module_name.split(".")

        # Look for "layers.N" pattern (common in transformer models)
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                # Return "model.layers.N" as the processing unit
                return ".".join(parts[: i + 2])
        return None

    def _process_layer_weights(
        self,
        layer: nn.Module,
        model_config: ModelConfig,
        target_device: torch.device,
    ) -> None:
        """Process weights for a single layer on GPU."""
        for _name, module in layer.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)

            if isinstance(module, (Attention, MLAAttention)) and hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading(model_config.dtype)
