# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# [FlexTensor] Import FlexTensor modules
import flextensor
from flextensor import OffloadConfig

model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

prompt = (
    "A penguin and a rabbit painting a canvas together in a sunny art studio. "
    "The penguin holds a palette of bright colors, while the rabbit carefully "
    "applies brushstrokes with a tiny brush. Paint splatters cover the wooden "
    "floor and their aprons."
)
negative_prompt = (
    "blurry, low resolution, oversaturated, underexposed, distorted anatomy, "
    "extra limbs, missing fingers, poorly drawn face, watermark, text overlay, "
    "static image, grainy, noisy background"
)
num_frames = 65

# [FlexTensor] Instead of pipe.to("cuda"), keep the transformer on CPU for offloading.
# Move all other components (VAE, text encoder, etc.) to GPU.
for name, component in pipe.components.items():
    if isinstance(component, torch.nn.Module) and name not in ("transformer", "transformer_2"):
        component.to("cuda")

# [FlexTensor] Configure weights streaming with proactive prefetching.
# include_patterns selects which submodules to offload (supports wildcards).
include_patterns = [
    "rope",
    "patch_embedding",
    "condition_embedder",
    "blocks.*",
    "norm_out",
    "proj_out",
]

offload_config = OffloadConfig(
    profiling_iters=20,
    min_blocks=2,
    include_patterns=include_patterns,
)

# [FlexTensor] Enable offloading on the transformer(s)
pipe.transformer = flextensor.offload(pipe.transformer, config=offload_config, name="transformer")
pipe.transformer_2 = flextensor.offload(pipe.transformer_2, config=offload_config, name="transformer2")

# [FlexTensor] First run: discovery + profiling pass (learns optimal prefetch schedule)
frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]

# [FlexTensor] Subsequent runs: optimized inference with proactive prefetching
frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
export_to_video(frames, "wan-t2v.mp4", fps=16)

# [FlexTensor] Release offload resources when done
flextensor.release("transformer")
flextensor.release("transformer2")
