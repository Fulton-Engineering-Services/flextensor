# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.1 T2V with FlexTensor offload + ``torch.compile`` via ``compile_fn``.

Same layout as ``../quickstart/wan_t2v.py``, but each offloaded block is compiled.

"""

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# [FlexTensor] Import FlexTensor modules
import flextensor
from flextensor import OffloadConfig

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

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
    if isinstance(component, torch.nn.Module) and name != "transformer":
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


# [FlexTensor compile] compile_fn (module -> module) is called once per offloaded
# unit, so each becomes its own graph. Swap in Torch-TensorRT, your own tuner, or
# `lambda m: m` (identity — no torch.compile, compiled-offload path still on) here.
def compile_fn(module: torch.nn.Module) -> torch.nn.Module:
    return torch.compile(module, fullgraph=True)


# [FlexTensor] Enable *compiled* offloading on the transformer -- the only API change
# vs plain offload is passing compile_fn.
pipe.transformer = flextensor.offload(
    pipe.transformer,
    config=offload_config,
    name="transformer",
    compile_fn=compile_fn,
)

# [FlexTensor] First run: discovery + compiled view-profile under compile_fn,
# then INFERENCE re-applies compile_fn. Strategy already uses compiled timings
# (no request_strategy_replan on this path).
frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]

# Second run: steady-state compiled + offloaded.
frames = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=num_frames).frames[0]
export_to_video(frames, "wan-t2v-compiled.mp4", fps=16)

# [FlexTensor] Release offload resources (also un-wraps the compiled modules).
flextensor.release("transformer")
