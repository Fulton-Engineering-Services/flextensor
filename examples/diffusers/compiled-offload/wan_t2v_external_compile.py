# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.1 T2V with external ``torch.compile`` after FlexTensor offload.

**You** call ``torch.compile`` instead of passing ``compile_fn``. Set
``external_compile=True`` so FlexTensor installs compile-transparent
``pre_compute/post_compute`` forwards, run one eager ``pipe(...)`` to reach INFERENCE,
compile each block, then ``request_strategy_replan()`` and one more ``pipe(...)``
to finish the re-plan tail before steady-state serving.
"""

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

import flextensor
from flextensor import OffloadConfig
from flextensor.compiled_offload import bump_dynamo_limits_for_compiled_offload
from flextensor.offload_manager import get_offload_manager

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

for name, component in pipe.components.items():
    if isinstance(component, torch.nn.Module) and name != "transformer":
        component.to("cuda")

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
    external_compile=True,
)

pipe.transformer = flextensor.offload(
    pipe.transformer,
    config=offload_config,
    name="transformer",
)

generate_kwargs = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "num_frames": num_frames,
}

# First run: discovery + profiling + INFERENCE (eager).
pipe(**generate_kwargs)

blocks = pipe.transformer.blocks
# Each block closes over a distinct offload-unit name; Dynamo specializes per name, so
# compiling N blocks needs a recompile limit >= N (PyTorch default is 8).
bump_dynamo_limits_for_compiled_offload(len(blocks))
# Manual compile after INFERENCE — one graph per block (required under offload).
for i in range(len(blocks)):
    blocks[i] = torch.compile(blocks[i], fullgraph=True)

get_offload_manager("transformer").request_strategy_replan()

# Second run: compiled graphs warm up and strategy re-plans (many transformer
# forwards per pipe() call advance the passive tail).
pipe(**generate_kwargs)

# Third run: compiled + offloaded steady state.
frames = pipe(**generate_kwargs).frames[0]
export_to_video(frames, "wan-t2v-external-compile.mp4", fps=16)

flextensor.release("transformer")
