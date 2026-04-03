# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile the Wan2.2 transformer and save the offload profile to disk.

Run this script once to generate the profile. Subsequent inference runs
can load the saved profile with ``run_infer.py`` and skip profiling entirely.

Note: This file is intentionally named ``run_profile.py`` (not ``profile.py``)
to avoid shadowing Python's stdlib ``profile`` module.

Usage:
    python run_profile.py [--profile-dir ./wan_profile]
"""

import argparse

import torch
from diffusers import WanPipeline

import flextensor
from flextensor import OffloadConfig

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODULE_PATTERNS = [
    "rope",
    "patch_embedding",
    "condition_embedder",
    "blocks.*",
    "norm_out",
    "proj_out",
]


def main():
    parser = argparse.ArgumentParser(description="Profile Wan2.2 transformer for FlexTensor offloading")
    parser.add_argument(
        "--profile-dir",
        default="./wan_profile",
        help="Base directory for offload profiles (default: ./wan_profile)",
    )
    args = parser.parse_args()

    transformer_profile_dir = f"{args.profile_dir}/transformer"
    transformer2_profile_dir = f"{args.profile_dir}/transformer2"

    pipe = WanPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    for name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module) and name not in ("transformer", "transformer_2"):
            component.to("cuda")

    offload_config = OffloadConfig(
        profile_iters=20,
        min_blocks=2,
        include_patterns=MODULE_PATTERNS,
    )

    pipe.transformer = flextensor.offload(pipe.transformer, config=offload_config, name="transformer")
    pipe.transformer_2 = flextensor.offload(pipe.transformer_2, config=offload_config, name="transformer2")

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

    pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=65)

    flextensor.save_profile(transformer_profile_dir, name="transformer")
    flextensor.save_profile(transformer2_profile_dir, name="transformer2")
    print(f"Profiles saved to {transformer_profile_dir}/ and {transformer2_profile_dir}/")  # noqa: T201

    flextensor.release("transformer")
    flextensor.release("transformer2")


if __name__ == "__main__":
    main()
