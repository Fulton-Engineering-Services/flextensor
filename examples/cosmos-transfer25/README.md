<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Cosmos Transfer2.5 Pipeline Weight Streaming

This example runs Cosmos-Transfer2.5 with one FlexTensor manager for the main Cosmos pipeline: diffusion `model.net`, the Qwen/Reason text encoder, the reference-image conditioner, and guardrail modules. The weights stay mostly in host RAM while runtime activations execute on CUDA.

It is intended for memory-constrained single-GPU experiments where the unmodified Cosmos initialization path places too much model state on CUDA before FlexTensor can wrap the pipeline. The example enables a guarded CPU-first construction path in Cosmos, registers pipeline-owned modules that are not normally visible through one `nn.Module` tree, and then switches runtime activations back to CUDA after FlexTensor wraps the composite model.

The single-manager approach is flexible for Cosmos because the pipeline has several independently expensive owners, but they share one execution path and one profile lifecycle. Users can adjust `include_patterns` or `exclude_patterns` in `run_infer.py` to change coverage while keeping one manager name, one state machine, one saved profile, and one cleanup path.

Minimum tested requirements:

- GPU: 32 GB VRAM. This path was validated on an NVIDIA GeForce RTX 5090 with default robot edge frames and a 2-step guarded smoke run.
- Host RAM: about 36 GB observed during the successful run. Use at least 64 GB system RAM for headroom; 128 GB is recommended for profiling and full-step experiments.

## Files

- `run_infer.py`: profile, save-profile, and saved-profile generation entrypoint.
- `Dockerfile`: builds a patched Cosmos image from a public Cosmos checkout plus this FlexTensor checkout.
- `cosmos-transfer25-flextensor.patch`: combined Cosmos compatibility patch for CPU-first construction, runtime device control, and FlexTensor-managed Qwen.

## Build

Clone Cosmos and check out the pinned commit:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git /tmp/cosmos-transfer2.5-ft
cd /tmp/cosmos-transfer2.5-ft
git checkout ce13887925722717a1148ddc46aaca0cf76d4d01
```

Build from the Cosmos checkout and pass the FlexTensor checkout as a named build context:

```bash
docker build \
  -f /path/to/flextensor/examples/cosmos-transfer25/Dockerfile \
  --build-context flextensor=/path/to/flextensor \
  -t cosmos-transfer25-flextensor \
  .
```

The Dockerfile applies the Cosmos patch, installs Cosmos dependencies, installs FlexTensor from `/opt/flextensor`, and copies the runnable example scripts into `/workspace`.
It also installs a small `uvx` wrapper so Cosmos checkpoint downloads that invoke `uvx 'hf>=1.3.5' ...` use the already-installed `hf` CLI instead of resolving a fresh tool environment during inference. Gated Cosmos assets still require a valid `HF_TOKEN`.

## Run Container

```bash
mkdir -p /tmp/cosmos-offload-outputs
docker run -it --gpus all --ipc=host \
  -v /tmp/cosmos-offload-outputs:/outputs \
  -e HF_TOKEN="$HF_TOKEN" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -w /workspace \
  --entrypoint bash \
  cosmos-transfer25-flextensor
```

## Profile Run

Use a short run to create the FlexTensor profile:

```bash
python3 run_infer.py \
  --output-dir /outputs/profile_2step/output \
  --profile-dir /outputs/profile_2step/profile \
  --num-steps 2 \
  --profiling-iters 1
```

The example always uses the composite `cosmos` manager and includes Qwen, conditioner, and guardrail coverage. It defaults to a conservative FlexTensor GPU-memory budget so the 32 GB RTX 5090 smoke has room for Cosmos tokenizer and guardrail activations. Override it with `--max-gpu-mem-fraction` when testing larger GPUs.

During profile generation, FlexTensor runs discovery/profiling passes under `_flextensor_phase/` before saving the profile and logs the offloaded block assignment table. The example uses `class:` patterns for repeated model blocks and path patterns for stable pipeline roots.

## Generation From Saved Profile

Run generation with the saved profile so profiling time is excluded from generation latency:

```bash
python3 run_infer.py \
  --output-dir /outputs/generate_full_from_profile/output \
  --profile-dir /outputs/profile_2step/profile \
  --from-profile
```

## Expected Behavior

CPU-first placement is the reachability unlock for the guarded RTX 5090 path. FlexTensor then manages the Qwen text encoder, diffusion network, conditioner, and guardrail weights from one saved profile with similar latency and lower process RSS/host RAM pressure. Peak VRAM can still be activation/cache dominated, especially in the Wan tokenizer path, so FlexTensor diagnostics are the first check for weight coverage rather than a guarantee of lower peak VRAM.
