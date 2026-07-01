<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# LTX 2.3 Outpaint IC-LoRA Pipeline Weight Streaming

This example serves the [`oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint`](https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint) IC-LoRA on top of the official two-stage `ICLoraPipeline`, with one FlexTensor manager per LTX diffusion transformer stage, plus an optional manager for the Gemma text encoder.

The Outpaint IC-LoRA extends the canvas of an input video: you letterbox the source to the target canvas with pure-black bars in the regions you want to fill, and the model paints those regions with content that is visually and temporally consistent with the original footage. It exercises FlexTensor on the default LTX-2.3 two-stage (latent upscaling) path.

The example is intended for memory-constrained single-GPU serving experiments. It separates startup/profile cost from request latency:

- `profile`: builds the LTX transformers (and, with `--offload-text`, the Gemma text encoder) on CPU, wraps each in its own FlexTensor manager, drives discovery/profiling passes, and saves one profile per manager.
- `serve`: loads the profiles, keeps the FlexTensor-managed models resident, and serves local HTTP POST requests.

The example assumes Hugging Face authentication and cache handling are already configured in the environment. It does not include a native/vanilla code path and does not include benchmark artifacts.

The default model artifacts are listed in [`EXTERNAL_MATERIALS.md`](../../../EXTERNAL_MATERIALS.md). These artifacts are not distributed with FlexTensor and are governed by their upstream terms. In particular, LTX artifacts are governed by the [LTX-2 Community License Agreement](https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE), which includes commercial-use and acceptable-use restrictions. Review and comply with the applicable upstream terms before downloading or running the default artifacts.

If any default repository is unavailable or unsuitable for your use case, replace it with CLI flags: `--distilled-checkpoint-repo`, `--spatial-upsampler-repo`, `--gemma-repo`, and `--lora-repo` select alternate Hugging Face repositories, while `--distilled-checkpoint-path`, `--spatial-upsampler-path`, `--gemma-root`, and `--lora-path` use local artifacts directly.

Related upstream references:

- [LTX-2.3 base model](https://huggingface.co/Lightricks/LTX-2.3)
- [LTX-2.3 Outpaint IC-LoRA](https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint)
- [Gemma text encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
- [Official `ICLoraPipeline` source](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)

## Files

- `serve_infer.py`: FlexTensor-only profile and serving entrypoint.
- `single_serve.sh`: single-GPU serving helper for an existing FlexTensor profile.
- `nginx_serve.sh`: single-node multi-GPU serving helper for an existing FlexTensor profile.
- `setup.sh`: environment setup helper for the LTX-2 checkout and FlexTensor install.
- `letterbox.py`: helper to pad a source video onto the target canvas with pure-black bars.

## How outpaint works

The IC-LoRA was trained with pure black pixels (RGB 0,0,0) as the sentinel for the region to generate. At inference you letterbox the source video to the target canvas with black bars on the sides / top / bottom you want to extend, and the model fills those regions. This example takes an already-letterboxed conditioning video and passes it to the pipeline as the IC-LoRA `video_conditioning`.

### Preparing the letterboxed input

Use `letterbox.py` to pad a raw source video onto the target canvas without rescaling the original content (the source frames are placed verbatim; the extension regions are filled with exact black). Specify either per-side margins or an explicit canvas:

```bash
# Margin mode: add 320 px of black on the left and right (extend horizontally).
/workspace/LTX-2/.venv/bin/python \
  /path/to/flextensor/examples/ltx23/outpaint/letterbox.py \
  --input /workspace/inputs/source.mp4 \
  --output /workspace/inputs/letterboxed_720p.mp4 \
  --left 320 --right 320

# Canvas mode: place the source centered on an explicit 1280x704 canvas.
/workspace/LTX-2/.venv/bin/python \
  /path/to/flextensor/examples/ltx23/outpaint/letterbox.py \
  --input /workspace/inputs/source.mp4 \
  --output /workspace/inputs/letterboxed_720p.mp4 \
  --width 1280 --height 704
```

The resulting canvas must be a multiple of 64 (e.g. `1280x704` for 720p); `letterbox.py` fails fast otherwise (use `--no-validate` to downgrade to a warning). Pass the same dimensions as `--height`/`--width` to `profile`/`serve`. The dark-scene gamma round-trip is handled separately by `serve_infer.py` (`--gamma`).

### Optional dark-scene gamma round-trip

Because pure black is the "generate here" signal, very dark source footage can be ambiguous and the bars may be left un-generated. The fix is a gamma round-trip, available via `--gamma` (default `1.0` = disabled, `2.0` matches the model card):

- Before generation, the letterboxed input is brightened with a per-channel RGB gamma (`out = (in/255) ** (1/g) * 255`, via ffmpeg `lutrgb`). Real dark content lifts into clearly-colored territory while the pure-black bars stay pure black (0 is an exact fixed point of the curve).
- After generation, the output is darkened by the exact inverse (gamma `1/g`) so the whole frame returns to the original exposure.

This matches the author's ComfyUI `Color Correct (mtb)` node, which also works in full-range RGB. The curve is exactly invertible, so the round-trip is lossless on continuous values (8-bit only adds <=~4/255 quantization), and the forward pass is encoded losslessly to keep the bars pure black. Note we deliberately do **not** use ffmpeg's `eq=gamma`: `eq` operates on limited-range luma (black is `Y=16`), so brightening lifts the bars to ~56/255 in the file the model consumes, destroying the "generate here" sentinel. For pixel-exact bars on heavily compressed sources, prefer preparing a brightened, letterboxed input yourself and leaving `--gamma` at `1.0`.

## Original LTX IC-LoRA flow

The official `ICLoraPipeline` builds several components for one request:

- prompt encoding through Gemma;
- image/video conditioning (here: the letterboxed reference video, via IC-LoRA);
- diffusion stage 1 at half resolution;
- spatial upsampling;
- diffusion stage 2 at target resolution;
- video and audio decode.

Inside the official pipeline, each `DiffusionStage` builds the large LTX transformer on demand. The pipeline runs two distinct stages (`stage_1`, `stage_2`), so a serving process that keeps both full CUDA transformers resident can leave too little room for request-time decode and activation memory.

```mermaid
flowchart TD
    request["Outpaint request"] --> prompt["PromptEncoder Gemma"]
    request --> videoCond["IC-LoRA video conditioner (letterboxed input)"]
    prompt --> stage1["DiffusionStage stage 1"]
    videoCond --> stage1
    stage1 --> upsample["Spatial upsampler"]
    upsample --> stage2["DiffusionStage stage 2"]
    prompt --> stage2
    stage2 --> decode["Video/audio decode"]
    decode --> output["MP4 output"]
```

## FlexTensor Serving Architecture

This example keeps the original LTX pipeline structure and only changes how the two diffusion transformers are constructed and cached. IC-LoRA video conditioning, spatial upsampling, and video/audio decoding still run through the official LTX pipeline components and are not wrapped by FlexTensor. Gemma prompt encoding also runs through the official component by default, and is only wrapped by FlexTensor when `--offload-text` is passed (see [Optional: Gemma text-encoder offload](#optional-gemma-text-encoder-offload)).

The key serving difference is lifecycle: FlexTensor initialization and profile loading happen once during server startup, not once per request. Requests reuse the cached FlexTensor-managed `stage1` and `stage2` transformers. Building both LTX transformers and restoring their FlexTensor profile is expensive; doing it per request would hide the benefit of weight streaming behind startup cost. Keeping the managed transformers resident gives request latency that reflects prompt/conditioning, denoising, upsampling, and decode work rather than model/profile construction.

The example builds both LTX transformer stages on CPU and registers each stage as its own FlexTensor root:

- `ltx_stage1`
- `ltx_stage2`

With `--offload-text`, the Gemma prompt encoder is also offloaded under a third manager:

- `ltx_text`

The default include pattern targets the stable module path for the repeated transformer blocks, which own almost all of the weight memory:

```text
velocity_model.transformer_blocks.*
```

```mermaid
classDiagram
    class FlexTensorStage1
    class FlexTensorStage2
    class X0Model_stage1
    class X0Model_stage2
    class LTXModel {
      velocity_model
    }
    class TransformerBlocks {
      transformer_blocks
    }

    FlexTensorStage1 --> X0Model_stage1 : "ltx_stage1"
    FlexTensorStage2 --> X0Model_stage2 : "ltx_stage2"
    X0Model_stage1 --> LTXModel : "velocity_model"
    X0Model_stage2 --> LTXModel : "velocity_model"
    LTXModel --> TransformerBlocks : "velocity_model.transformer_blocks.*"
```

The example deliberately uses name-based patterns instead of `class:Linear` / `class:BasicAVTransformerBlock`. Class patterns match the right parameter-owning modules, but they were too broad during profiling and did not reliably produce the intended per-block layer boundaries. The path pattern above lines up with the actual `named_modules()` tree under each stage root, producing layer statistics and per-block assignments (e.g. `velocity_model.transformer_blocks.0`) rather than one stage-level block. The practical effect is block size: coarse stage boundaries force a single huge transfer block, while the name-based pattern produces sub-GiB blocks that can be streamed under a tight memory budget.

```mermaid
flowchart TD
    subgraph requestFlow [Per Request Flow]
        server["HTTP server with cached FlexTensor transformers"] --> request["POST request"]
        request --> gammaIn["Optional gamma 2.0 on letterboxed input"]
        gammaIn --> prompt3["Original LTX prompt and IC-LoRA conditioning"]
        prompt3 --> stage1["Reuse cached FlexTensor stage1"]
        stage1 --> upsample2["Original LTX spatial upsampler"]
        upsample2 --> stage2["Reuse cached FlexTensor stage2"]
        stage2 --> decode["Original LTX video/audio decode"]
        decode --> gammaOut["Optional inverse gamma 0.5 on output"]
        gammaOut --> response["JSON response with output path and timing"]
    end
```

The startup flow pays the CPU construction and `offload_from_profile` cost once. The request flow does not call `flextensor.offload()` or `offload_from_profile()` again.

## Optional: Gemma text-encoder offload

By default only the two diffusion stages are offloaded. The official `PromptEncoder` builds the Gemma text encoder on GPU for every request and frees it afterward. On memory-constrained GPUs that transient spike during prompt encoding can be the peak that causes an OOM.

Pass `--offload-text` to also place Gemma under a FlexTensor manager (`ltx_text`). The example patches `PromptEncoder._text_encoder_ctx` to build the encoder on CPU once, wrap it with `flextensor.offload` / `offload_from_profile`, cache it, and reuse it across requests (instead of rebuilding it on GPU each time). The text encoder uses its own aggressive config (small resident footprint at the cost of a slower encode):

- `--text-mem-fraction` (default `0.05`): GPU memory fraction budget for the text manager. Lower it to keep less of Gemma resident (smaller peak, but more streaming and higher encode latency); raise it to keep more resident (faster encode at the cost of memory).
- `--text-include-pattern` (default `model.model.language_model.layers.*`): which Gemma submodules to offload. Override only for a different text-encoder layout.

When enabled, profiling writes a third profile under `text/`, and `serve` loads it alongside the stage profiles. The shell helpers enable text offload by default (`OFFLOAD_TEXT=1`) because the 40 GB path otherwise OOMs while rebuilding Gemma during profile or serve. Set `OFFLOAD_TEXT=0` only when you have enough GPU headroom or want to compare the non-text-offloaded path.

## Setup

Start from an NGC PyTorch container with the workspace mounted:

```bash
mkdir -p /tmp/ltx23-outpaint
docker run -it --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/workspace/hf \
  -e HUGGINGFACE_HUB_CACHE=/workspace/hf/hub \
  -v /tmp/ltx23-outpaint:/workspace \
  -w /workspace \
  --entrypoint bash \
  nvcr.io/nvidia/pytorch:26.03-py3
```

Clone LTX-2 and install the pipeline packages, then install this FlexTensor checkout into the LTX environment:

```bash
git clone https://github.com/Lightricks/LTX-2.git /workspace/LTX-2
cd /workspace/LTX-2
uv sync --frozen
uv pip install --python .venv/bin/python -e /path/to/flextensor
```

The serving script resolves model files from Hugging Face when authentication/cache is configured, or you can pass local paths:

- base checkpoint repo: `Lightricks/LTX-2.3`
- base checkpoint file: `ltx-2.3-22b-distilled-1.1.safetensors`
- spatial upsampler file: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- Outpaint LoRA repo: `oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint`
- Outpaint LoRA file: `ltx-2.3-22b-ic-lora-outpaint.safetensors`
- Gemma repo: `google/gemma-3-12b-it-qat-q4_0-unquantized`

The Gemma repo is gated. Request access on Hugging Face and export
`HF_TOKEN` before running `setup.sh`; the helper fails fast when it is missing.
Because `setup.sh` downloads external model snapshots, it also requires
`ACCEPT_EXTERNAL_LICENSES=1` after you have reviewed the upstream terms.

## Profile

Create the FlexTensor profile before serving. Profile at the resolution you intend to serve. Heights and widths must be multiples of 64; for 720p use `1280x704` (720 snaps to 704) and pass the same dimensions to `serve` and to each request.

On A100 40 GB, start with a low memory fraction such as `0.15`:

```bash
/workspace/LTX-2/.venv/bin/python \
  /path/to/flextensor/examples/ltx23/outpaint/serve_infer.py profile \
  --accept-external-licenses \
  --conditioning-video /workspace/inputs/letterboxed_720p.mp4 \
  --height 704 --width 1280 \
  --profile-dir /workspace/outputs/ltx23_outpaint_profile \
  --max-gpu-mem-fraction 0.15 \
  --output-path /workspace/outputs/profile_warmup.mp4
```

On A100 80 GB, a larger fraction such as `0.30` is a good starting point: at 720p the per-layer DiT compute already hides block transfers, so this is effectively latency-bound while staying memory-safe. Raise it toward lower latency if memory allows, or lower it if you OOM.

If your model files are already materialized locally, pass them explicitly:

```bash
  --distilled-checkpoint-path /workspace/models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upsampler-path /workspace/models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /workspace/models/gemma \
  --lora-path /workspace/models/lora/ltx-2.3-22b-ic-lora-outpaint.safetensors
```

The profile command runs one discovery request and one profiling request by default, then writes:

```text
/workspace/outputs/ltx23_outpaint_profile/stage1/profile.json
/workspace/outputs/ltx23_outpaint_profile/stage2/profile.json
```

With `--offload-text`, a third profile is also written under `text/`.

### Profiles are resolution-specific

The block assignment depends on per-layer compute time, which scales with the latent sequence length (and therefore with `--height`/`--width`). Profile at the resolution you intend to serve, and serve with the same dimensions. Reusing a low-resolution profile for high-resolution requests applies the wrong block strategy. At higher resolutions the per-layer DiT compute exceeds a block transfer, so transfers hide almost completely and FlexTensor can offload nearly the entire DiT with negligible latency cost; the peak then becomes activation-bound rather than weight-bound.

## Serve

Start the local server from the saved profile:

```bash
/workspace/LTX-2/.venv/bin/python \
  /path/to/flextensor/examples/ltx23/outpaint/serve_infer.py serve \
  --accept-external-licenses \
  --conditioning-video /workspace/inputs/letterboxed_720p.mp4 \
  --height 704 --width 1280 \
  --profile-dir /workspace/outputs/ltx23_outpaint_profile \
  --max-gpu-mem-fraction 0.15 \
  --host 127.0.0.1 \
  --port 8020 \
  --warmup-output-path /workspace/outputs/server_warmup.mp4 \
  --output-path /workspace/outputs/server_default.mp4
```

The server performs one warmup generation at startup so model/profile construction is not counted in later request latency.

## Request

Issue a local request. `conditioning_video` is the already-letterboxed source; `frame_rate` is flexible (15 or 24 fps both work). Add `"gamma": 2.0` for dark scenes.

```bash
curl -sS -X POST http://127.0.0.1:8020 \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "extend the scene naturally, consistent with the original footage",
    "conditioning_video": "/workspace/inputs/letterboxed_720p.mp4",
    "output_path": "/workspace/outputs/request_001.mp4",
    "height": 704,
    "width": 1280,
    "conditioning_strength": 1.0,
    "frame_rate": 24,
    "seed": 171198
  }'
```

The response includes the output path, resolved `num_frames`/`frame_rate`, request wall time, CUDA peak allocation, and before/after GPU memory snapshots.

`audio_mode` defaults to `copy`, which remuxes source audio up to the
generated video duration. This keeps capped-frame requests such as
`num_frames=49` from producing a short video stream with full-length source
audio continuing after the final generated frame. Set `audio_mode` to `none`
to omit audio.

## Multi-GPU serving with NGINX

For a single node with multiple GPUs, run one outpaint server process per GPU
and put NGINX in front of them. Each worker process has its own
`CUDA_VISIBLE_DEVICES` value, local port, warmup output, and default output.
NGINX exposes one front-end port and load balances POST requests across the
warmed workers.

The HTTP API does not change when NGINX is enabled. Requests still pass
`conditioning_video` and `output_path` as filesystem paths, so every worker
must see the same input and output directories. Use a unique `output_path` for
each request; generation requests are not idempotent.

### Scripted NGINX workflow

The helper scripts use these directory defaults:

```text
WORKSPACE_DIR=/workspace
BASE=$WORKSPACE_DIR/outpaint
INPUT_DIR=$BASE/inputs
OUTPUT_DIR=$BASE/outputs
PROFILE_DIR=$OUTPUT_DIR/ltx23_outpaint_profile
EX=/my_home/flex-tensor/examples/ltx23/outpaint
```

Place your letterboxed conditioning videos under `$INPUT_DIR`, or override
`LETTERBOXED_VIDEO` when running `setup.sh`. The setup helper installs NGINX,
creates or reuses the LTX-2 checkout and Python environment, downloads model
snapshots into the configured Hugging Face cache, and writes the FlexTensor
profile used by all workers. It defaults to `OFFLOAD_TEXT=1`,
`TEXT_MEM_FRACTION=0.05`, and `MAX_GPU_MEM_FRACTION=0.15` to preserve activation
headroom on A100 40 GB. Override these environment variables for larger GPUs or
throughput experiments.

1. Prepare the container/workspace and profile the target resolution:

```bash
export WORKSPACE_DIR=/workspace
export BASE=/workspace/outpaint
export EX=/my_home/flex-tensor/examples/ltx23/outpaint
export LETTERBOXED_VIDEO=$BASE/inputs/letterboxed_720p.mp4
export ACCEPT_EXTERNAL_LICENSES=1

mkdir -p "$BASE/inputs" "$BASE/outputs"
bash "$EX/setup.sh"
```

2. Start one worker per GPU behind NGINX:

```bash
NUM_WORKERS=8 \
BASE_PORT=8020 \
FRONTEND_HOST=0.0.0.0 \
FRONTEND_PORT=8080 \
BASE=/workspace/outpaint \
WORKSPACE_DIR=/workspace \
EX=/my_home/flex-tensor/examples/ltx23/outpaint \
bash /my_home/flex-tensor/examples/ltx23/outpaint/nginx_serve.sh
```

By default this starts workers on `127.0.0.1:8020` through
`127.0.0.1:8027` and exposes NGINX on `0.0.0.0:8080`. Override `GPU_IDS`
when the visible GPU IDs are not `0..NUM_WORKERS-1`:

```bash
GPU_IDS="0,2,4,6" \
FRONTEND_PORT=8080 \
bash /my_home/flex-tensor/examples/ltx23/outpaint/nginx_serve.sh
```

The launcher waits for every worker to finish warmup and answer `GET /healthz`
before starting NGINX. It writes worker logs to:

```text
$OUTPUT_DIR/serve_gpu<GPU_ID>_port<PORT>.log
```

3. In another shell, check the front end and send requests to NGINX:

```bash
curl -fsS http://127.0.0.1:8080/healthz

curl -sS -X POST http://127.0.0.1:8080 \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "extend the scene naturally, consistent with the original footage",
    "conditioning_video": "/workspace/outpaint/inputs/letterboxed_720p.mp4",
    "output_path": "/workspace/outpaint/outputs/request_001.mp4",
    "height": 704,
    "width": 1280,
    "num_frames": 49,
    "frame_rate": 24,
    "conditioning_strength": 1.0,
    "seed": 171198,
    "audio_mode": "copy"
  }'
```

NGINX uses its default round-robin upstream policy, so bursts of local requests
are spread predictably across the warmed workers. The generated access log
includes the selected upstream address, making it easy to verify which worker
handled each request. The generated config also sets `proxy_next_upstream off`
so NGINX does not replay non-idempotent POST requests against another worker.

### Concurrent request example

The following loop was used to test concurrent requests through the NGINX
front end. It sends three different conditioning videos for each seed and
writes each JSON response next to the generated MP4.

```bash
mkdir -p /workspace/outpaint/outputs/requests

for seed in 171198 171199 171200 171201 171202 171203; do
  curl -sS -X POST http://127.0.0.1:8080 \
    -H 'Content-Type: application/json' \
    -d "{
      \"prompt\": \"extend the scene naturally, consistent with the original footage\",
      \"conditioning_video\": \"/workspace/outpaint/inputs/full_portrait396_sides442_1280x704_audio_2s.mp4\",
      \"output_path\": \"/workspace/outpaint/outputs/requests/full_portrait396_sides442_seed${seed}.mp4\",
      \"height\": 704,
      \"width\": 1280,
      \"num_frames\": 49,
      \"frame_rate\": 24,
      \"conditioning_strength\": 1.0,
      \"seed\": ${seed},
      \"audio_mode\": \"copy\"
    }" \
    > "/workspace/outpaint/outputs/requests/full_portrait396_sides442_seed${seed}.json" 2>&1 &

  curl -sS -X POST http://127.0.0.1:8080 \
    -H 'Content-Type: application/json' \
    -d "{
      \"prompt\": \"extend the scene naturally, consistent with the original footage\",
      \"conditioning_video\": \"/workspace/outpaint/inputs/letterboxed_720p.mp4\",
      \"output_path\": \"/workspace/outpaint/outputs/requests/letterboxed_720p_seed${seed}.mp4\",
      \"height\": 704,
      \"width\": 1280,
      \"num_frames\": 49,
      \"frame_rate\": 24,
      \"conditioning_strength\": 1.0,
      \"seed\": ${seed},
      \"audio_mode\": \"copy\"
    }" \
    > "/workspace/outpaint/outputs/requests/letterboxed_720p_seed${seed}.json" 2>&1 &

  curl -sS -X POST http://127.0.0.1:8080 \
    -H 'Content-Type: application/json' \
    -d "{
      \"prompt\": \"extend the scene naturally, consistent with the original footage\",
      \"conditioning_video\": \"/workspace/outpaint/inputs/portrait_center_letterboxed_1280x704_49f.mp4\",
      \"output_path\": \"/workspace/outpaint/outputs/requests/portrait_center_letterboxed_seed${seed}.mp4\",
      \"height\": 704,
      \"width\": 1280,
      \"num_frames\": 49,
      \"frame_rate\": 24,
      \"conditioning_strength\": 1.0,
      \"seed\": ${seed},
      \"audio_mode\": \"copy\"
    }" \
    > "/workspace/outpaint/outputs/requests/portrait_center_letterboxed_seed${seed}.json" 2>&1 &
done

wait
```

## Notes

- This example targets the default two-stage (latent upscaling) LTX-2.3 path; the request validates that each generation invokes exactly two diffusion stages, so a future upstream pipeline shape change fails loudly instead of silently reusing the wrong cached stage.
- The server uses `torch.no_grad()` rather than a one-shot `torch.inference_mode()` wrapper. Persistent serving reuses cached objects across requests, and inference-mode tensors can fail later with version-counter errors.
- Requests are serialized around the shared pipeline/cache object.
- The NGINX serving path has been exercised with concurrent requests like the
  example above. Still verify output correctness for your own footage: the
  generated bars should be filled and temporally consistent, audio should be
  intact, and FlexTensor should not introduce corruption versus a native run.
