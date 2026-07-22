#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
BASE="${BASE:-$WORKSPACE_DIR/outpaint}"
EX="${EX:-/my_home/flex-tensor/examples/ltx23/outpaint}"
VENV_DIR="${VENV_DIR:-$WORKSPACE_DIR/.venv}"
INPUT_DIR="${INPUT_DIR:-$BASE/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/outputs}"
HF_HOME="${HF_HOME:-$BASE/ltx2.3/huggingface_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE/.cache}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
WORKER_HOST="${WORKER_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8020}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-8080}"
# NUM_WORKERS remains the number of GPUs for backward compatibility. The
# number of HTTP replicas is NUM_WORKERS / CONTEXT_PARALLEL_SIZE.
NUM_WORKERS="${NUM_WORKERS:-8}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
CONTROL_PLANE_TIMEOUT_SECONDS="${CONTROL_PLANE_TIMEOUT_SECONDS:-30}"
TORCHRUN_MAX_RESTARTS="${TORCHRUN_MAX_RESTARTS:-3}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
PY="$VENV_DIR/bin/python"
export HF_HOME HF_HUB_CACHE XDG_CACHE_HOME
export HF_XET_CACHE PYTORCH_CUDA_ALLOC_CONF TORCH_NCCL_ASYNC_ERROR_HANDLING

SNAP="${SNAP:-}"
DISTILLED="${DISTILLED:-}"
UPSAMPLER="${UPSAMPLER:-}"
GEMMA="${GEMMA:-}"
LORA="${LORA:-}"

DISTILLED_FILENAME="ltx-2.3-22b-distilled-1.1.safetensors"
UPSAMPLER_FILENAME="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LORA_FILENAME="ltx-2.3-22b-ic-lora-outpaint.safetensors"

DEFAULT_PROFILE_DIR="$OUTPUT_DIR/ltx23_outpaint_profile"
if [[ "$CONTEXT_PARALLEL_SIZE" != "1" ]]; then
  DEFAULT_PROFILE_DIR="${DEFAULT_PROFILE_DIR}_cp${CONTEXT_PARALLEL_SIZE}"
fi
PROFILE_DIR="${PROFILE_DIR:-$DEFAULT_PROFILE_DIR}"
LETTERBOXED_VIDEO="${LETTERBOXED_VIDEO:-$INPUT_DIR/letterboxed_720p.mp4}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-49}"
FRAME_RATE="${FRAME_RATE:-24}"
MAX_GPU_MEM_FRACTION="${MAX_GPU_MEM_FRACTION:-0.15}"
SERVER_READY_ATTEMPTS="${SERVER_READY_ATTEMPTS:-3600}"
OFFLOAD_TEXT="${OFFLOAD_TEXT:-1}"
TEXT_MEM_FRACTION="${TEXT_MEM_FRACTION:-0.05}"
NGINX_CONF="${NGINX_CONF:-/tmp/ltx23-outpaint-nginx-${USER:-user}.conf}"
NGINX_ERROR_LOG="${NGINX_ERROR_LOG:-$OUTPUT_DIR/nginx_error.log}"
NGINX_ACCESS_LOG="${NGINX_ACCESS_LOG:-$OUTPUT_DIR/nginx_access.log}"

WORKER_PIDS=()
WORKER_LOGS=()
NGINX_PID=""
CLEANED_UP=0
REPLICA_COUNT=0

validate_num_frames() {
  if [[ ! "$NUM_FRAMES" =~ ^[0-9]+$ ]] ||
    ((NUM_FRAMES < 1 || (NUM_FRAMES - 1) % 8 != 0)); then
    echo "NUM_FRAMES must be a positive integer of the form 8*k + 1; got: $NUM_FRAMES" >&2
    exit 1
  fi
}

validate_context_parallel_size() {
  case "$CONTEXT_PARALLEL_SIZE" in
    1 | 2 | 4 | 8) ;;
    *)
      echo "CONTEXT_PARALLEL_SIZE must be one of 1, 2, 4, or 8; got: $CONTEXT_PARALLEL_SIZE" >&2
      exit 1
      ;;
  esac
}

validate_torchrun_max_restarts() {
  if [[ ! "$TORCHRUN_MAX_RESTARTS" =~ ^[0-9]+$ ]]; then
    echo "TORCHRUN_MAX_RESTARTS must be a non-negative integer; got: $TORCHRUN_MAX_RESTARTS" >&2
    exit 1
  fi
}

require_path() {
  local label="$1"
  local path="$2"

  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"

  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

resolve_existing_path() {
  local explicit_path="$1"
  local fallback_path="$2"
  shift 2

  if [[ -n "$explicit_path" ]]; then
    echo "$explicit_path"
    return
  fi

  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "$fallback_path"
}

parse_gpu_ids() {
  local raw_gpu_ids="${GPU_IDS:-}"

  if [[ -n "$raw_gpu_ids" ]]; then
    # Accept either "0 1 2" or "0,1,2" for convenience.
    read -r -a GPU_ID_ARRAY <<<"${raw_gpu_ids//,/ }"
    NUM_WORKERS="${#GPU_ID_ARRAY[@]}"
    return
  fi

  GPU_ID_ARRAY=()
  local worker_index
  for worker_index in $(seq 0 $((NUM_WORKERS - 1))); do
    GPU_ID_ARRAY+=("$worker_index")
  done
}

# shellcheck disable=SC2329  # Invoked through trap handlers below.
cleanup() {
  set +e
  if ((CLEANED_UP)); then
    return
  fi
  CLEANED_UP=1
  echo "Stopping nginx and worker processes..."
  if [[ -n "$NGINX_PID" ]] && kill -0 "$NGINX_PID" 2>/dev/null; then
    kill "$NGINX_PID" 2>/dev/null || true
    wait "$NGINX_PID" 2>/dev/null || true
  fi
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  echo "Cleanup complete."
}

wait_for_worker() {
  local worker_pid="$1"
  local worker_port="$2"
  local serve_log="$3"
  local ready_line="Serving on http://$WORKER_HOST:$worker_port"
  local attempt

  echo "Waiting up to $SERVER_READY_ATTEMPTS seconds for worker on $WORKER_HOST:$worker_port; log: $serve_log"
  for attempt in $(seq 1 "$SERVER_READY_ATTEMPTS"); do
    if [[ -f "$serve_log" ]] && grep -Fq "$ready_line" "$serve_log" &&
      curl -fsS "http://$WORKER_HOST:$worker_port/healthz" >/dev/null 2>&1; then
      echo "Worker ready at http://$WORKER_HOST:$worker_port"
      return
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      echo "Worker on port $worker_port exited before becoming ready; see $serve_log" >&2
      exit 1
    fi
    if ((attempt % 30 == 0)) && [[ -f "$serve_log" ]]; then
      echo "Still waiting for port $worker_port; latest log line:"
      sed -n '$p' "$serve_log"
    fi
    sleep 1
  done

  echo "Worker did not become ready at http://$WORKER_HOST:$worker_port; see $serve_log" >&2
  exit 1
}

print_log_context() {
  local label="$1"
  local log_path="$2"
  local max_lines="${3:-40}"

  if [[ ! -f "$log_path" ]]; then
    echo "No $label log found at $log_path" >&2
    return
  fi
  if [[ ! -s "$log_path" ]]; then
    echo "$label log is empty: $log_path" >&2
    return
  fi

  local line_count
  local start_line
  line_count="$(wc -l <"$log_path")"
  start_line=$((line_count > max_lines ? line_count - max_lines + 1 : 1))

  echo "Last $max_lines lines from $label log ($log_path):" >&2
  sed -n "${start_line},${line_count}p" "$log_path" >&2
}

report_exited_processes() {
  local wait_status="$1"
  local reported=0

  echo "A worker or NGINX process exited; wait status: $wait_status" >&2
  if [[ -n "$NGINX_PID" ]] && ! kill -0 "$NGINX_PID" 2>/dev/null; then
    echo "NGINX process $NGINX_PID has exited." >&2
    print_log_context "NGINX error" "$NGINX_ERROR_LOG"
    print_log_context "NGINX access" "$NGINX_ACCESS_LOG"
    reported=1
  fi

  local worker_index
  for worker_index in "${!WORKER_PIDS[@]}"; do
    local worker_pid="${WORKER_PIDS[$worker_index]}"
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      local worker_port=$((BASE_PORT + worker_index))
      echo "Worker $worker_index process $worker_pid on port $worker_port has exited." >&2
      print_log_context "worker $worker_index" "${WORKER_LOGS[$worker_index]}"
      reported=1
    fi
  done

  if ((!reported)); then
    echo "No exited process could be identified; recent logs follow." >&2
    print_log_context "NGINX error" "$NGINX_ERROR_LOG"
    print_log_context "NGINX access" "$NGINX_ACCESS_LOG"
    for worker_index in "${!WORKER_LOGS[@]}"; do
      print_log_context "worker $worker_index" "${WORKER_LOGS[$worker_index]}"
    done
  fi
}

write_nginx_config() {
  local servers=""
  local replica_index

  for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
    local worker_port=$((BASE_PORT + replica_index))
    servers+="        server $WORKER_HOST:$worker_port max_fails=1 fail_timeout=30s;
"
  done

  cat >"$NGINX_CONF" <<EOF
worker_processes 1;
error_log $NGINX_ERROR_LOG warn;
pid /tmp/ltx23-outpaint-nginx-${USER:-user}-${FRONTEND_PORT}.pid;

events {
    worker_connections 4096;
}

http {
    log_format upstreamlog '\$remote_addr [\$time_local] "\$request" '
                           'status=\$status upstream=\$upstream_addr '
                           'request_time=\$request_time upstream_time=\$upstream_response_time';
    access_log $NGINX_ACCESS_LOG upstreamlog;
    client_max_body_size 1m;
    client_body_temp_path /tmp/ltx23-outpaint-client-body;
    proxy_temp_path /tmp/ltx23-outpaint-proxy;
    fastcgi_temp_path /tmp/ltx23-outpaint-fastcgi;
    uwsgi_temp_path /tmp/ltx23-outpaint-uwsgi;
    scgi_temp_path /tmp/ltx23-outpaint-scgi;

    upstream ltx_outpaint_workers {
$servers    }

    server {
        listen $FRONTEND_HOST:$FRONTEND_PORT;

        location / {
            proxy_pass http://ltx_outpaint_workers;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_connect_timeout 10m;
            proxy_send_timeout 10m;
            proxy_read_timeout 10m;
            proxy_next_upstream off;
        }
    }
}
EOF
}

validate_num_frames
validate_context_parallel_size
validate_torchrun_max_restarts
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME"
trap cleanup EXIT
trap 'trap - EXIT; cleanup; exit 130' HUP INT TERM

require_command curl
require_command nginx
require_path "Python venv interpreter" "$PY"
require_path "serve entrypoint" "$EX/serve_infer.py"
require_path "letterboxed conditioning video" "$LETTERBOXED_VIDEO"
require_path "FlexTensor profile directory" "$PROFILE_DIR"
parse_gpu_ids

if ((NUM_WORKERS < 1)); then
  echo "NUM_WORKERS must be at least 1" >&2
  exit 1
fi
if ((${#GPU_ID_ARRAY[@]} % CONTEXT_PARALLEL_SIZE != 0)); then
  echo "GPU count (${#GPU_ID_ARRAY[@]}) must be divisible by CONTEXT_PARALLEL_SIZE ($CONTEXT_PARALLEL_SIZE)" >&2
  exit 1
fi
REPLICA_COUNT=$((${#GPU_ID_ARRAY[@]} / CONTEXT_PARALLEL_SIZE))
if ((CONTEXT_PARALLEL_SIZE > 1)); then
  require_path "torchrun executable" "$VENV_DIR/bin/torchrun"
fi

shopt -s nullglob
LTX23_HF_REPO="models--Lightricks--LTX-2.3"
LORA_HF_REPO="models--oumoumad--LTX-2.3-22b-IC-LoRA-Outpaint"

distilled_candidates=(
  "$WORKSPACE_DIR/models/ltx23/$DISTILLED_FILENAME"
  "$BASE/models/ltx23/$DISTILLED_FILENAME"
)
upsampler_candidates=(
  "$WORKSPACE_DIR/models/ltx23/$UPSAMPLER_FILENAME"
  "$BASE/models/ltx23/$UPSAMPLER_FILENAME"
)

if [[ -n "$SNAP" ]]; then
  distilled_candidates=("$SNAP/$DISTILLED_FILENAME" "${distilled_candidates[@]}")
  upsampler_candidates=("$SNAP/$UPSAMPLER_FILENAME" "${upsampler_candidates[@]}")
fi

distilled_candidates+=(
  "$BASE/ltx2.3/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$DISTILLED_FILENAME"
  "$WORKSPACE_DIR/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$DISTILLED_FILENAME"
  "$CACHE_DIR/$LTX23_HF_REPO"/snapshots/*/"$DISTILLED_FILENAME"
)
upsampler_candidates+=(
  "$BASE/ltx2.3/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$UPSAMPLER_FILENAME"
  "$WORKSPACE_DIR/huggingface_cache/hub/$LTX23_HF_REPO"/snapshots/*/"$UPSAMPLER_FILENAME"
  "$CACHE_DIR/$LTX23_HF_REPO"/snapshots/*/"$UPSAMPLER_FILENAME"
)
gemma_candidates=(
  "$WORKSPACE_DIR/models/gemma"
  "$BASE/gemma3"
  "$BASE/models/gemma"
)
lora_candidates=(
  "$WORKSPACE_DIR/models/lora/$LORA_FILENAME"
  "$BASE/LTX-2.3-22b-IC-LoRA-Outpaint/$LORA_FILENAME"
  "$BASE/models/lora/$LORA_FILENAME"
  "$CACHE_DIR/$LORA_HF_REPO"/snapshots/*/"$LORA_FILENAME"
)

DISTILLED="$(resolve_existing_path "$DISTILLED" "$WORKSPACE_DIR/models/ltx23/$DISTILLED_FILENAME" "${distilled_candidates[@]}")"
UPSAMPLER="$(resolve_existing_path "$UPSAMPLER" "$WORKSPACE_DIR/models/ltx23/$UPSAMPLER_FILENAME" "${upsampler_candidates[@]}")"
GEMMA="$(resolve_existing_path "$GEMMA" "$WORKSPACE_DIR/models/gemma" "${gemma_candidates[@]}")"
LORA="$(resolve_existing_path "$LORA" "$WORKSPACE_DIR/models/lora/$LORA_FILENAME" "${lora_candidates[@]}")"
shopt -u nullglob

MODEL_ARGS=(--cache-dir "$CACHE_DIR")
if [[ -f "$DISTILLED" ]]; then
  MODEL_ARGS+=(--distilled-checkpoint-path "$DISTILLED")
  echo "Using local distilled checkpoint: $DISTILLED"
else
  echo "No local distilled checkpoint found; resolving $DISTILLED_FILENAME from Lightricks/LTX-2.3"
fi
if [[ -f "$UPSAMPLER" ]]; then
  MODEL_ARGS+=(--spatial-upsampler-path "$UPSAMPLER")
  echo "Using local spatial upsampler: $UPSAMPLER"
else
  echo "No local spatial upsampler found; resolving $UPSAMPLER_FILENAME from Lightricks/LTX-2.3"
fi
if [[ -d "$GEMMA" ]]; then
  MODEL_ARGS+=(--gemma-root "$GEMMA")
  echo "Using local Gemma root: $GEMMA"
else
  echo "No local Gemma root found; resolving google/gemma-3-12b-it-qat-q4_0-unquantized"
fi
if [[ -f "$LORA" ]]; then
  MODEL_ARGS+=(--lora-path "$LORA")
  echo "Using local LoRA checkpoint: $LORA"
else
  echo "No local LoRA checkpoint found; resolving $LORA_FILENAME from oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint"
fi
if [[ "$OFFLOAD_TEXT" == "1" ]]; then
  MODEL_ARGS+=(--offload-text --text-mem-fraction "$TEXT_MEM_FRACTION")
  echo "Using FlexTensor text-encoder offload with text mem fraction: $TEXT_MEM_FRACTION"
fi

for replica_index in $(seq 0 $((REPLICA_COUNT - 1))); do
  group_start=$((replica_index * CONTEXT_PARALLEL_SIZE))
  replica_gpu_ids=("${GPU_ID_ARRAY[@]:group_start:CONTEXT_PARALLEL_SIZE}")
  visible_devices="$(
    IFS=,
    echo "${replica_gpu_ids[*]}"
  )"
  WORKER_PORT=$((BASE_PORT + replica_index))
  SERVE_LOG="$OUTPUT_DIR/serve_cp${CONTEXT_PARALLEL_SIZE}_replica${replica_index}_port${WORKER_PORT}.log"
  WORKER_LOGS+=("$SERVE_LOG")
  echo "Starting CP${CONTEXT_PARALLEL_SIZE} replica $replica_index on GPUs $visible_devices at http://$WORKER_HOST:$WORKER_PORT"

  LAUNCH_COMMAND=("$PY")
  if ((CONTEXT_PARALLEL_SIZE > 1)); then
    LAUNCH_COMMAND=(
      "$VENV_DIR/bin/torchrun"
      --standalone
      --nproc-per-node="$CONTEXT_PARALLEL_SIZE"
      --max-restarts="$TORCHRUN_MAX_RESTARTS"
    )
  fi

  CUDA_VISIBLE_DEVICES="$visible_devices" nohup "${LAUNCH_COMMAND[@]}" "$EX/serve_infer.py" serve \
    --conditioning-video "$LETTERBOXED_VIDEO" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --frame-rate "$FRAME_RATE" \
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE" \
    --control-plane-timeout-seconds "$CONTROL_PLANE_TIMEOUT_SECONDS" \
    "${MODEL_ARGS[@]}" \
    --max-gpu-mem-fraction "$MAX_GPU_MEM_FRACTION" \
    --profile-dir "$PROFILE_DIR" \
    --host "$WORKER_HOST" \
    --port "$WORKER_PORT" \
    --warmup-output-path "$OUTPUT_DIR/server_warmup_cp${CONTEXT_PARALLEL_SIZE}_replica${replica_index}.mp4" \
    --output-path "$OUTPUT_DIR/server_default_cp${CONTEXT_PARALLEL_SIZE}_replica${replica_index}.mp4" \
    >"$SERVE_LOG" 2>&1 &
  WORKER_PIDS+=("$!")
done

for worker_index in "${!WORKER_PIDS[@]}"; do
  wait_for_worker "${WORKER_PIDS[$worker_index]}" "$((BASE_PORT + worker_index))" "${WORKER_LOGS[$worker_index]}"
done

write_nginx_config
echo "Generated nginx config:"
sed -n '1,200p' "$NGINX_CONF"
nginx -t -c "$NGINX_CONF"
nginx -c "$NGINX_CONF" -g "daemon off;" &
NGINX_PID="$!"

echo "Frontend ready at http://$FRONTEND_HOST:$FRONTEND_PORT"
echo "Worker logs:"
printf '  %s\n' "${WORKER_LOGS[@]}"
wait_status=0
wait -n || wait_status=$?
report_exited_processes "$wait_status"
exit "$wait_status"
