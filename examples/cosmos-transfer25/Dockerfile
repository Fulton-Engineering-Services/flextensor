# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libx11-dev \
        tree \
        wget

WORKDIR /workspace

COPY docker/nightly-requirements.txt /tmp/cosmos-nightly-requirements.txt
COPY packages /tmp/cosmos-packages
COPY --from=flextensor examples/cosmos-transfer25/cosmos-transfer25-flextensor.patch /tmp/cosmos-transfer25-flextensor.patch

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --break-system-packages -r /tmp/cosmos-nightly-requirements.txt

RUN mv /usr/local/bin/uvx /usr/local/bin/uvx.real && \
    cat > /usr/local/bin/uvx <<'EOF' && \
    chmod +x /usr/local/bin/uvx
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == hf* ]]; then
    shift
    if [[ "${1:-}" == "download" ]]; then
        command="$1"
        repo_id="$2"
        shift 2
        positional=()
        options=()
        while (($#)); do
            case "$1" in
                --repo-type|--revision|--include|--exclude|--local-dir|--cache-dir)
                    options+=("$1" "$2")
                    shift 2
                    ;;
                --*)
                    options+=("$1")
                    shift
                    ;;
                *)
                    positional+=("$1")
                    shift
                    ;;
            esac
        done
        exec hf "$command" "$repo_id" "${positional[@]}" "${options[@]}"
    fi
    exec hf "$@"
fi

exec /usr/local/bin/uvx.real "$@"
EOF

RUN pip install --no-deps /tmp/cosmos-packages/*

COPY . /workspace
RUN git apply /tmp/cosmos-transfer25-flextensor.patch

COPY --from=flextensor examples/cosmos-transfer25/run_infer.py /workspace/run_infer.py
COPY --from=flextensor . /opt/flextensor
RUN python3 -m pip install -e /opt/flextensor

ENTRYPOINT ["/bin/bash"]
