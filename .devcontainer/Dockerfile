# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${BASE_IMAGE}

# Remove conflicting user and group with UID/GID 1000 to allow devcontainer to create it at runtime
# This ensures proper file permissions on mounted volumes matching the host user
RUN  if getent passwd 1000 > /dev/null 2>&1; then \
  userdel -r $(getent passwd 1000 | cut -d: -f1); \
 fi \
 && if getent group 1000 > /dev/null 2>&1; then \
  groupdel $(getent group 1000 | cut -d: -f1); \
 fi

# libb64-0d and libcurl4 are required by perf_analyzer
# bubblewrap and socat are required by the coding agent sandbox environment.
RUN apt-get update \
 && apt-get install -y \
  sudo vim curl less htop direnv ranger telnet git skopeo \
  bubblewrap socat ripgrep \
  libb64-0d libcurl4 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN curl https://mise.run | env MISE_INSTALL_PATH=/usr/local/bin/mise sh
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
 && UV_TOOL_BIN_DIR=/usr/local/bin uv tool install ruff \
 && UV_TOOL_BIN_DIR=/usr/local/bin uv tool install pre-commit
