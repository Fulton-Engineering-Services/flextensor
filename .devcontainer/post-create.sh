#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Per-user settings
git config --global rebase.updateRefs true

# Set up zsh plugins list
sed -i -E 's/^plugins=\(.*\)/plugins=(git python direnv ssh-agent mise)/' ~/.zshrc
