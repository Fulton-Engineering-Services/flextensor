# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

_requirements_file_mentions_vllm() {
  grep -Eiq '^[[:space:]]*vllm(\[|[<>=!~;[:space:]]|$)' "$1"
}

_installed_vllm_version() {
  python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("vllm"))
except PackageNotFoundError:
    pass
PY
}

install_requirements_preserving_vllm() {
  local requirements_file=$1
  local before_vllm_version=""
  local check_vllm_version=false

  if _requirements_file_mentions_vllm "$requirements_file"; then
    before_vllm_version="$(_installed_vllm_version)"
    if [ -n "$before_vllm_version" ]; then
      check_vllm_version=true
    fi
  fi

  local pip_exit=0
  pip install -r "$requirements_file" --quiet || pip_exit=$?
  if [ "$pip_exit" -ne 0 ]; then
    return "$pip_exit"
  fi

  if [ "$check_vllm_version" = true ]; then
    local after_vllm_version
    after_vllm_version="$(_installed_vllm_version)"
    if [ "$after_vllm_version" != "$before_vllm_version" ]; then
      echo "ERROR: vLLM changed during requirements install: ${before_vllm_version} -> ${after_vllm_version}" >&2
      return 1
    fi
  fi
}
