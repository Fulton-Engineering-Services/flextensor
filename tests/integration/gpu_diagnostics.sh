# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

require_nvidia_gpu() {
  local gpu_diag_timeout_s="${GPU_DIAGNOSTICS_TIMEOUT_S:-15}"

  _run_gpu_diag() {
    if command -v timeout >/dev/null 2>&1; then
      timeout "$gpu_diag_timeout_s" "$@"
    else
      "$@"
    fi
  }

  if ! _run_gpu_diag nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU not detected. These tests require a GPU."
    exit 1
  fi

  echo "=== nvidia-smi ==="
  if ! _run_gpu_diag nvidia-smi; then
    echo "WARNING: failed to run NVIDIA GPU summary diagnostics."
  fi

  echo "=== nvidia-smi PCIe details ==="
  if ! _run_gpu_diag nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,pcie.link.gen.gpucurrent,pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.gen.hostmax,pcie.link.width.current,pcie.link.width.max,driver_version --format=csv; then
    echo "WARNING: failed to query detailed NVIDIA GPU PCIe information."
  fi

  echo "=== nvidia-smi topo -m ==="
  if ! _run_gpu_diag nvidia-smi topo -m; then
    echo "WARNING: failed to query NVIDIA GPU topology."
  fi
}
