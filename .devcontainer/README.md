<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
Environment variables sourced from host using ${localEnv:ENV_VAR_NAME}:
- HF_TOKEN: HuggingFace token for downloading models

Set them in your shell configuration file.

Use `.envrc` file to store private environment variables.

## Security Model

This devcontainer uses an elevated security posture required by two subsystems:
the Claude Code sandbox (`bubblewrap`) and Docker-based inference server workflows.

### Trust Model

**Designed for trusted single-user environments only:**
- Developer workstations (primary use case)
- Single-tenant leased compute clusters (occasional use for debugging/benchmarking)

**Not suitable for:**
- Shared multi-tenant systems where other users have access to the host
- Untrusted or public cloud infrastructure without verified sole tenancy

> **Cluster users:** Before using this devcontainer on a leased cluster, confirm
> you are the sole tenant. The Docker socket mount grants access to the host
> Docker daemon, which runs as root.

### Security Settings Rationale

| Setting | Reason |
|---------|--------|
| `docker-outside-of-docker` feature | Mounts `/var/run/docker.sock` to allow managing Docker containers from inside the devcontainer. Required for launching inference servers (vLLM, Triton) for debugging, benchmarking, and integration tests. Grants root-equivalent access to the host Docker daemon — acceptable only in single-user trusted environments. |
| `--cap-add=SYS_PTRACE` | Enables debuggers (gdb, strace, py-spy) and GPU profiling tools that inspect process memory and system calls. |
| `--cap-add=SYS_ADMIN` | Required by `bubblewrap` (Claude Code sandbox) to create user namespaces for process isolation. |
| `--ipc=host` | Shares the host IPC namespace, enabling CUDA IPC for PyTorch and NCCL shared memory across GPU processes. |
| `--gpus=all` | Passes all host GPUs into the container for ML workloads. |
| `--security-opt=seccomp=unconfined` | The default Docker seccomp profile blocks `clone` and `unshare` syscalls that `bubblewrap` requires. Must be disabled for the Claude Code sandbox to function. |
| `--security-opt=apparmor=unconfined` | AppArmor's default profile blocks namespace operations that `bubblewrap` requires. Must be disabled for the Claude Code sandbox to function. |

### Alternatives Considered

- **Docker socket proxy**: Would limit which Docker API calls are permitted (e.g., block privileged container creation). Adds dependency complexity; deferred unless the threat model changes.
- **Rootless Docker**: Eliminates daemon root access. Has known GPU passthrough rough edges that risk breaking vLLM workflows; deferred for now.
