# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ulysses context parallelism helpers for the LTX 2.3 outpaint server."""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import traceback
from dataclasses import dataclass, replace
from datetime import timedelta
from threading import Event, Thread
from typing import Any, NoReturn

import torch
import torch.distributed as dist
import torch.nn.functional as functional

SUPPORTED_CONTEXT_PARALLEL_SIZES = (1, 2, 4, 8)
VIDEO_VAE_TEMPORAL_SCALE_FACTOR = 8
VIDEO_VAE_SPATIAL_SCALE_FACTOR = 32
COMMAND_WAIT_TIMEOUT = timedelta(days=365)
MAX_ERROR_MESSAGE_CHARS = 4096
MAX_ERROR_TRACEBACK_CHARS = 8192
UNSAFE_RUNTIME_ERROR_MARKERS = (
    "cuda error",
    "cuda driver",
    "cuda kernel",
    "cuda-capable device",
    "device-side assert",
    "illegal memory access",
    "cublas_status",
    "cudnn_status",
    "nccl",
    "c10d",
    "gloo",
    "process group",
    "processgroup",
    "communicator",
    "connection closed by peer",
    "connection reset by peer",
    "pair closure",
)
LOGGER = logging.getLogger(__name__)


class DistributedRequestError(RuntimeError):
    """One or more ranks failed during a coordinated request."""

    def __init__(self, request_id: int, failures: list[dict[str, Any]]) -> None:
        self.request_id = request_id
        self.failures = failures
        summary = "; ".join(
            f"rank {failure['rank']}: {failure['error_type']}: {failure['error']}" for failure in failures
        )
        super().__init__(f"Context-parallel request {request_id} failed: {summary}")


class ReplicaFatalError(RuntimeError):
    """The replica can no longer safely execute distributed requests."""


def _validate_runtime_configuration(size: int, timeout_seconds: int, control_timeout_seconds: int) -> None:
    if size not in SUPPORTED_CONTEXT_PARALLEL_SIZES:
        raise ValueError(f"context_parallel_size must be one of {SUPPORTED_CONTEXT_PARALLEL_SIZES}; got {size}.")
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive; got {timeout_seconds}.")
    if control_timeout_seconds <= 0:
        raise ValueError(f"control_timeout_seconds must be positive; got {control_timeout_seconds}.")


def _terminate_process(exit_code: int) -> NoReturn:
    """Exit without invoking Python destructors for distributed/CUDA objects."""
    with contextlib.suppress(Exception):
        sys.stdout.flush()
    with contextlib.suppress(Exception):
        sys.stderr.flush()
    os._exit(exit_code)


def _is_replica_fatal_error(error: BaseException) -> bool:
    if isinstance(error, ReplicaFatalError):
        return True
    # A completed per-request outcome exchange explicitly establishes that the
    # reported request failure is recoverable, even if its text mentions NCCL.
    if isinstance(error, DistributedRequestError):
        return False
    # Keep typed OOM recoverable even if PyTorch changes its exception hierarchy.
    out_of_memory_error = getattr(torch, "OutOfMemoryError", None)
    if isinstance(out_of_memory_error, type) and isinstance(error, out_of_memory_error):
        return False
    dist_error = getattr(dist, "DistError", None)
    if isinstance(dist_error, type) and isinstance(error, dist_error):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if isinstance(accelerator_error, type) and isinstance(error, accelerator_error):
        return True
    cuda_error = getattr(torch.cuda, "CudaError", None)
    if isinstance(cuda_error, type) and isinstance(error, cuda_error):
        return True
    # PyTorch often reports device asserts and other fatal asynchronous CUDA
    # failures as plain RuntimeError.
    if not isinstance(error, RuntimeError):
        return False
    with contextlib.suppress(Exception):
        message = str(error).lower()
        return any(marker in message for marker in UNSAFE_RUNTIME_ERROR_MARKERS)
    return False


def is_replica_fatal_error(error: BaseException) -> bool:
    """Return whether ``error`` or a wrapped cause makes process reuse unsafe."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_replica_fatal_error(current):
            return True
        # An explicitly recoverable outer error is authoritative.
        if isinstance(current, DistributedRequestError):
            return False
        out_of_memory_error = getattr(torch, "OutOfMemoryError", None)
        if isinstance(out_of_memory_error, type) and isinstance(current, out_of_memory_error):
            return False
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return False


def shard_sequence(tensor: torch.Tensor | None, dim: int, size: int, rank: int) -> torch.Tensor | None:
    """Return the contiguous sequence shard owned by ``rank``."""
    if tensor is None:
        return None
    sequence_length = tensor.shape[dim]
    if sequence_length % size != 0:
        raise ValueError(f"CP{size} does not divide tensor dimension {dim} with length {sequence_length}.")
    shard_length = sequence_length // size
    return tensor.narrow(dim, rank * shard_length, shard_length).contiguous()


def shard_sequence_or_broadcast(
    tensor: torch.Tensor | None,
    dim: int,
    size: int,
    rank: int,
) -> torch.Tensor | None:
    """Shard a per-token tensor while preserving a singleton broadcast axis."""
    if tensor is None or tensor.shape[dim] == 1:
        return tensor
    return shard_sequence(tensor, dim, size, rank)


def _video_latent_token_count(*, frames: int, height: int, width: int) -> int:
    """Mirror ``VideoLatentShape.from_pixel_shape(...).token_count()`` without importing LTX."""
    if frames < 1 or height < 1 or width < 1:
        raise ValueError(f"Video dimensions must be positive; got frames={frames}, height={height}, width={width}.")
    latent_frames = (frames - 1) // VIDEO_VAE_TEMPORAL_SCALE_FACTOR + 1
    latent_height = height // VIDEO_VAE_SPATIAL_SCALE_FACTOR
    latent_width = width // VIDEO_VAE_SPATIAL_SCALE_FACTOR
    if latent_height < 1 or latent_width < 1:
        raise ValueError(
            "Video dimensions are too small for the LTX VAE's 32x spatial compression; "
            f"got height={height}, width={width}."
        )
    return latent_frames * latent_height * latent_width


def _temporally_subsampled_frame_count(frames: int, scale_factor: int) -> int:
    """Return the frame count produced by LTX's causal ``[0, 1::scale]`` sampling."""
    if frames < 1:
        raise ValueError(f"Reference video must contain at least one frame; got {frames}.")
    if scale_factor < 1:
        raise ValueError(f"Reference temporal scale factor must be positive; got {scale_factor}.")
    return 1 + (frames - 1 + scale_factor - 1) // scale_factor


def context_parallel_video_sequence_lengths(
    *,
    height: int,
    width: int,
    num_frames: int,
    available_reference_frames: int,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int,
) -> tuple[int, int]:
    """Return Stage 1/2 target-plus-reference video token counts."""
    if height < 1 or width < 1 or height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Two-stage video dimensions must be positive and even; got {height}x{width}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive; got {num_frames}.")
    if available_reference_frames < 1:
        raise ValueError(f"Conditioning video must contain at least one frame; got {available_reference_frames}.")
    if reference_downscale_factor < 1:
        raise ValueError(f"Reference downscale factor must be positive; got {reference_downscale_factor}.")

    decoded_reference_frames = min(num_frames, available_reference_frames)
    reference_frames = _temporally_subsampled_frame_count(
        decoded_reference_frames,
        reference_temporal_scale_factor,
    )
    stage_dimensions = ((height // 2, width // 2), (height, width))
    sequence_lengths = []
    for stage_index, (stage_height, stage_width) in enumerate(stage_dimensions, start=1):
        if stage_height % reference_downscale_factor != 0 or stage_width % reference_downscale_factor != 0:
            raise ValueError(
                f"Stage {stage_index} dimensions {stage_height}x{stage_width} must be divisible by "
                f"the IC-LoRA reference downscale factor {reference_downscale_factor}."
            )
        reference_height = stage_height // reference_downscale_factor
        reference_width = stage_width // reference_downscale_factor
        if (
            reference_height % VIDEO_VAE_SPATIAL_SCALE_FACTOR != 0
            or reference_width % VIDEO_VAE_SPATIAL_SCALE_FACTOR != 0
        ):
            raise ValueError(
                f"Stage {stage_index} reference dimensions {reference_height}x{reference_width} must be "
                f"divisible by the LTX VAE spatial scale factor {VIDEO_VAE_SPATIAL_SCALE_FACTOR}."
            )
        target_tokens = _video_latent_token_count(
            frames=num_frames,
            height=stage_height,
            width=stage_width,
        )
        reference_tokens = _video_latent_token_count(
            frames=reference_frames,
            height=reference_height,
            width=reference_width,
        )
        sequence_lengths.append(target_tokens + reference_tokens)
    return sequence_lengths[0], sequence_lengths[1]


def validate_context_parallel_video_sequence_lengths(
    *,
    size: int,
    height: int,
    width: int,
    num_frames: int,
    available_reference_frames: int,
    reference_downscale_factor: int,
    reference_temporal_scale_factor: int,
) -> None:
    """Reject a request whose augmented video sequence cannot be evenly sharded."""
    if size < 1:
        raise ValueError(f"Context parallel size must be positive; got {size}.")
    if size == 1:
        return
    sequence_lengths = context_parallel_video_sequence_lengths(
        height=height,
        width=width,
        num_frames=num_frames,
        available_reference_frames=available_reference_frames,
        reference_downscale_factor=reference_downscale_factor,
        reference_temporal_scale_factor=reference_temporal_scale_factor,
    )
    indivisible = [
        f"Stage {stage_index}: {sequence_length} tokens"
        for stage_index, sequence_length in enumerate(sequence_lengths, start=1)
        if sequence_length % size != 0
    ]
    if indivisible:
        details = ", ".join(indivisible)
        raise ValueError(
            f"CP{size} cannot evenly shard the target-plus-reference video sequence ({details}). "
            f"Choose a height, width, or frame count whose sequence length is divisible by {size}."
        )


@dataclass
class ContextParallelRuntime:
    """Process-local context-parallel state for one ``torchrun`` replica."""

    size: int
    rank: int
    local_rank: int
    group: Any = None
    command_group: Any = None
    control_group: Any = None
    control_timeout_seconds: int = 30
    poisoned: bool = False
    poison_reason: str | None = None

    @classmethod
    def initialize(
        cls,
        size: int,
        *,
        timeout_seconds: int = 1800,
        control_timeout_seconds: int = 30,
    ) -> ContextParallelRuntime:
        _validate_runtime_configuration(size, timeout_seconds, control_timeout_seconds)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        if size == 1:
            return cls(
                size=1,
                rank=0,
                local_rank=local_rank,
                control_timeout_seconds=control_timeout_seconds,
            )

        if not torch.cuda.is_available():
            raise RuntimeError("Context parallel serving requires CUDA.")
        if not dist.is_available():
            raise RuntimeError("This PyTorch build does not provide torch.distributed.")
        if not dist.is_gloo_available():
            raise RuntimeError("Context parallel serving requires Gloo for request error propagation.")

        # This must be visible before ProcessGroupNCCL is constructed. Respect an
        # explicit operator override while making fail-fast teardown the default.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))

        # From this point onward, a setup exception can be asymmetric while the
        # default NCCL group already exists. Never enter normal interpreter teardown.
        try:
            world_size = dist.get_world_size()
            if world_size != size:
                raise RuntimeError(
                    f"torchrun world size ({world_size}) must equal context parallel size ({size}). "
                    "Launch one torchrun job per serving replica."
                )

            rank = dist.get_rank()
            # Followers may legitimately wait while the HTTP server is idle, so the
            # group itself needs a long timeout. Leader sends are bounded separately
            # by the watchdog in broadcast_object(). Both groups use CPU/Gloo and
            # remain independent of CUDA/NCCL failures.
            command_group = dist.new_group(backend="gloo", timeout=COMMAND_WAIT_TIMEOUT)
            control_group = dist.new_group(
                backend="gloo",
                timeout=timedelta(seconds=control_timeout_seconds),
            )
        except Exception:
            LOGGER.critical("Context-parallel process-group initialization failed", exc_info=True)
            _terminate_process(1)
        runtime = cls(
            size=size,
            rank=rank,
            local_rank=local_rank,
            group=dist.group.WORLD,
            command_group=command_group,
            control_group=control_group,
            control_timeout_seconds=control_timeout_seconds,
        )
        LOGGER.info(
            "Initialized LTX context parallelism: rank=%s local_rank=%s size=%s device=%s",
            runtime.rank,
            runtime.local_rank,
            runtime.size,
            runtime.device,
        )
        return runtime

    @property
    def enabled(self) -> bool:
        return self.size > 1

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.local_rank)

    def barrier(self) -> None:
        self.ensure_usable()
        if self.enabled:
            try:
                dist.barrier(group=self.group, device_ids=[self.local_rank])
            except Exception as exc:
                self.fail_replica("Context-parallel barrier failed", cause=exc)

    def broadcast_object(self, value: Any = None, *, timeout_seconds: float | None = None) -> Any:
        """Broadcast a small control-plane object with a bounded leader send."""
        self.ensure_usable()
        if not self.enabled:
            return value

        completed = Event()
        watchdog = None
        if self.is_leader:
            timeout = self.control_timeout_seconds if timeout_seconds is None else timeout_seconds
            if timeout <= 0:
                self.terminate(f"Command broadcast timeout must be positive; got {timeout}.")
            reason = f"Context-parallel command broadcast exceeded the {timeout:g}s control-plane timeout"

            def terminate_if_stalled() -> None:
                if completed.wait(timeout=timeout):
                    return
                LOGGER.critical("%s", reason)
                self.terminate(reason, exit_code=1)

            watchdog = Thread(
                target=terminate_if_stalled,
                name="ltx-command-broadcast-watchdog",
                daemon=True,
            )
            try:
                watchdog.start()
            except Exception as exc:
                self.terminate(
                    f"Could not start context-parallel command broadcast watchdog: {type(exc).__name__}",
                    exit_code=1,
                )

        values = [value if self.is_leader else None]
        try:
            dist.broadcast_object_list(values, src=0, group=self.command_group)
        except Exception as exc:
            self.fail_replica("Context-parallel command broadcast failed", cause=exc)
        finally:
            completed.set()
            if watchdog is not None:
                watchdog.join(timeout=1)
        return values[0]

    def exchange_request_status(self, status: dict[str, Any]) -> list[Any]:
        """Gather one bounded request outcome from every rank on the CPU control group."""
        self.ensure_usable()
        if not self.enabled:
            return [status]
        statuses: list[Any] = [None] * self.size
        try:
            dist.all_gather_object(statuses, status, group=self.control_group)
        except Exception as exc:
            self.fail_replica(
                f"Request outcome exchange exceeded the {self.control_timeout_seconds}s control-plane timeout",
                cause=exc,
            )
        return statuses

    def fail_replica(self, reason: str, *, cause: BaseException | None = None) -> NoReturn:
        """Poison this process and raise an error that must terminate the replica."""
        self.mark_poisoned(reason)
        error = ReplicaFatalError(reason)
        if cause is not None:
            raise error from cause
        raise error

    def ensure_usable(self) -> None:
        """Refuse to enter another collective after this process is poisoned."""
        if self.poisoned:
            raise ReplicaFatalError(
                f"Context-parallel replica is already poisoned: {self.poison_reason or 'unknown reason'}"
            )

    def mark_poisoned(self, reason: str) -> None:
        """Prevent graceful distributed cleanup after an uncoordinated failure."""
        if self.poisoned:
            return
        self.poisoned = True
        self.poison_reason = reason
        LOGGER.critical("Context-parallel replica is no longer usable: %s", reason)

    def terminate(self, reason: str | None = None, *, exit_code: int = 1) -> NoReturn:
        """Exit immediately, bypassing potentially blocking NCCL/CUDA teardown."""
        if not self.poisoned:
            self.mark_poisoned(reason or "Fatal distributed failure")
        _terminate_process(exit_code)

    def close(self) -> None:
        # On a divergent/failing path, graceful group destruction can itself wait
        # for missing peers. Process exit lets torchrun reap the whole replica.
        if self.enabled and not self.poisoned and dist.is_initialized():
            dist.destroy_process_group()


class RequestOutcomeCoordinator:
    """Perform the exactly-once outcome rendezvous for one distributed request."""

    def __init__(self, runtime: ContextParallelRuntime, request_id: int) -> None:
        self.runtime = runtime
        self.request_id = request_id
        self.attempted = False

    def synchronize(self, error: BaseException | None = None) -> list[dict[str, Any]]:
        if self.runtime.poisoned:
            raise ReplicaFatalError(
                f"Context-parallel replica is already poisoned: {self.runtime.poison_reason or 'unknown reason'}"
            )
        if self.attempted:
            self.runtime.fail_replica(f"Request {self.request_id} attempted its outcome exchange more than once")
        self.attempted = True

        # Successful ranks drain their queued data-plane work before declaring
        # completion. A failing rank must publish immediately: touching CUDA here
        # could wait on the same divergent NCCL collective this handshake protects.
        fatal = error is not None and is_replica_fatal_error(error)
        if error is None and self.runtime.enabled and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.runtime.device)
            except Exception as exc:
                error = exc
                fatal = True

        status = self._build_status(error, fatal=fatal)
        raw_statuses = self.runtime.exchange_request_status(status)
        statuses = self._validate_statuses(raw_statuses)
        failures = [item for item in statuses if not item["ok"]]
        fatal_failures = [item for item in failures if item["fatal"]]
        if fatal_failures:
            summary = "; ".join(
                f"rank {failure['rank']}: {failure['error_type']}: {failure['error']}" for failure in fatal_failures
            )
            self.runtime.fail_replica(f"Request {self.request_id} encountered an unsafe distributed error: {summary}")
        if failures:
            raise DistributedRequestError(self.request_id, failures)
        return statuses

    def _build_status(self, error: BaseException | None, *, fatal: bool) -> dict[str, Any]:
        if error is None:
            return {
                "request_id": self.request_id,
                "rank": self.runtime.rank,
                "ok": True,
                "fatal": False,
                "error_type": None,
                "error": None,
                "traceback": None,
            }

        try:
            message = str(error)
        except Exception:
            try:
                message = repr(error)
            except Exception:
                message = f"<{type(error).__name__} with unprintable message>"
        with contextlib.suppress(Exception):
            formatted_traceback = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            return {
                "request_id": self.request_id,
                "rank": self.runtime.rank,
                "ok": False,
                "fatal": fatal,
                "error_type": type(error).__name__,
                "error": message[:MAX_ERROR_MESSAGE_CHARS],
                "traceback": formatted_traceback[-MAX_ERROR_TRACEBACK_CHARS:],
            }
        return {
            "request_id": self.request_id,
            "rank": self.runtime.rank,
            "ok": False,
            "fatal": fatal,
            "error_type": type(error).__name__,
            "error": message[:MAX_ERROR_MESSAGE_CHARS],
            "traceback": None,
        }

    def _validate_statuses(self, raw_statuses: list[Any]) -> list[dict[str, Any]]:
        expected_ranks = set(range(self.runtime.size))
        observed_ranks: set[int] = set()
        statuses: list[dict[str, Any]] = []
        for status in raw_statuses:
            if not isinstance(status, dict):
                self.runtime.fail_replica(f"Request {self.request_id} received a malformed rank outcome: {status!r}")
            if status.get("request_id") != self.request_id:
                self.runtime.fail_replica(
                    f"Request outcome ID mismatch: expected {self.request_id}, got {status.get('request_id')!r}"
                )
            rank = status.get("rank")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank not in expected_ranks
                or rank in observed_ranks
            ):
                self.runtime.fail_replica(f"Request {self.request_id} received an invalid rank outcome: {rank!r}")
            if not isinstance(status.get("ok"), bool):
                self.runtime.fail_replica(f"Request {self.request_id} rank {rank} has a malformed ok flag")
            if not isinstance(status.get("fatal"), bool):
                self.runtime.fail_replica(f"Request {self.request_id} rank {rank} has a malformed fatal flag")
            if status["ok"] and (
                status["fatal"]
                or status.get("error_type") is not None
                or status.get("error") is not None
                or status.get("traceback") is not None
            ):
                self.runtime.fail_replica(f"Request {self.request_id} rank {rank} has inconsistent success details")
            if not status["ok"] and (
                not isinstance(status.get("error_type"), str) or not isinstance(status.get("error"), str)
            ):
                self.runtime.fail_replica(f"Request {self.request_id} rank {rank} has malformed error details")
            observed_ranks.add(rank)
            statuses.append(status)

        if observed_ranks != expected_ranks:
            self.runtime.fail_replica(
                f"Request {self.request_id} outcomes covered ranks {sorted(observed_ranks)}, "
                f"expected {sorted(expected_ranks)}"
            )
        return statuses


class UlyssesAttention:
    """All-to-all video self-attention over a context-parallel group."""

    label = "UlyssesSDPA"

    def __init__(self, runtime: ContextParallelRuntime) -> None:
        self.runtime = runtime

    def _all_to_all(self, tensor: torch.Tensor) -> torch.Tensor:
        self.runtime.ensure_usable()
        output = torch.empty_like(tensor)
        dist.all_to_all_single(output, tensor, group=self.runtime.group)
        return output

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, local_tokens, inner_dim = query.shape
        if inner_dim % heads != 0:
            raise ValueError(f"Attention width {inner_dim} is not divisible by {heads} heads.")
        if heads % self.runtime.size != 0:
            raise ValueError(
                f"Context parallel size {self.runtime.size} does not divide the attention head count {heads}."
            )

        head_dim = inner_dim // heads
        local_heads = heads // self.runtime.size

        def sequence_to_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.view(batch, local_tokens, self.runtime.size, local_heads, head_dim)
            tensor = tensor.permute(2, 0, 1, 3, 4).contiguous()
            tensor = self._all_to_all(tensor)
            tensor = tensor.permute(1, 0, 2, 3, 4)
            return tensor.reshape(batch, self.runtime.size * local_tokens, local_heads, head_dim).transpose(1, 2)

        query_heads = sequence_to_heads(query)
        key_heads = sequence_to_heads(key)
        value_heads = sequence_to_heads(value)
        output = functional.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=mask,
        )
        output = output.transpose(1, 2).reshape(
            batch,
            self.runtime.size,
            local_tokens,
            local_heads,
            head_dim,
        )
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        output = self._all_to_all(output).permute(1, 2, 0, 3, 4)
        return output.reshape(batch, local_tokens, heads * head_dim)


class GatherKeyValueAttention:
    """Gather sharded video keys/values for video-to-audio attention.

    This is correct only while every rank has the same replicated audio query,
    its keys/values are equal-sized contiguous video-sequence shards ordered by
    process-group rank, and any mask addresses the reconstructed full sequence.
    The gather then lets every rank compute the same replicated audio output.
    """

    label = "GatherKVSDPA"

    def __init__(self, runtime: ContextParallelRuntime) -> None:
        self.runtime = runtime

    def _gather_sequence(self, tensor: torch.Tensor) -> torch.Tensor:
        self.runtime.ensure_usable()
        shards = [torch.empty_like(tensor) for _ in range(self.runtime.size)]
        dist.all_gather(shards, tensor.contiguous(), group=self.runtime.group)
        return torch.cat(shards, dim=1)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_tokens, inner_dim = query.shape
        if inner_dim % heads != 0:
            raise ValueError(f"Attention width {inner_dim} is not divisible by {heads} heads.")
        key = self._gather_sequence(key)
        value = self._gather_sequence(value)
        head_dim = inner_dim // heads
        query = query.view(batch, query_tokens, heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, heads, head_dim).transpose(1, 2)
        output = functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return output.transpose(1, 2).reshape(batch, query_tokens, heads * head_dim)


_PATCHED_RUNTIME: ContextParallelRuntime | None = None


def _install_process_blocks_patch(runtime: ContextParallelRuntime) -> None:
    global _PATCHED_RUNTIME

    from ltx_core.model.transformer.model import LTXModel

    if _PATCHED_RUNTIME is not None:
        if (_PATCHED_RUNTIME.size, _PATCHED_RUNTIME.rank) != (runtime.size, runtime.rank):
            raise RuntimeError("LTX context parallelism was already installed with a different runtime.")
        return

    original_process_blocks = LTXModel._process_transformer_blocks  # noqa: SLF001 - CP wraps LTX block execution.

    @functools.wraps(original_process_blocks)
    def process_blocks_with_context_parallel(
        model: Any,
        video: Any,
        audio: Any,
        perturbations: Any,
    ) -> tuple[Any, Any]:
        full_embedded_timestep = None
        if video is not None:
            full_embedded_timestep = video.embedded_timestep
            video = replace(
                video,
                x=shard_sequence(video.x, 1, runtime.size, runtime.rank),
                timesteps=shard_sequence_or_broadcast(video.timesteps, 1, runtime.size, runtime.rank),
                embedded_timestep=shard_sequence_or_broadcast(
                    video.embedded_timestep,
                    1,
                    runtime.size,
                    runtime.rank,
                ),
                positional_embeddings=tuple(
                    shard_sequence(embedding, 2, runtime.size, runtime.rank)
                    for embedding in video.positional_embeddings
                ),
                cross_positional_embeddings=(
                    tuple(
                        shard_sequence(embedding, 2, runtime.size, runtime.rank)
                        for embedding in video.cross_positional_embeddings
                    )
                    if video.cross_positional_embeddings is not None
                    else None
                ),
                cross_scale_shift_timestep=shard_sequence_or_broadcast(
                    video.cross_scale_shift_timestep,
                    1,
                    runtime.size,
                    runtime.rank,
                ),
                cross_gate_timestep=shard_sequence_or_broadcast(
                    video.cross_gate_timestep,
                    1,
                    runtime.size,
                    runtime.rank,
                ),
            )

        video_output, audio_output = original_process_blocks(model, video, audio, perturbations)
        if video_output is not None:
            runtime.ensure_usable()
            output_shards = [torch.empty_like(video_output.x) for _ in range(runtime.size)]
            dist.all_gather(output_shards, video_output.x.contiguous(), group=runtime.group)
            video_output = replace(
                video_output,
                x=torch.cat(output_shards, dim=1),
                embedded_timestep=full_embedded_timestep,
            )
        return video_output, audio_output

    LTXModel._process_transformer_blocks = (  # noqa: SLF001
        process_blocks_with_context_parallel
    )
    _PATCHED_RUNTIME = runtime


def install_context_parallel(transformer: torch.nn.Module, runtime: ContextParallelRuntime) -> None:
    """Install Ulysses attention on a freshly built LTX transformer."""
    if not runtime.enabled:
        return

    velocity_model = getattr(transformer, "velocity_model", transformer)
    ulysses_attention = UlyssesAttention(runtime)
    gather_key_value_attention = GatherKeyValueAttention(runtime)
    block_count = 0
    for block in velocity_model.transformer_blocks:
        if block.attn1.heads % runtime.size != 0:
            raise ValueError(
                f"Context parallel size {runtime.size} does not divide video attention head count {block.attn1.heads}."
            )
        block.attn1.attention_function = ulysses_attention
        block.attn1.masked_attention_function = ulysses_attention
        block_count += 1
        if getattr(block, "video_to_audio_attn", None) is not None:
            block.video_to_audio_attn.attention_function = gather_key_value_attention
            block.video_to_audio_attn.masked_attention_function = gather_key_value_attention

    _install_process_blocks_patch(runtime)
    if runtime.is_leader:
        LOGGER.info("Installed Ulysses context parallelism on %s LTX transformer blocks", block_count)
