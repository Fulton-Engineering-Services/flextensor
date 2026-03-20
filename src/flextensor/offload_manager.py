# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import logging
import os
import re
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from flextensor.state_handler import TensorManagerState

import psutil
import torch
from proxytypes3 import ObjectWrapper
from torch import nn

from flextensor.benchmark_tensor_mode import BenchmarkReplace, NoOpBenchmark, PreloadToDevice, TensorBenchmarkMode
from flextensor.config import OffloadConfig
from flextensor.helpers import NoOpTensorManager
from flextensor.instrumentation import dump_to_directory, get_registry
from flextensor.strategy import AdaptiveStrategy
from flextensor.types import GPUMemoryUsage  # noqa: TC001

LOGGER = logging.getLogger(__name__)
_GiB = 1 << 30


def _log_diagnostic_snapshot(label: str, om: OffloadManager | None) -> None:
    """Capture and log full diagnostic context. Called only on error path.

    Every diagnostic collection is wrapped in its own try/except so that
    capture failures never mask the original exception.

    Args:
        label: Human-readable context for the error (e.g. method name, forward block name).
        om: The OffloadManager instance, or None if unavailable.
    """
    parts = [f"FlexTensor error in {label}"]

    if om is not None:
        parts.append(f"  state={om._current_state.value}")  # noqa: SLF001
        parts.append(f"  iteration={om._iteration_count}")  # noqa: SLF001
        parts.append(f"  manager={om.name!r}")

    try:
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            parts.append(
                f"  gpu{i}: alloc={alloc / _GiB:.2f}GiB reserved={reserved / _GiB:.2f}GiB total={total / _GiB:.2f}GiB"
            )
    except Exception:
        parts.append("  gpu: <unavailable>")

    try:
        vm = psutil.virtual_memory()
        parts.append(
            f"  host: used={vm.used / _GiB:.2f}GiB avail={vm.available / _GiB:.2f}GiB total={vm.total / _GiB:.2f}GiB"
        )
    except Exception:
        parts.append("  host: <unavailable>")

    parts.append(f"  pid={os.getpid()}")

    LOGGER.error("\n".join(parts), exc_info=True)


_F = TypeVar("_F", bound=Callable[..., Any])


def _error_boundary(func: _F) -> _F:
    """Decorator that logs full diagnostics on unhandled exceptions.

    Captures FlexTensor state, GPU/host memory, traceback, then re-raises.
    Zero overhead on the happy path (CPython try/except is free when no
    exception is raised).

    Only for OffloadManager methods — ``self`` must be an OffloadManager.

    Args:
        func: The OffloadManager method to wrap.

    Returns:
        Wrapped method with identical signature.
    """

    @functools.wraps(func)
    def wrapper(self: OffloadManager, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception:
            _log_diagnostic_snapshot(func.__qualname__, self)
            raise

    return wrapper  # type: ignore[return-value]


@runtime_checkable
class _ShmCoordinatorLike(Protocol):
    """Structural interface for ShmCoordinator used by the follower init path."""

    namespace: str
    is_creator: bool

    def wait_for_ready(self) -> None: ...
    def read_profile(self) -> TensorManagerState: ...


class OffloadState(Enum):
    """State of the offload manager during profiling and inference."""

    NOT_INITIALIZED = "not_initialized"
    WARMUP = "warmup"
    PROFILE = "profile"
    INFERENCE = "inference"


DEFAULT_MANAGER_NAME: str = "default"
"""Name used by :func:`get_offload_manager` when no explicit name is provided."""

# Global singleton map for OffloadManager instances
OFFLOAD_MANAGER_MAP = {}
_MANAGER_MAP_LOCK = threading.Lock()


class OffloadModelProxy(ObjectWrapper):
    """Proxy that delegates to the current model version during offload state transitions."""

    def __init__(self, model, offload_manager: OffloadManager):
        ObjectWrapper.__init__(self, model)
        # Store offload_manager directly on the proxy, bypassing ObjectWrapper delegation
        object.__setattr__(self, "offload_manager", offload_manager)

    def _get_model(self):
        """Helper method to get the current model, avoiding code duplication."""
        # Access offload_manager directly from proxy, bypassing ObjectWrapper delegation
        offload_manager = object.__getattribute__(self, "offload_manager")
        model = offload_manager.model
        if model is None:
            message = "Model not initialized. Ensure offload() has been called and the model hasn't been released."
            raise RuntimeError(message)
        return model

    def __call__(self, *args, **kwargs):
        """Intercept calls to handle state transitions."""
        return self._forward_impl(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Handle state transitions and delegate to current model."""
        return self._forward_impl(*args, **kwargs)

    def _forward_impl(self, *args, **kwargs):
        """Handle state transitions and delegate to current model."""
        # Get the current active model
        model = self._get_model()
        res = model(*args, **kwargs)
        # Call update_state to handle automatic state transitions
        # Access offload_manager directly from proxy, bypassing ObjectWrapper delegation
        offload_manager = object.__getattribute__(self, "offload_manager")
        offload_manager.update_state()
        return res

    # Special methods that need explicit delegation (not handled by __getattr__)
    # These are required for proper proxy behavior:
    # - __iter__: iteration support
    # - __enter__, __exit__: context manager protocol
    # - __dir__: introspection support

    def __iter__(self):
        """Delegate __iter__ to underlying model if it supports it."""
        model = self._get_model()
        if not hasattr(model, "__iter__"):
            raise TypeError(f"'{type(model).__name__}' object is not iterable")
        return iter(model)

    def __enter__(self):
        """Delegate __enter__ to underlying model for context manager protocol."""
        model = self._get_model()
        if not hasattr(model, "__enter__"):
            raise AttributeError(f"'{type(model).__name__}' object has no attribute '__enter__'")
        return model.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Delegate __exit__ to underlying model for context manager protocol."""
        model = self._get_model()
        if not hasattr(model, "__exit__"):
            raise AttributeError(f"'{type(model).__name__}' object has no attribute '__exit__'")
        return model.__exit__(exc_type, exc_val, exc_tb)

    def __dir__(self):
        """Delegate __dir__ to underlying model to include model's attributes."""
        model = self._get_model()
        # Combine proxy's own attributes with model's attributes
        proxy_attrs = set(object.__dir__(self))
        model_attrs = set(dir(model))
        return sorted(proxy_attrs | model_attrs)


class OffloadManager:
    """Simplified offload manager API.

    Provides a high-level interface for tensor offloading with automatic
    state management, profiling, and module patching capabilities.
    """

    def __init__(self, name: str):
        self.name = name
        self.config = OffloadConfig()
        self._tensor_manager = None
        self._initialized = False
        self._current_state = OffloadState.NOT_INITIALIZED
        self._iteration_count = 0
        self._model = None
        self._model_proxy = None
        self._patched_modules: list[nn.Module] = []

    def set_config(self, config: OffloadConfig):
        """Set configuration for the offload manager.

        Args:
            config: OffloadConfig instance with settings
        """
        self.config = config

        # Enable/disable instrumentation based on config
        registry = get_registry()
        registry.enabled = config.enable_instrumentation

    def _resolve_profile_directory(self, profile_directory: str | None) -> str:
        """Resolve profile directory from argument or config.

        Args:
            profile_directory: Explicit directory, or None to use config.profile_storage_dir.

        Returns:
            Resolved profile directory path.

        Raises:
            ValueError: If no directory is provided and config.profile_storage_dir is not set.
        """
        resolved = profile_directory or self.config.profile_storage_dir
        if resolved is None:
            raise ValueError(
                "No profile directory specified. Provide profile_directory argument "
                "or set profile_storage_dir in OffloadConfig."
            )
        return resolved

    def save_profile(self, profile_directory: str | None = None) -> None:
        """Save current offload profile to directory.

        Args:
            profile_directory: Directory to save profile to.
                If None, uses config.profile_storage_dir.

        Raises:
            RuntimeError: If tensor manager is not initialized or no profile data exists.
            ValueError: If no directory is provided and config.profile_storage_dir is not set,
                or if config.profile_read_only is True.
        """
        if self.config.profile_read_only:
            raise ValueError(
                "Cannot save profile: config.profile_read_only is True. "
                "Set profile_read_only=False in OffloadConfig to enable saving."
            )

        if self._tensor_manager is None:
            raise RuntimeError("Tensor manager not initialized. Call offload() first.")

        resolved_dir = self._resolve_profile_directory(profile_directory)
        self._tensor_manager.save_profile(resolved_dir)

    def load_profile(self, profile_directory: str | None = None, model: nn.Module | None = None) -> None:
        """Load offload profile from directory and restore state to model.

        Args:
            profile_directory: Directory containing the profile.
                If None, uses config.profile_storage_dir.
            model: Model to restore state to. If None, uses the current model.

        Raises:
            RuntimeError: If no model is available to restore state to.
            FileNotFoundError: If profile directory or file doesn't exist.
            ValueError: If no directory is provided and config.profile_storage_dir is not set.
        """
        if self._tensor_manager is None:
            self._initialize_tensor_manager()

        resolved_dir = self._resolve_profile_directory(profile_directory)
        target_model = model if model is not None else self._model
        if target_model is None:
            raise RuntimeError("No model provided and no model has been offloaded yet.")
        self._tensor_manager.load_profile(resolved_dir, target_model)

    def benchmark_transfers(self):
        """Benchmark tensor transfer speeds."""
        if self._tensor_manager is None:
            return
        # TODO: Implement transfer benchmarking

    def get_tensor_manager(self):
        """Get the internal tensor manager instance.

        Returns:
            TensorManager instance or None if not initialized
        """
        return self._tensor_manager

    def get_gpu_memory_usage(self) -> GPUMemoryUsage:
        """Get GPU memory usage by FlexTensor in inference mode.

        Returns the memory used by GPU transfer blocks and unmapped tensors
        that were moved to GPU. This method should be called after the manager
        has transitioned to inference mode (after warmup and profile phases).

        Returns:
            GPUMemoryUsage: Memory breakdown with per-component bytes and MB values.
                See `GPUMemoryUsage` for field details.

        Raises:
            RuntimeError: If called before inference mode (before all warmup
                and profile iterations have completed)

        Examples:
            >>> om = get_offload_manager()
            >>> config = OffloadConfig(module_patterns=["layers.*"])
            >>> model = om.offload(model, config=config)
            >>> # Run warmup and profile iterations
            >>> for _ in range(config.all_warmup_iters):
            ...     model(input)
            >>> # Now in inference mode, get memory usage
            >>> usage = om.get_gpu_memory_usage()
            >>> print(f"FlexTensor GPU memory: {usage.total_mb:.2f} MB")
            >>> print(f"  Transfer blocks: {usage.blocks_mb:.2f} MB")
            >>> print(f"  Unmapped tensors: {usage.unmapped_tensors_mb:.2f} MB")
        """
        if self._current_state != OffloadState.INFERENCE:
            msg = (
                f"Cannot get GPU memory usage in state {self._current_state.value}. "
                "This method is only available after transitioning to inference mode."
            )
            raise RuntimeError(msg)

        if self._tensor_manager is None:
            msg = "Tensor manager not initialized"
            raise RuntimeError(msg)

        return self._tensor_manager.get_gpu_memory_usage()

    @property
    def model(self) -> nn.Module | None:
        """Get the current active model.

        Returns:
            The current model instance or None if not initialized
        """
        return self._model

    def _initialize_tensor_manager(self):
        """Initialize TensorManager with config settings."""
        # Import here to avoid circular dependency between offload_manager and tensor_manager modules
        from flextensor.tensor_manager import TensorManager

        device_gpu = torch.device(f"cuda:{self.config.gpu_device}")

        if self.config.load_strategy is not None:
            tensor_manager_load_strategy = self.config.load_strategy
        else:
            tensor_manager_load_strategy = AdaptiveStrategy(
                scale=self.config.knapsack_scale,
                loader_type=self.config.transfer_mode,
                n_blocks=self.config.num_blocks,
                min_blocks=self.config.min_blocks,
            )

        if not self.config.enabled:
            benchmark_cls = PreloadToDevice
            # pyright: ignore[reportUnknownReturnType]
            self._tensor_manager = NoOpTensorManager(device_gpu, benchmark_cls=benchmark_cls)
        else:
            benchmark_cls: type[TensorBenchmarkMode] = BenchmarkReplace if self.config.enable_tracing else NoOpBenchmark
            self._tensor_manager = TensorManager(
                device_gpu,
                self.config.pinned_memory,
                tensor_manager_load_strategy,
                self.config.release_tensors,
                benchmark_cls=benchmark_cls,
                direct_mode=self.config.enable_direct_mode,
                use_trace_tensor=self.config.enable_tracing,
                loader_type=self.config.transfer_mode,
                min_compute_transfer_gap=self.config.compute_transfer_gap,
                rearrange_transfers=self.config.rearrange_transfers,
                blocks=self.config.num_blocks,
                remove_layers_operations=[],
                enable_untraced_tensor_discovery=self.config.enable_untraced_tensor_discovery,
                enable_module_tracker=self.config.enable_module_tracker,
                use_shm=self.config.shm_enabled,
                enable_diagnostics=self.config.enable_diagnostics,
                max_gpu_mem_fraction=self.config.max_gpu_mem_fraction,
            )

    def _transfer_hooks(self, old_model: nn.Module | None, new_model: nn.Module):  # noqa: C901
        """Transfer hooks from old model to new model, preserving module hierarchy.

        Args:
            old_model: Model to extract hooks from (None if no previous model)
            new_model: Model to register hooks on
        """
        # Skip if old model is None or models are the same object (no transfer needed)
        if old_model is None or old_model is new_model:
            return

        # Get module mappings by name
        old_modules = dict(old_model.named_modules())
        new_modules = dict(new_model.named_modules())

        for module_name, old_module in old_modules.items():
            if module_name not in new_modules:
                continue

            new_module = new_modules[module_name]

            # Transfer forward hooks (create list copy to avoid mutation during iteration)
            if forward_hooks := getattr(old_module, "_forward_hooks", None):
                for _, hook_fn in list(forward_hooks.items()):
                    new_module.register_forward_hook(hook_fn)

            # Transfer forward pre-hooks (create list copy to avoid mutation during iteration)
            if forward_pre_hooks := getattr(old_module, "_forward_pre_hooks", None):
                for _, hook_fn in list(forward_pre_hooks.items()):
                    new_module.register_forward_pre_hook(hook_fn)

            # Transfer backward hooks (create list copy to avoid mutation during iteration)
            if backward_hooks := getattr(old_module, "_backward_hooks", None):
                for _, hook_fn in list(backward_hooks.items()):
                    new_module.register_full_backward_hook(hook_fn)

            # Transfer backward pre-hooks (create list copy to avoid mutation during iteration)
            if backward_pre_hooks := getattr(old_module, "_backward_pre_hooks", None):
                for _, hook_fn in list(backward_pre_hooks.items()):
                    if hasattr(new_module, "register_full_backward_pre_hook"):
                        new_module.register_full_backward_pre_hook(hook_fn)

    def _patch_module_forward(self, module: nn.Module, offload_name: str) -> None:
        """Patch module's forward method to include offload context.

        Args:
            module: The module to patch
            offload_name: Name for the offload block

        Note:
            Stores the original forward class method as module._ft_original_forward_func for restoration.
            Uses the UNBOUND forward function from the class to avoid bound method issues when
            models are copied during state transitions.
        """

        # Skip if already patched
        if hasattr(module, "_ft_original_forward_func"):
            return

        # Get the UNBOUND forward function from the class, not a bound method.
        # This is critical: bound methods capture `self`, which breaks when models are copied.
        original_forward_func = type(module).forward
        offload_manager = self

        def patched_forward(self_module, *args, **kwargs):
            try:
                with offload_manager.offload_block(offload_name):
                    # Call the unbound forward function with self_module explicitly
                    return original_forward_func(self_module, *args, **kwargs)
            except Exception:
                _log_diagnostic_snapshot(f"forward({offload_name})", offload_manager)
                raise

        # Make patched_forward look like the original for introspection
        patched_forward = functools.wraps(original_forward_func)(patched_forward)

        # Store original for restoration (using the UNBOUND function, not bound method)
        module._ft_original_forward_func = original_forward_func  # noqa: SLF001
        module._ft_offload_name = offload_name  # noqa: SLF001

        # Bind patched_forward as a method to the module.
        # This ensures it receives `self` when called.
        module.forward = types.MethodType(patched_forward, module)  # type: ignore[method-assign]

        # Track patched module for cleanup
        self._patched_modules.append(module)

    def _restore_module_forward(self, module: nn.Module) -> None:
        """Restore module's original forward method.

        Args:
            module: The module to restore
        """
        if hasattr(module, "_ft_original_forward_func"):
            # Restore by removing the patched forward from __dict__,
            # allowing the class method to be used again
            if "forward" in module.__dict__:
                del module.__dict__["forward"]
            delattr(module, "_ft_original_forward_func")
            delattr(module, "_ft_offload_name")

    @_error_boundary
    def _transition_to_warmup(self):
        """Transition to warmup state."""

        self._tensor_manager.set_model(self._model)
        old_model = self._model
        self._model = self._tensor_manager.initialize_warmup()

        # Transfer hooks from old model to new model (if re-initializing)
        self._transfer_hooks(old_model, self._model)

        # Update proxy to point to new warmup model (if proxy already exists)
        if self._model_proxy is not None:
            self._model_proxy.__subject__ = self._model

        self._current_state = OffloadState.WARMUP
        self._iteration_count = 0

    @_error_boundary
    def _transition_to_profile(self):
        """Transition to profile state."""
        if self._tensor_manager is None:
            return

        old_model = self._model
        self._model = self._tensor_manager.initialize_profile()

        # Transfer hooks from old model to new model
        self._transfer_hooks(old_model, self._model)

        self._model_proxy.__subject__ = self._model
        self._current_state = OffloadState.PROFILE
        self._iteration_count = 0

    @_error_boundary
    def _transition_to_inference(self):
        """Transition to inference state."""
        try:
            if self._tensor_manager is None:
                return

            old_model = self._model
            self._model = self._tensor_manager.initialize_inference()

            # Transfer hooks from old model to new model
            self._transfer_hooks(old_model, self._model)

            self._model_proxy.__subject__ = self._model
            self._current_state = OffloadState.INFERENCE
            self._iteration_count = 0
        finally:
            # Dump instrumentation data if enabled and output directory is configured
            if (
                self.config.enable_instrumentation
                and self.config.instrumentation_output_dir
                and self._tensor_manager is not None
            ):
                dump_to_directory(
                    self.config.instrumentation_output_dir,
                    extra={"memory_transfer_stats": self._tensor_manager.get_memory_transfer_stats()},
                )

    def update_state(self):
        """Check and update state if necessary."""
        if self._current_state in {OffloadState.NOT_INITIALIZED, OffloadState.INFERENCE}:
            return
        self._iteration_count += 1

        if self._current_state == OffloadState.WARMUP and self._iteration_count >= self.config.warmup_iters:
            self._transition_to_profile()
        elif self._current_state == OffloadState.PROFILE and self._iteration_count >= self.config.profile_iters:
            self._transition_to_inference()

    def init(self, config: OffloadConfig | None = None):
        if config is not None:
            self.set_config(config)
        if self._tensor_manager is None:
            self._initialize_tensor_manager()

    def offload(
        self,
        model: nn.Module,
        config: OffloadConfig | None = None,
    ) -> nn.Module:
        """Offload modules in the model based on config patterns.

        Patches modules matching config.module_patterns to automatically manage
        tensor transfers. Supports wildcards in module patterns.

        Args:
            model: PyTorch model to patch
            config: OffloadConfig to use for offload (patterns from config.module_patterns)

        Returns:
            A wrapper around the model that tracks state transitions

        Examples:
            >>> model = MyModel()
            >>> om = get_offload_manager()
            >>> config = OffloadConfig(module_patterns=["embed", "layers.*", "head"])
            >>> model = om.offload(model, config)
        """
        self.init(config)

        self._tensor_manager.build_parameters_mapping(model)
        # Save old model for hook transfer before overwriting
        old_model = self._model
        self._model = model

        for module_path in self.config.module_patterns:
            self._offload_modules(self._model, self._model, "root", ["root", *module_path.split(".")])

        # Create proxy that delegates to current model
        if self._model_proxy is None:
            self._model_proxy = OffloadModelProxy(self._model, self)

        # Transfer hooks from old model to new model (if re-initializing)
        self._transfer_hooks(old_model, self._model)

        self._transition_to_warmup()

        return self._model_proxy

    def offload_block(self, name: str) -> Any:
        """Create an offload block context manager.

        Args:
            name: Name of the offload block (used for tracking)

        Returns:
            OffloadBlock context manager (Trap, WarmupTrap, TrapInfer, etc.)

        Examples:
            >>> with om.offload_block("encoder"):
            ...     output = encoder(input)
        """
        if self._tensor_manager is None:
            raise RuntimeError(
                "Tensor manager not initialized. Call offload() on a model first to initialize the manager."
            )
        return self._tensor_manager.trap(name)

    @_error_boundary
    def _initialize_from_shm(self, coordinator: _ShmCoordinatorLike, model: nn.Module) -> OffloadModelProxy:
        """Initialize as a follower from shared memory.

        Loads profile, restores state, and jumps directly to INFERENCE.

        Args:
            coordinator: ShmCoordinator in follower mode.
            model: Model constructed on meta device.

        Returns:
            Proxy-wrapped model in INFERENCE state.

        Raises:
            RuntimeError: If coordinator is a creator (not a follower).
            ValueError: If the profile's loader type is incompatible or tensors
                from the profile are missing from the model.
        """
        if coordinator.is_creator:
            raise RuntimeError("_initialize_from_shm called on creator coordinator")

        coordinator.wait_for_ready()
        state = coordinator.read_profile()

        self._initialize_tensor_manager()
        self._tensor_manager.shm_namespace = coordinator.namespace
        self._tensor_manager.restore_state(model, state)

        self._model = self._tensor_manager.initialize_warmup()
        self._model = self._tensor_manager.initialize_profile()
        self._model = self._tensor_manager.initialize_inference()

        for module_path in self.config.module_patterns:
            self._offload_modules(self._model, self._model, "root", ["root", *module_path.split(".")])

        self._current_state = OffloadState.INFERENCE

        # Wrap in proxy for API consistency with offload() — follower is already
        # in INFERENCE so update_state() is a no-op on each forward call.
        if self._model_proxy is None:
            self._model_proxy = OffloadModelProxy(self._model, self)
        self._model_proxy.__subject__ = self._model
        return self._model_proxy

    def release(self):
        # Restore original forward methods for all patched modules
        for module in self._patched_modules:
            self._restore_module_forward(module)
        self._patched_modules.clear()

        if self._tensor_manager is not None:
            self._tensor_manager.release_memory()
            self._tensor_manager.shutdown()
            self._tensor_manager = None
        self._initialized = False
        self._current_state = OffloadState.NOT_INITIALIZED
        self._model_proxy = None
        self._model = None

    def _offload_modules(
        self,
        parent_module: nn.Module,
        module: nn.Module,
        module_name: str,
        path_chunks: list[str],
    ):
        """Recursively offload modules matching the path pattern.

        Args:
            parent_module: Parent module (for replacing child modules)
            module: Current module being examined
            module_name: Name of current module
            path_chunks: Remaining path chunks to match (supports wildcards)
        """
        if len(path_chunks) == 0:
            return

        chunk_module_name = path_chunks[0]
        # Escape regex special characters except * and ?
        chunk_module_name_escaped = re.escape(chunk_module_name)
        # Convert wildcards: * -> .* and ? -> .
        chunk_module_name_escaped = chunk_module_name_escaped.replace(r"\*", ".*").replace(r"\?", ".")

        if not re.fullmatch(chunk_module_name_escaped, module_name):
            return

        if len(path_chunks) == 1:
            # This is the target module - patch its forward method
            offload_name = f"{parent_module.__class__.__name__}.{module_name}"
            self._patch_module_forward(module, offload_name)
            return

        # Recurse into child modules
        children = list(module.named_children())
        if len(children) == 0:
            return

        for name, child_module in children:
            self._offload_modules(module, child_module, name, path_chunks[1:])


@dataclass(frozen=True, slots=True)
class _ManagerEntry:
    """Internal entry pairing an OffloadManager with its owner thread."""

    manager: OffloadManager
    owner_thread: int


def get_offload_manager(name: str = DEFAULT_MANAGER_NAME) -> OffloadManager:
    """Get or create an OffloadManager singleton.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        OffloadManager instance

    Raises:
        RuntimeError: If the named manager was created by a different thread.

    Examples:
        >>> om = get_offload_manager()
        >>> om.set_config(OffloadConfig(gpu_device=0))
    """
    current_thread = threading.get_ident()
    with _MANAGER_MAP_LOCK:
        if name not in OFFLOAD_MANAGER_MAP:
            OFFLOAD_MANAGER_MAP[name] = _ManagerEntry(
                manager=OffloadManager(name),
                owner_thread=current_thread,
            )
        entry = OFFLOAD_MANAGER_MAP[name]
    if entry.owner_thread != current_thread:
        raise RuntimeError(
            f"OffloadManager '{name}' belongs to thread {entry.owner_thread}, "
            f"but accessed from thread {current_thread}. "
            f"Each named manager must be used from a single thread."
        )
    return entry.manager


# Convenience functions for the default offload manager


def init(config: OffloadConfig | None = None, name: str = DEFAULT_MANAGER_NAME):
    om = get_offload_manager(name)
    om.init(config)


def offload(model: nn.Module, config: OffloadConfig | None = None, name: str = DEFAULT_MANAGER_NAME) -> nn.Module:
    """Offload modules using the specified offload manager.

    Args:
        model: PyTorch model to patch.
        config: OffloadConfig to use for offload (patterns from config.module_patterns).
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        The patched model.
    """
    om = get_offload_manager(name)
    return om.offload(model, config)


def set_config(config: OffloadConfig, name: str = DEFAULT_MANAGER_NAME):
    """Set config for the specified offload manager.

    Args:
        config: OffloadConfig to apply.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.set_config(config)


def save_profile(profile_directory: str | None = None, name: str = DEFAULT_MANAGER_NAME) -> None:
    """Save profile using the specified offload manager.

    Args:
        profile_directory: Directory to save profile to.
            If None, uses config.profile_storage_dir.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.save_profile(profile_directory)


def load_profile(
    profile_directory: str | None = None, model: nn.Module | None = None, name: str = DEFAULT_MANAGER_NAME
) -> None:
    """Load profile using the specified offload manager.

    Args:
        profile_directory: Directory containing the profile.
            If None, uses config.profile_storage_dir.
        model: Model to restore state to. If None, uses the current model.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.load_profile(profile_directory, model)


def offload_from_profile(
    model: nn.Module,
    profile_directory: str,
    config: OffloadConfig | None = None,
    name: str = DEFAULT_MANAGER_NAME,
) -> nn.Module:
    """Initialize, load a saved profile, and offload the model in one step.

    Convenience wrapper that combines :func:`init`, :func:`load_profile`, and
    :func:`offload`.  The profile must be loaded *before* offloading so that
    ``initialize_warmup`` sees the saved state and skips warmup/profiling,
    going straight to inference mode.

    Args:
        model: PyTorch model to offload.
        profile_directory: Directory containing the saved profile.
        config: OffloadConfig to apply.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        The offloaded model proxy, ready for inference.
    """
    init(config=config, name=name)
    load_profile(profile_directory, model=model, name=name)
    return offload(model, config=config, name=name)


def offload_block(block_name: str, name: str = DEFAULT_MANAGER_NAME) -> Any:
    """Create offload block using the specified offload manager.

    Args:
        block_name: Name for the offload block.
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    return om.offload_block(block_name)


def release(name: str = DEFAULT_MANAGER_NAME):
    """Release the specified offload manager.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.
    """
    om = get_offload_manager(name)
    om.release()


def get_gpu_memory_usage(name: str = DEFAULT_MANAGER_NAME) -> GPUMemoryUsage:
    """Get GPU memory usage by the specified offload manager.

    Returns the memory used by GPU transfer blocks and unmapped tensors.
    Must be called after the manager has transitioned to inference mode.

    Args:
        name: Name for the offload manager. Defaults to DEFAULT_MANAGER_NAME.

    Returns:
        GPUMemoryUsage dataclass with memory breakdown in bytes and megabytes.

    Raises:
        RuntimeError: If called before inference mode.
    """
    om = get_offload_manager(name)
    return om.get_gpu_memory_usage()
