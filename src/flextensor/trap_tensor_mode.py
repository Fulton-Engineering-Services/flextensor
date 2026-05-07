# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.overrides import TorchFunctionMode

try:
    import torch._dynamo as _dynamo
except ImportError:  # pragma: no cover - torch built without dynamo
    _dynamo = None  # type: ignore[assignment]


def _graph_break() -> None:
    """Insert a Dynamo graph break at a trap boundary.

    Called from each trap's ``__enter__`` and ``__exit__`` so that
    ``torch.compile`` splits its graph at offload-block boundaries.  The
    tensor ops *between* traps compile into subgraphs; the loader's eager
    tensor movement runs in the resume functions Dynamo emits between
    subgraphs.

    No-op when ``torch._dynamo`` is not importable (e.g. a torch build
    without Dynamo support).  Any future failure from ``graph_break()``
    is deliberately not caught — real failures from Dynamo should surface
    rather than silently degrade trap isolation.
    """
    if _dynamo is None:
        return
    _dynamo.graph_break()


class Trap(TorchFunctionMode):
    def __init__(self, tensor_manager, trace_id, device_gpu):
        self.current_trace_id = trace_id
        self.device_gpu = device_gpu
        self.tensor_layer_loader = tensor_manager.tensor_layer_loader
        self.tensors_ids = set()
        self.tensor_manager = tensor_manager
        self.start_event = tensor_manager.trap_start_event
        self.end_event = tensor_manager.trap_end_event
        self._nesting_guard = tensor_manager.trap_nesting_guard

    def __enter__(self):
        _graph_break()
        try:
            self._nesting_guard.acquire(self.current_trace_id)
            self.tensor_layer_loader.enter(self.current_trace_id)
            self.start_event.record()
            return super().__enter__()
        except BaseException:
            # Pair the enter-side break: if __enter__ raises, __exit__ never
            # runs, so we emit the matching exit-side break here to keep
            # Dynamo from waiting on a continuation that will never arrive.
            _graph_break()
            raise

    def __exit__(self, _type, _value, _traceback):
        self.end_event.record()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)

        self.tensor_layer_loader.exit(self.current_trace_id)

        self.tensor_manager.record_all(self.current_trace_id, self.tensors_ids, duration_ms)
        self._nesting_guard.release()

        _graph_break()
        super().__exit__(_type, _value, _traceback)

    def __torch_function__(self, func, _types, args, kwargs=None):
        # rewrite args
        new_args = []
        for arg in args:
            new_arg = arg
            if self.tensor_manager.is_traced(arg):
                tensor_id = id(arg)
                tensor = self.tensor_layer_loader.get(tensor_id)
                # collect tensor information
                self.tensors_ids.add(tensor_id)
                if tensor is not None:
                    new_arg = tensor
            new_args.append(new_arg)

        new_kwargs = kwargs
        if kwargs is not None:
            new_kwargs = {}
            for name, arg in kwargs.items():
                new_arg = arg
                if self.tensor_manager.is_traced(arg):
                    tensor_id = id(arg)
                    tensor = self.tensor_layer_loader.get(tensor_id)
                    # collect tensor information
                    self.tensors_ids.add(tensor_id)
                    if tensor is not None:
                        new_arg = tensor
                new_kwargs[name] = new_arg

        return func(*new_args, **(new_kwargs or {}))


class WarmupTrap(TorchFunctionMode):
    def __init__(self, tensor_manager, trace_id, device_gpu):
        self.current_trace_id = trace_id
        self.device_gpu = device_gpu
        self.tensors_ids = set()
        self.tensor_manager = tensor_manager
        self._nesting_guard = tensor_manager.trap_nesting_guard

    def __enter__(self):
        _graph_break()
        try:
            self._nesting_guard.acquire(self.current_trace_id)
            if self.tensor_manager.module_tracker is not None:
                self.tensor_manager.module_tracker.enter_trap(self.current_trace_id)
            return super().__enter__()
        except BaseException:
            # See Trap.__enter__: pair the graph break on the failure path.
            _graph_break()
            raise

    def __exit__(self, _type, _value, _traceback):
        if self.tensor_manager.module_tracker is not None:
            self.tensor_manager.module_tracker.exit_trap(self.current_trace_id)
        self.tensor_manager.record_tensors(self.current_trace_id, self.tensors_ids)
        self._nesting_guard.release()
        _graph_break()
        super().__exit__(_type, _value, _traceback)

    def _update_tensors_ids(self, args, kwargs):
        for arg in args:
            if self.tensor_manager.is_traced(arg):
                tensor_id = id(arg)
                self.tensors_ids.add(tensor_id)
        if kwargs is not None:
            for _name, arg in kwargs.items():
                if self.tensor_manager.is_traced(arg):
                    tensor_id = id(arg)
                    self.tensors_ids.add(tensor_id)

    def __torch_function__(self, func, _types, args, kwargs=None):
        self._update_tensors_ids(args, kwargs)

        new_args = []
        release_args = []
        for arg in args:
            new_arg = arg
            if self.tensor_manager.is_traced(arg) and arg.device != self.device_gpu:
                new_arg = arg.to(device=self.device_gpu, copy=True)
                torch.cuda.synchronize()
                release_args.append(new_arg)
            new_args.append(new_arg)

        new_kwargs = kwargs
        if kwargs is not None:
            new_kwargs = {}
            for name, arg in kwargs.items():
                new_arg = arg
                if self.tensor_manager.is_traced(arg) and arg.device != self.device_gpu:
                    new_arg = arg.to(device=self.device_gpu, copy=True)
                    torch.cuda.synchronize()
                    release_args.append(new_arg)
                new_kwargs[name] = new_arg

        # TODO: skip compute, we only need trace tensors
        res = func(*new_args, **(new_kwargs or {}))
        torch.cuda.synchronize()
        # release loaded tensors
        for tensor in release_args:
            del tensor

        return res


class TrapInfer(TorchFunctionMode):
    def __init__(self, tensor_manager, trace_id, device_gpu):
        self.current_trace_id = trace_id
        self.device_gpu = device_gpu
        self.tensor_layer_loader = tensor_manager.tensor_layer_loader
        self.tensor_manager = tensor_manager

    def __enter__(self):
        _graph_break()
        try:
            self.tensor_layer_loader.enter(self.current_trace_id)
            return super().__enter__()
        except BaseException:
            # See Trap.__enter__: pair the graph break on the failure path.
            _graph_break()
            raise

    def __exit__(self, _type, _value, _traceback):
        self.tensor_layer_loader.exit(self.current_trace_id)
        _graph_break()
        super().__exit__(_type, _value, _traceback)

    def __torch_function__(self, func, types, args, kwargs=None):
        new_args = []
        for arg in args:
            new_arg = arg
            if self.tensor_manager.is_traced(arg):
                tensor_id = id(arg)
                tensor = self.tensor_layer_loader.get(tensor_id)
                if tensor is not None:
                    new_arg = tensor
            new_args.append(new_arg)
        new_kwargs = kwargs
        if kwargs is not None:
            new_kwargs = {}
            for name, arg in kwargs.items():
                new_arg = arg
                if self.tensor_manager.is_traced(arg):
                    tensor_id = id(arg)
                    tensor = self.tensor_layer_loader.get(tensor_id)
                    if tensor is not None:
                        new_arg = tensor
                new_kwargs[name] = new_arg

        return func(*new_args, **(new_kwargs or {}))


class TrapDirect:
    def __init__(self, tensor_manager, trace_id, device_gpu):
        self.current_trace_id = trace_id
        self.device_gpu = device_gpu
        self.tensor_layer_loader = tensor_manager.tensor_layer_loader
        self.tensors_ids = set()
        self.tensor_manager = tensor_manager
        self.start_event = tensor_manager.trap_start_event
        self.end_event = tensor_manager.trap_end_event
        self._nesting_guard = tensor_manager.trap_nesting_guard

    def __enter__(self):
        _graph_break()
        try:
            self._nesting_guard.acquire(self.current_trace_id)
            self.tensor_layer_loader.enter(self.current_trace_id)
            self.start_event.record()
            return self
        except BaseException:
            # See Trap.__enter__: pair the graph break on the failure path.
            _graph_break()
            raise

    def __exit__(self, _type, _value, _traceback):
        self.end_event.record()
        self.end_event.synchronize()
        duration_ms = self.start_event.elapsed_time(self.end_event)
        self.tensor_manager.record_duration(self.current_trace_id, duration_ms)

        self.tensor_layer_loader.exit(self.current_trace_id)
        self._nesting_guard.release()
        _graph_break()


class TrapInferDirect:
    def __init__(self, tensor_manager, trace_id, device_gpu):
        self.current_trace_id = trace_id
        self.device_gpu = device_gpu
        self.tensor_layer_loader = tensor_manager.tensor_layer_loader
        self.tensor_manager = tensor_manager

    def __enter__(self):
        _graph_break()
        try:
            self.tensor_layer_loader.enter(self.current_trace_id)
            return self
        except BaseException:
            # See Trap.__enter__: pair the graph break on the failure path.
            _graph_break()
            raise

    def __exit__(self, _type, _value, _traceback):
        self.tensor_layer_loader.exit(self.current_trace_id)
        _graph_break()
