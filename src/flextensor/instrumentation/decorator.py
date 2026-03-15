# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decorator for instrumenting class initialization.

This module provides the @instrumentable decorator that automatically captures
initialization arguments when a class is instantiated.
"""

import functools
import inspect
from typing import Any, TypeVar

from flextensor.instrumentation.registry import InstrumentationRegistry
from flextensor.instrumentation.serializers import serialize_args

T = TypeVar("T")


def instrumentable(cls: type[T]) -> type[T]:
    """Class decorator that instruments __init__ to capture initialization arguments.

    When instrumentation is enabled via the registry, this decorator captures
    all arguments passed to __init__ and registers them for later analysis.

    Args:
        cls: The class to instrument.

    Returns:
        The instrumented class with wrapped __init__.

    Example:
        >>> @instrumentable
        ... class MyComponent:
        ...     def __init__(self, arg1: str, arg2: int = 10):
        ...         self.arg1 = arg1
        ...         self.arg2 = arg2
        ...
        >>> # Enable instrumentation
        >>> registry = InstrumentationRegistry.get_instance()
        >>> registry.enabled = True
        >>> obj = MyComponent("test", arg2=20)
        >>> records = registry.get_records()
        >>> records[0].args  # {'arg1': 'test', 'arg2': 20}
    """
    original_init = cls.__init__

    @functools.wraps(original_init)
    def instrumented_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Call original __init__ first
        original_init(self, *args, **kwargs)

        # Check if instrumentation is enabled
        registry = InstrumentationRegistry.get_instance()
        if not registry.enabled:
            return

        # Capture init arguments
        captured_args = _capture_init_args(original_init, args, kwargs)

        # Serialize and register
        serialized_args = serialize_args(captured_args)
        registry.register(
            class_name=cls.__name__,
            module_path=f"{cls.__module__}.{cls.__name__}",
            args=serialized_args,
        )

    cls.__init__ = instrumented_init  # type: ignore[method-assign]
    return cls


def _capture_init_args(
    init_method: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Capture initialization arguments using inspect.signature.

    Args:
        init_method: The original __init__ method.
        args: Positional arguments passed to __init__.
        kwargs: Keyword arguments passed to __init__.

    Returns:
        Dictionary mapping parameter names to their values.
    """
    sig = inspect.signature(init_method)
    params = list(sig.parameters.items())

    captured: dict[str, Any] = {}

    # Skip 'self' parameter
    params = [(name, param) for name, param in params if name != "self"]

    # Map positional args to parameter names
    for i, arg_value in enumerate(args):
        if i < len(params):
            param_name = params[i][0]
            captured[param_name] = arg_value

    # Add keyword arguments
    captured.update(kwargs)

    # Add default values for parameters not provided
    for param_name, param in params:
        if param_name not in captured:
            if param.default is not inspect.Parameter.empty:
                captured[param_name] = param.default
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                # *args - skip
                pass
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # **kwargs - skip
                pass

    return captured
