# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for bound method rebinding in model copy operations.

This module tests that bound methods stored in __dict__ are correctly rebound
to new module instances when copying modules. This is critical for vLLM's
CustomOp classes which store `_forward_method = self.forward_cuda` in __dict__.

Bug scenario: When create_model_with_shared_tensors or extend_nn_module copies
a module, bound methods should be rebound to the new module, not remain bound
to the original module.
"""

import copy
import types

import torch
import torch.nn as nn

from flextensor.tensor_processors import create_model_with_shared_tensors


class CustomOpModule(nn.Module):
    """Mock module that mimics vLLM's CustomOp pattern.

    vLLM's CustomOp stores bound methods in __dict__ during __init__:
        self._forward_method = self.dispatch_forward()  # returns self.forward_cuda

    When this module is copied, _forward_method should be rebound to the new
    instance, not remain bound to the original.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        # Store a bound method in __dict__ (mimics vLLM's CustomOp pattern)
        self._forward_method = self.forward_impl

    def forward_impl(self, x):
        """Implementation method that gets bound and stored in _forward_method."""
        # Return self's id to verify which instance is executing
        return id(self)

    def forward(self, x):
        """Forward calls the stored bound method."""
        return self._forward_method(x)


class NestedModelWithCustomOp(nn.Module):
    """Model containing a CustomOp-style module to test nested copying."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 4)
        self.custom_op = CustomOpModule()
        self.output = nn.Linear(4, 4)

    def forward(self, x):
        x = self.embed(x)
        x = self.custom_op(x)
        return self.output(x)


class TestBoundMethodRebinding:
    """Test cases for bound method rebinding during module copy operations."""

    def test_bound_method_stored_in_dict(self):
        """Verify that CustomOpModule stores a bound method in __dict__."""
        module = CustomOpModule()

        # Verify _forward_method is a bound method
        assert hasattr(module, "_forward_method")
        assert isinstance(module._forward_method, types.MethodType)
        assert module._forward_method.__self__ is module

        # Verify forward returns the module's id
        dummy_input = torch.randn(1, 4)
        result = module.forward(dummy_input)
        assert result == id(module)

    def test_shallow_copy_bound_method_issue(self):
        """Demonstrate the bug: shallow copy doesn't rebind methods."""
        original = CustomOpModule()
        copied = copy.copy(original)

        # After shallow copy, _forward_method is still bound to original!
        # This is the bug we want to fix
        assert copied._forward_method.__self__ is original  # Bug: should be copied
        assert copied._forward_method.__self__ is not copied  # Bug confirmed

        # When we call forward on the copy, it executes on the original!
        dummy_input = torch.randn(1, 4)
        result = copied.forward(dummy_input)
        assert result == id(original)  # Bug: returns original's id, not copy's id

    def test_create_model_with_shared_tensors_rebinds_methods(self):
        """Test that create_model_with_shared_tensors rebinds bound methods.

        This test will FAIL until the bug is fixed.
        """
        original = CustomOpModule()
        copied = create_model_with_shared_tensors(original)

        # After create_model_with_shared_tensors, _forward_method should be rebound
        assert hasattr(copied, "_forward_method")
        assert isinstance(copied._forward_method, types.MethodType)

        # The bound method should be bound to the NEW module, not the original
        # THIS ASSERTION WILL FAIL UNTIL THE BUG IS FIXED
        assert copied._forward_method.__self__ is copied, (
            f"Bug: _forward_method is bound to {id(copied._forward_method.__self__)} "
            f"but should be bound to copied module {id(copied)}"
        )
        assert copied._forward_method.__self__ is not original

        # When we call forward on the copy, it should execute on the copy
        dummy_input = torch.randn(1, 4)
        result = copied.forward(dummy_input)
        assert result == id(copied), (
            f"Bug: forward() returned {result} (original's id is {id(original)}) "
            f"but should return {id(copied)} (copy's id)"
        )

    def test_nested_model_bound_method_rebinding(self):
        """Test that nested modules with bound methods are properly handled.

        This test will FAIL until the bug is fixed.
        """
        original = NestedModelWithCustomOp()
        copied = create_model_with_shared_tensors(original)

        # The nested custom_op's _forward_method should be rebound to the new instance
        original_custom_op = original.custom_op
        copied_custom_op = copied.custom_op

        # Verify they are different instances
        assert copied_custom_op is not original_custom_op

        # The bound method in the copy should be bound to the copied module
        # THIS ASSERTION WILL FAIL UNTIL THE BUG IS FIXED
        assert copied_custom_op._forward_method.__self__ is copied_custom_op, (
            f"Bug: nested custom_op._forward_method is bound to "
            f"{id(copied_custom_op._forward_method.__self__)} "
            f"but should be bound to {id(copied_custom_op)}"
        )

        # forward() should return the copy's id
        dummy_input = torch.randn(1, 4)
        result = copied_custom_op.forward(dummy_input)
        assert result == id(copied_custom_op)

    def test_tensors_are_shared_after_copy(self):
        """Verify that tensors are shared (not copied) between original and copy."""
        original = CustomOpModule()
        copied = create_model_with_shared_tensors(original)

        # The weight parameter should be shared (same tensor id)
        assert id(original.weight) == id(copied.weight)
        assert original.weight is copied.weight

    def test_multiple_bound_methods_rebinding(self):
        """Test module with multiple bound methods in __dict__.

        This test will FAIL until the bug is fixed.
        """

        class MultiMethodModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))
                # Multiple bound methods stored in __dict__
                self._method_a = self.impl_a
                self._method_b = self.impl_b

            def impl_a(self):
                return ("a", id(self))

            def impl_b(self):
                return ("b", id(self))

        original = MultiMethodModule()
        copied = create_model_with_shared_tensors(original)

        # Both methods should be rebound
        assert copied._method_a.__self__ is copied
        assert copied._method_b.__self__ is copied

        # Calling them should return the copy's id
        assert copied._method_a()[1] == id(copied)
        assert copied._method_b()[1] == id(copied)
