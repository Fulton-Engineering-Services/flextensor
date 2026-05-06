# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that AllocationBlock and RawBlockController preserve tensor stride layouts.

These tests verify the fix for Fortran-contiguous (column-major) tensors created
by operations like ``weight.t()`` — used by vLLM's modelopt FP8 quantization.
Without stride preservation, packing into a block silently converts column-major
data to row-major, breaking downstream kernels that expect a specific layout.
"""

import torch

from flextensor.allocation_block import AllocationBlock
from flextensor.host_pinning import HostPinner
from flextensor.loaders import RawBlockController
from flextensor.utils import is_dense_layout


def _make_fortran_contiguous(rows: int, cols: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a Fortran-contiguous (column-major) tensor, mimicking ``weight.t()``."""
    base = torch.arange(rows * cols, dtype=dtype).reshape(cols, rows)
    return base.t()


class TestAllocationBlockStridePreservation:
    """Verify AllocationBlock preserves tensor strides through pack/copy."""

    def test_fortran_contiguous_view_has_original_strides(self):
        """Views for Fortran-contiguous tensors must keep column-major strides."""
        tensor = _make_fortran_contiguous(4, 3)
        assert not tensor.is_contiguous(), "precondition: tensor should be Fortran-contiguous, not C-contiguous"
        assert tensor.is_contiguous(memory_format=torch.contiguous_format) is False
        original_strides = tensor.stride()

        block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        block.add(tensor)
        views = block.allocate()

        assert views[0].stride() == original_strides
        assert views[0].shape == tensor.shape

    def test_fortran_contiguous_data_layout_preserved_after_copy(self):
        """After copy_, the raw bytes in the block must match column-major order."""
        tensor = _make_fortran_contiguous(4, 3)
        original_data = tensor.clone()
        original_strides = tensor.stride()

        block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        block.add(tensor)
        views = block.allocate()

        view = views[0]
        assert view.stride() == original_strides
        assert torch.equal(view, original_data)

        # The raw bytes in the view's storage region must match column-major
        # layout — i.e. iterating over the flat storage should give elements
        # in column-major order (col0, col1, col2, ...).
        nbytes = view.numel() * view.element_size()
        raw = torch.empty(0, dtype=torch.uint8, device="cpu")
        raw.set_(view.untyped_storage(), view.storage_offset() * view.element_size(), (nbytes,))
        flat_from_block = raw.view(view.dtype)

        raw_orig = torch.empty(0, dtype=torch.uint8, device="cpu")
        raw_orig.set_(original_data.untyped_storage(), 0, (nbytes,))
        flat_from_orig = raw_orig.view(original_data.dtype)

        assert torch.equal(flat_from_block, flat_from_orig), (
            "Raw byte order in the block must match the original column-major layout"
        )

    def test_c_contiguous_tensor_unchanged(self):
        """C-contiguous tensors should still work exactly as before."""
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert tensor.is_contiguous()
        original_strides = tensor.stride()

        block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        block.add(tensor)
        views = block.allocate()

        assert views[0].stride() == original_strides
        assert torch.equal(views[0], tensor)

    def test_mixed_contiguous_and_fortran_tensors(self):
        """A block with both C-contiguous and Fortran-contiguous tensors."""
        c_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        f_tensor = _make_fortran_contiguous(4, 3)

        block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        block.add(c_tensor)
        block.add(f_tensor)
        views = block.allocate()

        assert views[0].stride() == c_tensor.stride()
        assert views[1].stride() == f_tensor.stride()
        assert torch.equal(views[0], c_tensor)
        assert torch.equal(views[1], f_tensor)

    def test_non_dense_strided_falls_back_to_c_contiguous(self):
        """Non-dense tensors (gaps in memory) must be packed as C-contiguous."""
        base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        sliced = base[:, ::2]  # shape (4, 3), stride (5, 2) — has gaps
        expected_data = sliced.clone()  # clone() produces C-contiguous copy

        block = AllocationBlock(device="cpu", host_pinner=HostPinner())
        block.add(sliced)
        views = block.allocate()

        view = views[0]
        assert view.is_contiguous(), "Non-dense tensors should fall back to C-contiguous strides"
        assert view.shape == sliced.shape
        assert torch.equal(view, expected_data)


class TestRawBlockControllerStridePreservation:
    """Verify RawBlockController preserves tensor strides through combine/reconstruct."""

    def test_fortran_contiguous_roundtrip_preserves_strides(self):
        """combine + reconstruct must preserve Fortran-contiguous strides."""
        tensor = _make_fortran_contiguous(4, 3)
        original_data = tensor.clone()
        original_strides = tensor.stride()
        nbytes = tensor.numel() * tensor.element_size()

        combined = torch.zeros(nbytes, dtype=torch.uint8, device="cpu")

        controller = RawBlockController.__new__(RawBlockController)
        metadata = controller.combine_tensors([tensor], combined)
        reconstructed = controller.reconstruct_original_shapes(combined, metadata)

        result = reconstructed[0]
        assert result.stride() == original_strides, (
            f"Strides not preserved: expected {original_strides}, got {result.stride()}"
        )
        assert result.shape == original_data.shape
        assert torch.equal(result, original_data)

    def test_fortran_contiguous_raw_byte_order_preserved(self):
        """Raw bytes in the combined block must be in column-major order."""
        tensor = _make_fortran_contiguous(4, 3)
        nbytes = tensor.numel() * tensor.element_size()

        # Capture original raw byte order
        raw_orig = torch.empty(0, dtype=torch.uint8, device="cpu")
        raw_orig.set_(tensor.untyped_storage(), 0, (nbytes,))
        expected_bytes = raw_orig.clone()

        combined = torch.zeros(nbytes, dtype=torch.uint8, device="cpu")
        controller = RawBlockController.__new__(RawBlockController)
        controller.combine_tensors([tensor], combined)

        assert torch.equal(combined, expected_bytes), (
            "Raw bytes in combined block must match original column-major layout"
        )

    def test_c_contiguous_roundtrip_unchanged(self):
        """C-contiguous tensors must roundtrip without any change."""
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        original_data = tensor.clone()
        original_strides = tensor.stride()
        nbytes = tensor.numel() * tensor.element_size()

        combined = torch.zeros(nbytes, dtype=torch.uint8, device="cpu")
        controller = RawBlockController.__new__(RawBlockController)
        metadata = controller.combine_tensors([tensor], combined)
        reconstructed = controller.reconstruct_original_shapes(combined, metadata)

        result = reconstructed[0]
        assert result.stride() == original_strides
        assert torch.equal(result, original_data)

    def test_non_dense_strided_roundtrip_produces_correct_data(self):
        """Non-dense tensors are made contiguous before packing; data must survive."""
        base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        sliced = base[:, ::2]  # shape (4, 3), stride (5, 2) — has gaps
        expected_data = sliced.clone()
        nbytes = sliced.numel() * sliced.element_size()

        combined = torch.zeros(nbytes, dtype=torch.uint8, device="cpu")
        controller = RawBlockController.__new__(RawBlockController)
        metadata = controller.combine_tensors([sliced], combined)
        reconstructed = controller.reconstruct_original_shapes(combined, metadata)

        result = reconstructed[0]
        assert result.shape == expected_data.shape
        assert torch.equal(result, expected_data)

    def test_mixed_tensors_roundtrip(self):
        """Multiple tensors with different layouts roundtrip correctly."""
        c_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        f_tensor = _make_fortran_contiguous(4, 3)
        total_bytes = (c_tensor.numel() + f_tensor.numel()) * 4

        combined = torch.zeros(total_bytes, dtype=torch.uint8, device="cpu")
        controller = RawBlockController.__new__(RawBlockController)

        c_clone = c_tensor.clone()
        f_clone = f_tensor.clone()

        metadata = controller.combine_tensors([c_tensor, f_tensor], combined)
        reconstructed = controller.reconstruct_original_shapes(combined, metadata)

        assert reconstructed[0].stride() == c_clone.stride()
        assert reconstructed[1].stride() == f_clone.stride()
        assert torch.equal(reconstructed[0], c_clone)
        assert torch.equal(reconstructed[1], f_clone)


class TestIsDenseLayout:
    """Verify is_dense_layout correctly classifies tensor memory layouts."""

    def test_c_contiguous(self):
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert tensor.is_contiguous()
        assert is_dense_layout(tensor) is True

    def test_fortran_contiguous(self):
        tensor = _make_fortran_contiguous(4, 3)
        assert not tensor.is_contiguous()
        assert is_dense_layout(tensor) is True

    def test_strided_with_gaps(self):
        base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
        sliced = base[:, ::2]  # shape (4, 3), stride (5, 2)
        assert is_dense_layout(sliced) is False

    def test_scalar(self):
        assert is_dense_layout(torch.tensor(1.0)) is True

    def test_1d_contiguous(self):
        assert is_dense_layout(torch.arange(10)) is True

    def test_1d_strided_with_gaps(self):
        base = torch.arange(10)
        sliced = base[::3]  # elements 0, 3, 6, 9 — stride (3,)
        assert is_dense_layout(sliced) is False

    def test_empty_tensor(self):
        assert is_dense_layout(torch.empty(0)) is True

    def test_3d_permuted_dense(self):
        tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).permute(2, 0, 1)
        assert not tensor.is_contiguous()
        assert is_dense_layout(tensor) is True

    def test_3d_sliced_not_dense(self):
        base = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5)
        sliced = base[:, ::2, :]  # shape (3, 2, 5), stride (20, 10, 1)
        assert is_dense_layout(sliced) is False
