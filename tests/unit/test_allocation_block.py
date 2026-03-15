# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from flextensor.allocation_block import AllocationBlock


class TestAllocationBlock:
    def setup_method(self):
        """Setup test fixtures before each test method."""
        # Create some tensors for testing
        # Note: Using float32 instead of float64 for MPS compatibility
        # Force CPU device for tests
        self.tensors = [
            torch.ones(12, dtype=torch.float32, device="cpu"),
            (torch.ones(15, dtype=torch.float32, device="cpu") * 2).to("cpu"),
            (torch.ones(13, dtype=torch.int32, device="cpu") * 3).to("cpu"),
        ]
        self.block = AllocationBlock(device="cpu")

    def test_allocate_and_allocate(self):
        # Allocate all tensors
        for tensor in self.tensors:
            self.block.add(tensor)
        # Commit and get the views
        views = self.block.allocate()
        # Check that the returned views have the same content as the original tensors
        assert len(views) == len(self.tensors)
        for orig, view in zip(self.tensors, views, strict=False):
            # Views should have same shape and dtype
            assert orig.shape == view.shape
            assert orig.dtype == view.dtype
            # Views should have same content
            if orig.dtype in [torch.float32, torch.float64]:
                assert torch.allclose(orig, view)
            else:
                assert torch.equal(orig, view)
        # Check that the data is correct
        assert torch.allclose(views[0], torch.ones(12, dtype=torch.float32))
        assert torch.allclose(views[1], torch.ones(15, dtype=torch.float32) * 2)
        assert torch.equal(views[2], torch.ones(13, dtype=torch.int32) * 3)

    def test_view_from_block(self):
        # Test the _view_from_block method for correct dtype and shape
        base = self.block._make_base_block(64, device="cpu")
        t_i32 = self.block._view_from_block(base, torch.int32, shape=(16,), offset_bytes=0)
        t_f64 = self.block._view_from_block(base, torch.float64, shape=(8,), offset_bytes=0)
        assert t_i32.shape == (16,)
        assert t_i32.dtype == torch.int32
        assert t_f64.shape == (8,)
        assert t_f64.dtype == torch.float64
        # Fill one view and check reinterpretation
        t_i32[:] = torch.arange(16, dtype=torch.int32)
        # The same memory, so t_f64 should see the same bytes
        assert t_f64.untyped_storage().data_ptr() == t_i32.untyped_storage().data_ptr()

    def test_offset_and_stride(self):
        # Test offset_bytes and stride
        base = self.block._make_base_block(32, device="cpu")
        # Offset by 16 bytes (4 float32 elements)
        t = self.block._view_from_block(base, torch.float32, shape=(2,), offset_bytes=16)
        assert t.storage_offset() == 4
        assert t.shape == (2,)
        assert t.dtype == torch.float32

    def test_copy_to_cpu_block(self):
        """Test copy_to with CPU target block."""
        # Create source block with one tensor
        source_block = AllocationBlock(device="cpu")
        test_tensor = torch.ones(10, dtype=torch.float32) * 5
        source_block.add(test_tensor)
        source_block.allocate()

        # Create target CPU block
        target_block = AllocationBlock(device="cpu", block_size=source_block.block.numel())

        # Copy to target block
        result = source_block.copy_to(target_block.block, non_blocking=True)

        # Should return None for blocking copy
        assert result is None

        # Verify data was copied correctly
        target_view = source_block._view_from_block(target_block.block, torch.float32, (10,), 0)
        assert torch.allclose(target_view, test_tensor)

    def test_copy_to_cuda_block(self):
        """Test copy_to with CUDA target block."""
        # Create source block with one tensor
        source_block = AllocationBlock(device="cpu")
        test_tensor = torch.ones(8, dtype=torch.float32) * 3
        source_block.add(test_tensor)
        source_block.allocate()

        # Skip test if CUDA not available
        if not torch.cuda.is_available():
            print("CUDA not available, skipping CUDA copy test")
            return

        # Create target CUDA block
        target_block = AllocationBlock(device="cuda", block_size=source_block.block.numel())

        # Copy to CUDA block with non-blocking
        gpu_views, gpu_block_view = source_block.project_views(target_block)
        source_block.copy_to(gpu_block_view, non_blocking=True)

        # Should return a CUDA event for non-blocking GPU copy
        event = torch.Event(device=gpu_views[0].device)
        event.record()

        # Synchronize before checking results
        event.synchronize()

        # Verify data was copied correctly to GPU
        assert torch.allclose(gpu_views[0].cpu(), test_tensor)
