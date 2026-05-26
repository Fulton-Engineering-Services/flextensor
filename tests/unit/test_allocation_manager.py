# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from flextensor.allocation_block import AllocationBlock, AllocationManager
from flextensor.host_pinning import HostPinner


class TestAllocationManager:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.manager = AllocationManager(host_pinner=HostPinner())
        # Force CPU device for tests (MPS doesn't support float64)
        self.device = "cpu"

    def test_manager_initialization(self):
        """Test that AllocationManager initializes correctly."""
        manager = AllocationManager(host_pinner=HostPinner())
        assert isinstance(manager.blocks, list)
        assert len(manager.blocks) == 0

    def test_manager_allocate(self):
        """Test the manager's direct tensor allocation functionality."""
        # Test basic allocation
        block = self.manager.block(device="cpu")
        block.add(torch.ones(10, dtype=torch.float32, device="cpu"))
        views = block.allocate()
        assert len(views) == 1
        assert views[0].shape == (10,)
        assert views[0].dtype == torch.float32
        assert views[0].device.type == "cpu"

        # Test multi-dimensional allocation
        block2 = self.manager.block(device="cpu")
        block2.add(torch.ones(5, 3, dtype=torch.int64))
        views2 = block2.allocate()
        assert len(views2) == 1
        assert views2[0].shape == (5, 3)
        assert views2[0].dtype == torch.int64
        assert views2[0].device.type == "cpu"

    def test_manager_block_creation(self):
        """Test that the manager can create blocks and tracks them."""
        # Create a block
        block = self.manager.block(device="cpu")
        assert isinstance(block, AllocationBlock)
        assert len(self.manager.blocks) == 1
        assert self.manager.blocks[0] is block

        # Create multiple blocks
        block2 = self.manager.block(device="cpu")
        assert len(self.manager.blocks) == 2
        assert self.manager.blocks[1] is block2


class TestMultipleBlockWorkflow:
    """Test the complex workflow from the main code with multiple blocks."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.manager = AllocationManager(host_pinner=HostPinner())

    def test_two_block_workflow(self):
        """Test the workflow with two separate blocks as shown in main code."""
        # Create first block with tensors from main code
        b = self.manager.block()
        t1 = torch.ones(12, dtype=torch.float32, device="cpu")
        t2 = torch.ones(15, dtype=torch.float32, device="cpu") * 2
        t3 = torch.ones(13, dtype=torch.int32, device="cpu") * 3

        b.add(t1)
        b.add(t2)
        b.add(t3)
        views = b.allocate()

        # Verify first block
        assert len(views) == 3
        assert torch.allclose(views[0], t1)
        assert torch.allclose(views[1], t2)
        assert torch.equal(views[2], t3)

        # Create second block with different tensors
        b2 = self.manager.block()
        t4 = torch.ones(18, dtype=torch.float32)
        t5 = torch.ones(19, dtype=torch.float32, device="cpu") * 2

        b2.add(t4)
        b2.add(t5)
        views2 = b2.allocate()

        # Verify second block
        assert len(views2) == 2
        assert torch.allclose(views2[0], t4)
        assert torch.allclose(views2[1], t5)

        # Verify both blocks are tracked by manager
        assert len(self.manager.blocks) == 2

    def test_create_max_block_functionality(self):
        """Test the create_max_block functionality from main code."""
        # Create blocks of different sizes
        b1 = self.manager.block()
        b1.add(torch.ones(10, dtype=torch.float32))
        b1.allocate()

        b2 = self.manager.block()
        b2.add(torch.ones(20, dtype=torch.float32, device="cpu"))
        b2.allocate()

        b3 = self.manager.block()
        b3.add(torch.ones(5, dtype=torch.int32))
        b3.allocate()

        # Create max block - should be sized for the largest block (b2)
        max_block = self.manager.create_max_block(device="cpu")
        assert isinstance(max_block, AllocationBlock)

        # Verify the max block size matches the largest committed block
        expected_size = b2.block.numel() * b2.block.element_size()
        actual_size = max_block.block.numel() * max_block.block.element_size()
        assert actual_size == expected_size

    def test_create_max_block_with_no_committed_blocks(self):
        """Test create_max_block when no blocks are committed."""
        # Create blocks but don't commit them
        b1 = self.manager.block()
        b1.add(torch.ones(10, dtype=torch.float32))
        # Don't commit

        # Should create a block with size 0
        max_block = self.manager.create_max_block(device="cpu")
        assert max_block.block.numel() == 0


class TestViewProjectionWorkflow:
    """Test view projection functionality from the main code."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.manager = AllocationManager(host_pinner=HostPinner())

    def test_view_projection_workflow(self):
        """Test the complete view projection workflow from main code."""
        # Create and populate first block
        b = self.manager.block()
        t1 = torch.ones(5, dtype=torch.float32, device="cpu")
        t2 = torch.ones(3, dtype=torch.float32, device="cpu") * 2
        b.add(t1)
        b.add(t2)
        views = b.allocate()

        # Create and populate second block
        b2 = self.manager.block()
        t3 = torch.ones(4, dtype=torch.float32)
        t4 = torch.ones(6, dtype=torch.float32, device="cpu") * 3
        b2.add(t3)
        b2.add(t4)
        views2 = b2.allocate()

        # Create GPU block
        gpu_block = self.manager.create_max_block(device="cpu")

        # Test view projection
        gpu_views1, _block_view1 = b.project_views(gpu_block)
        gpu_views2, _block_view2 = b2.project_views(gpu_block)

        # Verify projections
        assert len(gpu_views1) == len(views)
        assert len(gpu_views2) == len(views2)

        # Verify projected views have correct shapes and dtypes
        for orig_view, gpu_view in zip(views, gpu_views1, strict=False):
            assert orig_view.shape == gpu_view.shape
            assert orig_view.dtype == gpu_view.dtype

        for orig_view, gpu_view in zip(views2, gpu_views2, strict=False):
            assert orig_view.shape == gpu_view.shape
            assert orig_view.dtype == gpu_view.dtype


class TestDataCopyingWorkflow:
    """Test data copying functionality from the main code."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.manager = AllocationManager(host_pinner=HostPinner())

    def test_copy_to_workflow(self):
        """Test the copy_to workflow from main code."""
        # Create source block with known data
        b = self.manager.block()
        t1 = torch.ones(8, dtype=torch.float32) * 5
        t2 = torch.ones(6, dtype=torch.int32) * 7
        b.add(t1)
        b.add(t2)
        b.allocate()

        # Create target block
        gpu_block = self.manager.create_max_block(device="cpu")  # Use CPU for testing
        gpu_views, block_view = b.project_views(gpu_block)

        # Copy data
        b.copy_to(block_view)

        # Verify data was copied correctly
        assert torch.allclose(gpu_views[0], torch.ones(8, dtype=torch.float32) * 5)
        assert torch.equal(gpu_views[1], torch.ones(6, dtype=torch.int32) * 7)

    def test_multiple_copy_workflow(self):
        """Test copying from multiple blocks to the same GPU block."""
        # Create first source block
        b1 = self.manager.block()
        t1 = torch.ones(4, dtype=torch.float32) * 10
        b1.add(t1)
        b1.allocate()

        # Create second source block
        b2 = self.manager.block()
        t2 = torch.ones(3, dtype=torch.float32, device="cpu") * 20
        b2.add(t2)
        b2.allocate()

        # Create GPU block large enough for both
        gpu_block = self.manager.create_max_block(device="cpu")  # Use CPU for testing

        # Project and copy first block
        gpu_views1, block_view1 = b1.project_views(gpu_block)
        b1.copy_to(block_view1)

        # Verify first copy
        assert torch.allclose(gpu_views1[0], torch.ones(4, dtype=torch.float32) * 10)

        # Project and copy second block (should overwrite)
        gpu_views2, block_view2 = b2.project_views(gpu_block)
        b2.copy_to(block_view2)

        # Verify second copy
        assert torch.allclose(gpu_views2[0], torch.ones(3, dtype=torch.float32, device="cpu") * 20)


class TestComplexIntegrationWorkflow:
    """Test the complete integration workflow from main code."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.manager = AllocationManager(host_pinner=HostPinner())

    def test_full_main_workflow(self):
        """Test the complete workflow exactly as shown in main code."""
        # Replicate the exact main code workflow
        b = self.manager.block()
        t1 = torch.ones(12, dtype=torch.float32, device="cpu")
        t2 = torch.ones(15, dtype=torch.float32, device="cpu") * 2
        t3 = torch.ones(13, dtype=torch.int32, device="cpu") * 3
        b.add(t1)
        b.add(t2)
        b.add(t3)
        views = b.allocate()

        b2 = self.manager.block()
        t4 = torch.ones(18, dtype=torch.float32)
        t5 = torch.ones(19, dtype=torch.float32, device="cpu") * 2
        b2.add(t4)
        b2.add(t5)
        views2 = b2.allocate()

        gpu_block = self.manager.create_max_block(device="cpu")  # Use CPU for testing
        gpu_views1, block_view1 = b.project_views(gpu_block)
        gpu_views2, block_view2 = b2.project_views(gpu_block)

        # Copy data and verify integrity
        b.copy_to(block_view1)

        # Verify all data in first block is correct
        for orig, gpu_view in zip([t1, t2, t3], gpu_views1, strict=False):
            if orig.dtype in [torch.float32, torch.float64]:
                assert torch.allclose(orig, gpu_view)
            else:
                assert torch.equal(orig, gpu_view)

        # Copy second block and verify
        b2.copy_to(block_view2)

        for orig, gpu_view in zip([t4, t5], gpu_views2, strict=False):
            if orig.dtype in [torch.float32, torch.float64]:
                assert torch.allclose(orig, gpu_view)
            else:
                assert torch.equal(orig, gpu_view)

        # Verify block properties match expectations
        assert len(views) == 3
        assert len(views2) == 2
        assert len(gpu_views1) == 3
        assert len(gpu_views2) == 2
