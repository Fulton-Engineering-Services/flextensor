# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ShmCoordinator — creator/follower orchestration."""

import time

import pytest

from flextensor.shm.coordinator import ShmCoordinator
from flextensor.shm.namespace import SHM_PROTOCOL_VERSION
from flextensor.state_handler import TensorManagerState


@pytest.fixture
def namespace():
    return f"ft_test_{int(time.time() * 1000) % 100000}"


def _make_minimal_state() -> TensorManagerState:
    return TensorManagerState(
        loader_type="allocation_block_transfer",
        tensor_id_to_name_map={},
        allocation_ordered={},
        label_to_size_map={},
        block_sizes={},
        load_strategy={},
        release_strategy={},
        label_to_block_id={},
        stats=[],
        transfer_to_compute_map={},
        view_tensors_ids=[],
        view_tensors_names=[],
        gpu_tensors_names=[],
        shm_block_name_map={},
    )


class TestShmCoordinatorCreator:
    """Tests for ShmCoordinator in creator role."""

    def test_first_process_becomes_creator(self, namespace):
        """First process to create coordinator becomes creator."""
        coord = ShmCoordinator(namespace)
        try:
            assert coord.is_creator is True
        finally:
            coord.close()

    def test_creator_can_write_and_notify(self, namespace):
        """Creator can write profile and notify followers."""
        coord = ShmCoordinator(namespace)
        try:
            coord.write_profile(_make_minimal_state())
            coord.notify_ready()
        finally:
            coord.close()


class TestShmCoordinatorFollower:
    """Tests for ShmCoordinator in follower role."""

    def test_follower_detects_existing_shm(self, namespace):
        """Second coordinator detects existing SHM and becomes follower."""
        creator = ShmCoordinator(namespace)
        try:
            creator.write_profile(_make_minimal_state())
            creator.notify_ready()
            follower = ShmCoordinator(namespace)
            try:
                assert follower.is_creator is False
            finally:
                follower.close()
        finally:
            creator.close()

    def test_follower_reads_profile(self, namespace):
        """Follower can read profile written by creator."""
        state = _make_minimal_state()
        creator = ShmCoordinator(namespace)
        try:
            creator.write_profile(state)
            creator.notify_ready()
            follower = ShmCoordinator(namespace)
            try:
                loaded = follower.read_profile()
                assert loaded.loader_type == state.loader_type
            finally:
                follower.close()
        finally:
            creator.close()


class TestShmCoordinatorRoleGuards:
    """Tests for role enforcement on write_profile/read_profile."""

    def test_follower_cannot_write_profile(self, namespace):
        """write_profile raises RuntimeError on follower."""
        creator = ShmCoordinator(namespace)
        try:
            creator.write_profile(_make_minimal_state())
            creator.notify_ready()
            follower = ShmCoordinator(namespace)
            try:
                with pytest.raises(RuntimeError, match="follower"):
                    follower.write_profile(_make_minimal_state())
            finally:
                follower.close()
        finally:
            creator.close()

    def test_creator_cannot_read_profile(self, namespace):
        """read_profile raises RuntimeError on creator."""
        creator = ShmCoordinator(namespace)
        try:
            with pytest.raises(RuntimeError, match="creator"):
                creator.read_profile()
        finally:
            creator.close()


class TestShmCoordinatorVersionGate:
    """Tests for version gate enforcement."""

    def test_version_mismatch_raises(self, namespace):
        """Follower rejects SHM from a different FlexTensor version."""
        creator = ShmCoordinator(namespace)
        try:
            from flextensor.shm.coord_block import CoordBlockHeader

            fake_header = CoordBlockHeader(
                flextensor_version="0.0.0+fake",
                protocol_version=SHM_PROTOCOL_VERSION,
            )
            fake_header.write_to(creator._coord_shm.block.buf, offset=0)

            with pytest.raises(RuntimeError, match="FlexTensor version mismatch"):
                ShmCoordinator(namespace)
        finally:
            creator.close()

    def test_version_mismatch_cleans_up_shm(self, namespace):
        """Version mismatch closes SHM handle (no leak)."""
        creator = ShmCoordinator(namespace)
        try:
            from flextensor.shm.coord_block import CoordBlockHeader

            fake_header = CoordBlockHeader(
                flextensor_version="0.0.0+fake",
                protocol_version=SHM_PROTOCOL_VERSION,
            )
            fake_header.write_to(creator._coord_shm.block.buf, offset=0)

            with pytest.raises(RuntimeError):
                follower = ShmCoordinator(namespace)
                # Should not reach here, but if it does, clean up
                follower.close()  # pragma: no cover
        finally:
            creator.close()
