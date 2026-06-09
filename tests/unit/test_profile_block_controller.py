# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`flextensor.profile_block_controller`.

These tests run on either CPU or CUDA. The controller is allocator-only
on the device passed in, so the GPU vs. CPU device argument is the only
thing that changes the underlying storage location.
"""

from __future__ import annotations

import pytest
import torch

from flextensor.collectors import IterativeLayerStatistics
from flextensor.profile_block_controller import (
    _DEFAULT_SLOT_ALIGNMENT_BYTES,
    ProfileBlockController,
    _SlotMeta,
    _tensor_byte_view,
)
from flextensor.utils import is_dense_layout


def _align_up(offset: int, alignment: int = _DEFAULT_SLOT_ALIGNMENT_BYTES) -> int:
    """Mirror :meth:`ProfileBlockController._build_slot_meta`'s rounding so
    tests can predict block layouts without duplicating the formula. The
    default mirrors :data:`_DEFAULT_SLOT_ALIGNMENT_BYTES` (matches
    :attr:`AllocationBlock.memory_alignment`).
    """
    return -(-offset // alignment) * alignment


def _device() -> torch.device:
    """Return CUDA when available, otherwise CPU.

    The controller works on either; CPU keeps these tests CI-friendly.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _make_tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "shared": torch.randn(8, 4, dtype=torch.float32),
        "priv1a": torch.randn(16, 8, dtype=torch.float32),
        "priv1b": torch.randn(4, 4, dtype=torch.float32),
        "priv2": torch.randn(32, 4, dtype=torch.float32),
    }


def _build_two_label_setup() -> tuple[
    list[IterativeLayerStatistics],
    dict[int, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Two labels, one shared tensor between them.

    Returns (layer_stats, tensors_map_by_id, named_tensors).
    """
    named = _make_tensors()
    tensors_map = {id(t): t for t in named.values()}
    stats = [
        IterativeLayerStatistics(
            label="L1",
            tensor_ids=[id(named["shared"]), id(named["priv1a"]), id(named["priv1b"])],
        ),
        IterativeLayerStatistics(
            label="L2",
            tensor_ids=[id(named["shared"]), id(named["priv2"])],
        ),
    ]
    return stats, tensors_map, named


class TestProfileBlockControllerBudgetPreflight:
    def test_raises_when_block_exceeds_budget(self) -> None:
        stats, tensors_map, _ = _build_two_label_setup()

        with pytest.raises(ValueError, match="profile_mode='getter'"):
            ProfileBlockController(stats, tensors_map, _device(), gpu_budget_bytes=1)

    def test_allows_block_within_budget(self) -> None:
        stats, tensors_map, _ = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device(), gpu_budget_bytes=1 << 30)

        assert ctrl.block_size > 0

    def test_none_budget_skips_preflight(self) -> None:
        stats, tensors_map, _ = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device(), gpu_budget_bytes=None)

        assert ctrl.block_size > 0


class TestProfileBlockControllerLayout:
    def test_block_size_partition_matches_shared_and_max_label(self) -> None:
        """Predicted layout mirrors :meth:`_layout_shared_prefix` and
        :meth:`_layout_rotating_region`: each slot's ``start`` is rounded up
        via :func:`_align_up` before ``nbytes`` is added.
        """
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())

        shared_t = named["shared"]
        shared_bytes = shared_t.numel() * shared_t.element_size()
        assert ctrl.shared_size == shared_bytes

        def _label_rotating_size(*priv_tensors: torch.Tensor) -> int:
            cursor = ctrl.shared_size
            for t in priv_tensors:
                cursor = _align_up(cursor) + t.numel() * t.element_size()
            return cursor - ctrl.shared_size

        l1_priv = _label_rotating_size(named["priv1a"], named["priv1b"])
        l2_priv = _label_rotating_size(named["priv2"])

        assert ctrl.rotating_size == max(l1_priv, l2_priv)
        assert ctrl.block_size == ctrl.shared_size + ctrl.rotating_size

    def test_shared_ids_only_when_referenced_by_multiple_labels(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())

        shared_ids = list(ctrl.shared_tensor_ids())
        assert shared_ids == [id(named["shared"])]

    def test_view_map_covers_every_offloadable_tensor_id(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        view_map = ctrl.get_tensor_id_to_view_mapping()

        assert set(view_map.keys()) == {id(t) for t in named.values()}

    def test_views_have_original_shapes_and_dtypes(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        view_map = ctrl.get_tensor_id_to_view_mapping()

        for original in named.values():
            view = view_map[id(original)]
            assert view.shape == original.shape
            assert view.dtype == original.dtype

    def test_skips_tensors_not_in_tensors_map(self) -> None:
        """``layer_stats`` may reference ids not in ``tensors_map`` (e.g. already
        moved to GPU as non-offloaded). The controller must drop them silently.
        """
        named = _make_tensors()
        ghost_id = 0xDEADBEEF
        tensors_map = {id(named["priv1a"]): named["priv1a"]}
        stats = [
            IterativeLayerStatistics(label="L1", tensor_ids=[id(named["priv1a"]), ghost_id]),
        ]

        ctrl = ProfileBlockController(stats, tensors_map, _device())

        assert set(ctrl.get_tensor_id_to_view_mapping().keys()) == {id(named["priv1a"])}

    def test_empty_layer_stats_yields_zero_size_blocks(self) -> None:
        ctrl = ProfileBlockController([], {}, _device())

        assert ctrl.block_size == 0
        assert ctrl.shared_size == 0
        assert ctrl.rotating_size == 0
        assert ctrl.get_gpu_memory_bytes() == 0
        assert ctrl.get_tensor_id_to_view_mapping() == {}


class TestProfileBlockControllerData:
    def test_shared_prefix_loaded_at_construction(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())

        view = ctrl.get(id(named["shared"]))
        assert view is not None
        assert torch.equal(view.cpu(), named["shared"])

    def test_enter_packs_label_private_tensors(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())

        ctrl.enter("L1")
        assert torch.equal(ctrl.get(id(named["priv1a"])).cpu(), named["priv1a"])
        assert torch.equal(ctrl.get(id(named["priv1b"])).cpu(), named["priv1b"])

        ctrl.enter("L2")
        assert torch.equal(ctrl.get(id(named["priv2"])).cpu(), named["priv2"])

    def test_shared_prefix_survives_subsequent_enters(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.enter("L1")
        ctrl.enter("L2")

        assert torch.equal(ctrl.get(id(named["shared"])).cpu(), named["shared"])

    def test_enter_unknown_label_is_noop(self) -> None:
        """Unknown labels (patched modules not in ``layer_stats``, e.g.
        vLLM's ``logits_processor`` reached via ``_dummy_sampler_run``)
        are tolerated as a no-op and leave the shared prefix and any
        previously-loaded rotating region untouched. Mirrors
        :meth:`flextensor.loaders.TensorLayerLoader.enter`.
        """
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.enter("nonexistent")
        assert torch.equal(ctrl.get(id(named["shared"])).cpu(), named["shared"])

        ctrl.enter("L1")
        ctrl.enter("nonexistent")
        assert torch.equal(ctrl.get(id(named["priv1a"])).cpu(), named["priv1a"])
        assert torch.equal(ctrl.get(id(named["priv1b"])).cpu(), named["priv1b"])
        assert torch.equal(ctrl.get(id(named["shared"])).cpu(), named["shared"])

    def test_enter_label_with_only_shared_tensors_is_noop(self) -> None:
        """Label whose tensors are all in the shared prefix has no rotating
        slots — ``enter`` returns without re-reading the source tensors and
        without touching the rotating region. The shared view stays intact.
        """
        torch.manual_seed(0)
        s1 = torch.randn(8, 4, dtype=torch.float32)
        s2 = torch.randn(4, 4, dtype=torch.float32)
        tensors_map = {id(s1): s1, id(s2): s2}
        stats = [
            IterativeLayerStatistics(label="L1", tensor_ids=[id(s1), id(s2)]),
            IterativeLayerStatistics(label="L2", tensor_ids=[id(s1), id(s2)]),
        ]

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        assert ctrl.rotating_size == 0

        ctrl.enter("L1")
        ctrl.enter("L2")

        assert torch.equal(ctrl.get(id(s1)).cpu(), s1)
        assert torch.equal(ctrl.get(id(s2)).cpu(), s2)

    def test_exit_is_noop_and_does_not_affect_views(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.enter("L1")
        ctrl.exit("L1")

        # View references are unchanged after exit; data still reflects L1.
        assert torch.equal(ctrl.get(id(named["priv1a"])).cpu(), named["priv1a"])


class TestProfileBlockControllerTeardown:
    def test_teardown_drops_views_and_blocks(self) -> None:
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.teardown(model=None, tensors_map=tensors_map)

        assert ctrl.get_tensor_id_to_view_mapping() == {}
        assert ctrl.get_gpu_memory_bytes() == 0
        for original in named.values():
            assert ctrl.get(id(original)) is None

    def test_teardown_restores_dict_entries_to_originals(self) -> None:
        """Regression: dict-shaped models replace entries wholesale with block
        views, so teardown must rewrite the container back to the originals or
        the returned dict keeps pinning the freed block.
        """
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        view_map = ctrl.get_tensor_id_to_view_mapping()

        # Patch a dict model with the views, as TensorReplacementProcessor does.
        model = {name: view_map[id(t)] for name, t in named.items()}
        assert all(model[name] is view_map[id(t)] for name, t in named.items())

        ctrl.teardown(model, tensors_map)

        # Every entry must point back at the original tensor, not a block view.
        for name, original in named.items():
            assert model[name] is original

    def test_teardown_restores_param_data_on_module(self) -> None:
        device = _device()
        torch.manual_seed(1)

        weight_a_init = torch.randn(4, 8, dtype=torch.float32)
        weight_b_init = torch.randn(8, 4, dtype=torch.float32)
        shared_init = torch.randn(2, 8, dtype=torch.float32)

        # Capture initial bytes by value so teardown restoration is observable
        # even after the controller has packed the rotating region.
        weight_a_baseline = weight_a_init.clone()
        weight_b_baseline = weight_b_init.clone()
        shared_baseline = shared_init.clone()

        weight_a = torch.nn.Parameter(weight_a_init, requires_grad=False)
        weight_b = torch.nn.Parameter(weight_b_init, requires_grad=False)
        shared = torch.nn.Parameter(shared_init, requires_grad=False)

        # Two-leaf module so the parameter walk has to recurse.
        leaf_a = torch.nn.Module()
        leaf_a.weight = weight_a
        leaf_b = torch.nn.Module()
        leaf_b.weight = weight_b

        root = torch.nn.Module()
        root.shared = shared
        root.leaf_a = leaf_a
        root.leaf_b = leaf_b

        tensors_map = {
            id(weight_a): weight_a,
            id(weight_b): weight_b,
            id(shared): shared,
        }
        stats = [
            IterativeLayerStatistics(label="L1", tensor_ids=[id(shared), id(weight_a)]),
            IterativeLayerStatistics(label="L2", tensor_ids=[id(shared), id(weight_b)]),
        ]

        ctrl = ProfileBlockController(stats, tensors_map, device)
        view_map = ctrl.get_tensor_id_to_view_mapping()

        # Patch the model to use views (mimicking what TensorReplacementProcessor
        # does in the real flow).
        weight_a.data = view_map[id(weight_a)]
        weight_b.data = view_map[id(weight_b)]
        shared.data = view_map[id(shared)]

        ctrl.enter("L1")  # writes priv slots; shared prefix already populated.

        ctrl.teardown(root, tensors_map)

        # After teardown, ``.data`` must point back at the original tensors.
        assert torch.equal(weight_a.data.cpu(), weight_a_baseline)
        assert torch.equal(weight_b.data.cpu(), weight_b_baseline)
        assert torch.equal(shared.data.cpu(), shared_baseline)

    def test_teardown_restores_tensor_valued_attrs(self) -> None:
        """Tensor-valued instance attrs (e.g. ``weight.scale`` on FP8/quantized
        params) are rebound to profile-block views by the view patcher on the
        *shared* parameter object. ``.data`` restoration alone leaves the attr
        pointing into the block, which keeps the storage alive and feeds stale
        profile views into inference. ``teardown`` must rebind the attr back to
        the original tensor object.
        """
        device = _device()
        torch.manual_seed(3)

        weight = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.float32), requires_grad=False)
        # Plain ``Tensor`` (not ``nn.Parameter``) attached as an attribute --
        # this is the shape that leaks, because the patcher substitutes the view
        # object wholesale rather than swapping ``.data`` in place.
        scale = torch.randn(4, dtype=torch.float32)
        weight.scale = scale

        weight_baseline = weight.detach().clone()
        scale_baseline = scale.clone()
        scale_orig_ptr = scale.data_ptr()

        tensors_map = {id(weight): weight, id(scale): scale}
        stats = [
            IterativeLayerStatistics(label="L1", tensor_ids=[id(weight), id(scale)]),
        ]

        ctrl = ProfileBlockController(stats, tensors_map, device)
        view_map = ctrl.get_tensor_id_to_view_mapping()

        # Mimic ``TensorReplacementProcessor``: ``.data`` repointed at the view,
        # and the plain-tensor attr rebound to the scale's block view -- a
        # *different* object than the original ``scale``.
        ctrl.enter("L1")
        weight.data = view_map[id(weight)]
        weight.scale = view_map[id(scale)]
        assert weight.scale is not scale
        assert weight.scale.data_ptr() != scale_orig_ptr

        ctrl.teardown(model=None, tensors_map=tensors_map)

        # The attribute is rebound to the original object, not the released view.
        assert weight.scale is scale
        assert weight.scale.data_ptr() == scale_orig_ptr
        assert torch.equal(weight.scale.cpu(), scale_baseline)
        # ``.data`` is restored too, and the GPU block is released.
        assert torch.equal(weight.data.cpu(), weight_baseline)
        assert ctrl.get_gpu_memory_bytes() == 0

    def test_snapshot_tensor_attrs_only_captures_tensor_instance_attrs(self) -> None:
        """``_snapshot_tensor_attrs`` records public, tensor-valued instance
        attributes and ignores non-tensor attrs and dunder/private names, so
        teardown does not clobber unrelated metadata.
        """
        weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32), requires_grad=False)
        scale = torch.randn(4, dtype=torch.float32)
        weight.scale = scale
        weight.quant_method = "fp8"  # non-tensor metadata must be ignored

        snapshot = ProfileBlockController._snapshot_tensor_attrs(weight)

        assert snapshot == {"scale": scale}
        assert snapshot["scale"] is scale

    def test_teardown_aggregates_restore_failures_and_still_releases_block(self) -> None:
        """Per-tid restoration failures must not skip ``shutdown()``: the GPU
        block is released, all failing tids end up in the chained error, and
        ``__cause__`` points at the first underlying exception.
        """
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        assert ctrl.get_gpu_memory_bytes() > 0

        priv1a_id = id(named["priv1a"])
        priv2_id = id(named["priv2"])

        # Replace two entries in ``tensors_map`` with proxy objects whose
        # ``.data = ...`` setter raises; the ``shared`` and ``priv1b`` entries
        # remain real so the success-path branch is also exercised.
        class _RaisingDataProxy:
            def __init__(self, message: str) -> None:
                self._message = message

            def __setattr__(self, name: str, value: object) -> None:
                if name == "data":
                    raise ValueError(self._message)
                super().__setattr__(name, value)

        patched_map = dict(tensors_map)
        patched_map[priv1a_id] = _RaisingDataProxy("priv1a restoration failed")
        patched_map[priv2_id] = _RaisingDataProxy("priv2 restoration failed")

        with pytest.raises(RuntimeError) as exc_info:
            ctrl.teardown(model=None, tensors_map=patched_map)

        msg = str(exc_info.value)
        assert "failed to restore 2 tensor(s)" in msg
        assert str(priv1a_id) in msg
        assert str(priv2_id) in msg
        # ``__cause__`` chains to the first underlying exception (priv1a is
        # iterated before priv2 because the controller preserves layer-stats
        # order and L1's ``priv1a`` is registered before L2's ``priv2``).
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "priv1a restoration failed" in str(exc_info.value.__cause__)
        # ``shutdown()`` ran inside ``finally`` despite the failures.
        assert ctrl.get_gpu_memory_bytes() == 0
        assert ctrl.get_tensor_id_to_view_mapping() == {}

    def test_teardown_round_trips_tied_parameters(self) -> None:
        """Tied parameters (e.g. ``lm_head.weight is embedding.weight``) share a
        single ``nn.Parameter`` object referenced from multiple attribute paths.
        ``tensors_map`` keys by ``id``, so there is one controller entry; a
        single ``.data`` patch and a single restoration must leave *both*
        attribute paths consistent. Pin this so future processor changes that
        copy parameters instead of mutating them surface immediately.
        """
        device = _device()
        torch.manual_seed(2)

        tied = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32), requires_grad=False)
        other = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32), requires_grad=False)

        baseline = tied.detach().clone()
        baseline_ptr = tied.data_ptr()

        # Two attribute paths -> one Parameter object (tied embedding/lm_head).
        embedding = torch.nn.Module()
        embedding.weight = tied
        lm_head = torch.nn.Module()
        lm_head.weight = tied
        leaf_other = torch.nn.Module()
        leaf_other.weight = other

        root = torch.nn.Module()
        root.embedding = embedding
        root.lm_head = lm_head
        root.leaf_other = leaf_other

        # Sanity: the two attribute paths really are the same object.
        assert root.embedding.weight is root.lm_head.weight

        tensors_map = {id(tied): tied, id(other): other}
        stats = [
            IterativeLayerStatistics(label="L1", tensor_ids=[id(tied), id(other)]),
            IterativeLayerStatistics(label="L2", tensor_ids=[id(tied)]),
        ]

        ctrl = ProfileBlockController(stats, tensors_map, device)
        view_map = ctrl.get_tensor_id_to_view_mapping()
        # Tied tid resolves to a single view shared by both attribute paths.
        assert id(tied) in view_map

        # Mimic patching: one ``.data`` assignment, observable from both paths.
        tied.data = view_map[id(tied)]
        other.data = view_map[id(other)]
        assert root.embedding.weight.data_ptr() == root.lm_head.weight.data_ptr()
        assert root.embedding.weight.data_ptr() != baseline_ptr

        ctrl.teardown(root, tensors_map)

        # Both attribute paths see the restored original storage.
        assert root.embedding.weight is root.lm_head.weight
        assert root.embedding.weight.data_ptr() == baseline_ptr
        assert root.lm_head.weight.data_ptr() == baseline_ptr
        assert torch.equal(root.embedding.weight.data.cpu(), baseline)

    def test_teardown_partial_failure_restores_successful_tids(self) -> None:
        """Mixed-outcome teardown: surviving tids must be restored to their
        original storage. A regression that left them pointing at the
        controller's view (or at freed storage) would silently corrupt the
        next inference pass.
        """
        stats, tensors_map, named = _build_two_label_setup()

        shared_orig_ptr = named["shared"].data_ptr()
        priv1b_orig_ptr = named["priv1b"].data_ptr()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        view_map = ctrl.get_tensor_id_to_view_mapping()

        # Mimic the production flow: model patching repoints ``.data`` at the
        # controller's view for the real tensors. The proxies (failing tids)
        # don't get patched -- their setters raise on ``.data`` assignment.
        named["shared"].data = view_map[id(named["shared"])]
        named["priv1b"].data = view_map[id(named["priv1b"])]
        assert named["shared"].data_ptr() != shared_orig_ptr
        assert named["priv1b"].data_ptr() != priv1b_orig_ptr

        priv1a_id = id(named["priv1a"])
        priv2_id = id(named["priv2"])

        class _RaisingDataProxy:
            def __setattr__(self, name: str, value: object) -> None:
                if name == "data":
                    raise ValueError(f"restore failed for {id(self)}")
                super().__setattr__(name, value)

        patched_map = dict(tensors_map)
        patched_map[priv1a_id] = _RaisingDataProxy()
        patched_map[priv2_id] = _RaisingDataProxy()

        with pytest.raises(RuntimeError) as exc_info:
            ctrl.teardown(model=None, tensors_map=patched_map)

        # Error must list exactly the failed tids and nothing else.
        msg = str(exc_info.value)
        assert "failed to restore 2 tensor(s)" in msg
        assert str(priv1a_id) in msg
        assert str(priv2_id) in msg
        assert str(id(named["shared"])) not in msg
        assert str(id(named["priv1b"])) not in msg

        # Surviving tids point back at their original CPU storage, not at the
        # rotating-block view.
        assert named["shared"].data_ptr() == shared_orig_ptr
        assert named["priv1b"].data_ptr() == priv1b_orig_ptr

        # And the GPU block is gone regardless.
        assert ctrl.get_gpu_memory_bytes() == 0

    def test_teardown_blanks_binding_when_restore_fails(self) -> None:
        """When restoring ``.data`` raises, teardown must blank the binding so
        no profile-block view survives (the failed param ends up empty rather
        than aliasing the released GPU block). A setter that rejects the
        non-empty original but accepts an empty tensor exercises the fail-safe.
        """
        stats, tensors_map, named = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        priv1a_id = id(named["priv1a"])

        class _RejectsNonEmptyData:
            """Accepts only empty ``.data`` -- mimics a wrapper that validates
            the restored value but tolerates blanking."""

            def __setattr__(self, name: str, value: object) -> None:
                if name == "data" and isinstance(value, torch.Tensor) and value.numel() > 0:
                    raise ValueError("rejects non-empty restore")
                super().__setattr__(name, value)

        proxy = _RejectsNonEmptyData()
        patched_map = dict(tensors_map)
        patched_map[priv1a_id] = proxy

        with pytest.raises(RuntimeError):
            ctrl.teardown(model=None, tensors_map=patched_map)

        # Fail-safe ran: the binding is an empty tensor, not the block view.
        assert isinstance(proxy.data, torch.Tensor)
        assert proxy.data.numel() == 0
        assert ctrl.get_gpu_memory_bytes() == 0

    def test_teardown_idempotent(self) -> None:
        stats, tensors_map, _ = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.teardown(model=None, tensors_map=tensors_map)
        # A second call must not blow up.
        ctrl.teardown(model=None, tensors_map=tensors_map)

    def test_tensors_map_invariant_preserved_through_lifecycle(self) -> None:
        """Every tid the controller manages at init must still be present in
        ``tensors_map`` when ``teardown`` runs. ``teardown`` tolerates a
        missing entry silently (it skips the ``.data`` restore), so this
        invariant is what guarantees every managed parameter is actually
        restored. Pin it down so any future change that mutates
        ``tensors_map`` between init and teardown fails loudly here.
        """
        stats, tensors_map, _ = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        managed = set(ctrl.get_tensor_id_to_view_mapping().keys())
        assert managed, "test setup must produce at least one managed tid"
        assert managed.issubset(tensors_map.keys())

        for label in ("L1", "L2"):
            ctrl.enter(label)
            ctrl.exit(label)
        assert managed.issubset(tensors_map.keys())

        ctrl.teardown(model=None, tensors_map=tensors_map)
        assert managed.issubset(tensors_map.keys())

    def test_shutdown_alias(self) -> None:
        stats, tensors_map, _ = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.shutdown()

        assert ctrl.get_tensor_id_to_view_mapping() == {}
        assert ctrl.get_gpu_memory_bytes() == 0

    def test_release_memory_is_noop_and_preserves_block(self) -> None:
        """``TensorManager.release_memory()`` calls into whichever loader is
        currently bound, and during the profile phase that's the controller.
        ``release_memory`` must therefore exist (mirrors
        ``TensorLayerLoader.release_memory``) and must not free the block —
        block release belongs to ``shutdown`` / ``teardown``.
        """
        stats, tensors_map, _ = _build_two_label_setup()

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        bytes_before = ctrl.get_gpu_memory_bytes()
        view_map_before = ctrl.get_tensor_id_to_view_mapping()
        assert bytes_before > 0

        ctrl.release_memory()

        assert ctrl.get_gpu_memory_bytes() == bytes_before
        assert ctrl.get_tensor_id_to_view_mapping() == view_map_before


def test_cpu_staging_block_is_pinned_on_cuda() -> None:
    """With ``pinned_memory=True`` on CUDA, the host staging block is page-locked
    so the per-label H2D runs at full bandwidth."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stats, tensors_map, _ = _build_two_label_setup()
    ctrl = ProfileBlockController(stats, tensors_map, torch.device("cuda:0"), pinned_memory=True)

    assert ctrl.block_size > 0
    assert ctrl._cpu_block.is_pinned()


def test_cpu_staging_block_pageable_when_pinning_disabled() -> None:
    """Default (``pinned_memory=False``) leaves the staging block pageable —
    the controller honors the opt-out rather than pinning unconditionally."""
    stats, tensors_map, _ = _build_two_label_setup()
    ctrl = ProfileBlockController(stats, tensors_map, _device())

    assert ctrl.block_size > 0
    assert ctrl._cpu_block.is_pinned() is False


@pytest.mark.parametrize("device_str", ["cpu", "cuda:0"])
def test_get_gpu_memory_bytes_matches_block_size(device_str: str) -> None:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    stats, tensors_map, _ = _build_two_label_setup()
    ctrl = ProfileBlockController(stats, tensors_map, torch.device(device_str))

    assert ctrl.get_gpu_memory_bytes() == ctrl.block_size


class TestTensorByteView:
    """Cover all four branches of ``_tensor_byte_view``.

    The function returns a flat ``uint8`` view used to pack tensors into the
    rotating block. Slot sizes (computed by ``_build_slot_meta`` as
    ``numel * element_size``) must match what this helper produces, or the
    rotating region gets corrupted. The branches differ in *how* the bytes
    are obtained but must all agree on the byte count.
    """

    def test_empty_tensor_returns_empty_uint8(self) -> None:
        """``numel() == 0`` branch: empty input → empty uint8 output."""
        t = torch.zeros(0, dtype=torch.float32)
        out = _tensor_byte_view(t)

        assert out.dtype == torch.uint8
        assert out.numel() == 0

    def test_contiguous_tensor_round_trips_through_uint8(self) -> None:
        """Contiguous branch: standard C-contiguous input → flat uint8 view."""
        t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        assert t.is_contiguous()

        out = _tensor_byte_view(t)

        assert out.dtype == torch.uint8
        assert out.numel() == t.numel() * t.element_size()
        # Round-trip: reinterpreting the uint8 bytes back as float32 reproduces
        # the flattened original — proves the byte order is preserved.
        assert torch.equal(out.view(torch.float32), t.flatten())

    def test_dense_non_contiguous_uses_storage_offset(self) -> None:
        """Dense-but-not-contiguous branch: permutation-contiguous input
        (e.g. a transpose) reaches the ``.set_()`` path that maps a uint8
        view directly onto the existing storage starting at ``storage_offset``.

        The bytes returned reflect the underlying *storage order*, not the
        transposed iteration order. ``_build_slot_meta`` records the
        contiguous stride for such tensors, so writing storage-order bytes
        into the slot and reading them back through the recorded stride is
        what reconstructs the original tensor's data view.
        """
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        t = base.t()
        assert not t.is_contiguous()
        assert is_dense_layout(t)

        out = _tensor_byte_view(t)

        assert out.dtype == torch.uint8
        assert out.numel() == t.numel() * t.element_size()
        # The branch maps onto ``t.untyped_storage()`` starting at
        # ``storage_offset() * element_size()`` for ``nbytes``. Build the
        # expected storage-byte slice the same way and compare.
        expected = torch.empty(0, dtype=torch.uint8)
        expected.set_(
            t.untyped_storage(),
            t.storage_offset() * t.element_size(),
            (t.numel() * t.element_size(),),
        )
        assert torch.equal(out, expected)

    def test_non_dense_falls_back_to_contiguous_copy(self) -> None:
        """Non-dense (gapped) branch: a strided slice with element gaps falls
        through to ``tensor.contiguous().view(uint8).flatten()`` so the
        returned bytes are packed (no inherited gaps).
        """
        base = torch.arange(20, dtype=torch.float32)
        t = base[::2]
        assert not is_dense_layout(t)

        out = _tensor_byte_view(t)

        assert out.dtype == torch.uint8
        assert out.numel() == t.numel() * t.element_size()
        # Round-trip with the contiguous copy (what the branch actually does).
        assert torch.equal(out.view(torch.float32), t.contiguous().flatten())

    @pytest.mark.parametrize(
        "tensor_factory",
        [
            pytest.param(lambda: torch.zeros(0, dtype=torch.float32), id="empty"),
            pytest.param(lambda: torch.randn(3, 5, dtype=torch.float32), id="contiguous"),
            pytest.param(lambda: torch.randn(3, 5, dtype=torch.float32).t(), id="dense_transposed"),
            pytest.param(lambda: torch.randn(20, dtype=torch.float32)[::2], id="non_dense_sliced"),
        ],
    )
    def test_byte_count_matches_slot_size_invariant(self, tensor_factory) -> None:
        """All four branches must return ``numel * element_size`` bytes.

        ``ProfileBlockController._build_slot_meta`` sizes each slot as
        ``numel * element_size`` regardless of layout. If any
        ``_tensor_byte_view`` branch returned a different byte count, the
        ``copy_`` into the slot would either truncate or overflow into the
        next slot — silent rotating-region corruption.
        """
        t = tensor_factory()
        out = _tensor_byte_view(t)

        assert out.numel() == t.numel() * t.element_size()


class TestSlotMetaAlignment:
    """Verify ``_SlotMeta`` / ``_build_slot_meta`` keep slots dtype-aligned.

    ``_slot_view`` reconstructs the typed view by integer-dividing
    ``meta.start`` by the dtype's element size. A misaligned ``start`` is
    silently truncated to the prior aligned offset, so the view ends up
    pointing ``start % itemsize`` bytes *before* where the bytes were copied
    — producing garbage on read. These tests pin both the layout-side
    fix (``_build_slot_meta`` rounds ``start`` up) and the construction-side
    guard (``_SlotMeta.__post_init__`` rejects misaligned input).
    """

    def test_post_init_rejects_misaligned_start(self) -> None:
        """Constructing ``_SlotMeta`` with ``start`` not divisible by the
        dtype's itemsize must fail loudly. This guards against future layout
        code that forgets to round up.
        """
        with pytest.raises(ValueError, match="not aligned"):
            _SlotMeta(
                start=1,
                end=5,
                shape=torch.Size([2]),
                dtype=torch.float16,
                stride=(1,),
            )

    def test_post_init_rejects_span_not_multiple_of_itemsize(self) -> None:
        """``end - start`` must cover an integer number of elements; otherwise
        the slot's byte span doesn't match the typed view's footprint. Pinned
        as a defensive guard against future direct constructions that bypass
        ``_build_slot_meta``.
        """
        with pytest.raises(ValueError, match="not a non-negative multiple"):
            _SlotMeta(
                start=0,
                end=5,
                shape=torch.Size([2]),
                dtype=torch.float16,
                stride=(1,),
            )

    def test_build_slot_meta_pads_start_to_dtype_alignment(self) -> None:
        """``alignment=1`` collapses the formula to historical dtype-only
        rounding: an odd ``start`` for an fp16 tensor rounds up to the next
        2-byte boundary. Pins the original semantics so future regressions
        of the rounding formula itself are caught even when the module
        default is overridden.
        """
        fp16 = torch.zeros(4, dtype=torch.float16)
        meta = ProfileBlockController._build_slot_meta(fp16, start=1, alignment=1)

        assert meta.start == 2
        assert meta.end == meta.start + fp16.numel() * fp16.element_size()
        assert meta.start % fp16.element_size() == 0

    def test_mixed_dtype_layout_round_trips_through_views(self) -> None:
        """End-to-end: int8 followed by fp16 in the same label must produce
        an fp16 view that round-trips its original values. Without the
        alignment fix, the fp16 slot would sit on an odd byte and read 1
        byte off, returning garbage.

        Construction also exercises shared-prefix alignment: a single int8
        scalar referenced by both labels lands at offset 0, advances the
        cursor by 1 byte, and must not break the fp16 slot that follows.
        """
        torch.manual_seed(0)
        # Shared int8 scalar to push the rotating-region cursor onto an odd
        # byte. Two labels reference it so it lands in the shared prefix.
        shared_i8 = torch.full((1,), 7, dtype=torch.int8)
        # Per-label fp16 weights — the slot whose alignment was broken.
        priv_fp16_a = torch.randn(8, dtype=torch.float16)
        priv_fp16_b = torch.randn(4, dtype=torch.float16)

        named = {
            "shared_i8": shared_i8,
            "priv_fp16_a": priv_fp16_a,
            "priv_fp16_b": priv_fp16_b,
        }
        tensors_map = {id(t): t for t in named.values()}
        stats = [
            IterativeLayerStatistics(
                label="L1",
                tensor_ids=[id(shared_i8), id(priv_fp16_a)],
            ),
            IterativeLayerStatistics(
                label="L2",
                tensor_ids=[id(shared_i8), id(priv_fp16_b)],
            ),
        ]

        ctrl = ProfileBlockController(stats, tensors_map, _device())

        ctrl.enter("L1")
        assert torch.equal(ctrl.get(id(priv_fp16_a)).cpu(), priv_fp16_a)
        ctrl.enter("L2")
        assert torch.equal(ctrl.get(id(priv_fp16_b)).cpu(), priv_fp16_b)
        assert torch.equal(ctrl.get(id(shared_i8)).cpu(), shared_i8)


class TestNonContiguousRoundTrip:
    """Verify non-contiguous source tensors round-trip through the controller.

    ``_tensor_byte_view`` has dedicated branches for dense-but-not-contiguous
    (transpose) and non-dense (strided slice) inputs; these tests pin the
    end-to-end pipeline (``__init__`` → ``enter`` → ``get``) for both.
    """

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(
                lambda: torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
                id="dense_transposed",
            ),
            pytest.param(
                lambda: torch.arange(20, dtype=torch.float32)[::2],
                id="non_dense_sliced",
            ),
        ],
    )
    def test_non_contiguous_source_round_trips(self, factory) -> None:
        src = factory()
        tensors_map = {id(src): src}
        stats = [IterativeLayerStatistics(label="L1", tensor_ids=[id(src)])]

        ctrl = ProfileBlockController(stats, tensors_map, _device())
        ctrl.enter("L1")

        view = ctrl.get(id(src))
        assert view is not None
        assert view.shape == src.shape
        assert view.dtype == src.dtype
        # Logical values must match. Stride may differ for non-dense sources;
        # ``_build_slot_meta`` records the contiguous stride so the view
        # iterates over packed bytes.
        assert torch.equal(view.cpu(), src.contiguous())
