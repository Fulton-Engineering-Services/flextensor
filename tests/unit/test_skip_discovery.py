# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``skip_discovery`` config option.

Covers:

* ``OffloadConfig.skip_discovery`` field default and round-trip.
* ``TensorManager.set_skip_discovery`` flag wiring.
* ``TensorManager._build_layer_stats_from_forward_patching`` against
  unpatched, partially patched, and fully patched models.
* ``prepare_profile_mode`` / ``prepare_profile_direct_mode`` idempotence
  when ``layer_stats`` is pre-populated by the skip path.
* ``OffloadManager._transition_to_warmup`` short-circuit to profiling
  (and the guard that requires patched modules to be present).
* ``IterativeLayerStatisticsCollector._labels_with_durations`` filtering.
"""

import logging
import os
import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.collectors import IterativeLayerStatistics, IterativeLayerStatisticsCollector
from flextensor.config import OffloadConfig, load_config_from_env
from flextensor.offload_manager import OffloadManager, OffloadPhase
from flextensor.tensor_manager import TensorManager


def _mark_module_patched(module: nn.Module, name: str) -> None:
    """Add the markers that ``is_offload_patched_module`` and ``get_offload_name`` look for.

    Mirrors what ``OffloadManager._patch_module_forward`` writes, without going
    through the full forward-patching pipeline (which the high-level API
    triggers via ``OffloadManager.offload``).
    """
    module._ft_original_forward_func = type(module).forward  # noqa: SLF001
    module._ft_offload_name = name  # noqa: SLF001


class _Layer(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_tensor_manager(model: nn.Module) -> TensorManager:
    """Construct a TensorManager wired up enough for the skip-path helpers.

    Avoids the full ``initialize_warmup`` pipeline (which touches CUDA)
    while still populating ``tensors_map`` so ``get_offload_module_tensor_ids``
    can resolve parameter IDs.
    """
    tm = TensorManager(device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False)
    tm.model = model
    tm.tensors_map = {id(p): p for p in model.parameters()}
    return tm


class TestSkipDiscoveryConfig:
    def test_default_is_false(self) -> None:
        assert OffloadConfig().skip_discovery is False

    def test_round_trip_true(self) -> None:
        config = OffloadConfig(skip_discovery=True)
        assert config.skip_discovery is True

    def test_model_copy_preserves_value(self) -> None:
        config = OffloadConfig().model_copy(update={"skip_discovery": True})
        assert config.skip_discovery is True

    def test_accepts_zero_profiling_iters_under_both_defaults(self) -> None:
        """``profiling_iters=0`` is functionally equivalent to ``profiling_iters=1``.

        Because ``OffloadManager.update_state()`` runs as a post-forward
        hook, the PROFILING→INFERENCE transition can only fire *after* the
        first profile forward completes. Both ``profiling_iters=0`` (guard
        ``1 >= 0``) and ``profiling_iters=1`` (guard ``1 >= 1``) transition
        after exactly one profile forward — so rejecting only ``=0`` (as an
        earlier ``_validate_skip_discovery_requires_profiling`` did) was
        inconsistent with the state-machine timing. Neither combination
        raises now; ``report_profiling_quality`` still WARNs on low sample
        counts.
        """
        for skip in (True, False):
            config = OffloadConfig(skip_discovery=skip, profiling_iters=0)
            assert config.profiling_iters == 0

    def test_accepts_skip_discovery_with_positive_profiling_iters(self) -> None:
        config = OffloadConfig(skip_discovery=True, profiling_iters=1)
        assert config.profiling_iters == 1

    def test_model_copy_rejects_illegal_combination(self) -> None:
        """``model_copy(update=...)`` must re-validate the resulting config.

        Pydantic v2 bypasses ``@model_validator`` on ``model_copy`` by
        design, so a bare copy could smuggle in an illegal combination.
        The ``OffloadConfig`` override re-parses through the validator
        chain — vLLM's worker relies on ``model_copy`` in production
        (see ``FlexTensorOffloadWorker.load_model``), so an invalid
        override there must fail at construction, not silently activate.

        Exercises ``_validate_profile_mode``: ``profile_mode='torch_function'``
        is incompatible with a block ``transfer_mode``.
        """
        base = OffloadConfig(profile_mode="getter", transfer_mode="allocation_block_transfer")
        with pytest.raises(ValueError, match="profile_mode='torch_function' is incompatible"):
            base.model_copy(update={"profile_mode": "torch_function"})

    def test_model_copy_accepts_legal_combination_via_update(self) -> None:
        """The re-validation must not reject legal updates that would
        satisfy the invariant only after the update is applied."""
        base = OffloadConfig(profile_mode="getter", transfer_mode="allocation_block_transfer")
        copied = base.model_copy(update={"profile_mode": "torch_function", "transfer_mode": "strategy"})
        assert copied.profile_mode == "torch_function"
        assert copied.transfer_mode == "strategy"


class TestTensorManagerSetSkipDiscovery:
    def test_default_flag_is_false(self) -> None:
        tm = TensorManager(
            device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False
        )
        assert tm.skip_discovery_requested is False

    def test_set_true(self) -> None:
        tm = TensorManager(
            device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False
        )
        tm.set_skip_discovery(True)
        assert tm.skip_discovery_requested is True

    def test_set_false(self) -> None:
        tm = TensorManager(
            device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False
        )
        tm.set_skip_discovery(True)
        tm.set_skip_discovery(False)
        assert tm.skip_discovery_requested is False

    def test_layer_stats_initialized_to_none(self) -> None:
        tm = TensorManager(
            device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False
        )
        assert tm.layer_stats is None


class TestBuildLayerStatsFromForwardPatching:
    def test_returns_empty_when_model_is_none(self) -> None:
        tm = _build_tensor_manager(_Model())
        tm.model = None
        assert tm._build_layer_stats_from_forward_patching() == []

    def test_returns_empty_when_no_modules_patched(self) -> None:
        tm = _build_tensor_manager(_Model())
        assert tm._build_layer_stats_from_forward_patching() == []

    def test_collects_patched_module_tensors(self) -> None:
        model = _Model()
        for i, layer in enumerate(model.layers):
            _mark_module_patched(layer, f"layers.{i}")
        tm = _build_tensor_manager(model)

        stats = tm._build_layer_stats_from_forward_patching()

        assert {s.label for s in stats} == {f"layers.{i}" for i in range(3)}
        for stat in stats:
            assert stat.duration == pytest.approx(0.0)
            assert isinstance(stat.tensor_ids, set)
            assert stat.tensor_ids
        # Each layer contributes exactly its own parameters (weight + bias).
        all_tensor_ids = {tid for stat in stats for tid in stat.tensor_ids}
        assert all_tensor_ids == set(tm.tensors_map.keys())

    def test_drops_patched_modules_with_no_matching_tensors(self) -> None:
        # tensors_map contains no parameters from the model, so even a patched
        # module yields an empty tensor_ids set and is filtered out.
        model = _Model()
        _mark_module_patched(model.layers[0], "layers.0")
        tm = _build_tensor_manager(model)
        tm.tensors_map = {}

        assert tm._build_layer_stats_from_forward_patching() == []

    def test_warns_when_patched_modules_have_no_tensors(self, caplog: pytest.LogCaptureFixture) -> None:
        # Silent drops mask "include_patterns matched a patched module whose
        # params are all excluded" — the user loses offload coverage for that
        # label with no breadcrumb. The WARNING surfaces the dropped labels
        # so they can correct their patterns.
        model = _Model()
        _mark_module_patched(model.layers[0], "layers.0")
        _mark_module_patched(model.layers[1], "layers.1")
        tm = _build_tensor_manager(model)
        tm.tensors_map = {}

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_manager"):
            tm._build_layer_stats_from_forward_patching()

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("had no offloadable tensors" in r.message for r in warning_records), (
            "must log a WARNING listing dropped patched-module labels"
        )
        assert any("layers.0" in r.message and "layers.1" in r.message for r in warning_records)

    def test_no_warning_when_all_patched_modules_contribute(self, caplog: pytest.LogCaptureFixture) -> None:
        model = _Model()
        for i, layer in enumerate(model.layers):
            _mark_module_patched(layer, f"layers.{i}")
        tm = _build_tensor_manager(model)

        with caplog.at_level(logging.WARNING, logger="flextensor.tensor_manager"):
            tm._build_layer_stats_from_forward_patching()

        assert not any("had no offloadable tensors" in r.message for r in caplog.records), (
            "no patched modules were dropped; the WARNING must stay quiet"
        )


class TestInitializeWarmupSkipDiscoverySeeding:
    """``initialize_warmup`` must wire the static seed into the collector.

    The seeding loop at ``tensor_manager.py:initialize_warmup`` is what
    bridges :meth:`_build_layer_stats_from_forward_patching` to the
    collector that downstream consumers (``UntimedTrapsReport``, layer-stat
    aggregation) read. A refactor that drops the loop or renames
    ``add_tensors`` would otherwise pass every other test in this file
    while silently breaking the skip-discovery contract end-to-end.
    """

    def test_initialize_warmup_seeds_collector_from_patched_modules(self) -> None:
        model = _Model()
        for i, layer in enumerate(model.layers):
            _mark_module_patched(layer, f"layers.{i}")

        tm = _build_tensor_manager(model)
        tm.set_skip_discovery(True)

        # Mock the heavyweight tensor-mapping/move steps — they're not what
        # this test pins. The collector wiring is the sole subject.
        with (
            patch("flextensor.tensor_manager.preprocess_model"),
            patch.object(tm, "_move_non_offloaded_tensors_to_gpu"),
        ):
            tm.initialize_warmup()

        expected_labels = {f"layers.{i}" for i in range(3)}
        assert set(tm.layer_statistics_collector.tensor_measurements.keys()) == expected_labels

        # Each label was seeded exactly once with the matching layer's parameters.
        for i, layer in enumerate(model.layers):
            label = f"layers.{i}"
            measurements = tm.layer_statistics_collector.tensor_measurements[label]
            assert len(measurements) == 1, f"label {label!r} should be seeded once, got {len(measurements)}"
            assert measurements[0] == {id(p) for p in layer.parameters()}

    def test_initialize_warmup_does_not_seed_when_flag_disabled(self) -> None:
        model = _Model()
        for i, layer in enumerate(model.layers):
            _mark_module_patched(layer, f"layers.{i}")

        tm = _build_tensor_manager(model)
        # skip_discovery_requested left at default False

        with (
            patch("flextensor.tensor_manager.preprocess_model"),
            patch.object(tm, "_move_non_offloaded_tensors_to_gpu"),
        ):
            tm.initialize_warmup()

        # Collector is allocated by prepare_warmup_mode but stays empty —
        # the seeding loop only runs when skip_discovery_requested is True.
        assert tm.layer_statistics_collector.tensor_measurements == {}
        assert tm.layer_stats is None


class TestPrepareProfileModeIdempotence:
    """``prepare_profile_*_mode`` must not overwrite a pre-populated ``layer_stats``.

    The skip path seeds ``layer_stats`` during ``initialize_warmup`` and marks
    it with ``_layer_stats_seeded_statically``; the profile-prep methods should
    honor that seed and only fall back to the collector when it is absent. The
    marker is explicit rather than inferred from ``layer_stats is None`` so a
    second offload cycle cannot mistake the previous cycle's stats for a seed.
    """

    def _setup_tm(self) -> TensorManager:
        tm = _build_tensor_manager(_Model())
        # Drive ``prepare_profile_direct_mode`` directly (without going through
        # ``prepare_profile_direct_mode_model``); the view path needs a
        # pre-built ``ProfileBlockController`` that this fixture doesn't stage.
        tm.profile_mode = "getter"
        tm.layer_statistics_collector = IterativeLayerStatisticsCollector()
        tm.enable_untraced_tensor_discovery = False  # avoid extra discovery
        tm.module_tracker = None
        return tm

    def _seed(self, tm: TensorManager) -> list[IterativeLayerStatistics]:
        """Stage a static seed the way ``initialize_warmup`` does.

        Uses real ids from ``tensors_map``: the seed is filtered against known
        tensors, exactly as ``_build_layer_stats_from_forward_patching``'s
        output would be.
        """
        seeded = [IterativeLayerStatistics(label="layers.0", tensor_ids=set(tm.tensors_map), duration=0.0)]
        tm.layer_stats = seeded
        tm._layer_stats_seeded_statically = True
        return seeded

    def test_prepare_profile_mode_does_not_overwrite_prepopulated(self) -> None:
        tm = self._setup_tm()
        seeded = self._seed(tm)

        with patch("flextensor.tensor_manager.TensorLayerLoader"):
            tm.prepare_profile_mode()

        assert tm.layer_stats == seeded

    def test_prepare_profile_direct_mode_does_not_overwrite_prepopulated(self) -> None:
        tm = self._setup_tm()
        seeded = self._seed(tm)

        with patch("flextensor.tensor_manager.TensorLayerLoader"):
            tm.prepare_profile_direct_mode()

        assert tm.layer_stats == seeded

    def test_unmarked_stats_are_rebuilt_not_reused(self) -> None:
        """Without the seed marker, stale stats must not survive.

        This is the second-offload-cycle case: ``layer_stats`` is non-``None``
        but belongs to the previous model.
        """
        tm = self._setup_tm()
        tm.layer_stats = [IterativeLayerStatistics(label="stale", tensor_ids=set(tm.tensors_map), duration=1.0)]

        with patch("flextensor.tensor_manager.TensorLayerLoader"):
            tm.prepare_profile_mode()

        assert "stale" not in {stat.label for stat in (tm.layer_stats or [])}

    def test_prepare_profile_mode_falls_back_to_collector(self) -> None:
        tm = self._setup_tm()
        # Collector returns an empty list; we just verify it is consulted.
        tm.layer_stats = None
        with (
            patch.object(tm.layer_statistics_collector, "get_layer_stats", return_value=[]) as mock_get,
            patch("flextensor.tensor_manager.TensorLayerLoader"),
        ):
            tm.prepare_profile_mode()

        mock_get.assert_called_once()
        assert tm.layer_stats == []


class TestOffloadManagerSkipDiscoveryShortCircuit:
    """``_transition_to_warmup`` should jump to profiling only when both
    ``skip_discovery`` is set and at least one module has been patched.
    """

    @staticmethod
    def _model_with_patched_child() -> nn.Module:
        """Build an nn.Module whose `model.modules()` walk visits a patched leaf."""
        model = nn.Sequential(nn.Identity())
        _mark_module_patched(model[0], "leaf")
        return model

    def _make_manager(
        self, *, skip_discovery: bool, model: nn.Module | None = None
    ) -> tuple[OffloadManager, MagicMock]:
        # The model passed in is used for both ``om._model`` AND as the value
        # returned by ``mock_tm.initialize_warmup``, because ``_swap_to_new_model``
        # reassigns ``om._model`` from that return value before the
        # short-circuit gate fires.
        if model is None:
            model = nn.Identity()
        om = OffloadManager(f"test_skip_{skip_discovery}")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=skip_discovery))
        mock_tm = MagicMock()
        mock_tm.initialize_warmup.return_value = model
        om._tensor_manager = mock_tm
        om._model = model
        om._patched_modules = []
        return om, mock_tm

    def test_short_circuits_when_flag_set_and_modules_patched(self) -> None:
        om, _mock_tm = self._make_manager(skip_discovery=True, model=self._model_with_patched_child())
        with patch.object(om, "_transition_to_profile") as mock_profile:
            om._transition_to_warmup()

        mock_profile.assert_called_once()
        assert om.skip_discovery_honored is True

    def test_no_short_circuit_when_flag_set_but_no_modules_patched(self, caplog: pytest.LogCaptureFixture) -> None:
        om, _mock_tm = self._make_manager(skip_discovery=True)
        # Plain ``nn.Identity`` — has_offload_modules returns False.
        with (
            patch.object(om, "_transition_to_profile") as mock_profile,
            caplog.at_level("WARNING", logger="flextensor.offload_manager"),
        ):
            om._transition_to_warmup()

        mock_profile.assert_not_called()
        assert om._current_phase == OffloadPhase.DISCOVERY
        # Silent fallback would mask a perf regression — pin the warning.
        assert any("skip_discovery=True but no patched modules" in m for m in caplog.messages)
        # And the programmatic signal must flip so callers don't need to grep logs.
        assert om.skip_discovery_honored is False

    def test_no_warning_when_flag_set_and_modules_patched(self, caplog: pytest.LogCaptureFixture) -> None:
        om, _mock_tm = self._make_manager(skip_discovery=True, model=self._model_with_patched_child())
        with (
            patch.object(om, "_transition_to_profile"),
            caplog.at_level("WARNING", logger="flextensor.offload_manager"),
        ):
            om._transition_to_warmup()

        assert not any("skip_discovery" in m for m in caplog.messages)

    def test_no_short_circuit_when_flag_not_set(self) -> None:
        om, _mock_tm = self._make_manager(skip_discovery=False, model=self._model_with_patched_child())
        with patch.object(om, "_transition_to_profile") as mock_profile:
            om._transition_to_warmup()

        mock_profile.assert_not_called()
        assert om._current_phase == OffloadPhase.DISCOVERY
        # ``skip_discovery=False`` means the request was honored by definition.
        assert om.skip_discovery_honored is True

    def test_skip_discovery_honored_starts_undetermined(self) -> None:
        """Before any warmup the verdict is unknown, not optimistic.

        Reporting ``True`` here conflated "the skip fired" with "nothing has
        run yet", which is what made the property easy to misread.
        """
        om = OffloadManager("test_skip_honored_init")
        assert om.skip_discovery_honored is None

    def test_release_resets_the_verdict_to_undetermined(self) -> None:
        """After ``release()`` the verdict belongs to a manager that is gone."""
        om = OffloadManager("test_skip_honored_release")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=True))
        om._skip_discovery_honored = False

        om.release()

        assert om.skip_discovery_honored is None


class TestSetSkipDiscoveryPropagation:
    """``OffloadManager`` must forward ``config.skip_discovery`` to the
    ``TensorManager`` after constructing it."""

    @patch("flextensor.offload_manager.AdaptiveStrategy")
    @patch("flextensor.tensor_manager.TensorManager")
    def test_set_skip_discovery_called_with_config_value(
        self,
        mock_tm_cls: MagicMock,
        _mock_strategy_cls: MagicMock,
    ) -> None:
        mock_tm = MagicMock()
        mock_tm_cls.return_value = mock_tm

        om = OffloadManager("test_propagation_true")
        om.set_config(OffloadConfig(skip_discovery=True))
        om._initialize_tensor_manager()

        mock_tm.set_skip_discovery.assert_called_once_with(True)

    @patch("flextensor.offload_manager.AdaptiveStrategy")
    @patch("flextensor.tensor_manager.TensorManager")
    def test_set_skip_discovery_called_with_explicit_false(
        self,
        mock_tm_cls: MagicMock,
        _mock_strategy_cls: MagicMock,
    ) -> None:
        mock_tm = MagicMock()
        mock_tm_cls.return_value = mock_tm

        om = OffloadManager("test_propagation_false")
        om.set_config(OffloadConfig(skip_discovery=False))
        om._initialize_tensor_manager()

        mock_tm.set_skip_discovery.assert_called_once_with(False)

    @patch("flextensor.offload_manager.AdaptiveStrategy")
    @patch("flextensor.tensor_manager.TensorManager")
    def test_set_skip_discovery_called_before_initialize_warmup(
        self,
        mock_tm_cls: MagicMock,
        _mock_strategy_cls: MagicMock,
    ) -> None:
        """Regression: ``set_skip_discovery`` must reach the manager BEFORE
        ``initialize_warmup`` reads ``skip_discovery_requested``.

        The existing propagation tests only check that ``set_skip_discovery``
        was called at all — swapping the order at offload_manager.py:704 to
        run after ``_transition_to_warmup`` would silently break the entire
        skip-discovery feature with every other test still green, because
        ``initialize_warmup`` reads the flag at line 586 of tensor_manager.py.
        """
        mock_tm = MagicMock()
        mock_tm_cls.return_value = mock_tm
        mock_tm.initialize_warmup.return_value = nn.Identity()

        om = OffloadManager("test_propagation_order")
        om.set_config(OffloadConfig(skip_discovery=True))
        om._model = nn.Identity()
        # Drive the public flow: init + transition. _transition_to_warmup is
        # the consumer of skip_discovery_requested via initialize_warmup.
        om._initialize_tensor_manager()
        om._transition_to_warmup()

        # mock_calls captures every attribute/method access on the manager
        # mock in chronological order; we assert set_skip_discovery appears
        # before initialize_warmup.
        method_names = [call[0] for call in mock_tm.mock_calls if call[0]]
        try:
            set_idx = method_names.index("set_skip_discovery")
        except ValueError:
            raise AssertionError(f"set_skip_discovery never called; calls: {method_names}") from None
        try:
            init_idx = method_names.index("initialize_warmup")
        except ValueError:
            raise AssertionError(f"initialize_warmup never called; calls: {method_names}") from None
        assert set_idx < init_idx, (
            "set_skip_discovery must precede initialize_warmup so the flag is set "
            "before the warmup path reads it; got order "
            f"set_skip_discovery@{set_idx}, initialize_warmup@{init_idx}: {method_names}"
        )


class TestSkipDiscoveryEnvVar:
    """``FT_SKIP_DISCOVERY`` must round-trip through the env-var loader.

    ``config._load_from_env`` auto-generates ``FT_<FIELD>`` for every
    ``OffloadConfig`` field, so a refactor that drops a field from the
    auto-loop or coerces non-canonical booleans inconsistently would
    silently change the production default. Pin the round-trip plus a
    typo-rejecting case.
    """

    def setup_method(self) -> None:
        self._original_env = os.environ.copy()

    def teardown_method(self) -> None:
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_ft_skip_discovery_false_round_trips(self) -> None:
        os.environ["FT_SKIP_DISCOVERY"] = "0"
        config = load_config_from_env()
        assert config.skip_discovery is False

    def test_ft_skip_discovery_true_overrides_default(self) -> None:
        os.environ["FT_SKIP_DISCOVERY"] = "1"
        config = load_config_from_env()
        assert config.skip_discovery is True

    def test_ft_skip_discovery_rejects_unparseable_value(self) -> None:
        # ``"maybe"`` is neither truthy nor falsy; the env-var bool parser
        # must surface this as a ValueError rather than silently defaulting.
        os.environ["FT_SKIP_DISCOVERY"] = "maybe"
        with pytest.raises(ValueError, match="FT_SKIP_DISCOVERY"):
            load_config_from_env()


class TestEndToEndNoWarmupTrapFires:
    """End-to-end pin: ``skip_discovery=True`` short-circuits warmup so
    ``WarmupTrap.__enter__`` never fires.

    The short-circuit is verified elsewhere via mocked
    ``_transition_to_profile``. This test takes the complementary angle:
    drive the public flow and assert ``WarmupTrap`` is never entered. A
    future regression where the static seed populates correctly but the
    transition somehow stays in DISCOVERY (e.g. the gate is moved earlier
    than the seed) would slip past the mocked tests but fail here.
    """

    @patch("flextensor.trap_tensor_mode.WarmupTrap.__enter__")
    @patch("flextensor.offload_manager.AdaptiveStrategy")
    @patch("flextensor.tensor_manager.TensorManager")
    def test_warmup_trap_not_entered_when_skip_discovery_honored(
        self,
        mock_tm_cls: MagicMock,
        _mock_strategy_cls: MagicMock,
        mock_warmup_enter: MagicMock,
    ) -> None:
        # Mock the manager so it claims initialize_warmup returned a
        # patched model — that's the gate has_offload_modules will read.
        # The short-circuit also drives _transition_to_profile (and on through
        # _transition_to_inference depending on config), so every phase init
        # must return a real Module to satisfy beartype on _swap_to_new_model.
        model = nn.Sequential(nn.Identity())
        _mark_module_patched(model[0], "leaf")
        mock_tm = MagicMock()
        mock_tm_cls.return_value = mock_tm
        mock_tm.initialize_warmup.return_value = model
        mock_tm.initialize_profile.return_value = model
        mock_tm.initialize_inference.return_value = model

        om = OffloadManager("test_e2e_no_warmup")
        om.set_config(OffloadConfig(skip_discovery=True))
        om._model = model
        om._initialize_tensor_manager()
        om._transition_to_warmup()

        mock_warmup_enter.assert_not_called()
        assert om.skip_discovery_honored is True
        assert om._current_phase == OffloadPhase.PROFILING


class TestSetConfigRejectsOneShotFieldChange:
    """``set_config`` must reject skip_discovery flips against the live manager.

    ``skip_discovery`` is read once when the underlying ``TensorManager`` is
    first constructed (during the first ``offload()`` call). Silently
    accepting a later change would leave ``self.config.skip_discovery`` and
    ``self._tensor_manager.skip_discovery_requested`` diverged; because
    ``offload_block()``'s guard reads the live config, a ``True → False``
    flip would then permit manual blocks while the manager stays in
    skip-mode and never captured their tensor mappings. Reject instead so
    the inconsistent execution path cannot exist.
    """

    def test_raises_when_skip_discovery_changes_with_active_manager(self) -> None:
        om = OffloadManager("test_set_config_skip_change")
        om.set_config(OffloadConfig(skip_discovery=True))
        om._tensor_manager = MagicMock()  # simulate post-offload() state

        with pytest.raises(RuntimeError, match="cannot change skip_discovery from True to False"):
            om.set_config(OffloadConfig(skip_discovery=False))

    def test_leaves_config_untouched_when_change_rejected(self) -> None:
        """Reject must fire before ``self.config`` is reassigned."""
        om = OffloadManager("test_set_config_skip_reject_atomic")
        original = OffloadConfig(skip_discovery=True)
        om.set_config(original)
        om._tensor_manager = MagicMock()

        with pytest.raises(RuntimeError, match="cannot change skip_discovery"):
            om.set_config(OffloadConfig(skip_discovery=False))

        assert om.config.skip_discovery is True, (
            "set_config must not partially apply — leaving self.config in the "
            "new state while raising would recreate the divergence the reject "
            "is meant to prevent"
        )

    def test_accepts_unchanged_skip_discovery_with_active_manager(self) -> None:
        """Same-value ``set_config`` must pass (other fields may still change)."""
        om = OffloadManager("test_set_config_skip_unchanged")
        om.set_config(OffloadConfig(skip_discovery=True))
        om._tensor_manager = MagicMock()

        # No RuntimeError expected.
        om.set_config(OffloadConfig(skip_discovery=True, profiling_iters=5))
        assert om.config.profiling_iters == 5

    def test_accepts_skip_discovery_change_before_manager_is_initialized(self) -> None:
        # Without an active manager there's nothing to diverge from;
        # set_config is allowed to change skip_discovery freely.
        om = OffloadManager("test_set_config_skip_pre_init")
        om.set_config(OffloadConfig(skip_discovery=True))

        # No RuntimeError — the manager hasn't captured a value yet.
        om.set_config(OffloadConfig(skip_discovery=False))
        assert om.config.skip_discovery is False


class TestOffloadBlockGuard:
    """``offload_block()`` is the manual-mode entry point.

    With ``skip_discovery=True``, tensors inside manual blocks
    cannot be discovered statically, so the call must raise. Auto-trap (the
    patched-forward path) bypasses this guard internally.
    """

    def _make_manager(self, *, skip_discovery: bool, with_tensor_manager: bool = True) -> OffloadManager:
        om = OffloadManager(f"test_guard_{skip_discovery}_{with_tensor_manager}")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=skip_discovery))
        if with_tensor_manager:
            mock_tm = MagicMock()
            mock_tm.trap.return_value = MagicMock()
            om._tensor_manager = mock_tm
        return om

    def test_raises_when_skip_discovery_true(self) -> None:
        om = self._make_manager(skip_discovery=True)
        with pytest.raises(RuntimeError, match="skip_discovery"):
            om.offload_block("encoder")

    def test_allows_when_skip_discovery_false(self) -> None:
        om = self._make_manager(skip_discovery=False)
        # Should not raise; just delegates to tensor_manager.trap.
        om.offload_block("encoder")
        om._tensor_manager.trap.assert_called_once_with("encoder")

    def test_tensor_manager_uninitialized_error_takes_precedence(self) -> None:
        om = self._make_manager(skip_discovery=True, with_tensor_manager=False)
        with pytest.raises(RuntimeError, match="Tensor manager not initialized"):
            om.offload_block("encoder")

    def test_auto_trap_patched_forward_is_not_blocked(self) -> None:
        """The patched-forward closure built by ``_patch_module_forward`` must
        keep working when ``skip_discovery=True`` — it bypasses
        ``offload_block()`` and goes straight to ``_tensor_manager.trap``.
        """
        model = _Model()
        om = self._make_manager(skip_discovery=True)
        om._model = model

        om._patch_module_forward(model.layers[0], "layers.0")

        x = torch.randn(2, 4)
        # Forward must run end-to-end without raising the guard.
        model.layers[0](x)
        om._tensor_manager.trap.assert_called_once_with("layers.0")


class TestLabelsWithDurations:
    """``_labels_with_durations`` is the helper backing ``get_median_duration_ms``
    and ``get_min_duration_ms`` so labels with tensor data but no duration
    samples (e.g. modules whose forward was never called) don't get summarised
    with empty input. ``get_layer_stats`` keeps those labels with
    ``duration=None`` and the strategy layer drops them downstream.
    """

    def test_returns_only_labels_with_both_measurements(self) -> None:
        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("layer_with_both", {1, 2})
        collector.add_duration("layer_with_both", 5.0)
        collector.add_tensors("layer_only_tensors", {3, 4})  # forward never called
        collector.add_duration("layer_only_duration", 7.0)  # tensors never seen

        labels = collector._labels_with_durations()

        assert labels == ["layer_with_both"]

    def test_preserves_insertion_order(self) -> None:
        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("b", {1})
        collector.add_tensors("a", {2})
        collector.add_tensors("c", {3})
        collector.add_duration("a", 1.0)
        collector.add_duration("b", 2.0)
        collector.add_duration("c", 3.0)

        # Order follows ``tensor_measurements``, the insertion order of add_tensors.
        assert collector._labels_with_durations() == ["b", "a", "c"]

    def test_get_median_skips_labels_without_durations(self) -> None:
        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("kept", {1})
        collector.add_duration("kept", 4.0)
        collector.add_tensors("dropped", {2})

        assert collector.get_median_duration_ms() == {"kept": pytest.approx(4.0)}

    def test_get_min_skips_labels_without_durations(self) -> None:
        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("kept", {1})
        collector.add_duration("kept", 2.5)
        collector.add_tensors("dropped", {2})

        assert collector.get_min_duration_ms() == {"kept": pytest.approx(2.5)}


class TestOffloadBlockGuardHonorsFallback:
    """The guard must key off what actually happened, not the requested value.

    When ``skip_discovery=True`` is requested but no patched modules are
    reachable, ``_transition_to_warmup`` falls back to a full DISCOVERY phase
    and flips ``skip_discovery_honored`` to ``False``. That fallback topology —
    no patched modules, discovery running — is exactly the manual
    ``offload_block()`` case, so the guard must not fire there.
    """

    def _make_manager(self, *, honored: bool) -> OffloadManager:
        om = OffloadManager(f"test_guard_honored_{honored}")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=True))
        mock_tm = MagicMock()
        mock_tm.trap.return_value = MagicMock()
        om._tensor_manager = mock_tm
        om._skip_discovery_honored = honored
        return om

    def test_allows_manual_block_when_skip_fell_back_to_discovery(self) -> None:
        om = self._make_manager(honored=False)

        om.offload_block("encoder")

        om._tensor_manager.trap.assert_called_once_with("encoder")

    def test_still_raises_when_skip_was_honored(self) -> None:
        om = self._make_manager(honored=True)

        with pytest.raises(RuntimeError, match="skip_discovery"):
            om.offload_block("encoder")


class TestItersBeforeInferenceFloor:
    """``iters_before_inference`` must return a bound that can reach INFERENCE.

    ``update_state`` runs as a post-forward hook, so PROFILING → INFERENCE
    needs at least one profile forward. A returned bound of ``0`` makes the
    documented drive loop (``for _ in range(om.iters_before_inference)``) a
    no-op, leaving the model stuck in PROFILING with plausible-looking output
    and nothing logged.
    """

    def _make_manager(self, **config_kwargs: object) -> OffloadManager:
        om = OffloadManager(f"test_iters_floor_{sorted(config_kwargs.items())}")
        om.set_config(OffloadConfig(**config_kwargs))  # type: ignore[arg-type]
        return om

    def test_skip_discovery_with_zero_profiling_iters_still_drives_one_forward(self) -> None:
        om = self._make_manager(skip_discovery=True, profiling_iters=0)
        om._skip_discovery_honored = True

        assert om.iters_before_inference >= 1

    def test_full_discovery_with_zero_profiling_iters_still_drives_a_profile_forward(self) -> None:
        om = self._make_manager(skip_discovery=False, discovery_iters=3, profiling_iters=0)

        # 3 discovery forwards + at least 1 profile forward.
        assert om.iters_before_inference >= 4

    def test_nonzero_profiling_iters_is_not_inflated(self) -> None:
        om = self._make_manager(skip_discovery=True, profiling_iters=4)
        om._skip_discovery_honored = True

        assert om.iters_before_inference == 4


class TestSetConfigWarnsOnOneShotFields:
    """One-shot fields baked into the live ``TensorManager`` must not change silently.

    ``_initialize_tensor_manager`` runs only once, so every constructor
    argument it reads is one-shot. Accepting a later ``set_config`` for those
    fields leaves ``om.config`` reporting a value the live manager never saw.
    """

    def _make_manager(self, config: OffloadConfig) -> OffloadManager:
        om = OffloadManager(f"test_oneshot_warn_{id(config)}")
        om.set_config(config)
        om._tensor_manager = MagicMock()
        return om

    def test_warns_when_one_shot_field_changes_after_init(self) -> None:
        om = self._make_manager(OffloadConfig(max_gpu_mem_fraction=0.9))

        with pytest.warns(UserWarning, match="max_gpu_mem_fraction"):
            om.set_config(OffloadConfig(max_gpu_mem_fraction=0.5))

    def test_warning_names_every_changed_one_shot_field(self) -> None:
        om = self._make_manager(OffloadConfig(include_patterns=["a.*"], num_blocks=4))

        with pytest.warns(UserWarning) as record:
            om.set_config(OffloadConfig(include_patterns=["b.*"], num_blocks=8))

        message = "\n".join(str(w.message) for w in record)
        assert "include_patterns" in message
        assert "num_blocks" in message

    def test_no_warning_when_only_reapplied_fields_change(self) -> None:
        om = self._make_manager(OffloadConfig(profiling_iters=3))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            om.set_config(OffloadConfig(profiling_iters=7))

    def test_no_warning_before_manager_is_initialized(self) -> None:
        om = OffloadManager("test_oneshot_warn_pre_init")
        om.set_config(OffloadConfig(num_blocks=4))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            om.set_config(OffloadConfig(num_blocks=8))

    def test_warns_when_offload_timing_changes_after_init(self) -> None:
        """Construction-only timing mode must not flip silently under set_config."""
        om = self._make_manager(OffloadConfig(offload_timing="off"))

        with pytest.warns(UserWarning, match="offload_timing"):
            om.set_config(OffloadConfig(offload_timing="eager"))

    def test_warns_when_piecewise_prefetch_changes_after_init(self) -> None:
        """Construction-only piecewise policy must not flip silently under set_config."""
        om = self._make_manager(OffloadConfig(piecewise_prefetch="warn"))

        with pytest.warns(UserWarning, match="piecewise_prefetch"):
            om.set_config(OffloadConfig(piecewise_prefetch="error"))

    def test_offload_timing_and_piecewise_are_oneshot_fields(self) -> None:
        from flextensor.offload_manager import _TENSOR_MANAGER_ONESHOT_FIELDS

        assert "offload_timing" in _TENSOR_MANAGER_ONESHOT_FIELDS
        assert "piecewise_prefetch" in _TENSOR_MANAGER_ONESHOT_FIELDS


class TestLayerStatsResetPerWarmupCycle:
    """A fresh discovery cycle must rebuild ``layer_stats``.

    ``prepare_warmup_mode`` allocates a new collector and clears
    ``_layer_stats_computed`` precisely so the next profile setup rebuilds.
    Keying the skip path off ``layer_stats is None`` would make a second
    ``offload()`` on the same ``TensorManager`` silently reuse the previous
    cycle's stats — built from a different model's tensor IDs.
    """

    def _make_tensor_manager(self) -> TensorManager:
        return TensorManager(
            device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False
        )

    def test_prepare_warmup_mode_clears_previous_cycle_stats(self) -> None:
        tm = self._make_tensor_manager()
        tm.layer_stats = [IterativeLayerStatistics(label="stale", tensor_ids={1}, duration=1.0)]

        tm.prepare_warmup_mode()

        assert tm.layer_stats is None, (
            "stale stats from the previous offload cycle survived into a fresh "
            "discovery cycle; the profile loader would be wired with old tensor IDs"
        )

    def test_second_cycle_rebuilds_from_the_new_collector(self) -> None:
        tm = self._make_tensor_manager()
        tm.tensors_map = {}
        tm.layer_stats = [IterativeLayerStatistics(label="stale", tensor_ids={1}, duration=1.0)]

        tm.prepare_warmup_mode()
        tm.layer_statistics_collector.add_tensors("fresh", set())
        tm._compute_profile_layer_stats()

        labels = {stat.label for stat in (tm.layer_stats or [])}
        assert "stale" not in labels

    def test_prepare_warmup_mode_clears_observed_cross_refs(self) -> None:
        """Stale cross-ref ids must not leak into a new cycle.

        These are CPython ``id()`` values whose tensors are freed between
        cycles, so ids can be recycled. A stale entry would suppress the new
        tensor's promotion warning and silently drop it from ``layer_stats``
        in ``prepare_infer_mode``.
        """
        tm = self._make_tensor_manager()
        tm.observed_cross_refs.add(123456)

        tm.prepare_warmup_mode()

        assert tm.observed_cross_refs == set()


class TestOneShotWarningValueSemantics:
    """The one-shot diff must compare values, not object identity."""

    def _make_manager(self, config: OffloadConfig) -> OffloadManager:
        om = OffloadManager(f"test_oneshot_semantics_{id(config)}")
        om.set_config(config)
        om._tensor_manager = MagicMock()
        return om

    def test_deep_copied_strategy_is_not_reported_as_a_change(self) -> None:
        """``model_copy(deep=True)`` clones ``load_strategy``; that is not a user change.

        ``Strategy`` defines no ``__eq__``, so a naive ``!=`` compares identity
        and fires a false positive. A warning that cries wolf on a legitimate
        copy trains users to ignore the real ones.
        """
        from flextensor.strategy import KnapsackStrategy

        om = self._make_manager(OffloadConfig(load_strategy=KnapsackStrategy()))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            om.set_config(om.config.model_copy(update={"profiling_iters": 5}, deep=True))

    def test_genuinely_different_strategy_still_warns(self) -> None:
        """The tolerance must not mask a real strategy swap."""
        from flextensor.strategy import GreedyStrategy, KnapsackStrategy

        om = self._make_manager(OffloadConfig(load_strategy=KnapsackStrategy()))

        with pytest.warns(UserWarning, match="load_strategy"):
            om.set_config(OffloadConfig(load_strategy=GreedyStrategy()))


class TestItersBeforeInferenceFloorsDiscoveryToo:
    """``discovery_iters=0`` must not strand the model in DISCOVERY.

    ``update_state`` compares ``_iteration_count >= discovery_iters`` as a
    post-forward hook, so an active DISCOVERY phase always consumes at least
    one forward. Flooring only the profile component left the identical
    stop-short bug reachable from the other side.
    """

    def _make_manager(self, **config_kwargs: object) -> OffloadManager:
        om = OffloadManager(f"test_discovery_floor_{sorted(config_kwargs.items())}")
        om.set_config(OffloadConfig(**config_kwargs))  # type: ignore[arg-type]
        return om

    def test_zero_discovery_iters_still_budgets_a_discovery_forward(self) -> None:
        om = self._make_manager(skip_discovery=False, discovery_iters=0, profiling_iters=10)

        # 1 discovery forward + 10 profile forwards.
        assert om.iters_before_inference == 11

    def test_both_components_zero_budgets_one_each(self) -> None:
        om = self._make_manager(skip_discovery=False, discovery_iters=0, profiling_iters=0)

        assert om.iters_before_inference == 2

    def test_honored_skip_still_excludes_discovery_entirely(self) -> None:
        """The floor must not resurrect the discovery component when skipped."""
        om = self._make_manager(skip_discovery=True, discovery_iters=0, profiling_iters=3)
        om._skip_discovery_honored = True

        assert om.iters_before_inference == 3


class TestSkipDiscoveryHonoredOnRestoredProfile:
    """The restored-profile path must not leave a stale honored flag.

    ``_transition_to_warmup`` returns early when a profile was restored, so it
    previously never touched ``_skip_discovery_honored``. A ``False`` left over
    from an earlier fallback cycle would let ``offload_block()`` past its guard
    on a path where no discovery phase ran at all.
    """

    def test_restored_profile_resets_stale_honored_flag(self) -> None:
        from flextensor.state_handler import TensorManagerState

        om = OffloadManager("test_restored_profile_honored")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=True))
        om._model = _Model()

        mock_tm = MagicMock()
        mock_tm.initialize_warmup.return_value = om._model
        mock_tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
        # The manager-level gate keys off the explicit restore marker, not on
        # the presence of state — a live cycle leaves state behind too.
        mock_tm.state_restored_from_profile = True
        om._tensor_manager = mock_tm
        om._skip_discovery_honored = False  # stale from a previous fallback cycle

        with patch.object(om, "_transition_to_inference"):
            om._transition_to_warmup()

        assert om.skip_discovery_honored is True

    def test_offload_block_stays_guarded_after_restored_profile(self) -> None:
        """The whole point: the guard must still reject manual blocks here."""
        from flextensor.state_handler import TensorManagerState

        om = OffloadManager("test_restored_profile_guard")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=True))
        om._model = _Model()

        mock_tm = MagicMock()
        mock_tm.initialize_warmup.return_value = om._model
        mock_tm.tensor_manager_state = MagicMock(spec=TensorManagerState)
        # The manager-level gate keys off the explicit restore marker, not on
        # the presence of state — a live cycle leaves state behind too.
        mock_tm.state_restored_from_profile = True
        om._tensor_manager = mock_tm
        om._skip_discovery_honored = False

        with patch.object(om, "_transition_to_inference"):
            om._transition_to_warmup()

        with pytest.raises(RuntimeError, match="skip_discovery"):
            om.offload_block("encoder")


class TestOneShotWarningComparesAgainstCapturedValues:
    """The diff baseline must be what the TensorManager captured.

    ``set_config`` overwrites ``self.config`` on every call, so diffing against
    it means a second application of an already-diverged config compares equal
    and stays silent while the live manager is still out of sync.
    """

    def _make_manager(self, config: OffloadConfig) -> OffloadManager:
        """Install a live manager with the complete snapshot ``offload()`` would take."""
        from flextensor.offload_manager import _TENSOR_MANAGER_ONESHOT_FIELDS

        om = OffloadManager(f"test_oneshot_captured_{id(config)}")
        om.set_config(config)
        om._tensor_manager = MagicMock()
        om._tensor_manager_oneshot_snapshot = {
            name: getattr(config, name, None) for name in (*_TENSOR_MANAGER_ONESHOT_FIELDS, "skip_discovery")
        }
        return om

    def test_repeated_divergent_set_config_keeps_warning(self) -> None:
        om = self._make_manager(OffloadConfig(num_blocks=4))

        with pytest.warns(UserWarning, match="num_blocks"):
            om.set_config(OffloadConfig(num_blocks=8))

        with pytest.warns(UserWarning, match="num_blocks"):
            om.set_config(OffloadConfig(num_blocks=8))

    def test_reverting_to_the_captured_value_stops_warning(self) -> None:
        om = self._make_manager(OffloadConfig(num_blocks=4))

        with pytest.warns(UserWarning, match="num_blocks"):
            om.set_config(OffloadConfig(num_blocks=8))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            om.set_config(OffloadConfig(num_blocks=4))


class TestEagerProfilingItersFloor:
    """The public property must not advertise a budget that can't reach INFERENCE.

    ``OffloadManager.eager_profiling_iters`` is documented as "profiling
    forwards required before the PROFILING→INFERENCE transition" and is driven
    as a loop bound in production (``contrib/vllm/worker.py``). ``update_state``
    is a post-forward hook, so one forward is always required; returning ``0``
    at ``profiling_iters=0`` makes that loop a no-op. vLLM is shielded by its
    own ``VLLM_PROFILING_ITER_FLOOR``, but a direct caller is not.
    """

    def _make_manager(self, **config_kwargs: object) -> OffloadManager:
        om = OffloadManager(f"test_eager_floor_{sorted(config_kwargs.items())}")
        om.set_config(OffloadConfig(**config_kwargs))  # type: ignore[arg-type]
        return om

    def test_zero_profiling_iters_still_advertises_one_forward(self) -> None:
        om = self._make_manager(profiling_iters=0)

        assert om.eager_profiling_iters >= 1

    def test_nonzero_profiling_iters_is_not_inflated(self) -> None:
        om = self._make_manager(profiling_iters=6)

        assert om.eager_profiling_iters == 6

    def test_agrees_with_the_transition_threshold(self) -> None:
        """The advertised count must actually reach INFERENCE.

        ``update_state`` compares against the unfloored internal threshold, so
        driving the advertised number of forwards must satisfy it.
        """
        om = self._make_manager(profiling_iters=0)

        assert om.eager_profiling_iters >= om._eager_profiling_iters()


class TestSkipDiscoveryHonoredTriState:
    """``None`` must be distinguishable from ``True``/``False`` at every gate."""

    def _make_manager(self) -> OffloadManager:
        om = OffloadManager("test_tristate")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=True, discovery_iters=2, profiling_iters=7))
        om._compiled.active = False
        om._compiled.replan_active = False
        return om

    def test_undetermined_budgets_for_discovery(self) -> None:
        """Under-counting strands the model; over-counting costs a spare forward."""
        om = self._make_manager()
        assert om.skip_discovery_honored is None

        assert om.iters_before_inference == 9

    def test_honored_drops_the_discovery_component(self) -> None:
        om = self._make_manager()
        om._skip_discovery_honored = True

        assert om.iters_before_inference == 7

    def test_fallback_keeps_the_discovery_component(self) -> None:
        om = self._make_manager()
        om._skip_discovery_honored = False

        assert om.iters_before_inference == 9

    def test_offload_block_blocked_while_undetermined(self) -> None:
        """Permitting manual blocks before any discovery ran would be unsafe."""
        om = self._make_manager()
        om._tensor_manager = MagicMock()
        assert om.skip_discovery_honored is None

        with pytest.raises(RuntimeError, match="skip_discovery"):
            om.offload_block("encoder")

    def test_offload_block_allowed_only_on_a_known_fallback(self) -> None:
        om = self._make_manager()
        om._tensor_manager = MagicMock()
        om._skip_discovery_honored = False

        om.offload_block("encoder")

        om._tensor_manager.trap.assert_called_once_with("encoder")


class TestSecondOffloadIsNotTreatedAsRestoredProfile:
    """A previous live cycle must not look like a restored profile.

    ``_create_loader(prepare_state=True)`` stores a real ``TensorManagerState``
    once a cycle reaches INFERENCE. Branching the ``initialize_*`` short-circuits
    on ``tensor_manager_state is not None`` therefore made a second ``offload()``
    on the same manager jump straight to INFERENCE with the previous model's
    plan and tensor IDs — the reset in ``prepare_warmup_mode`` is never reached,
    because ``initialize_warmup`` returns above it.
    """

    def _tensor_manager(self) -> TensorManager:
        tm = TensorManager(
            device_gpu=torch.device("cpu"), tensor_manager_load_strategy=MagicMock(), pinned_memory=False
        )
        tm.model = _Model()
        tm.tensors_map = {id(p): p for p in tm.model.parameters()}
        return tm

    def test_live_cycle_state_does_not_short_circuit_warmup(self) -> None:
        tm = self._tensor_manager()
        # What a completed cycle leaves behind.
        tm.tensor_manager_state = MagicMock(name="state_from_previous_live_cycle")
        tm.prepare_infer_load_mode = MagicMock()

        tm.initialize_warmup()

        tm.prepare_infer_load_mode.assert_not_called(), "a live-cycle state must not replay the inference path"

    def test_live_cycle_state_does_not_short_circuit_profile(self) -> None:
        tm = self._tensor_manager()
        tm.tensor_manager_state = MagicMock(name="state_from_previous_live_cycle")
        tm._move_non_offloaded_tensors_to_gpu = MagicMock()
        tm.prepare_profile_mode = MagicMock()
        tm.prepare_profile_direct_mode_model = MagicMock(return_value=tm.model)
        tm.prepare_profile_direct_mode = MagicMock()

        tm.initialize_profile()

        tm._move_non_offloaded_tensors_to_gpu.assert_called_once()

    def test_restored_profile_still_short_circuits(self) -> None:
        """The externally restored path must keep working."""
        tm = self._tensor_manager()
        tm.tensor_manager_state = MagicMock(name="restored_state")
        tm._state_restored_from_profile = True
        tm.prepare_infer_load_mode = MagicMock()
        tm.prepare_final_model = MagicMock(return_value=tm.model)

        tm.initialize_warmup()

        tm.prepare_infer_load_mode.assert_called_once()

    def test_restore_marker_is_consumed_by_inference(self) -> None:
        """One restore, one short-circuit — the next cycle runs fresh."""
        tm = self._tensor_manager()
        tm.tensor_manager_state = MagicMock(name="restored_state")
        tm._state_restored_from_profile = True

        tm.initialize_inference()

        assert tm._state_restored_from_profile is False


class TestPreInferenceItersRemainsAnUpperBound:
    """``pre_inference_iters`` must never undercount ``iters_before_inference``.

    Both phase counts accept ``0`` while the state machine always consumes at
    least one forward per phase it drives, so the raw sum stopped being an
    upper bound once the runtime counts were floored.
    """

    @pytest.mark.parametrize(
        ("discovery_iters", "profiling_iters"),
        [(0, 0), (0, 10), (2, 0), (2, 7)],
    )
    def test_never_below_the_runtime_count(self, discovery_iters: int, profiling_iters: int) -> None:
        config = OffloadConfig(skip_discovery=False, discovery_iters=discovery_iters, profiling_iters=profiling_iters)
        om = OffloadManager(f"test_upper_bound_{discovery_iters}_{profiling_iters}")
        om.set_config(config)
        om._compiled.active = False
        om._compiled.replan_active = False

        assert config.pre_inference_iters >= om.iters_before_inference


def _live_cycle_state():
    """A ``TensorManagerState`` of the shape a completed live cycle leaves behind."""
    from flextensor.state_handler import TensorManagerState

    return TensorManagerState(
        loader_type="strategy",
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
        shm_block_name_map=None,
    )


class TestManagerLevelRestoredProfileGate:
    """``OffloadManager._transition_to_warmup`` must use the restore marker too.

    Fixing only the ``TensorManager.initialize_*`` checks left the orchestration
    branch deciding "a profile was restored" from
    ``isinstance(tensor_manager_state, TensorManagerState)``. A completed live
    cycle leaves exactly that object, so a second ``offload()`` still jumped to
    INFERENCE with the previous model's plan.

    These drive ``OffloadManager`` rather than ``TensorManager`` directly —
    exercising the tensor-manager entry points in isolation is what let the
    partial fix look complete.
    """

    def _manager(self, *, restored: bool) -> OffloadManager:
        om = OffloadManager(f"test_mgr_gate_{restored}")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=False))
        model = _Model()
        om._model = model

        mock_tm = MagicMock()
        mock_tm.initialize_warmup.return_value = model
        mock_tm.initialize_profile.return_value = model
        mock_tm.initialize_inference.return_value = model
        # What a completed live cycle leaves behind, either way.
        mock_tm.tensor_manager_state = _live_cycle_state()
        mock_tm.state_restored_from_profile = restored
        om._tensor_manager = mock_tm
        return om

    def test_live_cycle_state_does_not_jump_to_inference(self) -> None:
        om = self._manager(restored=False)

        om._transition_to_warmup()

        assert om.phase is not OffloadPhase.INFERENCE, (
            "a second offload() must run a fresh cycle, not replay the previous model's plan"
        )

    def test_restored_profile_still_jumps_to_inference(self) -> None:
        om = self._manager(restored=True)

        om._transition_to_warmup()

        assert om.phase is OffloadPhase.INFERENCE

    def test_bare_mock_manager_does_not_trigger_the_restore_path(self) -> None:
        """A bare MagicMock attribute is truthy and must not read as restored."""
        om = OffloadManager("test_mgr_gate_bare_mock")
        om.set_config(OffloadConfig(enabled=True, skip_discovery=False))
        model = _Model()
        om._model = model
        mock_tm = MagicMock()  # every attribute is a truthy MagicMock
        mock_tm.initialize_warmup.return_value = model
        om._tensor_manager = mock_tm

        om._transition_to_warmup()

        assert om.phase is not OffloadPhase.INFERENCE
