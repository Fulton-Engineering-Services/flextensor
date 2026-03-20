# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for flextensor.strategy.utils."""

import numpy as np
import pytest

from flextensor.strategy.utils import EarlyStopCallback, validate_memory_params

# ===========================================================================
# validate_memory_params
# ===========================================================================


class TestValidateMemoryParams:
    def test_returns_none_when_max_is_none(self):
        result = validate_memory_params(1.0, None)
        assert result is None

    def test_returns_max_when_set(self):
        result = validate_memory_params(1.0, 1000)
        assert result == 1000

    def test_scale_zero_raises(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            validate_memory_params(0.0, None)

    def test_scale_negative_raises(self):
        with pytest.raises(ValueError, match="scale must be positive"):
            validate_memory_params(-1.0, None)

    def test_fractional_scale(self):
        result = validate_memory_params(0.01, 1000)
        assert result == 1000

    def test_max_gpu_mem_bytes_zero_raises(self):
        with pytest.raises(ValueError, match="max_gpu_mem_bytes must be positive"):
            validate_memory_params(1.0, 0)

    def test_max_gpu_mem_bytes_negative_raises(self):
        with pytest.raises(ValueError, match="max_gpu_mem_bytes must be positive"):
            validate_memory_params(1.0, -1)


# ===========================================================================
# EarlyStopCallback
# ===========================================================================


class TestEarlyStopCallback:
    @staticmethod
    def _objective(x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def test_no_stop_before_max_stall(self):
        cb = EarlyStopCallback(max_stall=3, objective_func=self._objective)
        x = np.array([1.0])
        assert cb(x) is False  # sets best=1.0, stall=0
        assert cb(x) is False  # no improvement, stall=1
        assert cb(x) is False  # stall=2
        assert cb(x) is True  # stall=3 == max_stall

    def test_improvement_resets_stall(self):
        cb = EarlyStopCallback(max_stall=2, objective_func=self._objective)
        x_bad = np.array([10.0])
        x_good = np.array([1.0])
        assert cb(x_bad) is False  # stall 0, best=100
        assert cb(x_bad) is False  # stall 1, no improvement
        assert cb(x_good) is False  # improvement resets stall to 0
        assert cb(x_good) is False  # stall 1
        assert cb(x_good) is True  # stall 2 == max_stall

    def test_dual_annealing_mode_uses_f_directly(self):
        """When context is not None, f_or_convergence is used as the value."""
        calls = []

        def spy_objective(x: np.ndarray) -> float:
            calls.append(x)
            return float(np.sum(x**2))

        cb = EarlyStopCallback(max_stall=5, objective_func=spy_objective)
        x = np.array([99.0])

        cb(x, f_or_convergence=10.0, context=0)
        assert len(calls) == 0, "objective_func should NOT be called in dual_annealing mode"
        assert cb._best_value == pytest.approx(10.0)

    def test_differential_evolution_mode_calls_objective(self):
        """When context is None, objective_func is evaluated."""
        calls = []

        def spy_objective(x: np.ndarray) -> float:
            calls.append(x.copy())
            return float(np.sum(x**2))

        cb = EarlyStopCallback(max_stall=5, objective_func=spy_objective)
        x = np.array([3.0])

        cb(x, f_or_convergence=0.5)
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], x)
        assert cb._best_value == pytest.approx(9.0)

    def test_stops_exactly_at_max_stall(self):
        cb = EarlyStopCallback(max_stall=1, objective_func=self._objective)
        x = np.array([5.0])
        assert cb(x) is False  # first call sets best
        assert cb(x) is True  # stall 1 == max_stall

    @pytest.mark.parametrize("bad_value", [0, -1, -100])
    def test_max_stall_below_one_raises(self, bad_value: int):
        with pytest.raises(ValueError, match=r"max_stall must be >= 1"):
            EarlyStopCallback(max_stall=bad_value, objective_func=self._objective)

    def test_tiny_improvement_below_tolerance_counts_as_stall(self):
        def stub(x: np.ndarray) -> float:
            return 0.0

        cb = EarlyStopCallback(max_stall=2, objective_func=stub)
        cb._best_value = 1.0

        # Improvement smaller than 1e-10 should not reset stall
        cb.objective_func = lambda _: 1.0 - 1e-11
        assert cb(np.array([0.0])) is False  # stall 1
        assert cb(np.array([0.0])) is True  # stall 2

    def test_large_improvement_resets_stall(self):
        cb = EarlyStopCallback(max_stall=2, objective_func=lambda _: 0.0)
        cb._best_value = 100.0

        cb.objective_func = lambda _: 50.0
        assert cb(np.array([0.0])) is False  # large improvement resets
        assert cb._stall_count == 0
        assert cb._best_value == pytest.approx(50.0)
