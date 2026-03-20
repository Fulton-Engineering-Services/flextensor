# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive strategy that runs all available strategies and picks the best result.

The :class:`AdaptiveStrategy` conforms to the :class:`Strategy` protocol so it
can be used as a drop-in replacement in :class:`OffloadConfig`.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import ClassVar

from flextensor.collectors import LayerStatistics  # noqa: TC001
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator
from flextensor.strategy.assignment import (
    OptimizedRoundRobinAssignment,
    StrictRoundRobinAssignment,
)
from flextensor.strategy.evaluation import StrategyScore, evaluate_strategy_result
from flextensor.strategy.global_strategy import (
    GlobalOffloadStrategy,
    GlobalTensorSelectionStrategy,
)
from flextensor.strategy.knapsack import (
    KnapsackBlockStrategy,
    KnapsackStrategy,
)
from flextensor.strategy.protocol import Strategy, StrategyComputeError, StrategyResult
from flextensor.strategy.utils import validate_memory_params

logger = logging.getLogger(__name__)


class AdaptiveStrategy:
    """Run all available strategies and return the best result.

    For block-based loaders (the common path), the following candidates are
    evaluated:

    * :class:`KnapsackBlockStrategy`
    * :class:`GlobalOffloadStrategy` (Optimized assignment)
    * :class:`GlobalOffloadStrategy` (Strict assignment)
    * :class:`GlobalTensorSelectionStrategy` (Optimized assignment)
    * :class:`GlobalTensorSelectionStrategy` (Strict assignment)

    For non-block loaders only :class:`KnapsackStrategy` is used.

    After all candidates run, the one with the lowest overhead that also
    satisfies the memory constraint (if any) is selected.  The winning
    strategy's name is available via :attr:`selected_strategy_name`.

    Example:
        >>> strategy = AdaptiveStrategy(
        ...     scale=1.0,
        ...     max_gpu_mem_bytes=50 * 1024**3,
        ...     n_blocks=4,
        ... )
        >>> result = strategy.compute(layer_stats, memory_stats)
        >>> print(strategy.selected_strategy_name)
        'KnapsackBlock'
    """

    BLOCK_LOADER_TYPES: ClassVar[set[str]] = {"allocation_block_transfer", "raw_block_transfer"}

    def __init__(
        self,
        scale: float = 1.0,
        loader_type: str = "allocation_block_transfer",
        cyclic: bool = False,
        group_size: int = 1,
        threshold_mb: float = 0.1,
        min_blocks: int = 2,
        n_blocks: int = 4,
        max_gpu_mem_bytes: int | None = None,
        extra_optimization: bool = False,
    ):
        """Initialize AdaptiveStrategy.

        Args:
            scale: Duration multiplier (baseline). ``0.9`` = 10% safety margin,
                ``1.0`` = exact fit, ``2.0`` = allow 100% overhead.
            loader_type: Type of tensor loader. Determines which strategy
                candidates are instantiated.
            cyclic: Whether to apply cyclic strategy extension (KnapsackStrategy only).
            group_size: Number of layers to merge into groups (Knapsack strategies).
            threshold_mb: Minimum tensor size threshold in MB.
            min_blocks: Minimum number of blocks for OptimizedRoundRobinAssignment
                search range. Does not affect Strict or Knapsack candidates.
            n_blocks: Number of memory blocks.
            max_gpu_mem_bytes: Hard GPU memory limit in bytes (never exceed).
                When ``None``, latency mode — respect ``scale`` strictly.
            extra_optimization: If True, include additional slower strategy candidates
                (GlobalTensorSelectionStrategy) that may find better solutions at the
                cost of significantly longer optimization time. Default is False.
        """
        max_gpu_mem_bytes = validate_memory_params(scale, max_gpu_mem_bytes)
        if min_blocks < 2:
            raise ValueError(f"min_blocks must be >= 2, got {min_blocks}")
        if min_blocks > n_blocks:
            raise ValueError(f"min_blocks ({min_blocks}) must be <= n_blocks ({n_blocks})")

        self.scale = scale
        self.loader_type = loader_type
        self.cyclic = cyclic
        self.group_size = group_size
        self.threshold_mb = threshold_mb
        self.n_blocks = n_blocks
        self.min_blocks = min_blocks
        self.max_gpu_mem_bytes = max_gpu_mem_bytes
        self.extra_optimization = extra_optimization

        # Set after compute()
        self._selected_strategy_name: str | None = None
        self._all_scores: list[StrategyScore] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
    ) -> StrategyResult:
        """Run all strategy candidates and return the best result.

        Args:
            layer_stats: List of layer statistics containing tensor info and durations.
            memory_stats: Optional dict mapping tensor size (bytes) to transfer time (ms).

        Returns:
            :class:`StrategyResult` from the best-scoring candidate.
        """
        interpolator = MemoryTransferInterpolator(memory_stats) if memory_stats else None
        candidates = self._build_candidates()

        best_result: StrategyResult | None = None
        best_score: StrategyScore | None = None
        all_scores: list[StrategyScore] = []

        for name, strategy in candidates:
            t0 = time.perf_counter()
            try:
                result = strategy.compute(layer_stats, memory_stats)
            except StrategyComputeError:
                logger.warning("Strategy %s failed, skipping", name, exc_info=True)
                continue
            elapsed = time.perf_counter() - t0

            score = evaluate_strategy_result(
                result,
                layer_stats,
                strategy_name=name,
                interpolator=interpolator,
                max_gpu_mem_bytes=self.max_gpu_mem_bytes,
            )
            all_scores.append(score)

            logger.debug(
                "Candidate %-35s  overhead=%.1f%%  peak=%.2f GB  violations=%d  valid=%s  time=%.1fs",
                name,
                score.estimated_overhead * 100,
                score.peak_memory_bytes / 1024**3,
                score.consecutive_violations,
                score.is_valid,
                elapsed,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_result = result

        self._all_scores = all_scores

        if best_result is None or best_score is None:
            raise RuntimeError("All strategy candidates failed — no result produced")

        self._selected_strategy_name = best_score.strategy_name

        if not best_score.is_valid:
            warnings.warn(
                f"AdaptiveStrategy: best result ({best_score.strategy_name}) is not fully valid — "
                f"peak={best_score.peak_memory_bytes / 1024**3:.2f} GB, "
                f"violations={best_score.consecutive_violations}, "
                f"overhead={best_score.estimated_overhead * 100:.1f}%",
                UserWarning,
                stacklevel=2,
            )

        return best_result

    @property
    def selected_strategy_name(self) -> str:
        """Name of the strategy that produced the best result.

        Raises:
            RuntimeError: If accessed before :meth:`compute` has been called.
        """
        if self._selected_strategy_name is None:
            raise RuntimeError("selected_strategy_name is not available until compute() has been called")
        return self._selected_strategy_name

    @property
    def all_scores(self) -> list[StrategyScore]:
        """Scores from all candidates evaluated in the last :meth:`compute` call."""
        return list(self._all_scores)

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def _build_candidates(self) -> list[tuple[str, Strategy]]:
        """Build the list of ``(name, strategy_instance)`` candidates."""
        if self.loader_type in self.BLOCK_LOADER_TYPES:
            return self._block_candidates()
        return self._non_block_candidates()

    def _block_candidates(self) -> list[tuple[str, Strategy]]:
        """Candidates for block-based loaders."""
        scale = self.scale
        threshold_mb = self.threshold_mb
        n_blocks = self.n_blocks
        min_blocks = self.min_blocks
        max_mem = self.max_gpu_mem_bytes

        candidates: list[tuple[str, Strategy]] = [
            (
                "KnapsackBlock",
                KnapsackBlockStrategy(
                    scale=scale,
                    group_size=self.group_size,
                    threshold_mb=threshold_mb,
                    n_blocks=n_blocks,
                    max_gpu_mem_bytes=max_mem,
                ),
            ),
            (
                "GlobalOffload(Optimized)",
                GlobalOffloadStrategy(
                    scale=scale,
                    threshold_mb=threshold_mb,
                    n_blocks=n_blocks,
                    max_gpu_mem_bytes=max_mem,
                    assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=min_blocks, max_blocks=n_blocks),
                ),
            ),
            (
                "GlobalOffload(Strict)",
                GlobalOffloadStrategy(
                    scale=scale,
                    threshold_mb=threshold_mb,
                    n_blocks=n_blocks,
                    max_gpu_mem_bytes=max_mem,
                    assignment_strategy=StrictRoundRobinAssignment(),
                ),
            ),
        ]

        if self.extra_optimization:
            candidates.extend([
                (
                    "TensorSelection(Optimized)",
                    GlobalTensorSelectionStrategy(
                        scale=scale,
                        threshold_mb=threshold_mb,
                        n_blocks=n_blocks,
                        max_gpu_mem_bytes=max_mem,
                        pop_size=50,
                        epoch=100,
                        max_early_stop=50,
                        assignment_strategy=OptimizedRoundRobinAssignment(min_blocks=min_blocks, max_blocks=n_blocks),
                    ),
                ),
                (
                    "TensorSelection(Strict)",
                    GlobalTensorSelectionStrategy(
                        scale=scale,
                        threshold_mb=threshold_mb,
                        n_blocks=n_blocks,
                        max_gpu_mem_bytes=max_mem,
                        pop_size=50,
                        epoch=100,
                        max_early_stop=50,
                        assignment_strategy=StrictRoundRobinAssignment(),
                    ),
                ),
            ])

        return candidates

    def _non_block_candidates(self) -> list[tuple[str, Strategy]]:
        """Candidates for non-block loaders."""
        return [
            (
                "Knapsack",
                KnapsackStrategy(
                    scale=self.scale,
                    cyclic=self.cyclic,
                    group_size=self.group_size,
                    threshold_mb=self.threshold_mb,
                    max_gpu_mem_bytes=self.max_gpu_mem_bytes,
                ),
            ),
        ]
