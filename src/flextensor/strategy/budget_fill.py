# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory-first offload strategies under a hard GPU budget.

Solvers (run alone for comparison)
    * :class:`BudgetFillGreedyStrategy` — ranked-prefix scan under a hard peak budget
    * :class:`BudgetFillLayerDEStrategy` — DE over per-layer offload fractions
    * :class:`BudgetFillTensorDEStrategy` — DE over per-tensor binaries (slow)

Facade
    * :class:`BudgetFillStrategy` — runs greedy, then optionally layer / tensor
      DE, keeping the best feasible / lowest-offload plan. A pinched peak alone
      does not skip DE (less offload may still exist via non-prefix subsets).

Hard constraints
    * Peak ≤ ``max_gpu_mem_bytes`` when feasible (warns and returns the best-effort
      full-offload plan if the budget is unreachable, so callers can still compare)
    * Neighbor transfers on different blocks (via
      :class:`~flextensor.strategy.assignment.StrictRoundRobinAssignment` or
      :class:`~flextensor.strategy.assignment.OptimizedRoundRobinAssignment`,
      same machinery as :class:`~flextensor.strategy.global_strategy.GlobalOffloadStrategy`)

Soft objectives (in order)
    1. Minimize offloaded bytes (maximize GPU-resident weights)
    2. Prefer tensors whose H2D transfer fits in ``prev_duration * scale``
       (from ``memory_stats`` / interpolator, else ``load_time_ms``)

Layer 0 stays resident (nothing to pipeline into). Greedy ranks eligible tensors
fit-then-size and scans ranked prefixes (peak is not monotonic in prefix length
under round-robin assignment). The DE solvers may select non-prefix subsets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

from flextensor.collectors import LayerStatistics, TensorStatistics  # noqa: TC001
from flextensor.memory_transfer_interpolator import MemoryTransferInterpolator
from flextensor.strategy.assignment import AssignmentStrategy, StrictRoundRobinAssignment
from flextensor.strategy.protocol import BlockStrategyData, StrategyComputeError, StrategyResult
from flextensor.strategy.utils import (
    calculate_transfer_to_compute_map,
    compute_label_to_size_map,
    validate_memory_params,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """One eligible tensor with transfer-fit metadata."""

    layer_index: int
    tensor: TensorStatistics
    fits: bool
    transfer_ms: float


def _transfer_ms(
    tensor: TensorStatistics,
    interpolator: MemoryTransferInterpolator | None,
) -> float:
    if interpolator is not None:
        return float(interpolator.bytes_to_duration(tensor.size_bytes))
    return float(tensor.load_time_ms)


def _rank_candidates(
    layer_stats: list[LayerStatistics],
    *,
    threshold_mb: float,
    scale: float,
    interpolator: MemoryTransferInterpolator | None,
) -> list[_Candidate]:
    """Eligible tensors (layers 1..) ranked fit-first, then larger size."""
    threshold_bytes = threshold_mb * 1024 * 1024
    scored: list[tuple[tuple[float | int, ...], _Candidate]] = []
    for index, layer in enumerate(layer_stats):
        if index == 0:
            continue
        window_ms = max(0.0, layer_stats[index - 1].duration * scale)
        for tensor in layer.tensors:
            if tensor.size_bytes <= threshold_bytes:
                continue
            transfer_ms = _transfer_ms(tensor, interpolator)
            fits = transfer_ms <= window_ms
            candidate = _Candidate(
                layer_index=index,
                tensor=tensor,
                fits=fits,
                transfer_ms=transfer_ms,
            )
            scored.append((
                (
                    0 if fits else 1,
                    -tensor.size_bytes,
                    transfer_ms,
                    tensor.tensor_id,
                ),
                candidate,
            ))
    scored.sort(key=lambda item: item[0])
    return [item[1] for item in scored]


def _strategy_map_from_selection(
    layer_stats: list[LayerStatistics],
    selected: list[_Candidate],
) -> dict[str, list[TensorStatistics]]:
    """Pipeline map: transfer during layer i-1 for selected tensors of layer i."""
    by_layer: dict[int, list[TensorStatistics]] = {}
    for candidate in selected:
        by_layer.setdefault(candidate.layer_index, []).append(candidate.tensor)

    strategy: dict[str, list[TensorStatistics]] = {}
    for layer_index, tensors in by_layer.items():
        prev_label = layer_stats[layer_index - 1].label
        strategy[prev_label] = tensors
    return strategy


def _peak_memory(result: StrategyResult, layer_stats: list[LayerStatistics]) -> int:
    """Peak GPU ~ sum(block sizes) + (model - offloaded)."""
    block_memory = 0
    if result.block_data is not None:
        sizes = result.block_data.block_sizes
        block_memory = sum(sizes.values()) if isinstance(sizes, dict) else sum(sizes)
    total_model = sum(tensor.size_bytes for layer in layer_stats for tensor in layer.tensors)
    total_offloaded = sum(tensor.size_bytes for tensors in result.strategy_map.values() for tensor in tensors)
    return block_memory + total_model - total_offloaded


def _build_block_data(
    layer_stats: list[LayerStatistics],
    strategy_map: dict[str, list[TensorStatistics]],
    *,
    n_blocks: int,
    assignment_strategy: AssignmentStrategy,
) -> BlockStrategyData | None:
    """Assign transfer slots with ``assignment_strategy`` and build block data."""
    if n_blocks <= 0 or not strategy_map:
        return None

    # Transfer slots in model order (neighbor constraint applies along this sequence).
    ordered_labels = [layer.label for layer in layer_stats if layer.label in strategy_map]
    transfer_sizes = [sum(t.size_bytes for t in strategy_map[label]) for label in ordered_labels]
    if not transfer_sizes:
        return None

    assignment = assignment_strategy.compute(transfer_sizes, n_blocks)
    if len(assignment) != len(ordered_labels):
        raise StrategyComputeError(f"Assignment length {len(assignment)} != transfer slots {len(ordered_labels)}")

    block_sizes = [0] * n_blocks
    label_to_block_id: dict[str, int] = {}
    allocation_ordered: dict[int, list[str]] = {}
    for label, block_id, size in zip(ordered_labels, assignment, transfer_sizes, strict=True):
        label_to_block_id[label] = block_id
        block_sizes[block_id] = max(block_sizes[block_id], size)
        allocation_ordered.setdefault(block_id, []).append(label)

    block_sizes_dict = {index: size for index, size in enumerate(block_sizes) if size > 0}
    return BlockStrategyData(
        label_to_size_map=compute_label_to_size_map(layer_stats, strategy_map),
        allocation_ordered=allocation_ordered,
        block_sizes=block_sizes_dict,
        label_to_block_id=label_to_block_id,
        transfer_to_compute_map=calculate_transfer_to_compute_map(layer_stats, strategy_map),
    )


def _result_for_selection(
    layer_stats: list[LayerStatistics],
    selected: list[_Candidate],
    *,
    n_blocks: int,
    assignment_strategy: AssignmentStrategy,
) -> StrategyResult:
    strategy_map = _strategy_map_from_selection(layer_stats, selected)
    if not strategy_map:
        return StrategyResult(strategy_map={}, block_data=None)
    block_data = _build_block_data(
        layer_stats,
        strategy_map,
        n_blocks=n_blocks,
        assignment_strategy=assignment_strategy,
    )
    return StrategyResult(strategy_map=strategy_map, block_data=block_data)


def _search_block_count(n_blocks: int) -> int:
    """Block count used inside DE hot loops (Strict RR).

    Optimized assignment typically collapses to 2 blocks on balanced MoE-like
    models; evaluating with Strict@2 matches that peak closely without the
    expensive Optimized search on every objective call.
    """
    if n_blocks <= 0:
        return 0
    if n_blocks == 1:
        return 1
    return 2


def _strict_peak_from_transfer_sizes(
    transfer_sizes: list[int],
    *,
    total_model: float,
    offloaded: float,
    n_blocks: int,
) -> float:
    """Peak = sum(block maxes under Strict RR) + (model - offloaded)."""
    if n_blocks <= 0 or not transfer_sizes:
        return total_model - offloaded
    block_sizes = [0] * n_blocks
    for index, size in enumerate(transfer_sizes):
        block_id = index % n_blocks
        if size > block_sizes[block_id]:
            block_sizes[block_id] = size
    return float(sum(block_sizes)) + total_model - offloaded


def _best_ranked_prefix(
    candidates: list[_Candidate],
    *,
    total_model: float,
    budget: float,
    search_blocks: int,
) -> int:
    """Pick a ranked-prefix length under Strict RR peak.

    Peak is **not** monotonic in prefix length (dropping a transfer can reassign
    later slots and lower ``sum(block maxes)``). Scan all prefixes once:

    * among feasible peaks, minimize offloaded bytes
    * if none are feasible, return the prefix with the lowest peak (best-effort)
    """
    n_cand = len(candidates)
    layer_xfer: dict[int, int] = {}
    offloaded = 0
    best_feasible_k: int | None = None
    best_feasible_off = 0
    best_any_k = 0
    best_any_peak = total_model

    for prefix_len in range(n_cand + 1):
        if prefix_len > 0:
            candidate = candidates[prefix_len - 1]
            size = candidate.tensor.size_bytes
            layer_xfer[candidate.layer_index] = layer_xfer.get(candidate.layer_index, 0) + size
            offloaded += size
        transfer_sizes = [layer_xfer[index] for index in sorted(layer_xfer)]
        peak = _strict_peak_from_transfer_sizes(
            transfer_sizes,
            total_model=total_model,
            offloaded=float(offloaded),
            n_blocks=search_blocks,
        )
        if peak < best_any_peak:
            best_any_peak = peak
            best_any_k = prefix_len
        if peak <= budget and (best_feasible_k is None or offloaded < best_feasible_off):
            best_feasible_k = prefix_len
            best_feasible_off = offloaded

    return best_feasible_k if best_feasible_k is not None else best_any_k


def _soft_objective(offloaded: float | int, nonfit: float | int, total_model: float | int) -> float:
    """Lexicographic soft cost: minimize offload bytes, then non-fitting bytes.

    The nonfit term is strictly dominated by one byte of offload:
    ``0 <= nonfit / (total_model + 1) < 1`` whenever ``nonfit <= total_model``.
    """
    return float(offloaded) + float(nonfit) / (max(float(total_model), 0.0) + 1.0)


def _layer_fraction_metrics(  # noqa: C901
    by_layer: dict[int, list[_Candidate]],
    layer_indices: list[int],
    max_bytes: list[int],
    fractions: list[float],
    *,
    total_model: float,
    search_blocks: int,
    layer_size_prefix: list[list[int]] | None = None,
    layer_nonfit_prefix: list[list[int]] | None = None,
) -> tuple[float, float, float]:
    """Return ``(peak, offloaded_bytes, nonfit_offload_bytes)`` with Strict RR peak.

    Optional ``layer_*_prefix`` cumsums make DE objectives O(layers) instead of
    O(tensors).
    """
    transfer_sizes: list[int] = []
    offloaded = 0
    nonfit = 0
    for index, layer_index in enumerate(layer_indices):
        target = max(0.0, float(fractions[index])) * float(max_bytes[index])
        if target <= 0:
            continue
        if layer_size_prefix is not None:
            sizes = layer_size_prefix[index]
            # Smallest i with cumsize[i] >= target (0-based); take i+1 tensors.
            take = 0
            for i, cum in enumerate(sizes):
                take = i + 1
                if cum >= target:
                    break
            if take <= 0:
                continue
            total = sizes[take - 1]
            offloaded += total
            if layer_nonfit_prefix is not None:
                nonfit += layer_nonfit_prefix[index][take - 1]
            transfer_sizes.append(total)
            continue

        total = 0
        for candidate in by_layer[layer_index]:
            if total >= target:
                break
            size = candidate.tensor.size_bytes
            total += size
            offloaded += size
            if not candidate.fits:
                nonfit += size
        if total > 0:
            transfer_sizes.append(total)
    peak = _strict_peak_from_transfer_sizes(
        transfer_sizes,
        total_model=total_model,
        offloaded=float(offloaded),
        n_blocks=search_blocks,
    )
    return peak, float(offloaded), float(nonfit)


class BudgetFillGreedyStrategy:
    """Ranked-prefix BudgetFill: scan fit-then-size prefixes under a hard peak budget.

    Args:
        n_blocks: Pipeline block count. ``0`` skips block planning (peak is
            residency only). ``1`` is rejected; use ``0`` or ``>= 2``.
        threshold_mb: Ignore tensors at or below this size (MiB).
        scale: Multiplier on previous-layer duration for transfer-fit ranking.
        assignment_strategy: Trap-to-block assignment. Defaults to
            :class:`~flextensor.strategy.assignment.StrictRoundRobinAssignment`.
            Pass :class:`~flextensor.strategy.assignment.OptimizedRoundRobinAssignment`
            to search block counts / patterns like GlobalOffload.
    """

    def __init__(
        self,
        n_blocks: int = 0,
        *,
        threshold_mb: float = 1.0,
        scale: float = 1.0,
        assignment_strategy: AssignmentStrategy | None = None,
    ) -> None:
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if n_blocks == 1:
            raise ValueError("n_blocks=1 is unsupported for BudgetFillGreedyStrategy; use 0 (no blocks) or >= 2")
        validate_memory_params(scale)
        if threshold_mb < 0:
            raise ValueError(f"threshold_mb must be >= 0, got {threshold_mb}")

        self.n_blocks = n_blocks
        self.threshold_mb = threshold_mb
        self.scale = scale
        self.assignment_strategy: AssignmentStrategy = (
            assignment_strategy if assignment_strategy is not None else StrictRoundRobinAssignment()
        )

    def compute(  # noqa: C901
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Select a minimal ranked offload prefix that meets ``max_gpu_mem_bytes``."""
        if max_gpu_mem_bytes is None:
            raise StrategyComputeError("BudgetFillGreedyStrategy requires max_gpu_mem_bytes (hard GPU memory budget)")
        if len(layer_stats) < 2:
            raise StrategyComputeError("BudgetFillGreedyStrategy requires at least two layers for pipelined offload")

        interpolator: MemoryTransferInterpolator | None = None
        if memory_stats:
            interpolator = MemoryTransferInterpolator(memory_stats)

        candidates = _rank_candidates(
            layer_stats,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            interpolator=interpolator,
        )

        empty = StrategyResult(strategy_map={}, block_data=None)
        if _peak_memory(empty, layer_stats) <= max_gpu_mem_bytes:
            return empty

        if not candidates:
            warnings.warn(
                "BudgetFillGreedyStrategy could not meet max_gpu_mem_bytes: no eligible tensors "
                f"above threshold_mb={self.threshold_mb}",
                UserWarning,
                stacklevel=2,
            )
            return empty

        assignment = self.assignment_strategy
        n_blocks = self.n_blocks
        total_model = float(sum(tensor.size_bytes for layer in layer_stats for tensor in layer.tensors))
        budget = float(max_gpu_mem_bytes)
        search_blocks = _search_block_count(n_blocks)
        n_cand = len(candidates)

        def result_for_prefix(prefix_len: int) -> StrategyResult:
            return _result_for_selection(
                layer_stats,
                candidates[:prefix_len],
                n_blocks=n_blocks,
                assignment_strategy=assignment,
            )

        # Peak is non-monotonic in prefix length — scan, do not treat full offload
        # as an infeasibility proof.
        prefix_len = _best_ranked_prefix(
            candidates,
            total_model=total_model,
            budget=budget,
            search_blocks=search_blocks,
        )
        result = result_for_prefix(prefix_len)
        if _peak_memory(result, layer_stats) <= max_gpu_mem_bytes:
            return result

        # Cheap Strict peak can disagree with the configured assignment. Rescan
        # with the real assignment; full scan when small, else expand around the
        # Strict-chosen prefix so a nearby feasible plan is not missed.
        best_k = prefix_len
        best_peak = _peak_memory(result, layer_stats)
        best_feasible_k: int | None = None
        best_feasible_off = 0

        def _consider(k: int) -> None:
            nonlocal best_k, best_peak, best_feasible_k, best_feasible_off
            plan = result_for_prefix(k)
            peak = _peak_memory(plan, layer_stats)
            off = sum(t.size_bytes for ts in plan.strategy_map.values() for t in ts)
            if peak < best_peak:
                best_peak = peak
                best_k = k
            if peak <= max_gpu_mem_bytes and (best_feasible_k is None or off < best_feasible_off):
                best_feasible_k = k
                best_feasible_off = off

        if n_cand <= 64:
            for k in range(n_cand + 1):
                _consider(k)
        else:
            considered: set[int] = set()
            for radius in range(n_cand + 1):
                ks = (prefix_len,) if radius == 0 else (prefix_len - radius, prefix_len + radius)
                for k in ks:
                    if 0 <= k <= n_cand and k not in considered:
                        considered.add(k)
                        _consider(k)
                if best_feasible_k is not None:
                    break
                if len(considered) >= n_cand + 1:
                    break

        result = result_for_prefix(best_feasible_k if best_feasible_k is not None else best_k)

        if _peak_memory(result, layer_stats) > max_gpu_mem_bytes:
            warnings.warn(
                "BudgetFillGreedyStrategy could not meet max_gpu_mem_bytes "
                f"(peak={_peak_memory(result, layer_stats)}, budget={max_gpu_mem_bytes})",
                UserWarning,
                stacklevel=2,
            )
        return result


class BudgetFillTensorDEStrategy:
    """Per-tensor differential-evolution BudgetFill (slow; opt-in fallback).

    Same hard/soft goals as :class:`BudgetFillGreedyStrategy`, but searches
    arbitrary subsets via differential evolution
    (``scipy.optimize.differential_evolution``).

    Needed when the best feasible set is **not** a within-layer fit→size prefix
    (e.g. take a mid-ranked tensor without its higher-ranked neighbors). Prefer
    :class:`BudgetFillLayerDEStrategy` for speed; enable via
    :class:`BudgetFillStrategy` ``enable_tensor_de=True`` or run alone.
    Adaptive leaves this off and uses :class:`GlobalTensorSelectionStrategy`
    for expensive tensor-level search under ``extra_optimization``.

    Seeds the population with the greedy ranked-prefix plan.
    """

    def __init__(
        self,
        n_blocks: int = 0,
        *,
        threshold_mb: float = 1.0,
        scale: float = 1.0,
        assignment_strategy: AssignmentStrategy | None = None,
        pop_size: int = 30,
        epoch: int = 80,
        max_early_stop: int = 25,
        seed: int | None = None,
    ) -> None:
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if n_blocks == 1:
            raise ValueError("n_blocks=1 is unsupported for BudgetFillTensorDEStrategy; use 0 (no blocks) or >= 2")
        validate_memory_params(scale)
        if threshold_mb < 0:
            raise ValueError(f"threshold_mb must be >= 0, got {threshold_mb}")
        if pop_size < 5:
            raise ValueError(f"pop_size must be >= 5, got {pop_size}")
        if epoch < 1:
            raise ValueError(f"epoch must be >= 1, got {epoch}")
        if max_early_stop < 1:
            raise ValueError(f"max_early_stop must be >= 1, got {max_early_stop}")

        self.n_blocks = n_blocks
        self.threshold_mb = threshold_mb
        self.scale = scale
        self.assignment_strategy: AssignmentStrategy = (
            assignment_strategy if assignment_strategy is not None else StrictRoundRobinAssignment()
        )
        self.pop_size = pop_size
        self.epoch = epoch
        self.max_early_stop = max_early_stop
        self.seed = seed

    def compute(  # noqa: C901
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Optimize a binary offload mask under ``max_gpu_mem_bytes`` via DE."""
        import numpy as np
        from scipy.optimize import Bounds, differential_evolution

        from flextensor.strategy.utils import EarlyStopCallback

        if max_gpu_mem_bytes is None:
            raise StrategyComputeError("BudgetFillTensorDEStrategy requires max_gpu_mem_bytes (hard GPU memory budget)")
        if len(layer_stats) < 2:
            raise StrategyComputeError("BudgetFillTensorDEStrategy requires at least two layers for pipelined offload")

        interpolator: MemoryTransferInterpolator | None = None
        if memory_stats:
            interpolator = MemoryTransferInterpolator(memory_stats)

        candidates = _rank_candidates(
            layer_stats,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            interpolator=interpolator,
        )

        empty = StrategyResult(strategy_map={}, block_data=None)
        if _peak_memory(empty, layer_stats) <= max_gpu_mem_bytes:
            return empty

        if not candidates:
            warnings.warn(
                "BudgetFillTensorDEStrategy could not meet max_gpu_mem_bytes: no eligible tensors "
                f"above threshold_mb={self.threshold_mb}",
                UserWarning,
                stacklevel=2,
            )
            return empty

        n_tensors = len(candidates)
        sizes = np.array([c.tensor.size_bytes for c in candidates], dtype=np.float64)
        fits = np.array([1.0 if c.fits else 0.0 for c in candidates], dtype=np.float64)
        total_model = float(sum(tensor.size_bytes for layer in layer_stats for tensor in layer.tensors))
        budget = float(max_gpu_mem_bytes)
        assignment = self.assignment_strategy
        n_blocks = self.n_blocks

        def result_for_mask(mask: np.ndarray) -> StrategyResult:
            selected = [candidates[i] for i in range(n_tensors) if mask[i] > 0.5]
            return _result_for_selection(
                layer_stats,
                selected,
                n_blocks=n_blocks,
                assignment_strategy=assignment,
            )

        def peak_for_mask(mask: np.ndarray) -> float:
            return float(_peak_memory(result_for_mask(mask), layer_stats))

        full_mask = np.ones(n_tensors, dtype=np.float64)

        # Warm-start from greedy ranked-prefix search (peak is non-monotonic in
        # offload amount — do not treat full offload as an infeasibility proof).
        greedy = BudgetFillGreedyStrategy(
            n_blocks=n_blocks,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            assignment_strategy=assignment,
        ).compute(layer_stats, memory_stats, max_gpu_mem_bytes)
        greedy_ids = {t.tensor_id for tensors in greedy.strategy_map.values() for t in tensors}
        greedy_mask = np.array(
            [1.0 if c.tensor.tensor_id in greedy_ids else 0.0 for c in candidates],
            dtype=np.float64,
        )

        # Objective: feasible first, then min offload, then prefer fit (tie-break).
        def objective(x: np.ndarray) -> float:
            mask = (x > 0.5).astype(np.float64)
            peak = peak_for_mask(mask)
            offloaded = float(np.dot(mask, sizes))
            if peak > budget:
                return 1e18 + (peak - budget)
            nonfit = float(np.dot(mask * (1.0 - fits), sizes))
            return _soft_objective(offloaded, nonfit, total_model)

        # Tiny problems: enumerate all 2^n masks (n ≤ 4 → ≤16). Ranked-prefix
        # greedy can miss cheaper non-prefix subsets here, and DE overhead is
        # unnecessary when exhaustive search is trivial.
        if n_tensors <= 4:
            best_mask = greedy_mask.copy()
            best_obj = objective(best_mask)
            for bits in range(1 << n_tensors):
                mask = np.array(
                    [float((bits >> i) & 1) for i in range(n_tensors)],
                    dtype=np.float64,
                )
                obj = objective(mask)
                if obj + 1e-12 < best_obj:
                    best_obj = obj
                    best_mask = mask
            return result_for_mask(best_mask)

        pop_size, epoch, max_early_stop = _auto_de_params(
            n_tensors,
            pop_size=self.pop_size,
            epoch=self.epoch,
            max_early_stop=self.max_early_stop,
            binary=True,
        )

        rng = np.random.default_rng(self.seed)
        population: list[np.ndarray] = [greedy_mask.copy(), full_mask.copy()]
        # Sparse random masks (prefer few offloads — primary soft goal).
        while len(population) < pop_size:
            density = float(rng.uniform(0.05, 0.6))
            population.append((rng.random(n_tensors) < density).astype(np.float64))

        init = np.asarray(population[:pop_size], dtype=np.float64)
        bounds = Bounds([0.0] * n_tensors, [1.0] * n_tensors)
        integrality = [True] * n_tensors
        early_stop = EarlyStopCallback(max_stall=max_early_stop, objective_func=objective)

        de_result = differential_evolution(
            objective,
            bounds=bounds,
            integrality=integrality,
            init=init,
            maxiter=epoch,
            tol=0.05 if n_tensors >= 1_000 else 0.01,
            seed=self.seed if self.seed is not None else 42,
            polish=False,
            # EarlyStopCallback returns bool to halt DE; stubs omit that return type.
            callback=cast("Any", early_stop),
        )

        best = de_result.x
        if best is None or np.any(np.isnan(best)):
            return greedy
        best_mask = (best > 0.5).astype(np.float64)
        # Prefer greedy if DE somehow worsens primary objective while remaining feasible.
        if objective(best_mask) > objective(greedy_mask) + 1e-12:
            return greedy
        return result_for_mask(best_mask)


def _auto_de_params(
    n_vars: int,
    *,
    pop_size: int,
    epoch: int,
    max_early_stop: int,
    binary: bool = False,
) -> tuple[int, int, int]:
    """Scale DE effort with problem size; constructor values are upper caps.

    Small / easy problems get a shallow search (like Global's lighter budgets).
    Huge binary spaces stay intentionally shallow — prefer the layer variant.
    """
    if n_vars <= 0:
        return max(5, pop_size), max(1, epoch), max(1, max_early_stop)

    if binary:
        if n_vars >= 10_000:
            heuristic_pop, heuristic_epoch, heuristic_stall = 10, 15, 6
        elif n_vars >= 1_000:
            heuristic_pop, heuristic_epoch, heuristic_stall = 14, 30, 10
        elif n_vars >= 100:
            heuristic_pop, heuristic_epoch, heuristic_stall = 16, 40, 12
        else:
            heuristic_pop, heuristic_epoch, heuristic_stall = 12, 25, 8
    # Continuous per-layer fractions
    elif n_vars <= 8:
        heuristic_pop, heuristic_epoch, heuristic_stall = 8, 12, 5
    elif n_vars <= 24:
        heuristic_pop, heuristic_epoch, heuristic_stall = 10, 20, 7
    elif n_vars <= 48:
        heuristic_pop, heuristic_epoch, heuristic_stall = 10, 25, 8
    else:
        # DeepSeek-scale (~60 layers): shallow — greedy seed is already strong
        heuristic_pop, heuristic_epoch, heuristic_stall = 10, 20, 8

    return (
        max(5, min(pop_size, heuristic_pop)),
        max(1, min(epoch, heuristic_epoch)),
        max(1, min(max_early_stop, heuristic_stall)),
    )


def _candidates_by_layer(candidates: list[_Candidate]) -> dict[int, list[_Candidate]]:
    """Group candidates by layer; within each layer keep fit-then-size order."""
    by_layer: dict[int, list[_Candidate]] = {}
    for candidate in candidates:
        by_layer.setdefault(candidate.layer_index, []).append(candidate)
    for layer_index, items in by_layer.items():
        items.sort(
            key=lambda c: (
                0 if c.fits else 1,
                -c.tensor.size_bytes,
                c.transfer_ms,
                c.tensor.tensor_id,
            )
        )
        by_layer[layer_index] = items
    return by_layer


def _select_for_layer_fractions(
    by_layer: dict[int, list[_Candidate]],
    layer_indices: list[int],
    max_bytes: list[int],
    fractions: list[float],
) -> list[_Candidate]:
    """Take fit-ranked tensors in each layer up to ``fraction * layer_max`` bytes."""
    selected: list[_Candidate] = []
    for index, layer_index in enumerate(layer_indices):
        target = max(0.0, float(fractions[index])) * float(max_bytes[index])
        total = 0
        for candidate in by_layer[layer_index]:
            if total >= target:
                break
            selected.append(candidate)
            total += candidate.tensor.size_bytes
    return selected


class BudgetFillLayerDEStrategy:
    """Per-layer differential-evolution BudgetFill (one offload fraction per layer).

    Same hard/soft goals as :class:`BudgetFillGreedyStrategy`, with search space
    ~``n_layers``. Within each layer, tensors are taken in fit-then-size order up
    to the chosen byte budget. Used by :class:`BudgetFillStrategy` when
    ``enable_layer_de=True``, or alone for comparison.
    """

    def __init__(
        self,
        n_blocks: int = 0,
        *,
        threshold_mb: float = 1.0,
        scale: float = 1.0,
        assignment_strategy: AssignmentStrategy | None = None,
        pop_size: int = 20,
        epoch: int = 60,
        max_early_stop: int = 20,
        seed: int | None = None,
    ) -> None:
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if n_blocks == 1:
            raise ValueError("n_blocks=1 is unsupported for BudgetFillLayerDEStrategy; use 0 or >= 2")
        validate_memory_params(scale)
        if threshold_mb < 0:
            raise ValueError(f"threshold_mb must be >= 0, got {threshold_mb}")
        if pop_size < 5:
            raise ValueError(f"pop_size must be >= 5, got {pop_size}")
        if epoch < 1:
            raise ValueError(f"epoch must be >= 1, got {epoch}")
        if max_early_stop < 1:
            raise ValueError(f"max_early_stop must be >= 1, got {max_early_stop}")

        self.n_blocks = n_blocks
        self.threshold_mb = threshold_mb
        self.scale = scale
        self.assignment_strategy: AssignmentStrategy = (
            assignment_strategy if assignment_strategy is not None else StrictRoundRobinAssignment()
        )
        self.pop_size = pop_size
        self.epoch = epoch
        self.max_early_stop = max_early_stop
        self.seed = seed

    def compute(  # noqa: C901
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Optimize per-layer offload fractions under ``max_gpu_mem_bytes`` via DE."""
        import numpy as np
        from scipy.optimize import Bounds, differential_evolution

        from flextensor.strategy.utils import EarlyStopCallback

        if max_gpu_mem_bytes is None:
            raise StrategyComputeError("BudgetFillLayerDEStrategy requires max_gpu_mem_bytes (hard GPU memory budget)")
        if len(layer_stats) < 2:
            raise StrategyComputeError("BudgetFillLayerDEStrategy requires at least two layers for pipelined offload")

        interpolator: MemoryTransferInterpolator | None = None
        if memory_stats:
            interpolator = MemoryTransferInterpolator(memory_stats)

        candidates = _rank_candidates(
            layer_stats,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            interpolator=interpolator,
        )

        empty = StrategyResult(strategy_map={}, block_data=None)
        if _peak_memory(empty, layer_stats) <= max_gpu_mem_bytes:
            return empty

        if not candidates:
            warnings.warn(
                "BudgetFillLayerDEStrategy could not meet max_gpu_mem_bytes: no eligible "
                f"tensors above threshold_mb={self.threshold_mb}",
                UserWarning,
                stacklevel=2,
            )
            return empty

        by_layer = _candidates_by_layer(candidates)
        layer_indices = sorted(by_layer.keys())
        max_bytes = [sum(c.tensor.size_bytes for c in by_layer[i]) for i in layer_indices]
        n_vars = len(layer_indices)
        total_model = float(sum(tensor.size_bytes for layer in layer_stats for tensor in layer.tensors))
        budget = float(max_gpu_mem_bytes)
        assignment = self.assignment_strategy
        n_blocks = self.n_blocks
        # Cheap Strict peak during DE; final plan uses configured assignment.
        search_blocks = _search_block_count(n_blocks)

        # Cumsums for O(layers) objective evaluations.
        layer_size_prefix: list[list[int]] = []
        layer_nonfit_prefix: list[list[int]] = []
        for layer_index in layer_indices:
            sizes: list[int] = []
            nonfits: list[int] = []
            size_cum = 0
            nonfit_cum = 0
            for candidate in by_layer[layer_index]:
                size_cum += candidate.tensor.size_bytes
                if not candidate.fits:
                    nonfit_cum += candidate.tensor.size_bytes
                sizes.append(size_cum)
                nonfits.append(nonfit_cum)
            layer_size_prefix.append(sizes)
            layer_nonfit_prefix.append(nonfits)

        def result_for_fractions(
            fractions: np.ndarray,
            *,
            assign: AssignmentStrategy | None = None,
        ) -> StrategyResult:
            selected = _select_for_layer_fractions(
                by_layer,
                layer_indices,
                max_bytes,
                [float(v) for v in fractions],
            )
            return _result_for_selection(
                layer_stats,
                selected,
                n_blocks=n_blocks,
                assignment_strategy=assign if assign is not None else assignment,
            )

        full = np.ones(n_vars, dtype=np.float64)

        # Seed from ranked-prefix search (peak is non-monotonic — full offload is
        # not an infeasibility proof).
        prefix_len = _best_ranked_prefix(
            candidates,
            total_model=total_model,
            budget=budget,
            search_blocks=search_blocks,
        )
        greedy_ids = {c.tensor.tensor_id for c in candidates[:prefix_len]}
        greedy_frac = np.zeros(n_vars, dtype=np.float64)
        for index, layer_index in enumerate(layer_indices):
            off = sum(c.tensor.size_bytes for c in by_layer[layer_index] if c.tensor.tensor_id in greedy_ids)
            greedy_frac[index] = 0.0 if max_bytes[index] == 0 else off / float(max_bytes[index])

        greedy_peak, _, _ = _layer_fraction_metrics(
            by_layer,
            layer_indices,
            max_bytes,
            [float(v) for v in greedy_frac],
            total_model=total_model,
            search_blocks=search_blocks,
            layer_size_prefix=layer_size_prefix,
            layer_nonfit_prefix=layer_nonfit_prefix,
        )
        # Tiny problems, or greedy already pinches the budget: DE won't beat the
        # ranked prefix without raising peak (soft objective has no room).
        near_budget = greedy_peak <= budget and (budget - greedy_peak) <= max(1.0, 0.001 * budget)
        if n_vars <= 2 or near_budget:
            final = result_for_fractions(greedy_frac)
            if _peak_memory(final, layer_stats) <= max_gpu_mem_bytes:
                return final
            return BudgetFillGreedyStrategy(
                n_blocks=n_blocks,
                threshold_mb=self.threshold_mb,
                scale=self.scale,
                assignment_strategy=assignment,
            ).compute(layer_stats, memory_stats, max_gpu_mem_bytes)

        pop_size, epoch, max_early_stop = _auto_de_params(
            n_vars,
            pop_size=self.pop_size,
            epoch=self.epoch,
            max_early_stop=self.max_early_stop,
            binary=False,
        )

        def objective(x: np.ndarray) -> float:
            fracs = [float(v) for v in np.clip(x, 0.0, 1.0)]
            peak, offloaded, nonfit = _layer_fraction_metrics(
                by_layer,
                layer_indices,
                max_bytes,
                fracs,
                total_model=total_model,
                search_blocks=search_blocks,
                layer_size_prefix=layer_size_prefix,
                layer_nonfit_prefix=layer_nonfit_prefix,
            )
            if peak > budget:
                return 1e18 + (peak - budget)
            return _soft_objective(offloaded, nonfit, total_model)

        rng = np.random.default_rng(self.seed)
        population: list[np.ndarray] = [greedy_frac.copy(), full.copy(), np.zeros(n_vars)]
        while len(population) < pop_size:
            population.append(rng.random(n_vars))

        init = np.asarray(population[:pop_size], dtype=np.float64)
        bounds = Bounds([0.0] * n_vars, [1.0] * n_vars)
        early_stop = EarlyStopCallback(max_stall=max_early_stop, objective_func=objective)

        de_result = differential_evolution(
            objective,
            bounds=bounds,
            init=init,
            maxiter=epoch,
            tol=0.05 if n_vars <= 24 else 0.01,
            seed=self.seed if self.seed is not None else 42,
            polish=False,
            # EarlyStopCallback returns bool to halt DE; stubs omit that return type.
            callback=cast("Any", early_stop),
        )

        best = de_result.x
        if best is None or np.any(np.isnan(best)):
            best = greedy_frac
        else:
            best = np.clip(best, 0.0, 1.0)
            if objective(best) > objective(greedy_frac) + 1e-12:
                best = greedy_frac

        # Final plan with the real (possibly Optimized) assignment strategy.
        final = result_for_fractions(best)
        if _peak_memory(final, layer_stats) <= max_gpu_mem_bytes:
            return final
        # Search peak used Strict@2; if configured assignment is worse, fall back
        # to ranked-prefix greedy under the real assignment.
        return BudgetFillGreedyStrategy(
            n_blocks=n_blocks,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            assignment_strategy=assignment,
        ).compute(layer_stats, memory_stats, max_gpu_mem_bytes)


def _offload_bytes(result: StrategyResult) -> int:
    return sum(tensor.size_bytes for tensors in result.strategy_map.values() for tensor in tensors)


def _prefer_budget_fill_result(
    current: StrategyResult,
    candidate: StrategyResult,
    layer_stats: list[LayerStatistics],
    max_gpu_mem_bytes: int,
) -> StrategyResult:
    """Prefer feasible peak, then fewer offloaded bytes."""
    cur_peak = _peak_memory(current, layer_stats)
    cand_peak = _peak_memory(candidate, layer_stats)
    cur_ok = cur_peak <= max_gpu_mem_bytes
    cand_ok = cand_peak <= max_gpu_mem_bytes
    if cand_ok and not cur_ok:
        return candidate
    if cur_ok and not cand_ok:
        return current
    if _offload_bytes(candidate) < _offload_bytes(current):
        return candidate
    return current


class BudgetFillStrategy:
    """BudgetFill facade: greedy, then optional layer / tensor DE solvers.

    Always runs :class:`BudgetFillGreedyStrategy`. Optionally runs
    :class:`BudgetFillLayerDEStrategy` (``enable_layer_de``, default True) and
    :class:`BudgetFillTensorDEStrategy` (``enable_tensor_de``, default True),
    keeping the best feasible plan with the fewest offloaded bytes.
    Pinching the budget does not prove min-offload, so DE is not skipped solely
    for a tight peak — Adaptive constructs BudgetFill with
    ``enable_tensor_de=False`` and uses
    :class:`~flextensor.strategy.global_strategy.GlobalTensorSelectionStrategy`
    under ``extra_optimization`` instead.

    Each solver remains independently usable for A/B comparison.

    Args:
        n_blocks: Pipeline block count (``0`` or ``>= 2``; ``1`` rejected).
        threshold_mb: Ignore tensors at or below this size (MiB).
        scale: Multiplier on previous-layer duration for transfer-fit ranking.
        assignment_strategy: Trap-to-block assignment (default Strict).
        enable_layer_de: Run per-layer DE after greedy.
        enable_tensor_de: Run per-tensor DE after greedy/layer DE (default True;
            Adaptive keeps this False).
        pop_size / epoch / max_early_stop: Caps forwarded to DE solvers.
        seed: RNG seed for DE solvers.
    """

    def __init__(
        self,
        n_blocks: int = 0,
        *,
        threshold_mb: float = 1.0,
        scale: float = 1.0,
        assignment_strategy: AssignmentStrategy | None = None,
        enable_layer_de: bool = True,
        enable_tensor_de: bool = True,
        pop_size: int = 20,
        epoch: int = 60,
        max_early_stop: int = 20,
        seed: int | None = None,
    ) -> None:
        if n_blocks < 0:
            raise ValueError(f"n_blocks must be >= 0, got {n_blocks}")
        if n_blocks == 1:
            raise ValueError("n_blocks=1 is unsupported for BudgetFillStrategy; use 0 (no blocks) or >= 2")
        validate_memory_params(scale)
        if threshold_mb < 0:
            raise ValueError(f"threshold_mb must be >= 0, got {threshold_mb}")

        self.n_blocks = n_blocks
        self.threshold_mb = threshold_mb
        self.scale = scale
        self.assignment_strategy: AssignmentStrategy = (
            assignment_strategy if assignment_strategy is not None else StrictRoundRobinAssignment()
        )
        self.enable_layer_de = enable_layer_de
        self.enable_tensor_de = enable_tensor_de
        self.pop_size = pop_size
        self.epoch = epoch
        self.max_early_stop = max_early_stop
        self.seed = seed
        self._selected_solver_name: str | None = None

    @property
    def selected_solver_name(self) -> str | None:
        """Name of the solver that produced the last :meth:`compute` result."""
        return self._selected_solver_name

    def compute(
        self,
        layer_stats: list[LayerStatistics],
        memory_stats: dict[int, float] | None = None,
        max_gpu_mem_bytes: int | None = None,
    ) -> StrategyResult:
        """Run greedy, then optional scipy solvers; keep the best feasible plan."""
        if max_gpu_mem_bytes is None:
            raise StrategyComputeError("BudgetFillStrategy requires max_gpu_mem_bytes (hard GPU memory budget)")

        greedy = BudgetFillGreedyStrategy(
            n_blocks=self.n_blocks,
            threshold_mb=self.threshold_mb,
            scale=self.scale,
            assignment_strategy=self.assignment_strategy,
        ).compute(layer_stats, memory_stats, max_gpu_mem_bytes)
        best = greedy
        selected = "BudgetFillGreedy"

        if self.enable_layer_de:
            layer_result = BudgetFillLayerDEStrategy(
                n_blocks=self.n_blocks,
                threshold_mb=self.threshold_mb,
                scale=self.scale,
                assignment_strategy=self.assignment_strategy,
                pop_size=self.pop_size,
                epoch=self.epoch,
                max_early_stop=self.max_early_stop,
                seed=self.seed,
            ).compute(layer_stats, memory_stats, max_gpu_mem_bytes)
            preferred = _prefer_budget_fill_result(best, layer_result, layer_stats, max_gpu_mem_bytes)
            if preferred is not best:
                best = preferred
                selected = "BudgetFillLayerDE"

        # Non-prefix fallback: a pinched peak does not prove min offload.
        if self.enable_tensor_de:
            tensor_result = BudgetFillTensorDEStrategy(
                n_blocks=self.n_blocks,
                threshold_mb=self.threshold_mb,
                scale=self.scale,
                assignment_strategy=self.assignment_strategy,
                pop_size=max(self.pop_size, 30),
                epoch=max(self.epoch, 80),
                max_early_stop=max(self.max_early_stop, 25),
                seed=self.seed,
            ).compute(layer_stats, memory_stats, max_gpu_mem_bytes)
            preferred = _prefer_budget_fill_result(best, tensor_result, layer_stats, max_gpu_mem_bytes)
            if preferred is not best:
                best = preferred
                selected = "BudgetFillTensorDE"

        self._selected_solver_name = selected
        return best
