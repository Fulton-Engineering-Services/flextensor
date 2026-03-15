# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer-to-block assignment strategies for pipelined tensor offloading.

This module provides strategies for assigning layers to memory blocks
in block-based offloading schemes.
"""

from typing import Protocol, runtime_checkable

# =============================================================================
# Assignment Strategies
# =============================================================================


@runtime_checkable
class AssignmentStrategy(Protocol):
    """Protocol for layer-to-block assignment strategies."""

    def compute(self, layer_sizes: list[int], n_blocks: int) -> list[int]:
        """
            Compute layer-to-block assignment.

        Args:
                layer_sizes: Size of each layer in bytes.
                n_blocks: Number of available blocks.

        Returns:
                List of block indices, one per layer.
        """
        ...


class StrictRoundRobinAssignment:
    """
    Strict round-robin assignment.

    Assigns layers to blocks using strict pattern: layer_i -> block (i % n_blocks).
    Always uses all available blocks in a fixed cyclic pattern (0,1,2,3,0,1,2,3,...).
    Does not optimize for memory - all blocks sized to fit the largest layer.
    """

    def compute(self, layer_sizes: list[int], n_blocks: int) -> list[int]:
        """Compute simple cyclic assignment."""
        if n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")
        return [i % n_blocks for i in range(len(layer_sizes))]


class OptimizedRoundRobinAssignment:
    """
    Optimized round-robin assignment that minimizes memory while satisfying constraints.

    Features:
    - Tries different numbers of blocks (min_blocks to max_blocks)
    - Uses exhaustive search for small problems, greedy for larger ones
    - Balances memory efficiency with reuse distance (distribution_weight)

    Args:
        min_blocks: Minimum number of blocks to use.
        max_blocks: Maximum number of blocks to use.
        distribution_weight: Weight for reuse distance vs memory (0.0-1.0).
            0.0 = pure memory optimization, 1.0 = pure distribution optimization.
    """

    def __init__(
        self,
        min_blocks: int = 3,  # 3+ blocks to avoid cyclic violations; 2 is enough if cyclic not required
        max_blocks: int = 4,
        distribution_weight: float = 0.5,
    ):
        if min_blocks < 2:
            raise ValueError("min_blocks must be >= 2")
        if max_blocks < min_blocks:
            raise ValueError(f"max_blocks ({max_blocks}) must be >= min_blocks ({min_blocks})")
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.distribution_weight = distribution_weight

    def compute(self, layer_sizes: list[int], n_blocks: int) -> list[int]:
        """Compute optimized assignment within block count range."""
        return self._find_optimal_assignment_with_range(layer_sizes, n_blocks, self.min_blocks, self.max_blocks)

    def _find_optimal_assignment_with_range(  # noqa: C901
        self,
        layer_sizes: list[int],
        n_blocks: int,
        min_blocks: int,
        max_blocks: int,
    ) -> list[int]:
        """Find the optimal layer-to-block assignment within a block count range."""
        if n_blocks <= 0:
            raise ValueError("n_blocks must be > 0")

        num_layers = len(layer_sizes)

        # Adjust max_blocks based on number of layers and available blocks
        effective_max_blocks = min(max_blocks, num_layers, n_blocks)
        effective_min_blocks = min(min_blocks, effective_max_blocks)

        best_assignment: list[int] = []
        best_score = float("-inf")

        # Normalization factor for memory
        max_possible_memory = float(sum(layer_sizes))

        def calculate_reuse_distance(assignment: list[int], target_blocks: int) -> float:
            """Calculate average reuse distance score."""
            if len(assignment) <= target_blocks:
                return 1.0

            last_use: dict[int, int] = {}
            distances: list[int] = []

            for i, block in enumerate(assignment):
                if block in last_use:
                    distances.append(i - last_use[block])
                last_use[block] = i

            if not distances:
                return 1.0

            avg_distance = sum(distances) / len(distances)
            return min(avg_distance / target_blocks, 1.0)

        # Try each possible number of blocks in the range
        for target_blocks in range(effective_min_blocks, effective_max_blocks + 1):
            assignment = self._find_assignment_for_block_count(layer_sizes, n_blocks, target_blocks)

            if not assignment:
                continue

            # Calculate combined score
            total_mem = self._calculate_assignment_memory(layer_sizes, assignment, n_blocks)
            memory_score = 1.0 - (total_mem / max_possible_memory)
            dist_score = calculate_reuse_distance(assignment, target_blocks)

            mem_weight = 1.0 - self.distribution_weight
            combined_score = mem_weight * memory_score + self.distribution_weight * dist_score

            if combined_score > best_score:
                best_score = combined_score
                best_assignment = assignment

        # Fallback to simple round-robin if nothing found
        if not best_assignment:
            best_assignment = [i % effective_min_blocks for i in range(num_layers)]

        return best_assignment

    def _find_assignment_for_block_count(
        self,
        layer_sizes: list[int],
        n_blocks: int,
        target_blocks: int,
    ) -> list[int]:
        """Find optimal assignment using exactly target_blocks blocks."""
        num_layers = len(layer_sizes)

        if target_blocks > num_layers:
            return []  # Can't use more blocks than layers

        # For small problems, use exhaustive search
        if num_layers <= 12 and target_blocks <= 4:
            return self._find_assignment_exhaustive(layer_sizes, n_blocks, target_blocks)
        else:
            return self._find_assignment_greedy(layer_sizes, n_blocks, target_blocks)

    def _find_assignment_exhaustive(  # noqa: C901
        self,
        layer_sizes: list[int],
        n_blocks: int,
        target_blocks: int,
    ) -> list[int]:
        """Find optimal assignment using exhaustive search."""
        num_layers = len(layer_sizes)
        best_assignment: list[int] = []
        best_score = float("-inf")

        max_possible_memory = float(sum(layer_sizes))

        def uses_exactly_n_blocks(assignment: list[int], target: int) -> bool:
            return len(set(assignment)) == target

        def calculate_reuse_distance_score(assignment: list[int]) -> float:
            if len(assignment) <= target_blocks:
                return 1.0

            last_use: dict[int, int] = {}
            distances: list[int] = []

            for i, block in enumerate(assignment):
                if block in last_use:
                    distances.append(i - last_use[block])
                last_use[block] = i

            if not distances:
                return 1.0

            avg_distance = sum(distances) / len(distances)
            ideal_distance = target_blocks
            return min(avg_distance / ideal_distance, 1.0)

        def calculate_combined_score(assignment: list[int]) -> float:
            memory = self._calculate_assignment_memory(layer_sizes, assignment, n_blocks)
            memory_score = 1.0 - (memory / max_possible_memory)
            dist_score = calculate_reuse_distance_score(assignment)
            mem_weight = 1.0 - self.distribution_weight
            return mem_weight * memory_score + self.distribution_weight * dist_score

        def generate_valid_assignments(current: list[int], depth: int) -> None:
            nonlocal best_assignment, best_score

            if depth == num_layers:
                if uses_exactly_n_blocks(current, target_blocks):
                    score = calculate_combined_score(current)
                    if score > best_score:
                        best_score = score
                        best_assignment = current.copy()
                return

            blocks_used = len(set(current)) if current else 0
            remaining_layers = num_layers - depth

            if blocks_used > target_blocks:
                return

            max_possible_blocks = blocks_used + remaining_layers
            if max_possible_blocks < target_blocks:
                return

            prev_block = current[-1] if current else -1
            for block_idx in range(target_blocks):
                if block_idx != prev_block:
                    current.append(block_idx)
                    generate_valid_assignments(current, depth + 1)
                    current.pop()

        generate_valid_assignments([], 0)

        # Verify no cyclic violation (last block != first block) for repeated inference
        if best_assignment and len(best_assignment) >= 2 and best_assignment[-1] == best_assignment[0]:
            # Cyclic violation - try to find a better assignment
            # Fall back to strict round-robin which guarantees no cyclic violations with 3+ blocks
            best_assignment = [i % target_blocks for i in range(num_layers)]

        return best_assignment

    def _find_assignment_greedy(
        self,
        layer_sizes: list[int],
        n_blocks: int,
        target_blocks: int,
    ) -> list[int]:
        """Find good assignment using greedy heuristic."""
        num_layers = len(layer_sizes)
        assignment: list[int] = []
        block_maxes = [0.0] * n_blocks
        blocks_used: set[int] = set()

        for layer_idx in range(num_layers):
            layer_size = float(layer_sizes[layer_idx])
            prev_block = assignment[-1] if assignment else -1

            available_blocks = [b for b in range(target_blocks) if b != prev_block]
            unused_blocks = [b for b in available_blocks if b not in blocks_used]

            candidates = unused_blocks if unused_blocks and len(blocks_used) < target_blocks else available_blocks

            if not candidates:
                candidates = [b for b in range(target_blocks) if b != prev_block]

            best_block = candidates[0]
            best_increase = float("inf")

            for block_idx in candidates:
                current_max = block_maxes[block_idx]
                new_max = max(current_max, layer_size)
                increase = new_max - current_max

                if increase < best_increase:
                    best_increase = increase
                    best_block = block_idx

            assignment.append(best_block)
            block_maxes[best_block] = max(block_maxes[best_block], layer_size)
            blocks_used.add(best_block)

        # Check for cyclic violation (last block == first block) and fix if needed
        if len(assignment) >= 2 and assignment[-1] == assignment[0]:
            # Try to reassign the last layer to a different block
            last_layer_size = float(layer_sizes[-1])
            prev_block = assignment[-2] if len(assignment) >= 2 else -1
            first_block = assignment[0]

            # Find a block that's not the previous layer's block and not the first layer's block
            alternative_blocks = [b for b in range(target_blocks) if b != prev_block and b != first_block]
            if alternative_blocks:
                # Choose the block that minimizes memory increase
                best_alt = alternative_blocks[0]
                best_increase = float("inf")
                for block_idx in alternative_blocks:
                    current_max = block_maxes[block_idx]
                    new_max = max(current_max, last_layer_size)
                    increase = new_max - current_max
                    if increase < best_increase:
                        best_increase = increase
                        best_alt = block_idx
                assignment[-1] = best_alt

        return assignment

    def _calculate_assignment_memory(
        self,
        layer_sizes: list[int],
        assignment: list[int],
        n_blocks: int,
    ) -> float:
        """Calculate total block memory for an assignment."""
        block_maxes = [0.0] * n_blocks
        for layer_idx, block_idx in enumerate(assignment):
            block_maxes[block_idx] = max(block_maxes[block_idx], float(layer_sizes[layer_idx]))
        return sum(block_maxes)
