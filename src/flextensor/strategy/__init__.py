# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlexTensor strategy package.

This package provides offloading strategies for determining which tensors
to offload from GPU to CPU memory during inference.
"""

# Adaptive strategy
from .adaptive import AdaptiveStrategy

# Assignment strategies
from .assignment import (
    AssignmentStrategy,
    OptimizedRoundRobinAssignment,
    StrictRoundRobinAssignment,
)

# Budget-first strategy
from .budget_fill import (
    BudgetFillGreedyStrategy,
    BudgetFillLayerDEStrategy,
    BudgetFillStrategy,
    BudgetFillTensorDEStrategy,
)

# Evaluation utilities
from .evaluation import StrategyScore, evaluate_strategy_result

# Global optimization strategies
from .global_strategy import (
    GlobalOffloadStrategy,
    GlobalTensorSelectionStrategy,
)

# Knapsack strategies
from .knapsack import (
    AdaptiveKnapsackStrategy,
    KnapsackBlockStrategy,
    KnapsackStrategy,
)

# Protocol, result types, and exceptions
from .protocol import (
    BlockStrategyData,
    Strategy,
    StrategyComputeError,
    StrategyResult,
)

# Simple strategies
from .simple import (
    GreedyStrategy,
    NthLayerStrategy,
)

# Transfer window calculators
from .transfer_window import (
    GapAwareWindow,
    SingleLayerWindow,
    TransferWindowCalculator,
)

# Utility functions
from .utils import strategy_has_transfer_gaps

__all__ = [  # noqa: RUF022
    # Protocol, result types, and exceptions
    "Strategy",
    "StrategyResult",
    "StrategyComputeError",
    "BlockStrategyData",
    # Evaluation
    "StrategyScore",
    "evaluate_strategy_result",
    # Assignment strategies
    "AssignmentStrategy",
    "StrictRoundRobinAssignment",
    "OptimizedRoundRobinAssignment",
    # Adaptive strategy
    "AdaptiveStrategy",
    # Knapsack strategies
    "KnapsackStrategy",
    "KnapsackBlockStrategy",
    "AdaptiveKnapsackStrategy",
    # Budget-first strategy
    "BudgetFillStrategy",
    "BudgetFillGreedyStrategy",
    "BudgetFillTensorDEStrategy",
    "BudgetFillLayerDEStrategy",
    # Simple strategies
    "GreedyStrategy",
    "NthLayerStrategy",
    # Global strategies
    "GlobalOffloadStrategy",
    "GlobalTensorSelectionStrategy",
    # Transfer window calculators
    "TransferWindowCalculator",
    "SingleLayerWindow",
    "GapAwareWindow",
    # Utility functions
    "strategy_has_transfer_gaps",
]
