<!--
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Strategies

Offloading strategies determine which model weights to move from GPU to CPU memory.
Pass a strategy instance to [`OffloadConfig(load_strategy=...)`](configuration.md#flextensor.config.OffloadConfig).

## Strategy Protocol

::: flextensor.strategy.protocol.Strategy

## Result Types

::: flextensor.strategy.protocol.StrategyResult

::: flextensor.strategy.protocol.BlockStrategyData

## Knapsack Strategies

::: flextensor.strategy.knapsack.KnapsackStrategy

::: flextensor.strategy.knapsack.KnapsackBlockStrategy

::: flextensor.strategy.knapsack.AdaptiveKnapsackStrategy

## Simple Strategies

::: flextensor.strategy.simple.GreedyStrategy

::: flextensor.strategy.simple.NthLayerStrategy

## Global Strategies

::: flextensor.strategy.global_strategy.GlobalOffloadStrategy

::: flextensor.strategy.global_strategy.GlobalTensorSelectionStrategy

!!! note "Optimizer parameter values"
    `GlobalTensorSelectionStrategy` accepts an `optimizer` parameter with two accepted values: `"DE"` (differential evolution via `scipy.optimize.differential_evolution`) and `"SA"` (dual annealing via `scipy.optimize.dual_annealing`). `"DE"` is recommended for most cases as it natively handles binary variables. `"SA"` treats variables as continuous and is less efficient for binary decision problems; it may be useful for small problems with few tensors.

## Adaptive Strategy

::: flextensor.strategy.adaptive.AdaptiveStrategy

## Assignment Strategies

::: flextensor.strategy.assignment.AssignmentStrategy

::: flextensor.strategy.assignment.StrictRoundRobinAssignment

::: flextensor.strategy.assignment.OptimizedRoundRobinAssignment

## Transfer Windows

::: flextensor.strategy.transfer_window.TransferWindowCalculator

::: flextensor.strategy.transfer_window.SingleLayerWindow

::: flextensor.strategy.transfer_window.GapAwareWindow

## Evaluation Utilities

::: flextensor.strategy.protocol.StrategyComputeError

::: flextensor.strategy.evaluation.StrategyScore

::: flextensor.strategy.evaluation.evaluate_strategy_result

::: flextensor.strategy.utils.strategy_has_transfer_gaps
