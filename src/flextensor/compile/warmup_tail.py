# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passive warm → measure → done counter used by compiled-offload replan."""

from enum import Enum


class CompiledOffloadTailState(Enum):
    """State of the passive compiled-offload warm → measure → replan tail."""

    IDLE = "idle"
    WARMING = "warming"
    MEASURING = "measuring"
    DONE = "done"
    FAILED = "failed"


class WarmupTail:
    """Count unmeasured warmup forwards, then measured forwards, then finish.

    Used by the compiled replan tail (and similar fixed-warmup patterns).
    """

    def __init__(self, *, warmup_forwards: int, measure_forwards: int) -> None:
        self.warmup_forwards = max(warmup_forwards, 0)
        self.measure_forwards = max(measure_forwards, 0)
        self.state = CompiledOffloadTailState.IDLE
        self.warm_seen = 0
        self.measure_seen = 0
        self.failure: BaseException | None = None

    def reset(self) -> None:
        self.state = CompiledOffloadTailState.IDLE
        self.warm_seen = 0
        self.measure_seen = 0
        self.failure = None

    def arm(self, *, credited_warm: int = 0) -> int:
        """Arm warm→measure; return remaining forwards until finish."""
        credited = min(max(credited_warm, 0), self.warmup_forwards)
        self.warm_seen = credited
        self.measure_seen = 0
        self.failure = None
        remaining_warm = self.warmup_forwards - credited
        if credited >= self.warmup_forwards:
            self.state = CompiledOffloadTailState.MEASURING
        else:
            self.state = CompiledOffloadTailState.WARMING
        return remaining_warm + self.measure_forwards

    def mark_done(self) -> None:
        self.state = CompiledOffloadTailState.DONE

    def mark_failed(self, exc: BaseException) -> None:
        self.state = CompiledOffloadTailState.FAILED
        self.failure = exc
