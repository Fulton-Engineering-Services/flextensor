# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer window calculators for pipelined weight offloading.

This module provides strategies for computing the effective transfer window
per trap in a pipelined offloading scheme.  The transfer window determines
how much compute time is available for transferring a trap's weights from
CPU to GPU.

Note: code identifiers use "layer" (e.g. ``layer_durations``) for historical
reasons — a future refactor will rename them to "trap".

Two implementations are provided:

* :class:`SingleLayerWindow` - uses only the immediately preceding trap's
  compute duration (conservative, original behaviour).
* :class:`GapAwareWindow` - sums backward through consecutive traps that
  have no competing transfer, exploiting "gap traps" (traps with no
  offloaded weights) to increase the effective transfer budget.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TransferWindowCalculator(Protocol):
    """Protocol for computing effective transfer windows per trap.

    In the pipelined offload model, the transfer for trap *i* starts during
    an earlier trap's compute.  How much earlier depends on the strategy:

    * ``SingleLayerWindow`` assumes transfer happens during layer *i-1* only.
    * ``GapAwareWindow`` extends the window backward through consecutive
      layers that carry no transfer of their own.
    """

    def compute_window(
        self,
        layer_idx: int,
        layer_offload_sizes: np.ndarray,
        layer_durations: np.ndarray,
    ) -> float:
        """Compute the effective transfer window for a given layer.

        Args:
            layer_idx: Index of the layer whose transfer window to compute.
            layer_offload_sizes: Per-layer offload sizes in bytes.  For
                pre-optimisation checks this reflects permanent (known)
                offload; during optimisation it reflects the candidate
                solution's offload pattern.
            layer_durations: Per-layer compute durations in milliseconds.

        Returns:
            Effective transfer duration in milliseconds.
        """
        ...


class SingleLayerWindow:
    """Transfer window limited to the immediately preceding layer's compute duration.

    This is the conservative (original) behaviour: the transfer for layer *i*
    must complete within layer *i-1*'s compute time.
    """

    def compute_window(
        self,
        layer_idx: int,
        layer_offload_sizes: np.ndarray,
        layer_durations: np.ndarray,
    ) -> float:
        """Return ``layer_durations[layer_idx - 1]``."""
        if layer_idx <= 0:
            return 0.0
        return float(layer_durations[layer_idx - 1])


class GapAwareWindow:
    """Transfer window that exploits gap layers for a larger budget.

    Sums backward from layer *i-1* through consecutive layers whose
    *next* layer has no offloaded tensors (i.e. no competing transfer).
    This captures both **permanent gaps** (layers with no offloadable
    tensors at all) and **dynamic gaps** (layers the optimiser chose
    not to offload in the current candidate solution).

    A layer *j* is considered "busy" (stops the backward scan) when
    ``layer_offload_sizes[j + 1] > 0`` — meaning that during layer *j*'s
    compute, the transfer stream is occupied moving layer *j+1*'s data.
    """

    def compute_window(
        self,
        layer_idx: int,
        layer_offload_sizes: np.ndarray,
        layer_durations: np.ndarray,
    ) -> float:
        """Return the sum of durations of consecutive free layers before *layer_idx*."""
        if layer_idx <= 0:
            return 0.0
        window = float(layer_durations[layer_idx - 1])
        for j in range(layer_idx - 2, -1, -1):
            if layer_offload_sizes[j + 1] > 0.0:
                break
            window += float(layer_durations[j])
        return window
