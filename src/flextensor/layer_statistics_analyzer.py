# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer statistics analyzer for checking measurement consistency.

This module provides tools for analyzing layer duration measurements and
detecting potential issues with profiling quality.
"""

import logging

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

from flextensor.collectors import IterativeLayerStatisticsCollector, LayerStatistics
from flextensor.instrumentation.decorator import instrumentable

LOGGER = logging.getLogger(__name__)


class LayerDurationStatistics(BaseModel):
    """Comprehensive duration statistics for a single layer."""

    model_config = ConfigDict(frozen=True)

    label: str
    min_ms: float
    max_ms: float
    median_ms: float
    avg_ms: float
    std_ms: float
    count: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coefficient_of_variation(self) -> float:
        """Coefficient of variation (CV = std/mean). Higher values indicate more variability."""
        if self.avg_ms == 0:
            return 0.0
        return self.std_ms / self.avg_ms


@instrumentable
class LayerStatisticsAnalyzer:
    """Analyzer that computes comprehensive statistics from layer measurements.

    This class is decorated with @instrumentable to capture all init arguments
    for later inspection and debugging.

    Args:
        layer_statistics_collector: The collector containing raw duration measurements.
    """

    def __init__(self, layer_statistics_collector: IterativeLayerStatisticsCollector) -> None:
        self.statistics: dict[str, LayerDurationStatistics] = {}
        self._compute_statistics(layer_statistics_collector)

    def _compute_statistics(self, collector: IterativeLayerStatisticsCollector) -> None:
        """Compute min, max, median, avg, std dev of duration for each layer."""
        for label, durations in collector.duration_measurements.items():
            if not durations:
                continue

            durations_array = np.array(durations)
            self.statistics[label] = LayerDurationStatistics(
                label=label,
                min_ms=float(np.min(durations_array)),
                max_ms=float(np.max(durations_array)),
                median_ms=float(np.median(durations_array)),
                avg_ms=float(np.mean(durations_array)),
                std_ms=float(np.std(durations_array)),
                count=len(durations),
            )

    def get_statistics(self) -> list[LayerDurationStatistics]:
        """Get all computed layer statistics as a list."""
        return list(self.statistics.values())

    def get_layer_statistics(self, label: str) -> LayerDurationStatistics | None:
        """Get statistics for a specific layer by label."""
        return self.statistics.get(label)

    def format_statistics_table(self) -> str:
        """Format all layer statistics as a table string."""
        if not self.statistics:
            return "No layer statistics available."

        header = "=" * 110
        col_header = (
            f"{'Layer':<40} {'Min':>10} {'Max':>10} {'Median':>10} {'Avg':>10} {'Std':>10} {'CV':>8} {'Count':>6}"
        )

        lines = [
            header,
            "Layer Duration Statistics (ms)",
            header,
            col_header,
            "-" * 110,
        ]

        for s in self.statistics.values():
            label = s.label[:38] + ".." if len(s.label) > 40 else s.label
            row = (
                f"{label:<40} {s.min_ms:>10.3f} {s.max_ms:>10.3f} {s.median_ms:>10.3f} "
                f"{s.avg_ms:>10.3f} {s.std_ms:>10.3f} {s.coefficient_of_variation:>7.1%} {s.count:>6}"
            )
            lines.append(row)

        lines.append(header)
        return "\n".join(lines)

    def check_measurement_consistency(
        self,
        min_samples: int = 3,
        cv_threshold: float = 0.25,
        mean_median_ratio_outlier_skip: float = 1.2,
    ) -> bool:
        """Check if duration measurements are consistent and warn if not.

        This method checks for two potential issues:
        1. Too few samples (less than min_samples) - makes statistics unreliable
        2. High coefficient of variation (CV > cv_threshold) - indicates inconsistent measurements

        When the pipeline uses median (not mean) for timing decisions, high CV driven by a few
        outliers does not make the median unreliable. If mean >= median * mean_median_ratio_outlier_skip,
        we treat the layer as outlier-driven and do not flag high CV (median is still reliable).
        This also covers short layers (e.g. embed) where a few slow samples inflate mean and CV.

        Args:
            min_samples: Minimum number of samples recommended for reliable statistics (default: 3)
            cv_threshold: Maximum acceptable coefficient of variation (default: 0.25 = 25%)
            mean_median_ratio_outlier_skip: Skip CV warning when mean/median >= this ratio (default: 1.2),
                i.e. when high CV is likely from outliers and median is still reliable. Set to 0 to disable.

        Returns:
            True if measurements are consistent, False if warnings were issued.
        """
        is_consistent = True
        low_sample_layers: list[tuple[str, int]] = []
        high_variation_layers: list[tuple[str, float, int]] = []

        for stats in self.statistics.values():
            if stats.count < min_samples:
                low_sample_layers.append((stats.label, stats.count))
                is_consistent = False

            if stats.coefficient_of_variation <= cv_threshold:
                continue
            # Pipeline uses median; high CV from outliers (mean >> median) does not make median unreliable
            if (
                mean_median_ratio_outlier_skip > 0
                and stats.median_ms > 0
                and stats.avg_ms >= stats.median_ms * mean_median_ratio_outlier_skip
            ):
                continue
            high_variation_layers.append((stats.label, stats.coefficient_of_variation, stats.count))
            is_consistent = False

        if low_sample_layers:
            labels = ", ".join(f"'{label}' ({count} samples)" for label, count in low_sample_layers[:5])
            suffix = f" and {len(low_sample_layers) - 5} more..." if len(low_sample_layers) > 5 else ""
            LOGGER.warning(
                "Low sample count detected for %d layer(s): %s%s. "
                "Consider increasing profile iterations (recommended: >=%d) for more reliable profiling.",
                len(low_sample_layers),
                labels,
                suffix,
                min_samples,
            )

        if high_variation_layers:
            labels = ", ".join(f"'{label}' (CV={cv:.1%}, n={count})" for label, cv, count in high_variation_layers[:5])
            suffix = f" and {len(high_variation_layers) - 5} more..." if len(high_variation_layers) > 5 else ""
            LOGGER.warning(
                "High duration variability detected for %d layer(s): %s%s. "
                "This may indicate inconsistent execution times. Consider increasing profile iterations "
                "or checking for external interference (other GPU workloads, thermal throttling).",
                len(high_variation_layers),
                labels,
                suffix,
            )

        return is_consistent


def format_effective_layer_duration_table(layer_stats: list[LayerStatistics]) -> str:
    """Format effective saved compute durations without implying raw samples exist."""
    header = "=" * 64
    lines = [
        header,
        "Layer Duration Statistics (ms)",
        header,
        f"{'Layer':<48} {'Compute':>12}",
        "-" * 64,
    ]
    for stats in layer_stats:
        label = stats.label[:46] + ".." if len(stats.label) > 48 else stats.label
        lines.append(f"{label:<48} {stats.duration:>12.3f}")
    lines.append(header)
    return "\n".join(lines)


class UntimedTrapsReport:
    """Diagnostic: traps with tensor IDs recorded but no duration samples.

    Typical causes: the trap only fires at decode time (e.g. vLLM's
    ``logits_processor``), or only while profiling was suspended.
    Such labels are dropped from strategy input (which requires a real
    duration per layer).
    """

    def __init__(self, layer_statistics_collector: IterativeLayerStatisticsCollector) -> None:
        self.labels: list[str] = [
            label
            for label in layer_statistics_collector.tensor_measurements
            if not layer_statistics_collector.duration_measurements.get(label)
        ]

    def is_empty(self) -> bool:
        return not self.labels

    def format(self) -> str:
        """Render a single-line report. Empty string when nothing to report."""
        if not self.labels:
            return ""
        return f"Traps with no duration samples (dropped from strategy input): {', '.join(self.labels)}"


def report_profiling_quality(
    layer_statistics_collector: IterativeLayerStatisticsCollector,
) -> LayerStatisticsAnalyzer:
    """Surface profiling-quality signals after collection completes.

    * Untimed-trap report — emitted at WARNING on this module's logger
      whenever any label has tensor IDs but no duration samples. Such
      labels are silently dropped from strategy input, so users need
      to know which layers were excluded.

    The per-issue warnings emitted by ``check_measurement_consistency``
    itself (low samples, high CV) fire from this module unconditionally
    as part of the consistency check.

    Returns the analyzer so ``TensorManager`` can emit all inference
    diagnostics together after the effective strategy has been computed.

    Anchored by ``tests/unit/test_tensor_manager_diagnostics_emission.py``.
    """
    analyzer = LayerStatisticsAnalyzer(layer_statistics_collector)
    analyzer.check_measurement_consistency()

    untimed_traps = UntimedTrapsReport(layer_statistics_collector)
    if not untimed_traps.is_empty():
        LOGGER.warning("%s", untimed_traps.format())

    return analyzer
