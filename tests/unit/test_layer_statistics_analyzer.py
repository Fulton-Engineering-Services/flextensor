# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LayerStatisticsAnalyzer and LayerDurationStatistics.

This test suite validates:
1. LayerDurationStatistics model behavior and computed properties
2. LayerStatisticsAnalyzer statistics computation
3. Measurement consistency checking and warning generation
4. Statistics table formatting
"""

import logging

import pytest
from pydantic import ValidationError

from flextensor.collectors import IterativeLayerStatisticsCollector
from flextensor.layer_statistics_analyzer import LayerDurationStatistics, LayerStatisticsAnalyzer


class TestLayerDurationStatistics:
    """Test LayerDurationStatistics Pydantic model."""

    def test_basic_creation(self):
        """Test that LayerDurationStatistics can be created with valid values."""
        stats = LayerDurationStatistics(
            label="test_layer",
            min_ms=10.0,
            max_ms=20.0,
            median_ms=15.0,
            avg_ms=15.0,
            std_ms=3.0,
            count=5,
        )
        assert stats.label == "test_layer"
        assert stats.min_ms == 10.0
        assert stats.max_ms == 20.0
        assert stats.median_ms == 15.0
        assert stats.avg_ms == 15.0
        assert stats.std_ms == 3.0
        assert stats.count == 5

    def test_coefficient_of_variation(self):
        """Test that coefficient of variation is computed correctly."""
        stats = LayerDurationStatistics(
            label="test_layer",
            min_ms=10.0,
            max_ms=20.0,
            median_ms=15.0,
            avg_ms=100.0,
            std_ms=25.0,
            count=5,
        )
        # CV = std / avg = 25 / 100 = 0.25
        assert stats.coefficient_of_variation == pytest.approx(0.25)

    def test_coefficient_of_variation_zero_avg(self):
        """Test that coefficient of variation handles zero average gracefully."""
        stats = LayerDurationStatistics(
            label="test_layer",
            min_ms=0.0,
            max_ms=0.0,
            median_ms=0.0,
            avg_ms=0.0,
            std_ms=0.0,
            count=3,
        )
        assert stats.coefficient_of_variation == 0.0

    def test_frozen_model(self):
        """Test that LayerDurationStatistics is immutable."""
        stats = LayerDurationStatistics(
            label="test_layer",
            min_ms=10.0,
            max_ms=20.0,
            median_ms=15.0,
            avg_ms=15.0,
            std_ms=3.0,
            count=5,
        )
        with pytest.raises(ValidationError):
            stats.min_ms = 5.0


class TestLayerStatisticsAnalyzer:
    """Test LayerStatisticsAnalyzer class."""

    def test_compute_statistics_single_layer(self):
        """Test statistics computation for a single layer."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 20.0)
        collector.add_duration("layer1", 30.0)

        analyzer = LayerStatisticsAnalyzer(collector)
        stats = analyzer.get_layer_statistics("layer1")

        assert stats is not None
        assert stats.label == "layer1"
        assert stats.min_ms == pytest.approx(10.0)
        assert stats.max_ms == pytest.approx(30.0)
        assert stats.median_ms == pytest.approx(20.0)
        assert stats.avg_ms == pytest.approx(20.0)
        assert stats.count == 3

    def test_compute_statistics_multiple_layers(self):
        """Test statistics computation for multiple layers."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 20.0)
        collector.add_duration("layer2", 100.0)
        collector.add_duration("layer2", 200.0)

        analyzer = LayerStatisticsAnalyzer(collector)

        stats1 = analyzer.get_layer_statistics("layer1")
        stats2 = analyzer.get_layer_statistics("layer2")

        assert stats1 is not None
        assert stats1.avg_ms == pytest.approx(15.0)

        assert stats2 is not None
        assert stats2.avg_ms == pytest.approx(150.0)

    def test_get_statistics_returns_all(self):
        """Test that get_statistics returns all layer statistics."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer2", 20.0)
        collector.add_duration("layer3", 30.0)

        analyzer = LayerStatisticsAnalyzer(collector)
        all_stats = analyzer.get_statistics()

        assert len(all_stats) == 3
        labels = {s.label for s in all_stats}
        assert labels == {"layer1", "layer2", "layer3"}

    def test_get_layer_statistics_nonexistent(self):
        """Test that get_layer_statistics returns None for nonexistent layer."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)

        analyzer = LayerStatisticsAnalyzer(collector)
        assert analyzer.get_layer_statistics("nonexistent") is None

    def test_empty_collector(self):
        """Test analyzer with empty collector."""
        collector = IterativeLayerStatisticsCollector()
        analyzer = LayerStatisticsAnalyzer(collector)

        assert len(analyzer.get_statistics()) == 0


class TestMeasurementConsistency:
    """Test measurement consistency checking."""

    def test_consistent_measurements(self):
        """Test that consistent measurements return True."""
        collector = IterativeLayerStatisticsCollector()
        # Add enough samples with low variation
        for i in range(5):
            collector.add_duration("layer1", 100.0 + i * 0.1)

        analyzer = LayerStatisticsAnalyzer(collector)
        assert analyzer.check_measurement_consistency() is True

    def test_low_sample_count_warning(self, caplog):
        """Test that low sample count triggers warning."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 15.0)  # Only 2 samples

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(min_samples=3)

        assert result is False
        assert "Low sample count" in caplog.text
        assert "layer1" in caplog.text

    def test_high_variation_warning(self, caplog):
        """Test that high variation triggers warning."""
        collector = IterativeLayerStatisticsCollector()
        # Add samples with high variation (CV > 25%)
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 50.0)
        collector.add_duration("layer1", 90.0)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(cv_threshold=0.25)

        assert result is False
        assert "High duration variability" in caplog.text
        assert "layer1" in caplog.text

    def test_check_measurement_consistency_does_not_emit_full_table(self, caplog):
        """check_measurement_consistency must not dump the full table.

        Per-issue WARNINGs (low samples, high CV) are the only output
        from the consistency check itself; the full troubleshooting
        table is emitted separately by ``report_profiling_quality``
        (and therefore ``TensorManager.prepare_infer_mode``) only when
        ``enable_diagnostics`` is set *and* the consistency check
        flagged a problem. The emission contract for that path is
        anchored in
        ``tests/unit/test_tensor_manager_diagnostics_emission.py``.
        """
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 15.0)  # Only 2 samples

        analyzer = LayerStatisticsAnalyzer(collector)

        # INFO floor (rather than WARNING) so a regression that re-introduces
        # the table at INFO would also flip this assertion.
        with caplog.at_level(logging.INFO):
            analyzer.check_measurement_consistency()

        assert "Low sample count" in caplog.text, "per-issue warning must still fire"
        assert "Layer Duration Statistics" not in caplog.text, "full table must not be dumped by the consistency check"

    def test_custom_thresholds(self):
        """Test that custom thresholds are respected."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 15.0)  # 2 samples

        analyzer = LayerStatisticsAnalyzer(collector)

        # With min_samples=2, should pass the sample count check
        result = analyzer.check_measurement_consistency(min_samples=2, cv_threshold=0.5)
        assert result is True

    def test_multiple_issues_detected(self, caplog):
        """Test that multiple issues are detected and reported."""
        collector = IterativeLayerStatisticsCollector()
        # Layer with low sample count
        collector.add_duration("low_samples", 10.0)
        collector.add_duration("low_samples", 12.0)

        # Layer with high variation
        collector.add_duration("high_var", 10.0)
        collector.add_duration("high_var", 50.0)
        collector.add_duration("high_var", 90.0)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency()

        assert result is False
        assert "Low sample count" in caplog.text
        assert "High duration variability" in caplog.text

    def test_short_duration_layer_outlier_driven_not_flagged(self, caplog):
        """Short layer with high CV from one slow sample (mean > 1.2*median) is not flagged."""
        collector = IterativeLayerStatisticsCollector()
        # Short layer: mostly ~0.2 ms, one outlier 1.0 ms -> mean/median > 1.2
        for _ in range(9):
            collector.add_duration("embed", 0.2)
        collector.add_duration("embed", 1.0)
        # Long layer with low CV
        collector.add_duration("layer0", 100.0)
        collector.add_duration("layer0", 101.0)
        collector.add_duration("layer0", 102.0)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(
                cv_threshold=0.25,
                mean_median_ratio_outlier_skip=1.2,
            )

        assert result is True
        assert "High duration variability" not in caplog.text

    def test_short_layer_high_cv_genuine_spread_flagged(self, caplog):
        """Short layer with high CV and mean≈median (genuine spread) is still flagged."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("embed", 0.2)
        collector.add_duration("embed", 0.4)
        collector.add_duration("embed", 0.35)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(cv_threshold=0.25)

        assert result is False
        assert "High duration variability" in caplog.text
        assert "embed" in caplog.text

    def test_high_cv_outlier_driven_skipped_when_using_median(self, caplog):
        """High CV from outliers (mean >> median) is not flagged when pipeline uses median."""
        collector = IterativeLayerStatisticsCollector()
        # Typical runs ~236ms, one outlier 815ms -> high CV, but median ~236 is stable
        for _ in range(9):
            collector.add_duration("ModuleList.36", 236.0)
        collector.add_duration("ModuleList.36", 815.0)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(
                cv_threshold=0.25,
                mean_median_ratio_outlier_skip=1.2,
            )

        assert result is True
        assert "High duration variability" not in caplog.text

    def test_high_cv_outlier_skip_disabled_still_flags(self, caplog):
        """With mean_median_ratio_outlier_skip=0, outlier-driven high CV is still flagged."""
        collector = IterativeLayerStatisticsCollector()
        for _ in range(9):
            collector.add_duration("layer", 236.0)
        collector.add_duration("layer", 815.0)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(
                cv_threshold=0.25,
                mean_median_ratio_outlier_skip=0.0,
            )

        assert result is False
        assert "High duration variability" in caplog.text

    def test_high_cv_genuine_variation_still_flagged(self, caplog):
        """High CV from genuine spread (not one outlier) is still flagged; mean/median ~1."""
        collector = IterativeLayerStatisticsCollector()
        # Spread across all samples: no single outlier, mean and median both ~200
        for x in [100.0, 150.0, 200.0, 200.0, 250.0, 250.0, 300.0]:
            collector.add_duration("unstable_layer", x)

        analyzer = LayerStatisticsAnalyzer(collector)

        with caplog.at_level(logging.WARNING):
            result = analyzer.check_measurement_consistency(
                cv_threshold=0.25,
                mean_median_ratio_outlier_skip=1.2,
            )

        assert result is False
        assert "High duration variability" in caplog.text
        assert "unstable_layer" in caplog.text


class TestStatisticsTableFormatting:
    """Test statistics table formatting."""

    def test_format_statistics_table_basic(self):
        """Test basic table formatting."""
        collector = IterativeLayerStatisticsCollector()
        collector.add_duration("layer1", 10.0)
        collector.add_duration("layer1", 20.0)
        collector.add_duration("layer1", 30.0)

        analyzer = LayerStatisticsAnalyzer(collector)
        table = analyzer.format_statistics_table()

        assert "Layer Duration Statistics" in table
        assert "layer1" in table
        assert "Min" in table
        assert "Max" in table
        assert "Median" in table
        assert "Avg" in table
        assert "Std" in table
        assert "CV" in table
        assert "Count" in table

    def test_format_statistics_table_long_label_truncation(self):
        """Test that long layer labels are truncated."""
        collector = IterativeLayerStatisticsCollector()
        long_label = "a" * 50  # 50 characters
        collector.add_duration(long_label, 10.0)
        collector.add_duration(long_label, 20.0)
        collector.add_duration(long_label, 30.0)

        analyzer = LayerStatisticsAnalyzer(collector)
        table = analyzer.format_statistics_table()

        # Should be truncated to 38 chars + ".."
        assert ".." in table
        assert long_label not in table  # Full label should not appear

    def test_format_statistics_table_empty(self):
        """Test formatting with no statistics."""
        collector = IterativeLayerStatisticsCollector()
        analyzer = LayerStatisticsAnalyzer(collector)
        table = analyzer.format_statistics_table()

        assert table == "No layer statistics available."


class TestUntimedTrapsReport:
    """Traps that registered tensor IDs but have no duration samples.

    These are labels whose ``add_tensors`` path fired but ``add_duration``
    never did — e.g. decode-only traps like vLLM's ``logits_processor`` or
    traps that ran only while profiling was suspended.
    """

    def test_empty_collector_reports_nothing(self):
        from flextensor.layer_statistics_analyzer import UntimedTrapsReport

        report = UntimedTrapsReport(IterativeLayerStatisticsCollector())
        assert report.is_empty()
        assert report.labels == []
        assert report.format() == ""

    def test_fully_timed_collector_reports_nothing(self):
        """Labels that got durations are not in the report."""
        from flextensor.layer_statistics_analyzer import UntimedTrapsReport

        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("layer1", {1})
        collector.add_duration("layer1", 10.0)

        report = UntimedTrapsReport(collector)
        assert report.is_empty()
        assert report.format() == ""

    def test_tensors_without_duration_are_reported(self):
        """Tensors registered without any duration sample appear in the report."""
        from flextensor.layer_statistics_analyzer import UntimedTrapsReport

        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("logits_processor", {42})

        report = UntimedTrapsReport(collector)
        assert not report.is_empty()
        assert report.labels == ["logits_processor"]
        line = report.format()
        assert "Traps with no duration samples (dropped from strategy input):" in line
        assert "logits_processor" in line

    def test_mixed_timed_and_untimed_reports_only_untimed(self):
        """Only labels without durations show up; timed ones are filtered out."""
        from flextensor.layer_statistics_analyzer import UntimedTrapsReport

        collector = IterativeLayerStatisticsCollector()
        collector.add_tensors("timed", {1})
        collector.add_duration("timed", 10.0)
        collector.add_tensors("untimed_a", {2})
        collector.add_tensors("untimed_b", {3})

        report = UntimedTrapsReport(collector)
        assert set(report.labels) == {"untimed_a", "untimed_b"}
        line = report.format()
        assert "timed" not in line.replace("untimed_a", "").replace("untimed_b", "")
        assert "untimed_a" in line and "untimed_b" in line
