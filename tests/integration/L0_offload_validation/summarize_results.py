#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Summarize Test Results

Read all JSON result files from a directory and create a summary table.
This is run after running multiple tests with run_single_test.py.

Usage:
    python summarize_results.py test_results/
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TestSummary(BaseModel):
    """Summary of a single test result."""

    model: str = Field(description="Model name/preset used in test")
    test: str = Field(description="Test name/preset used in test")
    success: bool = Field(description="Whether the test succeeded")
    inference_time_ms: float = Field(default=0.0, description="Inference time in milliseconds")
    inference_memory_mb: float = Field(default=0.0, description="Inference memory usage in MB")
    profile_time_ms: float = Field(default=0.0, description="Profile time in milliseconds")
    profile_memory_mb: float = Field(default=0.0, description="Profile memory usage in MB")
    error_message: str = Field(default="", description="Error message if test failed")
    is_baseline: bool = Field(default=False, description="Whether this is a baseline test")
    output_validation: str = Field(default="N/A", description="Test output validation status")
    output_checksum: str = Field(default="", description="MD5 checksum of inference output")
    checksum_correctness: str = Field(
        default="N/A",
        description="Correctness vs baseline: ✓ MATCH / ✗ DIFFER / BASELINE / N/A",
    )
    config: dict[str, Any] | None = Field(default=None, description="Full configuration from the test")

    @property
    def validation_status(self) -> str:
        """Calculate validation status (inference vs profile)."""
        if not self.success or self.profile_time_ms == 0:
            return "N/A"

        # For baseline tests, expect inference ≈ profile (strict threshold)
        # For offload tests, they can be very different (lenient threshold)
        time_threshold = 10.0 if self.is_baseline else 100.0  # 10% for baseline, 100% for offload
        memory_threshold = 15.0 if self.is_baseline else 300.0  # 15% for baseline, 300% for offload

        time_diff = abs((self.inference_time_ms - self.profile_time_ms) / self.profile_time_ms * 100)
        memory_diff = abs((self.inference_memory_mb - self.profile_memory_mb) / self.profile_memory_mb * 100)

        if time_diff <= time_threshold and memory_diff <= memory_threshold:
            return "PASS"
        return "FAIL"

    @property
    def validation_message(self) -> str:
        """Get validation message."""
        if not self.success or self.profile_time_ms == 0:
            return self.error_message if self.error_message else "N/A"

        time_diff = (self.inference_time_ms - self.profile_time_ms) / self.profile_time_ms * 100
        memory_diff = (self.inference_memory_mb - self.profile_memory_mb) / self.profile_memory_mb * 100

        return f"Time: {time_diff:+.1f}%, Memory: {memory_diff:+.1f}%"


def load_result_file(filepath: Path) -> TestSummary:
    """Load a single result file and extract summary."""
    try:
        with filepath.open("r") as f:
            data = json.load(f)

        # Extract model and test from test_name
        test_name = data.get("test_name", "")
        if "+" in test_name:
            model, test = test_name.split("+", 1)
        else:
            model = "unknown"
            test = "unknown"

        # Determine if this is a baseline test
        is_baseline = "baseline" in test.lower()

        # Extract configuration
        config = data.get("config", {})

        # Check if test succeeded
        success = "error" not in data or not data.get("error")

        # Extract output validation (from experiment validation checks)
        output_validation = "N/A"
        if success and "results" in data:
            results = data["results"]

            # Check for validation results
            warmup_vs_profile = results.get("warmup_vs_profile_match")
            profile_vs_inference = results.get("profile_vs_inference_match")

            if warmup_vs_profile is not None and profile_vs_inference is not None:
                output_validation = "✓ PASS" if warmup_vs_profile and profile_vs_inference else "✗ FAIL"

            return TestSummary(
                model=model,
                test=test,
                success=True,
                inference_time_ms=results.get("infer_time_ms", 0.0),
                inference_memory_mb=results.get("infer_memory_mb", 0.0),
                profile_time_ms=results.get("profile_time_ms", 0.0),
                profile_memory_mb=results.get("profile_memory_mb", 0.0),
                is_baseline=is_baseline,
                output_validation=output_validation,
                output_checksum=results.get("output_checksum", ""),
                config=config,
            )
        return TestSummary(
            model=model,
            test=test,
            success=False,
            error_message=data.get("error", "Unknown error"),
            is_baseline=is_baseline,
            output_validation="N/A",
            config=config,
        )

    except Exception as e:
        return TestSummary(
            model="unknown",
            test="unknown",
            success=False,
            error_message=f"Failed to load file: {e}",
        )


def validate_checksums_against_baseline(summaries: list[TestSummary]) -> list[TestSummary]:
    """
    Validate checksums of offload tests against their baseline.

    Args:
        summaries: List of test summaries

    Returns:
        Updated list with checksum_correctness field populated
    """
    # Group by model to find baseline for each model type
    model_baselines: dict[str, str] = {}

    for summary in summaries:
        if summary.is_baseline and summary.output_checksum:
            model_baselines[summary.model] = summary.output_checksum

    # Validate each test against its baseline
    for summary in summaries:
        if not summary.success or not summary.output_checksum:
            summary.checksum_correctness = "N/A"
        elif summary.is_baseline:
            summary.checksum_correctness = "BASELINE"
        elif summary.model in model_baselines:
            baseline_checksum = model_baselines[summary.model]
            if summary.output_checksum == baseline_checksum:
                summary.checksum_correctness = "✓ MATCH"
            else:
                summary.checksum_correctness = "✗ DIFFER"
        else:
            # No baseline found for this model
            summary.checksum_correctness = "NO BASELINE"

    return summaries


def print_configurations(summaries: list[TestSummary]):  # noqa: C901
    """Print model and test configurations used in the tests."""
    if not summaries:
        return

    # Extract unique model and test configurations
    model_configs = {}
    test_configs = {}

    for summary in summaries:
        if not summary.config:
            continue

        # Group by model preset name
        if summary.model not in model_configs:
            model_configs[summary.model] = summary.config

        # Group by test preset name
        if summary.test not in test_configs:
            test_configs[summary.test] = summary.config

    # Print Model Configurations
    print("\n" + "=" * 100)
    print("MODEL PRESETS")
    print("=" * 100)

    # Model parameters to display with defaults
    model_params = {
        "model_type": "basic",
        "layers": 64,
        "iterations": 50,
        "tensor_shape": (14336, 4096),
        "tensor_dtype": "torch.bfloat16",
        "dim": 4096,
        "inter_dim": 14336,
        "num_experts": 8,
        "batch_size": 1,
        "seq_len": 1024,
        "use_non_uniform": False,
        "num_experts_list": None,
    }

    for model_name in sorted(model_configs.keys()):
        config = model_configs[model_name]
        print(f"\n{model_name}:")
        for param, default_value in model_params.items():
            value = config.get(param, default_value)
            # Format tensor_shape nicely
            if param == "tensor_shape" and isinstance(value, list):
                value = f"({value[0]}, {value[1]})"
            # Shorten long lists
            if param == "num_experts_list" and value and isinstance(value, list) and len(value) > 10:
                value = f"[{len(value)} values]"
            print(f"  {param:25} = {value}")

    # Print Test Configurations
    print("\n" + "=" * 100)
    print("TEST PRESETS")
    print("=" * 100)

    # Test parameters to display with defaults
    test_params = {
        "baseline_mode": False,
        "transfer_mode": "strategy",
        "pinned_memory": True,
        "knapsack_scale": 1.0,
        "rearrange_transfers": False,
        "compute_transfer_gap": 1,
    }

    for test_name in sorted(test_configs.keys()):
        config = test_configs[test_name]
        print(f"\n{test_name}:")
        for param, default_value in test_params.items():
            value = config.get(param, default_value)
            print(f"  {param:25} = {value}")

    print()


def print_summary_table(summaries: list[TestSummary]):  # noqa: C901
    """Print summary table to console."""
    if not summaries:
        print("No results to summarize.")
        return

    print("\n" + "=" * 140)
    print("TEST RESULTS SUMMARY")
    print("=" * 140)

    # Header
    print(f"{'Model':<20} {'Test':<40} {'Infer Time':<12} {'Infer Mem':<12} {'Output Val':<12} {'vs Baseline':<13}")
    print("-" * 140)

    # Rows
    for summary in summaries:
        infer_time = f"{summary.inference_time_ms:.2f} ms" if summary.success else "N/A"
        infer_mem = f"{summary.inference_memory_mb:.2f} MB" if summary.success else "N/A"
        # Show "BASELINE" for baseline tests in Output Val column
        output_val = "BASELINE" if summary.is_baseline else summary.output_validation
        correctness = summary.checksum_correctness  # Cross-test validation (vs baseline)

        print(
            f"{summary.model:<20} {summary.test:<40} {infer_time:<12} {infer_mem:<12} {output_val:<12} {correctness:<13}",  # noqa: E501
        )

    print("=" * 140)

    # Summary statistics
    total = len(summaries)
    passed = sum(1 for s in summaries if s.success)
    failed = total - passed

    # Internal validation - only count non-baseline tests
    non_baseline_tests = [s for s in summaries if not s.is_baseline]
    output_validation_passed = sum(1 for s in non_baseline_tests if s.output_validation == "✓ PASS")
    output_validation_failed = sum(1 for s in non_baseline_tests if s.output_validation == "✗ FAIL")

    # Checksum correctness analysis
    checksum_match = sum(1 for s in summaries if s.checksum_correctness == "✓ MATCH")
    checksum_differ = sum(1 for s in summaries if s.checksum_correctness == "✗ DIFFER")
    baselines = sum(1 for s in summaries if s.checksum_correctness == "BASELINE")

    print(f"\nSummary: {total} tests, {passed} passed, {failed} failed")
    if non_baseline_tests:
        print(
            f"Internal Validation (offload only): {output_validation_passed} passed, {output_validation_failed} failed",
        )
    print(
        f"Baseline Validation (offload vs baseline): {checksum_match} match, {checksum_differ} differ, {baselines} baselines",  # noqa: E501
    )

    # Show warnings for internal validation failures
    if output_validation_failed > 0:
        print("\n" + "⚠️ " * 30)
        print("⚠️  INTERNAL VALIDATION FAILURE!")
        print("⚠️ " * 30)
        print(f"\n{output_validation_failed} test(s) failed internal validation (profile vs inference)!")
        print("\nTests with internal validation failures:")
        for s in summaries:
            if s.output_validation == "✗ FAIL":
                print(f"  ✗ {s.model} + {s.test}")
        print("\nThis indicates:")
        print("  - Inconsistent results within the same test run")
        print("  - Profile and inference phases produce different outputs")
        print("  - Critical correctness issue that needs immediate attention")
        print("\nRecommendations:")
        print("  1. Review tensor manager implementation")
        print("  2. Check for synchronization issues")
        print("  3. Verify tensor transfer correctness")

    # Show warnings if checksums differ
    if checksum_differ > 0:
        print("\n" + "⚠️ " * 30)
        print("⚠️  CHECKSUM MISMATCH DETECTED!")
        print("⚠️ " * 30)
        print(f"\n{checksum_differ} test(s) have different checksums than their baseline!")
        print("\nTests with checksum mismatches:")
        for s in summaries:
            if s.checksum_correctness == "✗ DIFFER":
                print(f"  ✗ {s.model} + {s.test}")
                print(f"    Test checksum:     {s.output_checksum[:16]}...")
                # Find baseline
                baseline_checksum = "N/A"
                for baseline in summaries:
                    if baseline.model == s.model and baseline.is_baseline:
                        baseline_checksum = baseline.output_checksum[:16]
                        break
                print(f"    Baseline checksum: {baseline_checksum}...")
        print("\nThis indicates:")
        print("  - Different computation results (CORRECTNESS ISSUE)")
        print("  - Floating point precision differences (may be acceptable)")
        print("  - CUDA non-determinism (check with multiple runs)")
        print("\nRecommendations:")
        print("  1. Verify with same seed: ./run_multiple_tests.sh --seed 42")
        print("  2. Run verification: ./verify_seed.sh")
        print("  3. Compare numerical differences (small differences may be OK)")
        print("  4. Review offload implementation for correctness")

    # Show success message if all validations pass
    if output_validation_failed == 0 and checksum_differ == 0:
        if output_validation_passed > 0 or checksum_match > 0:
            print("\n" + "✓" * 60)
            print("✓✓✓ ALL VALIDATIONS PASSED! ✓✓✓")
            print("✓" * 60)
            if output_validation_passed > 0:
                print(
                    f"✓ Internal validation (offload): {output_validation_passed}/{output_validation_passed} passed",
                )
            if checksum_match > 0:
                print(f"✓ Baseline validation: {checksum_match}/{checksum_match} offload tests match baseline")
            print("\nAll tests are producing correct and reproducible results!")
    elif checksum_differ == 0 and checksum_match > 0:
        print(f"\n✓ All {checksum_match} offload tests match their baseline checksums!")
        print("  Results are correct and reproducible.")

    # Checksum uniqueness analysis (for debugging)
    checksums = [s.output_checksum for s in summaries if s.output_checksum]
    unique_checksums = len(set(checksums))
    if checksums:
        print(f"\nChecksum diversity: {len(checksums)} results, {unique_checksums} unique outputs")

    print()


def save_summary_csv(summaries: list[TestSummary], output_file: Path):
    """Save summary to CSV file."""
    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Test",
                "Success",
                "Inference Time (ms)",
                "Inference Memory (MB)",
                "Profile Time (ms)",
                "Profile Memory (MB)",
                "Output Validation",
                "Output Checksum",
                "Checksum Correctness",
                "Metric Validation",
                "Validation Message",
                "Error Message",
            ],
        )

        for summary in summaries:
            writer.writerow(
                [
                    summary.model,
                    summary.test,
                    summary.success,
                    f"{summary.inference_time_ms:.2f}" if summary.success else "",
                    f"{summary.inference_memory_mb:.2f}" if summary.success else "",
                    f"{summary.profile_time_ms:.2f}" if summary.success else "",
                    f"{summary.profile_memory_mb:.2f}" if summary.success else "",
                    summary.output_validation,
                    summary.output_checksum,
                    summary.checksum_correctness,
                    summary.validation_status,
                    summary.validation_message,
                    summary.error_message,
                ],
            )

    print(f"💾 Summary saved to: {output_file}")


def save_summary_json(summaries: list[TestSummary], output_file: Path):
    """Save summary to JSON file."""
    data = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "total_tests": len(summaries),
        "passed": sum(1 for s in summaries if s.success),
        "failed": sum(1 for s in summaries if not s.success),
        "output_validation_passed": sum(1 for s in summaries if s.output_validation == "✓ PASS"),
        "metric_validation_passed": sum(1 for s in summaries if s.validation_status == "PASS"),
        "results": [
            {
                "model": s.model,
                "test": s.test,
                "success": s.success,
                "inference_time_ms": s.inference_time_ms,
                "inference_memory_mb": s.inference_memory_mb,
                "profile_time_ms": s.profile_time_ms,
                "profile_memory_mb": s.profile_memory_mb,
                "output_validation": s.output_validation,
                "output_checksum": s.output_checksum,
                "checksum_correctness": s.checksum_correctness,
                "metric_validation": s.validation_status,
                "validation_message": s.validation_message,
                "error_message": s.error_message,
            }
            for s in summaries
        ],
    }

    with output_file.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"💾 Summary saved to: {output_file}")


def main() -> int:
    """Main function.

    Returns:
        Exit code: 0 for success, 1 for validation failures.
    """
    parser = argparse.ArgumentParser(
        description="Summarize test results from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize results in directory
  python summarize_results.py test_results/

  # Save summary to files
  python summarize_results.py test_results/ --save-summary

  # Custom output location
  python summarize_results.py test_results/ --save-summary --output-dir summaries/
        """,
    )

    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save summary to CSV and JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save summary files (default: current directory)",
    )

    args = parser.parse_args()

    # Find all JSON result files
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Error: Directory not found: {results_dir}")
        return 1

    result_files = sorted(results_dir.glob("test_*.json"))

    if not result_files:
        print(f"❌ Error: No result files found in: {results_dir}")
        print("   Looking for files matching pattern: test_*.json")
        return 1

    print(f"\n📊 Loading {len(result_files)} result files...")

    # Load all results
    summaries = []
    for filepath in result_files:
        summary = load_result_file(filepath)
        summaries.append(summary)

    # Validate checksums against baseline
    summaries = validate_checksums_against_baseline(summaries)

    # Print configurations
    print_configurations(summaries)

    # Print table
    print_summary_table(summaries)

    # Save summary files if requested
    if args.save_summary:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

        csv_file = output_dir / f"summary_{timestamp}.csv"
        save_summary_csv(summaries, csv_file)

        json_file = output_dir / f"summary_{timestamp}.json"
        save_summary_json(summaries, json_file)

    # Determine exit code based on validation results
    failed = sum(1 for s in summaries if not s.success)
    non_baseline_tests = [s for s in summaries if not s.is_baseline]
    output_validation_failed = sum(1 for s in non_baseline_tests if s.output_validation == "✗ FAIL")
    checksum_differ = sum(1 for s in summaries if s.checksum_correctness == "✗ DIFFER")

    if failed > 0:
        print(f"\n❌ FAILED: {failed} test(s) failed to execute")
        return 1

    if output_validation_failed > 0:
        print(f"\n❌ FAILED: {output_validation_failed} test(s) failed internal validation")
        return 1

    if checksum_differ > 0:
        print(f"\n❌ FAILED: {checksum_differ} test(s) have checksum mismatches vs baseline")
        return 1

    print("\n✅ All validations passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
