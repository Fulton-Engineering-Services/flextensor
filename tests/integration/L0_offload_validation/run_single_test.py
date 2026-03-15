#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run Single Test with Model and Test Presets

This script runs a single test with specific model and test preset combinations.
Use this to avoid memory issues from running multiple tests in batch (pytest).

Each test run is isolated and cleans up properly after completion.
"""

import warnings

from beartype.roar import BeartypeClawDecorWarning

# Fail on beartype decorator warnings - these indicate type hint issues
# This mirrors the pytest filterwarnings configuration in pyproject.toml
# IMPORTANT: This MUST be set before importing flextensor modules, as
# beartype warnings are emitted during import/decoration time.
warnings.filterwarnings("error", category=BeartypeClawDecorWarning)

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

try:
    from config_manager import ConfigManager
    from offload_validation import (
        CombinedTensorOffloadExperiment,
        ExperimentConfig,
        ModelFactory,
        OffloadManagerExperiment,
        TensorManagerFactory,
        set_seed,
    )
except ImportError:
    from .config_manager import ConfigManager
    from .offload_validation import (
        CombinedTensorOffloadExperiment,
        ExperimentConfig,
        ModelFactory,
        OffloadManagerExperiment,
        TensorManagerFactory,
        set_seed,
    )


class SingleTestRunner:
    """Run a single test with proper cleanup."""

    def __init__(self):
        """Initialize the test runner."""
        self.manager = ConfigManager()

    def cleanup_memory(self):
        """Aggressive memory cleanup."""

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        gc.collect()

    def run_test(
        self,
        model_preset: str,
        test_preset: str,
        save_results: bool = False,
        results_dir: str = "results",
        seed: int | None = None,
    ) -> bool:
        """
        Run a single test with specified presets.

        Args:
            model_preset: Name of the model preset
            test_preset: Name of the test preset
            save_results: Whether to save results to file
            results_dir: Directory to save results
            seed: Random seed for reproducibility (None = random)

        Returns:
            True if test succeeded, False otherwise
        """
        test_name = f"{model_preset}+{test_preset}"

        print("\n" + "=" * 80)
        print(f"RUNNING SINGLE TEST: {test_name}")
        print("=" * 80)

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Skipping test.")
            return False

        device = torch.device("cuda")

        try:
            config = self.manager.compose_presets(model_preset, test_preset)

            # Use distinct model name for high-level API to avoid baseline conflicts
            model_display = f"[om] {model_preset}" if config.api_type == "high_level" else model_preset
            test_name = f"{model_display}+{test_preset}"

            # Override seed if provided
            if seed is not None:
                config.seed = seed

            # Set random seed if specified (MUST be done before creating model!)
            if config.seed is not None:
                set_seed(config.seed)

            tensor_manager = None
            if config.api_type == "high_level":
                # High-level API: OffloadManager with auto-trap models
                model, input_tensor = ModelFactory.create_auto_trap_model(config, device)
                experiment = OffloadManagerExperiment(config, model, input_tensor)
            else:
                # Low-level API: TensorManager with manual trap models
                tensor_manager = TensorManagerFactory.create_tensor_manager(config, device)
                model, input_tensor = ModelFactory.create_model(config, tensor_manager, device)
                experiment = CombinedTensorOffloadExperiment(
                    config,
                    tensor_manager,
                    model,
                    input_tensor,
                )

            experiment.run_experiment()

            # Print results
            print("\n" + "=" * 80)
            print(f"✅ TEST PASSED: {test_name}")
            print("=" * 80)
            print("\n📊 RESULTS:")
            print(f"   Model Size:       {experiment.results.model_size_mb:.2f} MB")
            print(f"   Warmup Time:      {experiment.results.warmup_time_ms:.2f} ms")
            print(f"   Warmup Memory:    {experiment.results.warmup_memory_mb:.2f} MB")
            print(f"   Profile Time:     {experiment.results.profile_time_ms:.2f} ms")
            print(f"   Profile Memory:   {experiment.results.profile_memory_mb:.2f} MB")
            print(f"   Inference Time:   {experiment.results.infer_time_ms:.2f} ms")
            print(f"   Inference Memory: {experiment.results.infer_memory_mb:.2f} MB")
            if experiment.results.output_checksum:
                print(f"   Output Checksum:  {experiment.results.output_checksum[:16]}...")

            # Save results if requested
            if save_results:
                self._save_results(test_name, config, experiment, results_dir)

            # Cleanup
            del experiment
            del model
            del input_tensor
            if tensor_manager is not None:
                del tensor_manager
            self.cleanup_memory()

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"❌ TEST FAILED: {test_name}")
            print("=" * 80)
            print(f"\n💥 Error: {e}")

            print("\n🔍 Traceback:")
            traceback.print_exc()

            # Cleanup on failure
            self.cleanup_memory()

            return False
        return True

    def _save_results(
        self,
        test_name: str,
        config: ExperimentConfig,
        experiment: CombinedTensorOffloadExperiment,
        results_dir: str,
    ):
        """Save test results to file."""

        results_path = Path(results_dir)
        results_path.mkdir(exist_ok=True)

        # Create result dictionary
        result_data = {
            "test_name": test_name,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "config": config.model_dump(),
            "results": {
                "model_size_mb": experiment.results.model_size_mb,
                "warmup_time_ms": experiment.results.warmup_time_ms,
                "warmup_memory_mb": experiment.results.warmup_memory_mb,
                "profile_time_ms": experiment.results.profile_time_ms,
                "profile_memory_mb": experiment.results.profile_memory_mb,
                "infer_time_ms": experiment.results.infer_time_ms,
                "infer_memory_mb": experiment.results.infer_memory_mb,
                "output_checksum": experiment.results.output_checksum,
            },
        }

        # Convert torch.dtype to string for JSON serialization
        if "tensor_dtype" in result_data["config"]:
            result_data["config"]["tensor_dtype"] = str(result_data["config"]["tensor_dtype"])

        # Add validation results from experiment
        if experiment:
            result_data["results"]["warmup_vs_profile_match"] = experiment.results.warmup_vs_profile_match
            result_data["results"]["profile_vs_inference_match"] = experiment.results.profile_vs_inference_match

        # Save to file
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = results_path / f"test_{test_name}_{timestamp}.json"

        with filename.open("w") as f:
            json.dump(result_data, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

    def list_available_presets(self):
        """List all available model and test presets."""
        print("\n" + "=" * 80)
        print("AVAILABLE PRESETS")
        print("=" * 80)

        print("\n📦 Model Presets:")
        print("-" * 80)
        for preset in self.manager.list_model_presets():
            model = self.manager.get_model_preset(preset)
            print(f"  • {preset:<25} ({model.model_type}, {model.layers} layers)")

        print("\n🧪 Test Presets:")
        print("-" * 80)
        for preset in self.manager.list_test_presets():
            test = self.manager.get_test_preset(preset)
            mode = "baseline" if test.baseline_mode else f"offload-{test.transfer_mode}"
            print(f"  • {preset:<30} ({mode}, {test.iterations} iter)")

        print("\n" + "=" * 80)


def main():
    """CLI for running single tests."""
    parser = argparse.ArgumentParser(
        description="Run a single test with model and test presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single test
  python run_single_test.py basic-small baseline-quick

  # Run with result saving
  python run_single_test.py basic-small offload-strategy-quick --save-results

  # List available presets
  python run_single_test.py --list

    """,
    )

    parser.add_argument(
        "model_preset",
        nargs="?",
        type=str,
        help="Name of the model preset (e.g., basic-small, expert-medium)",
    )
    parser.add_argument(
        "test_preset",
        nargs="?",
        type=str,
        help="Name of the test preset (e.g., baseline-quick, offload-strategy-standard)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available presets",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    args = parser.parse_args()

    runner = SingleTestRunner()

    # List presets if requested
    if args.list:
        runner.list_available_presets()
        return

    # Validate arguments
    if not args.model_preset or not args.test_preset:
        print("❌ Error: Both model_preset and test_preset are required")
        print("\nUse --list to see available presets")
        print("Use --help for usage information")
        sys.exit(1)

    # Run the test
    success = runner.run_test(
        args.model_preset,
        args.test_preset,
        save_results=args.save_results,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
