# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for flextensor.contrib.vllm worker with FlexTensor offloading."""

import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from tests.integration._vllm_server import (
    FLEXTENSOR_SNAPSHOT_WORKER_CLS,
    VllmOffloadSmokeCase,
    run_vllm_server_test,
)
from tests.integration._vllm_utils import (
    load_gpu_memory_snapshots,
    parse_block_assignment_layers,
    sanitize_test_name,
)


@pytest.fixture
def test_output_dir(request) -> Path:
    """Fixture that provides a unique output directory for each test case.

    Args:
        request: pytest request fixture

    Returns:
        Path object for the test-specific output directory
    """
    test_dir = Path(__file__).parent
    sanitized_name = sanitize_test_name(request.node.name)
    output_dir = test_dir / "test_results" / sanitized_name
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="class")
def device_gpu() -> torch.device:
    """Fixture to provide GPU device.

    Returns:
        GPU device for testing

    Raises:
        pytest.skip: If CUDA is not available
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


class TestContribVLLM:
    """Test cases for flextensor.contrib.vllm.worker.FlexTensorOffloadWorker."""

    @pytest.mark.gpu_vram_40g
    @pytest.mark.parametrize(
        "case",
        [
            VllmOffloadSmokeCase(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                output_dir_name="qwen2_5_7b_instruct",
                cli_args=("--enforce-eager",),
            ),
        ],
    )
    def test_vllm_offloading_reduces_memory(
        self,
        case: VllmOffloadSmokeCase,
        device_gpu: torch.device,
        test_output_dir: Path,
    ) -> None:
        """Test that FlexTensor offloading integrates with vLLM serving.

        This test validates that the flextensor.contrib.vllm.worker.FlexTensorOffloadWorker
        correctly integrates FlexTensor offloading into vLLM's serving pipeline. It:
        1. Runs vLLM without offloading to establish baseline memory usage
        2. Runs vLLM with FlexTensor offloading enabled
        3. Compares the weights_memory and KV cache metrics between both runs
        4. Saves combined metrics to JSON for analysis

        The test asserts that with offloading enabled:
        - weights_memory is non-zero (some weights loaded during inference)
        - Both baseline and offload weights_memory are successfully parsed from vLLM logs

        Args:
            case: vLLM launch case to test
            device_gpu: GPU device for testing (from fixture)
            test_output_dir: Directory for test output
        """
        # Phase 1: Run baseline (without offloading)
        print("\n" + "=" * 60)
        print("PHASE 1: Running baseline (without offloading)")
        print("=" * 60)

        baseline_dir = test_output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        baseline_memory, baseline_metrics, _ = run_vllm_server_test(case, output_dir=baseline_dir)

        # Phase 2: Run with offloading
        print("\n" + "=" * 60)
        print("PHASE 2: Running with FlexTensor offloading")
        print("=" * 60)

        offload_dir = test_output_dir / "offload"
        offload_dir.mkdir(parents=True, exist_ok=True)

        offload_memory, offload_metrics, _ = run_vllm_server_test(
            case.with_flextensor_offload(), output_dir=offload_dir
        )

        # Phase 3: Compare results
        print("\n" + "=" * 60)
        print("COMPARISON: Baseline vs Offloading")
        print("=" * 60)

        assert baseline_memory.weights_memory_gib is not None, "Could not parse baseline weights_memory from vLLM logs"
        assert offload_memory.weights_memory_gib is not None, "Could not parse offload weights_memory from vLLM logs"

        memory_reduction = baseline_memory.weights_memory_gib - offload_memory.weights_memory_gib
        print(f"Baseline weights memory: {baseline_memory.weights_memory_gib:.2f} GiB")
        print(f"Offload weights memory:  {offload_memory.weights_memory_gib:.2f} GiB")
        print(f"Memory reduction:        {memory_reduction:.2f} GiB")

        if baseline_memory.available_kv_cache_memory_gib and offload_memory.available_kv_cache_memory_gib:
            kv_increase = offload_memory.available_kv_cache_memory_gib - baseline_memory.available_kv_cache_memory_gib
            print(f"Baseline KV cache:       {baseline_memory.available_kv_cache_memory_gib:.2f} GiB")
            print(f"Offload KV cache:        {offload_memory.available_kv_cache_memory_gib:.2f} GiB")
            print(f"KV cache increase:       {kv_increase:.2f} GiB")

        # Assertions
        assert offload_memory.weights_memory_gib > 0, (
            f"weights_memory should be non-zero with offloading, got {offload_memory.weights_memory_gib}"
        )
        assert memory_reduction > 0, (
            "Expected offloading to reduce weights memory, "
            f"but baseline={baseline_memory.weights_memory_gib:.2f} GiB and "
            f"offload={offload_memory.weights_memory_gib:.2f} GiB"
        )

        # Save combined metrics
        combined_metrics = {
            "model_name": case.model_name,
            "baseline": baseline_metrics,
            "offload": offload_metrics,
            "comparison": {
                "baseline_weights_memory_gib": baseline_memory.weights_memory_gib,
                "offload_weights_memory_gib": offload_memory.weights_memory_gib,
                "memory_reduction_gib": memory_reduction,
                "baseline_kv_cache_gib": baseline_memory.available_kv_cache_memory_gib,
                "offload_kv_cache_gib": offload_memory.available_kv_cache_memory_gib,
            },
        }

        metrics_file = test_output_dir / "comparison_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(combined_metrics, f, indent=2)

        print(f"\nCombined metrics saved to: {metrics_file}")

    @pytest.mark.gpu_vram_40g
    @pytest.mark.parametrize(
        "case",
        [
            VllmOffloadSmokeCase(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                output_dir_name="qwen2_5_7b_instruct",
                cli_args=("--enforce-eager",),
            ),
        ],
    )
    def test_vllm_no_wildcard_traps_by_default(
        self,
        case: VllmOffloadSmokeCase,
        device_gpu: torch.device,
        test_output_dir: Path,
    ) -> None:
        """Test that FlexTensorOffloadWorker creates per-layer traps by default.

        Regression test: the worker must NOT default to wildcard include_patterns=['*'],
        which collapses the entire model into one coarse trap and prevents per-layer
        pipelining. Per-layer traps are validated via the BLOCK ASSIGNMENT table,
        which lists every trap that was actually created during profiling.

        Incorrect (wildcard default):
            model  ← entire model = 1 trap, no pipelining

        Correct (specific layer patterns such as class:*DecoderLayer):
            model.layers.0  ← each layer has its own trap
            model.layers.1
            ...

        Args:
            case: vLLM launch case to test
            device_gpu: GPU device for testing (from fixture)
            test_output_dir: Directory for test output
        """
        output_dir = test_output_dir / "default_patterns"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Do NOT set FT_INCLUDE_PATTERNS — rely on worker defaults.
        # FT_ENABLE_DIAGNOSTICS=1 makes the BLOCK ASSIGNMENT table available for
        # trap enumeration regardless of measurement consistency.
        offload_memory, _, log_lines = run_vllm_server_test(
            case.with_flextensor_offload().with_env_vars(("FT_ENABLE_DIAGNOSTICS", "1")),
            output_dir=output_dir,
        )

        trap_labels = parse_block_assignment_layers(log_lines)
        assert trap_labels, (
            "BLOCK ASSIGNMENT table not found in logs. "
            "Set FT_ENABLE_DIAGNOSTICS=1 was not honoured or profiling did not complete."
        )

        print(f"\nBLOCK ASSIGNMENT ({len(trap_labels)} traps):")
        for name in trap_labels:
            print(f"  {name}")

        # Coarse-trap check: with wildcard ['*'], 'model' is the whole transformer
        # wrapped in a single trap — each forward pass is one giant block with no
        # opportunity to pipeline transfers for subsequent layers.
        coarse_trap_keys = [k for k in trap_labels if k == "model" or k.endswith(".model")]
        assert not coarse_trap_keys, (
            f"Coarse trap detected: {coarse_trap_keys} — entire model wrapped in one trap. "
            "Worker must use specific include_patterns (for example class:*DecoderLayer) "
            "so each transformer layer gets its own trap."
        )

        # Per-layer check: specific layer patterns produce one trap
        # per layer. Trap names use the full module path (e.g. model.layers.0,
        # model.layers.1, ...), so we match by numeric suffix.
        layer_entries = [k for k in trap_labels if re.search(r"\.\d+$", k)]
        assert len(layer_entries) >= 10, (
            f"Expected at least 10 individual layer traps, found {len(layer_entries)}: "
            f"{layer_entries}. Check that include_patterns includes layer-level patterns."
        )

        # Offloading must have moved tensors (weights_memory < full-model size)
        assert offload_memory.weights_memory_gib is not None, (
            "Could not parse weights_memory from vLLM logs — offloading may have failed"
        )
        assert offload_memory.weights_memory_gib > 0, (
            f"weights_memory should be non-zero with offloading, got {offload_memory.weights_memory_gib}"
        )

        print(f"\nPer-layer trap check passed: {len(layer_entries)} layer traps found")
        print(f"Weights memory: {offload_memory.weights_memory_gib:.2f} GiB")

    @pytest.mark.gpu_vram_40g
    @pytest.mark.parametrize(
        "case",
        [
            VllmOffloadSmokeCase(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                output_dir_name="qwen2_5_7b_instruct",
                cli_args=("--enforce-eager",),
            ),
        ],
    )
    def test_diagnostics_tables_appear_under_vllm(
        self,
        case: VllmOffloadSmokeCase,
        device_gpu: torch.device,
        test_output_dir: Path,
    ) -> None:
        """Block Assignment and Memory Transfer tables must appear in the vLLM server log.

        Regression test for issue 139: with FT_ENABLE_DIAGNOSTICS=1 set, the
        Block Assignment and Memory Transfer Statistics tables must appear in
        the captured server log at INFO level. Each table must appear exactly
        once per profiling run. (Layer Duration Statistics is intentionally
        emitted only via the WARNING path when measurements are inconsistent;
        consistent runs are covered by the strategy table.)
        """
        output_dir = test_output_dir / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)

        _, _, log_lines = run_vllm_server_test(
            case.with_flextensor_offload().with_env_vars(("FT_ENABLE_DIAGNOSTICS", "1")),
            output_dir=output_dir,
        )

        log_text = "\n".join(log_lines)

        assert "BLOCK ASSIGNMENT:" in log_text, (
            "Block Assignment table missing from vLLM server log — diagnostics bridge may not be installed."
        )
        assert "Memory Transfer Statistics" in log_text, (
            "Memory Transfer Statistics table missing from vLLM server log "
            "— INFO-path diagnostic records are being dropped."
        )

        # No duplicate emission: exactly one BLOCK ASSIGNMENT header per run.
        block_assignment_headers = [line for line in log_lines if "BLOCK ASSIGNMENT:" in line]
        assert len(block_assignment_headers) == 1, (
            f"Expected exactly 1 'BLOCK ASSIGNMENT:' header, got {len(block_assignment_headers)}: "
            f"{block_assignment_headers}"
        )
        memory_transfer_headers = [line for line in log_lines if "Memory Transfer Statistics" in line]
        assert len(memory_transfer_headers) == 1, (
            f"Expected exactly 1 'Memory Transfer Statistics' header, got {len(memory_transfer_headers)}: "
            f"{memory_transfer_headers}"
        )

    @pytest.mark.gpu_vram_40g
    @pytest.mark.parametrize(
        "case",
        [
            VllmOffloadSmokeCase(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                output_dir_name="qwen2_5_7b_instruct",
                cli_args=("--enforce-eager",),
            ),
        ],
    )
    def test_snapshot_worker_captures_all_stages(
        self,
        case: VllmOffloadSmokeCase,
        device_gpu: torch.device,
        test_output_dir: Path,
    ) -> None:
        """Test that FlexTensorSnapshotWorker dumps exactly one file with all 5 stages.

        Regression test for issue #88: FlexTensorSnapshotWorker was calling
        _dump_snapshots() inside compile_or_warm_up_model(), which is invoked
        twice internally during warmup_and_profile_model(). This produced
        premature dump files before after_load_model / after_kv_cache_init /
        final after_compile_warmup stages had run.

        This test validates the fix end-to-end by:
        1. Running FlexTensorSnapshotWorker with FT_VLLM_SNAPSHOT_OUTPUT_DIR set
        2. Reading the JSON snapshot file directly (no log parsing)
        3. Asserting exactly 1 file with all 5 expected labels in order
        4. Asserting GPU cuda_memory increases from init → load_model → kv_cache_init

        Args:
            case: vLLM launch case to test
            device_gpu: GPU device for testing (from fixture)
            test_output_dir: Directory for test output
        """
        snapshot_dir = test_output_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        _, _, _ = run_vllm_server_test(
            case.with_flextensor_offload(worker_cls=FLEXTENSOR_SNAPSHOT_WORKER_CLS).with_env_vars((
                "FT_VLLM_SNAPSHOT_OUTPUT_DIR",
                str(snapshot_dir),
            )),
            output_dir=test_output_dir,
        )

        # Read snapshot file directly — no log parsing needed
        json_files = list(snapshot_dir.glob("gpu_snapshots_rank0_device*.json"))
        assert len(json_files) == 1, (
            f"Expected exactly 1 snapshot file, got {len(json_files)}. "
            "Premature dumps during internal warmup were not suppressed."
        )

        snapshots = load_gpu_memory_snapshots(snapshot_dir)
        labels = [s["label"] for s in snapshots]

        expected_labels = [
            "after_init_device",
            "after_load_model",
            "after_determine_available_memory",
            "after_kv_cache_init",
            "after_compile_warmup",
        ]
        assert labels == expected_labels, (
            f"Expected snapshot labels {expected_labels}, got {labels}. "
            "Some lifecycle stages may be missing or in the wrong order."
        )

        # Validate GPU memory progression: cuda_memory should increase as
        # model weights load and KV cache is allocated.
        cuda_by_label = {s["label"]: s["gpu_memory"]["cuda_memory"] for s in snapshots}
        assert cuda_by_label["after_load_model"] > cuda_by_label["after_init_device"], (
            "GPU memory should increase after model weights are loaded. "
            f"after_init_device={cuda_by_label['after_init_device']}, "
            f"after_load_model={cuda_by_label['after_load_model']}"
        )
        assert cuda_by_label["after_kv_cache_init"] > cuda_by_label["after_load_model"], (
            "GPU memory should increase after KV cache is allocated. "
            f"after_load_model={cuda_by_label['after_load_model']}, "
            f"after_kv_cache_init={cuda_by_label['after_kv_cache_init']}"
        )

        # Validate host_memory is present on all snapshots
        for snap in snapshots:
            assert "host_memory" in snap, f"Missing host_memory in snapshot: {snap['label']}"
            assert snap["host_memory"].get("host_memory_total", 0) > 0, (
                f"host_memory_total should be positive for snapshot: {snap['label']}"
            )

        print(f"\nSnapshot validation passed: {len(snapshots)} stages captured")
        for snap in snapshots:
            cuda_gib = snap["gpu_memory"]["cuda_memory"] / (1024**3)
            print(f"  {snap['label']:45s} cuda={cuda_gib:.2f} GiB")

    @pytest.mark.gpu_vram_40g
    @pytest.mark.parametrize(
        "case",
        [
            VllmOffloadSmokeCase(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                output_dir_name="qwen2_5_7b_instruct_external_compile",
                # Non-eager native compile path: no --enforce-eager, CUDA graphs off.
                cli_args=(
                    "--compilation-config",
                    '{"cudagraph_mode": "NONE"}',
                    "--max-model-len",
                    "256",
                    "--max-num-seqs",
                    "1",
                ),
            ),
        ],
    )
    def test_vllm_external_compile_activates_compiled_offload(
        self,
        case: VllmOffloadSmokeCase,
        device_gpu: torch.device,
        test_output_dir: Path,
    ) -> None:
        """Smoke the real vLLM non-eager compiled-offload path.

        Existing L0 cases all pass ``--enforce-eager``, so they never exercise
        import-time compile-cache/AOT env handling, worker compile gating, or
        native fullgraph serving. This case enables ``FT_EXTERNAL_COMPILE=1``,
        leaves eager mode off, disables CUDA graphs, serves one request, and
        asserts compiled-offload activation from worker logs.
        """
        assert "--enforce-eager" not in case.cli_args

        output_dir = test_output_dir / "external_compile"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Native compile + replan warmups run during server startup before /health.
        memory, metrics, log_lines = run_vllm_server_test(
            case.with_flextensor_offload().with_env_vars(("FT_EXTERNAL_COMPILE", "1")),
            output_dir=output_dir,
            server_ready_timeout=1800,
        )
        log_text = "\n".join(log_lines)

        assert "FT-COMPILE-PATH: compiled-offload running under vLLM native fullgraph=True" in log_text, (
            "Import-time compiled-offload env setup did not run — FT_EXTERNAL_COMPILE may not "
            "have been visible when flextensor.contrib.vllm.worker was imported."
        )
        assert "external_compile=True" in log_text, (
            "Offload config log did not show external_compile=True; compiled-offload was not activated."
        )
        assert "FlexTensor compiled-offload: suppressed torch.compile" in log_text, (
            "Worker compile gate did not suppress vLLM compile during discovery/profiling."
        )
        assert "FlexTensor compiled-offload: re-enabled torch.compile" in log_text, (
            "Worker compile gate did not re-enable vLLM compile at INFERENCE."
        )
        assert "FlexTensor compiled-offload: installed compile-transparent forwards" in log_text, (
            "Compiled-offload forwards were not installed at the INFERENCE transition."
        )
        assert "FlexTensor compiled-offload: armed passive re-plan tail" in log_text, (
            "External-compile replan tail was not armed after INFERENCE."
        )
        assert "FlexTensor compiled offload does not support CUDA graphs yet" not in log_text, (
            'CUDA graphs were not disabled; pass --compilation-config \'{"cudagraph_mode": "NONE"}\'.'
        )

        assert memory.weights_memory_gib is not None and memory.weights_memory_gib > 0, (
            f"weights_memory should be non-zero with compiled offload, got {memory.weights_memory_gib}"
        )
        assert metrics.get("response_content"), "Expected a non-empty chat response from the compiled-offload serve"

        print("\nExternal-compile compiled-offload smoke passed")
        print(f"  Weights memory: {memory.weights_memory_gib:.2f} GiB")
        print(f"  Response: {metrics['response_content']}")
