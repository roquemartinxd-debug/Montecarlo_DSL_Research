"""Regression tests for the separate AWS four-core campaign profile."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks import run_v1_3_experiments as runner
from montecarlo_dsl.generator import GeneratorConfig


def test_performance_matrix_includes_a_true_in_process_serial_baseline() -> None:
    configs = runner.performance_configs()
    assert configs[0] == ("v1.3_serial_in_process", "ast_only", 1)
    assert ("v1.3_fixed_w1", "fixed", 1) in configs
    assert ("v1.3_fixed_w2", "fixed", 2) in configs
    assert ("v1.3_fixed_w4", "fixed", 4) in configs
    assert ("v1.3_adaptive_hybrid", "hybrid", 4) in configs


def test_serial_baseline_does_not_create_a_process_pool() -> None:
    source = runner.source_with_iterations("normal_mean", 2_000)
    config = GeneratorConfig(
        batch_size=500,
        logical_block_size=250,
        max_workers=1,
        scheduler_mode="ast_only",
        max_iterations=2_000,
    )
    with tempfile.TemporaryDirectory(prefix="mcdsl_serial_baseline_test_") as tmp:
        result = runner.run_simulation(source, config, temp_dir=Path(tmp), tag="aws_serial_baseline_test")
    metadata = result["complete_event"]["metadata"]
    assert metadata["use_process_pool"] is False
    assert metadata["worker_count"] == 1
    assert result["complete_event"]["valid_results"] == 2_000


def test_aws_memory_profile_adds_four_worker_config() -> None:
    standard = runner.memory_worker_configs("standard")
    aws = runner.memory_worker_configs("aws-4core")
    assert len(standard) == 2
    assert ("multiprocess_w4", "fixed", 4) not in standard
    assert ("multiprocess_w4", "fixed", 4) in aws
    assert len(aws) == 3


def test_lscpu_core_socket_parser_counts_unique_physical_cores() -> None:
    sample = """# The following is the parsable format, which can be fed to external programs.
# CORE,SOCKET
0,0
0,0
1,0
1,0
2,0
2,0
3,0
3,0
"""
    assert runner.parse_lscpu_core_socket_csv(sample) == 4


def test_generator_cli_exposes_aws_profile_and_output_isolation() -> None:
    import subprocess
    import sys

    project_root = Path(runner.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(project_root / "benchmarks" / "run_v1_3_experiments.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--output-dir" in proc.stdout
    assert "aws-4core" in proc.stdout
    assert "--instance-type" in proc.stdout
    assert "--cloud-region" in proc.stdout


def test_exp2_writes_paired_speedups_against_serial_and_fixed_pools(
    tmp_path: Path, monkeypatch
) -> None:
    import csv

    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "bootstrap_ci95", lambda values, stat_fn: (float(stat_fn(values)), float(stat_fn(values))))

    def fake_run_simulation(source, config, *, temp_dir, tag, cost_model=None):
        if config.scheduler_mode == "ast_only":
            seconds = 1.0
            workers = 1
            use_pool = False
        elif config.scheduler_mode == "fixed" and config.max_workers == 1:
            seconds = 1.2
            workers = 1
            use_pool = True
        elif config.scheduler_mode == "fixed" and config.max_workers == 2:
            seconds = 0.8
            workers = 2
            use_pool = True
        elif config.scheduler_mode == "fixed" and config.max_workers == 4:
            seconds = 0.6
            workers = 4
            use_pool = True
        else:
            seconds = 0.55
            workers = 4
            use_pool = True
        return {
            "compile_seconds": 0.01,
            "execution_seconds": seconds - 0.01,
            "total_seconds": seconds,
            "sha256": "fixed-result-hash",
            "complete_event": {"metadata": {
                "worker_count": workers,
                "batch_count": 4,
                "use_process_pool": use_pool,
                "peak_retained_coordinator_bytes": 1024,
                "sum_of_process_peak_rss_bytes": 4096,
            }},
        }

    monkeypatch.setattr(runner, "run_simulation", fake_run_simulation)
    runner.run_exp2_performance_and_scalability(
        tmp_path, confirmatory_reps=2, pilot_reps=2, cost_model=None
    )
    with (tmp_path / "performance_comparison_summary.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    one_million = [row for row in rows if row["iterations"] == "1000000"]
    assert {row["configuration"] for row in one_million} == {
        "v1.3_serial_in_process", "v1.3_fixed_w1", "v1.3_fixed_w2",
        "v1.3_fixed_w4", "v1.3_adaptive_hybrid",
    }
    serial_row = next(row for row in one_million if row["configuration"] == "v1.3_serial_in_process")
    w4_row = next(row for row in one_million if row["configuration"] == "v1.3_fixed_w4")
    assert float(serial_row["median_speedup_vs_serial_in_process"]) == 1.0
    assert float(w4_row["median_speedup_vs_serial_in_process"]) > 1.0
    ten_million = [row for row in rows if row["iterations"] == "10000000"]
    assert len(ten_million) == 5
    assert all("median_speedup_vs_serial_in_process" in row for row in ten_million)
