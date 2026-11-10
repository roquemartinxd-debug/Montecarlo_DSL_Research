from __future__ import annotations

import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks import run_v1_3_experiments as base
from benchmarks import run_v1_3_targeted_experiments as targeted
from montecarlo_dsl import GeneratorConfig, InterpretableASTCostModel


def test_delayed_first_requires_positive_delay() -> None:
    with pytest.raises(ValueError, match="requires task_delay_first_seconds"):
        GeneratorConfig(task_completion_order="delayed_first")
    with pytest.raises(ValueError, match="finite non-negative"):
        GeneratorConfig(task_delay_first_seconds=-0.1)


def test_break_even_configurations_keep_scheduler_modes_unchanged() -> None:
    configs = targeted._break_even_configs(2)
    by_label = {item["label"]: item for item in configs}
    assert by_label["serial_in_process"]["mode"] == "ast_only"
    assert by_label["fixed_pool_w2"]["mode"] == "fixed"
    assert by_label["fixed_pool_w4"]["workers"] == 4
    assert by_label["adaptive_hybrid"]["mode"] == "hybrid"
    assert by_label["adaptive_hybrid"]["workers"] == 2


def test_memory_configurations_follow_physical_core_profile() -> None:
    assert [c["workers"] for c in targeted._memory_configs(2)] == [1, 2]
    assert [c["workers"] for c in targeted._memory_configs(4)] == [1, 2, 4]


def test_fidelity_comparison_includes_paired_delta() -> None:
    histogram = {"total_count": 2, "bins": [{"count": 1}, {"count": 1}]}
    item = {
        "stats": {"avg": 1.0, "std": 0.5, "min": 0.0, "max": 2.0,
                  "p05": 0.1, "p50": 1.0, "p95": 1.9},
        "valid_results": 2,
        "histogram": histogram,
    }
    event = {
        "branches": [{"id": "main", **item}],
        "paired": {**item, "correlation_plus_minus_mean": 0.25},
    }
    result = targeted._fidelity_between(event, event)
    assert result["exact_moments_preserved"] is True
    assert result["exact_counts_preserved"] is True
    assert result["max_quantile_relative_error_pct"] == 0.0
    assert result["max_histogram_tv_distance_pct"] == 0.0
    assert result["paired_delta_compared"] is True


def test_delayed_first_stress_preserves_canonical_result_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "_CHECKPOINT", None)
    source = base.source_with_iterations("normal_mean", 40_000)
    model = InterpretableASTCostModel()
    with tempfile.TemporaryDirectory() as work:
        normal = base.run_simulation(
            source,
            GeneratorConfig(
                seed=101, batch_size=5_000, logical_block_size=1_000,
                max_workers=2, scheduler_mode="fixed", max_iterations=40_000,
                task_completion_order="as_completed",
            ),
            temp_dir=Path(work), tag="exp11_order_normal", cost_model=model,
        )
        delayed = base.run_simulation(
            source,
            GeneratorConfig(
                seed=101, batch_size=5_000, logical_block_size=1_000,
                max_workers=2, scheduler_mode="fixed", max_iterations=40_000,
                task_completion_order="delayed_first", task_delay_first_seconds=0.05,
            ),
            temp_dir=Path(work), tag="exp11_order_delayed", cost_model=model,
        )
    assert normal["sha256"] == delayed["sha256"]
    meta = delayed["complete_event"]["metadata"]
    assert meta["task_submission_order"] == "delayed_first"
    assert meta["task_delay_first_seconds"] == 0.05
    assert "max_completed_results_observed" in meta
    assert "max_paired_correlations_observed" in meta
    assert meta["max_futures_observed"] <= meta["max_in_flight_tasks"]
    assert meta["max_completed_results_observed"] <= meta["max_in_flight_tasks"]
    assert meta["max_block_coverage_entries_observed"] <= meta["max_pending_logical_blocks"]


def test_monitored_memory_probe_records_process_tree_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(base, "_CHECKPOINT", None)
    source = base.source_with_iterations("normal_mean", 5_000)
    result = targeted._run_monitored_memory_probe(
        source,
        GeneratorConfig(
            seed=101, batch_size=5_000, logical_block_size=1_000,
            max_workers=1, scheduler_mode="ast_only", memory_mode="bounded",
            max_iterations=5_000,
        ),
        temp_dir=tmp_path,
        tag="test_memory_monitor",
        cost_model=InterpretableASTCostModel(),
        monitor_interval=0.01,
        timeout_seconds=60,
    )
    monitor = result["memory_monitor"]
    assert result["complete_event"]["total_iterations"] == 5_000
    assert monitor["method"] == "psutil_process_tree_rss_sampled"
    assert monitor["peak_coordinator_rss_bytes"] > 0
    assert monitor["peak_process_tree_rss_sum_bytes"] >= monitor["peak_coordinator_rss_bytes"]
    assert monitor["samples"] > 0


def test_break_even_runner_writes_paired_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(base, "_CHECKPOINT", None)
    monkeypatch.setattr(targeted, "BREAK_EVEN_MODELS", ("normal_mean",))
    monkeypatch.setattr(targeted, "BREAK_EVEN_LEVELS", (10_000,))
    monkeypatch.setattr(targeted, "BREAK_EVEN_LARGE_ANCHORS", ())
    args = Namespace(
        phase="exploratory", exploration_reps=1, large_anchor_reps=1,
        confirmatory_reps=1, confirmatory_iterations="", physical_cores=2,
        platform_label="test-2core", cost_model_weights=(1e-7, 1e-8, 1e-8, 1e-8),
        output_dir=tmp_path,
    )
    with tempfile.TemporaryDirectory() as work:
        raw, summary, thresholds = targeted._run_break_even(
            args, temp_dir=Path(work), cost_model=InterpretableASTCostModel(),
        )
    assert len(raw) == 5
    assert len(summary) == 5
    assert len(thresholds) == 3
    assert (tmp_path / "exp10_break_even_exploratory_raw.csv").is_file()


def test_memory_study_small_smoke_writes_scaling_and_fidelity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(base, "_CHECKPOINT", None)
    monkeypatch.setattr(targeted, "MEMORY_MODELS", ("normal_mean",))
    monkeypatch.setattr(targeted, "MEMORY_LEVELS_BASE", (250_000,))
    monkeypatch.setattr(targeted, "MEMORY_RETAIN_LEVELS", (250_000,))
    monkeypatch.setattr(targeted, "MEMORY_STRESS_ORDERS", ())
    monkeypatch.setattr(targeted, "_memory_configs", lambda _physical: [
        {"label": "serial_in_process_w1", "mode": "ast_only", "workers": 1}
    ])
    args = Namespace(
        profile="windows-2core", max_memory_iterations=250_000,
        physical_cores=2, platform_label="test-2core", memory_reps=1,
        retain_reps=1, stress_reps=1, include_retain_5m=False,
        monitor_interval=0.01, memory_timeout_seconds=60.0,
        first_task_delay_seconds=1.0, output_dir=tmp_path,
    )
    with tempfile.TemporaryDirectory() as work:
        raw, summary, fidelity, slopes = targeted._run_memory_study(
            args, temp_dir=Path(work), cost_model=InterpretableASTCostModel(),
        )
    assert len(raw) == 2
    assert len(summary) == 2
    assert len(fidelity) == 1
    assert len(slopes) == 2
    assert (tmp_path / "exp11_memory_scaling_raw.csv").is_file()
    assert (tmp_path / "exp11_memory_fidelity.csv").is_file()


def test_bounded_serial_task_larger_than_reorder_limit_is_streamed() -> None:
    """A contiguous serial task may contain >64 logical blocks without false overflow."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(base, "_CHECKPOINT", None)
    source = base.source_with_iterations("normal_mean", 5_000_000)
    model = InterpretableASTCostModel(
        weights_seconds_per_sample=(2.2583574403375308e-07, 1.7661118774571211e-08,
                                    8.9818060129072783e-08, 5.5e-09)
    )
    try:
        with tempfile.TemporaryDirectory() as work:
            result = base.run_simulation(
                source,
                GeneratorConfig(
                    seed=91_337, batch_size=25_000, logical_block_size=5_000,
                    max_workers=1, scheduler_mode="ast_only",
                    seed_strategy="philox_logical_block", memory_mode="bounded",
                    optimize=True, max_iterations=50_000_000,
                ),
                temp_dir=Path(work), tag="exp10_bounded_serial_5m_frontier_regression",
                cost_model=model,
            )
        event = result["complete_event"]
        assert event["total_iterations"] == 5_000_000
        assert event["metadata"]["use_process_pool"] is False
        assert event["metadata"]["max_pending_logical_blocks_observed"] <= 64
        assert event["metadata"]["max_branch_batch_stats_observed"] <= 64
    finally:
        monkeypatch.undo()
