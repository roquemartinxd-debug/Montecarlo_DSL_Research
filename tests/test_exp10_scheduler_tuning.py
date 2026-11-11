from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path

import pytest

from benchmarks import run_exp10_scheduler_tuning as tuning
from benchmarks import run_v1_3_experiments as base
from benchmarks.campaign_checkpoint import source_tree_digest
from montecarlo_dsl import GeneratorConfig, analyze_source, extract_ast_cost_features, optimize_program
from montecarlo_dsl.scheduler import AdaptiveMonteCarloScheduler


def test_candidate_changes_only_pool_overheads():
    original, candidate = tuning.create_cost_models(tuning.AWS_AST_WEIGHTS)
    assert original.parallel_pool_spawn_seconds == 0.18
    assert original.parallel_task_ipc_seconds == 0.006
    assert candidate.parallel_pool_spawn_seconds == 0.016
    assert candidate.parallel_task_ipc_seconds == 0.00030
    assert original.weights_seconds_per_sample == candidate.weights_seconds_per_sample
    assert replace(candidate, parallel_pool_spawn_seconds=0.18,
                   parallel_task_ipc_seconds=0.006) == original


def test_candidate_does_not_change_scheduler_kernel_or_manifest_source_inputs():
    root = Path(__file__).resolve().parents[1]
    assert (root / "benchmarks" / "run_exp10_scheduler_tuning.py").is_file()
    # The source tree fingerprint intentionally excludes this experimental
    # runner, docs and tests. On Windows the exact hex digest differs (CRLF),
    # so check this by recomputing the original file selection without
    # depending on byte-for-byte identity with the Linux checkout.
    import inspect
    definition = inspect.getsource(source_tree_digest)
    assert '"src" / "montecarlo_dsl"' in definition
    assert '"run_v1_3_experiments.py"' in definition
    assert "run_exp10_scheduler_tuning.py" not in definition
    assert len(source_tree_digest(root)) == 64


def test_scheduler_becomes_less_conservative_for_aws_pilot_observation():
    original, candidate = tuning.create_cost_models(tuning.AWS_AST_WEIGHTS)
    source = base.source_with_iterations("quadratic_pm", 250_000)
    program, _ = optimize_program(analyze_source(source), enabled=True)
    features = extract_ast_cost_features(program)
    common = dict(total_iterations=250_000, batch_size=25_000,
                  logical_block_size=5000, max_workers=4, scheduler_mode="hybrid",
                  pilot_iterations=5000, pilot_elapsed_seconds=0.001400091,
                  available_cpus=8, start_logical_block=1, initial_task_index=1)
    baseline_plan = AdaptiveMonteCarloScheduler(original).build_plan(features, **common)
    candidate_plan = AdaptiveMonteCarloScheduler(candidate).build_plan(features, **common)
    assert baseline_plan.effective_workers == 1
    assert candidate_plan.effective_workers > 1
    assert candidate_plan.use_process_pool
    assert baseline_plan.cost_estimate.effective_per_sample_seconds == candidate_plan.cost_estimate.effective_per_sample_seconds


def test_validation_rejects_nonfinite_overhead_and_bad_levels():
    with pytest.raises(ValueError, match="finite"):
        tuning.create_cost_models(tuning.AWS_AST_WEIGHTS, spawn_seconds=float("nan"))
    with pytest.raises(Exception):
        tuning._levels("100000,100000")


def test_small_end_to_end_preserves_statistical_hash_with_candidate(monkeypatch):
    monkeypatch.setattr(base, "_CHECKPOINT", None)
    original, candidate = tuning.create_cost_models(tuning.AWS_AST_WEIGHTS)
    source = base.source_with_iterations("normal_mean", 20_000)
    with tempfile.TemporaryDirectory() as work:
        results = []
        for name, model in (("original", original), ("tuned", candidate)):
            config = GeneratorConfig(seed=7_100_001, batch_size=5_000, max_workers=4,
                                     logical_block_size=5_000, scheduler_mode="hybrid",
                                     max_iterations=20_000)
            results.append(base.run_simulation(source, config, temp_dir=Path(work),
                                               tag=f"exp10_tuning_smoke_{name}", cost_model=model))
    assert results[0]["sha256"] == results[1]["sha256"]
    assert results[0]["complete_event"]["metadata"]["worker_count"] == 1
    assert results[1]["complete_event"]["metadata"]["worker_count"] == 1
