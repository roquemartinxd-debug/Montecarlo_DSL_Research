from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from montecarlo_dsl import (
    AdaptiveMonteCarloScheduler,
    GeneratorConfig,
    InterpretableASTCostModel,
    analyze_source,
    compile_source,
    extract_ast_cost_features,
    optimize_program,
)
from montecarlo_dsl.job_manager import JobManager
from montecarlo_dsl.server import create_app
from montecarlo_dsl.scheduler import build_runtime_schedule_plan


SIMPLE_MODEL = r"""\model{y = x}
\normal x{10, 2}
\iter{10000}
\response{avg,var}"""

HEAVY_PM_MODEL = r"""\model{y = \frac{-b \pm \sqrt{b^2 - 4a c}}{2a} + \sqrt{a^2 + c^2}}
\normal a{10, 0.5}
\distrib b{{10, 0.25}, {15, 0.25}, {20, 0.25}, {25, 0.25}}
\uniform c{-20, 30}
\iter{1000000}
\response{avg,var,std,p05,p50,p95}"""


def test_ast_cost_features_distinguish_model_complexity():
    simple_prog = analyze_source(SIMPLE_MODEL)
    heavy_prog = analyze_source(HEAVY_PM_MODEL)

    f_simple = extract_ast_cost_features(simple_prog)
    f_heavy = extract_ast_cost_features(heavy_prog)

    assert f_simple.num_random_variables == 1
    assert f_simple.num_normal_vars == 1
    assert f_simple.branch_count == 1
    assert f_simple.requires_quantiles is False

    assert f_heavy.num_random_variables == 3
    assert f_heavy.num_normal_vars == 1
    assert f_heavy.num_uniform_vars == 1
    assert f_heavy.num_discrete_vars == 1
    assert f_heavy.discrete_support_total == 4
    assert f_heavy.branch_count == 2
    assert f_heavy.requires_quantiles is True
    assert f_heavy.num_sqrt == 2
    assert f_heavy.weighted_operation_score > f_simple.weighted_operation_score * 3.0
    assert f_heavy.estimated_bytes_per_sample > f_simple.estimated_bytes_per_sample


def test_cost_model_calibration_modes_and_shrinkage():
    model = InterpretableASTCostModel()
    features = extract_ast_cost_features(analyze_source(HEAVY_PM_MODEL))
    static_c = model.estimate_static_per_sample_seconds(features)

    ast_est = model.calibrate_with_pilot(
        features,
        pilot_iterations=5_000,
        pilot_elapsed_seconds=0.010,
        mode="ast_only",
    )
    assert ast_est.effective_per_sample_seconds == static_c
    assert ast_est.pilot_weight_alpha == 0.0

    pilot_est = model.calibrate_with_pilot(
        features,
        pilot_iterations=5_000,
        pilot_elapsed_seconds=0.015,
        mode="pilot_only",
    )
    assert pilot_est.pilot_weight_alpha == 1.0
    assert pilot_est.effective_per_sample_seconds == pilot_est.pilot_per_sample_seconds

    hybrid_est = model.calibrate_with_pilot(
        features,
        pilot_iterations=5_000,
        pilot_elapsed_seconds=0.015,
        mode="hybrid",
    )
    assert 0.0 < hybrid_est.pilot_weight_alpha < 1.0
    low = min(static_c, pilot_est.effective_per_sample_seconds)
    high = max(static_c, pilot_est.effective_per_sample_seconds)
    assert low <= hybrid_est.effective_per_sample_seconds <= high


def test_cost_model_ridge_fitting_generalizes_and_satisfies_exact_kkt():
    f1 = extract_ast_cost_features(analyze_source(SIMPLE_MODEL))
    f2 = extract_ast_cost_features(analyze_source(HEAVY_PM_MODEL))

    observations = [
        (f1, 100_000, 0.008),
        (f1, 1_000_000, 0.075),
        (f2, 100_000, 0.045),
        (f2, 1_000_000, 0.440),
    ]
    fitted = InterpretableASTCostModel.fit_from_observations(observations, ridge_lambda=1e-2)
    assert fitted.estimate_static_per_sample_seconds(f2) > fitted.estimate_static_per_sample_seconds(f1)

    # Verificamos condiciones de optimalidad KKT (factibilidad primal, dual y holgura complementaria)
    prior_w = np.asarray(fitted.prior_weights_seconds_per_sample, dtype=np.float64)
    w_hat = np.asarray(fitted.weights_seconds_per_sample, dtype=np.float64)
    lower_w = 0.25 * prior_w
    assert np.all(w_hat >= lower_w - 1e-15)

    X = np.vstack([f.compact_feature_vector() for f, _, _ in observations])
    y = np.array(
        [
            max(1e-6, sec - (iters // 5000) * fitted.serial_task_overhead_seconds * 0.25) / iters
            for _, iters, sec in observations
        ],
        dtype=np.float64,
    )
    X_s = X * prior_w
    y_res = y - X @ prior_w
    d_hat = (w_hat - prior_w) / prior_w
    lam_eff = 1e-2 * (float(np.trace(X_s.T @ X_s)) / X.shape[1])
    grad_d = 2.0 * ((X_s.T @ X_s + lam_eff * np.eye(4)) @ d_hat - X_s.T @ y_res)

    for j in range(4):
        if d_hat[j] > -0.75 + 1e-9:
            assert abs(grad_d[j]) < 1e-12
        else:
            assert grad_d[j] >= -1e-12


def test_design_matrix_diagnostics_match_reference_training_suite():
    root = Path(__file__).resolve().parents[1]
    train_models = [
        "normal_mean",
        "uniform_mean",
        "discrete_mean",
        "uniform_linear",
        "normal_quadratic",
        "discrete_heavy",
        "radical_sum_pm",
        "heavy_expression",
    ]
    feats = []
    for m in train_models:
        p = next((root / "benchmarks" / "models").rglob(f"{m}.dsl"))
        prog, _ = optimize_program(analyze_source(p.read_text(encoding="utf-8")), enabled=True)
        feats.append(extract_ast_cost_features(prog))

    diag = InterpretableASTCostModel.design_matrix_diagnostics(feats)
    assert diag["matrix_rank"] == 4.0
    assert diag["condition_number_raw"] == pytest.approx(11.3601, abs=0.01)
    assert diag["condition_number_standardized"] == pytest.approx(3.4054, abs=0.01)
    assert diag["arith_trans_correlation"] == pytest.approx(0.8234, abs=0.01)


def test_adaptive_scheduler_selects_in_process_for_lightweight_and_parallel_for_heavy():
    scheduler = AdaptiveMonteCarloScheduler()
    f_simple = extract_ast_cost_features(analyze_source(SIMPLE_MODEL))
    f_heavy = extract_ast_cost_features(analyze_source(HEAVY_PM_MODEL))

    light_plan = scheduler.build_plan(
        f_simple,
        total_iterations=10_000,
        logical_block_size=5_000,
        max_workers=4,
        scheduler_mode="ast_only",
        available_cpus=4,
    )
    assert light_plan.effective_workers == 1
    assert light_plan.use_process_pool is False

    heavy_plan = scheduler.build_plan(
        f_heavy,
        total_iterations=1_000_000,
        logical_block_size=5_000,
        max_workers=4,
        scheduler_mode="hybrid",
        pilot_iterations=5_000,
        pilot_elapsed_seconds=0.015,
        available_cpus=4,
    )
    assert heavy_plan.effective_workers > 1
    assert heavy_plan.use_process_pool is True
    assert sum(task.iterations for task in heavy_plan.tasks) == 1_000_000


def test_python_scheduler_and_embedded_runtime_kernel_choose_identical_plan():
    model = InterpretableASTCostModel()
    features = extract_ast_cost_features(analyze_source(HEAVY_PM_MODEL))
    pilot_iterations = 5_000
    pilot_elapsed = 0.035
    plan = AdaptiveMonteCarloScheduler(model).build_plan(
        features,
        total_iterations=1_000_000,
        batch_size=25_000,
        logical_block_size=5_000,
        max_workers=4,
        scheduler_mode="hybrid",
        pilot_iterations=pilot_iterations,
        pilot_elapsed_seconds=pilot_elapsed,
        start_logical_block=1,
        initial_task_index=1,
    )
    runtime_plan = build_runtime_schedule_plan(
        total_iterations=1_000_000,
        batch_size=25_000,
        logical_block_size=5_000,
        max_workers=4,
        scheduler_mode="hybrid",
        feature_values=features.to_dict(),
        static_per_sample_seconds=model.estimate_static_per_sample_seconds(features),
        prior_equivalent_samples=model.prior_equivalent_samples,
        serial_task_overhead_seconds=model.serial_task_overhead_seconds,
        parallel_pool_spawn_seconds=model.parallel_pool_spawn_seconds,
        parallel_task_ipc_seconds=model.parallel_task_ipc_seconds,
        estimated_bytes_per_sample=features.estimated_bytes_per_sample,
        pilot_iterations=pilot_iterations,
        pilot_elapsed_seconds=pilot_elapsed,
        start_logical_block=1,
        initial_task_index=1,
    )
    assert plan.task_tuples() == list(runtime_plan["tasks"])
    assert plan.effective_workers == runtime_plan["effective_workers"]
    assert plan.use_process_pool == runtime_plan["use_process_pool"]


def test_compile_source_and_server_propagate_trained_cost_model_and_static_schedule_plan():
    custom_weights = (8.1e-8, 1.5e-8, 2.9e-8, 1.1e-8)
    trained_model = InterpretableASTCostModel(weights_seconds_per_sample=custom_weights)
    compiled = compile_source(
        HEAVY_PM_MODEL,
        GeneratorConfig(seed=101, scheduler_mode="ast_only"),
        cost_model=trained_model,
    )
    assert compiled.cost_features is not None
    assert compiled.optimization_metrics is not None
    assert compiled.static_schedule_plan is not None
    assert compiled.cost_model == trained_model
    assert f"COST_MODEL_WEIGHTS = {custom_weights!r}" in compiled.script
    assert "STATIC_TASK_GENERATION = {" in compiled.script
    assert "STATIC_SCHEDULE_TASKS = [" not in compiled.script

    manager = JobManager(batch_size=100, max_workers=1, cost_model=trained_model)
    app = create_app(manager)
    assert app.state.job_manager.cost_model == trained_model
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200


def test_schedule_plan_generates_physical_tasks_lazily_for_large_iteration_counts():
    from itertools import islice

    scheduler = AdaptiveMonteCarloScheduler(memory_budget_bytes=1)
    features = extract_ast_cost_features(analyze_source(SIMPLE_MODEL))
    plan = scheduler.build_plan(
        features,
        total_iterations=100_000_000,
        batch_size=5_000,
        logical_block_size=1,
        max_workers=4,
        scheduler_mode="ast_only",
        available_cpus=4,
    )

    assert plan.task_count > 1_000_000
    assert len(plan.tasks) == plan.task_count
    assert plan.tasks[0].task_index == 0
    assert [task.task_index for task in plan.tasks[:3]] == [0, 1, 2]
    first_tasks = list(islice(iter(plan.tasks), 3))
    assert [task.task_index for task in first_tasks] == [0, 1, 2]
    assert first_tasks[0].start_sample == 0
    assert first_tasks[-1].end_sample > first_tasks[0].end_sample


def test_runtime_schedule_kernel_returns_lazy_tasks_and_constant_size_generation_config():
    features = extract_ast_cost_features(analyze_source(SIMPLE_MODEL))
    model = InterpretableASTCostModel()
    plan = build_runtime_schedule_plan(
        total_iterations=100_000_000,
        batch_size=5_000,
        logical_block_size=1,
        max_workers=4,
        scheduler_mode="ast_only",
        feature_values=features.to_dict(),
        static_per_sample_seconds=model.estimate_static_per_sample_seconds(features),
        prior_equivalent_samples=model.prior_equivalent_samples,
        serial_task_overhead_seconds=model.serial_task_overhead_seconds,
        parallel_pool_spawn_seconds=model.parallel_pool_spawn_seconds,
        parallel_task_ipc_seconds=model.parallel_task_ipc_seconds,
        estimated_bytes_per_sample=features.estimated_bytes_per_sample,
        memory_budget_bytes=1,
        available_cpus=4,
    )

    assert plan["task_count"] > 1_000_000
    assert plan["task_generation"]["total_iterations"] == 100_000_000
    assert next(plan["tasks"])[0] == 0


def test_lazy_runtime_task_counter_matches_generated_task_stream_for_edge_cases():
    from montecarlo_dsl.scheduler import count_runtime_schedule_tasks, iter_runtime_schedule_tasks

    for total_iterations in (1, 2, 9, 10, 11, 99, 100, 101, 1_001):
        for logical_block_size in (1, 2, 7, 10, 50):
            for start_logical_block in (0, 1, 3, 10):
                for blocks_per_task in (1, 2, 3, 8):
                    for effective_workers in (1, 2, 4):
                        generation = {
                            "total_iterations": total_iterations,
                            "logical_block_size": logical_block_size,
                            "start_logical_block": start_logical_block,
                            "initial_task_index": 17,
                            "blocks_per_task": blocks_per_task,
                            "effective_workers": effective_workers,
                        }
                        generated = list(iter_runtime_schedule_tasks(**generation))
                        counted = count_runtime_schedule_tasks(
                            total_iterations=total_iterations,
                            logical_block_size=logical_block_size,
                            start_logical_block=start_logical_block,
                            blocks_per_task=blocks_per_task,
                            effective_workers=effective_workers,
                        )
                        assert counted == len(generated)
                        assert [task[0] for task in generated] == list(
                            range(17, 17 + counted)
                        )
                        if generated:
                            assert generated[0][2] == start_logical_block * logical_block_size
                            assert generated[-1][3] == total_iterations
                            assert all(
                                left[3] == right[2]
                                for left, right in zip(generated, generated[1:])
                            )


def test_fixed_schedule_plan_is_lazy_for_large_task_counts():
    from itertools import islice

    scheduler = AdaptiveMonteCarloScheduler()
    features = extract_ast_cost_features(analyze_source(SIMPLE_MODEL))
    plan = scheduler.build_plan(
        features,
        total_iterations=100_000_000,
        batch_size=1,
        logical_block_size=1,
        max_workers=4,
        scheduler_mode="fixed",
        available_cpus=4,
    )

    assert plan.task_count == 100_000_000
    first_tasks = list(islice(iter(plan.tasks), 3))
    assert [task.to_tuple() for task in first_tasks] == [
        (0, 1, 0, 1),
        (1, 1, 1, 2),
        (2, 1, 2, 3),
    ]
