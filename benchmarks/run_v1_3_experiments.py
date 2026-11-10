from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

# When invoked as ``python benchmarks/run_v1_3_experiments.py``, Python puts
# the benchmarks directory (not the project root) at sys.path[0]. Bootstrap
# the project paths before importing the sibling ``benchmarks`` package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for project_path in (ROOT, SRC):
    resolved = str(project_path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)

from benchmarks.campaign_checkpoint import (
    CampaignCheckpoint, campaign_identity, simulation_identity, _write_json_atomic,
)

import numpy as np
from scipy.stats import pearsonr, spearmanr

from montecarlo_dsl import (
    ASTCostFeatures,
    GeneratorConfig,
    InterpretableASTCostModel,
    analyze_source,
    compile_source,
    extract_ast_cost_features,
    optimize_program,
)

EVENT_PREFIX = "MC_DSL_EVENT:"
OUTPUT_DIR = ROOT / "benchmarks" / "results" / "final" / "v1.3.1"
EXPERIMENT_PROFILE = "standard"
CAMPAIGN_LABEL = "local-v1.3.1"
INSTANCE_TYPE: str | None = None
CLOUD_REGION: str | None = None


def parse_lscpu_core_socket_csv(text: str) -> int | None:
    """Return physical core count from `lscpu -p=CORE,SOCKET` output."""
    pairs: set[tuple[str, str]] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[0] and fields[1]:
            pairs.add((fields[0], fields[1]))
    return len(pairs) or None


def detect_cpu_topology() -> dict[str, Any]:
    """Best-effort physical/logical CPU inventory for experiment provenance."""
    logical = os.cpu_count()
    if sys.platform.startswith("linux"):
        try:
            proc = subprocess.run(
                ["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True,
                check=True, timeout=10,
            )
            physical = parse_lscpu_core_socket_csv(proc.stdout)
            if physical is not None:
                return {
                    "logical_cpu_count": logical,
                    "physical_core_count": physical,
                    "topology_source": "lscpu -p=CORE,SOCKET",
                }
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
            records = [block for block in cpuinfo.split("\n\n") if block.strip()]
            pairs = set()
            current: dict[str, str] = {}
            for block in records:
                current = {}
                for line in block.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        if key.strip() in {"physical id", "core id"}:
                            current[key.strip()] = value.strip()
                if "physical id" in current and "core id" in current:
                    pairs.add((current["physical id"], current["core id"]))
            if pairs:
                return {
                    "logical_cpu_count": logical,
                    "physical_core_count": len(pairs),
                    "topology_source": "/proc/cpuinfo physical id/core id",
                }
        except OSError:
            pass
    return {
        "logical_cpu_count": logical,
        "physical_core_count": None,
        "topology_source": "unavailable; see captured lscpu output",
    }


def performance_configs() -> list[tuple[str, str, int]]:
    """Comparable execution variants, including a true in-process serial baseline."""
    return [
        ("v1.3_serial_in_process", "ast_only", 1),
        ("v1.3_fixed_w1", "fixed", 1),
        ("v1.3_fixed_w2", "fixed", 2),
        ("v1.3_fixed_w4", "fixed", 4),
        ("v1.3_adaptive_hybrid", "hybrid", 4),
    ]


def memory_worker_configs(profile: str | None = None) -> list[tuple[str, str, int]]:
    """Return memory stress configurations for the selected campaign profile."""
    selected = profile or EXPERIMENT_PROFILE
    configs = [
        ("in_process_w1", "hybrid", 1),
        ("multiprocess_w2", "fixed", 2),
    ]
    if selected == "aws-4core":
        configs.append(("multiprocess_w4", "fixed", 4))
    return configs

MODELS: dict[str, Path] = {
    "normal_mean": ROOT / "benchmarks" / "models" / "analytic" / "normal_mean.dsl",
    "uniform_mean": ROOT / "benchmarks" / "models" / "analytic" / "uniform_mean.dsl",
    "discrete_mean": ROOT / "benchmarks" / "models" / "analytic" / "discrete_mean.dsl",
    "linear_combination": ROOT / "benchmarks" / "models" / "analytic" / "linear_combination.dsl",
    "analytic_pm": ROOT / "benchmarks" / "models" / "analytic" / "analytic_pm.dsl",
    "heavy_expression": ROOT / "benchmarks" / "models" / "performance" / "heavy_expression.dsl",
    "quadratic_pm": ROOT / "benchmarks" / "models" / "performance" / "quadratic_pm.dsl",
    "medium_rational": ROOT / "benchmarks" / "models" / "performance" / "medium_rational.dsl",
    "heavy_polynomial_pm": ROOT / "benchmarks" / "models" / "performance" / "heavy_polynomial_pm.dsl",
    "multi_discrete_complex": ROOT / "benchmarks" / "models" / "performance" / "multi_discrete_complex.dsl",
    "optimizable_heavy": ROOT / "benchmarks" / "models" / "performance" / "optimizable_heavy.dsl",
    "uniform_linear": ROOT / "benchmarks" / "models" / "performance" / "uniform_linear.dsl",
    "normal_quadratic": ROOT / "benchmarks" / "models" / "performance" / "normal_quadratic.dsl",
    "discrete_heavy": ROOT / "benchmarks" / "models" / "performance" / "discrete_heavy.dsl",
    "radical_sum_pm": ROOT / "benchmarks" / "models" / "performance" / "radical_sum_pm.dsl",
    "sqrt_partial_invalid": ROOT / "benchmarks" / "models" / "robustness" / "sqrt_partial_invalid.dsl",
}

ANALYTIC_SPECS: dict[str, dict[str, Any]] = {
    "normal_mean": {
        "branches": {"main": {"mean": 10.0, "variance": 4.0}},
    },
    "uniform_mean": {
        "branches": {"main": {"mean": 5.0, "variance": 100.0 / 12.0}},
    },
    "discrete_mean": {
        "branches": {"main": {"mean": 17.0, "variance": 61.0}},
    },
    "linear_combination": {
        "branches": {"main": {"mean": 29.0, "variance": 43.0}},
    },
    "analytic_pm": {
        "branches": {
            "plus": {"mean": 16.0, "variance": 16.0},
            "minus": {"mean": 4.0, "variance": 16.0},
        },
        "paired": {"mean": 12.0, "variance": 48.0},
    },
}


def git_value(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def git_is_dirty() -> bool:
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return bool(proc.stdout.strip())


def source_with_iterations(model_name: str, iterations: int) -> str:
    text = MODELS[model_name].read_text(encoding="utf-8")
    replaced, count = re.subn(r"\\iter\{\d+\}", f"\\\\iter{{{iterations}}}", text)
    if count != 1:
        raise RuntimeError(f"Expected 1 \\iter in {model_name}, got {count}")
    return replaced


def canonical_result(complete: dict[str, Any]) -> dict[str, Any]:
    return {
        "processed_iterations": complete.get("processed_iterations"),
        "total_iterations": complete.get("total_iterations"),
        "candidate_results": complete.get("candidate_results"),
        "valid_results": complete.get("valid_results"),
        "discarded_results": complete.get("discarded_results"),
        "branches": complete.get("branches"),
        "paired": complete.get("paired"),
        "stats": complete.get("stats"),
        "histogram": complete.get("histogram"),
        "diagnostics": complete.get("diagnostics"),
    }


def result_sha256(complete: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_result(complete),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_MODULE_COUNTER = 0
_CHECKPOINT: CampaignCheckpoint | None = None


def _checkpoint_identity(source: str, config: GeneratorConfig, *, tag: str, kind: str,
                         cost_model: InterpretableASTCostModel | None) -> dict[str, Any] | None:
    if _CHECKPOINT is None:
        return None
    return simulation_identity(tag=tag, kind=kind, source=source,
                               config=config, cost_model=cost_model)



def run_simulation(
    source: str,
    config: GeneratorConfig,
    *,
    temp_dir: Path,
    tag: str = "sim",
    cost_model: InterpretableASTCostModel | None = None,
) -> dict[str, Any]:
    record_id = _checkpoint_identity(source, config, tag=tag, kind="simulation", cost_model=cost_model)
    is_warmup = tag.startswith(("warmup_", "w0_"))
    if _CHECKPOINT is not None:
        if is_warmup:
            _CHECKPOINT.mark_warmup(tag)
        elif record_id is not None:
            cached = _CHECKPOINT.load(record_id)
            if cached is not None:
                # Return the ORIGINAL compile metrics and script size, not a fresh
                # compilation whose timing-dependent contents might differ.
                return cached

    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    mod_name = f"mcdsl_gen_{_MODULE_COUNTER}"

    t_compile_start = time.perf_counter()
    compiled = compile_source(source, config, cost_model=cost_model)
    compile_seconds = time.perf_counter() - t_compile_start

    script_path = temp_dir / f"{mod_name}.py"
    script_path.write_text(compiled.script, encoding="utf-8")

    # Windows uses the "spawn" start method. ProcessPoolExecutor must be
    # able to import _simulate_batch from this dynamically generated module
    # by its real module name when unpickling a task in a child interpreter.
    # Registering it in the parent's sys.modules is not sufficient: the
    # temporary script's directory also has to be importable by children.
    module_dir = str(script_path.parent.resolve())
    sys.path.insert(0, module_dir)
    importlib.invalidate_caches()
    buf = io.StringIO()
    try:
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)

        t_exec_start = time.perf_counter()
        with redirect_stdout(buf):
            rc = module.main()
        exec_seconds = time.perf_counter() - t_exec_start
    finally:
        sys.modules.pop(mod_name, None)
        sys.path.remove(module_dir)
        importlib.invalidate_caches()
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    if rc != 0:
        raise RuntimeError(f"Generated simulation failed with code {rc}: {buf.getvalue()}")

    complete = None
    start_evt = None
    for line in buf.getvalue().splitlines():
        if line.startswith(EVENT_PREFIX):
            evt = json.loads(line[len(EVENT_PREFIX):])
            if evt.get("type") == "start":
                start_evt = evt
            elif evt.get("type") == "complete":
                complete = evt
    if complete is None:
        raise RuntimeError("Simulation did not emit complete event")

    result = {
        "compiled": compiled,
        "compiled_metrics": compiled.optimization_metrics.to_dict(),
        "script_bytes": len(compiled.script.encode("utf-8")),
        "start_event": start_evt,
        "complete_event": complete,
        "compile_seconds": compile_seconds,
        "execution_seconds": exec_seconds,
        "total_seconds": compile_seconds + exec_seconds,
        "sha256": result_sha256(complete),
    }
    if _CHECKPOINT is not None and not is_warmup and record_id is not None:
        _CHECKPOINT.save(record_id, {key: val for key, val in result.items() if key != "compiled"})
    return result


def bootstrap_ci95(
    values: list[float],
    stat_fn=np.median,
    reps: int = 10_000,
    seed: int = 120026,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        val = float(arr[0]) if arr.size == 1 else 0.0
        return val, val
    rng = np.random.default_rng(seed)
    n = arr.size
    samples = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        samples[i] = stat_fn(arr[idx])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # The final stage CSV is itself committed atomically: an interrupted write
    # never leaves a truncated or partially new published table on disk.
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=".csv_", suffix=".tmp", delete=False,
        ) as fh:
            tmp_name = fh.name
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)


def run_exp5_ast_cost_generalization(
    temp_dir: Path,
    *,
    reps: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], InterpretableASTCostModel]:
    print("=== Experimento 5: Modelo de Coste del AST (4 Indices Compuestos Interpretables, 8 Train, 6 Test, 10 Reps) ===")
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
    test_models = [
        "linear_combination",
        "analytic_pm",
        "quadratic_pm",
        "medium_rational",
        "heavy_polynomial_pm",
        "multi_discrete_complex",
    ]
    iterations = 500_000

    train_observations = []
    train_features_list = []
    measured_data: dict[str, dict[str, Any]] = {}

    for model_name in train_models + test_models:
        source = source_with_iterations(model_name, iterations)
        prog, _ = optimize_program(analyze_source(source), enabled=True)
        features = extract_ast_cost_features(prog)
        if model_name in train_models:
            train_features_list.append(features)

        times = []
        pilot_times = []
        for rep in range(reps):
            res = run_simulation(
                source,
                GeneratorConfig(
                    seed=200 + rep,
                    batch_size=50_000,
                    logical_block_size=5_000,
                    max_workers=1,
                    scheduler_mode="hybrid",
                ),
                temp_dir=temp_dir,
                tag=f"exp5_{model_name}_r{rep}",
            )
            meta = res["complete_event"]["metadata"]
            times.append(float(meta.get("simulation_compute_seconds") or res["execution_seconds"]))
            pilot_times.append(float(meta.get("pilot_elapsed_seconds") or 0.001))

        med_exec = float(np.median(times))
        med_pilot = float(np.median(pilot_times))
        measured_data[model_name] = {
            "features": features,
            "runs": reps,
            "median_exec_seconds": med_exec,
            "observed_per_sample_ns": (med_exec / iterations) * 1e9,
            "median_pilot_seconds": med_pilot,
            "split": "train" if model_name in train_models else "test",
        }
        if model_name in train_models:
            train_observations.append((features, iterations, med_exec))

    fitted_model = InterpretableASTCostModel.fit_from_observations(train_observations)
    design_diag = InterpretableASTCostModel.design_matrix_diagnostics(train_features_list)

    rows = []
    for model_name, info in measured_data.items():
        feat: ASTCostFeatures = info["features"]
        obs_per_sample = info["median_exec_seconds"] / iterations
        static_est = fitted_model.estimate_static_per_sample_seconds(feat)
        hybrid_est = fitted_model.calibrate_with_pilot(
            feat,
            pilot_iterations=5_000,
            pilot_elapsed_seconds=info["median_pilot_seconds"],
            mode="hybrid",
        ).effective_per_sample_seconds

        static_rel_err = abs(static_est - obs_per_sample) / obs_per_sample
        hybrid_rel_err = abs(hybrid_est - obs_per_sample) / obs_per_sample

        rows.append({
            "model": model_name,
            "split": info["split"],
            "runs": info["runs"],
            "iterations": iterations,
            "num_random_vars": feat.num_random_variables,
            "branch_count": feat.branch_count,
            "ast_node_count": feat.ast_node_count,
            "weighted_op_score": round(feat.weighted_operation_score, 4),
            "observed_per_sample_ns": round(obs_per_sample * 1e9, 3),
            "static_ast_predicted_ns": round(static_est * 1e9, 3),
            "hybrid_calibrated_predicted_ns": round(hybrid_est * 1e9, 3),
            "static_relative_error_pct": round(static_rel_err * 100.0, 2),
            "hybrid_relative_error_pct": round(hybrid_rel_err * 100.0, 2),
        })

    summary_rows = []
    for split_name in ("train", "test"):
        sub = [r for r in rows if r["split"] == split_name]
        obs_arr = np.array([float(r["observed_per_sample_ns"]) for r in sub], dtype=np.float64)
        stat_arr = np.array([float(r["static_ast_predicted_ns"]) for r in sub], dtype=np.float64)
        hyb_arr = np.array([float(r["hybrid_calibrated_predicted_ns"]) for r in sub], dtype=np.float64)
        stat_err = np.array([float(r["static_relative_error_pct"]) for r in sub], dtype=np.float64)
        hyb_err = np.array([float(r["hybrid_relative_error_pct"]) for r in sub], dtype=np.float64)

        stat_rmse = float(np.sqrt(np.mean((stat_arr - obs_arr) ** 2)))
        hyb_rmse = float(np.sqrt(np.mean((hyb_arr - obs_arr) ** 2)))
        stat_r, _ = pearsonr(obs_arr, stat_arr)
        hyb_r, _ = pearsonr(obs_arr, hyb_arr)
        stat_rho, _ = spearmanr(obs_arr, stat_arr)
        hyb_rho, _ = spearmanr(obs_arr, hyb_arr)

        summary_rows.append({
            "split": split_name,
            "models_count": len(sub),
            "parameters_fitted": 4,
            "runs_per_model": reps,
            "matrix_rank": int(design_diag["matrix_rank"]),
            "cond_raw": design_diag["condition_number_raw"],
            "cond_standardized": design_diag["condition_number_standardized"],
            "arith_trans_corr": design_diag["arith_trans_correlation"],
            "fitted_theta_ns": ";".join(f"{w * 1e9:.3f}" for w in fitted_model.weights_seconds_per_sample),
            "static_mape_pct": round(float(np.mean(stat_err)), 2),
            "static_median_rel_error_pct": round(float(np.median(stat_err)), 2),
            "static_rmse_ns": round(stat_rmse, 2),
            "static_pearson_r": round(float(stat_r), 4),
            "static_spearman_rho": round(float(stat_rho), 4),
            "hybrid_mape_pct": round(float(np.mean(hyb_err)), 2),
            "hybrid_median_rel_error_pct": round(float(np.median(hyb_err)), 2),
            "hybrid_rmse_ns": round(hyb_rmse, 2),
            "hybrid_pearson_r": round(float(hyb_r), 4),
            "hybrid_spearman_rho": round(float(hyb_rho), 4),
        })

    write_csv(OUTPUT_DIR / "ast_cost_generalization.csv", rows)
    write_csv(OUTPUT_DIR / "ast_cost_generalization_summary.csv", summary_rows)
    return rows, summary_rows, fitted_model


def run_exp0_v1_2_vs_v1_3_baseline_comparison(
    temp_dir: Path,
    *,
    reps: int = 30,
    cost_model: InterpretableASTCostModel | None = None,
) -> list[dict[str, Any]]:
    print("=== Experimento 0: Runtime v1.3.1 vs emulacion de configuracion historica v1.2.0 ===")
    models = ["normal_mean", "linear_combination", "heavy_expression", "quadratic_pm"]
    iteration_levels = [100_000, 1_000_000]
    variants = [
        (
            "v1.2.0_legacy_configuration_emulation",
            {
                "scheduler_mode": "fixed",
                "seed_strategy": "batch_index",
                "memory_mode": "retain_all",
                "optimize": False,
                "max_workers": 2,
            },
        ),
        (
            "v1.3.1_fixed_w2",
            {
                "scheduler_mode": "fixed",
                "seed_strategy": "philox_logical_block",
                "memory_mode": "bounded",
                "optimize": True,
                "max_workers": 2,
            },
        ),
        (
            "v1.3.1_adaptive_hybrid",
            {
                "scheduler_mode": "hybrid",
                "seed_strategy": "philox_logical_block",
                "memory_mode": "bounded",
                "optimize": True,
                "max_workers": 4,
            },
        ),
    ]

    summary_rows = []
    for model_name in models:
        for iterations in iteration_levels:
            source = source_with_iterations(model_name, iterations)
            times_by_var: dict[str, list[float]] = {v[0]: [] for v in variants}
            ret_mem_by_var: dict[str, list[int]] = {v[0]: [] for v in variants}
            rss_mem_by_var: dict[str, list[int]] = {v[0]: [] for v in variants}
            meta_by_var: dict[str, dict[str, Any]] = {}

            for v_name, kwargs in variants:
                run_simulation(
                    source,
                    GeneratorConfig(seed=101, batch_size=25_000, logical_block_size=5_000, **kwargs),
                    temp_dir=temp_dir,
                    tag=f"w0_{model_name}_{iterations}_{v_name}",
                    cost_model=cost_model,
                )

            for r in range(1, reps + 1):
                for v_name, kwargs in variants:
                    res = run_simulation(
                        source,
                        GeneratorConfig(seed=100 + r, batch_size=25_000, logical_block_size=5_000, **kwargs),
                        temp_dir=temp_dir,
                        tag=f"exp0_{model_name}_{iterations}_{v_name}_r{r}",
                        cost_model=cost_model,
                    )
                    meta = res["complete_event"]["metadata"]
                    times_by_var[v_name].append(res["total_seconds"])
                    ret_mem_by_var[v_name].append(int(meta.get("peak_retained_coordinator_bytes", meta.get("peak_memory_bytes", 0))))
                    rss_mem_by_var[v_name].append(
                        int(meta.get("sum_of_process_peak_rss_bytes", meta.get("aggregate_peak_rss_bytes", 0)))
                    )
                    meta_by_var[v_name] = meta

            v12_times = times_by_var["v1.2.0_legacy_configuration_emulation"]
            v12_ret_mem = float(np.median(ret_mem_by_var["v1.2.0_legacy_configuration_emulation"]))

            for v_name, kwargs in variants:
                times = times_by_var[v_name]
                ret_mems = ret_mem_by_var[v_name]
                rss_mems = rss_mem_by_var[v_name]
                meta = meta_by_var[v_name]
                ratios = [t12 / t_cur for t12, t_cur in zip(v12_times, times)]
                ci_l, ci_h = bootstrap_ci95(ratios, np.median)
                med_ret_mem = float(np.median(ret_mems))
                med_rss_mem = float(np.median(rss_mems))
                summary_rows.append({
                    "model": model_name,
                    "iterations": iterations,
                    "baseline_reference": "v1.2.0_legacy_configuration_emulation",
                    "implementation": v_name,
                    "runs": reps,
                    "effective_workers": meta["worker_count"],
                    "batch_count": meta["batch_count"],
                    "median_seconds": round(float(np.median(times)), 5),
                    "iqr_seconds": round(float(np.percentile(times, 75) - np.percentile(times, 25)), 5),
                    "mean_seconds": round(float(np.mean(times)), 5),
                    "std_seconds": round(float(np.std(times, ddof=1)), 5),
                    "median_speedup_vs_v1_2_0": round(float(np.median(ratios)), 4),
                    "speedup_vs_v1_2_0_ci95_low": round(ci_l, 4),
                    "speedup_vs_v1_2_0_ci95_high": round(ci_h, 4),
                    "median_retained_payload_kb": round(med_ret_mem / 1024.0, 2),
                    "median_peak_memory_kb": round(med_ret_mem / 1024.0, 2),
                    "median_sum_of_process_peak_rss_mb": round(med_rss_mem / (1024.0 * 1024.0), 2),
                    "retained_payload_reduction_vs_v1_2_0_pct": round((1.0 - med_ret_mem / max(1.0, v12_ret_mem)) * 100.0, 2),
                    "memory_reduction_vs_v1_2_0_pct": round((1.0 - med_ret_mem / max(1.0, v12_ret_mem)) * 100.0, 2),
                })

    write_csv(OUTPUT_DIR / "v1_2_vs_v1_3_baseline_comparison.csv", summary_rows)
    return summary_rows


def run_exp1_functional_equivalence(
    temp_dir: Path,
    *,
    cost_model: InterpretableASTCostModel | None = None,
) -> list[dict[str, Any]]:
    print("=== Experimento 1: Equivalencia Funcional y Reproducibilidad entre Modos de v1.3.1 ===")
    models = [
        "normal_mean",
        "uniform_mean",
        "discrete_mean",
        "linear_combination",
        "analytic_pm",
        "heavy_expression",
        "quadratic_pm",
        "sqrt_partial_invalid",
    ]
    modes = ("fixed", "ast_only", "pilot_only", "hybrid")
    rows = []
    for model_name in models:
        source = source_with_iterations(model_name, 100_000)
        ref_hash = None
        for mode in modes:
            res = run_simulation(
                source,
                GeneratorConfig(
                    seed=101,
                    batch_size=25_000,
                    logical_block_size=5_000,
                    max_workers=2,
                    scheduler_mode=mode,
                ),
                temp_dir=temp_dir,
                tag=f"exp1_{model_name}_{mode}",
                cost_model=cost_model,
            )
            complete = res["complete_event"]
            meta = complete["metadata"]
            if ref_hash is None:
                ref_hash = res["sha256"]
            rows.append({
                "model": model_name,
                "scheduler_mode": mode,
                "iterations": 100_000,
                "seed": 101,
                "has_plus_minus": len(complete["branches"]) == 2,
                "worker_count": meta["worker_count"],
                "batch_count": meta["batch_count"],
                "use_process_pool": meta.get("use_process_pool"),
                "valid_results": complete["valid_results"],
                "discarded_results": complete["discarded_results"],
                "result_sha256": res["sha256"],
                "matches_fixed_baseline": res["sha256"] == ref_hash,
                "total_seconds": round(res["total_seconds"], 6),
            })
    write_csv(OUTPUT_DIR / "functional_equivalence.csv", rows)
    return rows


def run_exp2_performance_and_scalability(
    temp_dir: Path,
    *,
    confirmatory_reps: int = 30,
    pilot_reps: int = 10,
    cost_model: InterpretableASTCostModel | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("=== Experimento 2: Rendimiento y Escalabilidad entre Configuraciones de v1.3.1 (30 confirmatorias / 10 piloto) ===")
    raw_rows = []
    summary_rows = []

    models = ["normal_mean", "linear_combination", "heavy_expression", "quadratic_pm"]
    iteration_levels = [10_000, 100_000, 1_000_000]
    configs = performance_configs()

    for model_name in models:
        for iterations in iteration_levels:
            source = source_with_iterations(model_name, iterations)
            reps = (
                confirmatory_reps
                if (model_name in {"normal_mean", "heavy_expression"} and iterations in {100_000, 1_000_000})
                else pilot_reps
            )

            for label, mode, w in configs:
                run_simulation(
                    source,
                    GeneratorConfig(
                        seed=101,
                        batch_size=25_000,
                        logical_block_size=5_000,
                        max_workers=w,
                        scheduler_mode=mode,
                    ),
                    temp_dir=temp_dir,
                    tag=f"warmup_{model_name}_{iterations}_{label}",
                    cost_model=cost_model,
                )

            per_config_times: dict[str, list[float]] = {label: [] for label, _, _ in configs}
            per_config_mem: dict[str, list[int]] = {label: [] for label, _, _ in configs}
            per_config_rss: dict[str, list[int]] = {label: [] for label, _, _ in configs}
            per_config_meta: dict[str, dict[str, Any]] = {}

            for rep in range(1, reps + 1):
                for label, mode, w in configs:
                    res = run_simulation(
                        source,
                        GeneratorConfig(
                            seed=100 + rep,
                            batch_size=25_000,
                            logical_block_size=5_000,
                            max_workers=w,
                            scheduler_mode=mode,
                        ),
                        temp_dir=temp_dir,
                        tag=f"exp2_{model_name}_{iterations}_{label}_r{rep}",
                        cost_model=cost_model,
                    )
                    meta = res["complete_event"]["metadata"]
                    ret_b = int(meta.get("peak_retained_coordinator_bytes", meta.get("peak_memory_bytes", 0)))
                    rss_b = int(meta.get("sum_of_process_peak_rss_bytes", meta.get("aggregate_peak_rss_bytes", 0)))
                    per_config_times[label].append(res["total_seconds"])
                    per_config_mem[label].append(ret_b)
                    per_config_rss[label].append(rss_b)
                    per_config_meta[label] = meta
                    raw_rows.append({
                        "model": model_name,
                        "iterations": iterations,
                        "configuration": label,
                        "scheduler_mode": mode,
                        "requested_workers": w,
                        "effective_workers": meta["worker_count"],
                        "batch_count": meta["batch_count"],
                        "use_process_pool": meta.get("use_process_pool"),
                        "repetition": rep,
                        "compile_seconds": res["compile_seconds"],
                        "execution_seconds": res["execution_seconds"],
                        "total_seconds": res["total_seconds"],
                        "retained_coordinator_payload_bytes": ret_b,
                        "peak_memory_bytes": ret_b,
                        "sum_of_process_peak_rss_bytes": rss_b,
                        "result_sha256": res["sha256"],
                    })

            serial_in_process_times = per_config_times["v1.3_serial_in_process"]
            fixed_w1_times = per_config_times["v1.3_fixed_w1"]
            fixed_w2_times = per_config_times["v1.3_fixed_w2"]

            for label, mode, w in configs:
                times = per_config_times[label]
                mems = per_config_mem[label]
                rss_list = per_config_rss[label]
                meta = per_config_meta[label]
                paired_speedups_vs_serial = [t_serial / t_cur for t_serial, t_cur in zip(serial_in_process_times, times)]
                paired_speedups_vs_w1 = [t_w1 / t_cur for t_w1, t_cur in zip(fixed_w1_times, times)]
                paired_speedups_vs_w2 = [t_w2 / t_cur for t_w2, t_cur in zip(fixed_w2_times, times)]
                ci_serial_low, ci_serial_high = bootstrap_ci95(paired_speedups_vs_serial, np.median)
                ci_low, ci_high = bootstrap_ci95(paired_speedups_vs_w1, np.median)
                ci_w2_low, ci_w2_high = bootstrap_ci95(paired_speedups_vs_w2, np.median)
                med = float(np.median(times))
                q1 = float(np.percentile(times, 25))
                q3 = float(np.percentile(times, 75))
                eff_w = int(meta["worker_count"])
                med_speedup = float(np.median(paired_speedups_vs_w1))
                summary_rows.append({
                    "model": model_name,
                    "iterations": iterations,
                    "configuration": label,
                    "scheduler_mode": mode,
                    "requested_workers": w,
                    "effective_workers": eff_w,
                    "batch_count": meta["batch_count"],
                    "use_process_pool": meta.get("use_process_pool"),
                    "runs": reps,
                    "median_seconds": med,
                    "mean_seconds": float(np.mean(times)),
                    "std_seconds": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                    "iqr_seconds": q3 - q1,
                    "min_seconds": float(np.min(times)),
                    "max_seconds": float(np.max(times)),
                    "median_speedup_vs_serial_in_process": float(np.median(paired_speedups_vs_serial)),
                    "speedup_vs_serial_ci95_low": ci_serial_low,
                    "speedup_vs_serial_ci95_high": ci_serial_high,
                    "median_speedup_vs_fixed_w1": med_speedup,
                    "speedup_vs_w1_ci95_low": ci_low,
                    "speedup_vs_w1_ci95_high": ci_high,
                    "median_speedup_vs_fixed_w2": float(np.median(paired_speedups_vs_w2)),
                    "speedup_vs_w2_ci95_low": ci_w2_low,
                    "speedup_vs_w2_ci95_high": ci_w2_high,
                    "parallel_efficiency": (
                        float(np.median(paired_speedups_vs_serial)) / max(1, eff_w)
                        if bool(meta.get("use_process_pool")) else ""
                    ),
                    "parallel_efficiency_reference": "serial_in_process",
                    "median_retained_coordinator_bytes": int(np.median(mems)),
                    "median_peak_memory_bytes": int(np.median(mems)),
                    "median_sum_of_process_peak_rss_bytes": int(np.median(rss_list)),
                })

    # Escalabilidad a 10^7 iteraciones (10 repeticiones por configuración).
    # Include serial in-process, fixed process pools at 1/2/4 workers, and adaptive.
    source_10m = source_with_iterations("normal_mean", 10_000_000)
    ten_m_configs = [
        ("v1.3_serial_in_process", "ast_only", 1),
        ("v1.3_fixed_w1", "fixed", 1),
        ("v1.3_fixed_w2", "fixed", 2),
        ("v1.3_fixed_w4", "fixed", 4),
        ("v1.3_adaptive_hybrid", "hybrid", 4),
    ]
    ten_m_data: dict[str, dict[str, Any]] = {}
    for label, mode, w in ten_m_configs:
        times_10m: list[float] = []
        mems_10m: list[int] = []
        rss_10m: list[int] = []
        meta_10m: dict[str, Any] = {}
        for rep in range(1, pilot_reps + 1):
            res = run_simulation(
                source_10m,
                GeneratorConfig(
                    seed=100 + rep,
                    batch_size=250_000,
                    logical_block_size=50_000,
                    max_workers=w,
                    scheduler_mode=mode,
                    max_iterations=10_000_000,
                ),
                temp_dir=temp_dir,
                tag=f"exp2_10m_{label}_r{rep}",
                cost_model=cost_model,
            )
            meta_10m = res["complete_event"]["metadata"]
            ret_b = int(meta_10m.get("peak_retained_coordinator_bytes", meta_10m.get("peak_memory_bytes", 0)))
            rss_b = int(meta_10m.get("sum_of_process_peak_rss_bytes", meta_10m.get("aggregate_peak_rss_bytes", 0)))
            times_10m.append(res["total_seconds"])
            mems_10m.append(ret_b)
            rss_10m.append(rss_b)
            raw_rows.append({
                "model": "normal_mean",
                "iterations": 10_000_000,
                "configuration": label,
                "scheduler_mode": mode,
                "requested_workers": w,
                "effective_workers": meta_10m["worker_count"],
                "batch_count": meta_10m["batch_count"],
                "use_process_pool": meta_10m.get("use_process_pool"),
                "repetition": rep,
                "compile_seconds": res["compile_seconds"],
                "execution_seconds": res["execution_seconds"],
                "total_seconds": res["total_seconds"],
                "retained_coordinator_payload_bytes": ret_b,
                "peak_memory_bytes": ret_b,
                "sum_of_process_peak_rss_bytes": rss_b,
                "result_sha256": res["sha256"],
            })
        ten_m_data[label] = {
            "times": times_10m, "memory": mems_10m, "rss": rss_10m, "metadata": meta_10m,
            "mode": mode, "requested_workers": w,
        }

    serial_10m = ten_m_data["v1.3_serial_in_process"]["times"]
    fixed_w1_10m = ten_m_data["v1.3_fixed_w1"]["times"]
    fixed_w2_10m = ten_m_data["v1.3_fixed_w2"]["times"]
    for label, mode, w in ten_m_configs:
        data = ten_m_data[label]
        times_10m = data["times"]
        meta_10m = data["metadata"]
        serial_sp = [a / b for a, b in zip(serial_10m, times_10m)]
        w1_sp = [a / b for a, b in zip(fixed_w1_10m, times_10m)]
        w2_sp = [a / b for a, b in zip(fixed_w2_10m, times_10m)]
        serial_ci = bootstrap_ci95(serial_sp, np.median)
        w1_ci = bootstrap_ci95(w1_sp, np.median)
        w2_ci = bootstrap_ci95(w2_sp, np.median)
        summary_rows.append({
            "model": "normal_mean",
            "iterations": 10_000_000,
            "configuration": label,
            "scheduler_mode": mode,
            "requested_workers": w,
            "effective_workers": meta_10m["worker_count"],
            "batch_count": meta_10m["batch_count"],
            "use_process_pool": meta_10m.get("use_process_pool"),
            "runs": len(times_10m),
            "median_seconds": float(np.median(times_10m)),
            "mean_seconds": float(np.mean(times_10m)),
            "std_seconds": float(np.std(times_10m, ddof=1)),
            "iqr_seconds": float(np.percentile(times_10m, 75) - np.percentile(times_10m, 25)),
            "min_seconds": float(np.min(times_10m)),
            "max_seconds": float(np.max(times_10m)),
            "median_speedup_vs_serial_in_process": float(np.median(serial_sp)),
            "speedup_vs_serial_ci95_low": serial_ci[0],
            "speedup_vs_serial_ci95_high": serial_ci[1],
            "median_speedup_vs_fixed_w1": float(np.median(w1_sp)),
            "speedup_vs_w1_ci95_low": w1_ci[0],
            "speedup_vs_w1_ci95_high": w1_ci[1],
            "median_speedup_vs_fixed_w2": float(np.median(w2_sp)),
            "speedup_vs_w2_ci95_low": w2_ci[0],
            "speedup_vs_w2_ci95_high": w2_ci[1],
            "parallel_efficiency": (
                float(np.median(serial_sp)) / max(1, int(meta_10m["worker_count"]))
                if bool(meta_10m.get("use_process_pool")) else ""
            ),
            "parallel_efficiency_reference": "serial_in_process",
            "median_retained_coordinator_bytes": int(np.median(data["memory"])),
            "median_peak_memory_bytes": int(np.median(data["memory"])),
            "median_sum_of_process_peak_rss_bytes": int(np.median(data["rss"])),
        })

    write_csv(OUTPUT_DIR / "performance_comparison_raw.csv", raw_rows)
    write_csv(OUTPUT_DIR / "performance_comparison_summary.csv", summary_rows)
    return raw_rows, summary_rows


def run_exp3_batch_adaptation(
    temp_dir: Path,
    *,
    reps: int = 10,
    cost_model: InterpretableASTCostModel | None = None,
) -> list[dict[str, Any]]:
    print("=== Experimento 3: Adaptacion de Lotes (Tamanos Fisicos Fijos vs Agrupacion Adaptativa) ===")
    models = ["normal_mean", "heavy_expression", "quadratic_pm"]
    fixed_sizes = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
    iterations = 1_000_000
    rows = []

    for model_name in models:
        source = source_with_iterations(model_name, iterations)
        hashes = set()
        for b_size in fixed_sizes:
            times = []
            meta = {}
            sha = ""
            for rep in range(reps):
                res = run_simulation(
                    source,
                    GeneratorConfig(
                        seed=101,
                        batch_size=b_size,
                        logical_block_size=5_000,
                        max_workers=2,
                        scheduler_mode="fixed",
                    ),
                    temp_dir=temp_dir,
                    tag=f"exp3_{model_name}_b{b_size}_r{rep}",
                    cost_model=cost_model,
                )
                times.append(res["total_seconds"])
                meta = res["complete_event"]["metadata"]
                sha = res["sha256"]
                hashes.add(sha)
            rows.append({
                "model": model_name,
                "iterations": iterations,
                "strategy": f"fixed_{b_size}",
                "runs": reps,
                "physical_batch_size": b_size,
                "batch_count": meta["batch_count"],
                "worker_count": meta["worker_count"],
                "median_seconds": float(np.median(times)),
                "mean_seconds": float(np.mean(times)),
                "std_seconds": float(np.std(times, ddof=1)),
                "result_sha256": sha,
            })

        adapt_times = []
        adapt_meta = {}
        adapt_sha = ""
        for rep in range(reps):
            res = run_simulation(
                source,
                GeneratorConfig(
                    seed=101,
                    batch_size=25_000,
                    logical_block_size=5_000,
                    max_workers=2,
                    scheduler_mode="hybrid",
                ),
                temp_dir=temp_dir,
                tag=f"exp3_{model_name}_hybrid_r{rep}",
                cost_model=cost_model,
            )
            adapt_times.append(res["total_seconds"])
            adapt_meta = res["complete_event"]["metadata"]
            adapt_sha = res["sha256"]
            hashes.add(adapt_sha)

        assert len(hashes) == 1, f"All batch sizes must produce identical hash for {model_name}"
        rows.append({
            "model": model_name,
            "iterations": iterations,
            "strategy": "adaptive_hybrid",
            "runs": reps,
            "physical_batch_size": iterations // max(1, adapt_meta["batch_count"]),
            "batch_count": adapt_meta["batch_count"],
            "worker_count": adapt_meta["worker_count"],
            "median_seconds": float(np.median(adapt_times)),
            "mean_seconds": float(np.mean(adapt_times)),
            "std_seconds": float(np.std(adapt_times, ddof=1)),
            "result_sha256": adapt_sha,
        })

    write_csv(OUTPUT_DIR / "batch_adaptation_summary.csv", rows)
    return rows


def run_exp4_schedule_invariance_reproducibility(
    temp_dir: Path,
    *,
    cost_model: InterpretableASTCostModel | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    print("=== Experimento 4: Reproducibilidad Independiente de la Planificacion Fisica ===")
    models = ["normal_mean", "uniform_mean", "discrete_mean", "linear_combination", "analytic_pm", "quadratic_pm"]
    iterations = 1_000_000
    raw_rows = []
    summary_rows = []

    schedule_variations = [
        ("fixed_b10k_w1_asc", "fixed", 10_000, 1, "as_completed"),
        ("fixed_b20k_w1_asc", "fixed", 20_000, 1, "as_completed"),
        ("fixed_b25k_w1_asc", "fixed", 25_000, 1, "as_completed"),
        ("fixed_b25k_w2_asc", "fixed", 25_000, 2, "as_completed"),
        ("fixed_b25k_w4_asc", "fixed", 25_000, 4, "as_completed"),
        ("fixed_b50k_w2_asc", "fixed", 50_000, 2, "as_completed"),
        ("fixed_b100k_w4_asc", "fixed", 100_000, 4, "as_completed"),
        ("fixed_b25k_w2_rev", "fixed", 25_000, 2, "reverse"),
        ("fixed_b25k_w2_shuf", "fixed", 25_000, 2, "shuffled"),
        ("adaptive_ast_only", "ast_only", 25_000, 4, "as_completed"),
        ("adaptive_pilot_only", "pilot_only", 25_000, 4, "as_completed"),
        ("adaptive_hybrid", "hybrid", 25_000, 4, "as_completed"),
        ("adaptive_hybrid_rev", "hybrid", 25_000, 4, "reverse"),
        ("adaptive_hybrid_shuf", "hybrid", 25_000, 4, "shuffled"),
    ]

    for model_name in models:
        source = source_with_iterations(model_name, iterations)
        ref_hash = None
        model_hashes = set()
        for label, mode, b_size, w, order in schedule_variations:
            res = run_simulation(
                source,
                GeneratorConfig(
                    seed=101,
                    batch_size=b_size,
                    logical_block_size=5_000,
                    max_workers=w,
                    scheduler_mode=mode,
                    task_completion_order=order,
                ),
                temp_dir=temp_dir,
                tag=f"exp4_{model_name}_{label}",
                cost_model=cost_model,
            )
            sha = res["sha256"]
            if ref_hash is None:
                ref_hash = sha
            model_hashes.add(sha)
            meta = res["complete_event"]["metadata"]
            raw_rows.append({
                "model": model_name,
                "variation": label,
                "scheduler_mode": mode,
                "physical_batch_size": b_size,
                "logical_block_size": 5_000,
                "requested_workers": w,
                "effective_workers": meta["worker_count"],
                "completion_order": order,
                "batch_count": meta["batch_count"],
                "elapsed_seconds": round(res["total_seconds"], 6),
                "result_sha256": sha,
                "matches_reference": sha == ref_hash,
            })

        summary_rows.append({
            "model": model_name,
            "seed": 101,
            "iterations": iterations,
            "logical_block_size": 5_000,
            "schedules_tested": len(schedule_variations),
            "unique_result_hashes": len(model_hashes),
            "exact_schedule_invariance": len(model_hashes) == 1,
            "reference_sha256": ref_hash,
        })

    control_rows = []
    ctrl_source = source_with_iterations("normal_mean", iterations)
    h_seed101 = run_simulation(
        ctrl_source,
        GeneratorConfig(seed=101, batch_size=25_000, logical_block_size=5_000, scheduler_mode="hybrid"),
        temp_dir=temp_dir,
        tag="ctrl_seed101",
        cost_model=cost_model,
    )["sha256"]
    h_seed102 = run_simulation(
        ctrl_source,
        GeneratorConfig(seed=102, batch_size=25_000, logical_block_size=5_000, scheduler_mode="hybrid"),
        temp_dir=temp_dir,
        tag="ctrl_seed102",
        cost_model=cost_model,
    )["sha256"]
    control_rows.append({
        "control": "seed_sensitivity",
        "description": "Seeds 101 vs 102 with fixed logical_block_size=5000",
        "expected": "different_hashes",
        "unique_hashes": len({h_seed101, h_seed102}),
        "passed": h_seed101 != h_seed102,
    })

    new_b_hashes = {
        run_simulation(
            ctrl_source,
            GeneratorConfig(
                seed=101,
                batch_size=b,
                logical_block_size=5_000,
                scheduler_mode="fixed",
                seed_strategy="philox_logical_block",
            ),
            temp_dir=temp_dir,
            tag=f"ctrl_philox_b{b}",
            cost_model=cost_model,
        )["sha256"]
        for b in (20_000, 25_000, 50_000)
    }
    control_rows.append({
        "control": "v1.3_physical_batch_size_invariance",
        "description": "Philox logical block with physical batch_size in {20000, 25000, 50000}",
        "expected": "single_identical_hash",
        "unique_hashes": len(new_b_hashes),
        "passed": len(new_b_hashes) == 1,
    })

    legacy_b_hashes = {
        run_simulation(
            ctrl_source,
            GeneratorConfig(
                seed=101,
                batch_size=b,
                logical_block_size=5_000,
                scheduler_mode="fixed",
                seed_strategy="batch_index",
            ),
            temp_dir=temp_dir,
            tag=f"ctrl_legacy_b{b}",
            cost_model=cost_model,
        )["sha256"]
        for b in (20_000, 25_000, 50_000)
    }
    control_rows.append({
        "control": "v1.2_legacy_batch_size_sensitivity",
        "description": "Legacy batch_index seed strategy with batch_size in {20000, 25000, 50000}",
        "expected": "three_distinct_hashes",
        "unique_hashes": len(legacy_b_hashes),
        "passed": len(legacy_b_hashes) == 3,
    })

    write_csv(OUTPUT_DIR / "schedule_invariance_raw.csv", raw_rows)
    write_csv(OUTPUT_DIR / "schedule_invariance_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "schedule_invariance_controls.csv", control_rows)
    return raw_rows, summary_rows, control_rows


def run_exp6_compiler_optimization(
    temp_dir: Path,
    *,
    reps: int = 10,
    cost_model: InterpretableASTCostModel | None = None,
) -> list[dict[str, Any]]:
    print("=== Experimento 6: Optimizacion del Compilador y Pruebas Diferenciales (10 repeticiones) ===")
    models = [
        "optimizable_heavy",
        "linear_combination",
        "heavy_expression",
        "quadratic_pm",
        "heavy_polynomial_pm",
        "sqrt_partial_invalid",
    ]
    iterations = 500_000
    rows = []

    for model_name in models:
        source = source_with_iterations(model_name, iterations)
        unopt_times = []
        opt_times = []
        unopt_comp = []
        opt_comp = []
        res_unopt = None
        res_opt = None

        for rep in range(reps):
            r_u = run_simulation(
                source,
                GeneratorConfig(
                    seed=300 + rep,
                    batch_size=50_000,
                    logical_block_size=5_000,
                    max_workers=1,
                    scheduler_mode="hybrid",
                    optimize=False,
                ),
                temp_dir=temp_dir,
                tag=f"exp6_{model_name}_unopt_r{rep}",
                cost_model=cost_model,
            )
            r_o = run_simulation(
                source,
                GeneratorConfig(
                    seed=300 + rep,
                    batch_size=50_000,
                    logical_block_size=5_000,
                    max_workers=1,
                    scheduler_mode="hybrid",
                    optimize=True,
                ),
                temp_dir=temp_dir,
                tag=f"exp6_{model_name}_opt_r{rep}",
                cost_model=cost_model,
            )
            assert r_u["sha256"] == r_o["sha256"], f"Differential check failed for {model_name}"
            unopt_times.append(r_u["execution_seconds"])
            opt_times.append(r_o["execution_seconds"])
            unopt_comp.append(r_u["compile_seconds"])
            opt_comp.append(r_o["compile_seconds"])
            res_unopt = r_u
            res_opt = r_o

        assert res_opt is not None and res_unopt is not None
        metrics = res_opt["compiled_metrics"]
        rows.append({
            "model": model_name,
            "runs": reps,
            "iterations": iterations,
            "ast_nodes_before": metrics["ast_nodes_before"],
            "ast_nodes_after": metrics["ast_nodes_after"],
            "nodes_reduced": metrics["nodes_reduced"],
            "constant_folds": metrics["constant_folds"],
            "algebraic_simplifications": metrics["algebraic_simplifications"],
            "strength_reductions": metrics["strength_reductions"],
            "script_bytes_unoptimized": res_unopt["script_bytes"],
            "script_bytes_optimized": res_opt["script_bytes"],
            "median_compile_ms_unoptimized": round(float(np.median(unopt_comp)) * 1000.0, 4),
            "median_compile_ms_optimized": round(float(np.median(opt_comp)) * 1000.0, 4),
            "median_exec_seconds_unoptimized": round(float(np.median(unopt_times)), 6),
            "median_exec_seconds_optimized": round(float(np.median(opt_times)), 6),
            "speedup_ratio": round(float(np.median(unopt_times)) / float(np.median(opt_times)), 4),
            "exact_hash_match": res_unopt["sha256"] == res_opt["sha256"],
        })

    write_csv(OUTPUT_DIR / "compiler_optimization_benchmark.csv", rows)
    return rows


def run_exp7_statistical_validation_25seeds(
    temp_dir: Path,
    *,
    cost_model: InterpretableASTCostModel | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("=== Experimento 7: Validacion Estadistica Ampliada (25 Semillas + Cobertura IC 95%) ===")
    seeds = [101 + i * 101 for i in range(25)]
    iterations = 100_000
    rows = []

    for model_name, spec in ANALYTIC_SPECS.items():
        source = source_with_iterations(model_name, iterations)
        for seed in seeds:
            res = run_simulation(
                source,
                GeneratorConfig(
                    seed=seed,
                    batch_size=25_000,
                    logical_block_size=5_000,
                    max_workers=2,
                    scheduler_mode="hybrid",
                ),
                temp_dir=temp_dir,
                tag=f"exp7_{model_name}_s{seed}",
                cost_model=cost_model,
            )
            complete = res["complete_event"]
            for branch in complete["branches"]:
                b_id = branch["id"]
                if b_id not in spec["branches"]:
                    continue
                theo_mean = spec["branches"][b_id]["mean"]
                theo_var = spec["branches"][b_id]["variance"]
                est_mean = branch["stats"]["avg"]
                est_var = branch["stats"]["var"]
                mcse = branch["diagnostics"]["mcse_mean"]
                ci95 = branch["diagnostics"]["ci95_mean"]
                abs_err = abs(est_mean - theo_mean)
                err_mcse = abs_err / mcse if mcse and mcse > 0 else 0.0
                in_ci95 = bool(ci95[0] <= theo_mean <= ci95[1])
                rows.append({
                    "model": model_name,
                    "branch": b_id,
                    "seed": seed,
                    "iterations": iterations,
                    "theoretical_mean": theo_mean,
                    "estimated_mean": est_mean,
                    "absolute_error": abs_err,
                    "mcse_mean": mcse,
                    "error_in_mcse_units": err_mcse,
                    "within_3_mcse": err_mcse <= 3.0,
                    "ci95_low": ci95[0],
                    "ci95_high": ci95[1],
                    "ci95_covers_theoretical": in_ci95,
                    "theoretical_variance": theo_var,
                    "estimated_variance": est_var,
                    "variance_abs_error": abs(est_var - theo_var),
                    "valid_rate": branch["stats"]["valid_rate"],
                })

            if "paired" in spec and complete.get("paired"):
                paired = complete["paired"]
                theo_mean = spec["paired"]["mean"]
                theo_var = spec["paired"]["variance"]
                est_mean = paired["stats"]["avg"]
                est_var = paired["stats"]["var"]
                mcse = paired["diagnostics"]["mcse_mean"]
                ci95 = paired["diagnostics"]["ci95_mean"]
                abs_err = abs(est_mean - theo_mean)
                err_mcse = abs_err / mcse if mcse and mcse > 0 else 0.0
                in_ci95 = bool(ci95[0] <= theo_mean <= ci95[1])
                rows.append({
                    "model": model_name,
                    "branch": "paired_delta",
                    "seed": seed,
                    "iterations": iterations,
                    "theoretical_mean": theo_mean,
                    "estimated_mean": est_mean,
                    "absolute_error": abs_err,
                    "mcse_mean": mcse,
                    "error_in_mcse_units": err_mcse,
                    "within_3_mcse": err_mcse <= 3.0,
                    "ci95_low": ci95[0],
                    "ci95_high": ci95[1],
                    "ci95_covers_theoretical": in_ci95,
                    "theoretical_variance": theo_var,
                    "estimated_variance": est_var,
                    "variance_abs_error": abs(est_var - theo_var),
                    "valid_rate": paired["stats"]["valid_rate"],
                })

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["model"], r["branch"]), []).append(r)

    for (model_name, branch_id), grp in sorted(grouped.items()):
        n_seeds = len(grp)
        within_3 = sum(1 for r in grp if r["within_3_mcse"])
        cov_95 = sum(1 for r in grp if r["ci95_covers_theoretical"])
        errs = [r["error_in_mcse_units"] for r in grp]
        summary_rows.append({
            "model": model_name,
            "branch": branch_id,
            "seeds_evaluated": n_seeds,
            "iterations_per_seed": iterations,
            "theoretical_mean": grp[0]["theoretical_mean"],
            "mean_across_seeds": float(np.mean([r["estimated_mean"] for r in grp])),
            "mean_error_in_mcse": float(np.mean(errs)),
            "max_error_in_mcse": float(np.max(errs)),
            "within_3_mcse_count": within_3,
            "within_3_mcse_rate": within_3 / n_seeds,
            "ci95_coverage_count": cov_95,
            "ci95_empirical_coverage": cov_95 / n_seeds,
            "theoretical_variance": grp[0]["theoretical_variance"],
            "mean_estimated_variance": float(np.mean([r["estimated_variance"] for r in grp])),
        })

    write_csv(OUTPUT_DIR / "statistical_validation_25seeds.csv", rows)
    write_csv(OUTPUT_DIR / "ci95_coverage_summary.csv", summary_rows)
    return rows, summary_rows


def _run_subprocess_memory_probe(
    source: str,
    config: GeneratorConfig,
    *,
    temp_dir: Path,
    tag: str,
    cost_model: InterpretableASTCostModel | None = None,
) -> dict[str, Any]:
    """Ejecuta el script generado en un subproceso fresco para medir RSS real aislado del proceso y workers."""
    record_id = _checkpoint_identity(source, config, tag=tag, kind="subprocess_memory_probe", cost_model=cost_model)
    if _CHECKPOINT is not None and record_id is not None:
        cached = _CHECKPOINT.load(record_id)
        if cached is not None:
            return cached

    compiled = compile_source(source, config, cost_model=cost_model)
    script_path = temp_dir / f"{tag}_subproc.py"
    script_path.write_text(compiled.script, encoding="utf-8")
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        elapsed = time.perf_counter() - t0
    finally:
        script_path.unlink(missing_ok=True)
    complete = None
    for line in proc.stdout.splitlines():
        if line.startswith(EVENT_PREFIX):
            evt = json.loads(line[len(EVENT_PREFIX):])
            if evt.get("type") == "complete":
                complete = evt
    if complete is None:
        raise RuntimeError(f"Memory probe {tag} did not emit a complete event")
    result = {"complete_event": complete, "total_seconds": elapsed}
    if _CHECKPOINT is not None and record_id is not None:
        _CHECKPOINT.save(record_id, result)
    return result


def run_exp8_memory_consumption(
    temp_dir: Path,
    *,
    cost_model: InterpretableASTCostModel | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("=== Experimento 8: Memoria Retenida O(1) vs RSS Real (1 y 2 Workers) y Fidelidad de Cuantiles en Multiples Distribuciones y Ramas ===")
    models = ["normal_mean", "quadratic_pm"]
    iteration_levels = [10_000, 100_000, 500_000, 1_000_000, 2_500_000]
    worker_configs = memory_worker_configs()
    rows = []

    for w_label, s_mode, w_count in worker_configs:
        for model_name in models:
            for iterations in iteration_levels:
                if w_count > 1 and iterations not in (100_000, 1_000_000, 2_500_000):
                    continue
                source = source_with_iterations(model_name, iterations)
                res_retain = _run_subprocess_memory_probe(
                    source,
                    GeneratorConfig(
                        seed=101,
                        batch_size=25_000,
                        logical_block_size=5_000,
                        max_workers=w_count,
                        scheduler_mode=s_mode,
                        memory_mode="retain_all",
                        max_iterations=10_000_000,
                    ),
                    temp_dir=temp_dir,
                    tag=f"exp8_{w_label}_{model_name}_{iterations}_retain",
                    cost_model=cost_model,
                )
                res_bounded = _run_subprocess_memory_probe(
                    source,
                    GeneratorConfig(
                        seed=101,
                        batch_size=25_000,
                        logical_block_size=5_000,
                        max_workers=w_count,
                        scheduler_mode=s_mode,
                        memory_mode="bounded",
                        max_iterations=10_000_000,
                    ),
                    temp_dir=temp_dir,
                    tag=f"exp8_{w_label}_{model_name}_{iterations}_bounded",
                    cost_model=cost_model,
                )

                c_ret = res_retain["complete_event"]
                c_bnd = res_bounded["complete_event"]
                m_ret = c_ret["metadata"]
                m_bnd = c_bnd["metadata"]

                ret_payload_ret = int(m_ret["peak_retained_coordinator_bytes"])
                ret_payload_bnd = int(m_bnd["peak_retained_coordinator_bytes"])

                coord_rss_ret = int(m_ret["coordinator_peak_rss_bytes"])
                coord_rss_bnd = int(m_bnd["coordinator_peak_rss_bytes"])
                work_rss_ret = int(m_ret["workers_peak_rss_bytes"])
                work_rss_bnd = int(m_bnd["workers_peak_rss_bytes"])
                agg_rss_ret = int(m_ret.get("sum_of_process_peak_rss_bytes", m_ret.get("aggregate_peak_rss_bytes", 0)))
                agg_rss_bnd = int(m_bnd.get("sum_of_process_peak_rss_bytes", m_bnd.get("aggregate_peak_rss_bytes", 0)))

                b0_ret = c_ret["branches"][0]
                b0_bnd = c_bnd["branches"][0]
                exact_moments = (
                    b0_ret["stats"]["avg"] == b0_bnd["stats"]["avg"]
                    and b0_ret["stats"]["std"] == b0_bnd["stats"]["std"]
                    and b0_ret["histogram"]["total_count"] == b0_bnd["histogram"]["total_count"]
                    and b0_ret["histogram"]["data_min"] == b0_bnd["histogram"]["data_min"]
                    and b0_ret["histogram"]["data_max"] == b0_bnd["histogram"]["data_max"]
                )
                if c_ret.get("paired") and c_bnd.get("paired"):
                    exact_moments = exact_moments and (
                        abs(
                            float(c_ret["paired"]["correlation_plus_minus_mean"])
                            - float(c_bnd["paired"]["correlation_plus_minus_mean"])
                        )
                        < 1e-12
                    )

                q_errs_abs = []
                q_errs_rel = []
                for qk in ("p05", "p50", "p95"):
                    v_ret = float(b0_ret["stats"][qk])
                    v_bnd = float(b0_bnd["stats"][qk])
                    abs_e = abs(v_bnd - v_ret)
                    rel_e = abs_e / max(1e-9, abs(v_ret))
                    q_errs_abs.append(abs_e)
                    q_errs_rel.append(rel_e)

                counts_ret = np.array([b["count"] for b in b0_ret["histogram"]["bins"]], dtype=np.int64)
                counts_bnd = np.array([b["count"] for b in b0_bnd["histogram"]["bins"]], dtype=np.int64)
                valid_n = max(1, int(b0_ret["histogram"]["total_count"]))
                hist_tv_pct = (float(np.sum(np.abs(counts_bnd - counts_ret))) / (2.0 * valid_n)) * 100.0

                rows.append({
                    "execution_mode": w_label,
                    "campaign_profile": EXPERIMENT_PROFILE,
                    "effective_workers": int(m_bnd["worker_count"]),
                    "model": model_name,
                    "iterations": iterations,
                    "retain_all_peak_bytes": ret_payload_ret,
                    "bounded_peak_bytes": ret_payload_bnd,
                    "retain_all_peak_mb": round(ret_payload_ret / (1024 * 1024), 3),
                    "bounded_peak_mb": round(ret_payload_bnd / (1024 * 1024), 3),
                    "memory_reduction_pct": round((1.0 - ret_payload_bnd / max(1, ret_payload_ret)) * 100.0, 2),
                    "retain_all_coord_rss_mb": round(coord_rss_ret / (1024 * 1024), 2),
                    "bounded_coord_rss_mb": round(coord_rss_bnd / (1024 * 1024), 2),
                    "retain_all_workers_rss_mb": round(work_rss_ret / (1024 * 1024), 2),
                    "bounded_workers_rss_mb": round(work_rss_bnd / (1024 * 1024), 2),
                    "retain_all_sum_of_process_peak_rss_mb": round(agg_rss_ret / (1024 * 1024), 2),
                    "bounded_sum_of_process_peak_rss_mb": round(agg_rss_bnd / (1024 * 1024), 2),
                    "retain_all_seconds": round(res_retain["total_seconds"], 4),
                    "bounded_seconds": round(res_bounded["total_seconds"], 4),
                    "exact_moments_and_counts_preserved": exact_moments,
                    "p50_abs_error": round(q_errs_abs[1], 6),
                    "max_quantile_abs_error": round(max(q_errs_abs), 6),
                    "max_quantile_rel_error_pct": round(max(q_errs_rel) * 100.0, 4),
                    "histogram_tv_distance_pct": round(hist_tv_pct, 4),
                })

    # Evaluacion ampliada de cuantiles e histogramas en 6 familias de distribucion y todas las ramas
    quantile_models = [
        ("normal_mean", "symmetric_unimodal"),
        ("discrete_mean", "discrete_step"),
        ("multi_discrete_complex", "multimodal_mixture"),
        ("normal_quadratic", "right_skewed_quadratic"),
        ("heavy_polynomial_pm", "heavy_tailed_polynomial_pm"),
        ("quadratic_pm", "paired_nonlinear_filtered_pm"),
    ]
    q_rows = []
    q_iters = 1_000_000
    for m_name, dist_family in quantile_models:
        src = source_with_iterations(m_name, q_iters)
        r_ret = run_simulation(
            src,
            GeneratorConfig(
                seed=101,
                batch_size=25_000,
                logical_block_size=5_000,
                max_workers=1,
                scheduler_mode="hybrid",
                memory_mode="retain_all",
            ),
            temp_dir=temp_dir,
            tag=f"exp8_q_{m_name}_ret",
            cost_model=cost_model,
        )["complete_event"]
        r_bnd = run_simulation(
            src,
            GeneratorConfig(
                seed=101,
                batch_size=25_000,
                logical_block_size=5_000,
                max_workers=1,
                scheduler_mode="hybrid",
                memory_mode="bounded",
            ),
            temp_dir=temp_dir,
            tag=f"exp8_q_{m_name}_bnd",
            cost_model=cost_model,
        )["complete_event"]

        for b_ret, b_bnd in zip(r_ret["branches"], r_bnd["branches"]):
            q_abs = [abs(float(b_bnd["stats"][k]) - float(b_ret["stats"][k])) for k in ("p05", "p50", "p95")]
            q_rel = [
                abs(float(b_bnd["stats"][k]) - float(b_ret["stats"][k])) / max(1e-9, abs(float(b_ret["stats"][k])))
                for k in ("p05", "p50", "p95")
            ]
            c_r = np.array([b["count"] for b in b_ret["histogram"]["bins"]], dtype=np.int64)
            c_b = np.array([b["count"] for b in b_bnd["histogram"]["bins"]], dtype=np.int64)
            tv_pct = (float(np.sum(np.abs(c_b - c_r))) / (2.0 * max(1, int(b_ret["histogram"]["total_count"])))) * 100.0
            q_rows.append({
                "model": m_name,
                "distribution_family": dist_family,
                "branch": b_ret["id"],
                "iterations": q_iters,
                "valid_results": b_ret["valid_results"],
                "p05_abs_error": round(q_abs[0], 6),
                "p50_abs_error": round(q_abs[1], 6),
                "p95_abs_error": round(q_abs[2], 6),
                "max_quantile_rel_error_pct": round(max(q_rel) * 100.0, 4),
                "histogram_tv_distance_pct": round(tv_pct, 4),
                "exact_moments_preserved": (
                    b_ret["stats"]["avg"] == b_bnd["stats"]["avg"]
                    and b_ret["stats"]["std"] == b_bnd["stats"]["std"]
                ),
            })

        if r_ret.get("paired") and r_bnd.get("paired"):
            p_ret = r_ret["paired"]
            p_bnd = r_bnd["paired"]
            q_abs = [abs(float(p_bnd["stats"][k]) - float(p_ret["stats"][k])) for k in ("p05", "p50", "p95")]
            q_rel = [
                abs(float(p_bnd["stats"][k]) - float(p_ret["stats"][k])) / max(1e-9, abs(float(p_ret["stats"][k])))
                for k in ("p05", "p50", "p95")
            ]
            p_ret_counts = np.array(
                [item["count"] for item in p_ret["histogram"]["bins"]], dtype=np.int64
            )
            p_bnd_counts = np.array(
                [item["count"] for item in p_bnd["histogram"]["bins"]], dtype=np.int64
            )
            p_ret_n = int(p_ret["histogram"]["total_count"])
            p_bnd_n = int(p_bnd["histogram"]["total_count"])
            if p_ret_n != p_bnd_n or len(p_ret_counts) != len(p_bnd_counts):
                raise RuntimeError(
                    "Paired-delta histograms are not comparable: bin count or valid counts differ."
                )
            p_valid_n = max(1, p_ret_n)
            p_tv_pct = (
                float(np.sum(np.abs(p_bnd_counts - p_ret_counts))) / (2.0 * p_valid_n)
            ) * 100.0
            q_rows.append({
                "model": m_name,
                "distribution_family": dist_family,
                "branch": "paired_delta",
                "iterations": q_iters,
                "valid_results": p_ret["valid_results"],
                "p05_abs_error": round(q_abs[0], 6),
                "p50_abs_error": round(q_abs[1], 6),
                "p95_abs_error": round(q_abs[2], 6),
                "max_quantile_rel_error_pct": round(max(q_rel) * 100.0, 4),
                "histogram_tv_distance_pct": round(p_tv_pct, 4),
                "exact_moments_preserved": (
                    p_ret["stats"]["avg"] == p_bnd["stats"]["avg"]
                    and p_ret["stats"]["std"] == p_bnd["stats"]["std"]
                    and abs(float(p_ret["correlation_plus_minus_mean"]) - float(p_bnd["correlation_plus_minus_mean"])) < 1e-12
                ),
            })

    write_csv(OUTPUT_DIR / "memory_retention_comparison.csv", rows)
    write_csv(OUTPUT_DIR / "quantile_and_histogram_fidelity.csv", q_rows)
    return rows, q_rows


def run_exp9_ablation_study(
    temp_dir: Path,
    *,
    reps: int = 10,
    cost_model: InterpretableASTCostModel | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print("=== Experimento 9: Estudio de Ablacion 4x2 (10 repeticiones por celda) ===")
    models = ["normal_mean", "heavy_expression", "quadratic_pm", "optimizable_heavy"]
    iterations = 1_000_000
    scheduler_variants = ("fixed", "ast_only", "pilot_only", "hybrid")
    optimizer_flags = (False, True)

    raw_rows = []
    summary_rows = []

    for model_name in models:
        source = source_with_iterations(model_name, iterations)
        baseline_times = []
        grouped_times: dict[tuple[str, bool], list[float]] = {}
        grouped_meta: dict[tuple[str, bool], dict[str, Any]] = {}
        ref_hash = None

        for s_mode in scheduler_variants:
            for opt_flag in optimizer_flags:
                key = (s_mode, opt_flag)
                grouped_times[key] = []
                for rep in range(1, reps + 1):
                    res = run_simulation(
                        source,
                        GeneratorConfig(
                            seed=500 + rep,
                            batch_size=25_000,
                            logical_block_size=5_000,
                            max_workers=2,
                            scheduler_mode=s_mode,
                            optimize=opt_flag,
                        ),
                        temp_dir=temp_dir,
                        tag=f"exp9_{model_name}_{s_mode}_opt{opt_flag}_r{rep}",
                        cost_model=cost_model,
                    )
                    if rep == 1 and ref_hash is None:
                        ref_hash = res["sha256"]
                    if rep == 1:
                        assert res["sha256"] == ref_hash, "Ablation variant altered canonical result!"
                    meta = res["complete_event"]["metadata"]
                    grouped_times[key].append(res["total_seconds"])
                    grouped_meta[key] = meta
                    raw_rows.append({
                        "model": model_name,
                        "iterations": iterations,
                        "scheduler_variant": s_mode,
                        "optimizer_enabled": opt_flag,
                        "repetition": rep,
                        "effective_workers": meta["worker_count"],
                        "batch_count": meta["batch_count"],
                        "use_process_pool": meta.get("use_process_pool"),
                        "total_seconds": res["total_seconds"],
                        "peak_memory_bytes": meta.get("peak_memory_bytes"),
                        "result_sha256": res["sha256"],
                    })
                if key == ("fixed", False):
                    baseline_times = list(grouped_times[key])

        for (s_mode, opt_flag), times in grouped_times.items():
            meta = grouped_meta[(s_mode, opt_flag)]
            ratios = [b / t for b, t in zip(baseline_times, times)]
            ci_l, ci_h = bootstrap_ci95(ratios, np.median)
            summary_rows.append({
                "model": model_name,
                "iterations": iterations,
                "scheduler_variant": s_mode,
                "optimizer_enabled": opt_flag,
                "runs": len(times),
                "effective_workers": meta["worker_count"],
                "batch_count": meta["batch_count"],
                "use_process_pool": meta.get("use_process_pool"),
                "median_seconds": round(float(np.median(times)), 6),
                "mean_seconds": round(float(np.mean(times)), 6),
                "std_seconds": round(float(np.std(times, ddof=1)), 6),
                "iqr_seconds": round(float(np.percentile(times, 75) - np.percentile(times, 25)), 6),
                "median_speedup_vs_fixed_unopt": round(float(np.median(ratios)), 4),
                "speedup_ci95_low": round(ci_l, 4),
                "speedup_ci95_high": round(ci_h, 4),
            })

    write_csv(OUTPUT_DIR / "ablation_study_raw.csv", raw_rows)
    write_csv(OUTPUT_DIR / "ablation_study_summary.csv", summary_rows)
    return raw_rows, summary_rows


def write_environment_metadata(
    *,
    confirmatory_reps: int,
    pilot_reps: int,
    ablation_reps: int,
    fitted_model: InterpretableASTCostModel | None = None,
) -> dict[str, Any]:
    import fastapi
    import httpx
    import pydantic
    import scipy
    import starlette
    import uvicorn

    dependency_report = _check_declared_dependencies()
    cpu_topology = detect_cpu_topology()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        "version": "1.3.1",
        "campaign_label": CAMPAIGN_LABEL,
        "experiment_profile": EXPERIMENT_PROFILE,
        "requested_instance_type": INSTANCE_TYPE,
        "cloud_region": CLOUD_REGION,
        "output_directory": str(OUTPUT_DIR.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": git_is_dirty(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "fastapi_version": fastapi.__version__,
        "pydantic_version": pydantic.__version__,
        "starlette_version": starlette.__version__,
        "uvicorn_version": uvicorn.__version__,
        "httpx_version": httpx.__version__,
        "pyproject_dependencies_compliant": dependency_report["compliant"],
        "dependency_compliance": dependency_report,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_cpu_topology": cpu_topology,
        "seed_strategy": "numpy.SeedSequence([MASTER_SEED, logical_block_index, var_index]) -> Philox",
        "logical_block_size_default": 5_000,
        "memory_mode_default": "bounded",
        "max_global_sketch_points": 2048,
        "confirmatory_repetitions": confirmatory_reps,
        "pilot_repetitions": pilot_reps,
        "ablation_repetitions": ablation_reps,
        "statistical_validation_seeds": 25,
        "bootstrap_resamples": 10_000,
        "scheduler_modes_evaluated": ["fixed", "ast_only", "pilot_only", "hybrid"],
        "cost_model_weights_seconds_per_sample": list(
            (fitted_model or InterpretableASTCostModel()).weights_seconds_per_sample
        ),
        "checkpoint_provenance": _CHECKPOINT.summary() if _CHECKPOINT is not None else None,
    }
    (OUTPUT_DIR / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return env


def _check_declared_dependencies() -> dict[str, Any]:
    """Check installed versions against every dependency declared in pyproject.toml."""
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        from packaging.requirements import Requirement
        from packaging.version import Version

        project_data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = project_data.get("project", {})
        requirements = list(project.get("dependencies", []))
        for optional_group in project.get("optional-dependencies", {}).values():
            requirements.extend(optional_group)

        checked = []
        mismatches = []
        for raw_requirement in requirements:
            requirement = Requirement(raw_requirement)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            try:
                installed = importlib_metadata.version(requirement.name)
            except importlib_metadata.PackageNotFoundError:
                installed = None
            compatible = (
                installed is not None
                and (not requirement.specifier or Version(installed) in requirement.specifier)
            )
            item = {
                "requirement": raw_requirement,
                "package": requirement.name,
                "installed_version": installed,
                "compatible": compatible,
            }
            checked.append(item)
            if not compatible:
                mismatches.append(item)
        return {
            "compliant": not mismatches,
            "checked_count": len(checked),
            "mismatches": mismatches,
            "checked": checked,
        }
    except Exception as exc:
        return {
            "compliant": False,
            "checked_count": 0,
            "mismatches": [],
            "check_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Monte Carlo DSL v1.3.1 experimental campaign")
    parser.add_argument("--confirmatory-reps", type=int, default=30)
    parser.add_argument("--pilot-reps", type=int, default=10)
    parser.add_argument("--ablation-reps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "benchmarks" / "checkpoints" / "v1.3.1",
                        help="Directory containing crash-safe per-simulation checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks" / "results" / "final" / "v1.3.1",
                        help="Directory for this campaign's CSV and environment.json outputs")
    parser.add_argument("--profile", choices=("standard", "aws-4core"), default="standard",
                        help="aws-4core adds the four-worker memory stress matrix")
    parser.add_argument("--campaign-label", default=None,
                        help="Human-readable campaign label stored in environment metadata")
    parser.add_argument("--instance-type", default=None,
                        help="Cloud instance type label, e.g. c7i.2xlarge")
    parser.add_argument("--cloud-region", default=None,
                        help="Cloud region label, e.g. mx-central-1")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously interrupted campaign (requires exactly matching inputs/environment)")
    parser.add_argument("--progress-every", type=int, default=1,
                        help="Print progress for every N completed or reused simulations (default: 1)")
    args = parser.parse_args()
    global OUTPUT_DIR, EXPERIMENT_PROFILE, CAMPAIGN_LABEL, INSTANCE_TYPE, CLOUD_REGION
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    EXPERIMENT_PROFILE = args.profile
    INSTANCE_TYPE = args.instance_type
    CLOUD_REGION = args.cloud_region
    CAMPAIGN_LABEL = args.campaign_label or (
        f"aws-{args.instance_type or '4core'}-ubuntu-24.04"
        if args.profile == "aws-4core" else "local-v1.3.1"
    )
    if min(args.confirmatory_reps, args.pilot_reps, args.ablation_reps, args.progress_every) <= 0:
        parser.error("All repetition counts and --progress-every must be positive")
    if args.profile == "aws-4core":
        topology = detect_cpu_topology()
        physical = topology.get("physical_core_count")
        if physical is not None and int(physical) < 4:
            parser.error(
                f"The aws-4core profile requires at least 4 physical cores; detected {physical}. "
                "Use a 4-core instance or choose --profile standard."
            )

    global _CHECKPOINT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency_report = _check_declared_dependencies()
    if not dependency_report["compliant"]:
        raise RuntimeError(
            "The current environment does not satisfy all declared pyproject dependencies; "
            "install the project with `pip install -e .[dev]` before running the campaign. "
            f"Mismatches: {dependency_report['mismatches']}"
        )
    _CHECKPOINT = CampaignCheckpoint(
        args.checkpoint_dir,
        campaign_identity(
            ROOT, confirmatory_reps=args.confirmatory_reps,
            pilot_reps=args.pilot_reps, ablation_reps=args.ablation_reps,
            campaign_label=CAMPAIGN_LABEL, experiment_profile=EXPERIMENT_PROFILE,
            instance_type=INSTANCE_TYPE, cloud_region=CLOUD_REGION, output_dir=str(OUTPUT_DIR),
        ),
        resume=args.resume,
        progress_every=args.progress_every,
    )
    _CHECKPOINT.save_status("running")
    fitted_model = None
    try:
        with tempfile.TemporaryDirectory(prefix="mcdsl_v13_exp_") as tmp:
            temp_dir = Path(tmp)
            write_environment_metadata(
                confirmatory_reps=args.confirmatory_reps,
                pilot_reps=args.pilot_reps,
                ablation_reps=args.ablation_reps,
                fitted_model=None,
            )
            # Fit first; all subsequent experiments use the SAME fitted model.
            _, _, fitted_model = run_exp5_ast_cost_generalization(temp_dir, reps=args.pilot_reps)
            write_environment_metadata(
                confirmatory_reps=args.confirmatory_reps,
                pilot_reps=args.pilot_reps,
                ablation_reps=args.ablation_reps,
                fitted_model=fitted_model,
            )
            run_exp0_v1_2_vs_v1_3_baseline_comparison(
                temp_dir, reps=args.confirmatory_reps, cost_model=fitted_model
            )
            run_exp1_functional_equivalence(temp_dir, cost_model=fitted_model)
            run_exp2_performance_and_scalability(
                temp_dir,
                confirmatory_reps=args.confirmatory_reps,
                pilot_reps=args.pilot_reps,
                cost_model=fitted_model,
            )
            run_exp3_batch_adaptation(temp_dir, reps=args.pilot_reps, cost_model=fitted_model)
            run_exp4_schedule_invariance_reproducibility(temp_dir, cost_model=fitted_model)
            run_exp6_compiler_optimization(temp_dir, reps=args.pilot_reps, cost_model=fitted_model)
            run_exp7_statistical_validation_25seeds(temp_dir, cost_model=fitted_model)
            run_exp8_memory_consumption(temp_dir, cost_model=fitted_model)
            run_exp9_ablation_study(temp_dir, reps=args.ablation_reps, cost_model=fitted_model)

        # Update provenance after the last measurement, not just after Exp. 5.
        write_environment_metadata(
            confirmatory_reps=args.confirmatory_reps,
            pilot_reps=args.pilot_reps,
            ablation_reps=args.ablation_reps,
            fitted_model=fitted_model,
        )
        _CHECKPOINT.save_status("completed")
        print(f"\nALL v1.3.1 EXPERIMENTS COMPLETED SUCCESSFULLY -> {OUTPUT_DIR}", flush=True)
        return 0
    except BaseException:
        _CHECKPOINT.save_status("interrupted")
        raise
    finally:
        _CHECKPOINT = None


if __name__ == "__main__":
    raise SystemExit(main())
