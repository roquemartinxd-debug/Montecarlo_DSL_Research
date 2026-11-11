from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from typing import Any

# Make direct invocation (`python benchmarks/run_v1_3_targeted_experiments.py`) work.
ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import numpy as np
import psutil

from benchmarks import run_v1_3_experiments as base
from benchmarks.campaign_checkpoint import CampaignCheckpoint, campaign_identity
from montecarlo_dsl import GeneratorConfig, InterpretableASTCostModel


BREAK_EVEN_MODELS = ("normal_mean", "linear_combination", "heavy_expression", "quadratic_pm")
BREAK_EVEN_LEVELS = (100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000)
BREAK_EVEN_LARGE_ANCHORS = (5_000_000, 10_000_000)
MEMORY_MODELS = ("normal_mean", "quadratic_pm")
MEMORY_LEVELS_BASE = (250_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000, 20_000_000)
MEMORY_RETAIN_LEVELS = (250_000, 1_000_000, 2_500_000)
MEMORY_STRESS_ORDERS = ("as_completed", "reverse", "shuffled", "delayed_first")
FIRST_TASK_DELAY_SECONDS = 1.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _parse_weights(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weights must be four comma-separated numbers") from exc
    if len(values) != 4 or any(not math.isfinite(x) or x < 0 for x in values):
        raise argparse.ArgumentTypeError("weights must be four finite non-negative numbers")
    return values  # type: ignore[return-value]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bootstrap_ci(values: list[float], *, seed: int) -> tuple[float, float]:
    return base.bootstrap_ci95(values, np.median, reps=10_000, seed=seed)


def _make_cost_model(weights: tuple[float, float, float, float]) -> InterpretableASTCostModel:
    return InterpretableASTCostModel(weights_seconds_per_sample=weights)


def _break_even_configs(physical_cores: int) -> list[dict[str, Any]]:
    return [
        {"label": "serial_in_process", "mode": "ast_only", "workers": 1, "pool_expected": False},
        {"label": "fixed_pool_w1", "mode": "fixed", "workers": 1, "pool_expected": True},
        {"label": "fixed_pool_w2", "mode": "fixed", "workers": 2, "pool_expected": True},
        {"label": "fixed_pool_w4", "mode": "fixed", "workers": 4, "pool_expected": True},
        {"label": "adaptive_hybrid", "mode": "hybrid", "workers": physical_cores, "pool_expected": None},
    ]


def _memory_configs(physical_cores: int) -> list[dict[str, Any]]:
    configs = [
        {"label": "serial_in_process_w1", "mode": "ast_only", "workers": 1},
        {"label": "fixed_pool_w2", "mode": "fixed", "workers": 2},
    ]
    if physical_cores >= 4:
        configs.append({"label": "fixed_pool_w4", "mode": "fixed", "workers": 4})
    return configs


def _config_for(
    *,
    seed: int,
    iterations: int,
    mode: str,
    workers: int,
    memory_mode: str = "bounded",
    task_order: str = "as_completed",
    delay_first_seconds: float = 0.0,
    batch_size: int = 25_000,
    logical_block_size: int = 5_000,
) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        batch_size=batch_size,
        logical_block_size=logical_block_size,
        max_workers=workers,
        scheduler_mode=mode,
        seed_strategy="philox_logical_block",
        memory_mode=memory_mode,
        optimize=True,
        task_completion_order=task_order,
        task_delay_first_seconds=delay_first_seconds,
        max_iterations=max(iterations, 50_000_000),
    )


def _campaign_store(args: argparse.Namespace, study: str, phase: str) -> CampaignCheckpoint:
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() / f"{study}_{phase}"
    output_dir = args.output_dir.expanduser().resolve()
    identity = campaign_identity(
        ROOT,
        confirmatory_reps=args.confirmatory_reps,
        pilot_reps=args.exploration_reps,
        ablation_reps=args.memory_reps,
        campaign_label=args.campaign_label,
        experiment_profile=f"targeted-{args.profile}",
        instance_type=args.instance_type,
        cloud_region=args.cloud_region,
        output_dir=str(output_dir),
    )
    identity.update({
        "targeted_protocol_version": "1.0.0",
        "targeted_script_sha256": _sha256_file(Path(__file__).resolve()),
        "generator_sha256": _sha256_file(ROOT / "src" / "montecarlo_dsl" / "generator.py"),
        "study": study,
        "phase": phase,
        "platform_label": args.platform_label,
        "physical_cores_declared": args.physical_cores,
        "cost_model_weights_seconds_per_sample": list(args.cost_model_weights),
        "exploration_reps": args.exploration_reps,
        "confirmatory_reps": args.confirmatory_reps,
        "memory_reps": args.memory_reps,
        "retain_reps": args.retain_reps,
        "stress_reps": args.stress_reps,
        "monitor_interval_seconds": args.monitor_interval,
        "memory_timeout_seconds": args.memory_timeout_seconds,
        "first_task_delay_seconds": args.first_task_delay_seconds,
        "max_memory_iterations": args.max_memory_iterations or (
            50_000_000 if args.profile == "aws-4core" else 20_000_000
        ),
        "include_retain_5m": args.include_retain_5m,
        "confirmatory_iterations": [int(x.strip()) for x in args.confirmatory_iterations.split(",") if x.strip()],
        "break_even_models": list(BREAK_EVEN_MODELS),
        "break_even_levels": list(BREAK_EVEN_LEVELS),
        "break_even_large_anchors": list(BREAK_EVEN_LARGE_ANCHORS),
        "memory_models": list(MEMORY_MODELS),
        "memory_stress_orders": list(MEMORY_STRESS_ORDERS),
        "memory_levels_base": list(MEMORY_LEVELS_BASE),
        "memory_retain_levels": list(MEMORY_RETAIN_LEVELS),
        "psutil_version": psutil.__version__,
    })
    manifest = checkpoint_dir / "manifest.json"
    resume = bool(args.resume and manifest.is_file())
    return CampaignCheckpoint(checkpoint_dir, identity, resume=resume, progress_every=args.progress_every)


def _run_break_even(
    args: argparse.Namespace,
    *,
    temp_dir: Path,
    cost_model: InterpretableASTCostModel,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if args.phase == "exploratory":
        plan = [(model, n, args.exploration_reps, "exploratory")
                for model in BREAK_EVEN_MODELS for n in BREAK_EVEN_LEVELS]
        # Large anchors are intentionally exploratory and limited to two models.
        plan.extend((model, n, min(args.exploration_reps, args.large_anchor_reps), "large_anchor")
                    for model in ("normal_mean", "quadratic_pm") for n in BREAK_EVEN_LARGE_ANCHORS)
        seed_base = 3_100_000
    else:
        selected = [int(x.strip()) for x in args.confirmatory_iterations.split(",") if x.strip()]
        if not selected:
            raise ValueError("--confirmatory-iterations is required for --phase confirmatory")
        if any(n <= 0 for n in selected):
            raise ValueError("confirmatory iteration counts must be positive")
        plan = [(model, n, args.confirmatory_reps, "confirmatory")
                for model in BREAK_EVEN_MODELS for n in selected]
        seed_base = 4_100_000

    configs = _break_even_configs(args.physical_cores)
    raw_rows: list[dict[str, Any]] = []
    output_dir = args.output_dir.expanduser().resolve()

    for model_name, iterations, reps, stratum in plan:
        source = base.source_with_iterations(model_name, iterations)
        # One warm-up per model/size/configuration; never included in timing rows.
        for cfg in configs:
            warm_cfg = _config_for(
                seed=91_337, iterations=iterations, mode=cfg["mode"], workers=cfg["workers"]
            )
            base.run_simulation(
                source, warm_cfg, temp_dir=temp_dir,
                tag=f"warmup_exp10_{args.phase}_{model_name}_{iterations}_{cfg['label']}",
                cost_model=cost_model,
            )

        for rep in range(1, reps + 1):
            seed = seed_base + rep
            order = list(configs)
            random.Random(_stable_int(f"{args.platform_label}|{model_name}|{iterations}|{rep}|exp10")).shuffle(order)
            for cfg in order:
                config = _config_for(
                    seed=seed, iterations=iterations, mode=cfg["mode"], workers=cfg["workers"]
                )
                tag = f"exp10_{args.phase}_{model_name}_{iterations}_{cfg['label']}_r{rep}"
                result = base.run_simulation(
                    source, config, temp_dir=temp_dir, tag=tag, cost_model=cost_model
                )
                meta = result["complete_event"]["metadata"]
                raw_rows.append({
                    "phase": args.phase,
                    "stratum": stratum,
                    "platform_label": args.platform_label,
                    "physical_cores": args.physical_cores,
                    "logical_cpus": os.cpu_count(),
                    "model": model_name,
                    "iterations": iterations,
                    "configuration": cfg["label"],
                    "scheduler_mode": cfg["mode"],
                    "requested_workers": cfg["workers"],
                    "effective_workers": int(meta["worker_count"]),
                    "oversubscribed": bool(cfg["workers"] > args.physical_cores),
                    "use_process_pool": bool(meta.get("use_process_pool")),
                    "batch_count": int(meta["batch_count"]),
                    "repetition": rep,
                    "seed": seed,
                    "compile_seconds": float(result["compile_seconds"]),
                    "execution_seconds": float(result["execution_seconds"]),
                    "total_seconds": float(result["total_seconds"]),
                    "simulation_compute_seconds": float(meta.get("simulation_compute_seconds") or 0.0),
                    "pilot_elapsed_seconds": meta.get("pilot_elapsed_seconds"),
                    "pilot_per_sample_seconds": meta.get("pilot_per_sample_seconds"),
                    "effective_per_sample_seconds": meta.get("effective_per_sample_seconds"),
                    "max_in_flight_tasks": int(meta.get("max_in_flight_tasks", 1)),
                    "result_sha256": result["sha256"],
                    "cost_model_weights_seconds_per_sample": ";".join(f"{x:.12g}" for x in args.cost_model_weights),
                })
        print(f"[exp10] completed {model_name} N={iterations:,} reps={reps} ({stratum})", flush=True)

    # A resumed run replays cached checkpoints, so the final raw file is rebuilt
    # deterministically from the complete plan rather than appended incrementally.
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["model"], int(row["iterations"]))].append(row)

    for (model_name, iterations), group in sorted(grouped.items()):
        by_config: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            by_config[row["configuration"]][int(row["repetition"])] = row
        serial = by_config["serial_in_process"]
        medians = {label: float(np.median([float(r["execution_seconds"]) for r in reps_map.values()]))
                   for label, reps_map in by_config.items()}
        fixed_labels = ("serial_in_process", "fixed_pool_w1", "fixed_pool_w2", "fixed_pool_w4")
        best_fixed = min((label for label in fixed_labels if label in medians), key=lambda label: medians[label])
        adaptive_median = medians["adaptive_hybrid"]
        best_fixed_median = medians[best_fixed]
        adaptive_pairs = sorted(set(by_config["adaptive_hybrid"]).intersection(by_config[best_fixed]))
        adaptive_regret_values = [
            (float(by_config["adaptive_hybrid"][rep]["execution_seconds"])
             / max(1e-12, float(by_config[best_fixed][rep]["execution_seconds"])) - 1.0) * 100.0
            for rep in adaptive_pairs
        ]
        regret_ci = _bootstrap_ci(
            adaptive_regret_values,
            seed=_stable_int(f"{model_name}|{iterations}|adaptive-regret|{args.phase}"),
        )
        for label, reps_map in by_config.items():
            common_reps = sorted(set(serial).intersection(reps_map))
            speedups = [float(serial[r]["execution_seconds"]) / max(1e-12, float(reps_map[r]["execution_seconds"]))
                        for r in common_reps]
            ci_low, ci_high = _bootstrap_ci(speedups, seed=_stable_int(f"{model_name}|{iterations}|{label}|{args.phase}"))
            times = [float(r["execution_seconds"]) for r in reps_map.values()]
            adaptive_regret = (adaptive_median / best_fixed_median - 1.0) * 100.0
            sample = next(iter(reps_map.values()))
            summary_rows.append({
                "phase": args.phase,
                "stratum": sample["stratum"],
                "platform_label": args.platform_label,
                "physical_cores": args.physical_cores,
                "model": model_name,
                "iterations": iterations,
                "configuration": label,
                "runs": len(times),
                "requested_workers": sample["requested_workers"],
                "median_effective_workers": float(np.median([float(r["effective_workers"]) for r in reps_map.values()])),
                "oversubscribed": sample["oversubscribed"],
                "median_execution_seconds": float(np.median(times)),
                "mean_execution_seconds": float(np.mean(times)),
                "std_execution_seconds": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
                "iqr_execution_seconds": float(np.percentile(times, 75) - np.percentile(times, 25)),
                "median_speedup_vs_serial": float(np.median(speedups)),
                "speedup_vs_serial_ci95_low": ci_low,
                "speedup_vs_serial_ci95_high": ci_high,
                "best_fixed_configuration": best_fixed,
                "best_fixed_median_seconds": best_fixed_median,
                "adaptive_median_seconds": adaptive_median,
                "adaptive_regret_vs_best_fixed_pct": adaptive_regret,
                "adaptive_regret_ci95_low_pct": regret_ci[0],
                "adaptive_regret_ci95_high_pct": regret_ci[1],
                "adaptive_matches_best_fixed": bool(medians["adaptive_hybrid"] <= best_fixed_median * 1.001),
            })

    threshold_rows: list[dict[str, Any]] = []
    for model_name in BREAK_EVEN_MODELS:
        model_rows = [r for r in summary_rows if r["model"] == model_name]
        for label in ("fixed_pool_w1", "fixed_pool_w2", "fixed_pool_w4"):
            ordered = sorted((r for r in model_rows if r["configuration"] == label), key=lambda r: int(r["iterations"]))
            threshold = None
            for index, row in enumerate(ordered):
                if float(row["speedup_vs_serial_ci95_low"]) > 1.0 and all(
                    float(later["speedup_vs_serial_ci95_low"]) > 1.0 for later in ordered[index:]
                ):
                    threshold = int(row["iterations"])
                    break
            threshold_rows.append({
                "phase": args.phase,
                "platform_label": args.platform_label,
                "physical_cores": args.physical_cores,
                "model": model_name,
                "configuration": label,
                "first_sustained_break_even_iterations": threshold if threshold is not None else "",
                "criterion": "bootstrap paired speedup median CI95 lower bound > 1 at this and all larger tested N",
            })

    suffix = args.phase
    _write_rows(output_dir / f"exp10_break_even_{suffix}_raw.csv", raw_rows)
    _write_rows(output_dir / f"exp10_break_even_{suffix}_summary.csv", summary_rows)
    _write_rows(output_dir / f"exp10_break_even_{suffix}_thresholds.csv", threshold_rows)
    return raw_rows, summary_rows, threshold_rows


def _rss_snapshot(root: psutil.Process) -> tuple[int, int, int, int]:
    coordinator_rss = 0
    worker_rss = 0
    count = 0
    try:
        coordinator_rss = int(root.memory_info().rss)
        count = 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, 0, 0, 0
    try:
        children = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for child in children:
        try:
            worker_rss += int(child.memory_info().rss)
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return coordinator_rss, worker_rss, coordinator_rss + worker_rss, count


def _run_monitored_memory_probe(
    source: str,
    config: GeneratorConfig,
    *,
    temp_dir: Path,
    tag: str,
    cost_model: InterpretableASTCostModel,
    monitor_interval: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    checkpoint = base._CHECKPOINT
    identity = base.simulation_identity(
        tag=tag, kind="targeted_memory_probe", source=source, config=config, cost_model=cost_model
    )
    if checkpoint is not None:
        cached = checkpoint.load(identity)
        if cached is not None:
            return cached

    compiled = base.compile_source(source, config, cost_model=cost_model)
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    script_path = temp_dir / f"{safe_tag}.py"
    script_path.write_text(compiled.script, encoding="utf-8")
    proc: subprocess.Popen[str] | None = None
    output_state: dict[str, Any] = {"start": None, "complete": None, "error": None, "tail": []}
    start_seen = threading.Event()

    def _reader(stream) -> None:
        for line in stream:
            if line.startswith(base.EVENT_PREFIX):
                try:
                    event = json.loads(line[len(base.EVENT_PREFIX):])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "start":
                    output_state["start"] = event
                    start_seen.set()
                elif event.get("type") == "complete":
                    output_state["complete"] = event
                elif event.get("type") == "error":
                    output_state["error"] = event
            else:
                output_state["tail"].append(line.rstrip())
                output_state["tail"] = output_state["tail"][-40:]

    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        reader = threading.Thread(target=_reader, args=(proc.stdout,), daemon=True)
        reader.start()
        root = psutil.Process(proc.pid)
        baseline_tree_rss = None
        peak_coord = 0
        peak_workers = 0
        peak_tree = 0
        peak_processes = 0
        samples = 0
        while proc.poll() is None or reader.is_alive():
            if (time.perf_counter() - t0) > timeout_seconds:
                proc.kill()
                proc.wait(timeout=10)
                raise TimeoutError(f"Memory probe exceeded {timeout_seconds} seconds: {tag}")
            if start_seen.is_set():
                coord, workers, tree, process_count = _rss_snapshot(root)
                if tree > 0:
                    if baseline_tree_rss is None:
                        baseline_tree_rss = tree
                    peak_coord = max(peak_coord, coord)
                    peak_workers = max(peak_workers, workers)
                    peak_tree = max(peak_tree, tree)
                    peak_processes = max(peak_processes, process_count)
                    samples += 1
            time.sleep(monitor_interval)
        reader.join(timeout=5)
        return_code = proc.wait(timeout=5)
    finally:
        script_path.unlink(missing_ok=True)

    if return_code != 0:
        raise RuntimeError(
            f"Memory probe failed ({return_code}) for {tag}: "
            f"{output_state['error'] or output_state['tail'][-12:]}"
        )
    complete = output_state["complete"]
    start = output_state["start"]
    if complete is None or start is None:
        raise RuntimeError(f"Memory probe did not emit start/complete events: {tag}")
    meta = complete["metadata"]
    baseline = int(baseline_tree_rss or 0)
    result = {
        "complete_event": complete,
        "start_event": start,
        "wall_seconds": time.perf_counter() - t0,
        "memory_monitor": {
            "method": "psutil_process_tree_rss_sampled",
            "psutil_version": psutil.__version__,
            "sampling_interval_seconds": monitor_interval,
            "samples": samples,
            "baseline_process_tree_rss_bytes": baseline,
            "peak_coordinator_rss_bytes": peak_coord,
            "peak_workers_rss_sum_bytes": peak_workers,
            "peak_process_tree_rss_sum_bytes": peak_tree,
            "peak_incremental_process_tree_rss_bytes": max(0, peak_tree - baseline),
            "peak_process_count": peak_processes,
        },
        "runtime_metadata": {
            key: meta.get(key) for key in (
                "worker_count", "use_process_pool", "batch_count", "max_in_flight_tasks",
                "max_pending_logical_blocks", "max_pending_logical_blocks_observed",
                "max_block_coverage_entries_observed", "max_branch_batch_stats_observed",
                "max_paired_batch_stats_observed", "max_paired_correlations_observed",
                "max_futures_observed", "max_completed_results_observed",
                "max_global_sketch_points", "logical_block_size", "batch_size",
                "task_submission_order", "task_delay_first_seconds", "peak_retained_coordinator_bytes",
                "retained_coordinator_payload_bytes",
            )
        },
    }
    if checkpoint is not None:
        checkpoint.save(identity, result)
    return result


def _fidelity_between(retain_event: dict[str, Any], bounded_event: dict[str, Any]) -> dict[str, Any]:
    retain_branches = {item["id"]: item for item in retain_event["branches"]}
    bounded_branches = {item["id"]: item for item in bounded_event["branches"]}
    branch_ids = sorted(set(retain_branches).intersection(bounded_branches))
    quantile_rel_errors: list[float] = []
    tv_values: list[float] = []
    exact_moments = True
    exact_counts = True

    def compare_item(retain_item: dict[str, Any], bounded_item: dict[str, Any]) -> None:
        nonlocal exact_moments, exact_counts
        for stat in ("avg", "std", "min", "max"):
            if retain_item["stats"].get(stat) != bounded_item["stats"].get(stat):
                exact_moments = False
        if int(retain_item.get("valid_results", 0)) != int(bounded_item.get("valid_results", 0)):
            exact_counts = False
        if int(retain_item["histogram"]["total_count"]) != int(bounded_item["histogram"]["total_count"]):
            exact_counts = False
        for key in ("p05", "p50", "p95"):
            reference = float(retain_item["stats"][key])
            observed = float(bounded_item["stats"][key])
            quantile_rel_errors.append(abs(observed - reference) / max(1e-12, abs(reference)) * 100.0)
        a = np.asarray([b["count"] for b in retain_item["histogram"]["bins"]], dtype=np.float64)
        b = np.asarray([b["count"] for b in bounded_item["histogram"]["bins"]], dtype=np.float64)
        total = max(1.0, float(retain_item["histogram"]["total_count"]))
        tv_values.append(float(np.abs(a - b).sum() / (2.0 * total) * 100.0))

    for branch_id in branch_ids:
        compare_item(retain_branches[branch_id], bounded_branches[branch_id])
    retain_paired = retain_event.get("paired")
    bounded_paired = bounded_event.get("paired")
    if retain_paired is not None and bounded_paired is not None:
        compare_item(retain_paired, bounded_paired)
        corr_a = retain_paired.get("correlation_plus_minus_mean")
        corr_b = bounded_paired.get("correlation_plus_minus_mean")
        if corr_a != corr_b:
            exact_moments = False
    return {
        "exact_moments_preserved": exact_moments,
        "exact_counts_preserved": exact_counts,
        "max_quantile_relative_error_pct": max(quantile_rel_errors, default=0.0),
        "max_histogram_tv_distance_pct": max(tv_values, default=0.0),
        "branches_compared": len(branch_ids),
        "paired_delta_compared": retain_paired is not None and bounded_paired is not None,
    }


def _run_memory_study(args: argparse.Namespace, *, temp_dir: Path,
                      cost_model: InterpretableASTCostModel) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = args.output_dir.expanduser().resolve()
    max_n = args.max_memory_iterations or (50_000_000 if args.profile == "aws-4core" else 20_000_000)
    bounded_levels = [n for n in MEMORY_LEVELS_BASE if n <= max_n]
    if args.profile == "aws-4core" and max_n >= 50_000_000:
        bounded_levels.append(50_000_000)
    retain_levels = [n for n in MEMORY_RETAIN_LEVELS if n <= max_n]
    if args.include_retain_5m and 5_000_000 not in retain_levels and max_n >= 5_000_000:
        retain_levels.append(5_000_000)
    configs = _memory_configs(args.physical_cores)
    raw_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    structures_rows: list[dict[str, Any]] = []
    retained_lookup: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    bounded_lookup: dict[tuple[str, int, str, int], dict[str, Any]] = {}

    def run_cell(model: str, iterations: int, cfg: dict[str, Any], memory_mode: str,
                 order: str, rep: int, seed: int, delay: float, stratum: str) -> dict[str, Any]:
        source = base.source_with_iterations(model, iterations)
        config = _config_for(
            seed=seed,
            iterations=iterations,
            mode=cfg["mode"],
            workers=cfg["workers"],
            memory_mode=memory_mode,
            task_order=order,
            delay_first_seconds=delay,
        )
        tag = f"exp11_{stratum}_{model}_{iterations}_{cfg['label']}_{memory_mode}_{order}_r{rep}"
        result = _run_monitored_memory_probe(
            source, config, temp_dir=temp_dir, tag=tag, cost_model=cost_model,
            monitor_interval=args.monitor_interval, timeout_seconds=args.memory_timeout_seconds,
        )
        event = result["complete_event"]
        meta = event["metadata"]
        monitor = result["memory_monitor"]
        row = {
            "stratum": stratum,
            "platform_label": args.platform_label,
            "physical_cores": args.physical_cores,
            "logical_cpus": os.cpu_count(),
            "model": model,
            "iterations": iterations,
            "memory_mode": memory_mode,
            "worker_configuration": cfg["label"],
            "scheduler_mode": cfg["mode"],
            "requested_workers": cfg["workers"],
            "effective_workers": int(meta["worker_count"]),
            "use_process_pool": bool(meta.get("use_process_pool")),
            "task_submission_order": order,
            "task_delay_first_seconds": delay,
            "repetition": rep,
            "seed": seed,
            "wall_seconds": float(result["wall_seconds"]),
            "retained_payload_peak_bytes": int(meta.get("peak_retained_coordinator_bytes", 0)),
            "retained_payload_final_bytes": int(meta.get("retained_coordinator_payload_bytes", 0)),
            "rss_baseline_tree_bytes": int(monitor["baseline_process_tree_rss_bytes"]),
            "rss_coordinator_peak_bytes": int(monitor["peak_coordinator_rss_bytes"]),
            "rss_workers_sum_peak_bytes": int(monitor["peak_workers_rss_sum_bytes"]),
            "rss_process_tree_sum_peak_bytes": int(monitor["peak_process_tree_rss_sum_bytes"]),
            "rss_process_tree_incremental_peak_bytes": int(monitor["peak_incremental_process_tree_rss_bytes"]),
            "process_count_peak": int(monitor["peak_process_count"]),
            "rss_samples": int(monitor["samples"]),
            "rss_sampling_interval_seconds": float(monitor["sampling_interval_seconds"]),
            "psutil_version": monitor["psutil_version"],
            "valid_results": int(event["valid_results"]),
            "discarded_results": int(event["discarded_results"]),
            "result_sha256": base.result_sha256(event),
            "max_in_flight_tasks": meta.get("max_in_flight_tasks"),
            "max_futures_observed": meta.get("max_futures_observed"),
            "max_completed_results_observed": meta.get("max_completed_results_observed"),
            "max_pending_logical_blocks_observed": meta.get("max_pending_logical_blocks_observed"),
            "max_block_coverage_entries_observed": meta.get("max_block_coverage_entries_observed"),
            "max_branch_batch_stats_observed": meta.get("max_branch_batch_stats_observed"),
            "max_paired_batch_stats_observed": meta.get("max_paired_batch_stats_observed"),
            "max_paired_correlations_observed": meta.get("max_paired_correlations_observed"),
            "max_global_sketch_points": meta.get("max_global_sketch_points"),
            "max_pending_logical_blocks_limit": meta.get("max_pending_logical_blocks"),
            "logical_block_size": meta.get("logical_block_size"),
            "batch_size": meta.get("batch_size"),
        }
        raw_rows.append(row)
        structures_rows.append({k: row[k] for k in (
            "platform_label", "model", "iterations", "memory_mode", "worker_configuration",
            "task_submission_order", "repetition", "max_in_flight_tasks", "max_futures_observed",
            "max_completed_results_observed", "max_pending_logical_blocks_observed",
            "max_block_coverage_entries_observed", "max_branch_batch_stats_observed",
            "max_paired_batch_stats_observed", "max_paired_correlations_observed",
            "max_global_sketch_points", "max_pending_logical_blocks_limit", "logical_block_size", "batch_size",
        )})
        return result

    # Baseline memory scaling: retain_all is the positive control; bounded is the target.
    for cfg in configs:
        for model in MEMORY_MODELS:
            for n in bounded_levels:
                for rep in range(1, args.memory_reps + 1):
                    seed = 5_100_000 + rep
                    result = run_cell(model, n, cfg, "bounded", "as_completed", rep, seed, 0.0, "scaling")
                    bounded_lookup[(model, n, cfg["label"], rep)] = result
            for n in retain_levels:
                for rep in range(1, args.retain_reps + 1):
                    seed = 5_100_000 + rep
                    result = run_cell(model, n, cfg, "retain_all", "as_completed", rep, seed, 0.0, "scaling")
                    retained_lookup[(model, n, cfg["label"], rep)] = result

    # Fidelity is paired by model, N, worker configuration and seed.
    for key, bounded_result in bounded_lookup.items():
        retain_result = retained_lookup.get(key)
        if retain_result is None:
            continue
        model, n, worker_label, rep = key
        fidelity = _fidelity_between(retain_result["complete_event"], bounded_result["complete_event"])
        fidelity_rows.append({
            "platform_label": args.platform_label,
            "model": model,
            "iterations": n,
            "worker_configuration": worker_label,
            "repetition": rep,
            **fidelity,
        })

    # Adversarial ordering: only bounded mode, maximum available workers, high N.
    max_worker_cfg = max(configs, key=lambda item: item["workers"])
    stress_levels = [n for n in (10_000_000, 20_000_000) if n <= max_n]
    for model in MEMORY_MODELS:
        for n in stress_levels:
            for order in MEMORY_STRESS_ORDERS:
                for rep in range(1, args.stress_reps + 1):
                    seed = 6_100_000 + rep
                    delay = args.first_task_delay_seconds if order == "delayed_first" else 0.0
                    run_cell(model, n, max_worker_cfg, "bounded", order, rep, seed, delay, "order_stress")

    # Summaries and empirical log-log slopes. Slopes are descriptive, not a proof of O(1).
    summary_rows: list[dict[str, Any]] = []
    group_fields = ("stratum", "model", "iterations", "memory_mode", "worker_configuration", "task_submission_order")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        rss = [float(r["rss_process_tree_sum_peak_bytes"]) for r in rows]
        incremental = [float(r["rss_process_tree_incremental_peak_bytes"]) for r in rows]
        payload = [float(r["retained_payload_peak_bytes"]) for r in rows]
        seed = _stable_int("|".join(map(str, key)))
        rss_ci = _bootstrap_ci(rss, seed=seed)
        inc_ci = _bootstrap_ci(incremental, seed=seed + 1)
        payload_ci = _bootstrap_ci(payload, seed=seed + 2)
        summary_rows.append({
            "platform_label": args.platform_label,
            **dict(zip(group_fields, key)),
            "runs": len(rows),
            "median_wall_seconds": float(np.median([float(r["wall_seconds"]) for r in rows])),
            "median_retained_payload_bytes": float(np.median(payload)),
            "retained_payload_ci95_low": payload_ci[0],
            "retained_payload_ci95_high": payload_ci[1],
            "median_process_tree_rss_peak_bytes": float(np.median(rss)),
            "process_tree_rss_ci95_low": rss_ci[0],
            "process_tree_rss_ci95_high": rss_ci[1],
            "median_incremental_process_tree_rss_bytes": float(np.median(incremental)),
            "incremental_rss_ci95_low": inc_ci[0],
            "incremental_rss_ci95_high": inc_ci[1],
            "median_max_futures_observed": float(np.median([float(r["max_futures_observed"] or 0) for r in rows])),
            "median_max_completed_results_observed": float(np.median([float(r["max_completed_results_observed"] or 0) for r in rows])),
            "median_max_pending_logical_blocks_observed": float(np.median([float(r["max_pending_logical_blocks_observed"] or 0) for r in rows])),
            "median_max_paired_correlations_observed": float(np.median([float(r["max_paired_correlations_observed"] or 0) for r in rows])),
        })

    slope_rows: list[dict[str, Any]] = []
    slope_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        if row["stratum"] != "scaling" or row["task_submission_order"] != "as_completed":
            continue
        slope_groups[(row["model"], row["memory_mode"], row["worker_configuration"], row["platform_label"])].append(row)
    for key, rows in slope_groups.items():
        rows = sorted(rows, key=lambda r: int(r["iterations"]))
        def _slope(field: str) -> tuple[float | str, float | str]:
            xs = np.asarray([math.log(float(r["iterations"])) for r in rows if float(r[field]) > 0], dtype=np.float64)
            ys = np.asarray([math.log(float(r[field])) for r in rows if float(r[field]) > 0], dtype=np.float64)
            if len(xs) < 3:
                return "", ""
            beta, intercept = np.polyfit(xs, ys, 1)
            predicted = beta * xs + intercept
            ss_res = float(np.sum((ys - predicted) ** 2))
            ss_tot = float(np.sum((ys - float(np.mean(ys))) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
            return float(beta), float(r2)
        beta_payload, r2_payload = _slope("median_retained_payload_bytes")
        beta_rss, r2_rss = _slope("median_process_tree_rss_peak_bytes")
        beta_incremental, r2_incremental = _slope("median_incremental_process_tree_rss_bytes")
        slope_rows.append({
            "model": key[0], "memory_mode": key[1], "worker_configuration": key[2],
            "platform_label": key[3], "iteration_levels": ";".join(str(int(r["iterations"])) for r in rows),
            "beta_retained_payload": beta_payload, "r2_retained_payload": r2_payload,
            "beta_process_tree_rss": beta_rss, "r2_process_tree_rss": r2_rss,
            "beta_incremental_process_tree_rss": beta_incremental,
            "r2_incremental_process_tree_rss": r2_incremental,
            "interpretation": "descriptive log-log slope; not a formal proof of asymptotic complexity",
        })

    _write_rows(output_dir / "exp11_memory_scaling_raw.csv", raw_rows)
    _write_rows(output_dir / "exp11_memory_scaling_summary.csv", summary_rows)
    _write_rows(output_dir / "exp11_memory_fidelity.csv", fidelity_rows)
    _write_rows(output_dir / "exp11_memory_structural_bounds.csv", structures_rows)
    _write_rows(output_dir / "exp11_memory_scaling_slopes.csv", slope_rows)
    return raw_rows, summary_rows, fidelity_rows, slope_rows


def _write_environment(args: argparse.Namespace, study: str, phase: str, cost_model: InterpretableASTCostModel) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    topology = base.detect_cpu_topology()
    payload = {
        "protocol_version": "1.0.0",
        "project_version": "1.3.1",
        "study": study,
        "phase": phase,
        "campaign_label": args.campaign_label,
        "platform_label": args.platform_label,
        "profile": args.profile,
        "physical_cores_declared": args.physical_cores,
        "detected_cpu_topology": topology,
        "logical_cpus": os.cpu_count(),
        "python_version": sys.version,
        "platform": sys.platform,
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "cost_model_weights_seconds_per_sample": list(cost_model.weights_seconds_per_sample),
        "scheduler_policy_modified": False,
        "scheduler_policy_note": "EXP-10/11 observe the existing scheduler; only a zero-default first-task delay hook is added for bounded-memory stress tests.",
        "batch_size": 25_000,
        "logical_block_size": 5_000,
        "max_in_flight_tasks_per_worker": 2,
        "max_pending_logical_blocks": 64,
        "max_global_sketch_points": 2048,
        "monitor_interval_seconds": args.monitor_interval,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "targeted_script_sha256": _sha256_file(Path(__file__).resolve()),
        "generator_sha256": _sha256_file(ROOT / "src" / "montecarlo_dsl" / "generator.py"),
    }
    _write_json(output_dir / f"targeted_{study}_{phase}_environment.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MC DSL v1.3.1 targeted EXP-10 break-even and EXP-11 bounded-memory scaling studies"
    )
    parser.add_argument("--study", choices=("break-even", "memory"), required=True)
    parser.add_argument("--phase", choices=("exploratory", "confirmatory"), default="exploratory",
                        help="Only applies to --study break-even")
    parser.add_argument("--profile", choices=("windows-2core", "aws-4core"), required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--physical-cores", type=int, choices=(2, 4), required=True)
    parser.add_argument("--instance-type", default=None)
    parser.add_argument("--cloud-region", default=None)
    parser.add_argument("--campaign-label", default="mc-dsl-v1.3.1-targeted")
    parser.add_argument("--cost-model-weights", type=_parse_weights, required=True,
                        help="Four fitted weights from this platform's v1.3.1 environment.json, comma-separated")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "benchmarks" / "checkpoints" / "v1.3.1_targeted")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmarks" / "results" / "targeted" / "v1.3.1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--exploration-reps", type=int, default=10)
    parser.add_argument("--large-anchor-reps", type=int, default=5)
    parser.add_argument("--confirmatory-reps", type=int, default=30)
    parser.add_argument("--confirmatory-iterations", default="",
                        help="Comma-separated N values selected after the exploratory phase")
    parser.add_argument("--memory-reps", type=int, default=5)
    parser.add_argument("--retain-reps", type=int, default=3)
    parser.add_argument("--stress-reps", type=int, default=3)
    parser.add_argument("--max-memory-iterations", type=int, default=None,
                        help="Default: 20M on Windows profile, 50M on AWS profile")
    parser.add_argument("--include-retain-5m", action="store_true")
    parser.add_argument("--monitor-interval", type=float, default=0.01)
    parser.add_argument("--memory-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--first-task-delay-seconds", type=float, default=FIRST_TASK_DELAY_SECONDS)
    args = parser.parse_args()

    if args.profile == "windows-2core" and args.physical_cores != 2:
        parser.error("windows-2core profile requires --physical-cores 2")
    if args.profile == "aws-4core" and args.physical_cores != 4:
        parser.error("aws-4core profile requires --physical-cores 4")
    if min(args.progress_every, args.exploration_reps, args.large_anchor_reps,
           args.confirmatory_reps, args.memory_reps, args.retain_reps, args.stress_reps) <= 0:
        parser.error("all repetition counts and progress interval must be positive")
    if args.monitor_interval <= 0 or args.memory_timeout_seconds <= 0 or args.first_task_delay_seconds <= 0:
        parser.error("monitor interval, timeout and first-task delay must be positive")
    if args.study == "memory" and args.phase != "exploratory":
        parser.error("--phase only applies to --study break-even")

    if args.profile == "aws-4core":
        detected = base.detect_cpu_topology().get("physical_core_count")
        if detected is not None and int(detected) != 4:
            parser.error(f"AWS profile expects 4 physical cores; detected {detected}")

    cost_model = _make_cost_model(args.cost_model_weights)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    study_phase = args.phase if args.study == "break-even" else "scaling"
    store = _campaign_store(args, args.study, study_phase)
    base._CHECKPOINT = store
    store.save_status("running")
    try:
        with tempfile.TemporaryDirectory(prefix=f"mcdsl_{args.study}_") as work:
            temp_dir = Path(work)
            _write_environment(args, args.study, study_phase, cost_model)
            if args.study == "break-even":
                raw, summary, thresholds = _run_break_even(args, temp_dir=temp_dir, cost_model=cost_model)
                print(f"[exp10] raw={len(raw)} summary={len(summary)} thresholds={len(thresholds)}", flush=True)
            else:
                raw, summary, fidelity, slopes = _run_memory_study(args, temp_dir=temp_dir, cost_model=cost_model)
                print(f"[exp11] raw={len(raw)} summary={len(summary)} fidelity={len(fidelity)} slopes={len(slopes)}", flush=True)
        store.save_status("completed")
        print(f"TARGETED {args.study.upper()} STUDY COMPLETED -> {args.output_dir}", flush=True)
        return 0
    except BaseException:
        store.save_status("interrupted")
        raise
    finally:
        base._CHECKPOINT = None


if __name__ == "__main__":
    raise SystemExit(main())
