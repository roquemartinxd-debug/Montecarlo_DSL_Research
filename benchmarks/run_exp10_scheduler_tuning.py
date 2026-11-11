"""Opt-in EXP-10 scheduler overhead calibration A/B study (no scheduler code changes).

This is a NEW experiment, not the v1.3.1 EXP-10 protocol. The tuned arm
changes only the estimated process-pool startup and per-task IPC costs passed
through InterpretableASTCostModel. The original scheduler kernel, AST weights,
RNG, block reduction, and physical task iterator are untouched.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
from collections import defaultdict
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for project_dir in (ROOT, ROOT / "src"):
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

import numpy as np

from benchmarks import run_v1_3_experiments as base
from benchmarks.campaign_checkpoint import CampaignCheckpoint, campaign_identity, _write_json_atomic
from montecarlo_dsl import GeneratorConfig, InterpretableASTCostModel

# Exploratory hypothesis inferred from AWS EXP-10 fixed-pool timings, NOT
# independently measured pool-startup/IPC components or a validated optimum.
AWS_EXP10_CANDIDATE_SPAWN_SECONDS = 0.016
AWS_EXP10_CANDIDATE_IPC_SECONDS = 0.00030
AWS_AST_WEIGHTS = (4.118662634957425e-08, 7e-09,
                   2.4547267526488982e-08, 5.5e-09)
MODELS = ("normal_mean", "linear_combination", "heavy_expression", "quadratic_pm")
LEVELS = (100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000)
ANCHORS = (5_000_000, 10_000_000)


def _valid_nonnegative_float(value: str) -> float:
    try:
        v = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a finite non-negative float") from exc
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative float")
    return v


def _weights(value: str) -> tuple[float, float, float, float]:
    try:
        vals = tuple(float(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected four comma-separated floats") from exc
    if len(vals) != 4 or any(not math.isfinite(x) or x < 0 for x in vals):
        raise argparse.ArgumentTypeError("expected four finite non-negative coefficients")
    return vals  # type: ignore[return-value]


def _levels(value: str) -> tuple[int, ...]:
    try:
        nums = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not nums or any(x <= 0 or x > 50_000_000 for x in nums) or len(set(nums)) != len(nums):
        raise argparse.ArgumentTypeError("iteration levels must be unique, 1..50,000,000")
    return nums


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def create_cost_models(
    weights: tuple[float, float, float, float],
    *,
    spawn_seconds: float = AWS_EXP10_CANDIDATE_SPAWN_SECONDS,
    ipc_seconds: float = AWS_EXP10_CANDIDATE_IPC_SECONDS,
) -> tuple[InterpretableASTCostModel, InterpretableASTCostModel]:
    """Keep baseline AST weights and scheduler algorithm; change only two costs."""
    if any(not math.isfinite(x) or x < 0 for x in (*weights, spawn_seconds, ipc_seconds)):
        raise ValueError("weights and overheads must be finite and non-negative")
    baseline = InterpretableASTCostModel(weights_seconds_per_sample=tuple(weights))
    experimental = replace(
        baseline,
        parallel_pool_spawn_seconds=float(spawn_seconds),
        parallel_task_ipc_seconds=float(ipc_seconds),
    )
    return baseline, experimental


def _configs(physical_cores: int) -> tuple[dict[str, Any], ...]:
    return (
        {"label": "serial_in_process", "mode": "ast_only", "workers": 1, "model": "baseline"},
        {"label": "fixed_pool_w2", "mode": "fixed", "workers": 2, "model": "baseline"},
        {"label": "fixed_pool_w4", "mode": "fixed", "workers": 4, "model": "baseline"},
        {"label": "adaptive_original", "mode": "hybrid", "workers": physical_cores, "model": "baseline"},
        {"label": "adaptive_tuned", "mode": "hybrid", "workers": physical_cores, "model": "experimental"},
    )


def _config(seed: int, n: int, mode: str, workers: int) -> GeneratorConfig:
    return GeneratorConfig(
        seed=seed,
        batch_size=25_000,
        logical_block_size=5_000,
        max_workers=workers,
        scheduler_mode=mode,
        seed_strategy="philox_logical_block",
        memory_mode="bounded",
        optimize=True,
        max_iterations=max(n, 50_000_000),
    )


def _bootstrap_interval(samples: list[float], label: str) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    values = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(_stable_int(label))
    indexes = rng.integers(0, len(values), size=(5000, len(values)))
    medians = np.median(values[indexes], axis=1)
    return tuple(map(float, np.quantile(medians, [0.025, 0.975])))  # type: ignore[return-value]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), int(row["iterations"]))].append(row)
    out = []
    for (name, n), group in sorted(grouped.items()):
        by_label: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in group:
            key = int(row["repetition"])
            label = str(row["configuration"])
            if key in by_label[label]:
                raise RuntimeError(f"Duplicate cell: {name} {n} {label} r{key}")
            by_label[label][key] = row
        expected_labels = {item["label"] for item in _configs(4)}
        if set(by_label) != expected_labels:
            raise RuntimeError(f"Missing configuration for {name} {n}")
        serial = by_label["serial_in_process"]
        original = by_label["adaptive_original"]
        tuned = by_label["adaptive_tuned"]
        for rep in serial:
            hashes = {by_label[label][rep]["result_sha256"] for label in expected_labels}
            if len(hashes) != 1:
                raise RuntimeError(f"RNG/result invariance failure for {name} N={n} rep={rep}")
        for label, rep_map in sorted(by_label.items()):
            ordered = sorted(rep_map)
            times = [float(rep_map[r]["execution_seconds"]) for r in ordered]
            speedups = [float(serial[r]["execution_seconds"]) / max(1e-12, float(rep_map[r]["execution_seconds"])) for r in ordered]
            versus_original = [float(original[r]["execution_seconds"]) / max(1e-12, float(rep_map[r]["execution_seconds"])) for r in ordered]
            ci = _bootstrap_interval(speedups, f"{name}|{n}|{label}|serial")
            orig_ci = _bootstrap_interval(versus_original, f"{name}|{n}|{label}|original")
            out.append({
                "model": name,
                "iterations": n,
                "configuration": label,
                "repetitions": len(times),
                "median_execution_seconds": float(np.median(times)),
                "median_effective_workers": float(np.median([int(rep_map[r]["effective_workers"]) for r in ordered])),
                "median_speedup_vs_serial": float(np.median(speedups)),
                "speedup_vs_serial_ci95_low": ci[0],
                "speedup_vs_serial_ci95_high": ci[1],
                "median_speedup_vs_adaptive_original": float(np.median(versus_original)),
                "speedup_vs_original_ci95_low": orig_ci[0],
                "speedup_vs_original_ci95_high": orig_ci[1],
                "all_result_hashes_match_within_platform": True,
            })
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No experimental rows to write")
    import tempfile as _tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with _tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                                           prefix=".exp10_tuning_", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWS EXP-10 A/B: same scheduler, alternative overhead parameters")
    parser.add_argument("--profile", choices=("aws-4core",), required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--physical-cores", type=int, default=4)
    parser.add_argument("--instance-type", default="c7i.2xlarge")
    parser.add_argument("--cloud-region", default="us-east-2")
    parser.add_argument("--campaign-label", required=True)
    parser.add_argument("--cost-model-weights", type=_weights, required=True)
    parser.add_argument("--tuned-spawn-seconds", type=_valid_nonnegative_float,
                        default=AWS_EXP10_CANDIDATE_SPAWN_SECONDS)
    parser.add_argument("--tuned-ipc-seconds", type=_valid_nonnegative_float,
                        default=AWS_EXP10_CANDIDATE_IPC_SECONDS)
    parser.add_argument("--iterations", type=_levels, default=LEVELS)
    parser.add_argument("--include-large-anchors", action="store_true")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=7_100_000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.physical_cores != 4:
        parser.error("this exploratory candidate is defined only for the AWS four-core profile")
    if args.repetitions < 1 or args.seed_base < 0:
        parser.error("repetitions must be >=1 and seed-base must be non-negative")
    return args


def run(args: argparse.Namespace) -> int:
    baseline, experimental = create_cost_models(
        args.cost_model_weights,
        spawn_seconds=args.tuned_spawn_seconds,
        ipc_seconds=args.tuned_ipc_seconds,
    )
    models = {"baseline": baseline, "experimental": experimental}
    plan = [(model, n) for model in MODELS for n in args.iterations]
    if args.include_large_anchors:
        plan.extend((model, n) for model in ("normal_mean", "quadratic_pm") for n in ANCHORS if (model, n) not in plan)

    out_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if out_dir == checkpoint_dir or out_dir in checkpoint_dir.parents or checkpoint_dir in out_dir.parents:
        raise ValueError("output-dir and checkpoint-dir must be separate directories")
    script_sha = _sha256(Path(__file__))
    identity = campaign_identity(
        ROOT,
        confirmatory_reps=0,
        pilot_reps=args.repetitions,
        ablation_reps=0,
        campaign_label=args.campaign_label,
        experiment_profile="exp10-scheduler-overhead-aws-4core",
        instance_type=args.instance_type,
        cloud_region=args.cloud_region,
        output_dir=str(out_dir),
    )
    identity.update({
        "study": "exp10-scheduler-overhead-ab",
        "protocol_version": "1.0.0",
        "runner_sha256": script_sha,
        "platform_label": args.platform_label,
        "declared_physical_cores": args.physical_cores,
        "iterations": list(args.iterations),
        "include_large_anchors": args.include_large_anchors,
        "models": list(MODELS),
        "configurations": list(c["label"] for c in _configs(args.physical_cores)),
        "seed_base": args.seed_base,
        "repetitions": args.repetitions,
        "weights": list(args.cost_model_weights),
        "baseline_spawn_seconds": baseline.parallel_pool_spawn_seconds,
        "baseline_ipc_seconds": baseline.parallel_task_ipc_seconds,
        "tuned_spawn_seconds": experimental.parallel_pool_spawn_seconds,
        "tuned_ipc_seconds": experimental.parallel_task_ipc_seconds,
        "hypothesis": "AWS fixed-pool measurements suggest lower startup/IPC costs; not yet confirmed",
    })
    store = CampaignCheckpoint(checkpoint_dir, identity, resume=args.resume, progress_every=args.progress_every)
    if base._CHECKPOINT is not None:
        raise RuntimeError("Another experimental campaign is already using the checkpoint store")
    base._CHECKPOINT = store
    store.save_status("running")
    rows: list[dict[str, Any]] = []
    try:
        environment = {
            "identity": identity,
            "baseline_cost_model": {"weights": list(baseline.weights_seconds_per_sample),
                                    "spawn_seconds": baseline.parallel_pool_spawn_seconds,
                                    "ipc_seconds": baseline.parallel_task_ipc_seconds},
            "experimental_cost_model": {"weights": list(experimental.weights_seconds_per_sample),
                                        "spawn_seconds": experimental.parallel_pool_spawn_seconds,
                                        "ipc_seconds": experimental.parallel_task_ipc_seconds},
            "scheduler_kernel_modified": False,
            "fixed_worker_execution_modified": False,
            "pilot_weighting_modified": False,
            "parallel_speedup_margin": 1.12,
            "host": {"platform": platform.platform(), "logical_cpus": os.cpu_count()},
        }
        _write_json_atomic(out_dir / "scheduler_tuning_environment.json", environment)
        with tempfile.TemporaryDirectory(prefix="mcdsl_exp10_tuning_") as work:
            tempdir = Path(work)
            for model, n in plan:
                source = base.source_with_iterations(model, n)
                configs = _configs(args.physical_cores)
                for variant in configs:
                    warm_config = _config(91_337, n, variant["mode"], variant["workers"])
                    base.run_simulation(
                        source, warm_config, temp_dir=tempdir,
                        tag=f"warmup_tuning_{model}_{n}_{variant['label']}",
                        cost_model=models[variant["model"]],
                    )
                for rep in range(1, args.repetitions + 1):
                    seed = args.seed_base + rep
                    order = list(configs)
                    random.Random(_stable_int(f"{args.platform_label}|{model}|{n}|{rep}|exp10-tuning")).shuffle(order)
                    row_group = []
                    for variant in order:
                        cm = models[variant["model"]]
                        config = _config(seed, n, variant["mode"], variant["workers"])
                        tag = f"exp10_tuning_{model}_{n}_{variant['label']}_r{rep}"
                        result = base.run_simulation(source, config, temp_dir=tempdir, tag=tag, cost_model=cm)
                        meta = result["complete_event"]["metadata"]
                        row = {
                            "platform_label": args.platform_label,
                            "model": model,
                            "iterations": n,
                            "configuration": variant["label"],
                            "scheduler_mode": variant["mode"],
                            "repetition": rep,
                            "seed": seed,
                            "requested_workers": variant["workers"],
                            "effective_workers": int(meta["worker_count"]),
                            "use_process_pool": bool(meta.get("use_process_pool")),
                            "execution_seconds": float(result["execution_seconds"]),
                            "pilot_elapsed_seconds": meta.get("pilot_elapsed_seconds"),
                            "simulation_compute_seconds": meta.get("simulation_compute_seconds"),
                            "pool_spawn_cost_assumed_seconds": cm.parallel_pool_spawn_seconds,
                            "task_ipc_cost_assumed_seconds": cm.parallel_task_ipc_seconds,
                            "result_sha256": result["sha256"],
                        }
                        row_group.append(row)
                        rows.append(row)
                    if len({r["result_sha256"] for r in row_group}) != 1:
                        raise RuntimeError(f"Result hash disagreement: {model} N={n} repetition={rep}")
                print(f"[exp10-tuning] completed {model} N={n:,} reps={args.repetitions}", flush=True)
        summary = summarize(rows)
        _write_csv(out_dir / "exp10_scheduler_tuning_raw.csv", rows)
        _write_csv(out_dir / "exp10_scheduler_tuning_summary.csv", summary)
        store.save_status("completed")
        print(f"EXP10 SCHEDULER TUNING COMPLETED raw={len(rows)} summary={len(summary)} -> {out_dir}", flush=True)
        return 0
    except BaseException:
        store.save_status("interrupted")
        raise
    finally:
        base._CHECKPOINT = None


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
