from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from montecarlo_dsl.compiler import compile_source
from montecarlo_dsl.generator import GeneratorConfig


EVENT_PREFIX = "MC_DSL_EVENT:"

DEFAULT_SEED = 101
DEFAULT_ITERATIONS = 1_000_000
DEFAULT_BATCH_SIZE = 25_000
DEFAULT_WORKERS = (1, 2, 4)
DEFAULT_REPETITIONS = 10
DEFAULT_HISTOGRAM_BINS = 30
DEFAULT_LOGICAL_BLOCK_SIZE = 5_000
DEFAULT_CONTROL_MODEL = "normal_mean"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "rng_reproducibility" / "v1.3.1"

MODELS = {
    "normal_mean": ROOT / "benchmarks" / "models" / "analytic" / "normal_mean.dsl",
    "uniform_mean": ROOT / "benchmarks" / "models" / "analytic" / "uniform_mean.dsl",
    "discrete_mean": ROOT / "benchmarks" / "models" / "analytic" / "discrete_mean.dsl",
    "linear_combination": ROOT / "benchmarks" / "models" / "analytic" / "linear_combination.dsl",
}

MAIN_FIELDS = [
    "model",
    "repetition",
    "seed",
    "iterations",
    "batch_size",
    "requested_workers",
    "actual_workers",
    "batch_count",
    "elapsed_seconds",
    "result_sha256",
    "matches_reference",
    "result_mean",
    "result_variance",
    "result_stddev",
    "p05",
    "p50",
    "p95",
    "count",
    "valid_rate",
    "discard_rate",
]

SUMMARY_FIELDS = [
    "model",
    "seed",
    "iterations",
    "batch_size",
    "workers_tested",
    "repetitions_per_worker",
    "total_runs",
    "unique_result_hashes",
    "exact_reproducibility",
    "reference_sha256",
]

CONTROL_FIELDS = [
    "control",
    "seed_strategy",
    "model",
    "seed",
    "iterations",
    "batch_size",
    "workers",
    "result_sha256",
    "result_mean",
    "result_variance",
    "expected_relation",
    "observed_relation",
    "pass",
]


def git_value(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return "unknown"
    return process.stdout.strip()


def git_is_dirty() -> bool | None:
    process = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        return None
    return bool(process.stdout.strip())


def source_with_iterations(model: str, iterations: int) -> str:
    path = MODELS[model]
    source = path.read_text(encoding="utf-8")
    source, replacements = re.subn(
        r"\\iter\{\d+\}",
        lambda _: f"\\iter{{{iterations}}}",
        source,
    )
    if replacements != 1:
        raise RuntimeError(
            f"Expected exactly one \\iter block in {path}, found {replacements}"
        )
    return source


def parse_complete_event(stdout: str) -> dict[str, Any]:
    complete: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if not line.startswith(EVENT_PREFIX):
            continue
        event = json.loads(line[len(EVENT_PREFIX) :])
        if event.get("type") == "complete":
            complete = event
    if complete is None:
        raise RuntimeError("No MC_DSL_EVENT complete event found.")
    return complete


def canonical_result(complete: dict[str, Any]) -> dict[str, Any]:
    """Return only final numerical outputs that must be worker-invariant.

    Execution metadata is intentionally excluded because worker_count differs by
    construction and platform metadata is not a numerical simulation result.
    """
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


def canonical_json(complete: dict[str, Any]) -> str:
    return json.dumps(
        canonical_result(complete),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def result_hash(complete: dict[str, Any]) -> str:
    encoded = canonical_json(complete).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_dsl(
    *,
    model: str,
    iterations: int,
    seed: int,
    batch_size: int,
    workers: int,
    histogram_bins: int,
    timeout_seconds: float,
    seed_strategy: str = "philox_logical_block",
    scheduler_mode: str = "hybrid",
) -> tuple[dict[str, Any], float]:
    source = source_with_iterations(model, iterations)
    compiled = compile_source(
        source,
        GeneratorConfig(
            seed=seed,
            batch_size=batch_size,
            logical_block_size=DEFAULT_LOGICAL_BLOCK_SIZE,
            max_workers=workers,
            histogram_bins=histogram_bins,
            seed_strategy=seed_strategy,
            scheduler_mode=scheduler_mode,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="mcdsl_rng_") as temp:
        script_path = Path(temp) / "simulation.py"
        script_path.write_text(compiled.script, encoding="utf-8")

        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - started

    if process.returncode != 0:
        raise RuntimeError(
            "DSL execution failed.\n"
            f"Model: {model}\n"
            f"Iterations: {iterations}\n"
            f"Seed: {seed}\n"
            f"Batch size: {batch_size}\n"
            f"Workers: {workers}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    return parse_complete_event(process.stdout), elapsed


def primary_stats(complete: dict[str, Any]) -> dict[str, Any]:
    stats = complete.get("stats") or {}
    return {
        "result_mean": stats.get("avg"),
        "result_variance": stats.get("var"),
        "result_stddev": stats.get("std"),
        "p05": stats.get("p05"),
        "p50": stats.get("p50"),
        "p95": stats.get("p95"),
        "count": stats.get("count"),
        "valid_rate": stats.get("valid_rate"),
        "discard_rate": stats.get("discard_rate"),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_complete_event(
    raw_dir: Path,
    filename: str,
    complete: dict[str, Any],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / filename).write_text(
        json.dumps(complete, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_environment(
    output_dir: Path,
    *,
    models: list[str],
    seed: int,
    iterations: int,
    batch_size: int,
    workers: tuple[int, ...],
    repetitions: int,
    histogram_bins: int,
) -> None:
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": git_is_dirty(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpu_count": os.cpu_count(),
        "models": models,
        "simulation_seed": seed,
        "iterations": iterations,
        "batch_size": batch_size,
        "batch_count": (iterations + batch_size - 1) // batch_size,
        "logical_block_size": DEFAULT_LOGICAL_BLOCK_SIZE,
        "seed_strategy": "philox_logical_block",
        "scheduler_mode": "hybrid",
        "workers": list(workers),
        "repetitions_per_worker": repetitions,
        "histogram_bins": histogram_bins,
        "rng_strategy": "numpy.SeedSequence([MASTER_SEED, logical_block_index, var_index]) -> Philox",
        "comparison_scope": (
            "Exact SHA-256 equality of the canonical final numerical result. "
            "worker_count and other execution metadata are excluded from the canonical payload."
        ),
        "controls": {
            "seed": "Same configuration with seeds 101 and 102 must produce different hashes.",
            "physical_batch_size_invariance": (
                "Under philox_logical_block, the physical batch sizes 20000, 25000, and 50000 "
                "must produce exactly one hash while logical_block_size remains fixed."
            ),
            "legacy_batch_size_sensitivity": (
                "Under the explicitly selected legacy batch_index strategy, physical batch "
                "sizes 20000, 25000, and 50000 should produce more than one hash. "
                "This is a control using the current code, not an execution of historical v1.2.0 source."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def run_main_experiment(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    raw_dir = output_dir / "raw"

    total = len(args.models) * len(args.workers) * args.repetitions
    run_number = 0

    for model in args.models:
        model_rows: list[dict[str, Any]] = []
        reference_hash: str | None = None

        for workers in args.workers:
            for repetition in range(1, args.repetitions + 1):
                run_number += 1
                print(
                    f"[{run_number:03d}/{total:03d}] model={model} workers={workers} "
                    f"repetition={repetition}",
                    flush=True,
                )

                complete, elapsed = run_dsl(
                    model=model,
                    iterations=args.iterations,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    workers=workers,
                    histogram_bins=args.histogram_bins,
                    timeout_seconds=args.timeout,
                )
                sha256 = result_hash(complete)
                if reference_hash is None:
                    reference_hash = sha256

                metadata = complete.get("metadata") or {}
                row: dict[str, Any] = {
                    "model": model,
                    "repetition": repetition,
                    "seed": args.seed,
                    "iterations": args.iterations,
                    "batch_size": args.batch_size,
                    "requested_workers": workers,
                    "actual_workers": metadata.get("worker_count"),
                    "batch_count": metadata.get("batch_count"),
                    "elapsed_seconds": elapsed,
                    "result_sha256": sha256,
                    "matches_reference": sha256 == reference_hash,
                    **primary_stats(complete),
                }
                raw_rows.append(row)
                model_rows.append(row)

                write_complete_event(
                    raw_dir,
                    f"{model}_seed{args.seed}_n{args.iterations}_b{args.batch_size}_w{workers}_r{repetition:02d}.json",
                    complete,
                )

        unique_hashes = sorted({row["result_sha256"] for row in model_rows})
        summary_rows.append(
            {
                "model": model,
                "seed": args.seed,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "workers_tested": ",".join(str(value) for value in args.workers),
                "repetitions_per_worker": args.repetitions,
                "total_runs": len(model_rows),
                "unique_result_hashes": len(unique_hashes),
                "exact_reproducibility": len(unique_hashes) == 1,
                "reference_sha256": reference_hash,
            }
        )

    write_csv(output_dir / "rng_reproducibility_raw.csv", MAIN_FIELDS, raw_rows)
    write_csv(output_dir / "rng_reproducibility_summary.csv", SUMMARY_FIELDS, summary_rows)
    return raw_rows, summary_rows


def run_controls(args: argparse.Namespace, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_dir = output_dir / "raw" / "controls"
    model = args.control_model

    print("[control] seed sensitivity", flush=True)
    seed_results: dict[int, tuple[dict[str, Any], str]] = {}
    for seed in (args.seed, args.seed + 1):
        complete, _ = run_dsl(
            model=model,
            iterations=args.iterations,
            seed=seed,
            batch_size=args.batch_size,
            workers=1,
            histogram_bins=args.histogram_bins,
            timeout_seconds=args.timeout,
        )
        sha256 = result_hash(complete)
        seed_results[seed] = (complete, sha256)
        write_complete_event(
            raw_dir,
            f"seed_control_{model}_seed{seed}_n{args.iterations}_b{args.batch_size}_w1.json",
            complete,
        )

    seed_hashes_differ = seed_results[args.seed][1] != seed_results[args.seed + 1][1]
    for seed, (complete, sha256) in seed_results.items():
        stats = primary_stats(complete)
        rows.append(
            {
                "control": "seed_sensitivity",
                "seed_strategy": "philox_logical_block",
                "model": model,
                "seed": seed,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "workers": 1,
                "result_sha256": sha256,
                "result_mean": stats["result_mean"],
                "result_variance": stats["result_variance"],
                "expected_relation": "different hashes across seeds",
                "observed_relation": "different" if seed_hashes_differ else "same",
                "pass": seed_hashes_differ,
            }
        )

    print("[control] physical batch-size invariance (v1.3.1)", flush=True)
    batch_sizes = tuple(dict.fromkeys((20_000, args.batch_size, 50_000)))
    batch_results: dict[int, tuple[dict[str, Any], str]] = {}
    for batch_size in batch_sizes:
        complete, _ = run_dsl(
            model=model,
            iterations=args.iterations,
            seed=args.seed,
            batch_size=batch_size,
            workers=1,
            histogram_bins=args.histogram_bins,
            timeout_seconds=args.timeout,
        )
        sha256 = result_hash(complete)
        batch_results[batch_size] = (complete, sha256)
        write_complete_event(
            raw_dir,
            f"physical_batch_control_{model}_seed{args.seed}_n{args.iterations}_b{batch_size}_w1.json",
            complete,
        )

    unique_batch_hashes = {item[1] for item in batch_results.values()}
    physical_invariance_pass = len(unique_batch_hashes) == 1
    for batch_size, (complete, sha256) in batch_results.items():
        stats = primary_stats(complete)
        rows.append(
            {
                "control": "physical_batch_size_invariance",
                "seed_strategy": "philox_logical_block",
                "model": model,
                "seed": args.seed,
                "iterations": args.iterations,
                "batch_size": batch_size,
                "workers": 1,
                "result_sha256": sha256,
                "result_mean": stats["result_mean"],
                "result_variance": stats["result_variance"],
                "expected_relation": "single identical hash across physical batch sizes",
                "observed_relation": f"{len(unique_batch_hashes)} unique hashes",
                "pass": physical_invariance_pass,
            }
        )

    print("[control] legacy batch-index sensitivity", flush=True)
    legacy_results: dict[int, tuple[dict[str, Any], str]] = {}
    for batch_size in batch_sizes:
        complete, _ = run_dsl(
            model=model,
            iterations=args.iterations,
            seed=args.seed,
            batch_size=batch_size,
            workers=1,
            histogram_bins=args.histogram_bins,
            timeout_seconds=args.timeout,
            seed_strategy="batch_index",
            scheduler_mode="fixed",
        )
        sha256 = result_hash(complete)
        legacy_results[batch_size] = (complete, sha256)
        write_complete_event(
            raw_dir,
            f"legacy_batch_control_{model}_seed{args.seed}_n{args.iterations}_b{batch_size}_w1.json",
            complete,
        )

    legacy_hashes = {item[1] for item in legacy_results.values()}
    legacy_sensitivity_pass = len(legacy_hashes) > 1
    for batch_size, (complete, sha256) in legacy_results.items():
        stats = primary_stats(complete)
        rows.append(
            {
                "control": "legacy_batch_size_sensitivity",
                "seed_strategy": "batch_index",
                "model": model,
                "seed": args.seed,
                "iterations": args.iterations,
                "batch_size": batch_size,
                "workers": 1,
                "result_sha256": sha256,
                "result_mean": stats["result_mean"],
                "result_variance": stats["result_variance"],
                "expected_relation": "more than one hash under legacy batch-index seeding",
                "observed_relation": f"{len(legacy_hashes)} unique hashes",
                "pass": legacy_sensitivity_pass,
            }
        )

    write_csv(output_dir / "rng_reproducibility_controls.csv", CONTROL_FIELDS, rows)
    return rows


def parse_workers(value: str) -> tuple[int, ...]:
    workers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not workers or any(worker <= 0 for worker in workers):
        raise argparse.ArgumentTypeError("workers must be a comma-separated list of positive integers")
    return workers


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(MODELS)}"
        )
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test Monte Carlo DSL v1.3.1 reproducibility across worker counts and physical "
            "batch sizes under fixed logical blocks, and legacy batch-index sensitivity."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=parse_workers, default=DEFAULT_WORKERS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument(
        "--models",
        type=parse_models,
        default=list(MODELS),
        help="Comma-separated model names. Default: all analytic models.",
    )
    parser.add_argument("--control-model", choices=tuple(MODELS), default=DEFAULT_CONTROL_MODEL)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per generated simulation, seconds.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.histogram_bins <= 0:
        raise SystemExit("--histogram-bins must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_environment(
        output_dir,
        models=args.models,
        seed=args.seed,
        iterations=args.iterations,
        batch_size=args.batch_size,
        workers=args.workers,
        repetitions=args.repetitions,
        histogram_bins=args.histogram_bins,
    )

    _, summaries = run_main_experiment(args, output_dir)
    controls = [] if args.skip_controls else run_controls(args, output_dir)

    main_pass = all(row["exact_reproducibility"] for row in summaries)
    controls_pass = all(row["pass"] for row in controls) if controls else True

    print("\nRNG reproducibility summary")
    print("---------------------------")
    for row in summaries:
        status = "PASS" if row["exact_reproducibility"] else "FAIL"
        print(
            f"{row['model']:<20} {status}  "
            f"unique_hashes={row['unique_result_hashes']}  runs={row['total_runs']}"
        )
    if controls:
        seed_pass = all(row["pass"] for row in controls if row["control"] == "seed_sensitivity")
        batch_pass = all(row["pass"] for row in controls if row["control"] == "physical_batch_size_invariance")
        legacy_pass = all(row["pass"] for row in controls if row["control"] == "legacy_batch_size_sensitivity")
        print(f"seed control          {'PASS' if seed_pass else 'FAIL'}")
        print(f"batch-size invariance {'PASS' if batch_pass else 'FAIL'}")
        print(f"legacy batch control  {'PASS' if legacy_pass else 'FAIL'}")

    overall = main_pass and controls_pass
    print(f"overall               {'PASS' if overall else 'FAIL'}")
    print(f"results               {output_dir}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
