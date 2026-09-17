from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from montecarlo_dsl import GeneratorConfig, generate_source

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "benchmarks" / "results" / "raw"
PROCESSED_DIR = ROOT / "benchmarks" / "results" / "processed"


def _events(stdout: str) -> list[dict]:
    prefix = "MC_DSL_EVENT:"
    return [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]


def _run_once(model_path: Path, iterations: int, seed: int, batch_size: int, workers: int) -> dict:
    source = model_path.read_text(encoding="utf-8")
    # Mantiene el modelo como fuente de verdad y solo ajusta iteraciones para el benchmark.
    lines = []
    for line in source.splitlines():
        lines.append(f"\\iter{{{iterations}}}" if line.startswith("\\iter{") else line)
    source = "\n".join(lines)
    script = generate_source(source, GeneratorConfig(seed=seed, batch_size=batch_size, max_workers=workers))
    script_path = RAW_DIR / f"bench_{model_path.stem}_{iterations}_{workers}_{seed}.py"
    script_path.write_text(script, encoding="utf-8")
    t0 = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=600,
    )
    elapsed = time.perf_counter() - t0
    events = _events(completed.stdout)
    if completed.returncode != 0 or not events or events[-1].get("type") != "complete":
        raise RuntimeError(f"benchmark failed for {model_path}: {completed.stderr[-2000:]}")
    final = events[-1]
    return {
        "model": model_path.stem,
        "iterations": iterations,
        "seed": seed,
        "batch_size": batch_size,
        "workers_requested": workers,
        "workers_used": final.get("metadata", {}).get("worker_count"),
        "wall_time_seconds": elapsed,
        "valid_results": final["valid_results"],
        "discarded_results": final["discarded_results"],
        "statistical_core_version": final.get("metadata", {}).get("statistical_core_version"),
        "python_version": final.get("metadata", {}).get("python_version"),
        "numpy_version": final.get("metadata", {}).get("numpy_version"),
        "platform": final.get("metadata", {}).get("platform"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="usa una matriz pequena para validacion rapida")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    models = [
        ROOT / "benchmarks" / "models" / "analytic" / "normal_mean.dsl",
        ROOT / "benchmarks" / "models" / "performance" / "heavy_expression.dsl",
    ]
    iterations_grid = [10_000, 100_000] if args.quick else [10_000, 100_000, 1_000_000]
    workers_grid = [1, 2] if args.quick else [1, 2, 4]
    repetitions = 2 if args.quick else 10
    rows: list[dict] = []
    for model in models:
        for iterations in iterations_grid:
            for workers in workers_grid:
                for rep in range(repetitions):
                    rows.append(_run_once(model, iterations, 10_000 + rep, 25_000, workers))
    raw_csv = RAW_DIR / "benchmark_runs.csv"
    with raw_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model"], row["iterations"], row["workers_requested"])
        groups.setdefault(key, []).append(row)
    for (model, iterations, workers), group in groups.items():
        times = [float(row["wall_time_seconds"]) for row in group]
        summary_rows.append({
            "model": model,
            "iterations": iterations,
            "workers_requested": workers,
            "runs": len(times),
            "mean_time_seconds": statistics.fmean(times),
            "median_time_seconds": statistics.median(times),
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "std_time_seconds": statistics.stdev(times) if len(times) > 1 else 0.0,
        })
    summary_csv = PROCESSED_DIR / "benchmark_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {raw_csv}")
    print(f"wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
