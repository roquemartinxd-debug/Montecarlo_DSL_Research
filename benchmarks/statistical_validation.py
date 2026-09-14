from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from montecarlo_dsl import GeneratorConfig, generate_source

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "benchmarks" / "results" / "raw"
PROCESSED_DIR = ROOT / "benchmarks" / "results" / "processed"
MODELS = {
    "normal_mean": {
        "path": ROOT / "benchmarks" / "models" / "analytic" / "normal_mean.dsl",
        "theoretical_mean": 10.0,
        "theoretical_variance": 4.0,
    },
    "uniform_mean": {
        "path": ROOT / "benchmarks" / "models" / "analytic" / "uniform_mean.dsl",
        "theoretical_mean": 5.0,
        "theoretical_variance": 100.0 / 12.0,
    },
    "discrete_mean": {
        "path": ROOT / "benchmarks" / "models" / "analytic" / "discrete_mean.dsl",
        "theoretical_mean": 17.0,
        "theoretical_variance": 61.0,
    },
    "linear_combination": {
        "path": ROOT / "benchmarks" / "models" / "analytic" / "linear_combination.dsl",
        "theoretical_mean": 29.0,
        "theoretical_variance": 43.0,
    },
}


def _events_from_stdout(stdout: str) -> list[dict]:
    prefix = "MC_DSL_EVENT:"
    return [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]


def _run_model(name: str, source: str, seed: int, batch_size: int, max_workers: int) -> dict:
    script = generate_source(source, GeneratorConfig(seed=seed, batch_size=batch_size, max_workers=max_workers))
    script_path = RAW_DIR / f"{name}_seed_{seed}.py"
    script_path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    events = _events_from_stdout(completed.stdout)
    if completed.returncode != 0 or not events or events[-1].get("type") != "complete":
        raise RuntimeError(f"{name} seed={seed} failed: {completed.stderr[-2000:]}")
    raw_path = RAW_DIR / f"{name}_seed_{seed}.jsonl"
    with raw_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return events[-1]


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [101, 202, 303, 404, 505]
    rows: list[dict] = []
    for name, config in MODELS.items():
        source = config["path"].read_text(encoding="utf-8")
        for seed in seeds:
            final = _run_model(name, source, seed=seed, batch_size=20_000, max_workers=2)
            stats = final["stats"]
            diagnostics = final["diagnostics"]
            estimated_mean = stats["avg"]
            theoretical_mean = config["theoretical_mean"]
            mcse = diagnostics["mcse_mean"]
            abs_error = abs(estimated_mean - theoretical_mean)
            rows.append({
                "model_name": name,
                "seed": seed,
                "iterations": final["total_iterations"],
                "theoretical_mean": theoretical_mean,
                "estimated_mean": estimated_mean,
                "absolute_error": abs_error,
                "mcse": mcse,
                "within_3_mcse": bool(mcse is not None and abs_error <= 3.0 * mcse),
                "theoretical_variance": config["theoretical_variance"],
                "estimated_variance": stats.get("var"),
                "valid_rate": stats.get("valid_rate"),
                "discard_rate": stats.get("discard_rate"),
                "statistical_core_version": final.get("metadata", {}).get("statistical_core_version"),
                "rng_bit_generator": final.get("metadata", {}).get("rng_bit_generator"),
            })
    out = PROCESSED_DIR / "statistical_validation.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
