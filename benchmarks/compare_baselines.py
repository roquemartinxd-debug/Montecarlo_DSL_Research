from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from montecarlo_dsl import generate_source
from montecarlo_dsl.compiler import compile_source
from montecarlo_dsl import GeneratorConfig

from benchmarks.baselines.normal_mean_manual import run as normal_manual
from benchmarks.baselines.uniform_mean_manual import run as uniform_manual
from benchmarks.baselines.discrete_mean_manual import run as discrete_manual
from benchmarks.baselines.linear_combination_manual import run as linear_manual


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "benchmarks" / "baseline_results"
RAW_DIR = RESULTS_DIR / "generated"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "normal_mean": {
        "dsl": ROOT / "benchmarks/models/analytic/normal_mean.dsl",
        "manual": normal_manual,
    },
    "uniform_mean": {
        "dsl": ROOT / "benchmarks/models/analytic/uniform_mean.dsl",
        "manual": uniform_manual,
    },
    "discrete_mean": {
        "dsl": ROOT / "benchmarks/models/analytic/discrete_mean.dsl",
        "manual": discrete_manual,
    },
    "linear_combination": {
        "dsl": ROOT / "benchmarks/models/analytic/linear_combination.dsl",
        "manual": linear_manual,
    },
}


def extract_complete_event(stdout: str):

    prefix = "MC_DSL_EVENT:"

    events = [
        json.loads(line[len(prefix):])
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]

    for event in reversed(events):
        if event.get("type") == "complete":
            return event

    raise RuntimeError("No complete event found")


def extract_stats(event):

    stats = event["stats"]
    diagnostics = event["diagnostics"]

    return {
        "mean": stats["avg"],
        "variance": stats["var"],
        "std": stats["std"],
        "mcse": diagnostics["mcse_mean"],
        "valid_rate": stats["valid_rate"],
        "discard_rate": stats["discard_rate"],
    }


def run_generated_dsl(source, seed):

    start_generation = time.perf_counter()

    script = generate_source(
        source,
        GeneratorConfig(
            seed=seed,
            batch_size=20000,
            max_workers=2,
        ),
    )

    generation_time = time.perf_counter() - start_generation

    script_path = RAW_DIR / "generated_model.py"
    script_path.write_text(script, encoding="utf-8")

    start_execution = time.perf_counter()

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    execution_time = time.perf_counter() - start_execution

    event = extract_complete_event(completed.stdout)

    result = extract_stats(event)

    result.update(
        {
            "generation_time_seconds": generation_time,
            "execution_time_seconds": execution_time,
            "compile_time_seconds": 0,
            "total_time_seconds": generation_time + execution_time,
        }
    )

    return result


def run_full_dsl(source, seed):

    start_compile = time.perf_counter()

    compiled = compile_source(
        source,
        GeneratorConfig(
            seed=seed,
            batch_size=20000,
            max_workers=2,
        ),
    )

    compile_time = time.perf_counter() - start_compile

    script_path = RAW_DIR / "compiled_model.py"
    script_path.write_text(
        compiled.script,
        encoding="utf-8",
    )

    start_execution = time.perf_counter()

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    execution_time = time.perf_counter() - start_execution

    event = extract_complete_event(completed.stdout)

    result = extract_stats(event)

    result.update(
        {
            "compile_time_seconds": compile_time,
            "generation_time_seconds": 0,
            "execution_time_seconds": execution_time,
            "total_time_seconds": compile_time + execution_time,
        }
    )

    return result


def main():

    rows = []

    seed = 101
    iterations = 100000

    for model, config in MODELS.items():

        source = config["dsl"].read_text(
            encoding="utf-8"
        )

        manual = config["manual"](
            seed=seed,
            iterations=iterations,
        )

        rows.append(
            {
                "model": model,
                "implementation": "python_manual",
                "seed": seed,
                "iterations": iterations,
                **manual,
                "compile_time_seconds": 0,
                "generation_time_seconds": 0,
                "total_time_seconds": manual["execution_time_seconds"],
            }
        )


        generated = run_generated_dsl(
            source,
            seed,
        )

        rows.append(
            {
                "model": model,
                "implementation": "dsl_generated",
                "seed": seed,
                "iterations": iterations,
                **generated,
            }
        )


        full = run_full_dsl(
            source,
            seed,
        )

        rows.append(
            {
                "model": model,
                "implementation": "dsl_full",
                "seed": seed,
                "iterations": iterations,
                **full,
            }
        )


    output = RESULTS_DIR / "baseline_comparison.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


    print(f"Wrote {output}")


if __name__ == "__main__":
    main()