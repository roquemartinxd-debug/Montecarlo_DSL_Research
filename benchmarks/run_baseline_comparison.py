from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

from montecarlo_dsl import generate_source, GeneratorConfig
from montecarlo_dsl.compiler import compile_source

from benchmarks.baselines.normal_mean_manual import run as normal_manual
from benchmarks.baselines.uniform_mean_manual import run as uniform_manual
from benchmarks.baselines.discrete_mean_manual import run as discrete_manual
from benchmarks.baselines.linear_combination_manual import run as linear_manual


ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "benchmarks"
    / "baseline_results"
    / "runs"
)

RESULTS.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULTS / "baseline_timing_repeated.csv"


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


RUNS = 10
SEED = 101
ITERATIONS = 100000


def extract_event(stdout: str):

    prefix = "MC_DSL_EVENT:"

    for line in reversed(stdout.splitlines()):

        if line.startswith(prefix):

            event = json.loads(
                line[len(prefix):]
            )

            if event.get("type") == "complete":
                return event

    raise RuntimeError("No complete event found")


def run_generated(source):

    generation_start = time.perf_counter()

    script = generate_source(
        source,
        GeneratorConfig(
            seed=SEED,
            batch_size=20000,
            max_workers=2,
        ),
    )

    generation_time = (
        time.perf_counter() - generation_start
    )

    script_path = RESULTS / "generated_tmp.py"

    script_path.write_text(
        script,
        encoding="utf-8"
    )

    execution_start = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    execution_time = (
        time.perf_counter() - execution_start
    )

    extract_event(result.stdout)

    return generation_time + execution_time


def run_full(source):

    compile_start = time.perf_counter()

    compiled = compile_source(
        source,
        GeneratorConfig(
            seed=SEED,
            batch_size=20000,
            max_workers=2,
        ),
    )

    compile_time = (
        time.perf_counter() - compile_start
    )

    script_path = RESULTS / "full_tmp.py"

    script_path.write_text(
        compiled.script,
        encoding="utf-8"
    )

    execution_start = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    execution_time = (
        time.perf_counter() - execution_start
    )

    extract_event(result.stdout)

    return compile_time + execution_time


def main():

    rows = []

    for model, config in MODELS.items():

        source = config["dsl"].read_text(
            encoding="utf-8"
        )

        timings = {
            "python_manual": [],
            "dsl_generated": [],
            "dsl_full": [],
        }

        for _ in range(RUNS):

            manual = config["manual"](
                seed=SEED,
                iterations=ITERATIONS,
            )

            timings["python_manual"].append(
                manual["execution_time_seconds"]
            )

            timings["dsl_generated"].append(
                run_generated(source)
            )

            timings["dsl_full"].append(
                run_full(source)
            )

        for implementation, values in timings.items():

            rows.append(
                {
                    "model": model,
                    "implementation": implementation,
                    "runs": RUNS,
                    "mean_time_seconds": mean(values),
                    "std_time_seconds": (
                        stdev(values)
                        if len(values) > 1
                        else 0
                    ),
                    "min_time_seconds": min(values),
                    "max_time_seconds": max(values),
                }
            )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()