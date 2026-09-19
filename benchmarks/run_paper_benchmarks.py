from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from montecarlo_dsl.compiler import compile_source
from montecarlo_dsl.generator import GeneratorConfig

from benchmarks.baselines.normal_mean_manual import run as run_normal
from benchmarks.baselines.uniform_mean_manual import run as run_uniform
from benchmarks.baselines.discrete_mean_manual import run as run_discrete
from benchmarks.baselines.linear_combination_manual import run as run_linear


ROOT = Path(__file__).resolve().parents[1]

EVENT_PREFIX = "MC_DSL_EVENT:"

SIMULATION_SEED = 101
SCHEDULE_SEED = 120026

BATCH_SIZE = 25_000
HISTOGRAM_BINS = 30

DEFAULT_ITERATIONS = [
    10_000,
    100_000,
    1_000_000,
]

IMPLEMENTATIONS = [
    "python_manual",
    "dsl_full_w1",
    "dsl_full_w2",
]

DEFAULT_OUTPUT = (
    ROOT
    / "benchmarks"
    / "paper_results"
    / "final"
    / "v1.2.0"
)


MODELS = {
    "normal_mean": {
        "dsl": ROOT / "benchmarks" / "models" / "analytic" / "normal_mean.dsl",
        "manual": run_normal,
    },
    "uniform_mean": {
        "dsl": ROOT / "benchmarks" / "models" / "analytic" / "uniform_mean.dsl",
        "manual": run_uniform,
    },
    "discrete_mean": {
        "dsl": ROOT / "benchmarks" / "models" / "analytic" / "discrete_mean.dsl",
        "manual": run_discrete,
    },
    "linear_combination": {
        "dsl": ROOT
        / "benchmarks"
        / "models"
        / "analytic"
        / "linear_combination.dsl",
        "manual": run_linear,
    },
}


RAW_FIELDS = [
    "phase",
    "block_index",
    "run_index",
    "schedule_position",
    "model",
    "iterations",
    "implementation",
    "workers",
    "simulation_seed",
    "compile_time_seconds",
    "execution_time_seconds",
    "total_time_seconds",
    "result_mean",
    "result_variance",
    "valid_rate",
    "discard_rate",
    "valid_count",
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


def source_with_iterations(
    model: str,
    iterations: int,
) -> str:

    path = MODELS[model]["dsl"]

    source = path.read_text(
        encoding="utf-8"
    )

    source, replacements = re.subn(
        r"\\iter\{\d+\}",
        lambda _: f"\\iter{{{iterations}}}",
        source,
    )

    if replacements != 1:
        raise RuntimeError(
            f"Expected exactly one \\iter block "
            f"in {path}, found {replacements}"
        )

    return source


def parse_complete_event(stdout: str) -> dict:

    complete = None

    for line in stdout.splitlines():

        if not line.startswith(EVENT_PREFIX):
            continue

        event = json.loads(
            line[len(EVENT_PREFIX):]
        )

        if event.get("type") == "complete":
            complete = event

    if complete is None:
        raise RuntimeError(
            "No MC_DSL_EVENT complete event found."
        )

    return complete


def run_manual(
    model: str,
    iterations: int,
) -> dict:

    runner = MODELS[model]["manual"]

    module_name = runner.__module__

    code = (
        "import json;"
        f"from {module_name} import run;"
        f"result=run(seed={SIMULATION_SEED}, "
        f"iterations={iterations});"
        "print(json.dumps(result))"
    )

    start = time.perf_counter()

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    total_time = (
        time.perf_counter()
        - start
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"Manual Python execution failed.\n"
            f"Model: {model}\n"
            f"Iterations: {iterations}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    lines = [
        line
        for line in process.stdout.splitlines()
        if line.strip()
    ]

    if not lines:
        raise RuntimeError(
            "Manual Python process produced no output."
        )

    result = json.loads(
        lines[-1]
    )

    return {
        "workers": "",
        "compile_time_seconds": 0.0,
        "execution_time_seconds": total_time,
        "total_time_seconds": total_time,
        "result_mean": result.get("mean"),
        "result_variance": result.get("variance"),
        "valid_rate": result.get("valid_rate"),
        "discard_rate": result.get("discard_rate"),
        "valid_count": result.get("count"),
    }

    runner = MODELS[model]["manual"]

    start = time.perf_counter()

    result = runner(
        seed=SIMULATION_SEED,
        iterations=iterations,
    )

    total_time = time.perf_counter() - start

    return {
        "workers": "",
        "compile_time_seconds": 0.0,
        "execution_time_seconds": total_time,
        "total_time_seconds": total_time,
        "result_mean": result.get("mean"),
        "result_variance": result.get("variance"),
        "valid_rate": result.get("valid_rate"),
        "discard_rate": result.get("discard_rate"),
        "valid_count": result.get("count"),
    }


def run_dsl(
    model: str,
    iterations: int,
    workers: int,
) -> dict:

    source = source_with_iterations(
        model,
        iterations,
    )

    total_start = time.perf_counter()

    compile_start = time.perf_counter()

    compiled = compile_source(
        source,
        GeneratorConfig(
            seed=SIMULATION_SEED,
            batch_size=BATCH_SIZE,
            max_workers=workers,
            histogram_bins=HISTOGRAM_BINS,
        ),
    )

    compile_time = (
        time.perf_counter()
        - compile_start
    )

    with tempfile.TemporaryDirectory() as temp:

        script_path = (
            Path(temp)
            / "simulation.py"
        )

        script_path.write_text(
            compiled.script,
            encoding="utf-8",
        )

        execution_start = time.perf_counter()

        process = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        execution_time = (
            time.perf_counter()
            - execution_start
        )

    total_time = (
        time.perf_counter()
        - total_start
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"DSL execution failed.\n"
            f"Model: {model}\n"
            f"Iterations: {iterations}\n"
            f"Workers: {workers}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    complete = parse_complete_event(
        process.stdout
    )

    stats = complete["stats"]

    return {
        "workers": workers,
        "compile_time_seconds":
            compile_time,
        "execution_time_seconds":
            execution_time,
        "total_time_seconds":
            total_time,
        "result_mean":
            stats.get("avg"),
        "result_variance":
            stats.get("var"),
        "valid_rate":
            stats.get("valid_rate"),
        "discard_rate":
            stats.get("discard_rate"),
        "valid_count":
            stats.get("count"),
    }


def execute(
    implementation: str,
    model: str,
    iterations: int,
) -> dict:

    if implementation == "python_manual":
        return run_manual(
            model,
            iterations,
        )

    if implementation == "dsl_full_w1":
        return run_dsl(
            model,
            iterations,
            workers=1,
        )

    if implementation == "dsl_full_w2":
        return run_dsl(
            model,
            iterations,
            workers=2,
        )

    raise ValueError(
        f"Unknown implementation: "
        f"{implementation}"
    )


def write_environment(
    output_dir: Path,
    repetitions: int,
    warmups: int,
    iterations: list[int],
) -> None:

    environment = {
        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "git_commit":
            git_value(
                "rev-parse",
                "HEAD",
            ),
        "git_branch":
            git_value(
                "branch",
                "--show-current",
            ),
        "python_version":
            sys.version,
        "python_executable":
            sys.executable,
        "numpy_version":
            np.__version__,
        "platform":
            platform.platform(),
        "machine":
            platform.machine(),
        "processor":
            platform.processor(),
        "processor_identifier":
            os.environ.get(
                "PROCESSOR_IDENTIFIER",
                "",
            ),
        "logical_cpu_count":
            os.cpu_count(),
        "simulation_seed":
            SIMULATION_SEED,
        "schedule_seed":
            SCHEDULE_SEED,
        "batch_size":
            BATCH_SIZE,
        "histogram_bins":
            HISTOGRAM_BINS,
        "warmup_runs_per_configuration":
            warmups,
        "measured_runs_per_configuration":
            repetitions,
        "iterations":
            iterations,
        "implementations":
            IMPLEMENTATIONS,
        "timing_scope":
    (
        "End-to-end latency. "
        "Both the manual Python baseline "
        "and generated DSL program execute "
        "in fresh Python subprocesses. "
        "DSL total time additionally includes "
        "compilation and generated-script creation."
    ),
    }

    path = (
        output_dir
        / "environment.json"
    )

    path.write_text(
        json.dumps(
            environment,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_blocks(
    phase: str,
    repetitions: int,
    iterations: list[int],
    rng: random.Random,
) -> list[tuple]:

    blocks = []

    for model in MODELS:

        for iteration_count in iterations:

            for run_index in range(
                1,
                repetitions + 1,
            ):

                blocks.append(
                    (
                        phase,
                        model,
                        iteration_count,
                        run_index,
                    )
                )

    rng.shuffle(blocks)

    return blocks


def run_blocks(
    blocks: list[tuple],
    writer: csv.DictWriter,
    raw_file,
    measured_rows: list[dict],
    rng: random.Random,
    initial_block_index: int,
) -> int:

    block_index = initial_block_index

    for (
        phase,
        model,
        iterations,
        run_index,
    ) in blocks:

        block_index += 1

        order = IMPLEMENTATIONS.copy()
        rng.shuffle(order)

        for schedule_position, implementation in enumerate(
            order,
            start=1,
        ):

            print(
                f"[{phase}] "
                f"block={block_index} "
                f"model={model} "
                f"iterations={iterations} "
                f"run={run_index} "
                f"impl={implementation}"
            )

            result = execute(
                implementation,
                model,
                iterations,
            )

            row = {
                "phase":
                    phase,
                "block_index":
                    block_index,
                "run_index":
                    run_index,
                "schedule_position":
                    schedule_position,
                "model":
                    model,
                "iterations":
                    iterations,
                "implementation":
                    implementation,
                "workers":
                    result["workers"],
                "simulation_seed":
                    SIMULATION_SEED,
                "compile_time_seconds":
                    result[
                        "compile_time_seconds"
                    ],
                "execution_time_seconds":
                    result[
                        "execution_time_seconds"
                    ],
                "total_time_seconds":
                    result[
                        "total_time_seconds"
                    ],
                "result_mean":
                    result[
                        "result_mean"
                    ],
                "result_variance":
                    result[
                        "result_variance"
                    ],
                "valid_rate":
                    result[
                        "valid_rate"
                    ],
                "discard_rate":
                    result[
                        "discard_rate"
                    ],
                "valid_count":
                    result[
                        "valid_count"
                    ],
            }

            writer.writerow(row)
            raw_file.flush()

            if phase == "measured":
                measured_rows.append(row)

    return block_index


def create_summary(
    measured_rows: list[dict],
    output_dir: Path,
) -> None:

    grouped = {}

    for row in measured_rows:

        key = (
            row["model"],
            int(row["iterations"]),
            row["implementation"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    manual_means = {}

    for (
        model,
        iterations,
        implementation,
    ), rows in grouped.items():

        if implementation != "python_manual":
            continue

        manual_means[
            (model, iterations)
        ] = statistics.mean(
            float(
                row[
                    "total_time_seconds"
                ]
            )
            for row in rows
        )

    summary_rows = []

    for key in sorted(grouped):

        model, iterations, implementation = key

        rows = grouped[key]

        total_times = [
            float(
                row[
                    "total_time_seconds"
                ]
            )
            for row in rows
        ]

        compile_times = [
            float(
                row[
                    "compile_time_seconds"
                ]
            )
            for row in rows
        ]

        execution_times = [
            float(
                row[
                    "execution_time_seconds"
                ]
            )
            for row in rows
        ]

        mean_total = statistics.mean(
            total_times
        )

        manual_mean = manual_means[
            (model, iterations)
        ]

        ratio = (
            mean_total / manual_mean
        )

        overhead_percent = (
            (ratio - 1.0) * 100.0
        )

        workers = rows[0]["workers"]

        summary_rows.append(
            {
                "model":
                    model,
                "iterations":
                    iterations,
                "implementation":
                    implementation,
                "workers":
                    workers,
                "measured_runs":
                    len(rows),
                "mean_total_time_seconds":
                    mean_total,
                "median_total_time_seconds":
                    statistics.median(
                        total_times
                    ),
                "std_total_time_seconds":
                    (
                        statistics.stdev(
                            total_times
                        )
                        if len(total_times) > 1
                        else 0.0
                    ),
                "min_total_time_seconds":
                    min(total_times),
                "max_total_time_seconds":
                    max(total_times),
                "mean_compile_time_seconds":
                    statistics.mean(
                        compile_times
                    ),
                "mean_execution_time_seconds":
                    statistics.mean(
                        execution_times
                    ),
                "ratio_vs_python_manual":
                    ratio,
                "overhead_percent_vs_python_manual":
                    overhead_percent,
            }
        )

    output = (
        output_dir
        / "paper_benchmark_summary.csv"
    )

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=DEFAULT_ITERATIONS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_environment(
        output_dir,
        args.repetitions,
        args.warmups,
        args.iterations,
    )

    raw_path = (
        output_dir
        / "paper_benchmark_raw.csv"
    )

    rng = random.Random(
        SCHEDULE_SEED
    )

    warmup_blocks = build_blocks(
        "warmup",
        args.warmups,
        args.iterations,
        rng,
    )

    measured_blocks = build_blocks(
        "measured",
        args.repetitions,
        args.iterations,
        rng,
    )

    measured_rows = []

    expected_runs = (
        (
            len(warmup_blocks)
            + len(measured_blocks)
        )
        * len(IMPLEMENTATIONS)
    )

    print(
        f"Expected process runs: "
        f"{expected_runs}"
    )

    with raw_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as raw_file:

        writer = csv.DictWriter(
            raw_file,
            fieldnames=RAW_FIELDS,
        )

        writer.writeheader()
        raw_file.flush()

        block_index = run_blocks(
            warmup_blocks,
            writer,
            raw_file,
            measured_rows,
            rng,
            0,
        )

        run_blocks(
            measured_blocks,
            writer,
            raw_file,
            measured_rows,
            rng,
            block_index,
        )

    create_summary(
        measured_rows,
        output_dir,
    )

    print()
    print(
        f"Raw results: {raw_path}"
    )

    print(
        "Summary: "
        f"{output_dir / 'paper_benchmark_summary.csv'}"
    )

    print(
        "Environment: "
        f"{output_dir / 'environment.json'}"
    )

    print()
    print(
        "PAPER BENCHMARK COMPLETED"
    )


if __name__ == "__main__":
    main()