from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "benchmarks"
    / "paper_results"
    / "final"
    / "v1.2.0"
)

INPUT = (
    RESULTS_DIR
    / "paper_benchmark_raw.csv"
)

ENVIRONMENT = (
    RESULTS_DIR
    / "environment.json"
)

DESCRIPTIVE_OUTPUT = (
    RESULTS_DIR
    / "paper_benchmark_descriptive.csv"
)

PAIRED_RAW_OUTPUT = (
    RESULTS_DIR
    / "paper_benchmark_paired_ratios_raw.csv"
)

PAIRED_SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "paper_benchmark_paired_ratios_summary.csv"
)


BOOTSTRAP_SEED = 120026
BOOTSTRAP_REPETITIONS = 20_000


IMPLEMENTATIONS = {
    "numpy_native",
    "python_reference",
    "dsl_full_w1",
    "dsl_full_w2",
}


COMPARISONS = [
    (
        "dsl_w1_vs_numpy_native",
        "dsl_full_w1",
        "numpy_native",
    ),
    (
        "dsl_w2_vs_numpy_native",
        "dsl_full_w2",
        "numpy_native",
    ),
    (
        "python_reference_vs_numpy_native",
        "python_reference",
        "numpy_native",
    ),
    (
        "dsl_w1_vs_python_reference",
        "dsl_full_w1",
        "python_reference",
    ),
    (
        "dsl_w2_vs_python_reference",
        "dsl_full_w2",
        "python_reference",
    ),
    (
        "dsl_w2_vs_dsl_w1",
        "dsl_full_w2",
        "dsl_full_w1",
    ),
]


def percentile(values, q):
    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=float,
            ),
            q,
        )
    )


def geometric_mean(values):
    array = np.asarray(
        values,
        dtype=float,
    )

    if np.any(array <= 0):
        raise ValueError(
            "Geometric mean requires positive values."
        )

    return float(
        np.exp(
            np.mean(
                np.log(array)
            )
        )
    )


def bootstrap_ci(
    values,
    statistic,
    repetitions=BOOTSTRAP_REPETITIONS,
):
    array = np.asarray(
        values,
        dtype=float,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    n = len(array)

    results = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(repetitions):

        sample_indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = array[
            sample_indices
        ]

        results[index] = statistic(
            sample
        )

    return (
        float(
            np.percentile(
                results,
                2.5,
            )
        ),
        float(
            np.percentile(
                results,
                97.5,
            )
        ),
    )


def load_environment():

    with ENVIRONMENT.open(
        encoding="utf-8",
    ) as file:

        environment = json.load(
            file
        )

    recorded = set(
        environment[
            "implementations"
        ]
    )

    if recorded != IMPLEMENTATIONS:
        raise RuntimeError(
            "Environment implementations do not match "
            f"expected set.\n"
            f"Recorded: {sorted(recorded)}\n"
            f"Expected: {sorted(IMPLEMENTATIONS)}"
        )

    return environment


def load_measured_rows():

    rows = []

    with INPUT.open(
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            if row["phase"] != "measured":
                continue

            row["block_index"] = int(
                row["block_index"]
            )

            row["run_index"] = int(
                row["run_index"]
            )

            row["iterations"] = int(
                row["iterations"]
            )

            row["schedule_position"] = int(
                row["schedule_position"]
            )

            row["total_time_seconds"] = float(
                row[
                    "total_time_seconds"
                ]
            )

            rows.append(row)

    return rows


def create_descriptive(rows):

    grouped = {}

    for row in rows:

        key = (
            row["model"],
            row["iterations"],
            row["implementation"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(
            row[
                "total_time_seconds"
            ]
        )

    output_rows = []

    for key in sorted(grouped):

        (
            model,
            iterations,
            implementation,
        ) = key

        values = np.asarray(
            grouped[key],
            dtype=float,
        )

        mean = float(
            np.mean(values)
        )

        median = float(
            np.median(values)
        )

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        q1 = percentile(
            values,
            25,
        )

        q3 = percentile(
            values,
            75,
        )

        iqr = q3 - q1

        lower_fence = (
            q1
            - 1.5 * iqr
        )

        upper_fence = (
            q3
            + 1.5 * iqr
        )

        outlier_mask = (
            (values < lower_fence)
            |
            (values > upper_fence)
        )

        outlier_count = int(
            np.sum(
                outlier_mask
            )
        )

        cv = (
            std / mean
            if mean > 0
            else float("nan")
        )

        output_rows.append(
            {
                "model":
                    model,

                "iterations":
                    iterations,

                "implementation":
                    implementation,

                "runs":
                    len(values),

                "mean_seconds":
                    mean,

                "median_seconds":
                    median,

                "std_seconds":
                    std,

                "q1_seconds":
                    q1,

                "q3_seconds":
                    q3,

                "iqr_seconds":
                    iqr,

                "min_seconds":
                    float(
                        np.min(values)
                    ),

                "max_seconds":
                    float(
                        np.max(values)
                    ),

                "coefficient_of_variation":
                    cv,

                "iqr_outlier_count":
                    outlier_count,

                "iqr_lower_fence":
                    lower_fence,

                "iqr_upper_fence":
                    upper_fence,
            }
        )

    return output_rows


def build_paired_ratios(rows):

    blocks = {}

    for row in rows:

        key = (
            row["block_index"],
            row["model"],
            row["iterations"],
            row["run_index"],
        )

        blocks.setdefault(
            key,
            {},
        )[
            row["implementation"]
        ] = row[
            "total_time_seconds"
        ]

    paired_rows = []

    for key in sorted(blocks):

        (
            block_index,
            model,
            iterations,
            run_index,
        ) = key

        values = blocks[key]

        if set(values) != IMPLEMENTATIONS:
            raise RuntimeError(
                f"Incomplete block {key}.\n"
                f"Found: {sorted(values)}\n"
                f"Expected: {sorted(IMPLEMENTATIONS)}"
            )

        numpy_native = values[
            "numpy_native"
        ]

        python_reference = values[
            "python_reference"
        ]

        dsl_w1 = values[
            "dsl_full_w1"
        ]

        dsl_w2 = values[
            "dsl_full_w2"
        ]

        paired_rows.append(
            {
                "block_index":
                    block_index,

                "run_index":
                    run_index,

                "model":
                    model,

                "iterations":
                    iterations,

                "numpy_native_seconds":
                    numpy_native,

                "python_reference_seconds":
                    python_reference,

                "dsl_full_w1_seconds":
                    dsl_w1,

                "dsl_full_w2_seconds":
                    dsl_w2,

                "ratio_dsl_w1_vs_numpy_native":
                    dsl_w1
                    / numpy_native,

                "ratio_dsl_w2_vs_numpy_native":
                    dsl_w2
                    / numpy_native,

                "ratio_python_reference_vs_numpy_native":
                    python_reference
                    / numpy_native,

                "ratio_dsl_w1_vs_python_reference":
                    dsl_w1
                    / python_reference,

                "ratio_dsl_w2_vs_python_reference":
                    dsl_w2
                    / python_reference,

                "ratio_dsl_w2_vs_dsl_w1":
                    dsl_w2
                    / dsl_w1,
            }
        )

    return paired_rows


def ratio_statistics(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    mean_ratio = float(
        np.mean(values)
    )

    median_ratio = float(
        np.median(values)
    )

    geometric_mean_ratio = (
        geometric_mean(values)
    )

    std_ratio = float(
        np.std(
            values,
            ddof=1,
        )
    )

    q1 = percentile(
        values,
        25,
    )

    q3 = percentile(
        values,
        75,
    )

    median_ci = bootstrap_ci(
        values,
        np.median,
    )

    geomean_ci = bootstrap_ci(
        values,
        lambda sample: np.exp(
            np.mean(
                np.log(sample)
            )
        ),
    )

    return {
        "runs":
            len(values),

        "mean_ratio":
            mean_ratio,

        "median_ratio":
            median_ratio,

        "geometric_mean_ratio":
            geometric_mean_ratio,

        "std_ratio":
            std_ratio,

        "q1_ratio":
            q1,

        "q3_ratio":
            q3,

        "iqr_ratio":
            q3 - q1,

        "min_ratio":
            float(
                np.min(values)
            ),

        "max_ratio":
            float(
                np.max(values)
            ),

        "median_percent_change":
            (
                median_ratio
                - 1.0
            )
            * 100.0,

        "geomean_percent_change":
            (
                geometric_mean_ratio
                - 1.0
            )
            * 100.0,

        "median_bootstrap_ci95_low":
            median_ci[0],

        "median_bootstrap_ci95_high":
            median_ci[1],

        "geomean_bootstrap_ci95_low":
            geomean_ci[0],

        "geomean_bootstrap_ci95_high":
            geomean_ci[1],
    }


def create_paired_summary(
    paired_rows,
):

    grouped = {}

    for row in paired_rows:

        key = (
            row["model"],
            row["iterations"],
        )

        grouped.setdefault(
            key,
            {
                comparison_name: []
                for (
                    comparison_name,
                    _,
                    _,
                ) in COMPARISONS
            },
        )

        ratio_map = {
            "dsl_w1_vs_numpy_native":
                row[
                    "ratio_dsl_w1_vs_numpy_native"
                ],

            "dsl_w2_vs_numpy_native":
                row[
                    "ratio_dsl_w2_vs_numpy_native"
                ],

            "python_reference_vs_numpy_native":
                row[
                    "ratio_python_reference_vs_numpy_native"
                ],

            "dsl_w1_vs_python_reference":
                row[
                    "ratio_dsl_w1_vs_python_reference"
                ],

            "dsl_w2_vs_python_reference":
                row[
                    "ratio_dsl_w2_vs_python_reference"
                ],

            "dsl_w2_vs_dsl_w1":
                row[
                    "ratio_dsl_w2_vs_dsl_w1"
                ],
        }

        for (
            comparison,
            value,
        ) in ratio_map.items():

            grouped[key][
                comparison
            ].append(
                float(value)
            )

    output_rows = []

    for key in sorted(grouped):

        (
            model,
            iterations,
        ) = key

        for (
            comparison,
            values,
        ) in grouped[key].items():

            stats = ratio_statistics(
                values
            )

            output_rows.append(
                {
                    "model":
                        model,

                    "iterations":
                        iterations,

                    "comparison":
                        comparison,

                    **stats,
                }
            )

    return output_rows


def validate_counts(
    environment,
    measured_rows,
    descriptive_rows,
    paired_rows,
    paired_summary_rows,
):

    expected_repetitions = int(
        environment[
            "measured_runs_per_configuration"
        ]
    )

    expected_iterations = len(
        environment[
            "iterations"
        ]
    )

    models = {
        row["model"]
        for row in measured_rows
    }

    expected_models = len(
        models
    )

    expected_implementations = len(
        IMPLEMENTATIONS
    )

    expected_measured = (
        expected_models
        * expected_iterations
        * expected_implementations
        * expected_repetitions
    )

    expected_configurations = (
        expected_models
        * expected_iterations
        * expected_implementations
    )

    expected_blocks = (
        expected_models
        * expected_iterations
        * expected_repetitions
    )

    expected_paired_summary = (
        expected_models
        * expected_iterations
        * len(COMPARISONS)
    )

    if len(measured_rows) != expected_measured:
        raise RuntimeError(
            f"Expected {expected_measured} measured rows, "
            f"found {len(measured_rows)}"
        )

    if len(descriptive_rows) != expected_configurations:
        raise RuntimeError(
            f"Expected {expected_configurations} descriptive rows, "
            f"found {len(descriptive_rows)}"
        )

    if len(paired_rows) != expected_blocks:
        raise RuntimeError(
            f"Expected {expected_blocks} paired blocks, "
            f"found {len(paired_rows)}"
        )

    if len(paired_summary_rows) != expected_paired_summary:
        raise RuntimeError(
            f"Expected {expected_paired_summary} paired summary rows, "
            f"found {len(paired_summary_rows)}"
        )

    for row in descriptive_rows:

        if int(row["runs"]) != expected_repetitions:
            raise RuntimeError(
                "Unexpected repetition count for "
                f"{row['model']} / "
                f"{row['iterations']} / "
                f"{row['implementation']}: "
                f"{row['runs']}"
            )


def write_csv(
    path,
    rows,
):

    if not rows:
        raise RuntimeError(
            f"No rows to write: {path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():

    environment = load_environment()

    measured_rows = (
        load_measured_rows()
    )

    descriptive_rows = (
        create_descriptive(
            measured_rows
        )
    )

    paired_rows = (
        build_paired_ratios(
            measured_rows
        )
    )

    paired_summary_rows = (
        create_paired_summary(
            paired_rows
        )
    )

    validate_counts(
        environment,
        measured_rows,
        descriptive_rows,
        paired_rows,
        paired_summary_rows,
    )

    write_csv(
        DESCRIPTIVE_OUTPUT,
        descriptive_rows,
    )

    write_csv(
        PAIRED_RAW_OUTPUT,
        paired_rows,
    )

    write_csv(
        PAIRED_SUMMARY_OUTPUT,
        paired_summary_rows,
    )

    print(
        f"Measured rows: "
        f"{len(measured_rows)}"
    )

    print(
        f"Paired blocks: "
        f"{len(paired_rows)}"
    )

    print(
        f"Descriptive rows: "
        f"{len(descriptive_rows)}"
    )

    print(
        f"Paired summary rows: "
        f"{len(paired_summary_rows)}"
    )

    print()

    print(
        f"Wrote {DESCRIPTIVE_OUTPUT}"
    )

    print(
        f"Wrote {PAIRED_RAW_OUTPUT}"
    )

    print(
        f"Wrote {PAIRED_SUMMARY_OUTPUT}"
    )

    print()

    print(
        "PAPER BENCHMARK ANALYSIS COMPLETED"
    )


if __name__ == "__main__":
    main()