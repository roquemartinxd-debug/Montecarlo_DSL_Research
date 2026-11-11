from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "benchmarks"
    / "paper_results"
    / "final"
    / "v1.2.0"
)

DESCRIPTIVE_INPUT = (
    RESULTS_DIR
    / "paper_benchmark_descriptive.csv"
)

PAIRED_INPUT = (
    RESULTS_DIR
    / "paper_benchmark_paired_ratios_summary.csv"
)

TABLES_DIR = ROOT / "docs" / "tables"
FIGURES_DIR = ROOT / "docs" / "figures"

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_ORDER = [
    "normal_mean",
    "uniform_mean",
    "discrete_mean",
    "linear_combination",
]

MODEL_LABELS = {
    "normal_mean": "Normal mean",
    "uniform_mean": "Uniform mean",
    "discrete_mean": "Discrete mean",
    "linear_combination": "Linear combination",
}

IMPLEMENTATION_ORDER = [
    "numpy_native",
    "python_reference",
    "dsl_full_w1",
    "dsl_full_w2",
]

IMPLEMENTATION_LABELS = {
    "numpy_native": "NumPy native",
    "python_reference": "Python reference",
    "dsl_full_w1": "DSL — 1 worker",
    "dsl_full_w2": "DSL — 2 workers",
}

ITERATION_ORDER = [
    10_000,
    100_000,
    1_000_000,
]


def read_csv(path: Path):

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def load_descriptive():

    rows = read_csv(
        DESCRIPTIVE_INPUT
    )

    return {
        (
            row["model"],
            int(row["iterations"]),
            row["implementation"],
        ): row
        for row in rows
    }


def load_paired():

    rows = read_csv(
        PAIRED_INPUT
    )

    return {
        (
            row["model"],
            int(row["iterations"]),
            row["comparison"],
        ): row
        for row in rows
    }


def ratio_text(row):

    median = float(
        row["median_ratio"]
    )

    low = float(
        row[
            "median_bootstrap_ci95_low"
        ]
    )

    high = float(
        row[
            "median_bootstrap_ci95_high"
        ]
    )

    return (
        f"{median:.3f} "
        f"[{low:.3f}, {high:.3f}]"
    )


def write_summary_table(
    descriptive,
    paired,
):

    output = (
        TABLES_DIR
        / "v1.2.0_final_benchmark_summary.md"
    )

    lines = [
        "# Final Paper Benchmark Summary",
        "",
        (
            "Times are medians across 30 measured runs. "
            "Ratio columns report the median paired ratio "
            "with a 95% bootstrap confidence interval."
        ),
        "",
        (
            "| Model | Iterations | NumPy native median (s) | "
            "Python reference median (s) | DSL w1 median (s) | "
            "DSL w2 median (s) | DSL w1 / NumPy | "
            "DSL w1 / Python reference | DSL w2 / DSL w1 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for model in MODEL_ORDER:

        for iterations in ITERATION_ORDER:

            numpy_row = descriptive[
                (
                    model,
                    iterations,
                    "numpy_native",
                )
            ]

            reference_row = descriptive[
                (
                    model,
                    iterations,
                    "python_reference",
                )
            ]

            w1_row = descriptive[
                (
                    model,
                    iterations,
                    "dsl_full_w1",
                )
            ]

            w2_row = descriptive[
                (
                    model,
                    iterations,
                    "dsl_full_w2",
                )
            ]

            w1_numpy = paired[
                (
                    model,
                    iterations,
                    "dsl_w1_vs_numpy_native",
                )
            ]

            w1_reference = paired[
                (
                    model,
                    iterations,
                    "dsl_w1_vs_python_reference",
                )
            ]

            w2_w1 = paired[
                (
                    model,
                    iterations,
                    "dsl_w2_vs_dsl_w1",
                )
            ]

            lines.append(
                "| "
                f"{MODEL_LABELS[model]} | "
                f"{iterations:,} | "
                f"{float(numpy_row['median_seconds']):.3f} | "
                f"{float(reference_row['median_seconds']):.3f} | "
                f"{float(w1_row['median_seconds']):.3f} | "
                f"{float(w2_row['median_seconds']):.3f} | "
                f"{ratio_text(w1_numpy)} | "
                f"{ratio_text(w1_reference)} | "
                f"{ratio_text(w2_w1)} |"
            )

    lines.extend(
        [
            "",
            (
                "**Interpretation:** a ratio greater than 1 means "
                "the numerator implementation required more "
                "end-to-end execution time than the denominator."
            ),
            "",
        ]
    )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Wrote {output}"
    )


def plot_runtime_scaling(
    descriptive,
):

    for model in MODEL_ORDER:

        plt.figure(
            figsize=(8, 5)
        )

        for implementation in IMPLEMENTATION_ORDER:

            medians = []
            lower_errors = []
            upper_errors = []

            for iterations in ITERATION_ORDER:

                row = descriptive[
                    (
                        model,
                        iterations,
                        implementation,
                    )
                ]

                median = float(
                    row["median_seconds"]
                )

                q1 = float(
                    row["q1_seconds"]
                )

                q3 = float(
                    row["q3_seconds"]
                )

                medians.append(
                    median
                )

                lower_errors.append(
                    median - q1
                )

                upper_errors.append(
                    q3 - median
                )

            plt.errorbar(
                ITERATION_ORDER,
                medians,
                yerr=[
                    lower_errors,
                    upper_errors,
                ],
                marker="o",
                capsize=3,
                label=IMPLEMENTATION_LABELS[
                    implementation
                ],
            )

        plt.xscale("log")

        plt.xlabel(
            "Monte Carlo iterations"
        )

        plt.ylabel(
            "Median end-to-end time (seconds)"
        )

        plt.title(
            "Runtime scaling — "
            f"{MODEL_LABELS[model]}"
        )

        plt.legend()

        plt.tight_layout()

        output = (
            FIGURES_DIR
            / f"v1.2.0_runtime_{model}.png"
        )

        plt.savefig(
            output,
            dpi=300,
        )

        plt.close()

        print(
            f"Wrote {output}"
        )


def plot_ratio(
    paired,
    comparison,
    title,
    ylabel,
    filename,
):

    plt.figure(
        figsize=(8, 5)
    )

    for model in MODEL_ORDER:

        medians = []
        lower_errors = []
        upper_errors = []

        for iterations in ITERATION_ORDER:

            row = paired[
                (
                    model,
                    iterations,
                    comparison,
                )
            ]

            median = float(
                row["median_ratio"]
            )

            low = float(
                row[
                    "median_bootstrap_ci95_low"
                ]
            )

            high = float(
                row[
                    "median_bootstrap_ci95_high"
                ]
            )

            medians.append(
                median
            )

            lower_errors.append(
                median - low
            )

            upper_errors.append(
                high - median
            )

        plt.errorbar(
            ITERATION_ORDER,
            medians,
            yerr=[
                lower_errors,
                upper_errors,
            ],
            marker="o",
            capsize=3,
            label=MODEL_LABELS[model],
        )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.xscale("log")

    plt.xlabel(
        "Monte Carlo iterations"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURES_DIR
        / filename
    )

    plt.savefig(
        output,
        dpi=300,
    )

    plt.close()

    print(
        f"Wrote {output}"
    )


def main():

    descriptive = (
        load_descriptive()
    )

    paired = (
        load_paired()
    )

    write_summary_table(
        descriptive,
        paired,
    )

    plot_runtime_scaling(
        descriptive
    )

    plot_ratio(
        paired,
        comparison=(
            "dsl_w1_vs_numpy_native"
        ),
        title=(
            "DSL single-worker overhead "
            "relative to NumPy native"
        ),
        ylabel=(
            "Paired runtime ratio "
            "(DSL w1 / NumPy)"
        ),
        filename=(
            "v1.2.0_ratio_dsl_w1_vs_numpy.png"
        ),
    )

    plot_ratio(
        paired,
        comparison=(
            "python_reference_vs_numpy_native"
        ),
        title=(
            "Python reference overhead "
            "relative to NumPy native"
        ),
        ylabel=(
            "Paired runtime ratio "
            "(Python reference / NumPy)"
        ),
        filename=(
            "v1.2.0_ratio_python_reference_vs_numpy.png"
        ),
    )

    plot_ratio(
        paired,
        comparison=(
            "dsl_w2_vs_dsl_w1"
        ),
        title=(
            "Effect of two workers "
            "relative to one worker"
        ),
        ylabel=(
            "Paired runtime ratio "
            "(DSL w2 / DSL w1)"
        ),
        filename=(
            "v1.2.0_ratio_w2_vs_w1.png"
        ),
    )

    print()
    print(
        "PAPER ARTIFACT GENERATION COMPLETED"
    )


if __name__ == "__main__":
    main()