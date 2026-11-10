from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "benchmarks"
    / "baseline_results"
    / "final"
    / "v1.2.0"
    / "baseline_timing_repeated.csv"
)

OUTPUT = (
    ROOT
    / "benchmarks"
    / "baseline_results"
    / "final"
    / "v1.2.0"
    / "baseline_overhead_analysis.csv"
)


def load_results():

    rows = []

    with INPUT.open(
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    return rows


def main():

    data = load_results()

    grouped = {}

    for row in data:

        model = row["model"]

        if model not in grouped:
            grouped[model] = {}

        grouped[model][row["implementation"]] = float(
            row["mean_time_seconds"]
        )


    output_rows = []

    for model, values in grouped.items():

        manual = values["python_manual"]
        generated = values["dsl_generated"]
        full = values["dsl_full"]

        output_rows.append(
            {
                "model": model,
                "manual_time_seconds": manual,
                "dsl_generated_time_seconds": generated,
                "dsl_full_time_seconds": full,
                "generated_ratio": generated / manual,
                "full_ratio": full / manual,
                "generated_overhead_percent":
                    ((generated - manual) / manual) * 100,
                "full_overhead_percent":
                    ((full - manual) / manual) * 100,
            }
        )


    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(output_rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(output_rows)


    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()