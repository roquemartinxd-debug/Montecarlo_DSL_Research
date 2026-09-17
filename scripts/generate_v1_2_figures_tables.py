from pathlib import Path
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "final" / "v1.2.0"
FIGURES = ROOT / "docs" / "figures"
TABLES = ROOT / "docs" / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

benchmark_file = RESULTS / "benchmark_summary_with_speedup.csv"
validation_file = RESULTS / "statistical_validation.csv"

def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

benchmark = read_csv(benchmark_file)
validation = read_csv(validation_file)

# Figure 1: mean execution time by workers
for model in sorted(set(row["model"] for row in benchmark)):
    rows = [row for row in benchmark if row["model"] == model]
    iterations_values = sorted(set(int(row["iterations"]) for row in rows))

    for iterations in iterations_values:
        subset = [
            row for row in rows
            if int(row["iterations"]) == iterations
        ]
        subset = sorted(subset, key=lambda r: int(r["workers_requested"]))

        workers = [int(row["workers_requested"]) for row in subset]
        mean_times = [float(row["mean_time_seconds"]) for row in subset]

        plt.figure()
        plt.plot(workers, mean_times, marker="o")
        plt.xlabel("Requested workers")
        plt.ylabel("Mean execution time (seconds)")
        plt.title(f"{model} - {iterations} iterations")
        plt.xticks(workers)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out = FIGURES / f"{model}_{iterations}_mean_time.png"
        plt.savefig(out, dpi=300)
        plt.close()

# Figure 2: speedup by workers
for model in sorted(set(row["model"] for row in benchmark)):
    rows = [row for row in benchmark if row["model"] == model]
    iterations_values = sorted(set(int(row["iterations"]) for row in rows))

    for iterations in iterations_values:
        subset = [
            row for row in rows
            if int(row["iterations"]) == iterations
        ]
        subset = sorted(subset, key=lambda r: int(r["workers_requested"]))

        workers = [int(row["workers_requested"]) for row in subset]
        speedups = [float(row["speedup_vs_1_worker"]) for row in subset]

        plt.figure()
        plt.plot(workers, speedups, marker="o")
        plt.xlabel("Requested workers")
        plt.ylabel("Speedup vs 1 worker")
        plt.title(f"{model} - {iterations} iterations")
        plt.xticks(workers)
        plt.axhline(y=1.0, linestyle="--")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out = FIGURES / f"{model}_{iterations}_speedup.png"
        plt.savefig(out, dpi=300)
        plt.close()

# Table 1: validation summary
validation_by_model = defaultdict(lambda: {"total": 0, "passed": 0})

for row in validation:
    model = row["model_name"]
    validation_by_model[model]["total"] += 1
    if row["within_3_mcse"].lower() == "true":
        validation_by_model[model]["passed"] += 1

table_validation = TABLES / "v1.2.0_validation_summary.md"
with table_validation.open("w", encoding="utf-8") as f:
    f.write("| Model | Rows | Within 3 MCSE | Pass rate |\n")
    f.write("|---|---:|---:|---:|\n")

    for model, data in sorted(validation_by_model.items()):
        rate = data["passed"] / data["total"] * 100
        f.write(
            f"| {model} | {data['total']} | {data['passed']} | {rate:.2f}% |\n"
        )

# Table 2: benchmark summary
table_benchmark = TABLES / "v1.2.0_benchmark_summary.md"
with table_benchmark.open("w", encoding="utf-8") as f:
    f.write("| Model | Iterations | Workers | Runs | Mean time (s) | Speedup | Efficiency |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")

    for row in sorted(
        benchmark,
        key=lambda r: (r["model"], int(r["iterations"]), int(r["workers_requested"]))
    ):
        f.write(
            f"| {row['model']} | "
            f"{row['iterations']} | "
            f"{row['workers_requested']} | "
            f"{row['runs']} | "
            f"{float(row['mean_time_seconds']):.3f} | "
            f"{float(row['speedup_vs_1_worker']):.3f} | "
            f"{float(row['parallel_efficiency']):.3f} |\n"
        )

print("Generated figures in:", FIGURES)
print("Generated tables in:", TABLES)
