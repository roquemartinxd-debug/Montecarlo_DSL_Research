from pathlib import Path
import csv
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "benchmarks"
    / "baseline_results"
    / "final"
    / "v1.2.0"
    / "baseline_overhead_analysis.csv"
)

OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "v1.2.0_overhead_comparison.png"


models = []
manual = []
dsl_full = []

with INPUT.open(
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        models.append(row["model"])
        manual.append(float(row["manual_time_seconds"]))
        dsl_full.append(float(row["dsl_full_time_seconds"]))


x = range(len(models))

plt.figure(figsize=(8, 4))

plt.bar(
    [i - 0.2 for i in x],
    manual,
    width=0.4,
    label="Python manual"
)

plt.bar(
    [i + 0.2 for i in x],
    dsl_full,
    width=0.4,
    label="DSL full"
)

plt.xticks(
    list(x),
    models,
    rotation=20
)

plt.ylabel("Mean execution time (seconds)")
plt.title("Monte Carlo DSL v1.2.0 overhead comparison")

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300
)

plt.close()

print(f"Wrote {OUTPUT}")