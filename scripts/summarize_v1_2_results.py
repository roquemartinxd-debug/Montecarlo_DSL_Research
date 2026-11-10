from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "final" / "v1.2.0"
OUT_DIR = ROOT / "docs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "v1.2.0_experiment_summary.md"

def read_csv(name):
    path = RESULTS / name
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "si"}

def fnum(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)

validation = read_csv("statistical_validation.csv")
benchmark = read_csv("benchmark_summary_with_speedup.csv")

validation_total = len(validation)
validation_pass = sum(1 for row in validation if as_bool(row.get("within_3_mcse")))
validation_rate = validation_pass / validation_total if validation_total else 0.0

by_model = defaultdict(lambda: {"total": 0, "pass": 0})
for row in validation:
    model = row["model_name"]
    by_model[model]["total"] += 1
    by_model[model]["pass"] += int(as_bool(row.get("within_3_mcse")))

benchmark_sorted = sorted(
    benchmark,
    key=lambda r: (r["model"], int(r["iterations"]), int(r["workers_requested"]))
)

multi_worker = [
    row for row in benchmark_sorted
    if int(row["workers_requested"]) > 1
]

positive_speedups = [
    row for row in multi_worker
    if float(row["speedup_vs_1_worker"]) > 1.0
]

non_positive_speedups = [
    row for row in multi_worker
    if float(row["speedup_vs_1_worker"]) <= 1.0
]

best = max(
    benchmark_sorted,
    key=lambda r: float(r["speedup_vs_1_worker"])
)

lines = []
lines.append("# Experimental summary for Monte Carlo DSL v1.2.0")
lines.append("")
lines.append("## Reproducibility context")
lines.append("")
lines.append("- Version: v1.2.0")
lines.append("- Results directory: `benchmarks/results/final/v1.2.0/`")
lines.append("- Validation file: `statistical_validation.csv`")
lines.append("- Benchmark file: `benchmark_summary_with_speedup.csv`")
lines.append("")
lines.append("## Statistical validation")
lines.append("")
lines.append(f"- Validation rows: {validation_total}")
lines.append(f"- Rows within 3 MCSE: {validation_pass}")
lines.append(f"- Pass rate: {validation_rate:.2%}")
lines.append("")
lines.append("| Model | Rows | Within 3 MCSE | Pass rate |")
lines.append("|---|---:|---:|---:|")
for model, data in sorted(by_model.items()):
    rate = data["pass"] / data["total"] if data["total"] else 0.0
    lines.append(f"| {model} | {data['total']} | {data['pass']} | {rate:.2%} |")

lines.append("")
lines.append("## Benchmark summary")
lines.append("")
lines.append(f"- Benchmark configurations: {len(benchmark_sorted)}")
lines.append(f"- Multi-worker configurations: {len(multi_worker)}")
lines.append(f"- Multi-worker configurations with speedup > 1.0: {len(positive_speedups)}")
lines.append(f"- Multi-worker configurations with speedup <= 1.0: {len(non_positive_speedups)}")
lines.append("")
lines.append(
    "- Best observed speedup: "
    f"{fnum(best['speedup_vs_1_worker'])} "
    f"for `{best['model']}` with {best['iterations']} iterations "
    f"and {best['workers_requested']} worker(s)."
)
lines.append("")
lines.append("| Model | Iterations | Workers | Runs | Mean time (s) | Speedup | Efficiency |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for row in benchmark_sorted:
    lines.append(
        f"| {row['model']} | {row['iterations']} | {row['workers_requested']} | "
        f"{row['runs']} | {fnum(row['mean_time_seconds'])} | "
        f"{fnum(row['speedup_vs_1_worker'])} | {fnum(row['parallel_efficiency'])} |"
    )

lines.append("")
lines.append("## Preliminary interpretation")
lines.append("")
lines.append(
    "The statistical validation supports the correctness of the v1.2.0 statistical "
    "core for the tested analytical models, since the estimated means fall within "
    "the expected Monte Carlo error criterion."
)
lines.append("")
lines.append(
    "The benchmark results do not support claiming strong parallel scalability in "
    "the evaluated local setting. For the tested workloads, multi-worker execution "
    "often shows limited or negative speedup, suggesting that execution overhead "
    "dominates the computational cost of these models."
)
lines.append("")
lines.append(
    "These results should be presented as a reproducibility and validation baseline, "
    "not as evidence of optimized high-performance parallel execution."
)
lines.append("")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
