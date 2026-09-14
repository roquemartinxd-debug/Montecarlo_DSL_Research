from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "benchmarks" / "results" / "processed"


def main() -> int:
    summary = PROCESSED_DIR / "benchmark_summary.csv"
    if not summary.exists():
        print("No existe benchmark_summary.csv. Ejecuta primero benchmarks/run_benchmarks.py")
        return 1
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    baselines = {}
    for row in rows:
        if row["workers_requested"] == "1":
            baselines[(row["model"], row["iterations"])] = float(row["median_time_seconds"])
    out_rows = []
    for row in rows:
        key = (row["model"], row["iterations"])
        baseline = baselines.get(key)
        median = float(row["median_time_seconds"])
        workers = int(row["workers_requested"])
        speedup = baseline / median if baseline and median else None
        efficiency = speedup / workers if speedup else None
        row = dict(row)
        row["speedup_vs_1_worker"] = speedup
        row["parallel_efficiency"] = efficiency
        out_rows.append(row)
    out = PROCESSED_DIR / "benchmark_summary_with_speedup.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
