from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_table(path: Path, title: str, rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(label for _, label in columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, str):
                value = value.replace("|", "\\|").replace("\n", " ")
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_exp10(results: Path, tables: Path, figures: Path) -> None:
    for phase in ("exploratory", "confirmatory"):
        summary = read_csv(results / f"exp10_break_even_{phase}_summary.csv")
        thresholds = read_csv(results / f"exp10_break_even_{phase}_thresholds.csv")
        if not summary:
            continue
        write_table(
            tables / f"exp10_break_even_{phase}_summary.md",
            f"EXP-10 — Break-Even Scheduling ({phase})",
            summary,
            [
                ("platform_label", "Platform"), ("model", "Model"), ("iterations", "N"),
                ("configuration", "Configuration"), ("runs", "Runs"),
                ("median_effective_workers", "Median workers"),
                ("median_execution_seconds", "Median execution (s)"),
                ("median_speedup_vs_serial", "Speedup vs serial"),
                ("speedup_vs_serial_ci95_low", "Speedup CI95 low"),
                ("speedup_vs_serial_ci95_high", "Speedup CI95 high"),
                ("best_fixed_configuration", "Best fixed"),
                ("adaptive_regret_vs_best_fixed_pct", "Adaptive regret (%)"),
                ("adaptive_regret_ci95_low_pct", "Regret CI95 low (%)"),
                ("adaptive_regret_ci95_high_pct", "Regret CI95 high (%)"),
            ],
        )
        write_table(
            tables / f"exp10_break_even_{phase}_thresholds.md",
            f"EXP-10 — Sustained Break-Even Thresholds ({phase})",
            thresholds,
            [("platform_label", "Platform"), ("model", "Model"),
             ("configuration", "Configuration"),
             ("first_sustained_break_even_iterations", "First sustained N"),
             ("criterion", "Criterion")],
        )
        models = sorted({row["model"] for row in summary})
        for model in models:
            model_rows = [row for row in summary if row["model"] == model]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for config in sorted({row["configuration"] for row in model_rows}):
                rows = sorted((row for row in model_rows if row["configuration"] == config),
                              key=lambda row: int(row["iterations"]))
                xs = [int(row["iterations"]) for row in rows]
                ys = [float(row["median_speedup_vs_serial"]) for row in rows]
                lows = [float(row["speedup_vs_serial_ci95_low"]) for row in rows]
                highs = [float(row["speedup_vs_serial_ci95_high"]) for row in rows]
                ax.plot(xs, ys, marker="o", label=config)
                ax.fill_between(xs, lows, highs, alpha=0.12)
            ax.axhline(1.0, linestyle="--", linewidth=1, label="Break-even (1.0x)")
            ax.set_xscale("log")
            ax.set_xlabel("Iterations (N)")
            ax.set_ylabel("Paired speedup vs serial in-process")
            ax.set_title(f"EXP-10 {phase}: {model}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            figures.mkdir(parents=True, exist_ok=True)
            fig.savefig(figures / f"exp10_{phase}_{model}_break_even.png", dpi=220)
            plt.close(fig)


def generate_exp11(results: Path, tables: Path, figures: Path) -> None:
    summary = read_csv(results / "exp11_memory_scaling_summary.csv")
    slopes = read_csv(results / "exp11_memory_scaling_slopes.csv")
    fidelity = read_csv(results / "exp11_memory_fidelity.csv")
    structures = read_csv(results / "exp11_memory_structural_bounds.csv")
    if not summary:
        return
    write_table(
        tables / "exp11_memory_scaling_summary.md",
        "EXP-11 — Bounded Memory Scaling Summary",
        summary,
        [("platform_label", "Platform"), ("stratum", "Stratum"), ("model", "Model"),
         ("iterations", "N"), ("memory_mode", "Memory mode"),
         ("worker_configuration", "Workers"), ("task_submission_order", "Task order"),
         ("runs", "Runs"), ("median_retained_payload_bytes", "Median retained payload (bytes)"),
         ("median_process_tree_rss_peak_bytes", "Median sampled process-tree RSS peak (bytes)"),
         ("median_incremental_process_tree_rss_bytes", "Median incremental process-tree RSS (bytes)"),
         ("median_max_futures_observed", "Max futures observed"),
         ("median_max_completed_results_observed", "Max completed results observed"),
         ("median_max_pending_logical_blocks_observed", "Max pending logical blocks observed"),
         ("median_max_paired_correlations_observed", "Max paired correlations observed")],
    )
    write_table(
        tables / "exp11_memory_scaling_slopes.md",
        "EXP-11 — Descriptive Log-Log Memory Scaling Slopes",
        slopes,
        [("platform_label", "Platform"), ("model", "Model"), ("memory_mode", "Mode"),
         ("worker_configuration", "Workers"), ("iteration_levels", "Iteration levels"),
         ("beta_retained_payload", "Beta retained payload"), ("r2_retained_payload", "R2 retained payload"),
         ("beta_process_tree_rss", "Beta process-tree RSS"), ("r2_process_tree_rss", "R2 process-tree RSS"),
         ("beta_incremental_process_tree_rss", "Beta incremental RSS"),
         ("r2_incremental_process_tree_rss", "R2 incremental RSS"), ("interpretation", "Interpretation")],
    )
    write_table(
        tables / "exp11_memory_fidelity.md",
        "EXP-11 — Paired retain_all vs bounded Fidelity",
        fidelity,
        [("platform_label", "Platform"), ("model", "Model"), ("iterations", "N"),
         ("worker_configuration", "Workers"), ("repetition", "Repetition"),
         ("exact_moments_preserved", "Exact moments"), ("exact_counts_preserved", "Exact counts"),
         ("max_quantile_relative_error_pct", "Max quantile relative error (%)"),
         ("max_histogram_tv_distance_pct", "Max histogram TV (%)"),
         ("branches_compared", "Branches compared"), ("paired_delta_compared", "Paired delta compared")],
    )
    write_table(
        tables / "exp11_memory_structural_bounds.md",
        "EXP-11 — Observed Structural High-Water Marks",
        structures,
        [("platform_label", "Platform"), ("model", "Model"), ("iterations", "N"),
         ("memory_mode", "Mode"), ("worker_configuration", "Workers"),
         ("task_submission_order", "Task order"), ("max_in_flight_tasks", "In-flight limit"),
         ("max_futures_observed", "Futures observed"),
         ("max_completed_results_observed", "Completed results observed"),
         ("max_pending_logical_blocks_observed", "Pending logical blocks observed"),
         ("max_block_coverage_entries_observed", "Block coverage entries observed"),
         ("max_branch_batch_stats_observed", "Branch stats entries observed"),
         ("max_paired_batch_stats_observed", "Paired stats entries observed"),
         ("max_paired_correlations_observed", "Paired correlations observed"),
         ("max_global_sketch_points", "Sketch capacity"),
         ("max_pending_logical_blocks_limit", "Pending block limit"),
         ("logical_block_size", "Logical block size"), ("batch_size", "Physical batch size")],
    )

    models = sorted({row["model"] for row in summary})
    for model in models:
        model_rows = [row for row in summary if row["model"] == model and row["stratum"] == "scaling"]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for key in sorted({(row["memory_mode"], row["worker_configuration"]) for row in model_rows}):
            rows = sorted((row for row in model_rows
                           if (row["memory_mode"], row["worker_configuration"]) == key),
                          key=lambda row: int(row["iterations"]))
            xs = [int(row["iterations"]) for row in rows]
            ys = [max(1.0, float(row["median_incremental_process_tree_rss_bytes"])) / (1024 * 1024)
                  for row in rows]
            ax.plot(xs, ys, marker="o", label=f"{key[0]} / {key[1]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Iterations (N)")
        ax.set_ylabel("Incremental sampled process-tree RSS (MiB)")
        ax.set_title(f"EXP-11 memory scaling: {model}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures / f"exp11_{model}_memory_scaling.png", dpi=220)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate EXP-10/EXP-11 tables and figures")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()
    results = args.results_dir.expanduser().resolve()
    tables = args.tables_dir.expanduser().resolve()
    figures = args.figures_dir.expanduser().resolve()
    generate_exp10(results, tables, figures)
    generate_exp11(results, tables, figures)
    print(f"Generated targeted EXP-10/EXP-11 artifacts from {results}")
    print(f"Tables: {tables}")
    print(f"Figures: {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
