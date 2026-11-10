from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmarks" / "results" / "final" / "v1.3.1"
TABLES_DIR = ROOT / "docs" / "tables"
FIGURES_DIR = ROOT / "docs" / "figures"


def _read_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS_DIR / name
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def generate_markdown_tables() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Historical Baseline v1.2.0 vs v1.3.1 Comparison Table
    base_rows = _read_csv("v1_2_vs_v1_3_baseline_comparison.csv")
    base_lines = [
        "# Comparación contra Emulación de Configuración Histórica v1.2.0 (`v1.2.0_legacy_configuration_emulation` vs. `v1.3.1`)",
        "",
        "El numero de corridas por celda se conserva en el CSV; Exp. 0 usa 30 repeticiones por celda. La variante `v1.2.0_legacy_configuration_emulation` ejecuta el runtime actual con la configuración histórica; no es una ejecución del código fuente original v1.2.0. El RSS se reporta como suma de máximos históricos por proceso, no como pico simultáneo agregado.",
        "",
        "| Modelo | Iteraciones | Implementación | Corridas | Workers efectivos | Lotes | Mediana (s) | IQR (s) | Speedup vs emulación v1.2.0 [IC 95 %] | Carga Retenida Coordinador (KB) | Reducción Carga Retenida vs emulación (%) | Suma de máximos RSS por proceso (MB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in base_rows:
        sp = float(r["median_speedup_vs_v1_2_0"])
        sp_l = float(r["speedup_vs_v1_2_0_ci95_low"])
        sp_h = float(r["speedup_vs_v1_2_0_ci95_high"])
        rss_mb = float(r.get("median_sum_of_process_peak_rss_mb", "0.0"))
        base_lines.append(
            f"| `{r['model']}` | {int(r['iterations']):,} | `{r['implementation']}` | {r['runs']} | "
            f"{r['effective_workers']} | {r['batch_count']} | {float(r['median_seconds']):.4f} | "
            f"{float(r['iqr_seconds']):.4f} | {sp:.2f}x [{sp_l:.2f}, {sp_h:.2f}] | "
            f"{float(r['median_peak_memory_kb']):.1f} | {float(r['memory_reduction_vs_v1_2_0_pct']):.1f}% | "
            f"{rss_mb:.1f} |"
        )
    (TABLES_DIR / "v1.3.1_vs_v1.2.0_baseline.md").write_text("\n".join(base_lines) + "\n", encoding="utf-8")

    # 1. Performance & Scalability Summary Table (Internal v1.3.1 Configurations)
    perf_rows = _read_csv("performance_comparison_summary.csv")
    perf_lines = [
        "# Rendimiento y Escalabilidad entre Configuraciones de v1.3.1",
        "",
        "El CSV conserva las repeticiones reales por celda. Se incluye un baseline serial en proceso (`v1.3_serial_in_process`, `ast_only`, un worker, sin ProcessPoolExecutor). Los speedups frente a ese baseline separan el beneficio del cálculo del coste de creación/coordinación de procesos. Los speedups históricos frente a `v1.3_fixed_w1` y `v1.3_fixed_w2` se conservan como comparaciones secundarias.",
        "",
        "| Modelo | Iteraciones | Configuración | Corridas | Workers efectivos | Lotes | Pool | Mediana (s) | IQR (s) | Speedup vs serial en proceso [IC 95 %] | Speedup vs fixed w1 [IC 95 %] | Speedup vs fixed w2 [IC 95 %] | Carga retenida (KB) | Suma máximos RSS (MB) |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in perf_rows:
        sp1 = float(r["median_speedup_vs_fixed_w1"])
        sp1_l = float(r["speedup_vs_w1_ci95_low"])
        sp1_h = float(r["speedup_vs_w1_ci95_high"])
        sp2 = float(r["median_speedup_vs_fixed_w2"])
        sp2_l = float(r["speedup_vs_w2_ci95_low"])
        sp2_h = float(r["speedup_vs_w2_ci95_high"])
        sp_serial = float(r.get("median_speedup_vs_serial_in_process", "nan"))
        sp_serial_l = float(r.get("speedup_vs_serial_ci95_low", "nan"))
        sp_serial_h = float(r.get("speedup_vs_serial_ci95_high", "nan"))
        pool_label = str(r.get("use_process_pool", ""))
        mem_kb = int(r["median_peak_memory_bytes"]) / 1024.0
        rss_mb = int(r.get("median_sum_of_process_peak_rss_bytes", "0")) / (1024.0 * 1024.0)
        perf_lines.append(
            f"| `{r['model']}` | {int(r['iterations']):,} | `{r['configuration']}` | {r['runs']} | "
            f"{r['effective_workers']} | {r['batch_count']} | {pool_label} | "
            f"{float(r['median_seconds']):.4f} | {float(r['iqr_seconds']):.4f} | "
            f"{sp_serial:.2f}x [{sp_serial_l:.2f}, {sp_serial_h:.2f}] | "
            f"{sp1:.2f}x [{sp1_l:.2f}, {sp1_h:.2f}] | "
            f"{sp2:.2f}x [{sp2_l:.2f}, {sp2_h:.2f}] | {mem_kb:.1f} | {rss_mb:.1f} |"
        )
    perf_text = "\n".join(perf_lines) + "\n"
    (TABLES_DIR / "v1.3.1_performance_comparison.md").write_text(perf_text, encoding="utf-8")
    (TABLES_DIR / "v1.3.1_performance_summary.md").write_text(perf_text, encoding="utf-8")

    # 2. Schedule-Independent Reproducibility Summary
    repro_rows = _read_csv("schedule_invariance_summary.csv")
    ctrl_rows = _read_csv("schedule_invariance_controls.csv")
    repro_lines = [
        "# Reproducibilidad Independiente de la Planificación Física (v1.3.1)",
        "",
        "## Invariancia frente a workers, tamaño físico de tarea, modo de planificación y orden de finalización",
        "",
        "| Modelo | Semilla | Iteraciones | Tamaño bloque lógico | Planificaciones probadas | Hashes únicos | Invariancia exacta | Hash SHA-256 canónico |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in repro_rows:
        repro_lines.append(
            f"| `{r['model']}` | {r['seed']} | {int(r['iterations']):,} | "
            f"{int(r['logical_block_size']):,} | {r['schedules_tested']} | "
            f"{r['unique_result_hashes']} | {'PASS' if r['exact_schedule_invariance'] == 'True' else 'FAIL'} | "
            f"`{r['reference_sha256'][:16]}...` |"
        )
    repro_lines.extend([
        "",
        "## Controles Experimentales",
        "",
        "| Control | Descripción | Criterio esperado | Hashes únicos | Resultado |",
        "|---|---|---|---:|---|",
    ])
    for c in ctrl_rows:
        repro_lines.append(
            f"| `{c['control']}` | {c['description']} | `{c['expected']}` | "
            f"{c['unique_hashes']} | {'PASS' if c['passed'] == 'True' else 'FAIL'} |"
        )
    (TABLES_DIR / "v1.3.1_schedule_invariance.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")

    # 3. AST Cost Model Generalization (Per-Model + Aggregate Summary)
    cost_rows = _read_csv("ast_cost_generalization.csv")
    cost_sum_rows = _read_csv("ast_cost_generalization_summary.csv")
    d0 = cost_sum_rows[0]
    cost_lines = [
        "# Evaluación y Generalización del Modelo de Coste Basado en el AST (v1.3.1)",
        "",
        "El prior estático del AST emplea **4 índices estructurales compuestos interpretables** (`[intercepto, índice_distribuciones, índice_aritmético_pm, índice_trascendente]`) ajustados mediante regresión Ridge con restricciones de cota inferior (solución exacta KKT) sobre **8 modelos de entrenamiento** (10 repeticiones por modelo) y evaluados sobre **6 modelos de prueba no vistos**, propagando los coeficientes ajustados al planificador del runtime.",
        "",
        f"- **Diagnósticos de la matriz de diseño de entrenamiento ($8 \\times 4$)**: rango $= {d0.get('matrix_rank', '4')}$, número de condición sin estandarizar $\\kappa(X) = {d0.get('cond_raw', '11.3601')}$, número de condición tras estandarizar los índices $\\kappa(X_{{\\text{{std}}}}) = {d0.get('cond_standardized', '3.4054')}$, correlación entre índice aritmético e índice trascendental $r(I_{{\\text{{arith}}}}, I_{{\\text{{trans}}}}) = {d0.get('arith_trans_corr', '0.8234')}$.",
        f"- **Coeficientes ajustados $\\hat{{\\theta}}$ (ns/muestra)**: `{d0.get('fitted_theta_ns', '')}`.",
        "",
        "## Métricas Agregadas de Generalización (Train vs. Out-of-Sample Test)",
        "",
        "| Conjunto | Modelos | Parámetros ajustados | Corridas/modelo | MAPE Prior AST (%) | Error Mediano AST (%) | RMSE AST (ns) | Pearson $r$ AST | Spearman $\\rho$ AST | MAPE Híbrido (%) | Error Mediano Híbrido (%) | RMSE Híbrido (ns) | Pearson $r$ Híbrido | Spearman $\\rho$ Híbrido |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in cost_sum_rows:
        cost_lines.append(
            f"| `{s['split']}` | {s['models_count']} | {s['parameters_fitted']} | {s['runs_per_model']} | "
            f"{s['static_mape_pct']}% | {s['static_median_rel_error_pct']}% | {s['static_rmse_ns']} | "
            f"{s['static_pearson_r']} | {s['static_spearman_rho']} | "
            f"{s['hybrid_mape_pct']}% | {s['hybrid_median_rel_error_pct']}% | {s['hybrid_rmse_ns']} | "
            f"{s['hybrid_pearson_r']} | {s['hybrid_spearman_rho']} |"
        )
    cost_lines.extend([
        "",
        "## Desglose por Modelo (10 Repeticiones por Modelo a 500,000 Iteraciones)",
        "",
        "| Modelo | Conjunto | Corridas | Vars | Ramas | Nodos AST | Score complejidad | Observado (ns/muestra) | Prior Estático AST (ns/muestra) | Calibrado Híbrido (ns/muestra) | Error Relativo AST (%) | Error Relativo Híbrido (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in cost_rows:
        cost_lines.append(
            f"| `{r['model']}` | `{r['split']}` | {r['runs']} | {r['num_random_vars']} | {r['branch_count']} | "
            f"{r['ast_node_count']} | {r['weighted_op_score']} | {r['observed_per_sample_ns']} | "
            f"{r['static_ast_predicted_ns']} | {r['hybrid_calibrated_predicted_ns']} | "
            f"{r['static_relative_error_pct']}% | {r['hybrid_relative_error_pct']}% |"
        )
    (TABLES_DIR / "v1.3.1_ast_cost_generalization.md").write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    # 4. Compiler Optimization Benchmark
    opt_rows = _read_csv("compiler_optimization_benchmark.csv")
    opt_lines = [
        "# Evaluación del Optimizador del Compilador y Pruebas Diferenciales (v1.3.1)",
        "",
        "| Modelo | Corridas | Nodos antes | Nodos después | Reducción nodos | Folds constantes | Simplificaciones algebraicas | Bytes script (sin/con) | Ejecución mediana sin opt (s) | Ejecución mediana con opt (s) | Speedup | Equivalencia SHA-256 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in opt_rows:
        opt_lines.append(
            f"| `{r['model']}` | {r['runs']} | {r['ast_nodes_before']} | {r['ast_nodes_after']} | "
            f"{r['nodes_reduced']} | {r['constant_folds']} | {r['algebraic_simplifications']} | "
            f"{r['script_bytes_unoptimized']} / {r['script_bytes_optimized']} | "
            f"{float(r['median_exec_seconds_unoptimized']):.4f} | {float(r['median_exec_seconds_optimized']):.4f} | "
            f"{float(r['speedup_ratio']):.3f}x | {'PASS' if r['exact_hash_match'] == 'True' else 'FAIL'} |"
        )
    (TABLES_DIR / "v1.3.1_compiler_optimization.md").write_text("\n".join(opt_lines) + "\n", encoding="utf-8")

    # 5. Statistical Validation (25 Seeds) & CI95 Coverage
    stat_rows = _read_csv("ci95_coverage_summary.csv")
    stat_lines = [
        "# Validación Estadística con 25 Semillas Independientes y Cobertura de IC 95 % (v1.3.1)",
        "",
        "| Modelo | Rama | Semillas | Media teórica | Media global estimada | Error medio (MCSE) | Error máx (MCSE) | Dentro de 3 MCSE | Cobertura empírica IC 95 % | Varianza teórica | Varianza media estimada |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in stat_rows:
        stat_lines.append(
            f"| `{r['model']}` | `{r['branch']}` | {r['seeds_evaluated']} | "
            f"{float(r['theoretical_mean']):.4f} | {float(r['mean_across_seeds']):.5f} | "
            f"{float(r['mean_error_in_mcse']):.3f} | {float(r['max_error_in_mcse']):.3f} | "
            f"{r['within_3_mcse_count']}/{r['seeds_evaluated']} ({float(r['within_3_mcse_rate'])*100:.1f}%) | "
            f"{r['ci95_coverage_count']}/{r['seeds_evaluated']} ({float(r['ci95_empirical_coverage'])*100:.1f}%) | "
            f"{float(r['theoretical_variance']):.4f} | {float(r['mean_estimated_variance']):.4f} |"
        )
    (TABLES_DIR / "v1.3.1_statistical_validation_25seeds.md").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")

    # 6. Memory Retention & RSS Comparison + Extended Quantile & Histogram Fidelity
    mem_rows = _read_csv("memory_retention_comparison.csv")
    q_rows = _read_csv("quantile_and_histogram_fidelity.csv")
    mem_lines = [
        "# Comparación de Carga Retenida del Coordinador bajo Ventana Acotada vs. RSS por Proceso (`bounded` vs. `retain_all`) y Precisión de Cuantiles e Histograma",
        "",
        "El modo `bounded` combina un boceto ponderado de capacidad fija con una ventana acotada de tareas en vuelo y buffers de reordenamiento. Bajo tamaño máximo de tarea, número de workers y ventana fijos, la carga útil retenida por el coordinador está diseñada para permanecer acotada respecto a $N$. La suma de máximos RSS por proceso no representa un pico simultáneo agregado.",
        "",
        "## A. Escalado de Memoria Retenida del Coordinador y Suma de maximos RSS por proceso",
        "",
        "| Modo Ejecución | Workers | Modelo | Iteraciones | Carga Retenida `retain_all` (MB) | Carga Retenida `bounded` (MB) | Reducción Carga Retenida (%) | Suma máximos RSS `retain_all` (MB) | Suma máximos RSS `bounded` (MB) | Error Abs `p50` | Error Rel Máx Cuantiles (%) | Discrepancia TV Histograma (%) | Momentos, Correlación y Conteo Exactos |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in mem_rows:
        mem_lines.append(
            f"| `{r['execution_mode']}` | {r['effective_workers']} | `{r['model']}` | {int(r['iterations']):,} | "
            f"{float(r['retain_all_peak_mb']):.3f} | {float(r['bounded_peak_mb']):.3f} | "
            f"{float(r['memory_reduction_pct']):.1f}% | "
            f"{float(r['retain_all_sum_of_process_peak_rss_mb']):.1f} | {float(r['bounded_sum_of_process_peak_rss_mb']):.1f} | "
            f"{float(r['p50_abs_error']):.5f} | {float(r['max_quantile_rel_error_pct']):.3f}% | "
            f"{float(r['histogram_tv_distance_pct']):.3f}% | "
            f"{'PASS' if r['exact_moments_and_counts_preserved'] == 'True' else 'FAIL'} |"
        )
    mem_lines.extend([
        "",
        "## B. Fidelidad de Cuantiles Ponderados e Histograma en Distribuciones Multimodales, Asimétricas, Discretas, de Colas Pesadas y Ramas Pareadas ($N = 1,000,000$)",
        "",
        "| Modelo | Familia de Distribución | Rama | Muestras Válidas | Error Abs `p05` | Error Abs `p50` | Error Abs `p95` | Error Rel Máx Cuantiles (%) | Discrepancia TV Histograma (%) | Momentos y Correlación Exactos |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for q in q_rows:
        mem_lines.append(
            f"| `{q['model']}` | `{q['distribution_family']}` | `{q['branch']}` | {int(q['valid_results']):,} | "
            f"{float(q['p05_abs_error']):.5f} | {float(q['p50_abs_error']):.5f} | {float(q['p95_abs_error']):.5f} | "
            f"{float(q['max_quantile_rel_error_pct']):.3f}% | {float(q['histogram_tv_distance_pct']):.3f}% | "
            f"{'PASS' if q['exact_moments_preserved'] == 'True' else 'FAIL'} |"
        )
    (TABLES_DIR / "v1.3.1_memory_comparison.md").write_text("\n".join(mem_lines) + "\n", encoding="utf-8")

    # 7. Ablation Study Summary (10 Repetitions per Variant)
    abl_rows = _read_csv("ablation_study_summary.csv")
    abl_lines = [
        "# Estudio de Ablación $4 \\times 2$: Variantes del Planificador y Optimizador AST (10 Repeticiones por Variante)",
        "",
        "| Modelo | Variante Planificador | Optimizador AST | Corridas | Workers efectivos | Lotes | Mediana (s) | IQR (s) | Speedup vs `fixed (unopt)` [IC 95 %] |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in abl_rows:
        abl_lines.append(
            f"| `{r['model']}` | `{r['scheduler_variant']}` | `{r['optimizer_enabled']}` | {r['runs']} | "
            f"{r['effective_workers']} | {r['batch_count']} | {float(r['median_seconds']):.4f} | "
            f"{float(r['iqr_seconds']):.4f} | "
            f"{float(r['median_speedup_vs_fixed_unopt']):.2f}x [{float(r['speedup_ci95_low']):.2f}, {float(r['speedup_ci95_high']):.2f}] |"
        )
    (TABLES_DIR / "v1.3.1_ablation_study.md").write_text("\n".join(abl_lines) + "\n", encoding="utf-8")


def generate_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figura 1: tiempos de los baselines serial, fixed process-pool y adaptativo (1,000,000 iteraciones)
    perf_rows = [
        r for r in _read_csv("performance_comparison_summary.csv")
        if int(r["iterations"]) == 1_000_000
    ]
    models = ["normal_mean", "linear_combination", "heavy_expression", "quadratic_pm"]
    configs = ["v1.3_serial_in_process", "v1.3_fixed_w1", "v1.3_fixed_w2", "v1.3_fixed_w4", "v1.3_adaptive_hybrid"]
    labels = {
        "v1.3_serial_in_process": "Serial in-process",
        "v1.3_fixed_w1": "v1.3 Fixed (pool w=1)",
        "v1.3_fixed_w2": "v1.3 Fixed (w=2)",
        "v1.3_fixed_w4": "v1.3 Fixed (w=4)",
        "v1.3_adaptive_hybrid": "v1.3 Adaptive Hybrid (AST + Pilot)",
    }

    x = np.arange(len(models))
    width = 0.16
    plt.figure(figsize=(10, 5.2))
    for idx, cfg in enumerate(configs):
        vals = []
        for m in models:
            match = [r for r in perf_rows if r["model"] == m and r["configuration"] == cfg]
            vals.append(float(match[0]["median_seconds"]) if match else 0.0)
        plt.bar(x + (idx - 2.0) * width, vals, width=width, label=labels[cfg])

    plt.xticks(x, models, rotation=12)
    plt.ylabel("Median execution time (seconds)")
    plt.title("Monte Carlo DSL v1.3.1 — Serial, Fixed Pools and Adaptive Scheduler (1,000,000 iterations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v1.3.1_scheduler_performance_1m.png", dpi=300)
    plt.close()

    # Figura 2: carga útil retenida del coordinador (retain_all vs bounded)
    mem_rows = [
        r for r in _read_csv("memory_retention_comparison.csv")
        if r["model"] == "quadratic_pm" and r.get("execution_mode", "in_process_w1") == "in_process_w1"
    ]
    iters = [int(r["iterations"]) for r in mem_rows]
    ret_mb = [float(r["retain_all_peak_mb"]) for r in mem_rows]
    bnd_mb = [float(r["bounded_peak_mb"]) for r in mem_rows]

    plt.figure(figsize=(8, 4.8))
    plt.plot(iters, ret_mb, marker="o", linewidth=2, label="retain_all (full result retention)")
    plt.plot(iters, bnd_mb, marker="s", linewidth=2, label="bounded (fixed sketch + bounded in-flight window)")
    plt.xscale("log")
    plt.xlabel("Monte Carlo iterations")
    plt.ylabel("Retained coordinator payload memory (MB)")
    plt.title("Coordinator Payload Memory Scaling — Full Retention vs. Bounded Window")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v1.3.1_memory_scaling.png", dpi=300)
    plt.close()

    # Figura 3: Estudio de ablacion (4 variantes x optimizador, 10 repeticiones)
    abl_rows = [r for r in _read_csv("ablation_study_summary.csv") if r["model"] in ("heavy_expression", "optimizable_heavy")]
    variants = ["fixed", "ast_only", "pilot_only", "hybrid"]
    x_abl = np.arange(len(variants))
    plt.figure(figsize=(9, 4.8))
    for i, (m, opt) in enumerate([
        ("heavy_expression", "False"),
        ("heavy_expression", "True"),
        ("optimizable_heavy", "False"),
        ("optimizable_heavy", "True"),
    ]):
        speedups = []
        for v in variants:
            row = [r for r in abl_rows if r["model"] == m and r["scheduler_variant"] == v and r["optimizer_enabled"] == opt][0]
            speedups.append(float(row["median_speedup_vs_fixed_unopt"]))
        plt.bar(x_abl + (i - 1.5) * 0.18, speedups, width=0.18, label=f"{m} (opt={opt})")

    plt.axhline(1.0, linestyle="--", color="gray")
    plt.xticks(x_abl, ["Fixed", "AST Only", "Pilot Only", "Hybrid (AST+Pilot)"])
    plt.ylabel("Median Speedup vs. Fixed (unoptimized)")
    plt.title("Ablation Study: Scheduler Variants and AST Optimizer Effect (10 runs/variant)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v1.3.1_ablation_speedup.png", dpi=300)
    plt.close()

    # Figura 4: Generalizacion del modelo de coste del AST (8 Train + 6 Test)
    cost_rows = _read_csv("ast_cost_generalization.csv")
    names = [f"{r['model']} ({r['split'][0].upper()})" for r in cost_rows]
    obs = [float(r["observed_per_sample_ns"]) for r in cost_rows]
    ast_p = [float(r["static_ast_predicted_ns"]) for r in cost_rows]
    hyb_p = [float(r["hybrid_calibrated_predicted_ns"]) for r in cost_rows]

    x_c = np.arange(len(names))
    plt.figure(figsize=(11.5, 5.2))
    plt.bar(x_c - 0.25, obs, width=0.25, label="Observed (ns/sample)")
    plt.bar(x_c, ast_p, width=0.25, label="Static 4-Index KKT-Ridge AST Model (ns/sample)")
    plt.bar(x_c + 0.25, hyb_p, width=0.25, label="Hybrid Shrinkage Calibrated (ns/sample)")
    plt.xticks(x_c, names, rotation=28, ha="right")
    plt.ylabel("Time per sample (ns)")
    plt.title("AST Cost Model Generalization (8 Train [T] & 6 Out-of-Sample Test [T] Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v1.3.1_ast_cost_generalization.png", dpi=300)
    plt.close()


def main() -> int:
    global RESULTS_DIR, TABLES_DIR, FIGURES_DIR
    parser = argparse.ArgumentParser(description="Generate v1.3.1 research tables and figures")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir.expanduser().resolve()
    TABLES_DIR = args.tables_dir.expanduser().resolve()
    FIGURES_DIR = args.figures_dir.expanduser().resolve()
    generate_markdown_tables()
    generate_figures()
    print(f"Generated v1.3.1 tables and figures from {RESULTS_DIR}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
