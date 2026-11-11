# Experimental Evaluation — Historical v1.3.0 Campaign and v1.2.0 Audit

> **Nota editorial sobre versiones (Separación v1.2.0 vs. v1.3.0):**
> - La **Parte I** conserva los resultados historicos de **MC DSL v1.3.0**. Los tamaños muestrales varian por experimento: Exp. 0 usa 30 repeticiones por celda; Exp. 1 y Exp. 4 una; Exp. 2 usa 30 solo en celdas seleccionadas y 10 en las demas; Exp. 3, 5, 6 y 9 usan 10; Exp. 7 usa 25 semillas; Exp. 8 usa una ejecucion por configuracion. Exp. 0 compara contra una emulacion de configuracion historica, no contra el codigo original v1.2.0. El manuscrito historico esta en [`docs/paper/v1.3.0_manuscript.md`](v1.3.0_manuscript.md).
> - La **Parte II** conserva como registro histórico verificable la evaluación original de la línea base **v1.2.0** (`numpy_native`, `python_reference`, `dsl_full_w1`, `dsl_full_w2` con `SeedSequence([MASTER_SEED, batch_index])`), cuyos datos congelados residen en `benchmarks/results/final/v1.2.0/` y `benchmarks/paper_results/final/v1.2.0/`.

> **Estado:** documento historico de la campana v1.3.0. No es evidencia experimental de v1.3.1. La nueva version debe ejecutar la campana y regenerar tablas/figuras antes de publicacion.

---

## Parte I — Síntesis de la Evaluación Experimental de MC DSL v1.3.0

### 1. Entorno Experimental y Coherencia de Dependencias
El registro historico de la campana v1.3.0 no es conforme con el `pyproject.toml`:, registrado en `benchmarks/results/final/v1.3.0/environment.json`:
- **Python**: `3.11.2` (`GCC 12.2.0`, Linux `x86_64`, 2 CPUs lógicas)
- **NumPy**: `1.24.2` (incompatible con `numpy>=2.0`; el entorno historico no es conforme)
- **SciPy**: `1.15.2` (`scipy>=1.10.0`)
- **FastAPI / Pydantic / Uvicorn / HTTPX**: `FastAPI 0.92.0`, `Pydantic 1.10.4`, `Uvicorn 0.17.6`, `HTTPX 0.23.3`. El recuento de pruebas de v1.3.0 no se debe presentar como verificacion completa de v1.3.1.
- **Protocolo de repeticiones**:
  - **Exp. 0:** 30 repeticiones por celda de modelo/iteraciones/variante.
  - **Exp. 1:** una ejecucion por modelo y modo de planificacion.
  - **Exp. 2:** 30 repeticiones solo para `normal_mean` y `heavy_expression` a 100.000 y 1.000.000; 10 en las demas celdas y 10 en escalabilidad a 10.000.000.
  - **Exp. 3, 5, 6 y 9:** 10 repeticiones por celda/modelo segun el script.
  - **Exp. 4:** una ejecucion por modelo/configuracion.
  - **Exp. 7:** 25 semillas independientes por modelo analitico.
  - **Exp. 8:** una ejecucion por configuracion de memoria/fidelidad.
  - **Intervalos:** bootstrap de 10.000 remuestreos; cada CSV conserva el numero real de corridas.

### 2. Separación Explícita de Baselines
Para evitar cualquier ambigüedad metodológica, la campaña v1.3.0 distingue explícitamente dos tipos de comparaciones:
1. **Comparacion contra la emulacion de configuracion historica v1.2.0** (`benchmarks/results/final/v1.3.0/v1_2_vs_v1_3_baseline_comparison.csv` y `docs/tables/v1.3.0_vs_v1.2.0_baseline.md`): ejecuta el runtime actual con parametros de configuracion historicos (`seed_strategy="batch_index"`, `batch_size=25,000`, `memory_mode="retain_all"`, `ProcessPoolExecutor` fijo); no ejecuta el codigo fuente original v1.2.0 frente a `v1.3.0_fixed_w2` y `v1.3.0_adaptive_hybrid` con el numero de repeticiones registrado en el CSV. A $N = 1,000,000$, `v1.3.0_fixed_w2` alcanza **`2.05x`–`3.14x`** de aceleración solo por la compactacion de memoria bajo parametros fijos, y `v1.3.0_adaptive_hybrid` logra aceleraciones de **`2.41x` (`quadratic_pm`) a `5.17x` (`normal_mean`)** a $N = 1,000,000$ (y de **`2.72x` a `5.70x`** a $N = 100,000$), reduciendo la carga util retenida en el coordinador en los datos historicos; la suma de maximos RSS por proceso no es un pico simultaneo agregado.
2. **Comparación controlada entre variantes internas de v1.3.0** (`benchmarks/results/final/v1.3.0/performance_comparison_summary.csv` y `docs/tables/v1.3.0_performance_comparison.md`): mantiene fijo el motor de generación `philox_logical_block`, el modelo de coste entrenado y el modo de memoria acotada bajo parametros fijos para aislar el efecto puro del planificador (`v1.3_fixed_w1`, `v1.3_fixed_w2`, `v1.3_fixed_w4`, `v1.3_adaptive_hybrid`). A $N = 1,000,000$, `v1.3_adaptive_hybrid` alcanza speedups medianos de **`1.24x` a `1.43x`** frente a `v1.3_fixed_w1` y de **`1.16x` a `1.56x`** frente a `v1.3_fixed_w2` (y de **`2.49x` a `4.75x`** frente a `v1.3_fixed_w2` a $N = 100,000$). A $N = 10,000,000$, activa automáticamente `effective_workers = 2` logrando **`1.36x`** de speedup paralelo frente a `v1.3_fixed_w1`.

### 3. Resumen de Resultados por Pregunta de Investigación (`RQ1`–`RQ4`)
- **RQ1 (Invariancia de reproducibilidad frente a la planificación)**: En 6 modelos y 14 configuraciones de planificación física (workers $\in \{1, 2, 4\}$, tamaños físicos $\in \{10\text{k}, 20\text{k}, 25\text{k}, 50\text{k}, 100\text{k}\}$, modos `fixed`, `ast_only`, `pilot_only`, `hybrid` y órdenes `as_completed`, `reverse`, `shuffled`), cada modelo produjo exactamente **1 hash SHA-256 canónico único** (`84/84` ejecuciones idénticas).
- **RQ2 (Modelo de coste del AST con 4 índices estructurales compuestos)**: Al proyectar el AST sobre 4 índices estructurales compuestos interpretables ($\text{rango} = 4$, $\kappa_{\text{raw}} = 11.3601$, $\kappa_{\text{std}} = 3.4054$, $r_{\text{arith,trans}} = 0.8234$) y calibrar mediante regresión Ridge restringida exacta (KKT) sobre 8 modelos de entrenamiento, el prior estático alcanza **$\text{MAPE} = 12.35\%$** ($\rho = 1.0000$) en los 6 modelos no vistos, y la contracción ponderada por tamaño muestral con el bloque piloto ($b = 0$) reduce el MAPE fuera de muestra a **`5.55%`** (error mediano **`5.04%`**, Pearson $r = 0.9899$, Spearman $\rho = 1.0000$, `ast_cost_generalization_summary.csv`).
- **RQ3 (Rendimiento y ablación $4 \times 2$)**: El estudio de ablación (`ablation_study_summary.csv`, 10 repeticiones por variante) confirma que las variantes adaptativas superan a `fixed (unopt)` por **`1.75x`–`1.80x`** en `normal_mean`, **`1.40x`–`1.45x`** en `heavy_expression`, **`1.11x`–`1.24x`** en `quadratic_pm` y **`1.43x`–`1.59x`** en `optimizable_heavy` (`1.57x` para `hybrid + opt=True`).
- **RQ4 (Carga retenida acotada bajo parametros fijos, RSS por proceso y precision de cuantiles)**: Mediante compactación por frontera contigua de centroides equiprobables (`MAX_GLOBAL_SKETCH_POINTS = 2048`) y suma compensada de correlaciones pareadas, la carga util retenida en el coordinador se mantuvo estable en las configuraciones historicas medidas en **`0.207 MB`** (`normal_mean`) y **`0.583 MB`** (`quadratic_pm`) a $2.5 \times 10^6$ iteraciones (**reducción del `99.0%` y `98.1%`** en proceso, y **`98.8%` y `97.8%`** con 2 workers multiproceso), con error `0.000%` en distribuciones discretas y error relativo maximo historico en cuantiles **$\le 0.174\%$** a través de 6 familias de distribución y 10 ramas (`memory_retention_comparison.csv` y `quantile_and_histogram_fidelity.csv`).

---

## Parte II — Evaluación Histórica de la Línea Base v1.2.0 (Registro Congelado)

### Experimental Setup (v1.2.0 Baseline)

The historical v1.2.0 performance evaluation considered four representative Monte Carlo models: a normal mean, a uniform mean, a discrete mean, and a linear combination of random variables. Three workload sizes were evaluated: 10,000, 100,000, and 1,000,000 Monte Carlo iterations.

Four implementations were compared:

1. `numpy_native`, an independent implementation based directly on vectorized NumPy operations;
2. `python_reference`, a Python implementation using the statistical core of the project;
3. `dsl_full_w1`, the v1.2.0 DSL execution pipeline configured with one worker; and
4. `dsl_full_w2`, the same v1.2.0 DSL pipeline configured with two workers.

Each configuration was executed 30 times after two warm-up runs. Execution order was randomized using a deterministic scheduling seed. All implementations were executed in fresh Python subprocesses. For the DSL configurations, the reported end-to-end latency additionally includes DSL compilation and generated-script creation.

The complete benchmark comprised 1,536 process executions, of which 1,440 corresponded to measured runs and 96 to warm-up executions.

Performance results are reported primarily using the median because several configurations exhibited right-tailed timing outliers. Paired runtime ratios were also calculated within each randomized execution block. Median paired ratios were accompanied by percentile-bootstrap 95% confidence intervals.

### Comparison with Native NumPy (v1.2.0 Baseline)

The native NumPy implementation consistently achieved the lowest end-to-end execution time across all evaluated models and workload sizes.

For 10,000 iterations, the median paired runtime ratio between the single-worker DSL and NumPy ranged from 2.613 to 2.632. At 100,000 iterations, the ratio remained between 2.621 and 2.680. At 1,000,000 iterations, the relative cost increased, ranging from 3.363 for the discrete model to 4.041 for the uniform model.

Therefore, the proposed DSL should not be interpreted as a performance optimization over direct vectorized NumPy execution. Instead, its runtime overhead represents the computational cost associated with the higher-level execution infrastructure provided by the DSL.

### Comparison with the Python Reference Implementation (v1.2.0 Baseline)

A different scaling behavior was observed when the v1.2.0 DSL was compared with the Python reference implementation based on the project's statistical core.

At 10,000 iterations, the single-worker DSL required approximately 2.15–2.18 times the runtime of the Python reference implementation. At 100,000 iterations, this ratio decreased to approximately 1.56–1.58.

At 1,000,000 iterations, the relationship reversed. The median paired DSL/reference ratios ranged from 0.606 to 0.700 across the four models. Under these tested conditions, the single-worker DSL therefore required approximately 30–39% less end-to-end time than the reference implementation at the largest workload.

### Effect of Worker Count in v1.2.0 (Motivation for v1.3.0 Adaptive Scheduling)

Increasing the v1.2.0 DSL worker count from one to two did not produce a measurable benefit for the tested lightweight workloads:
- At 10,000 iterations, the paired `w2/w1` ratios remained close to 1.0.
- At 100,000 iterations, two workers were approximately 13–14% slower than one worker.
- At 1,000,000 iterations, the two-worker configuration remained approximately 8–10% slower.

These results motivated the v1.3.0 `AdaptiveMonteCarloScheduler`, which predicts per-sample execution cost and executes lightweight workloads serially in-process while reserving `ProcessPoolExecutor` for compute-intensive models where parallel speedup outweighs IPC overhead.

### Historical RNG Reproducibility Across Worker Configurations (v1.2.0 Baseline)

In v1.2.0, the four analytical models (`normal_mean`, `uniform_mean`, `discrete_mean`, and `linear_combination`) were executed with 1,000,000 Monte Carlo iterations, a master seed of 101, a fixed batch size of 25,000 samples (40 logical batches), and worker counts of 1, 2, and 4 (10 repetitions each, 120 primary executions).

| Model | Runs across 1/2/4 workers | Unique result hashes | Exact reproducibility |
|---|---:|---:|---|
| `normal_mean` | 30 | 1 | PASS |
| `uniform_mean` | 30 | 1 | PASS |
| `discrete_mean` | 30 | 1 | PASS |
| `linear_combination` | 30 | 1 | PASS |

However, the v1.2.0 negative control showed that changing the batch size among 20,000, 25,000, and 50,000 samples produced three distinct hashes because v1.2.0 derived one random-number substream per physical batch using `numpy.SeedSequence([MASTER_SEED, batch_index])`. This limitation was resolved in v1.3.0 by decoupling fixed logical blocks (`logical_block_size = 5,000`) with counter-based `Philox` streams from physical task granularity.
