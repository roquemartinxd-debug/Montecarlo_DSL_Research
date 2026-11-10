# Auditoría y Congelación de la Línea Base v1.2.0

## 1. Objetivo de la auditoría

Siguiendo la secuencia técnica establecida para la defensa ante la VAE (RNCP39278) y la preparación del manuscrito científico, este documento audita y congela el estado exacto de `montecarlo-dsl` v1.2.0 (commit base `428a54c`) antes de introducir la arquitectura v1.3.0. Todos los artefactos experimentales de la versión v1.2.0 se preservan intactos como línea base histórica y comparativa.

## 2. Artefactos congelados de la línea base v1.2.0

Se mantienen sin modificación los siguientes directorios y resultados experimentales registrados en v1.2.0:

- `benchmarks/results/final/v1.2.0/`:
  - `benchmark_runs.csv`, `benchmark_summary.csv`, `benchmark_summary_with_speedup.csv` (18 configuraciones: `normal_mean` y `heavy_expression`, $10^4, 10^5, 10^6$ iteraciones, 1, 2 y 4 workers, 10 repeticiones).
  - `statistical_validation.csv` (4 modelos analíticos $\times$ 5 semillas = 20 ejecuciones con 100,000 iteraciones; 20/20 dentro de $3 \times \text{MCSE}$).
  - `nonfinite_validation.csv` (modelo `sqrt_partial_invalid` con validación de política `discard_and_report` e intervalo de Wilson).
- `benchmarks/paper_results/final/v1.2.0/`:
  - `paper_benchmark_raw.csv`, `paper_benchmark_summary.csv`, `paper_benchmark_descriptive.csv`, `paper_benchmark_paired_ratios_raw.csv`, `paper_benchmark_paired_ratios_summary.csv`, `environment.json` (1,440 ejecuciones medidas + 96 de calentamiento comparando `numpy_native`, `python_reference`, `dsl_full_w1` y `dsl_full_w2`).
- `benchmarks/results/rng_reproducibility/v1.2.0/`:
  - `rng_reproducibility_raw.csv`, `rng_reproducibility_summary.csv`, `rng_reproducibility_controls.csv`, `environment.json` y los 125 eventos JSON en `raw/`.
- `benchmarks/baseline_results/final/v1.2.0/`:
  - `baseline_comparison.csv`, `baseline_overhead_analysis.csv`, `baseline_timing_repeated.csv`.

## 3. Hallazgos de la inspección técnica de v1.2.0 (Documentación vs. Comportamiento Real)

### 3.1. Acoplamiento entre partición lógica RNG y tamaño físico de tarea (`batch_size`)
- **Comportamiento en v1.2.0**: El runtime deriva semillas por lote mediante `np.random.SeedSequence([MASTER_SEED, batch_index])` con el generador por defecto `PCG64`, donde `batch_index` indexa directamente los lotes físicos de tamaño `BATCH_SIZE`.
- **Limitación verificada**: Aunque el resultado es invariante al número de workers (`1, 2, 4`) cuando `BATCH_SIZE` permanece fijo, el control negativo en `benchmarks/results/rng_reproducibility/v1.2.0/rng_reproducibility_controls.csv` demuestra que cambiar `batch_size` (`20,000`, `25,000`, `50,000`) altera el número de muestras extraídas de cada substream y produce 3 hashes SHA-256 distintos. Esto impide que un planificador adaptativo modifique el tamaño físico de lote sin alterar los resultados numéricos.

### 3.2. Retención de muestras en memoria (`all_chunks`) y sobrecoste $O(B \cdot N)$ en el coordinador
- **Comportamiento en v1.2.0**: Aunque `statistics.py` y `generator.py` implementan acumulación incremental de Welford y combinación de Chan–Golub–LeVeque para media y varianza, cada worker serializa y envía el arreglo completo `values` (`float64`) al proceso padre, que lo almacena en `state["all_chunks"]` (y `paired_state["all_chunks"]`).
- **Discrepancia y cuello de botella detectado**:
  1. En cada evento `batch` ($B = 40$ veces para $10^6$ iteraciones), `_branch_payload` invoca `_stats_payload` $\rightarrow$ `_quantile_payload(state["all_chunks"])`, ejecutando `np.concatenate(chunks)` y `np.quantile` sobre todas las muestras acumuladas hasta ese lote, incluso cuando `\response` no solicita cuantiles (`p05`, `p50`, `p95`). Esto introduce un coste cuadrático $O(B \cdot N)$ en el hilo coordinador.
  2. La retención completa en `all_chunks` hace que el consumo de memoria crezca linealmente $O(N)$ con el número total de iteraciones y penaliza la transferencia IPC entre workers.

### 3.3. Planificación fija sin modelo de coste ni adaptación a recursos
- **Comportamiento en v1.2.0**: El runtime divide siempre `TOTAL_ITERATIONS` en lotes fijos de `batch_size = 25,000` y lanza incondicionalmente un `ProcessPoolExecutor(max_workers=worker_count)`, incluso cuando `worker_count == 1` o cuando el coste de creación de procesos e IPC supera el tiempo de cómputo numérico del modelo.
- **Evidencia en línea base**: En `benchmark_summary_with_speedup.csv`, para `normal_mean` con $10^6$ iteraciones (en equipo de 2 CPUs lógicas), el tiempo mediano empeoró de `2.947 s` (1 worker) a `3.276 s` (2 workers) y `4.070 s` (4 workers). En `paper_benchmark_summary.csv`, para $10^4$ iteraciones, `dsl_full_w1` tardó `1.681 s` frente a `0.644 s` de `numpy_native`, dominado por el arranque del pool de procesos.

### 3.4. Ausencia de fase explícita de optimización del AST en el compilador
- **Comportamiento en v1.2.0**: `compiler.py` encadena `analyze_source` y `PythonCodeGenerator` sin una fase intermedia de optimización del AST. Subexpresiones completamente constantes, identidades algebraicas (`1 * x`, `x + 0`, `+x`, `-(-x)`) y potencias triviales (`x^1`, `x^2` traducido a `np.power`) se emiten sin simplificar.

### 3.5. Discrepancias menores de metadatos y entorno
- En `src/montecarlo_dsl/server.py`, `FastAPI(version="1.1.0")` no se había actualizado a `"1.2.0"` y `app.state.job_manager` se inicializaba únicamente dentro del contexto `lifespan`.

## 4. Criterio de comparación para v1.3.0

La auditoria y los artefactos de resultados de v1.2.0 se conservan como registro historico. **La comparacion Exp. 0 del runner v1.3.x no ejecuta este codigo fuente original**: utiliza el runtime actual con parametros de configuracion que emulan el comportamiento historico (`v1.2.0_legacy_configuration_emulation`). Una comparacion directa requiere recuperar y ejecutar el arbol de codigo original v1.2.0 bajo el mismo protocolo.
