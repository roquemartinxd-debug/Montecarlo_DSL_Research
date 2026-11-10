# Discussion and Interpretation of Results — MC DSL v1.3.1

> **Nota de version v1.3.1:** las cifras que aparecen en este documento proceden de la campana historica v1.3.0. Los cambios v1.3.1 en buffers, planificador, RSS, cuantiles pareados y optimizador requieren regenerar los resultados antes de publicar.

> **Nota editorial sobre versiones (Separación v1.2.0 vs. v1.3.0):**
> - La **Parte I** desarrolla la discusión científica correspondiente a la arquitectura y los resultados experimentales de **MC DSL v1.3.0** (integrada también en la Sección 7 de [`docs/paper/v1.3.0_manuscript.md`](v1.3.0_manuscript.md)).
> - La **Parte II** preserva la discusión original de la versión **v1.2.0**, que diagnosticó las limitaciones de escalabilidad multiproceso y sensibilidad al tamaño de lote físico que dieron origen al diseño de v1.3.0.

---

> **Nota de version:** las cifras de los CSV v1.3.0 son historicas. Los cambios v1.3.1 en buffers, planificador, RSS, cuantiles pareados y optimizador requieren regenerar los resultados antes de publicar.

## Parte I — Discusion cientifica de MC DSL v1.3.1

### 1. Por qué la Planificación Adaptativa Requiere PRNGs Indexados por Bloque Lógico y Doble Acumulador de Progreso
El hallazgo central al comparar v1.2.0 y v1.3.0 es que la planificación adaptativa de recursos y la reproducibilidad numérica estricta entran en conflicto directo cuando el estado del generador pseudoaleatorio se vincula a las unidades físicas de trabajo. En v1.2.0 (`seed_strategy="batch_index"`), modificar `batch_size` entre $20,000$, $25,000$ y $50,000$ producía tres realizaciones estocásticas distintas (`unique_hashes = 3`), haciendo imposible que el planificador fusionara o dividiera lotes dinámicamente sin alterar el resultado numérico del experimento.

En v1.3.0, al indexar instancias basadas en contador `numpy.random.Philox` mediante la tupla lógica `(MASTER_SEED, logical_block_index, var_index)` con un tamaño de bloque lógico canónico $L = 5,000$, la asignación de números pseudoaleatorios queda completamente determinada por la especificación del experimento. Además, para soportar tareas físicas más pequeñas que un bloque lógico ($B_{\text{phys}} < L$) sin violar el contrato de los eventos de progreso en tiempo real (`test_generator_contract.py`), el coordinador separa el **acumulador de progreso de tareas físicas** (`progress_sketch` y `stats_state`, donde `histogram.total_count == valid_results` en cada evento `batch`) del **acumulador canónico de bloques lógicos** (`canonical_sketch` y `canonical_stats`, que reduce cada bloque lógico completo exactamente una vez en orden ascendente $b = 0, 1, \dots, B-1$). Así, el runtime obtiene exactamente el mismo hash SHA-256 canónico (`unique_result_hashes = 1` en los 6 modelos y 14 configuraciones evaluadas) independientemente del número de workers, tamaño físico de tarea u orden de llegada.

### 2. Cuándo Compensa la Ejecución Serial en Proceso frente al Multiprocesamiento
Los CSV historicos de v1.3.0 (`performance_comparison_summary.csv` y `v1_2_vs_v1_3_baseline_comparison.csv`) usan distintos numeros de repeticiones segun el experimento; no todas las celdas tienen 30. Los resultados sugieren que ningun modo estatico es optimo en todos los regimenes, pero deben revalidarse en v1.3.1:
- **Regímenes dominados por coordinación ($N \le 10^6$ en modelos ligeros o de complejidad media)**: El coste de crear procesos de sistema operativo, serializar descriptores de tarea por tuberías IPC y deserializar resúmenes estadísticos supera el tiempo de evaluación vectorial en NumPy. Al estimar $\hat{T}_{\text{serial}} \le \hat{T}_{\text{parallel}}$ con el modelo de coste entrenado e inyectado en el compilador y runtime, `AdaptiveMonteCarloScheduler` selecciona ejecución serial directa en proceso (`effective_workers = 1`), evitando la sobrecarga de concurrencia e IPC y logrando aceleraciones de **`1.24x` a `1.43x`** frente a `v1.3_fixed_w1`, de **`1.16x` a `1.56x`** frente a `v1.3_fixed_w2` a $N = 1,000,000$ (`5.82x`–`7.62x` a $N = 10,000$ y `2.05x`–`4.75x` a $N = 100,000$), y de **`2.72x` a `5.70x`** ($N = 100,000$) y **`2.41x` a `5.17x`** ($N = 1,000,000$) frente a la emulacion de configuracion historica v1.2.0 (no al codigo fuente original).
- **Regímenes dominados por cómputo ($N = 10,000,000$ o modelos de muy alto coste por muestra)**: La calibración piloto mediante contracción ponderada por tamaño muestral ($\alpha = L/(L+N_0) = 0.20$) corrige cualquier desviación del prior estático del AST ($\text{MAPE} = 5.55\%$ fuera de muestra, mediana $5.04\%$, $r = 0.9899$, $\rho = 1.0000$), activando `ProcessPoolExecutor` (`effective_workers = 2`, `use_process_pool = True`) cuando el paralelismo amortiza la sobrecarga de IPC y alcanzando un speedup paralelo de **`1.36x`** (`1.0785 s` vs `1.4675 s`, IC 95 % `[1.32, 1.39]`).

### 3. Memoria retenida del coordinador, RSS por proceso y fidelidad de cuantiles

La version v1.3.1 limita la ventana multiproceso a `2 * worker_count` tareas y los buffers de reordenamiento de bloques a `MAX_PENDING_LOGICAL_BLOCKS = 64`. Bajo tamano maximo de tarea, workers y ventana fijos, el numero de resultados retenidos fuera de orden queda acotado respecto a N. Esta afirmacion no implica una cota independiente de esos parametros y debe verificarse con una campaña de escalado que incluya envio `forward`, `reverse` y `shuffled`.

`peak_retained_coordinator_bytes` estima el payload de arrays y buffers observables en los puntos de ingesta; no equivale a RSS total. `sum_of_process_peak_rss_bytes` suma maximos historicos por proceso y no es un pico simultaneo del arbol de procesos. Para reportar un pico simultaneo se requiere muestreo del arbol de procesos o cgroup.

La salida pareada `paired_delta` incluye ahora histograma propio. La distancia TV debe calcularse a partir de los histogramas `retain_all` y `bounded`; cualquier valor constante o ausente no constituye evidencia. Los resultados numericos de v1.3.0 no validan estos cambios y deben regenerarse.

### 4. Alcance del optimizador del compilador

El optimizador v1.3.1 es deliberadamente conservador: pliega operaciones binarias entre literales despues de convertir cada operando a float64 y conserva el orden de redondeo del runtime. Elimina signo unario positivo y doble negacion, pero no elimina identidades dinamicas como `x + 0` o `x * 1`, ni sustituye `sqrt` o potencias NumPy por operaciones escalares Python. Esto reduce el conjunto de transformaciones disponibles, pero evita afirmar equivalencia bit a bit sin demostrarla. Los tiempos, reducciones de nodos y resultados de ablacion de v1.3.0 deben volver a medirse con esta version del optimizador.

## Parte II — Discusión Histórica de la Línea Base v1.2.0 (Contexto de Referencia)

### Performance–Abstraction Trade-off in v1.2.0

The comparison with `numpy_native` in v1.2.0 quantified the computational cost associated with the DSL infrastructure. For the evaluated workloads, the single-worker v1.2.0 DSL required approximately 2.6 times the end-to-end runtime of native NumPy at 10,000 and 100,000 iterations, with the relative difference increasing at 1,000,000 iterations.

This overhead reflects the complete abstraction pipeline: semantic analysis, code generation, deterministic random-number configuration, batch-oriented execution, statistical aggregation, non-finite result handling, percentile computation, and structured runtime reporting.

### Scaling Relative to the Reference Statistical Implementation in v1.2.0

At smaller workloads, `python_reference` was faster than the v1.2.0 DSL, whereas at 1,000,000 iterations the single-worker v1.2.0 DSL required 30–39% less end-to-end time than `python_reference` because batch-vectorized evaluation scaled better than per-sample Python updates.

### Parallel Execution and Reproducibility Boundary in v1.2.0

In v1.2.0, increasing the worker count from one to two under a fixed batch size of 25,000 degraded runtime by 8–14% at $10^5$–$10^6$ iterations on lightweight models, and changing `batch_size` altered the canonical numerical hash because `SeedSequence([MASTER_SEED, batch_index])` tied random substreams to physical batch size. Both limitations were directly addressed and resolved by the v1.3.0 architecture discussed in Parte I.
