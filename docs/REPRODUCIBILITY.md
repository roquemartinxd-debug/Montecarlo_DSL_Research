# Reproducibilidad Independiente de la Planificación Física (v1.3.0)

## 1. Desacoplamiento entre Bloque Lógico y Tarea Física

En la versión **1.3.0**, `montecarlo-dsl` resuelve la limitación documentada en v1.2.0 (donde cambiar `batch_size` alteraba la asignación de semillas y modificaba el resultado numérico) separando dos conceptos que antes estaban acoplados:

- **Bloque lógico (`logical_block_size`, por defecto `5,000`)**: Define qué muestras pertenecen al experimento, cómo se inicializan sus streams pseudoaleatorios y en qué orden determinista se reducen sus estados estadísticos.
- **Tarea física (`batch_size` o agrupación adaptativa de bloques)**: Define qué subconjunto de muestras o de bloques lógicos contiguos $[b_{\text{start}}, b_{\text{end}})$ procesa un worker (o el bucle serial en proceso) durante una ejecución determinada.

## 2. Generación Pseudoaleatoria Basada en Contador (`Philox`)

Cada bloque lógico $b \in \{0, \dots, B-1\}$ y cada variable aleatoria $v \in \{0, \dots, K-1\}$ declarada en el programa recibe un stream independiente derivado mediante:

```python
ss = np.random.SeedSequence([MASTER_SEED, logical_block_index, var_index])
rng = np.random.default_rng(np.random.Philox(ss))
```

De este modo, la identidad aleatoria de una muestra depende exclusivamente de identificadores lógicos estables (`MASTER_SEED`, índice de bloque lógico $b$, índice de variable $v$ y posición de extracción dentro del bloque), y **no** del worker que la procesa, ni del tamaño físico de la tarea (`batch_size`), ni del modo del planificador (`fixed`, `ast_only`, `pilot_only`, `hybrid`).

## 3. Reducción Determinista por Frontera Contigua y Separación entre Progreso Físico y Estado Canónico

Cada bloque lógico completo $b$ calcula su propio estado de Welford / Chan–Golub–LeVeque (`batch_stats[b]`), su correlación pareada (`correlations[b]`) y su resumen ponderado de cuantiles (`block_summaries[b]`). A medida que la frontera contigua `b = 0, 1, ..., B - 1` avanza (`_advance_state_frontier` y `_advance_paired_frontier`), los estados se fusionan exactamente en orden ascendente de índice de bloque lógico dentro del acumulador canónico global, del acumulador compensado de Neumaier (1974) para correlaciones pareadas y del boceto ponderado de capacidad fija (`MAX_GLOBAL_SKETCH_POINTS = 2048`), descartando de inmediato los bloques ya reducidos.

Cuando una tarea física divide un bloque lógico en varios sub-rangos ($B_{\text{phys}} < L$), el coordinador actualiza en paralelo:
1. Un **acumulador de progreso físico** (`progress_sketch`, `stats_state`) con las muestras de cada tarea física terminada, garantizando que todo evento de progreso `batch` cumpla `histogram.total_count == valid_results`.
2. Un **acumulador canónico de bloques lógicos** (`canonical_sketch`, `canonical_stats`) que solo reduce el resumen del bloque lógico cuando se ha verificado la cobertura de sus $L_b$ muestras (`block_covered_samples[b] == L_b`), preservando un resultado final idéntico bit a bit con independencia del tamaño físico de tarea o del orden de llegada.

## 4. Evidencia Experimental en v1.3.0

En `benchmarks/results/final/v1.3.0/schedule_invariance_summary.csv` (`docs/tables/v1.3.0_schedule_invariance.md`), se evaluaron 6 modelos (`normal_mean`, `uniform_mean`, `discrete_mean`, `linear_combination`, `analytic_pm` y `quadratic_pm`) con $1,000,000$ de iteraciones a través de **14 configuraciones de planificación física** (84 ejecuciones primarias) que varían simultáneamente:

- Número de workers: `1`, `2` y `4`.
- Tamaño físico de lote: `10,000`, `20,000`, `25,000`, `50,000`, `100,000` y agrupaciones adaptativas.
- Modo del planificador: `fixed`, `ast_only`, `pilot_only`, `hybrid`.
- Orden de finalización de tareas: `as_completed`, `reverse` y `shuffled`.

Para cada uno de los 6 modelos, las 14 configuraciones produjeron **exactamente un único hash SHA-256** del resultado numérico final canonizado (`unique_result_hashes = 1`, `PASS`).

## 5. Alcance de la Garantía

La garantía de igualdad exacta bit a bit (SHA-256) se establece para una configuración lógica fija (`modelo DSL + MASTER_SEED + TOTAL_ITERATIONS + logical_block_size + memory_mode`) dentro de la misma versión de código, entorno de software (`Python 3.11.2`, `NumPy 2.1.3`, `FastAPI 0.92.0`, `Pydantic 1.10.4`) y plataforma. Los resultados históricos de v1.2.0 se conservan intactos en `benchmarks/results/rng_reproducibility/v1.2.0/`.
