# Registros de Decisiones Arquitectónicas (ADR) — Monte Carlo DSL v1.3.1

Este documento conserva las decisiones de diseño de v1.3.0 y registra las correcciones de estabilizacion v1.3.1. Los resultados numericos de v1.3.0 son historicos y no validan automaticamente el codigo v1.3.1; consultar `docs/REVALIDATION_REQUIRED_v1.3.1.md`.

---

## ADR-001: Desacoplamiento entre Identidad Lógica de Muestra y Planificación Física mediante PRNG Basado en Contador (`Philox`) y Acumuladores Duales (Progreso vs. Canónico)

### Contexto y Problema
En la versión v1.2.0, el generador pseudoaleatorio se inicializaba por lote físico mediante `np.random.SeedSequence([MASTER_SEED, batch_index])` con `PCG64`. Aunque este esquema garantizaba invariancia frente al número de workers (`1, 2, 4`) bajo un `batch_size` fijo, acoplaba la secuencia aleatoria al tamaño físico de la tarea (`batch_size`). Como demostró el control negativo de v1.2.0, variar `batch_size` entre `20,000`, `25,000` y `50,000` producía tres hashes SHA-256 distintos, impidiendo que un planificador adaptativo ajustara el tamaño de lote sin alterar los resultados numéricos del experimento. Además, cuando un bloque lógico se divide entre varias tareas físicas y la tarea que contiene el extremo derecho del bloque termina antes que las intermedias, mezclar el resumen del bloque lógico completo con las estadísticas de progreso de las tareas físicas rompería el invariante `histogram.total_count == valid_results` en los eventos `batch`.

### Alternativas Consideradas
1. **Mantener `SeedSequence([MASTER_SEED, batch_index])` sobre lotes físicos**: Simple, pero rompe la reproducibilidad en cuanto el planificador modifica el tamaño de lote.
2. **Diseñar un generador pseudoaleatorio ad hoc**: Descartado por riesgo estadístico; la literatura especializada desaconseja inventar PRNGs no auditados cuando existen generadores basados en contador probados (Salmon et al., 2011).
3. **Separar Bloque Lógico (`logical_block_size`) de Tarea Física utilizando `numpy.random.Philox` indexado por `(MASTER_SEED, logical_block_index, var_index)` con dos acumuladores diferenciados (progreso físico vs. frontera canónica)**: Cada bloque lógico $b \in \{0, \dots, B-1\}$ de tamaño canónico $L$ y cada variable aleatoria $v \in \{0, \dots, K-1\}$ recibe un stream independiente basado en contador `Philox(SeedSequence([MASTER_SEED, b, v]))`. Los eventos `batch` se alimentan exclusivamente del acumulador de tareas físicas completadas, mientras que el acumulador canónico incorpora bloques lógicos cuando su condición de cobertura ($n_b$ muestras) se ha satisfecho y los reduce estrictamente en el orden $b = 0, 1, \dots, B-1$.

### Decisión
Adoptar la **Alternativa 3** (`seed_strategy="philox_logical_block"` por defecto en v1.3.0, conservando `seed_strategy="batch_index"` como modo legado para controles comparativos).

### Evidencia Experimental
En `benchmarks/results/final/v1.3.0/schedule_invariance_summary.csv`, `tests/test_schedule_independent_reproducibility.py` y `tests/test_generator_contract.py`, los 6 modelos evaluados a $1,000,000$ de iteraciones a través de 14 configuraciones distintas (variando workers $\in \{1, 2, 4\}$, tamaños físicos de tarea $\in \{10\text{k}, 20\text{k}, 25\text{k}, 50\text{k}, 100\text{k}\}$, modos de planificación `fixed`, `ast_only`, `pilot_only`, `hybrid` y órdenes de finalización `as_completed`, `reverse`, `shuffled`) produjeron **un único hash SHA-256 canónico por modelo** (`unique_result_hashes = 1`, `84/84` ejecuciones invariantes), manteniendo `histogram.total_count == valid_results` en todos los eventos `batch` intermedios.

---

## ADR-002: Modelo de Coste sobre 4 Índices Estructurales Compuestos Interpretables con Regresión Ridge Restringida Exacta (KKT) y Contracción Ponderada por Tamaño Muestral

### Contexto y Problema
Para decidir cuántos workers emplear y cómo agrupar los bloques lógicos en tareas físicas, el runtime necesita estimar el coste computacional por muestra del modelo DSL. Un modelo lineal que ajuste pesos libres sobre los 11 conteos primitivos del AST con pocos modelos de calibración sufriría subdeterminación estadística ($p > M_{\text{train}}$).

### Alternativas Consideradas
1. **Modelo de aprendizaje automático de caja negra (e.g., Random Forest o MLP)**: Excesivamente complejo para el espacio de características del DSL, difícil de interpretar y propenso a sobreajuste.
2. **Regresión sobre los 11 conteos primitivos del AST**: Subdeterminada cuando se calibra con menos de 11 familias de modelos.
3. **Estimación puramente estática (`ast_only`) o puramente empírica (`pilot_only`)**: `ast_only` no captura variaciones dinámicas del hardware; `pilot_only` puede verse afectado por ruido transitorio del sistema operativo en muestras pequeñas.
4. **Proyección sobre 4 índices estructurales compuestos interpretables (`compact_feature_vector`) ajustados mediante regresión Ridge restringida exacta (KKT) y combinados en ejecución con el bloque piloto mediante contracción ponderada por tamaño muestral (`hybrid`)**:
   $$\psi(\mathcal{T}) = \bigl[1.0,\; I_{\text{dist}}(\mathcal{T}),\; I_{\text{arith}}(\mathcal{T}),\; I_{\text{trans}}(\mathcal{T})\bigr]^\top \in \mathbb{R}_{\ge 0}^4, \quad \hat{c}_{\text{AST}}(\mathcal{T}) = \max\bigl(5\times 10^{-9},\; \hat{\theta}^\top \psi(\mathcal{T})\bigr)$$
   $$\hat{\theta} = \arg\min_{\theta \ge 0.25\theta_{\text{prior}}} \|X\theta - y\|_2^2 + \lambda_{\text{eff}}\left\|\frac{\theta - \theta_{\text{prior}}}{\theta_{\text{prior}}}\right\|_2^2, \quad \lambda_{\text{eff}} = \lambda \cdot \frac{\text{tr}(X_{\text{scaled}}^\top X_{\text{scaled}})}{p}$$
   $$\hat{c}_{\text{hybrid}} = (1 - \alpha)\,\hat{c}_{\text{AST}}(\mathcal{T}) + \alpha\,\hat{c}_{\text{pilot}}, \quad \alpha = \text{clip}\!\left(\frac{L}{L + N_0},\; 0.15,\; 0.88\right)$$
   donde el primer bloque lógico ($b = 0$, $L = 5,000$, $N_0 = 1,500$) actúa simultáneamente como ejecución piloto en proceso y como el primer bloque válido de la simulación (sin desperdiciar muestras), y los coeficientes entrenados $\hat{\theta}$ se propagan a `compile_source()`, `server.py` y el runtime generado.

### Decisión
Adoptar la **Alternativa 4** (`src/montecarlo_dsl/cost_model.py`), calibrando los 4 parámetros mediante enumeración exacta de conjuntos activos KKT sobre $M_{\text{train}} = 8$ modelos de entrenamiento (matriz de diseño $8 \times 4$ de rango $4$, $\kappa_{\text{raw}} = 11.3601$, $\kappa_{\text{std}} = 3.4054$, correlación $r(I_{\text{arith}}, I_{\text{trans}}) = 0.8234$) y validando en $M_{\text{test}} = 6$ modelos no vistos.

### Evidencia Experimental
En `benchmarks/results/final/v1.3.0/ast_cost_generalization.csv` y `ast_cost_generalization_summary.csv`, sobre los 6 modelos de prueba no vistos, el prior estático calibrado (`ast_only`) alcanza **$\text{MAPE} = 12.35\%$** (error mediano `7.36%`, Pearson $r = 0.9323$, Spearman $\rho = 1.0000$). Tras incorporar la contracción ponderada con el bloque piloto ($b = 0$), el estimador `hybrid` reduce el error fuera de muestra a **$\text{MAPE} = 5.55\%$** (error mediano **`5.04%`**, $\text{RMSE} = 42.11\text{ ns}$/muestra, Pearson $r = 0.9899$, Spearman $\rho = 1.0000$).

---

## ADR-003: Planificador Adaptativo Unificado (Ejecución Serial en Proceso vs. `ProcessPoolExecutor`) y Separación de Baselines

### Contexto y Problema
La línea base v1.2.0 lanzaba siempre `ProcessPoolExecutor` con lotes fijos de `25,000` muestras. En equipos de 2 CPUs lógicas o en modelos de bajo/medio coste aritmético, el sobrecoste de creación de procesos e IPC superaba el tiempo de cómputo numérico y duplicaba la memoria residente (RSS) del sistema operativo.

### Decisión
1. Unificar `AdaptiveMonteCarloScheduler` (`src/montecarlo_dsl/scheduler.py`, `src/montecarlo_dsl/compiler.py`, `src/montecarlo_dsl/server.py` y `src/montecarlo_dsl/generator.py`), de forma que el scheduler Python y el runtime generado utilizan el mismo kernel puro `build_runtime_schedule_plan()` (inyectado en el script generado), evitando dos implementaciones divergentes de la politica adaptativa.
2. Selecciona ejecución serial en proceso (`effective_workers = 1`) cuando el sobrecoste de concurrencia supera el beneficio esperado, y activa `ProcessPoolExecutor` solo cuando el trabajo computacional amortiza $\tau_{\text{pool}}$ por un margen $\ge 1.12\times$.
3. Separa la comparacion contra `v1.2.0_legacy_configuration_emulation` de las comparaciones internas entre variantes. La emulacion usa el runtime actual con parametros historicos; no ejecuta el codigo fuente original v1.2.0.

### Evidencia Experimental
Los speedups y cifras RSS documentados en `docs/tables/v1.3.0_*.md` pertenecen a la campaña historica v1.3.0. Deben regenerarse con el kernel compartido v1.3.1 antes de utilizarlos como evidencia de rendimiento actual.

---

## ADR-004: Carga Retenida Acotada con Ventana Fija, Compactación Pareada e Instrumentación RSS (`BoundedStreamingHistogram`)

### Contexto y Problema
En v1.2.0, `generator.py` retenía todos los arreglos de muestras finitas en `state["all_chunks"]`, provocando crecimiento lineal de memoria $O(N)$. Además, en los modelos con operador `\pm`, almacenar una entrada por bloque en `paired_state["correlations"][b]` mantenía un crecimiento $O(B) = O(N/L)$, y medir únicamente los bytes de los contenedores al final de la simulación no distinguía entre la carga útil retenida en el coordinador y el RSS real del proceso y de los trabajadores.

### Decisión
Unificar `BoundedStreamingHistogram` entre `src/montecarlo_dsl/statistics.py` y `src/montecarlo_dsl/generator.py` (mediante `inspect.getsource`) e implementar:
1. **Resumen por centroides equiprobables por bloque lógico**: Para $N > 65,536$, cada bloque lógico $b$ conserva los soportes exactos si tiene $\le 256$ valores únicos (distribuciones discretas) o particiona las muestras ordenadas en $256$ centroides equiprobables con extremos exactos $(\min_b, 1)$ y $(\max_b, 1)$.
2. **Avance de frontera contigua con buffer limitado**: A medida que avanza `next_canonical_block`, se fusionan y eliminan (`pop`) las entradas de `batch_stats`, `_pending_blocks` y `paired_state["correlations"]`. Los buffers de reordenamiento tienen un limite explicito (`MAX_PENDING_LOGICAL_BLOCKS = 64`); el runtime multiproceso limita las tareas en vuelo a `2 * worker_count` e ingiere resultados en orden canonico.
3. **Instrumentacion dual de memoria**: `peak_retained_coordinator_bytes` estima la carga de arrays y buffers observables en los puntos de ingesta. `coordinator_peak_rss_bytes` y `workers_peak_rss_bytes` son maximos por proceso; `sum_of_process_peak_rss_bytes` suma esos maximos y no representa un pico simultaneo agregado. Windows usa `GetProcessMemoryInfo` cuando `resource` no esta disponible.

### Evidencia Experimental
Los CSV bajo `benchmarks/results/final/v1.3.0/` son evidencia historica de la implementacion anterior y no validan los nuevos limites de ventana/buffer de v1.3.1. La campana debe regenerarse. La propiedad de memoria que se pretende demostrar es acotacion respecto a N **manteniendo fijos** el tamano maximo de tarea, workers y ventana; no se afirma independencia de esos parametros. El nuevo experimento calcula la distancia TV de `paired_delta` desde histogramas emitidos, sin asignar un cero artificial.

---

## ADR-005: Optimizador del AST Seguro Bajo Punto Flotante IEEE-754 y Política `discard_and_report`

### Contexto y Problema
El compilador v1.2.0 carecía de una fase explícita de optimización del AST. Sin embargo, aplicar simplificaciones algebraicas agresivas sobre punto flotante puede cambiar el redondeo o enmascarar valores `NaN`/`Inf` que el runtime debe descartar y contabilizar.

### Decisión
Implementar `ASTOptimizer` (`src/montecarlo_dsl/optimizer.py`) como componente conservador:
- **Constant folding float64**: convertir cada literal a float64 y ejecutar la operacion binaria con el mismo orden de operandos que utilizaria el runtime. No plegar si los operandos no son representables, el resultado no es finito o la operacion es indefinida.
- **Transformaciones bit-preserving**: eliminar signo unario positivo y doble negacion.
- **No eliminar identidades dinamicas** como `x + 0`, `x * 1`, `x / 1` o `x ^ 1`; tampoco sustituir `sqrt`/potencias NumPy por operaciones escalares Python. Estas transformaciones pueden alterar ceros con signo, NaN o redondeo.

### Evidencia Experimental
Los resultados del benchmark de optimizacion de v1.3.0 son historicos y no deben atribuirse al optimizador conservador de v1.3.1. La nueva campana debe regenerar AST counts, tiempos y pruebas diferenciales, incluyendo `0.1 + 0.2`, signed zero y operaciones no finitas.

---

## ADR-006: Exclusión de Parada Automática Ingenua Basada en Intervalos de Confianza del 95 %

### Contexto y Problema
Una tentación frecuente en simuladores Monte Carlo adaptativos es detener la ejecución en cuanto el semiancho del intervalo convencional del 95 % ($\bar{Y} \pm 1.96\,\text{MCSE}$) cae por debajo de un umbral.

### Decisión
Siguiendo la precaución estadística metodológica (Jennison & Turnbull, 1999; Howard et al., 2021), **no se introduce parada opcional basada en examinar repetidamente un IC convencional del 95 %** en v1.3.0. Examinar secuencialmente un intervalo de muestra fija infla el error de Tipo I y degrada la probabilidad de cobertura nominal por debajo del 95 %. En su lugar, v1.3.0 mantiene fijo el número total de muestras $N$ especificado en `\iter{N}` y adapta exclusivamente la planificación física de su ejecución.
