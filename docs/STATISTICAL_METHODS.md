# Métodos Estadísticos y Gestión de Memoria Acotada bajo Ventana Fija — Monte Carlo DSL v1.3.1

> Los resultados cuantitativos citados en este documento proceden de la campaña histórica v1.3.0 y deben regenerarse antes de presentarlos como resultados v1.3.1.

## 1. Semántica Probabilística y Ramas Emparejadas (`\pm`)

Un programa DSL define una variable aleatoria de salida:

```text
Y = f(X1, X2, ..., Xk)
```

Cada variable `Xi` se declara con una distribución `\normal`, `\uniform` o `\distrib`. En v1.3.0, cada bloque lógico $b \in \{0, \dots, B-1\}$ de tamaño canónico $L = 5,000$ y cada variable aleatoria $v \in \{0, \dots, k-1\}$ recibe un substream pseudoaleatorio independiente basado en contador mediante `numpy.random.Philox(SeedSequence([MASTER_SEED, b, v]))`.

Cuando la expresión contiene `\pm`, el runtime evalúa dos ramas emparejadas sobre las mismas realizaciones aleatorias $(X_1, \dots, X_k)$:

```text
Y_plus  = f_plus(X)
Y_minus = f_minus(X)
Delta   = Y_plus - Y_minus
```

Por ello la diferencia `Delta` se reporta como estadístico pareado (incluyendo covarianza y correlación de Pearson entre `Y_plus` e `Y_minus`) y no como comparación de muestras independientes.

## 2. Resultados Finitos y Media Condicional (`discard_and_report`)

El runtime no propaga `NaN` ni infinitos a los estimadores principales. La política vigente es `nonfinite_policy = discard_and_report`: los estimadores de media, varianza, desviación estándar y cuantiles se calculan exclusivamente sobre resultados finitos. Si existen descartes, la media reportada es la media condicional sobre resultados válidos, acompañada de:
- `candidate_results`, `valid_results`, `discarded_results`
- `valid_rate`, `discard_rate`
- intervalos de confianza de Wilson del 95 % para las tasas de válidos y descartes (`wilson_interval_95`)

## 3. Estimadores Exactos por Reducción de Bloques Lógicos y Separación Progreso / Canónico

La directiva `\response` acepta:

```text
avg, min, max, var, std, count, valid_rate, discard_rate, p05, p50, p95
```

- `min` y `max` son extremos muestrales exactos observados.
- La varianza muestral se calcula con $n - 1$ grados de libertad mediante el algoritmo de Welford dentro de cada bloque lógico y la fórmula de reducción paralela de Chan–Golub–LeVeque entre bloques lógicos en orden estricto $b = 0, 1, \dots, B-1$.
- En modelos pareados `\pm`, las correlaciones por bloque se acumulan en orden canónico mediante suma compensada de Neumaier (1974) (`_advance_paired_frontier`). El buffer de reordenamiento tiene un limite explicito de 64 entradas.
- Cuando una tarea física es menor que un bloque lógico ($B_{\text{phys}} < L$), el coordinador mantiene dos acumuladores separados: un **acumulador de progreso físico** (`progress_sketch`, `stats_state`) que garantiza que cada evento `batch` cumpla exactamente `histogram.total_count == valid_results`, y un **acumulador canónico de bloques lógicos** (`canonical_sketch`, `canonical_stats`) que incorpora cada bloque lógico completo una única vez para garantizar invariancia SHA-256 al final de la simulación.
- El error estándar Monte Carlo de la media es $\text{MCSE}(\bar{Y}) = s / \sqrt{n_{\text{valid}}}$, y el intervalo de confianza al 95 % es $\bar{Y} \pm 1.96\,\text{MCSE}(\bar{Y})$.

## 4. Memoria Retenida Acotada bajo Parametros Fijos, RSS y Cuantiles Ponderados (`BoundedStreamingHistogram`)

En `memory_mode="bounded"` (modo predeterminado):
1. **Resumen por bloque lógico con preservación de átomos discretos y centroides de igual masa**: Para simulaciones con $N > 65,536$, cada bloque lógico $b$ resume sus muestras válidas en un boceto ordenado de a lo sumo $K_{\text{block}} = 256$ puntos y pesos enteros exactos (`summarize_block_samples`). Si el soporte del bloque tiene $\le 256$ valores únicos (distribuciones discretas), se conservan exactamente todos los átomos; en caso contrario, las muestras ordenadas se particionan en $K_{\text{block}}$ cubetas de igual masa muestral conservando extremos exactos (`min`, `max`) y $\sum w = n_{\text{valid}, b}$, descartando de inmediato el arreglo de muestras crudas.
2. **Compactación y backpressure**: El coordinador mantiene una frontera canonica y fusiona estados de Welford, resumenes y correlaciones. En v1.3.1, el buffer de bloques pendientes se limita a 64 y la ventana de tareas en vuelo a `2 * worker_count`; los resultados se ingieren en orden canonico. La acotacion respecto a N requiere mantener fijos el tamano maximo de tarea, workers y ventana.
3. **Cuantiles ponderados y validacion**: Los cuantiles `p05`, `p50` y `p95` se calculan usando pesos del sketch. Son aproximaciones, no cuantiles exactos. Los numeros de error de v1.3.0 son historicos; v1.3.1 debe generar `quantile_and_histogram_fidelity.csv` para ramas simples y `paired_delta`.

La carga retenida estimada (`peak_retained_coordinator_bytes`) no es RSS. `sum_of_process_peak_rss_bytes` suma maximos historicos por proceso y no representa un pico simultaneo del arbol de procesos.
