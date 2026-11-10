# Threats to Validity — MC DSL v1.3.1

> **Estado de revision:** este documento describe los limites de la implementacion v1.3.1. Los numeros de `benchmarks/results/final/v1.3.0/` son historicos y no constituyen una validacion experimental de v1.3.1. Consultar `docs/REVALIDATION_REQUIRED_v1.3.1.md` antes de usar el manuscrito para un envio.

## Parte I — Amenazas a la validez en v1.3.1

### 1. Validez interna

1. **Variabilidad de ejecucion:** el sistema operativo, la frecuencia de CPU, la recoleccion de basura, la creacion de procesos y la carga concurrente afectan las mediciones. Las comparaciones deben reportar mediana, IQR, desviacion y los intervalos bootstrap realmente calculados.
2. **Repeticiones heterogeneas:** el script no ejecuta 30 repeticiones en todas las celdas. Exp. 0 usa 30; Exp. 1 y Exp. 4 usan una ejecucion por configuracion; Exp. 2 usa 30 solo para `normal_mean` y `heavy_expression` a 100k y 1M, y 10 en las demas celdas; Exp. 6 y Exp. 9 usan 10; Exp. 7 usa 25 semillas; Exp. 8 usa una ejecucion por configuracion. Los CSV son la fuente de verdad para cada tamano muestral.
3. **Planificacion multiproceso:** v1.3.1 limita las tareas en vuelo a `2 * worker_count` y reduce los resultados en orden canonico. El orden de envio `reverse`/`shuffled` perturba el buffer de reordenamiento; el orden de finalizacion real depende del sistema operativo y no debe describirse como forzado salvo que el experimento mida esa secuencia.
4. **Coste del piloto:** el bloque piloto puede sufrir ruido transitorio. La politica `pilot_only` es especialmente sensible; la variante hibrida contrae la medicion hacia el prior estructural. Deben reportarse ambos componentes y no solo el error final.

### 2. Validez de constructo

1. **Carga retenida y RSS son metricas diferentes:** `peak_retained_coordinator_bytes` es una estimacion de payload de arrays y buffers observables en puntos de ingesta; no es un recuento completo del heap ni el RSS del proceso.
2. **Semantica RSS:** `coordinator_peak_rss_bytes` y `workers_peak_rss_bytes` son maximos historicos por proceso. `sum_of_process_peak_rss_bytes` es la suma de esos maximos, no un pico simultaneo agregado. Para afirmar pico simultaneo del arbol de procesos se necesita muestreo temporal conjunto o cgroup.
3. **Memoria acotada condicionada:** los buffers estan limitados (`MAX_PENDING_LOGICAL_BLOCKS = 64`, ventana de tareas en vuelo `2 * worker_count`). La afirmacion de acotacion respecto a N requiere mantener fijos el tamano maximo de tarea, el numero de workers y la ventana. No se afirma una cota independiente de esos parametros.
4. **Cuantiles e histogramas aproximados:** el sketch ponderado produce aproximaciones deterministas de cuantiles y frecuencias. La media, varianza, extremos y conteos se deben evaluar por separado. La distancia TV de `paired_delta` se calcula desde los histogramas reales; si faltan datos, debe marcarse como no disponible.
5. **Correlacion pareada:** `correlation_plus_minus_mean` es una media de correlaciones por bloque logico reducida en orden canonico; no es el coeficiente de Pearson global sobre todas las muestras.
6. **Optimizador:** el optimizador pliega constantes en float64 y conserva operaciones dinamicas como `x + 0` y `x * 1`; tampoco reemplaza `sqrt` o potencias NumPy por operaciones escalares. La equivalencia debe comprobarse con pruebas diferenciales y casos de signed zero, redondeo y valores no finitos.
7. **Modelo de coste:** los cuatro indices estructurales son interpretables, pero no ortogonales. En el conjunto historico se observo correlacion entre indices; el error fuera de muestra debe volver a medirse despues del cambio de optimizador.

### 3. Validez externa

1. **Entorno historico no conforme:** `benchmarks/results/final/v1.3.0/environment.json` registra NumPy 1.24.2, mientras que el `pyproject.toml` de esa entrega declaraba `numpy>=2.0`. El registro historico se marco como no conforme. Los resultados nuevos deben provenir de un entorno auditado por `write_environment_metadata()`.
2. **Plataforma:** el soporte Windows usa `GetProcessMemoryInfo` cuando `resource` no esta disponible, pero esta ruta debe ejecutarse realmente en `windows-latest`; la inspeccion estatica no equivale a una prueba de plataforma.
3. **Baseline:** `v1.2.0_legacy_configuration_emulation` ejecuta el runtime actual con parametros historicos. No es una ejecucion del codigo fuente original v1.2.0. Una comparacion con el codigo historico requiere recuperar el tag/commit original.
4. **Alcance del DSL:** MC DSL modela simulaciones Monte Carlo i.i.d. estaticas y ramas pareadas `+/-`; MCMC, SDE dependientes del tiempo y mallas espaciales quedan fuera del alcance actual.

### 4. Validez de conclusion estadistica

Los intervalos bootstrap de 95 % usan 10.000 remuestreos en `bootstrap_ci95()`. La interpretacion debe respetar el numero de repeticiones de cada experimento y no extrapolar la precision de celdas con 30 corridas a celdas con 1 o 10. La validacion de cobertura analitica de Exp. 7 usa 25 semillas independientes; no sustituye las repeticiones de rendimiento.

## Parte II — Alcance historico de la linea base v1.2.0

La evaluacion historica v1.2.0 cubrio modelos y tamanos de muestra especificos. Su estrategia `SeedSequence([MASTER_SEED, batch_index])` acoplaba los substreams a la particion fisica. La reproducibilidad de v1.3.x se apoya en bloques logicos y direccionamiento `Philox`; esa propiedad debe verificarse en cada version del compilador y entorno numerico declarado.
