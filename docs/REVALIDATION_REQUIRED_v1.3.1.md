# MC DSL v1.3.1 — checklist de estabilizacion y revalidacion

> Esta version incorpora correcciones de implementacion y pruebas. Los resultados de `benchmarks/results/final/v1.3.0/` siguen siendo historicos y no se han regenerado automaticamente. No se deben atribuir a v1.3.1.

## Correcciones incorporadas en el codigo

- [x] Generacion perezosa de tareas fisicas en los modos fijo y adaptativo: el plan almacena parametros de generacion y `task_count`, no una lista completa de descriptores.
- [x] Contador de tareas fisicas calculado en O(1) y verificado contra el iterador en multiples combinaciones de bloques, workers y umbrales de cola.
- [x] Buffer multiproceso de resultados limitado a `2 * worker_count` tareas en vuelo; el reordenamiento de envio solo materializa ventanas acotadas.
- [x] Resultados multiproceso ingeridos en orden canonico, aunque se perturbe el orden de envio de tareas.
- [x] Buffers pendientes de bloques logicos y del sketch limitados por `MAX_PENDING_LOGICAL_BLOCKS` / `max_pending_blocks`.
- [x] La carga retenida estimada incluye arrays de resultados actuales y resultados completados pendientes en el buffer observable.
- [x] El runtime usa una unica funcion pura de planificacion compartida con `AdaptiveMonteCarloScheduler`.
- [x] La importacion de `resource` es condicional; Windows usa `GetProcessMemoryInfo` como proveedor de peak working set.
- [x] RSS renombrado semanticamente como suma de maximos historicos por proceso; el alias antiguo se conserva solo por compatibilidad.
- [x] La rama `paired_delta` emite histograma; el experimento calcula TV a partir de los histogramas reales.
- [x] El optimizador pliega literales en float64 y evita identidades dinamicas y plegado escalar de `sqrt`/potencias.
- [x] Auditoria de dependencias en `write_environment_metadata()`; el registro historico v1.3.0 se marca no conforme por NumPy 1.24.2 frente a `numpy>=2.0`.
- [x] Baseline renombrado como `v1.2.0_legacy_configuration_emulation` para no insinuar que se ejecuta el codigo fuente original.

## Hotfix de importacion de workers con spawn

El runner experimental importa `mcdsl_gen_*.py` dinamicamente. Bajo Windows (`spawn`) los workers deben poder reimportar ese modulo por nombre para recibir trabajos de `ProcessPoolExecutor`. El runner ahora incorpora la carpeta temporal del modulo a `sys.path` durante `module.main()` y la retira al terminar, incluso cuando ocurre un error.

La prueba `tests/test_experiment_spawn.py` fuerza el metodo `spawn` en un proceso nuevo, ejecuta dos simulaciones con workers y verifica el contrato de limpieza. El PASS de esta prueba en Linux prueba la ruta `spawn`, pero no sustituye la comprobacion de la campana en Windows.

## Revalidaciones pendientes antes de publicacion

- [x] Corregir la asercion de `statistical_core_version` para esperar `1.3.1`.
- [x] Ejecutar los 242 tests en tres grupos independientes con Python 3.13.5 y NumPy 2.3.5; todos pasan. La ejecucion monolitica excedio el limite de tiempo del entorno de revision.
- [ ] Ejecutar la suite completa (incluida `test_experiment_spawn.py`) en el entorno objetivo de PowerShell (Python 3.11 y NumPy >=2.0) y guardar el log.
- [ ] Confirmar CI real en `ubuntu-latest` y `windows-latest`; la ruta Windows usa API Win32 y requiere ejecucion efectiva en runner Windows.
- [ ] Repetir las pruebas de memoria para `normal_mean` y `quadratic_pm` con N = 100k, 1M, 2.5M, 5M y 10M; incluir `forward`, `reverse` y `shuffled`, workers 1/2/4, y registrar `max_in_flight_tasks`.
- [ ] Verificar en el equipo objetivo que el crecimiento del buffer retenido permanece acotado al aumentar N manteniendo constantes L, tamano maximo de tarea, workers y ventana; incluir RSS del proceso/worker.
- [ ] Repetir RQ2 (entrenamiento/prueba del modelo de coste) porque el optimizador produce ahora AST/features diferentes.
- [ ] Repetir RQ3 y ablacion con el kernel de planificacion compartido.
- [ ] Repetir las pruebas diferenciales del optimizador, incluyendo `0.1 + 0.2`, `0 + x`, `x * 1`, division por cero, desbordamiento y valores no finitos.
- [ ] Generar `quantile_and_histogram_fidelity.csv` y verificar la TV de todas las ramas, incluida `paired_delta`.
- [ ] Comparar el `sum_of_process_peak_rss_bytes` con un muestreo simultaneo del arbol de procesos si el articulo necesita afirmar un pico RSS agregado real.
- [ ] Actualizar todas las tablas, figuras, README y manuscrito desde los CSV v1.3.1 regenerados.
- [ ] Verificar cada DOI/referencia bibliografica contra una fuente primaria.

## Generacion perezosa de tareas

`ExecutionSchedulePlan.tasks` es ahora una secuencia re-iterable perezosa. `build_runtime_schedule_plan()` devuelve un iterador de un solo uso, `task_generation` y `task_count`; el contador se calcula mediante aritmetica entera sin recorrer las tareas. El runtime generado crea tareas bajo demanda y conserva como maximo una ventana de envio de `2 * worker_count`, junto con futures y resultados completados dentro de la misma cota. `task_tuples()` sigue existiendo por compatibilidad, pero materializa una lista solo cuando un consumidor lo solicita expresamente; no se usa en el camino de compilacion ni de ejecucion.

## Baseline historico

El ZIP recibido no contiene un repositorio Git utilizable para recuperar el arbol de codigo v1.2.0 (el directorio `.git` esta incompleto). Por tanto, el experimento actual no puede ejecutar la implementacion original v1.2.0. Si se necesita una comparacion historica real, se debe recuperar el tag/commit original desde el repositorio fuente. Hasta entonces, usar exclusivamente el nombre `v1.2.0_legacy_configuration_emulation`.

## Comandos

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10
python scripts/generate_v1_3_figures_tables.py
```
