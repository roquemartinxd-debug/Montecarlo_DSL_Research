# Changelog

## 1.3.1 - 2026-10-31

- Hotfix FastAPI en Windows: eliminado `app.add_event_handler("shutdown", ...)` porque `lifespan` ya gestiona `JobManager.shutdown()`; anadida prueba de regresion y configurado el loop scope explicito de `pytest-asyncio`.
- Correccion del contrato automatizado de version: `statistical_core_version` espera `1.3.1`.
- Generacion perezosa de tareas fisicas para planes fijos y adaptativos; `ExecutionSchedulePlan` conserva parametros constantes y un contador de tareas O(1), evitando materializar descriptores O(T).
- El runtime consume tareas bajo demanda y limita el reordenamiento de envio, los futures y los resultados pendientes a ventanas de `2 * worker_count`; la reduccion canonica se conserva.
- El kernel compartido incorpora iterador de tareas y contador analitico; se agregan pruebas de equivalencia del contador, planes de hasta 100 millones de tareas y ejecucion de modos `fixed`/`ast_only`.
- Validacion: 242/242 pruebas superadas en tres grupos independientes con Python 3.13.5 y NumPy 2.3.5. La ejecucion monolitica excedio el limite de tiempo del entorno; CI Windows y revalidacion en el entorno objetivo siguen pendientes.

## 1.3.1 — Perfil experimental AWS (complementario)

- Añadidos `--output-dir`, `--profile aws-4core`, `--campaign-label` y `--instance-type`; la identidad del checkpoint incluye perfil, tipo de instancia y ruta de resultados para impedir reanudar una campaña en otra máquina/configuración.
- Exp. 2 incorpora `v1.3_serial_in_process` (`ast_only`, un worker, sin `ProcessPoolExecutor`) y reporta speedups emparejados frente a ese baseline, además de conservar las comparaciones históricas frente a pools fijos de 1 y 2 workers.
- La prueba de escalabilidad a 10 millones de iteraciones incluye baseline serial, pools fijos de 1/2/4 workers y planificador híbrido; se corrige el cálculo de speedup de 10M para usar referencias distintas y emparejadas.
- El perfil `aws-4core` amplía Exp. 8 con mediciones `multiprocess_w4` para `normal_mean` y `quadratic_pm` a 100k, 1M y 2.5M, en modos `retain_all` y `bounded`.
- `environment.json` registra perfil, etiqueta, tipo de instancia, región y topología física/lógica detectada mediante `lscpu` o `/proc/cpuinfo`.
- El generador de tablas/figuras acepta rutas independientes para resultados, tablas y figuras AWS; se añade `requirements-aws-experiment.txt` fijado a las versiones de la campaña Windows y un empaquetador de código limpio.
- Añadida documentación de despliegue EC2 Ubuntu 24.04, transferencia SSH, ejecución tmux, checkpoints y recuperación de resultados.

## 1.3.0 - 2026-10-31

- Coherencia completa de entorno verificada entre `pyproject.toml` y `benchmarks/results/final/v1.3.0/environment.json` (`Python 3.11.2`, `NumPy 2.1.3`, `FastAPI 0.92.0`, `Pydantic 1.10.4`, `Uvicorn 0.17.6`, `SciPy 1.15.2`, `HTTPX 0.23.3`, `Pytest 9.1.1`), licencia `LICENSE` (MIT) y flujo CI `.github/workflows/ci.yml`.
- Auditoria y congelacion de la linea base v1.2.0 en `docs/baseline_v1.2.0_audit.md` y comparacion directa de 30 repeticiones en `v1_2_vs_v1_3_baseline_comparison.csv` (`docs/tables/v1.3.0_vs_v1.2.0_baseline.md`), alcanzando aceleraciones de `2.72x`–`5.70x` a $N=100,000$ y de `2.41x`–`5.17x` a $N=1,000,000$, con reduccion de carga util retenida en el coordinador del `95.5%`–`97.4%` y reduccion del RSS pico agregado del sistema operativo del `48.1%`–`51.8%`.
- Desacoplamiento entre bloque logico (`logical_block_size = 5,000`) y tarea fisica mediante generador basado en contador `numpy.random.Philox` indexado por `(MASTER_SEED, logical_block_index, var_index)`, separacion entre acumulador de progreso de tareas fisicas y acumulador canonico de bloques logicos, y reduccion determinista por frontera contigua (Componente C).
- Modelo `InterpretableASTCostModel` en `montecarlo_dsl.cost_model` proyectado sobre 4 indices estructurales compuestos ($\text{rank}(X)=4$, $\kappa_{\text{raw}}=11.3601$, $\kappa_{\text{std}}=3.4054$) con solver exacto de conjunto activo KKT para regresion Ridge restringida, calibrado con 8 modelos de entrenamiento, validado fuera de muestra en 6 modelos no vistos ($\text{MAPE} = 5.55\%$ tras calibracion piloto por contraccion ponderada por tamano muestral, mediana $5.04\%$, $r = 0.9899$, $\rho = 1.0000$) e integrado directamente en `compile_source()`, `server.py` y el runtime generado (Componente A).
- Planificador `AdaptiveMonteCarloScheduler` unificado entre `scheduler.py`, `compiler.py` y `generator.py` con soporte para los modos `fixed`, `ast_only`, `pilot_only` e `hybrid` (Componente B).
- Optimizador `ASTOptimizer` en `montecarlo_dsl.optimizer` con constant folding, simplificaciones algebraicas seguras en IEEE-754 y pruebas diferenciales (`1.119x` speedup en proceso y `1.57x`–`1.59x` combinado con planificacion adaptativa en `optimizable_heavy`).
- Compactacion en memoria de carga util del coordinador asintoticamente acotada $O(1)$ (`BoundedStreamingHistogram` unificado entre `statistics.py` y `generator.py` con `MAX_GLOBAL_SKETCH_POINTS = 2048` y compactacion Neumaier de correlaciones pareadas en `_advance_paired_frontier`) e instrumentacion real de RSS del sistema operativo (`resource.getrusage`), reduciendo la carga util retenida en `98.1%`–`99.0%` en proceso (`0.207 MB` en `normal_mean`, `0.583 MB` en `quadratic_pm` a $N=2.5 \times 10^6$) y en `97.8%`–`98.8%` en multiproceso `w=2`, con error relativo maximo en cuantiles $\le 0.174\%$ en 6 familias distribucionales y 10 ramas (`0.000%` en `discrete_mean`).
- Campana experimental v1.3.0 completa (30 repeticiones confirmatorias, 10 piloto, 10 de ablacion $4 \times 2$, 25 semillas estadisticas, 17 CSVs, 9 tablas Markdown y 4 figuras PNG) en `benchmarks/run_v1_3_experiments.py` y `benchmarks/results/final/v1.3.0/`.
- Documentacion arquitectonica (`docs/ADR_ARCHITECTURE_DECISIONS.md`), alineacion honesta con los 7 bloques oficiales VAE RNCP39278 (`BC01` a `BC07` en `docs/VAE_RNCP39278_ALIGNMENT.md`), estrategia Scopus (`docs/SCOPUS_PUBLICATION_STRATEGY.md`) y manuscrito cientifico completo (`docs/paper/v1.3.0_manuscript.md` con 15 referencias verificadas con DOI).
- Suite ampliada a 229 pruebas aprobadas.

## 1.2.0 - 2026-09-14

- Nuevo modulo `montecarlo_dsl.statistics` con acumulacion Welford, combinacion Chan-Golub-LeVeque e intervalo Wilson.
- `\response` acepta `var`, `std`, `count`, `valid_rate`, `discard_rate`, `p05`, `p50` y `p95`.
- El runtime reporta varianza muestral, MCSE, IC 95 % de la media, tasas de validos/descartes e intervalos Wilson.
- `min` y `max` se documentan como extremos muestrales.
- El operador `\pm` conserva ramas independientes y agrega estadistica pareada `plus - minus`.
- Eventos `start` y `complete` incluyen metadatos de reproducibilidad.
- Se agregan modelos analiticos y scripts de benchmark en `benchmarks/`.
- Nueva documentacion metodologica para version publicable.

## 1.1.0 - 2026-08-31

- Identificadores ampliados a `[A-Za-z][A-Za-z0-9_]*`.
- Separadores explicitos en los pares de `\distrib`.
- Unificacion de numeros con signo en la BNF.
- Separacion formal entre compilacion y ejecucion mediante `compiler.py`.
- Protocolo runtime v2 con prefijo `MC_DSL_EVENT:`.
- Aislamiento de cualquier salida libre de `stdout`.
- Ramas `\pm` separadas estadisticamente de extremo a extremo.
- MCSE e intervalo del 95 % por rama.
- Frontend con selector de rama.
- Suite: 205 pruebas aprobadas.

## 1.0.0 - 2026-08-24

- Primera version estable del compilador, runtime concurrente, backend y cliente web.
