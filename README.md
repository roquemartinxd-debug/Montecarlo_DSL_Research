# Monte Carlo DSL v1.3.1

Compilador DSL para simulaciones Monte Carlo con validacion semantica, generacion NumPy, planificacion adaptativa guiada por el AST y calibracion piloto, reproducibilidad independiente de la planificacion fisica (`Philox`), estadistica incremental, cuantiles aproximados mediante sketch ponderado y backend FastAPI/WebSocket.

La version **1.3.1** es una entrega de estabilizacion. Corrige el buffer multiproceso, limita buffers de reordenamiento, unifica el kernel del planificador, hace portable la medicion RSS, incorpora histogramas pareados y restringe el optimizador para respetar el redondeo float64. **No debe considerarse aun una version de envio a revista**: la campana experimental debe regenerarse despues de estos cambios.

## Estado de la versión

- Version: **1.3.1** (Licencia MIT en [`LICENSE`](LICENSE), flujo CI en [`.github/workflows/ci.yml`](.github/workflows/ci.yml))
- Pipeline de compilación: `LEX -> SYN -> AST -> SEM -> OPT (ASTOptimizer) -> COST (ASTCostFeatures) -> GEN` antes de crear el trabajo runtime
- Optimizador del compilador (`optimizer.py`): *constant folding* en float64 para literales; no elimina identidades dinamicas como `x + 0` o `x * 1`, ni sustituye `sqrt`/potencias NumPy por aritmetica escalar Python
- Modelo de coste (`cost_model.py`): proyección sobre **4 índices estructurales compuestos interpretables** (`intercept`, `distribution_cost_index`, `arithmetic_cost_index`, `transcendental_cost_index`; rango $= 4$, $\kappa_{\text{raw}} = 11.3601$, $\kappa_{\text{std}} = 3.4054$, $r_{\text{arith,trans}} = 0.8234$), calibrada mediante regresión Ridge restringida exacta (KKT) sobre 8 modelos de entrenamiento ($\text{MAPE}_{\text{AST}} = 12.35\%$ fuera de muestra, $\rho = 1.0000$), propagada al runtime y combinada con el bloque piloto $b = 0$ mediante contracción ponderada por tamaño muestral (*sample-size-weighted shrinkage*), logrando un **$\text{MAPE}_{\text{híbrido}} = 5.55\%$** (error mediano **`5.04%`**, $r = 0.9899$, $\rho = 1.0000$) fuera de muestra en 6 modelos no vistos
- Planificador adaptativo (`scheduler.py`, `compiler.py`, `generator.py`): el mismo kernel puro se utiliza en el scheduler Python y en el runtime generado; los descriptores de tareas se generan perezosamente y el reordenamiento/futures se limita a ventanas de `2 * worker_count`, con ingesta final en orden canonico
- Reproducibilidad: los bloques logicos determinan las secuencias `Philox`; las tareas fisicas se pueden enviar en distinto orden y sus resultados se reducen en orden canonico
- Memoria: bajo tamano maximo de tarea, workers y ventana fijos, el modo `bounded` limita el buffer de reordenamiento y la carga retenida estimada; `peak_retained_coordinator_bytes` no es RSS. `sum_of_process_peak_rss_bytes` es suma de maximos por proceso, no pico simultaneo
- Validacion local de esta correccion: 242/242 pruebas superadas en tres grupos con Python 3.13.5 y NumPy 2.3.5; la ejecucion monolitica excedio el limite del entorno y el runner Windows sigue pendiente

## Inicio rápido

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp
.\.venv\Scripts\python.exe -m uvicorn montecarlo_dsl.server:app --host 127.0.0.1 --port 8000
```

Abrir:

- Aplicación: `http://127.0.0.1:8000/`
- Salud: `http://127.0.0.1:8000/api/health`
- OpenAPI/Swagger: `http://127.0.0.1:8000/docs`

## Campana experimental v1.3.1 y generacion de tablas/figuras

```bash
python benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10
python scripts/generate_v1_3_figures_tables.py
```

Los comandos regeneran resultados en `benchmarks/results/final/v1.3.1/`. Los resultados anteriores se conservan por separado:

- Baseline historico v1.2.0: `benchmarks/results/final/v1.2.0/` (la comparacion en Exp. 0 es una emulacion de configuracion, no ejecucion del codigo original)
- Resultados historicos v1.3.0: `benchmarks/results/final/v1.3.0/` (no reutilizar como resultados v1.3.1)
- Resultados nuevos v1.3.1: `benchmarks/results/final/v1.3.1/` despues de ejecutar la campana
- Tablas y figuras v1.3.1: generadas desde los CSV v1.3.1

## Documentación clave

Empiece por [`docs/00_INDICE.md`](docs/00_INDICE.md). Para la defensa VAE y el artículo científico destacan:

- [`docs/paper/v1.3.1_manuscript.md`](docs/paper/v1.3.1_manuscript.md) — Borrador actualizado; cifras heredadas aun requieren revalidacion
- [`docs/VAE_RNCP39278_ALIGNMENT.md`](docs/VAE_RNCP39278_ALIGNMENT.md) — Alineación rigurosa con los 7 bloques oficiales RNCP39278 (`BC01` a `BC07`)
- [`docs/ADR_ARCHITECTURE_DECISIONS.md`](docs/ADR_ARCHITECTURE_DECISIONS.md) — Registros de decisiones arquitectónicas (ADR-001 a ADR-006)
- [`docs/SCOPUS_PUBLICATION_STRATEGY.md`](docs/SCOPUS_PUBLICATION_STRATEGY.md) — Estrategia de publicación científica en Scopus y paquete de replicación
- [`docs/baseline_v1.2.0_audit.md`](docs/baseline_v1.2.0_audit.md) — Auditoría y congelación de la línea base v1.2.0
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — Garantías formales de reproducibilidad independiente de la planificación
- [`docs/STATISTICAL_METHODS.md`](docs/STATISTICAL_METHODS.md) — Métodos estadísticos, memoria acotada bajo parametros fijos y cuantiles ponderados
- [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) — Protocolo corregido y numero real de repeticiones por experimento
- [`docs/REVALIDATION_REQUIRED_v1.3.1.md`](docs/REVALIDATION_REQUIRED_v1.3.1.md) — Lista de verificaciones pendientes antes de publicar
- [`docs/VALIDATION_SUMMARY_v1.3.1_lazy_tasks.md`](docs/VALIDATION_SUMMARY_v1.3.1_lazy_tasks.md) — Resumen de los 242 tests y la correccion de generacion perezosa

## Campana experimental reanudable (v1.3.1 hotfix)

El runner experimental guarda checkpoints por simulacion, permite continuar
despues de un reinicio con `--resume` y detecta cambios de codigo, maquina,
dependencias o numero de repeticiones. Los calentamientos se repiten.

```powershell
python -u benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10
# Si se interrumpe:
python -u benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10 --resume
```

Consulta `docs/CHECKPOINT_RESUME_v1.3.1.md` para el procedimiento y las
limitaciones de comparaciones de rendimiento entre sesiones.

## Campaña complementaria AWS (Ubuntu 24.04, perfil de 4 núcleos)

Para una segunda campaña aislada en EC2, consulta [`docs/AWS_CAMPAIGN_v1.3.1_UBUNTU_4CORE.md`](docs/AWS_CAMPAIGN_v1.3.1_UBUNTU_4CORE.md). El perfil `aws-4core` añade un baseline serial en proceso a Exp. 2 y una matriz de memoria con cuatro workers a Exp. 8; `--output-dir` y `--checkpoint-dir` separan completamente los resultados AWS de la campaña Windows. Las dependencias de comparación están fijadas en `requirements-aws-experiment.txt`.

Paquete de código limpio para transferir desde Windows a EC2:

```powershell
python scripts/package_aws_source.py --output .\dist\Montecarlo_DSL_Research_v1.3.1_aws_source.zip
```
