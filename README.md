# Monte Carlo DSL v1.2.0

Compilador DSL para simulaciones Monte Carlo con runtime NumPy concurrente, backend
FastAPI/WebSocket e interfaz web local.

La version 1.2.0 esta orientada a preparar una version publicable: agrega nucleo
estadistico estable, respuesta estadistica extendida, metadatos de reproducibilidad,
validacion analitica y scripts de benchmark.

## Estado de la version

- Version: **1.2.0**
- Compilacion: LEX -> SYN -> AST -> SEM -> GEN antes de crear el trabajo runtime
- Backend: FastAPI + JobManager
- Runtime: script Python/NumPy generado desde un AST validado
- Paralelismo: ProcessPoolExecutor
- Streaming: WebSocket con eventos JSON prefijados con `MC_DSL_EVENT:`
- Operador `\pm`: ramas `plus` y `minus` separadas y bloque adicional `paired`
- Nucleo estadistico: media, varianza muestral, desviacion estandar, MCSE, IC 95 %, cuantiles y tasas de descarte
- Reproducibilidad: semilla maestra, estrategia por lote con NumPy `SeedSequence` y metadatos de entorno
- Persistencia productiva: ninguna; los benchmarks escriben artefactos CSV/JSONL locales

## Inicio rapido

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp
.\.venv\Scripts\python.exe -m uvicorn montecarlo_dsl.server:app --host 127.0.0.1 --port 8000
```

Abrir:

- Aplicacion: `http://127.0.0.1:8000/`
- Salud: `http://127.0.0.1:8000/api/health`
- OpenAPI/Swagger: `http://127.0.0.1:8000/docs`

El proyecto declara Python `>=3.10`.

## Ejemplo de modelo

```text
\model{resultado=\frac{-coef_b\pm\sqrt{coef_b^2-4coef_a coef_c}}{2coef_a}}
\normal coef_a{10,0.5}
\distrib coef_b{{10,0.5},{15,0.3},{20,0.2}}
\uniform coef_c{-20,30}
\iter{1000000}
\response{avg,std,count,valid_rate,discard_rate,p05,p50,p95}
```

## Estadisticos permitidos

`\response` acepta:

```text
avg, min, max, var, std, count, valid_rate, discard_rate, p05, p50, p95
```

Notas metodologicas:

- `min` y `max` son extremos muestrales observados, no limites teoricos.
- Si hay `NaN` o infinitos, se descartan de los estimadores y se reportan en `discard_rate`.
- La media se interpreta como condicional sobre resultados finitos cuando existen descartes.
- `ci95_mean` es una aproximacion normal de la media: `mean +/- 1.96 * MCSE`.

## Benchmarks y validacion

```powershell
python benchmarks/statistical_validation.py
python benchmarks/run_benchmarks.py --quick
python benchmarks/analyze_results.py
```

Resultados:

- `benchmarks/results/raw/`
- `benchmarks/results/processed/`

## Documentacion

Empiece por [`docs/00_INDICE.md`](docs/00_INDICE.md). Para la version publicable son especialmente importantes:

- [`docs/STATISTICAL_METHODS.md`](docs/STATISTICAL_METHODS.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md)
