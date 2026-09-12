# Monte Carlo DSL v1.1.0

Version endurecida del compilador y simulador Monte Carlo DSL.

El proyecto incluye compilacion completa del DSL, analisis semantico, generacion de un
runtime NumPy concurrente, backend FastAPI, streaming por WebSocket e interfaz web con
histogramas actualizados por lotes.

## Estado de la version

- Version: **1.1.0 estable**
- Suite automatizada final: **205 pruebas aprobadas**
- Identificadores: `[A-Za-z][A-Za-z0-9_]*`
- Arquitectura: cliente/servidor local
- Compilacion: LEX -> SYN -> AST -> SEM -> GEN antes de crear el trabajo runtime
- Backend: FastAPI + JobManager
- Runtime: script Python/NumPy generado desde un AST validado
- Paralelismo: ProcessPoolExecutor
- Streaming: WebSocket
- Protocolo subproceso/backend: eventos JSON prefijados con `MC_DSL_EVENT:`
- Operador `\pm`: ramas `plus` y `minus` conservadas de forma independiente
- Diagnosticos numericos: desviacion muestral, MCSE e intervalo de confianza del 95 % para la media
- Frontend: HTML/CSS/JavaScript + D3 local
- Persistencia: ninguna; no usa base de datos
- Simulaciones activas: una a la vez

## Inicio rapido

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
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
\response{avg,min,max}
```

## Cambios principales de v1.1.0

- Se unifico la produccion de numeros con signo en `<NUMERO_SIGNADO>`.
- `\distrib` utiliza separadores externos explicitos entre pares.
- Los identificadores pueden tener varias posiciones, digitos y guion bajo.
- El generador mapea los nombres del DSL a identificadores Python internos seguros.
- La compilacion termina completamente antes de crear el trabajo de ejecucion.
- `stdout` queda desacoplado del protocolo mediante un prefijo reservado para eventos.
- Las ramas producidas por `\pm` ya no se mezclan para estadisticos o histogramas.
- Cada rama incorpora MCSE e intervalo del 95 % para la media.

## Documentacion

Empiece por [`docs/00_INDICE.md`](docs/00_INDICE.md).
