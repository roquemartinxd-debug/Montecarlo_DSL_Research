# Arquitectura cliente-servidor - v1.1.0

## Flujo completo

```text
Usuario
  |
  v
Cliente HTML/CSS/JavaScript
  |
  | POST /api/jobs
  v
FastAPI
  |
  | compile_source(): LEX -> SYN -> SEM -> GEN
  |   error -> HTTP 422 (sin crear trabajo)
  v
CompiledSimulation
  |
  | JobManager.create_job()
  v
Subproceso Python generado
  |
  | ProcessPoolExecutor
  v
Workers / lotes
  |
  | stdout reservado: MC_DSL_EVENT:{JSON}
  v
JobManager / EventHub
  |
  | WebSocket /ws/jobs/{id}
  v
Cliente: progreso + ramas + estadisticos + histograma
```

## Separacion entre compilacion y ejecucion

El backend completa todas las fases del compilador antes de crear el objeto `Job`.
`compile_source()` ejecuta analisis lexico, parsing, analisis semantico y generacion. Un
`CompilationError` se devuelve como HTTP 422 y no existe todavia ningun WebSocket de
ejecucion asociado a un trabajo aceptado.

Solo despues de obtener un `CompiledSimulation` se crea el trabajo y se devuelve HTTP
202 con `job_id`.

## JobManager

`JobManager` ya no compila. Sus responsabilidades empiezan despues de la frontera de
compilacion:

- crear el directorio temporal del trabajo;
- escribir el script ya generado;
- iniciar el subproceso;
- consumir el protocolo de eventos;
- publicar eventos por WebSocket;
- administrar cancelacion y limpieza.

## Protocolo de stdout

El runtime estructurado utiliza exclusivamente lineas con el prefijo
`MC_DSL_EVENT:`. Tras el prefijo se encuentra un objeto JSON.

Una salida auxiliar sin prefijo no se interpreta como mensaje del protocolo y se conserva
solo como diagnostico. Una salida prefijada con JSON invalido o con un tipo de evento no
permitido provoca un fallo controlado de protocolo. `stderr` permanece separado.

## Eventos y ramas

Los eventos `start`, `batch` y `complete` incluyen descriptores de ramas. Si el modelo no
contiene `\pm`, existe la rama `main`. Si contiene `\pm`, existen `plus` y `minus`.

Cada rama publica de forma independiente sus conteos, estadisticos, diagnosticos de
precision e histograma. Esto evita construir una distribucion combinada para escenarios
que el propio operador define como ramas distintas.

## Contrato HTTP / WebSocket

- `POST /api/jobs`: compilacion completa y creacion del trabajo.
- HTTP 422: error LEX/SYN/SEM/GEN.
- HTTP 202: compilacion correcta y `job_id` creado.
- `/ws/jobs/{id}`: seguimiento de la ejecucion ya aceptada.
- WebSocket `error`: problema ocurrido durante el runtime/subproceso.

## Concurrencia

El script usa `ProcessPoolExecutor`. Los lotes poseen semilla derivada de una semilla
maestra y su indice. La agregacion conserva ramas separadas y publica el avance conforme
finalizan los futures.
