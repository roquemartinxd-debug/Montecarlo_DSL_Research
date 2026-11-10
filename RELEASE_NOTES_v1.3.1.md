# Release notes — MC DSL v1.3.1

## Enfoque

Version de estabilizacion tecnica y metodologica. No incorpora nuevas funcionalidades de dominio; corrige limites de memoria, consistencia de planificacion, semantica numerica del optimizador y procedencia experimental.

## Cambios

- Generacion perezosa de tareas fisicas en los modos fijo y adaptativo; los planes almacenan parametros constantes y un contador O(1), no una lista O(T) de descriptores.
- Reordenamiento de envio limitado a ventanas de hasta `2 * worker_count`; futures y resultados pendientes permanecen dentro de esa ventana, con backpressure e ingesta canonica.
- Limites explicitos para buffers de reordenamiento de bloques logicos y del sketch estadistico.
- Una unica funcion de planificacion adaptativa compartida entre el scheduler Python y el runtime generado.
- Medicion de memoria portable: `resource.getrusage` cuando existe y `GetProcessMemoryInfo` en Windows.
- La suma de RSS por proceso se etiqueta como suma de maximos historicos, no como pico simultaneo del arbol de procesos.
- Histogramas de `paired_delta` disponibles para medir la distancia TV real.
- Optimizador conservador: constant folding en float64; no elimina identidades dinamicas ni sustituye `sqrt`/potencias NumPy por operaciones escalares Python.
- Auditoria automatica de dependencias y correccion del registro historico de NumPy no conforme.
- Baseline de v1.2.0 renombrado como emulacion de configuracion historica.
- Suite ampliada con pruebas de generacion perezosa para mas de un millon de tareas, equivalencia entre contador e iterador, modos fijo/ast_only, ventana multiproceso, limites de buffers y redondeo float64.
- Corregida la asercion de version del contrato estadistico: el runtime v1.3.1 se valida contra `1.3.1`, no contra `1.2.0`.

### Compatibilidad FastAPI (hotfix de pruebas en Windows)

- Eliminada la llamada redundante `app.add_event_handler("shutdown", ...)`: el cierre asincrono de `JobManager` ya se gestiona mediante `FastAPI(..., lifespan=lifespan)`.
- Agregada prueba de regresion del ciclo de vida que verifica que `JobManager.shutdown()` se espera exactamente una vez al finalizar `TestClient`.
- Declarado `asyncio_default_fixture_loop_scope = "function"` en la configuracion pytest para evitar advertencias de cambio de comportamiento de `pytest-asyncio`.
- Este hotfix no modifica el compilador, la generacion perezosa, el planificador ni los resultados experimentales.

### Compatibilidad multiprocessing spawn (hotfix de campana en Windows)

- `benchmarks/run_v1_3_experiments.py` registra temporalmente en `sys.path` la carpeta del modulo `mcdsl_gen_*.py`, para que los procesos creados mediante `spawn` puedan importarlo al deserializar `_simulate_batch`.
- La ruta importable y el registro en `sys.modules` se eliminan tras cada ejecucion, incluso ante excepciones, y el archivo Python temporal se borra.
- Se agrega una prueba de regresion que fuerza `spawn` en un interprete aislado y ejecuta dos simulaciones multiproceso consecutivas, verificando resultados, uso real del pool y limpieza de los modulos temporales.
- El problema original era un `ModuleNotFoundError: No module named 'mcdsl_gen_N'` que abortaba Exp. 0 con `BrokenProcessPool`. La correccion se ha verificado con `spawn` en Linux; sigue pendiente confirmacion de toda la campana en Windows.
- No se han regenerado resultados estadisticos o de rendimiento, ni cambiado las formulas o la planificacion.

## Validacion local realizada durante esta correccion

- 242 pruebas descubiertas; las 242 pasaron al ejecutar la suite en tres grupos independientes en Linux con Python 3.13.5 y NumPy 2.3.5.
- La prueba de contrato de version estadistica ahora espera `1.3.1` y pasa.
- Se verifico la generacion perezosa para 100 millones de tareas planificadas sin materializar la lista de descriptores; `tracemalloc` registro 1,776 bytes de pico de asignaciones Python durante la construccion del plan en esa prueba (no es RSS ni memoria total del runtime). Tambien se comprobaron los modos `fixed` y `ast_only` en runtime.
- Una ejecucion monolitica de `pytest -q` excedio el limite de tiempo del entorno de revision; la verificacion de 242/242 se realizo por grupos.
- La ejecucion real en Windows y el CI remoto siguen pendientes.

## Resultados experimentales

Los resultados v1.3.0 se conservan como historicos. La campana v1.3.1 debe ejecutarse de nuevo porque cambiaron el planificador, el buffer multiproceso, la instrumentacion de memoria, la salida pareada y las reglas del optimizador. Consultar `docs/REVALIDATION_REQUIRED_v1.3.1.md`.

### Hotfix del protocolo RNG / smoke test de CI

- Actualizado `benchmarks/rng_reproducibility.py`: el control de tamano de lote
  ahora exige igualdad con `philox_logical_block` y comprueba por separado la
  sensibilidad de la estrategia heredada `batch_index`.
- Corregidos los metadatos del RNG, la ruta de resultados predeterminada y el
  documento `benchmarks/RNG_REPRODUCIBILITY_PROTOCOL.md`, heredados de v1.2.0.
- La configuracion experimental del compilador y del runtime no cambia.

### Hotfix: checkpoints para campana experimental larga (PowerShell / Windows)

- El runner escribe resultados exitosos por simulacion de forma atomica.
- `--resume` recupera simulaciones previamente terminadas y evita repetir
  cientos de mediciones despues de apagar el equipo.
- Se valida la huella del codigo, los modelos y el entorno antes de reusar
  mediciones. Los calentamientos se reejecutan.
- Los CSV de resultados finales se escriben atomica e idempotentemente al
  finalizar cada experimento; no se fabrican resultados nuevos.
- El experimento 6 conserva en el checkpoint las metricas del compilador y el
  tamano del script original para evitar recompilaciones que alteren su evidencia.
- Se registran sesiones de medicion para auditar comparaciones de rendimiento
  cuando se reanuda. Ver `docs/CHECKPOINT_RESUME_v1.3.1.md`.
