# Auditoria de integracion cliente-servidor - v1.1.0

## Resultado

**APROBADO.** La revision final reforzo la frontera entre compilacion y ejecucion, el
protocolo de subproceso y la representacion de resultados del operador `\pm`.

## Cambios verificados

1. `POST /api/jobs` ejecuta LEX/SYN/SEM/GEN antes de crear un trabajo.
2. Los errores de compilacion se devuelven mediante HTTP 422.
3. El WebSocket se utiliza despues de aceptar el trabajo y comunica eventos runtime.
4. `JobManager` recibe un `CompiledSimulation`; no vuelve a analizar ni generar codigo.
5. stdout usa un prefijo reservado `MC_DSL_EVENT:` para los eventos estructurados.
6. Las lineas no pertenecientes al protocolo no pueden convertirse accidentalmente en
eventos JSON consumidos por FastAPI.
7. `stderr` se conserva como canal de diagnostico independiente.
8. El frontend soporta resultados por rama para `\pm` y no presenta estadisticas
combinadas.
9. La suite incorpora pruebas especificas para la frontera compile/runtime y para salida
no protocolaria en stdout.

## Estado de pruebas

La suite final contiene **205 pruebas automatizadas aprobadas**.
