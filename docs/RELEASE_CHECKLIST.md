# Release checklist - v1.1.0

> **Estado v1.3.1 (2026-10-31):** estabilizacion de codigo, no liberacion cientifica final. 242/242 pruebas superadas por grupos en Linux/Python 3.13.5/NumPy 2.3.5; no publicar ni enviar el articulo hasta revalidar en PowerShell/Python 3.11, regenerar resultados y ejecutar CI en Linux/Windows.


- [x] Version del paquete: 1.1.0.
- [x] BNF sin no terminales equivalentes redundantes.
- [x] Identificadores descriptivos implementados.
- [x] `\distrib` con separadores explicitos entre pares.
- [x] Compilacion completa antes de crear un trabajo runtime.
- [x] HTTP 422 reservado para errores de compilacion.
- [x] WebSocket utilizado para seguimiento runtime despues de aceptar el trabajo.
- [x] Protocolo stdout aislado mediante `MC_DSL_EVENT:`.
- [x] `stderr` separado del canal estructurado.
- [x] Ramas `\pm` conservadas por separado.
- [x] MCSE e intervalo aproximado de 95 % por rama.
- [x] Identificadores DSL mapeados a nombres Python seguros.
- [x] Modelo de evidencia actualizado.
- [x] 205 pruebas automatizadas aprobadas.
