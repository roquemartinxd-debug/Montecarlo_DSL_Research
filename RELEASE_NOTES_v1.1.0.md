# Monte Carlo DSL v1.1.0 - Notas de release

## Objetivo

Esta version incorpora el endurecimiento tecnico derivado de la revision final del caso
practico y elimina ambiguedades detectadas en la definicion del lenguaje, el protocolo
de ejecucion y el tratamiento estadistico del operador `\pm`.

## Cambios

- BNF simplificada: se elimino la duplicacion entre `DECIMAL_SIGNADO` y `NUMERO_SIGNADO`.
- Distribuciones discretas con separador externo explicito entre pares.
- Identificadores de varias posiciones con letras, digitos y guion bajo.
- Mapeo de identificadores del DSL a nombres Python internos para evitar colisiones con palabras reservadas.
- Nueva capa `compiler.py` que completa LEX/SYN/SEM/GEN antes de crear un trabajo runtime.
- Protocolo runtime v2: solo las lineas con prefijo `MC_DSL_EVENT:` se interpretan como eventos JSON.
- Salidas libres de `stdout` se aislan como diagnostico y no pueden corromper el protocolo.
- `\pm` mantiene dos ramas independientes (`plus` y `minus`) con conteos, estadisticos e histogramas propios.
- Diagnosticos numericos por rama: desviacion estandar muestral, MCSE e intervalo de confianza del 95 % para la media.
- Interfaz actualizada con selector de rama y resultados separados.
- Suite final: **205 pruebas aprobadas**.
