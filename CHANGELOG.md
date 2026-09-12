# Changelog

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
