# Changelog

## 1.2.0 - 2026-09-14

- Nuevo modulo `montecarlo_dsl.statistics` con acumulacion Welford, combinacion Chan-Golub-LeVeque e intervalo Wilson.
- `\response` acepta `var`, `std`, `count`, `valid_rate`, `discard_rate`, `p05`, `p50` y `p95`.
- El runtime generado reporta varianza muestral, MCSE, IC 95 % de la media, tasas de validos/descartes e intervalos Wilson.
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
