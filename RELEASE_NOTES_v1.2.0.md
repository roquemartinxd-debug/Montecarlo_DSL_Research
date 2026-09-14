# Release notes v1.2.0

## Enfoque

Version orientada a publicacion: agrega nucleo estadistico estable, respuesta estadistica
extendida, metadatos de reproducibilidad, analisis pareado para `\pm` y scripts de validacion
experimental.

## Cambios principales

- Nuevo modulo `montecarlo_dsl.statistics` con `RunningStats`, combinacion de lotes y Wilson CI.
- `\response` acepta `var`, `std`, `count`, `valid_rate`, `discard_rate`, `p05`, `p50`, `p95`.
- El runtime reporta `sample_variance`, MCSE, intervalo de media, tasas de validez/descarte e intervalos Wilson.
- Las ramas `\pm` agregan bloque `paired` con diferencia pareada `plus - minus`.
- Se agregan metadatos de reproducibilidad en eventos `start` y `complete`.
- Se agregan modelos analiticos y scripts en `benchmarks/`.
- Nueva documentacion metodologica en `docs/STATISTICAL_METHODS.md`, `docs/REPRODUCIBILITY.md` y `docs/EXPERIMENTAL_PROTOCOL.md`.

## Compatibilidad

Los estadisticos existentes `avg`, `min` y `max` se conservan. `min` y `max` deben interpretarse
como extremos muestrales observados.
