# Reproducibilidad v1.2.0

La reproducibilidad de una ejecucion queda definida por la combinacion:

```text
modelo DSL + semilla + batch_size + workers + version del codigo + version de NumPy/Python
```

El runtime generado registra metadatos en los eventos `start` y `complete`:

- `statistical_core_version`
- `python_version`
- `numpy_version`
- `platform`
- `rng_bit_generator`
- `seed_strategy`
- `batch_count`
- `worker_count`
- `nonfinite_policy`

## Estrategia de semillas

Cada lote usa:

```text
numpy.SeedSequence([MASTER_SEED, batch_index])
```

Esto hace que el resultado sea estable para la misma semilla, modelo y plan de lotes.
Si cambia `batch_size`, cambia el particionado de la simulacion y por tanto puede cambiar
la secuencia exacta de muestras. Esta limitacion debe reportarse en articulos o anexos.

## Artefactos recomendados para publicacion

Para cada experimento conserve:

- fuente `.dsl` del modelo;
- semilla;
- tamano de lote;
- numero de workers;
- salida JSONL del runtime;
- CSV procesado;
- hash o commit de Git;
- version de Python y NumPy;
- sistema operativo y hardware.
