# Benchmarks reproducibles v1.2.0

Esta carpeta separa dos tipos de experimentos:

- `models/analytic/`: modelos con media y varianza teoricas conocidas para validar el nucleo estadistico.
- `models/performance/`: modelos mas costosos para medir tiempo, workers, lotes y escalabilidad.

Comandos sugeridos:

```powershell
python benchmarks/statistical_validation.py
python benchmarks/run_benchmarks.py --quick
python benchmarks/analyze_results.py
```

Los resultados crudos se escriben en `benchmarks/results/raw/` y los agregados en
`benchmarks/results/processed/`. Los scripts registran metadatos de reproducibilidad:
version de Python, NumPy, sistema operativo, semilla, workers, tamano de lote y version
del nucleo estadistico reportada por el runtime generado.

## Reproducibilidad RNG

La invariancia numerica frente al numero de workers se evalua por separado del benchmark de rendimiento:

```powershell
python benchmarks/rng_reproducibility.py
```

El protocolo completo se documenta en `benchmarks/RNG_REPRODUCIBILITY_PROTOCOL.md`. Los resultados oficiales v1.2.0 se conservan en `benchmarks/results/rng_reproducibility/v1.2.0/` e incluyen 120 ejecuciones principales (cuatro modelos, workers 1/2/4 y 10 repeticiones), controles de semilla y `batch_size`, hashes SHA-256 del resultado canonizado y el entorno de ejecucion.

## Control RNG actualizado para v1.3.1

El script `rng_reproducibility.py` tiene un protocolo v1.3.1 distinto del
control historico v1.2.0 anterior: exige invariancia frente al tamano de lote
fisico con `philox_logical_block` (bloques logicos fijos), y comprueba por
separado la sensibilidad de la estrategia heredada `batch_index`. Los
resultados nuevos se guardan en `benchmarks/results/rng_reproducibility/v1.3.1/`.
No confundir los resultados v1.2.0 archivados con la campana v1.3.1.
