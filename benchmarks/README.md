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
