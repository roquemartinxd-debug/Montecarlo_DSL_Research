# Protocolo experimental sugerido para version publicable

## Validacion estadistica

1. Ejecutar modelos con solucion teorica conocida en `benchmarks/models/analytic/`.
2. Usar varias semillas independientes.
3. Comparar media estimada contra media teorica.
4. Reportar error absoluto, MCSE y si el error cae dentro de `3 * MCSE`.
5. Reportar varianza estimada, tasa de validos y tasa de descartes.

Comando:

```powershell
python benchmarks/statistical_validation.py
```

Salida principal:

```text
benchmarks/results/processed/statistical_validation.csv
```

## Benchmark de rendimiento

1. Usar al menos una ejecucion de calentamiento si se toman tiempos para publicacion.
2. Usar 10 o mas repeticiones por configuracion; 30 es preferible si el tiempo lo permite.
3. Comparar `workers = 1, 2, 4` y mas solo si el hardware lo justifica.
4. Reportar mediana, media, desviacion y speedup contra 1 worker.
5. No afirmar escalabilidad lineal sin evidencia.

Comandos:

```powershell
python benchmarks/run_benchmarks.py --quick
python benchmarks/analyze_results.py
```

Para ejecucion completa retire `--quick`.

## Amenazas a la validez que deben declararse

- Las variables aleatorias son independientes por defecto.
- El DSL no implementa correlacion ni reduccion de varianza.
- Los intervalos de la media son aproximaciones normales.
- `min` y `max` son extremos muestrales.
- Si existen descartes, la media es condicional sobre resultados finitos.
- El rendimiento puede depender de hardware, sistema operativo y version de NumPy.
