# Protocolo experimental y revalidacion — MC DSL v1.3.1

> **Estado:** protocolo corregido para la campaña v1.3.1. Los CSV y las tablas de `benchmarks/results/final/v1.3.0/` son resultados historicos de la version anterior; no deben presentarse como resultados de v1.3.1 hasta regenerarlos con este codigo.

## 1. Entorno y procedencia

El registro historico `benchmarks/results/final/v1.3.0/environment.json` contiene Python 3.11.2, NumPy 1.24.2, FastAPI 0.92.0, Pydantic 1.10.4, Uvicorn 0.17.6, SciPy 1.15.2 y HTTPX 0.23.3, sobre Linux x86_64 con 2 CPU logicas. El `pyproject.toml` de esa entrega declaraba `numpy>=2.0`; por tanto, ese entorno **no era conforme** con las dependencias declaradas. El JSON historico se ha corregido para registrar expresamente la discrepancia.

La funcion `write_environment_metadata()` comprueba ahora las dependencias declaradas en `pyproject.toml` y registra versiones instaladas, requisitos incompatibles y errores de auditoria. Una campaña no debe marcarse como conforme si esa comprobacion falla. El nuevo registro se escribira en `benchmarks/results/final/v1.3.1/environment.json`.

Cada campaña debe conservar: commit exacto, estado del arbol Git, version de Python, versiones de dependencias, plataforma, CPU logicas, semilla, configuracion, repeticiones y archivos CSV originales. Si el repositorio no tiene metadatos Git validos, el commit debe registrarse como `unknown`; no se debe inventar.

## 2. Repeticiones reales definidas por el script

Los valores de CLI son limites por tipo de experimento, no implican que cada celda de todos los experimentos use 30 repeticiones.

| Experimento | Configuracion implementada | Repeticiones |
|---|---|---:|
| Exp. 0 — comparacion historica | 4 modelos, N = 100.000 y 1.000.000, 3 variantes | 30 por celda |
| Exp. 1 — equivalencia funcional | 8 modelos, 4 modos de planificacion, N = 100.000 | 1 por celda |
| Exp. 2 — rendimiento interno | 4 modelos, N = 10.000, 100.000 y 1.000.000, 4 configuraciones | 30 solo para `normal_mean` y `heavy_expression` a 100.000 y 1.000.000; 10 en las demas celdas |
| Exp. 2 — escalabilidad adicional | `normal_mean`, N = 10.000.000, `fixed_w1` y `adaptive_hybrid` | 10 por variante |
| Exp. 3 — adaptacion de lotes | Rejilla de tamanos de lote | 10 |
| Exp. 4 — invariancia de planificacion | 6 modelos y 14 configuraciones | 1 por configuracion |
| Exp. 5 — generalizacion del modelo de coste | 8 modelos de entrenamiento y 6 de prueba | 10 por modelo |
| Exp. 6 — optimizacion del compilador | 6 modelos, N = 500.000 | 10 pares optimizado/no optimizado |
| Exp. 7 — validacion estadistica | Modelos analiticos, N = 100.000 | 25 semillas por modelo |
| Exp. 8a — memoria | 2 modelos, 1 o 2 workers, varios tamanos N | 1 por configuracion y modo de memoria |
| Exp. 8b — fidelidad de cuantiles e histogramas | 6 familias, N = 1.000.000, todas las ramas y rama pareada cuando exista | 1 por modelo y modo |
| Exp. 9 — ablacion | 4 modelos, 4 modos x 2 estados del optimizador | 10 por celda |

Los intervalos bootstrap de las comparaciones de rendimiento usan 10.000 remuestreos en `bootstrap_ci95()`. La campaña de v1.3.1 debe reportar los tamaños muestrales realmente escritos en cada CSV; no debe generalizar el valor 30 a celdas que solo tienen 1 o 10 ejecuciones.

## 3. Definicion de baselines

El experimento Exp. 0 denomina su primera variante `v1.2.0_legacy_configuration_emulation`. Esta variante ejecuta el compilador/runtime actual con parametros que emulan la configuracion historica (`batch_index`, `retain_all`, optimizador desactivado y planificacion fija). **No ejecuta el codigo fuente original v1.2.0.**

Por ello, los speedups de Exp. 0 solo pueden describirse como comparaciones contra una emulacion de configuracion historica. Una comparacion contra la implementacion original v1.2.0 requiere recuperar y ejecutar ese codigo fuente bajo un entorno y protocolo comparables.

Exp. 2 compara configuraciones internas de v1.3.1 (`fixed_w1`, `fixed_w2`, `fixed_w4`, `adaptive_hybrid`) y debe mantenerse separada de Exp. 0.

## 4. Definiciones de memoria

Se reportan dos familias de medidas distintas:

1. `peak_retained_coordinator_bytes`: estimacion de carga util retenida en el coordinador en los puntos de ingesta. Incluye payloads de arrays NumPy, estructuras de reduccion y buffers de resultados/reordenamiento que la instrumentacion puede observar. No es un recuento exacto de toda la memoria del heap Python ni sustituye al RSS.
2. `coordinator_peak_rss_bytes`, `workers_peak_rss_bytes` y `sum_of_process_peak_rss_bytes`: maximos historicos por proceso. La suma de maximos individuales **no es** el maximo simultaneo del arbol de procesos y puede sobreestimar el pico concurrente.

El alias historico `aggregate_peak_rss_bytes` se conserva por compatibilidad, pero tiene la misma semantica de suma de maximos por proceso. Los nuevos analisis deben usar `sum_of_process_peak_rss_bytes` y describir la metrica correctamente. Para medir un pico agregado simultaneo se necesita muestreo del arbol de procesos o una medicion de cgroup equivalente.

La afirmacion de memoria acotada se limita a ejecuciones con tamano maximo de tarea, numero de workers, ventana de tareas en vuelo y buffer de reordenamiento fijos. La implementacion limita la ventana multiproceso a `2 * worker_count` y los buffers de bloques pendientes a `MAX_PENDING_LOGICAL_BLOCKS = 64`. No se debe afirmar un limite independiente de esos parametros.

## 5. Fidelidad estadistica

El modo `bounded` conserva momentos, extremos y conteos mediante acumuladores exactos/mergeables, mientras que cuantiles e histogramas son aproximaciones del sketch ponderado. La campaña debe producir `quantile_and_histogram_fidelity.csv` para las ramas individuales y `paired_delta` cuando exista.

La distancia de variacion total (TV) para `paired_delta` debe calcularse a partir de los histogramas realmente emitidos por ambos modos. No se admite un valor constante de cero ni una celda sintetica. Si el histograma no esta disponible, debe registrarse `N/A` y excluirse del calculo agregado.

La correlacion reportada por el runtime es la media de correlaciones por bloque logico, combinada en orden canonico; no debe denominarse coeficiente de Pearson global de todas las muestras sin una derivacion adicional basada en estadisticos suficientes globales.

## 6. Reproduccion de v1.3.1

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10
python scripts/generate_v1_3_figures_tables.py
```

La campana completa es costosa y puede tardar. Antes de iniciar, verifica que `pyproject_dependencies_compliant` sea `true` en el entorno y que el repositorio tenga un commit identificable. Los resultados nuevos deben guardarse en `benchmarks/results/final/v1.3.1/` y las tablas/figuras deben llevar el identificador v1.3.1.

## 7. Criterios de publicacion

No se debe congelar el manuscrito ni enviar el articulo hasta que:

- la suite completa pase en el entorno declarado y CI verifique Windows y Linux;
- se regeneren todos los resultados afectados por los cambios de memoria, planificador y optimizador;
- se compruebe la fidelidad de histogramas y cuantiles de `paired_delta`;
- las tablas incluyan las repeticiones reales de cada celda;
- las afirmaciones de complejidad espacial y RSS coincidan con las definiciones de esta seccion;
- el baseline se denomine emulacion, salvo que se ejecute el codigo historico original.

## 8. Checkpoints y recuperacion de una campana interrumpida

La campana v1.3.1 admite checkpoints a nivel de simulacion. El inicio crea
`benchmarks/checkpoints/v1.3.1/manifest.json` y guarda cada resultado completo
con tiempos, hash, configuracion, sesion y fecha en archivos atomicos. La
reanudacion requiere `--resume` y las mismas versiones/codigo/modelos, CPU,
plataforma y numeros de repeticiones. Los calentamientos se repiten. Los CSV
se regeneran al finalizar su experimento con las mismas filas y metricas.

Para las instrucciones PowerShell ve `docs/CHECKPOINT_RESUME_v1.3.1.md`.
Los tiempos combinados de distintas sesiones requieren documentar posible
sesgo ambiental; la integridad de los checkpoints no garantiza condiciones
de CPU o temperatura identicas.
