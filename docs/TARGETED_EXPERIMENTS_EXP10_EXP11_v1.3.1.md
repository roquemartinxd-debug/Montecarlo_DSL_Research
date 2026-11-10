# MC DSL v1.3.1 — Protocolos dirigidos EXP-10 y EXP-11

## Alcance y decisión de diseño

Estos estudios son complementarios a las campañas v1.3.1 ya completadas. **No cambian la política ni los coeficientes del scheduler**. EXP-10 observa el scheduler existente con los pesos calibrados de cada plataforma. EXP-11 estudia el comportamiento espacial del runtime `bounded`.

El único cambio de runtime es un hook experimental con valor predeterminado cero: `GeneratorConfig.task_delay_first_seconds`. Se utiliza exclusivamente con `task_completion_order="delayed_first"` para retrasar de forma controlada la primera tarea física y estresar los buffers de reordenamiento. No modifica el RNG ni las decisiones del scheduler; su valor queda registrado en la identidad del checkpoint y en los metadatos.

Los dos estudios deben ejecutarse por separado en:

- Windows local: 2 núcleos físicos declarados, 2 workers como máximo adaptativo. Fixed W4 se conserva únicamente como prueba de sobremultiplexación.
- AWS EC2: 4 núcleos físicos declarados, 4 workers como máximo adaptativo.

Las comparaciones temporales principales son internas a cada plataforma. No se agrupan tiempos Windows y AWS en una misma distribución.

## Dependencias adicionales

La monitorización del árbol de procesos usa `psutil==7.2.2`, fijado en `requirements-aws-experiment.txt` y en las dependencias de desarrollo. Instalar la misma versión en ambas plataformas:

```bash
python -m pip install psutil==7.2.2
```

Comprobar:

```bash
python -c "import psutil; print(psutil.__version__)"
```

## EXP-10 — Adaptive Scheduling Break-Even Analysis

### Pregunta

¿A partir de qué N resulta más rápido un pool de 1, 2 o 4 workers que ejecutar directamente en el proceso principal, y cómo se compara el scheduler híbrido actual con la mejor configuración fija observada?

### Modelos y N

Modelos: `normal_mean`, `linear_combination`, `heavy_expression`, `quadratic_pm`.

Exploración: 100k, 250k, 500k, 750k, 1M, 1.5M y 2M iteraciones para los cuatro modelos. Anchors adicionales de 5M y 10M para `normal_mean` y `quadratic_pm`.

Configuraciones:

1. `serial_in_process`: `ast_only`, 1 worker, sin pool.
2. `fixed_pool_w1`: `fixed`, 1 worker, pool.
3. `fixed_pool_w2`: `fixed`, 2 workers, pool.
4. `fixed_pool_w4`: `fixed`, 4 workers, pool (sobremultiplexado en Windows de 2 núcleos).
5. `adaptive_hybrid`: `hybrid`, límite de workers igual al número de núcleos físicos declarado.

El orden de configuración se aleatoriza de forma determinista dentro de cada repetición. Las comparaciones dentro de cada celda usan la misma semilla entre configuraciones. Se registra tiempo de compilación, tiempo de ejecución y tiempo total; el análisis del punto de equilibrio usa `execution_seconds`, que incluye piloto y ejecución, pero separa compilación.

### Fase exploratoria

10 repeticiones por celda (5 para los anchors grandes), una ejecución de calentamiento por configuración. Las semillas empiezan en 3,100,001.

### Fase confirmatoria

Después de inspeccionar el CSV exploratorio, seleccionar tres N que representen: por debajo del equilibrio, cerca del equilibrio y por encima. Ejecutar 30 repeticiones por configuración en esos N con semillas nuevas (base 4,100,001). El parámetro `--confirmatory-iterations` recibe una lista separada por comas y se aplica a los cuatro modelos.

### Métricas

- Mediana, media, desviación estándar e IQR de `execution_seconds`.
- Speedup emparejado `T_serial_in_process / T_configuration`, con IC bootstrap del 95 %.
- Workers solicitados y efectivos, uso de pool y coste de piloto.
- Mejor configuración fija observada y regret adaptativo respecto a ella.
- Primer N donde el límite inferior del IC bootstrap emparejado permanece por encima de 1 en ese N y en todos los N mayores ensayados.

La identificación del punto de equilibrio es específica de modelo, plataforma y configuración. No se extrapola fuera de los N medidos.

## EXP-11 — Bounded Memory Asymptotic Scaling Analysis

### Pregunta

Con tamaño de bloque, número de workers, ventana de tareas y capacidad de sketch fijos, ¿las estructuras del modo `bounded` permanecen limitadas al crecer N? ¿Qué ocurre bajo finalización adversa de la primera tarea?

### Modelos y N

Modelos: `normal_mean` y `quadratic_pm` (incluye ramas pareadas y correlaciones).

`bounded`: 250k, 1M, 2.5M, 5M, 10M y 20M en Windows; AWS añade 50M.

`retain_all` como control positivo: 250k, 1M y 2.5M; 5M opcional mediante `--include-retain-5m`, si la memoria disponible lo permite.

Workers: serial en proceso W1 y pool W2; AWS añade pool W4. No se fuerza W4 en el equipo Windows de dos núcleos para el estudio de memoria.

Repeticiones: 5 por celda `bounded`, 3 por celda `retain_all`. Cada sonda se ejecuta en un proceso nuevo.

### Orden adverso

En `bounded`, para 10M y 20M, con el máximo número de workers de la plataforma, se comparan `as_completed`, `reverse`, `shuffled` y `delayed_first`, con 3 repeticiones emparejadas. `delayed_first` somete primero las otras tareas de la ventana acotada y retrasa 1 segundo la tarea física 0. Es un estrés artificial de head-of-line, no una medición de rendimiento normal.

### Medición de memoria

`psutil` muestrea el proceso generado y todos sus descendientes cada 10 ms. Se guardan:

- RSS pico del coordinador.
- Suma RSS de workers en el mismo instante de muestreo.
- Suma RSS del árbol de procesos en el mismo instante de muestreo.
- RSS incremental sobre la línea base del árbol al recibir el evento `start`.
- Número máximo de procesos observados y número de muestras.
- Carga útil retenida estimada informada por el runtime.

La suma RSS simultánea puede contar páginas compartidas más de una vez; se etiqueta como suma del árbol de RSS, no como memoria física privada exacta. La frecuencia de muestreo puede no capturar picos muy breves.

También se registran límites y máximos observados de futures, resultados completados, bloques pendientes, cobertura, estadísticas de ramas y correlaciones pareadas, junto con capacidad máxima del sketch, tamaño de bloque y tamaño de lote.

La fidelidad se compara entre `retain_all` y `bounded` con la misma semilla: momentos, conteos, cuantiles e histogramas de todas las ramas y de `paired_delta` cuando exista.

### Interpretación de O(1)

Una pendiente log-log beta cercana a cero es evidencia empírica compatible con memoria constante respecto a N, no una demostración formal. La afirmación asintótica requiere además verificar en el código que las estructuras relevantes tienen límites fijos: bloque lógico, lote, ventana de tareas en vuelo, buffers de reordenamiento, bins y sketch.

El tamaño de los checkpoints en disco puede crecer con el número de simulaciones; esto no es memoria RAM del runtime. El estudio no debe afirmar que el almacenamiento persistente sea O(1).

## Ejecución en Windows (2 núcleos físicos)

Utilizar los pesos ajustados guardados en el `environment.json` de la campaña Windows v1.3.1:

```text
2.2583574403375308e-07,1.766111877457121e-08,8.981806012907278e-08,5.5e-09
```

EXP-10 exploratorio:

```powershell
python benchmarks/run_v1_3_targeted_experiments.py `
  --study break-even --phase exploratory `
  --profile windows-2core --platform-label windows-local-2core --physical-cores 2 `
  --cost-model-weights "2.2583574403375308e-07,1.766111877457121e-08,8.981806012907278e-08,5.5e-09" `
  --checkpoint-dir benchmarks/checkpoints/targeted/windows-exp10 `
  --output-dir benchmarks/results/targeted/windows-2core
```

EXP-10 confirmatorio (sustituir los N por los seleccionados después de explorar):

```powershell
python benchmarks/run_v1_3_targeted_experiments.py `
  --study break-even --phase confirmatory `
  --profile windows-2core --platform-label windows-local-2core --physical-cores 2 `
  --cost-model-weights "2.2583574403375308e-07,1.766111877457121e-08,8.981806012907278e-08,5.5e-09" `
  --confirmatory-iterations "500000,750000,1000000" `
  --checkpoint-dir benchmarks/checkpoints/targeted/windows-exp10 `
  --output-dir benchmarks/results/targeted/windows-2core
```

EXP-11:

```powershell
python benchmarks/run_v1_3_targeted_experiments.py `
  --study memory --profile windows-2core `
  --platform-label windows-local-2core --physical-cores 2 `
  --cost-model-weights "2.2583574403375308e-07,1.766111877457121e-08,8.981806012907278e-08,5.5e-09" `
  --max-memory-iterations 20000000 --include-retain-5m `
  --checkpoint-dir benchmarks/checkpoints/targeted/windows-exp11 `
  --output-dir benchmarks/results/targeted/windows-2core
```

Añadir `--resume` al mismo comando después de una interrupción. No cambiar código, pesos, repeticiones, plataforma ni rutas al reanudar.

## Ejecución en AWS (4 núcleos físicos)

Utilizar los pesos ajustados guardados en el `environment.json` de la campaña AWS v1.3.1:

```text
4.118662634957425e-08,7e-09,2.4547267526488982e-08,5.5e-09
```

EXP-10 exploratorio:

```bash
python -u benchmarks/run_v1_3_targeted_experiments.py \
  --study break-even --phase exploratory \
  --profile aws-4core --platform-label aws-c7i-2xlarge-us-east-2 --physical-cores 4 \
  --instance-type c7i.2xlarge --cloud-region us-east-2 \
  --cost-model-weights "4.118662634957425e-08,7e-09,2.4547267526488982e-08,5.5e-09" \
  --checkpoint-dir benchmarks/checkpoints/targeted/aws-exp10 \
  --output-dir benchmarks/results/targeted/aws-4core
```

EXP-10 confirmatorio:

```bash
python -u benchmarks/run_v1_3_targeted_experiments.py \
  --study break-even --phase confirmatory \
  --profile aws-4core --platform-label aws-c7i-2xlarge-us-east-2 --physical-cores 4 \
  --instance-type c7i.2xlarge --cloud-region us-east-2 \
  --cost-model-weights "4.118662634957425e-08,7e-09,2.4547267526488982e-08,5.5e-09" \
  --confirmatory-iterations "500000,750000,1000000" \
  --checkpoint-dir benchmarks/checkpoints/targeted/aws-exp10 \
  --output-dir benchmarks/results/targeted/aws-4core
```

EXP-11:

```bash
python -u benchmarks/run_v1_3_targeted_experiments.py \
  --study memory --profile aws-4core \
  --platform-label aws-c7i-2xlarge-us-east-2 --physical-cores 4 \
  --instance-type c7i.2xlarge --cloud-region us-east-2 \
  --cost-model-weights "4.118662634957425e-08,7e-09,2.4547267526488982e-08,5.5e-09" \
  --max-memory-iterations 50000000 --include-retain-5m \
  --checkpoint-dir benchmarks/checkpoints/targeted/aws-exp11 \
  --output-dir benchmarks/results/targeted/aws-4core
```

En ambos sistemas, si se interrumpe una fase, repetir exactamente el mismo comando añadiendo `--resume`.

## Archivos generados

EXP-10:

- `exp10_break_even_exploratory_raw.csv`
- `exp10_break_even_exploratory_summary.csv`
- `exp10_break_even_exploratory_thresholds.csv`
- equivalentes `confirmatory` después de ejecutar la fase confirmatoria.

EXP-11:

- `exp11_memory_scaling_raw.csv`
- `exp11_memory_scaling_summary.csv`
- `exp11_memory_fidelity.csv`
- `exp11_memory_structural_bounds.csv`
- `exp11_memory_scaling_slopes.csv`

Cada fase tiene su manifiesto/checkpoints y su archivo `targeted_*_environment.json`. El scheduler actual se mantiene sin ajuste; los resultados describen su comportamiento, no una versión optimizada posterior.
