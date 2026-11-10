# EXP-10 — calibracion exploratoria de sobrecoste del scheduler en AWS

## Alcance

Este parche **NO edita** `src/montecarlo_dsl/scheduler.py`, `generator.py`,
`cost_model.py`, el scheduler de v1.3.1, sus cuatro coeficientes AST, ni el
runner anterior EXP-10. Agrega un experimento A/B separado que utiliza la API
existente `InterpretableASTCostModel` y cambia **solo** los parametros del coste
estimado para iniciar un pool y enviar una tarea.

| Parametro | v1.3.1 original | Hipotesis AWS EXP-10 |
|---|---:|---:|
| `parallel_pool_spawn_seconds` | 0.180 s | 0.016 s |
| `parallel_task_ipc_seconds` | 0.006 s | 0.00030 s |
| `parallel_speedup_margin` | 1.12 | 1.12 (SIN cambio) |
| `weights_seconds_per_sample` | 4 coeficientes AWS | Identicos |

Los valores experimentales son **hipotesis informadas por la latencia global**
de las ejecuciones W2 y W4 de EXP-10 AWS. No son mediciones independientes de
`spawn`/`IPC` ni demuestran que esos valores sean optimos. No se deben citar
como tiempos de inicializacion medidos directamente.

Al compilar, se pasa el `InterpretableASTCostModel` original a las condiciones
`serial`, `fixed_w2`, `fixed_w4`, `adaptive_original` y el objeto con **solo esos
dos campos modificados** a `adaptive_tuned`. La politica de seleccion y el
mecanismo RNG permanecen exactamente iguales; las elecciones de workers pueden
cambiar por efecto de las estimaciones.

## Aplicacion (Windows / PowerShell)

Desde la raiz del proyecto donde se aplico el hotfix de reorder buffer:

```powershell
$patch = "$env:USERPROFILE\Downloads\MC_DSL_v1.3.1_EXP10_scheduler_tuning_optin.patch"
Test-Path $patch
git apply --check "$patch"
git apply "$patch"
python -m pytest -q tests/test_exp10_scheduler_tuning.py tests/test_cost_model_and_scheduler.py tests/test_targeted_experiments.py
```

No ejecutar este experimento en los directorios de checkpoints historicos.

## Ejecucion inicial en AWS (Linux, perfil `aws-4core`)

Agregar los dos archivos nuevos del parche al mismo arbol de codigo que se uso
para EXP-10 AWS. Con virtualenv activado:

```bash
python -m pytest -q tests/test_exp10_scheduler_tuning.py tests/test_cost_model_and_scheduler.py
mkdir -p logs
set -o pipefail
python -u benchmarks/run_exp10_scheduler_tuning.py \
  --profile aws-4core \
  --platform-label aws-c7i-2xlarge-us-east-2 \
  --physical-cores 4 \
  --instance-type c7i.2xlarge \
  --cloud-region us-east-2 \
  --campaign-label aws-exp10-scheduler-overhead-ab-run01 \
  --cost-model-weights '4.118662634957425e-08,7e-09,2.4547267526488982e-08,5.5e-09' \
  --iterations '100000,250000,500000,750000,1000000,1500000,2000000' \
  --repetitions 10 \
  --checkpoint-dir benchmarks/checkpoints/tuning/aws_exp10_scheduler_ab_run01 \
  --output-dir benchmarks/results/tuning/aws_exp10_scheduler_ab_run01 \
  --progress-every 10 \
  2>&1 | tee logs/aws-exp10-scheduler-ab-run01.log
```

Se esperan 4 modelos x 7 niveles x 5 variantes x 10 repeticiones = **1400
mediciones**, mas los warmups. Se comprueba identidad de resultados por SHA
entre las cinco variantes para cada modelo/N/semilla antes de publicar CSV.

Para incluir anclajes exploratorios de N=5M y 10M de `normal_mean` y
`quadratic_pm`, usar `--include-large-anchors` **desde el inicio de una campaña
nueva**, no agregarlo a una campaña empezada. En ese caso se esperan 1600
mediciones (1400 + 200). Opcionalmente iniciar primero una prueba minima con
`--iterations '100000,250000' --repetitions 1` y rutas de checkpoint/output
**distintas**, antes de la campaña principal.

Archivos generados:
- `exp10_scheduler_tuning_raw.csv`: configuracion, workers, tiempos, semilla, hashes.
- `exp10_scheduler_tuning_summary.csv`: medianas e intervalos bootstrap de aceleracion emparejada.
- `scheduler_tuning_environment.json`: perfil, parametros y huellas del codigo.
- `manifest.json`, `runs/`, `campaign_status.json`: checkpoints verificables.

Para reanudar, repetir **exactamente** los argumentos anteriores y agregar
`--resume`. El algoritmo y los pesos deben ser los mismos. Los warmups se
repetiran, pero no las mediciones ya guardadas.

## Limitaciones y decision

No se ha demostrado que la variante sea mas rapida; este A/B es la prueba.
Si la variante muestra mejoras confirmadas en AWS sin regresiones relevantes,
se puede proponer otro perfil experimental; **no aplicar estos parametros
automaticamente a Windows**, donde EXP-10 encontro sobrecostes del pool mucho
mayores. Conservar resultados historicos y versionar cualquier futura politica.

El script nuevo no forma parte de `source_tree_digest` de las campanas viejas:
la huella del ejecutor A/B se guarda separadamente en `runner_sha256`. No
reabrir/reusar rutas de resultado o checkpoint de v1.3.1 ni de EXP-10 previo.

## Uso optativo fuera del benchmark (solo despues de evaluar)

El mecanismo ya existe en v1.3.1; no hace falta editar `scheduler.py`:

```python
from dataclasses import replace
from montecarlo_dsl import InterpretableASTCostModel, GeneratorConfig, compile_source

weights_aws = (4.118662634957425e-08, 7e-09,
               2.4547267526488982e-08, 5.5e-09)
reference = InterpretableASTCostModel(weights_seconds_per_sample=weights_aws)
experimental = replace(
    reference,
    parallel_pool_spawn_seconds=0.016,
    parallel_task_ipc_seconds=0.00030,
)
config = GeneratorConfig(scheduler_mode="hybrid", max_workers=4, seed=101)
# `source` es el texto DSL existente, no se modifica el modelo.
compiled = compile_source(source, config, cost_model=experimental)
```

Este uso NO esta habilitado de forma predeterminada y no se recomienda aun
para uso productivo. El coste de `spawn` y `IPC` cambia con el equipo, su carga,
la version Python y el modo de inicio de procesos.
