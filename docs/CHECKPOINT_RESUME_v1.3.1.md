# Campana experimental v1.3.1: checkpoints y reanudacion (Windows/PowerShell)

Esta ampliacion afecta exclusivamente al **runner experimental**. No cambia el
planificador de produccion, la semilla, las simulaciones, la forma de agregar
resultados ni el formato de los CSV. A partir de ahora, cada simulacion **que
termine correctamente** se guarda por separado en
`benchmarks/checkpoints/v1.3.1/runs/`.

## 1. Inicio de una campana NUEVA

Situate en la raiz del proyecto y activa el entorno `.venv` ya instalado.

```powershell
New-Item -ItemType Directory -Force .\logs | Out-Null
python -m pytest -q tests/test_benchmark_checkpoint.py tests/test_experiment_spawn.py
python -u benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10 2>&1 | Tee-Object -FilePath .\logs\experiments-v1.3.1.log
if ($LASTEXITCODE -ne 0) { throw "Campana interrumpida o fallida; revisa el log antes de continuar." }
```

Muestra progreso por simulacion: `[checkpoint] SAVED new=... reused=... last=...`.
Los calentamientos muestran `[warmup]` y se ejecutan nuevamente despues de
reiniciar una sesion.

El directorio de checkpoints se crea automaticamente. Los resultados de una
campana lanzada **antes de instalar este parche** no se pueden reanudar con este
mecanismo: no se guardaron registros por simulacion.

## 2. Continuar DESPUES de cerrar PowerShell o reiniciar Windows

Deja terminados los procesos Python antiguos antes de reanudar. NO elimines
`benchmarks/checkpoints/v1.3.1/`. Ejecuta **exactamente la misma configuracion**:

```powershell
python -u benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10 --resume 2>&1 | Tee-Object -FilePath .\logs\experiments-v1.3.1.log -Append
if ($LASTEXITCODE -ne 0) { throw "La campana aun no ha terminado; comprueba el log." }
```

El runner revisa el manifiesto, reconstruye Exp. 5 con las mediciones
almacenadas, reejecuta los calentamientos y salta las simulaciones completadas.
Una simulacion interrumpida antes de su checkpoint se repite desde cero; no se
reanudan iteraciones *dentro* de una simulacion.

## 3. Integridad, proteccion frente a mezclas y evidencias

- `manifest.json`: huella SHA-256 del codigo fuente de compilador/runner,
  modelos DSL y `pyproject.toml`; incluye Python, versiones de NumPy/SciPy/FastAPI,
  plataforma, CPU y parametros `--*-reps`.
- `runs/xx/<hash>.json`: resultado individual original, tiempos medidos,
  metadatos, huella de integridad, identificador de sesion y fecha UTC.
- `campaign_status.json`: `running`, `interrupted` o `completed`, contadores
  y sesiones observadas. Si Windows se cierra abruptamente, puede conservar
  `running`; la existencia de checkpoints individuales sigue siendo valida.
- `benchmarks/results/final/v1.3.1/environment.json`: al terminar incluye
  los metadatos del checkpoint; los CSV siguen siendo los originales por
  experimento. Se escriben atomically (renombrado `os.replace`).

Un checkpoint existente exige `--resume`. Si el codigo, las dependencias,
la plataforma, la CPU o las repeticiones difieren, el programa **se detiene**
y solicita un directorio nuevo. No ignores esa proteccion. Para realizar
otra campana independiente usa:

```powershell
python -u benchmarks/run_v1_3_experiments.py --confirmatory-reps 30 --pilot-reps 10 --ablation-reps 10 --checkpoint-dir .\benchmarks\checkpoints\v1.3.1_campaign2
```

El checkpoint solo reutiliza resultados de simulaciones exitosas. Los errores,
las ejecuciones incompletas y los calentamientos **no** se guardan como datos
validos.

## 4. Consideraciones cientificas

El checkpoint preserva los **tiempos medidos originalmente**, no los calcula de
nuevo al reanudar. Mezclar mediciones obtenidas en sesiones distintas puede
introducir efectos de temperatura, carga del sistema, energia, version de
controladores o procesos de fondo. Conserva todos los checkpoints y sus
`session_id` para auditar esos posibles sesgos. Para conclusiones fuertes sobre
rendimiento, reporta sesiones o considera repetir de forma controlada los
benchmarks temporales afectados. La coherencia del codigo y de las versiones
se comprueba, pero no se puede garantizar identidad de las condiciones fisicas.

No ejecutes dos procesos de la campaña simultaneamente sobre el mismo
directorio. Espera a que el primero termine antes de ejecutar `--resume`.

## 5. Comprobar el estado y la salida final

```powershell
Get-Content .\benchmarks\checkpoints\v1.3.1\campaign_status.json -Raw | ConvertFrom-Json | Format-List
Get-ChildItem .\benchmarks\checkpoints\v1.3.1\runs -Recurse -Filter '*.json' | Measure-Object
Get-ChildItem .\benchmarks\results\final\v1.3.1 -File | Sort-Object Name | Format-Table Name,Length
```

El criterio para continuar con las tablas es que el programa haya terminado
con codigo 0 y el status sea `completed`.

```powershell
python scripts/generate_v1_3_figures_tables.py
```
