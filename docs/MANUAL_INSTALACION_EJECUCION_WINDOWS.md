# Manual de instalacion y ejecucion en Windows

## 1. Requisitos

- Windows 11 recomendado.
- Python compatible con el proyecto (`>=3.10`).
- PowerShell.
- Navegador moderno.

Durante las pruebas de desarrollo se utilizo Python 3.14.3 en Windows. Si esa version esta instalada, los comandos siguientes pueden usar `py -3.14`.

## 2. Preparar el proyecto

Descomprima el ZIP final. En PowerShell:

```powershell
cd C:\ruta\al\proyecto\montecarlo_dsl_v1.1.0
```

Crear entorno virtual:

```powershell
py -3.14 -m venv .venv
```

Si no tiene Python 3.14 pero si otra version compatible:

```powershell
py -m venv .venv
```

Actualizar pip e instalar:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`uvicorn[standard]` instala el soporte WebSocket necesario.

## 3. Ejecutar la suite final

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Resultado esperado para esta version:

```text
205 passed
```

El tiempo depende del equipo.

## 4. Iniciar el servidor

```powershell
.\.venv\Scripts\python.exe -m uvicorn montecarlo_dsl.server:app --host 127.0.0.1 --port 8000
```

Debe aparecer un mensaje similar a:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

Mantenga esa ventana abierta.

## 5. Abrir la interfaz

En el navegador:

```text
http://127.0.0.1:8000/
```

Pruebas auxiliares:

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/docs
```

`/api/health` debe devolver:

```json
{"status":"ok"}
```

## 6. Prueba minima en la interfaz

Pegue:

```text
\model{x=a}
\normal a{0,1}
\iter{100000}
\response{avg,min,max}
```

Pulse **Ejecutar**. Debe observar progreso, resultados validos, workers, estadisticas e histograma. Los valores numericos varian si no fija semilla.

## 7. Detener el servidor

En la ventana de PowerShell donde se ejecuta Uvicorn:

```text
CTRL+C
```

## 8. Problemas comunes

### ERR_CONNECTION_REFUSED

El servidor no esta ejecutandose. Vuelva a iniciar Uvicorn y no cierre esa ventana.

### WebSocket no conecta

Compruebe que instalo el proyecto con `pip install -e ".[dev]"` y que `uvicorn[standard]` aparece entre las dependencias instaladas.

### Puerto 8000 ocupado

Use otro puerto:

```powershell
.\.venv\Scripts\python.exe -m uvicorn montecarlo_dsl.server:app --host 127.0.0.1 --port 8001
```

Despues abra `http://127.0.0.1:8001/`.

### Compilacion rechazada

Lea el panel de diagnosticos. Los errores incluyen codigo, linea y columna. Corrija primero el error indicado; la simulacion solo se crea si el programa pasa todas las fases de compilacion.

### HTTP 409

Ya existe una simulacion activa. Espere a que termine o use **Cancelar**.

## 9. Prueba API opcional con PowerShell

Con el servidor activo:

```powershell
$body = @{
    source = '\model{x=a}\normal a{0,1}\iter{100000}\response{avg,min,max}'
    seed = '12345'
} | ConvertTo-Json

$job = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/jobs" `
    -ContentType "application/json" `
    -Body $body

$job
```

Luego consulte:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/api/jobs/$($job.job_id)"
```
