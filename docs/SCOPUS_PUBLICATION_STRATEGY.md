# Estrategia de Publicación Científica y Paquete de Replicación — MC DSL v1.3.1

Este documento organiza la estrategia editorial de **Monte Carlo DSL v1.3.1**. La version actual es una entrega de estabilizacion; no debe enviarse a una revista hasta completar la revalidacion indicada en `docs/REVALIDATION_REQUIRED_v1.3.1.md`.

> **Estado editorial:** borrador de investigacion; no listo para envio. Los resultados v1.3.0 son historicos y no validan la implementacion v1.3.1. Completar `docs/REVALIDATION_REQUIRED_v1.3.1.md` antes de seleccionar revista o someter el articulo.

---

## 1. Formulación Precisa de la Contribución Científica

### 1.1. Hipótesis y Contribución Investigada
> **Contribucion central:** Un metodo de planificacion de ejecucion para un DSL Monte Carlo que combina un modelo de coste lineal sobre cuatro indices del AST (Ridge restringida por conjunto activo KKT) y calibracion piloto para adaptar workers y agrupacion de tareas. La reduccion canonica conserva la identidad logica de las muestras; el payload retenido queda acotado respecto a N cuando se mantienen fijos el tamano maximo de tarea, el numero de workers y la ventana de tareas en vuelo.

### 1.2. Preguntas de Investigación (Research Questions)
- **RQ1 (Invariancia de Reproducibilidad frente a la Planificación):** ¿Permite la separación entre bloques lógicos indexados mediante un PRNG basado en contador (`Philox`) y tareas físicas —junto con la separación entre acumuladores de progreso físico y acumuladores canónicos de bloques lógicos— preservar resultados numéricos canonizados idénticos (igualdad SHA-256) al variar el número de workers, el tamaño físico de lote, el modo de planificación y el orden de finalización de tareas?
- **RQ2 (Precisión y Generalización del Modelo de Coste):** ¿Puede un modelo lineal compacto de cuatro indices del AST, calibrado sobre ocho modelos de entrenamiento y actualizado mediante un bloque piloto, estimar el coste por muestra en seis modelos no vistos e integrarse en el compilador/runtime? Las metricas de generalizacion deben regenerarse en v1.3.1.
- **RQ3 (Rendimiento, baselines y ablacion):** ¿En que condiciones el planificador hibrido reduce el tiempo extremo a extremo frente a la emulacion de configuracion historica v1.2.0 y frente a variantes fijas v1.3.1, usando los tamanos muestrales realmente ejecutados por celda?
- **RQ4 (Memoria retenida acotada bajo parametros fijos y fidelidad estadistica):** ¿Puede la compactacion por frontera, una ventana multiproceso de `2 * worker_count` y buffers de reordenamiento de 64 entradas acotar el payload retenido respecto a N manteniendo fijos el tamano maximo de tarea y workers, y cual es el error de cuantiles/histogramas, incluida la rama pareada?

---

## 2. Comparación Crítica con los Antecedentes Más Cercanos

El borrador `docs/paper/v1.3.1_manuscript.md` incluye una seccion de **Trabajos Relacionados**. La bibliografia y cada DOI deben verificarse contra la fuente primaria antes del envio:

1. **PPME — *A Domain-Specific Language and Editor for Parallel Particle Methods* (Karol, Nett, Castrillon & Sbalzarini, 2018, *ACM TOMACS*, `10.1145/3157898`)**:
   - *Qué resuelve PPME*: Modelo formal del lenguaje, inferencia de tipos, análisis estático y generación de código paralelo Fortran/MPI para simulaciones de partículas.
   - *Diferencia en MC DSL v1.3.1*: Nuestra contribución reside en el acoplamiento entre el modelo de coste guiado por el AST + calibración piloto y la invariancia de reproducibilidad estocástica frente a la planificación física de lotes Monte Carlo.
2. **STELLA — *A Python-based Domain-Specific Language for Simulations* (Mohr & Stefanovic, 2016, *Journal of Computational Science*, `10.1016/j.jocs.2016.03.005`)**:
   - *Qué resuelve STELLA*: Compilación especializada embebida en Python y comparación de tiempos con implementaciones de referencia.
   - *Diferencia en MC DSL v1.3.1*: MC DSL resuelve la decisión adaptativa de planificación (evitar la penalización de multiprocesamiento e IPC cuando domina la coordinación y agrupar bloques lógicos dinámicamente) garantizando invariancia numérica exacta entre configuraciones de ejecución.
3. **SimDSL — *A Domain-Specific Language for Large-Scale GPU-Resident Simulation* (Tasnim & Zhao, 2026, *Software*, `10.3390/software5030037`)**:
   - *Qué resuelve SimDSL*: Compilación de modelos de simulación hacia kernels residentes en GPU eliminando transferencias intermedias entre host y dispositivo.
   - *Diferencia en MC DSL v1.3.1*: En simulaciones Monte Carlo en CPU con ramas emparejadas y descartes dinamicos, el proyecto desacopla bloques logicos (`Philox`) y tareas fisicas y reduce los resultados en orden canonico.
4. **PRNGs Basados en Contador, Reducciones Paralelas Reproducibles, Cuantiles en Streaming y Planificación Guiada por Coste (Salmon et al., 2011; Demmel & Nguyen, 2013; Arteaga et al., 2014; Collange et al., 2015; Welford, 1962; Chan, Golub & LeVeque, 1979; Neumaier, 1974; Greenwald & Khanna, 2001; Karnin, Lang & Liberty, 2016; Dunning & Ertl, 2019; Giles, 2015; Kirsch et al., 2019; Adams et al., 2019; Ledoit & Wolf, 2004)**:
   - *Diferencia en MC DSL v1.3.1*: Integramos estas primitivas en el compilador/runtime de un DSL declarativo con compactacion por frontera y buffers limitados bajo parametros fijos; la fidelidad de cuantiles e histogramas debe revalidarse.

---

## 3. Protocolo de Selección de Revista en Scopus y Artefactos de Software Abierto

1. Antes del envío, verificar en la lista oficial de fuentes de Scopus (`scopus.com/sources`) que la revista objetivo tenga cobertura **activa** (*Active*).
2. Categorías editoriales adecuadas para este manuscrito:
   - *SoftwareX*, *Journal of Open Source Software (JOSS)*, *Science of Computer Programming*, *Simulation Modelling Practice and Theory (SIMPAT)*, *ACM Transactions on Modeling and Computer Simulation (TOMACS)*.
3. El repositorio incluye todos los requisitos de software cientifico reproducible que deben verificarse en la campana v1.3.1:
   - Licencia abierta explícita en la raíz: [`LICENSE`](../LICENSE) (MIT License).
   - Flujo de integración continua automatizado: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
   - Auditoria automatica de dependencias y metadatos de entorno para la nueva campana v1.3.1. El registro historico v1.3.0 no era conforme: NumPy 1.24.2 frente a `numpy>=2.0`. El ZIP recibido no contiene un repositorio Git utilizable; el commit debe registrarse desde el repositorio fuente.

---

## 4. Lista de Verificación del Paquete de Artefactos de Reproducción

| Artefacto | Ruta en el Repositorio | Descripción |
|---|---|---|
| Licencia y CI de software científico | `LICENSE`, `.github/workflows/ci.yml` | Licencia MIT y pipeline automatizado de pruebas e invariancia. |
| Codigo fuente del compilador, optimizador, modelo de coste y planificador | `src/montecarlo_dsl/` | Implementacion v1.3.1 (`optimizer.py`, `cost_model.py`, `scheduler.py`, `compiler.py`, `generator.py`, `statistics.py`, `server.py`). |
| Suite de validacion (mas de 200 tests automatizados; 41 focalizados verificados durante la preparacion, suite completa pendiente) | `tests/` | Incluye pruebas de contrato léxico/sintáctico/semántico, equivalencia diferencial, invariancia de eventos de progreso con bloques lógicos divididos, cuantiles ponderados y buffers pareados limitados y pruebas de orden de envio. |
| Auditoria historica v1.2.0 | `benchmarks/results/final/v1.2.0/`, `benchmarks/paper_results/final/v1.2.0/` | Datos historicos y auditoria; no implica que Exp. 0 ejecute el codigo fuente original. |
| Runner de revalidacion v1.3.1 | `benchmarks/run_v1_3_experiments.py` | Runner de los 10 experimentos; los tamanos muestrales varian por experimento y estan documentados en `docs/EXPERIMENTAL_PROTOCOL.md`. |
| Datos historicos v1.3.0 | `benchmarks/results/final/v1.3.0/` | Datos de la entrega anterior; no reutilizar como resultados v1.3.1. |
| Tablas y figuras historicas | `docs/tables/v1.3.0_*.md`, `docs/figures/v1.3.0_*.png` | Material de trazabilidad; regenerar desde los CSV v1.3.1 antes del envio. |
| Borrador cientifico v1.3.1 | `docs/paper/v1.3.1_manuscript.md` | Borrador con RQ1-RQ4; bibliografia y resultados requieren verificacion final. |
