# Documentacion tecnica - Monte Carlo DSL v1.1.0

## 1. Alcance de la version

Monte Carlo DSL implementa un compilador y runtime para modelos de simulacion Monte Carlo.
La version 1.1.0 incorpora los ajustes derivados de la revision tecnica final:

- identificadores con patron `[A-Za-z][A-Za-z0-9_]*`;
- lista discreta con separador explicito entre pares;
- una unica categoria sintactica `<NUMERO_SIGNADO>`;
- compilacion completa antes de crear un trabajo runtime;
- protocolo estructurado de eventos aislado de cualquier salida auxiliar;
- ramas de `\pm` mantenidas separadas en estadisticas e histogramas;
- diagnosticos numericos de precision Monte Carlo por rama.

## 2. Pipeline del compilador

El pipeline es:

`source -> lexer -> tokens -> parser -> AST -> analisis semantico -> generador -> script Python`

Las fases LEX, SYN, SEM y GEN concluyen dentro de `compile_source()` antes de que el
servidor cree un trabajo de ejecucion. Si alguna falla, `POST /api/jobs` responde con HTTP
422 y no existe un subproceso de simulacion asociado a esa solicitud.

El modulo `src/montecarlo_dsl/compiler.py` contiene la frontera explicita entre compilacion
y ejecucion. El resultado de compilacion es un `CompiledSimulation` inmutable con el
programa validado, el script generado y la semilla efectiva.

## 3. Lexer

El lexer conserva el despacho q0/qP/qR y los subautomatas AD, AI, AN, AS, AW y AE.

### Identificadores

AI reconoce identificadores con la forma:

`[A-Za-z][A-Za-z0-9_]*`

Ejemplos validos: `a`, `precio`, `demanda_2026`, `tasa_interes`.

Esto permite nombres descriptivos sin introducir directamente el identificador del usuario
en el codigo Python generado.

### Directivas

AD reconoce `\model`, `\normal`, `\distrib`, `\uniform`, `\iter`, `\response`, `\frac`,
`\sqrt` y `\pm`. El lexer comprueba el limite de la directiva para impedir que una palabra
mas larga sea aceptada por prefijo.

## 4. Parser y BNF

La BNF normativa se encuentra en `docs/BNF_DEFINITIVA.md`.

Los parametros con signo y los exponentes comparten la produccion sintactica:

`<NUMERO_SIGNADO> ::= NUMBER | "+" NUMBER | "-" NUMBER`

Las funciones internas del parser pueden producir representaciones diferentes de ese
mismo patron - `Decimal` para parametros o nodos AST para exponentes - sin que eso
requiera duplicar el simbolo no terminal en la gramática.

En `\distrib` cada par conserva sus llaves y los pares consecutivos se separan mediante
coma:

`\distrib demanda{{10,0.5},{15,0.3},{20,0.2}}`

El parser rechaza tanto una coma sin un par posterior como dos pares adyacentes sin el
separador requerido.

## 5. Analisis semantico

El analizador semantico trabaja sobre el AST y comprueba, entre otras reglas:

- distribucion definida para toda variable aleatoria utilizada;
- parametros validos para `\normal` y `\uniform`;
- probabilidades discretas dentro de rango y suma aproximadamente igual a 1;
- valores discretos no duplicados;
- limite de iteraciones;
- estadisticos duplicados;
- cardinalidad permitida de `\pm`;
- errores matematicos constantes que pueden resolverse antes de ejecutar.

Estas comprobaciones forman el contrato formal y computacional del DSL. Los parametros
y distribuciones especificados por el usuario se ejecutan exactamente bajo ese contrato.

## 6. Generador Python y nombres seguros

`PythonCodeGenerator` recibe exclusivamente un programa validado. Los identificadores
del DSL se mapean a nombres internos seguros (`_mc_var_0`, `_mc_var_1`, etc.). Por tanto,
un identificador valido en el DSL, incluso si coincide con una palabra reservada de
Python, nunca se inserta directamente como nombre de variable en el script generado.

El visitor traduce las expresiones a NumPy y no utiliza `eval` ni `exec` sobre el texto
fuente del usuario.

## 7. Semantica de `\pm`

Un `\pm` produce dos ramas independientes: `plus` y `minus`. Cada lote calcula, filtra y
acumula ambas ramas por separado.

Para cada rama se conservan independientemente:

- resultados validos y descartados;
- suma, minimo y maximo;
- promedio cuando fue solicitado;
- histograma acumulado;
- desviacion estandar muestral;
- error estandar Monte Carlo de la media (`mcse_mean`);
- intervalo aproximado del 95 % para la media (`ci95_mean`);
- error estandar relativo (`relative_mcse`).

Los contadores superiores `valid_results` y `discarded_results` son totales de candidatos
para conservar la identidad de contabilizacion. Los estadisticos e histogramas no se
combinan entre ramas.

## 8. Batching, concurrencia y reproducibilidad

El runtime divide `TOTAL_ITERATIONS` en lotes de `BATCH_SIZE` y usa:

`workers = min(MAX_WORKERS, numero_de_lotes)`

Cada lote obtiene una semilla determinista derivada de `MASTER_SEED` y de su indice
mediante BLAKE2b. Los lotes se ejecutan con `ProcessPoolExecutor` y se integran de
manera determinista por indice para las operaciones globales sensibles al orden.

## 9. Protocolo subproceso -> backend

El script generado reserva un canal de eventos en stdout con el prefijo:

`MC_DSL_EVENT:`

Cada evento valido tiene la forma conceptual:

`MC_DSL_EVENT:{...JSON...}`

`JobManager` ignora para el protocolo cualquier linea que no tenga ese prefijo y la
conserva unicamente en una cola de diagnosticos de stdout. Una linea prefijada debe ser
JSON valido y contener un tipo de evento permitido; de lo contrario, la ejecucion falla de
forma controlada. `stderr` se consume por separado.

Esto evita que una salida auxiliar accidental sea interpretada como un evento de la
simulacion.

## 10. HTTP y WebSocket

`POST /api/jobs` tiene dos responsabilidades antes de aceptar un trabajo:

1. ejecutar LEX/SYN/SEM/GEN;
2. crear el trabajo runtime solo si la compilacion termino correctamente.

Los errores de compilacion usan HTTP 422. Despues de una respuesta 202 con `job_id`, el
cliente abre `/ws/jobs/{id}` para recibir eventos de ejecucion: inicio, lotes, finalizacion,
error runtime o cancelacion.

Por tanto, los errores de compilacion y los eventos de ejecucion pertenecen a canales y
momentos distintos del ciclo de vida.

## 11. Cliente web

La interfaz permite:

- editar el DSL con numeracion de lineas;
- proporcionar semilla opcional;
- iniciar y cancelar trabajos;
- seguir progreso y conteos globales;
- seleccionar `Rama +` o `Rama -` cuando existe `\pm`;
- consultar estadisticos e indicadores de precision de la rama seleccionada;
- visualizar su histograma acumulado.

## 12. Verificacion automatizada

La version 1.1.0 contiene **205 pruebas automatizadas**. La bateria cubre lexer, parser,
AST, semantica, tabla de simbolos, generador, runtime concurrente, aislamiento del
protocolo stdout, frontera compilacion/runtime, FastAPI, WebSocket y frontend.

El comando de verificacion es:

`python -m pytest`
