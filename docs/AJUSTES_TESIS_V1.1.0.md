# Ajustes del escrito para Monte Carlo DSL v1.1.0

Este documento indica que sustituir en la version de 39 paginas del trabajo escrito
`Compilador DSL Monte Carlo(4).pdf` para que el texto corresponda con el proyecto
v1.1.0.

## 1. Seccion 4.1.3 - precision del alcance de validacion

Al final de 4.1.3 agregue este parrafo. No se presenta como una limitacion, sino como la
separacion entre formulacion probabilistica y verificacion computacional:

> A partir de esta diferencia resulta importante separar la formulacion probabilistica del
> modelo de aquellas comprobaciones que pueden realizarse automaticamente durante su
> procesamiento. Dentro de Monte Carlo DSL, el compilador verifica que la especificacion
> cumpla con la sintaxis y con las reglas semanticas definidas para el lenguaje, mientras
> que durante la ejecucion se incorporan indicadores numericos de precision sobre los
> resultados obtenidos, como el error estandar Monte Carlo de la media y un intervalo
> aproximado del 95 %. De esta manera, la herramienta mantiene de forma explicita los
> parametros y distribuciones definidos en el modelo y proporciona informacion adicional
> para evaluar la precision de la estimacion obtenida durante la simulacion.

## 2. Seccion 4.2.6 - separar compilacion HTTP de ejecucion WebSocket

Sustituya los dos ultimos parrafos de 4.2.6 por:

> Cuando el usuario solicita una nueva ejecucion, el cliente envia el modelo hacia el
> servidor mediante HTTP. Antes de crear un trabajo de simulacion, el backend completa
> las etapas de analisis lexico, analisis sintactico, construccion del AST, analisis
> semantico y generacion del script Python. Si se detecta un problema durante estas
> etapas, la solicitud termina mediante una respuesta HTTP de compilacion y la simulacion
> no llega a iniciarse. Solamente cuando el modelo ha sido compilado correctamente se
> crea el trabajo correspondiente y se devuelve al cliente el identificador utilizado para
> realizar su seguimiento.
>
> Por otra parte, una vez aceptado el trabajo, los resultados de ejecucion necesitan ser
> comunicados progresivamente hacia la interfaz conforme comienzan a encontrarse
> disponibles. Para este proposito se utiliza WebSocket, permitiendo mantener una
> conexion persistente asociada al identificador del trabajo. Podemos decir entonces que,
> HTTP se utiliza para recibir y compilar la solicitud inicial, mientras que WebSocket se
> emplea posteriormente para comunicar el avance y los eventos pertenecientes a la
> ejecucion de una simulacion que ya fue aceptada por el servidor.

## 3. Seccion 6.1.1 - sustituir la BNF completa

Reemplace el bloque actual por:

```bnf
<PROGRAMA> ::= <DECLARACION_MODELO> <CONFIGURACIONES> <ITERACIONES> <RESPUESTA> EOF
<DECLARACION_MODELO> ::= "\\model" "{" VARIABLE "=" <EXPRESION> "}"
<CONFIGURACIONES> ::= <CONFIGURACION> | <CONFIGURACION> <CONFIGURACIONES>
<CONFIGURACION> ::= <DIST_NORMAL> | <DIST_DISCRETA> | <DIST_UNIFORME>
<DIST_NORMAL> ::= "\\normal" VARIABLE "{" <NUMERO_SIGNADO> "," <NUMERO_SIGNADO> "}"
<DIST_DISCRETA> ::= "\\distrib" VARIABLE "{" <LISTA_PROBABILIDADES> "}"
<LISTA_PROBABILIDADES> ::= <PAR_PROBABILIDAD> | <PAR_PROBABILIDAD> "," <LISTA_PROBABILIDADES>
<PAR_PROBABILIDAD> ::= "{" <NUMERO_SIGNADO> "," NUMBER "}"
<DIST_UNIFORME> ::= "\\uniform" VARIABLE "{" <NUMERO_SIGNADO> "," <NUMERO_SIGNADO> "}"
<NUMERO_SIGNADO> ::= NUMBER | "+" NUMBER | "-" NUMBER
<ITERACIONES> ::= "\\iter" "{" NUMBER "}"
<RESPUESTA> ::= "\\response" "{" <LISTA_ESTADISTICOS> "}"
<LISTA_ESTADISTICOS> ::= <ESTADISTICO> | <ESTADISTICO> "," <LISTA_ESTADISTICOS>
<ESTADISTICO> ::= AVG | MIN | MAX
<EXPRESION> ::= <SUMA>
<SUMA> ::= <PRODUCTO> <COLA_SUMA>
<COLA_SUMA> ::= <OPERADOR_SUMA> <PRODUCTO> <COLA_SUMA> | epsilon
<OPERADOR_SUMA> ::= "+" | "-" | "\\pm"
<PRODUCTO> ::= <UNARIO> <COLA_PRODUCTO>
<COLA_PRODUCTO> ::= <OPERADOR_PRODUCTO> <UNARIO> <COLA_PRODUCTO> | <FACTOR_IMPLICITO> <COLA_PRODUCTO> | epsilon
<OPERADOR_PRODUCTO> ::= "*" | "/"
<UNARIO> ::= "+" <POTENCIA> | "-" <POTENCIA> | <POTENCIA>
<POTENCIA> ::= <ATOMO> | <ATOMO> "^" <NUMERO_SIGNADO>
<FACTOR_IMPLICITO> ::= <ATOMO_NO_NUMERICO> | <ATOMO_NO_NUMERICO> "^" <NUMERO_SIGNADO>
<ATOMO> ::= VARIABLE | NUMBER | <FRACCION> | <RAIZ> | <GRUPO>
<ATOMO_NO_NUMERICO> ::= VARIABLE | <FRACCION> | <RAIZ> | <GRUPO>
<FRACCION> ::= "\\frac" "{" <EXPRESION> "}" "{" <EXPRESION> "}"
<RAIZ> ::= "\\sqrt" "{" <EXPRESION> "}"
<GRUPO> ::= "{" <EXPRESION> "}"
```

Con este cambio desaparece `<DECIMAL_SIGNADO>` y queda unicamente
`<NUMERO_SIGNADO>`.

## 4. Seccion 6.1.1 - sustituir decisiones sintacticas afectadas

Cambie la viñeta de VARIABLE por:

> Una VARIABLE lexica corresponde a un identificador ASCII cuya primera posicion debe
> ser una letra y cuyas posiciones posteriores pueden contener letras, digitos o guion
> bajo. De esta manera, son validos identificadores como `a`, `precio`, `demanda2` y
> `tasa_interes`.

Cambie la viñeta de `\distrib` por:

> `\distrib` debe contener al menos un par `{valor,probabilidad}`. Cuando se declaran
> varios pares, estos se separan mediante comas; por ejemplo,
> `\distrib demanda{{10,0.5},{15,0.3},{20,0.2}}`.

Cambie la viñeta de multiplicacion implicita por:

> La multiplicacion implicita permite como factor derecho una VARIABLE, `\frac`, `\sqrt`
> o un grupo definido mediante llaves. Debido a que las variables pueden utilizar
> identificadores de varias posiciones, `2precio` se interpreta como `2 * precio`, mientras
> que `precio2` representa un unico identificador. Cuando se desea multiplicar dos
> variables con nombres descriptivos puede utilizarse de manera explicita el operador
> `*`, evitando cualquier ambiguedad entre el nombre de la variable y la operacion.

Sustituya la viñeta del exponente por:

> El exponente corresponde a un `<NUMERO_SIGNADO>` y no constituye una expresion
> general. Esta misma produccion se utiliza para representar los parametros numericos con
> signo de las distribuciones.

## 5. Seccion 6.1.2 - autómata AI

Sustituya la oracion que actualmente dice que AI acepta una unica letra por:

> Los demas automatas siguen el mismo principio de reconocimiento. AI reconoce
> identificadores cuya primera posicion corresponde a una letra ASCII y permite continuar
> con letras, digitos o guion bajo, generando posteriormente el token VARIABLE; AN
> reconoce los formatos numericos permitidos por el lenguaje; AS procesa los simbolos y
> operadores definidos; mientras que AW consume los espacios en blanco. Por su parte,
> AE reconoce exclusivamente `avg`, `min` y `max` cuando el analizador se encuentra
> dentro del contexto correspondiente a `\response`.

Si se incorpora el diagrama AI en anexos, use el nuevo archivo
`docs/diagramas/03_AI_variables.png` incluido en el proyecto v1.1.0.

## 6. Seccion 6.1.3 - cliente personalizado

Despues del parrafo que describe el editor, agregue:

> A partir de la version final del lenguaje, el editor tambien permite utilizar identificadores
> descriptivos para las variables, formados por una letra inicial seguida opcionalmente por
> letras, digitos o guion bajo. De igual forma, las distribuciones discretas presentan de
> manera explicita la separacion entre sus diferentes pares de valor y probabilidad,
> manteniendo una correspondencia directa con la gramática utilizada por el parser.

Sustituya los parrafos que mezclan errores y WebSocket por:

> Cuando el usuario solicita la ejecucion de un modelo, JavaScript realiza una peticion
> HTTP al endpoint `POST /api/jobs`. Antes de aceptar el trabajo, el backend completa las
> etapas correspondientes al lexer, parser, analisis semantico y generacion del script. Si
> alguna de estas etapas produce un diagnostico, la respuesta de compilacion se devuelve
> directamente mediante HTTP y el cliente la presenta dentro de la interfaz. En cambio,
> cuando la compilacion finaliza correctamente, el servidor devuelve un identificador
> `job_id` y solamente a partir de ese momento el cliente establece la conexion WebSocket
> mediante `/ws/jobs/{id}` para seguir la ejecucion.
>
> Por medio de WebSocket, el servidor transmite el avance producido por los diferentes
> lotes. Cuando el modelo contiene el operador `\pm`, la interfaz conserva de forma
> independiente la rama positiva y la rama negativa, permitiendo seleccionar cualquiera de
> ellas para consultar sus resultados validos, descartados, estadisticos e histograma. De
> igual forma, cada rama presenta indicadores numericos de precision asociados con la
> estimacion de la media, incluyendo el error estandar Monte Carlo y un intervalo
> aproximado del 95 %.

## 7. Seccion 6.1.4 - servicio de compilacion

Sustituya los primeros tres parrafos, hasta el flujo del compilador, por:

> Para atender el servicio de compilacion se establecio una frontera explicita entre las
> etapas encargadas de transformar el modelo y aquellas responsables de ejecutar la
> simulacion. Cuando el backend recibe una solicitud mediante `POST /api/jobs`, primero
> ejecuta de manera completa el analisis lexico, analisis sintactico, construccion del AST,
> analisis semantico y generacion del script Python. Solamente si todas estas etapas
> terminan correctamente se crea posteriormente el trabajo de ejecucion.
>
> De esta manera, los errores correspondientes a las fases lexico, sintactico, semantico o
> de generacion son devueltos como diagnosticos de compilacion mediante HTTP antes de
> iniciar cualquier subproceso. El proceso utilizado por el servicio puede representarse
> mediante la siguiente secuencia:
>
> **Modelo DSL -> Lexer -> Tokens -> Parser -> AST -> Analisis semantico -> Generador
> -> Script Python -> Creacion del trabajo runtime**

Despues del parrafo que explica la transformacion del AST, agregue:

> Debido a que el lenguaje admite identificadores descriptivos, el generador realiza ademas
> un mapeo interno de los nombres definidos mediante el DSL hacia identificadores Python
> controlados. Es por esto que, el texto utilizado por el usuario como nombre de una
> variable no se incorpora directamente dentro del script generado, evitando colisiones
> incluso cuando dicho nombre coincide con una palabra reservada del lenguaje Python.

Sustituya el parrafo actual sobre `\pm` por:

> Por otra parte, cuando la expresion contiene el operador `\pm`, el generador construye
> dos ramas independientes correspondientes a las alternativas positiva y negativa. Esta
> separacion se conserva durante toda la etapa de ejecucion, de manera que cada rama
> mantiene posteriormente sus propios resultados, estadisticos e histograma en lugar de
> integrarse dentro de una sola distribucion.

## 8. Seccion 6.1.5 - servicio concurrente

Despues del segundo parrafo, agregue el siguiente texto para documentar el canal del
subproceso:

> La comunicacion entre el subproceso y el administrador de trabajos utiliza `stdout` como
> un canal estructurado de eventos. Para evitar que alguna salida auxiliar pueda
> confundirse con los mensajes consumidos por el backend, cada evento valido se emite
> utilizando el prefijo reservado `MC_DSL_EVENT:` seguido de su objeto JSON. El
> administrador procesa unicamente las lineas que contienen dicho prefijo y mantiene por
> separado cualquier otra salida diagnostica; de igual forma, `stderr` se conserva como un
> canal independiente para los mensajes de error del subproceso.

Reemplace desde "Este comportamiento resulta particularmente importante cuando el
modelo utiliza el operador \pm" hasta antes del parrafo de WebSocket por:

> Este comportamiento adquiere especial importancia cuando el modelo utiliza el operador
> `\pm`. En estos casos una misma iteracion produce hasta dos resultados candidatos,
> correspondientes a las ramas positiva y negativa de la expresion. Para una simulacion de
> 1 000 000 de iteraciones pueden producirse hasta 2 000 000 de candidatos; sin embargo,
> ambas ramas permanecen identificadas de manera independiente durante el
> procesamiento. Cada una conserva su propio conteo de resultados validos y descartados,
> asi como su promedio, minimo, maximo e histograma cuando estos corresponden a la
> respuesta solicitada.
>
> Ademas de los estadisticos seleccionados mediante `\response`, el runtime calcula por
> rama algunos indicadores numericos destinados a describir la precision de la estimacion
> obtenida. Entre estos se encuentran la desviacion estandar muestral, el error estandar
> Monte Carlo de la media y un intervalo aproximado del 95 % para dicha media. Podemos
> decir entonces que, el operador `\pm` no solamente genera dos alternativas durante el
> calculo, sino que estas permanecen separadas hasta la presentacion final de los
> resultados.
>
> La contabilizacion global de candidatos puede comprobarse mediante la relacion:
> **Resultados validos totales + Resultados descartados totales = Resultados candidatos**.
> De manera complementaria, cada rama conserva su propia contabilizacion y su propia
> distribucion de resultados.

## 9. Seccion 7 - conclusiones

Sustituya el parrafo que comienza con "En lo que respecta a la ejecucion" por:

> En lo que respecta a la ejecucion, la arquitectura concurrente permite dividir las
> simulaciones en lotes de 25 000 iteraciones y distribuirlos entre un maximo de cuatro
> workers mediante `ProcessPoolExecutor`. Los resultados generados por estos procesos
> son integrados posteriormente conservando la identidad de cada rama cuando el modelo
> utiliza `\pm`. De esta manera, las alternativas positiva y negativa mantienen sus propios
> estadisticos e histogramas, mientras que los conteos globales permiten verificar la
> totalidad de resultados candidatos procesados. De igual forma, la reproducibilidad fue
> comprobada mediante ejecuciones independientes realizadas con un mismo modelo, una
> misma semilla y las mismas condiciones de ejecucion.

Sustituya el parrafo que comienza con "Sin embargo, el alcance actual tambien presenta
algunas limitaciones" y el siguiente parrafo de mejoras por este unico parrafo:

> A partir de la arquitectura desarrollada, la evolucion de Monte Carlo DSL puede
> realizarse mediante la incorporacion controlada de nuevas construcciones sin modificar
> el principio general utilizado por el compilador. La gramática, los autómatas, el parser,
> el analisis semantico y el generador mantienen responsabilidades separadas, mientras
> que el runtime conserva una estructura independiente para el procesamiento concurrente
> y la presentacion de resultados. Esta organizacion permite ampliar posteriormente las
> distribuciones, operadores, estadisticos o mecanismos de despliegue siguiendo el mismo
> flujo de reconocimiento, validacion, generacion y ejecucion utilizado por la version
> desarrollada.

Antes del parrafo final de conclusion, agregue:

> La revision final del sistema permitio ademas reforzar diferentes aspectos de la
> correspondencia entre la especificacion y la implementacion. La gramática utiliza una
> unica produccion para los numeros con signo, las distribuciones discretas cuentan con
> separadores explicitos, los identificadores permiten nombres descriptivos, los errores de
> compilacion permanecen separados de los eventos de ejecucion y el operador `\pm`
> conserva sus ramas de resultados de manera independiente. Estas caracteristicas fueron
> incorporadas a la suite automatizada, alcanzando un total de 205 pruebas aprobadas.

## 10. Anexo de pruebas

Cambie el texto de 195 a **205 pruebas aprobadas** y genere una nueva captura de
`python -m pytest` con la version 1.1.0.

## 11. Figuras que deben renovarse

Debido a que la sintaxis del ejemplo y la interfaz cambiaron, renueve las siguientes
capturas:

- Figura 6.3: interfaz inicial, para mostrar identificadores descriptivos y `\distrib` con
  comas entre pares.
- Figura 6.4: resultado final, para mostrar selector de Rama + / Rama -, MCSE e intervalo
  aproximado del 95 %.
- Figura 6.6: modelo DSL y script generado, debido al mapeo de identificadores a nombres
  Python internos.
- Figura 6.7: traduccion de la expresion, para mostrar las dos ramas generadas.
- Figura 6.8: `ProcessPoolExecutor`, porque el runtime v1.1.0 administra estructuras por
  rama.
- Figura 6.9: histograma progresivo de una rama seleccionada.
- Figura 6.10: reproducibilidad, usando el proyecto v1.1.0.
- Figura del anexo de Pytest: 205 pruebas.

Las Figuras 6.1 (dispatcher) y 6.2 (AD de directivas) pueden conservarse porque esos
componentes no cambiaron de principio. Si se incorpora AI en anexos, utilice el diagrama
nuevo incluido en `docs/diagramas/03_AI_variables.png`.
