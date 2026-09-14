# BNF definitiva del Monte Carlo DSL

Esta especificacion refleja el comportamiento implementado por `lexer.py` y `parser.py`
en la version 1.1.0. Las restricciones que dependen de valores (por ejemplo, limite de
iteraciones, probabilidades que suman 1 o maximo un `\pm`) pertenecen al analisis
semantico y no se fuerzan en esta BNF.

## Simbolo inicial

`<PROGRAMA>`

## Producciones P

```bnf
<PROGRAMA> ::= <DECLARACION_MODELO>
               <CONFIGURACIONES>
               <ITERACIONES>
               <RESPUESTA>
               EOF

<DECLARACION_MODELO> ::= "\model" "{" VARIABLE "=" <EXPRESION> "}"

<CONFIGURACIONES> ::= <CONFIGURACION>
                    | <CONFIGURACION> <CONFIGURACIONES>

<CONFIGURACION> ::= <DIST_NORMAL>
                  | <DIST_DISCRETA>
                  | <DIST_UNIFORME>

<DIST_NORMAL> ::= "\normal" VARIABLE "{" <NUMERO_SIGNADO> "," <NUMERO_SIGNADO> "}"

<DIST_DISCRETA> ::= "\distrib" VARIABLE "{" <LISTA_PROBABILIDADES> "}"

<LISTA_PROBABILIDADES> ::= <PAR_PROBABILIDAD>
                         | <PAR_PROBABILIDAD> "," <LISTA_PROBABILIDADES>

<PAR_PROBABILIDAD> ::= "{" <NUMERO_SIGNADO> "," NUMBER "}"

<DIST_UNIFORME> ::= "\uniform" VARIABLE "{" <NUMERO_SIGNADO> "," <NUMERO_SIGNADO> "}"

<NUMERO_SIGNADO> ::= NUMBER
                   | "+" NUMBER
                   | "-" NUMBER

<ITERACIONES> ::= "\iter" "{" NUMBER "}"

<RESPUESTA> ::= "\response" "{" <LISTA_ESTADISTICOS> "}"

<LISTA_ESTADISTICOS> ::= <ESTADISTICO>
                       | <ESTADISTICO> "," <LISTA_ESTADISTICOS>

<ESTADISTICO> ::= AVG | MIN | MAX

<EXPRESION> ::= <SUMA>

<SUMA> ::= <PRODUCTO> <COLA_SUMA>

<COLA_SUMA> ::= <OPERADOR_SUMA> <PRODUCTO> <COLA_SUMA>
              | epsilon

<OPERADOR_SUMA> ::= "+" | "-" | "\pm"

<PRODUCTO> ::= <UNARIO> <COLA_PRODUCTO>

<COLA_PRODUCTO> ::= <OPERADOR_PRODUCTO> <UNARIO> <COLA_PRODUCTO>
                  | <FACTOR_IMPLICITO> <COLA_PRODUCTO>
                  | epsilon

<OPERADOR_PRODUCTO> ::= "*" | "/"

<UNARIO> ::= "+" <POTENCIA>
           | "-" <POTENCIA>
           | <POTENCIA>

<POTENCIA> ::= <ATOMO>
             | <ATOMO> "^" <NUMERO_SIGNADO>

<FACTOR_IMPLICITO> ::= <ATOMO_NO_NUMERICO>
                     | <ATOMO_NO_NUMERICO> "^" <NUMERO_SIGNADO>

<ATOMO> ::= VARIABLE
          | NUMBER
          | <FRACCION>
          | <RAIZ>
          | <GRUPO>

<ATOMO_NO_NUMERICO> ::= VARIABLE
                      | <FRACCION>
                      | <RAIZ>
                      | <GRUPO>

<FRACCION> ::= "\frac" "{" <EXPRESION> "}" "{" <EXPRESION> "}"

<RAIZ> ::= "\sqrt" "{" <EXPRESION> "}"

<GRUPO> ::= "{" <EXPRESION> "}"
```

## Contrato lexico de VARIABLE

`VARIABLE` representa un identificador ASCII con la forma:

```text
[A-Za-z][A-Za-z0-9_]*
```

Por ejemplo, son identificadores validos `a`, `precio`, `demanda2` y `tasa_interes`.
Los nombres se transforman internamente a identificadores Python seguros durante la
generacion, por lo que incluso un nombre que coincida con una palabra reservada de
Python no se incorpora de manera literal al script generado.

## Decisiones sintacticas fijadas

- El lenguaje usa llaves `{}` para agrupacion; no usa parentesis ni corchetes.
- `avg`, `min` y `max` son tokens estadisticos propios dentro de `\response`.
- Debe existir al menos una configuracion de distribucion.
- `\distrib` debe contener al menos un par `{valor,probabilidad}` y los pares consecutivos se separan mediante comas.
- Los valores de distribuciones pueden llevar un signo opcional; la probabilidad no.
- `\response` debe contener al menos un estadistico.
- La precedencia es: potencia > signo unario > producto/division/multiplicacion implicita > suma/resta/`\pm`.
- `+`, `-` y `\pm` se procesan de izquierda a derecha al mismo nivel sintactico.
- `*` y `/` se procesan de izquierda a derecha.
- El exponente utiliza el mismo no terminal `<NUMERO_SIGNADO>` empleado por los parametros numericos de las distribuciones.
- La multiplicacion implicita permite un factor no numerico a la derecha. Con identificadores de varias posiciones, `2precio` representa `2 * precio`, mientras que `precio2` representa un unico identificador. Para multiplicar dos variables sin ambiguedad se recomienda `precio*demanda`.
- Un grupo con llaves afecta la precedencia pero no crea un nodo AST independiente.
- La duplicacion de estadisticos y el limite de un solo `\pm` se validan en la fase semantica, no en el parser.

## Extension v1.2.0 de `\response`

La version 1.2.0 mantiene `avg`, `min` y `max`, y agrega estadisticos para validacion
publicable:

```text
<ESTADISTICO> ::= AVG | MIN | MAX | VAR | STD | COUNT | VALID_RATE | DISCARD_RATE | P05 | P50 | P95
```

Donde `min` y `max` representan extremos muestrales observados. `valid_rate` y
`discard_rate` se calculan sobre resultados candidatos de cada rama. `p05`, `p50` y
`p95` son cuantiles muestrales.
