# Auditoria semantica y tabla de simbolos - v1.1.0

## Resultado

**APROBADO.** La implementacion actual de `semantic.py` y `symbols.py` es coherente con las
reglas semanticas definidas para el DSL. No fue necesario modificar la logica del
analizador semantico ni de la tabla de simbolos.

La auditoria agrego pruebas de contrato para limites, tolerancias, dominios constantes,
orden de resultados, spans de diagnostico, inmutabilidad y referencias de simbolos.

## Arquitectura del analisis semantico

`SemanticAnalyzer` trabaja en tres pasadas:

1. **Registro de declaraciones**: variable resultado y variables aleatorias.
2. **Referencias y expresion**: resolucion de variables y validaciones de dominio que
   pueden determinarse en compilacion.
3. **Reglas globales**: distribuciones, iteraciones, estadisticos y `\\pm`.

Si existe al menos un diagnostico semantico se lanza `CompilationError`. La tabla de
simbolos solo se congela y entrega dentro de `ValidatedProgram` cuando el programa es
semanticamente valido.

## Diagnosticos semanticos implementados

| Codigo | Regla |
|---|---|
| SEM001 | La variable resultado no puede declararse como variable aleatoria. |
| SEM002 | La variable resultado no puede aparecer en su propia expresion. |
| SEM003 | Toda variable usada por el modelo debe tener una distribucion. |
| SEM004 | Una variable aleatoria no puede tener mas de una distribucion. |
| SEM005 | Una distribucion declarada debe utilizarse en el modelo. |
| SEM006 | Debe existir al menos una variable aleatoria valida. |
| SEM010 | La desviacion estandar de `\\normal` debe ser mayor que cero. |
| SEM011 | En `\\uniform`, minimo < maximo. |
| SEM012 | Cada probabilidad discreta debe estar entre 0 y 1. |
| SEM013 | Las probabilidades discretas deben sumar 1, con tolerancia absoluta `1e-12`. |
| SEM014 | Un valor de una distribucion discreta no puede repetirse. |
| SEM020 | `\\iter` debe escribirse sin parte decimal. |
| SEM021 | `\\iter` debe estar entre 1 y 1,000,000 inclusive. |
| SEM030 | `\\response` no permite estadisticos repetidos. |
| SEM040 | El modelo puede contener como maximo un operador `\\pm`. |
| SEM050 | Un denominador constante no puede ser cero. Aplica a `/` y `\\frac`. |
| SEM051 | Un argumento constante de `\\sqrt` no puede ser negativo. |
| SEM052 | La potencia constante `0` con exponente negativo no esta definida. |

Los saltos numericos entre codigos son intencionales: la implementacion no define un
codigo para cada entero entre SEM001 y SEM052.

## Observaciones de capas lexer/parser/semantica

- Una probabilidad negativa escrita en el codigo fuente no alcanza SEM012: el parser
  exige que la probabilidad sea un `NUMBER` sin signo. El chequeo `probability < 0` en
  semantica funciona como defensa adicional para un AST construido directamente.
- Los exponentes del DSL son `NUMBER` con signo opcional; no son expresiones ni
  variables. Esta restriccion pertenece al parser.
- Los errores de dominio que dependen de variables aleatorias se dejan para ejecucion.
  Ejemplos: division por una variable que puede valer cero o `\\sqrt{a}` cuando `a`
  puede ser negativa.
- Una potencia con base negativa y exponente fraccionario constante no se rechaza en
  esta fase; queda sujeta al comportamiento numerico de ejecucion. Esta observacion se
  revisara de nuevo durante la auditoria del generador/runtime.

## Tabla de simbolos

La tabla contiene dos clases de simbolos:

- `OUTPUT_VARIABLE`: variable resultado de `\\model`.
- `RANDOM_VARIABLE`: variable con `\\normal`, `\\distrib` o `\\uniform`.

Cada `Symbol` conserva:

- nombre;
- clase de simbolo;
- span de declaracion;
- clase de distribucion (`NONE`, `NORMAL`, `DISCRETE`, `UNIFORM`);
- nodo AST de la distribucion, cuando corresponde;
- spans de todas las referencias validas en la expresion.

### Inmutabilidad

- `Symbol` es un `dataclass(frozen=True)`.
- `SymbolTable` es un `dataclass(frozen=True)`.
- El mapa interno se expone mediante `MappingProxyType`.
- Las referencias se congelan como una tupla.

Las pruebas de contrato verifican que ni el mapa ni los simbolos puedan modificarse una
vez validado el programa.

## `ValidatedProgram`

Al terminar correctamente, semantica entrega:

- AST validado;
- tabla de simbolos inmutable;
- variable resultado;
- variables aleatorias en orden de declaracion;
- iteraciones como `int`;
- estadisticos solicitados en el orden escrito;
- indicador de presencia de un unico `\\pm`.

## Pruebas agregadas

Se agregaron:

- **19** pruebas en `tests/test_semantic_contract.py`.
- **10** pruebas en `tests/test_symbol_table_contract.py`.

Cubren, entre otros puntos:

- limites inclusivos de `\\iter`;
- iteraciones con parte decimal;
- desviacion estandar positiva minima;
- uniformes con limites negativos;
- probabilidades 0 y 1;
- tolerancia de suma `1e-12`;
- valores discretos equivalentes (`1` y `1.0`) como duplicados;
- variables declaradas pero no usadas;
- conflicto con variable resultado;
- orden de `\\response`;
- bandera de `\\pm`;
- dominios constantes `/`, `\\sqrt` y potencia;
- spans de diagnosticos;
- preservacion exacta de `Decimal`;
- contenido, tipos, spans, referencias e inmutabilidad de la tabla de simbolos.

## Estado actual de pruebas

Las reglas semanticas forman parte de una suite global de **205 pruebas aprobadas**.
Los contratos dedicados a esta capa se mantienen en `test_semantic_contract.py` (19) y
`test_symbol_table_contract.py` (10), ademas de las pruebas funcionales existentes.

## Conclusion

No se encontro una discrepancia que justifique cambiar `semantic.py` o `symbols.py`.
La capa semantica queda integrada con la frontera de compilacion de v1.1.0; cualquier `CompilationError` semantico se produce antes de crear el trabajo runtime.
