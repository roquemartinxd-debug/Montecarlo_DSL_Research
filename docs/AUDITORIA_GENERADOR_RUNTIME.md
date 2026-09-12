# Auditoria del generador y runtime - v1.1.0

## Resultado

**APROBADO.** La version 1.1.0 refuerza la traduccion de identificadores, separa las
ramas de `\pm` y agrega diagnosticos de precision Monte Carlo.

## Nombres de variables

El DSL permite identificadores descriptivos. Antes de generar Python, cada identificador
se transforma a un nombre interno seguro como `_mc_var_0`. Por ello, nombres como
`class` o `for` pueden ser validos en el DSL sin convertirse en sintaxis Python invalida.

## AST -> NumPy

El visitor traduce operaciones aritmeticas, potencia, fraccion, raiz y referencias de
variables sobre arreglos NumPy. No se utiliza `eval` ni `exec` sobre texto proporcionado
por el usuario.

## `\pm`

Un unico `\pm` produce exactamente dos ramas: `plus` y `minus`. Las ramas se procesan
y acumulan por separado durante toda la simulacion.

Cada rama mantiene:

- resultados validos y descartados;
- suma, minimo, maximo y promedio solicitado;
- histograma propio;
- desviacion estandar muestral;
- MCSE de la media;
- intervalo aproximado de 95 % de la media;
- MCSE relativo.

No se publica un promedio ni histograma agregado que mezcle ambas ramas.

## Dominios numericos

El runtime filtra `NaN` e infinitos por rama mediante `np.isfinite`. Las protecciones
previas contra infinito, underflow estructural, colapso de limites uniformes y colision de
valores discretos en `float64` se mantienen.

## Batching y reproducibilidad

- los lotes cubren exactamente `TOTAL_ITERATIONS`;
- `workers = min(MAX_WORKERS, cantidad_de_lotes)`;
- cada lote deriva su semilla de `MASTER_SEED` y del indice mediante BLAKE2b;
- se conserva la configuracion necesaria para repetir una ejecucion.

## Protocolo

Los eventos se emiten como `MC_DSL_EVENT:{JSON}`. El prefijo separa el canal
estructurado de cualquier salida auxiliar del subproceso.

## Verificacion

La suite global de la version 1.1.0 contiene **205 pruebas** e incluye contratos de
separacion de ramas, MCSE/intervalos, aislamiento de stdout y compilacion previa al
runtime.
