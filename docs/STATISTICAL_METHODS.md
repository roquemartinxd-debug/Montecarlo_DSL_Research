# Metodos estadisticos de Monte Carlo DSL v1.2.0

## Semantica probabilistica

Un programa DSL define una variable aleatoria de salida:

```text
Y = f(X1, X2, ..., Xk)
```

Cada variable `Xi` se declara con una distribucion `\normal`, `\uniform` o `\distrib`.
En v1.2.0 las variables se tratan como independientes por defecto. Cada iteracion genera
una muestra pseudoaleatoria independiente bajo la semilla maestra y el indice de lote.

Cuando la expresion contiene `\pm`, el runtime genera dos ramas emparejadas para las
mismas entradas aleatorias:

```text
Y_plus  = f_plus(X)
Y_minus = f_minus(X)
Delta   = Y_plus - Y_minus
```

Por ello la diferencia `Delta` se reporta como estadistico pareado y no como comparacion
de muestras independientes.

## Resultados finitos y media condicional

El runtime no propaga `NaN` ni infinitos a los estimadores principales. La politica vigente es:

```text
nonfinite_policy = discard_and_report
```

Por tanto, los estimadores de media, varianza, desviacion y cuantiles se calculan sobre
resultados finitos. Si hay descartes, la media reportada debe interpretarse como una
media condicional sobre resultados validos. El sistema reporta tambien:

- `candidate_results`
- `valid_results`
- `discarded_results`
- `valid_rate`
- `discard_rate`
- intervalos Wilson de 95 % para tasas de validos y descartes

## Estimadores

La respuesta extendida acepta:

```text
avg, min, max, var, std, count, valid_rate, discard_rate, p05, p50, p95
```

`min` y `max` son extremos muestrales observados. No deben interpretarse como limites
teoricos del soporte de la distribucion.

La varianza muestral se define con `n - 1` grados de libertad. El error estandar Monte
Carlo de la media se calcula como:

```text
MCSE(mean) = s / sqrt(n)
```

El intervalo `ci95_mean` es una aproximacion normal:

```text
mean +/- 1.96 * MCSE
```

Para muestras pequenas debe tratarse como una aproximacion, no como garantia exacta.

## Acumulacion numerica

La version 1.2.0 agrega un nucleo estadistico con `RunningStats`, acumulacion incremental
Welford y combinacion Chan-Golub-LeVeque. El runtime generado mantiene estados por lote
para media y varianza, y combina lotes de forma determinista por indice de lote para reducir
la dependencia del orden de finalizacion de procesos concurrentes.
