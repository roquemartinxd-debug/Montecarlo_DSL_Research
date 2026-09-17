# Guia de usuario del DSL

## 1. Estructura obligatoria

Un programa contiene:

```text
\model{variable=expresion}
una o mas distribuciones
\iter{numero}
\response{estadisticos}
```

Ejemplo:

```text
\model{resultado=precio-demanda}
\normal precio{100,4}
\uniform demanda{20,30}
\iter{100000}
\response{avg,min,max}
```

## 2. Variables e identificadores

Los identificadores comienzan con una letra ASCII y pueden continuar con letras,
digitos o guion bajo. Ejemplos validos:

```text
a
precio
demanda2
tasa_interes
```

La forma lexica es `[A-Za-z][A-Za-z0-9_]*`. Los nombres del DSL no se insertan de
manera literal en el script Python; el generador los transforma a nombres internos
seguros.

## 3. Distribucion normal

```text
\normal precio{media,desviacion}
```

La desviacion debe ser mayor que cero.

## 4. Distribucion uniforme

```text
\uniform demanda{minimo,maximo}
```

El minimo debe ser estrictamente menor que el maximo.

## 5. Distribucion discreta

```text
\distrib escenario{{valor,probabilidad},{valor,probabilidad},...}
```

Ejemplo:

```text
\distrib escenario{{10,0.5},{15,0.3},{20,0.2}}
```

Los pares se separan mediante una coma externa. Las probabilidades deben pertenecer
a `[0,1]` y sumar 1. Los valores discretos no pueden repetirse.

## 6. Expresiones

Operadores:

```text
+  -  *  /  ^  \pm
```

Funciones/constructores:

```text
\frac{numerador}{denominador}
\sqrt{expresion}
```

La agrupacion se realiza con llaves. No se admiten parentesis `()` ni corchetes `[]`.

## 7. Multiplicacion implicita

Son validos ejemplos como:

```text
2precio
2\sqrt{precio}
precio{demanda+1}
```

Al admitirse identificadores de varias posiciones, `precio2` es un identificador y no
`precio * 2`. Para multiplicar una variable por un numero a la derecha utilice
`precio*2`; para multiplicar variables se recomienda `precio*demanda`.

## 8. Potencia

El exponente es un numero con signo opcional:

```text
precio^2
precio^-2
precio^+0.5
```

## 9. Plus/minus

Se permite como maximo un `\pm` en el modelo. El runtime genera dos ramas y conserva
**por separado** sus resultados, estadisticos, diagnosticos numericos e histogramas.

```text
\model{resultado=base\pmajuste}
```

La interfaz permite seleccionar `Rama +` o `Rama -`; no se mezclan las dos poblaciones
para calcular promedio, minimo, maximo o histograma.

## 10. Iteraciones

```text
\iter{100000}
```

Debe escribirse como entero y estar entre 1 y 1,000,000.

## 11. Response

```text
\response{avg,min,max}
```

Estadisticos permitidos: `avg`, `min`, `max`. No pueden repetirse y el orden solicitado
se conserva.

## 12. Diagnosticos numericos

Ademas de los estadisticos solicitados, cada rama expone el numero de observaciones,
desviacion estandar muestral, error estandar Monte Carlo de la media (MCSE) e intervalo
del 95 % para la media. Estos indicadores permiten observar la precision numerica de la
ejecucion sin mezclarla con las validaciones de compilacion.

## 13. Resultados no finitos

Los errores de dominio que dependen de valores aleatorios se resuelven durante la
ejecucion. Los `NaN` o infinitos se contabilizan como descartados dentro de la rama en
la que se produjeron. Si ninguna rama produce resultados finitos, el runtime finaliza con
un evento de error.

## 14. Semilla

Si no se proporciona semilla, el backend genera una de 128 bits y la muestra en la
interfaz. Guardarla junto con el modelo y la configuracion permite repetir la simulacion.

## 15. Ejemplo cuadratico

```text
\model{resultado=\frac{-coef_b\pm\sqrt{coef_b^2-4coef_a coef_c}}{2coef_a}}
\normal coef_a{10,0.5}
\distrib coef_b{{10,0.5},{15,0.3},{20,0.2}}
\uniform coef_c{-20,30}
\iter{1000000}
\response{avg,min,max}
```

## Estadisticos nuevos en v1.2.0

`\response` puede solicitar:

```text
avg, min, max, var, std, count, valid_rate, discard_rate, p05, p50, p95
```

Ejemplo:

```text
\model{y=x}
\normal x{10,2}
\iter{100000}
\response{avg,var,std,count,valid_rate,discard_rate,p05,p50,p95}
```

Si el modelo produce `NaN` o infinitos, esos resultados se descartan de los estimadores
y se reportan mediante `discard_rate`. Por esa razon, cuando hay descartes, `avg`, `var`,
`std` y los cuantiles corresponden solo a resultados finitos.
