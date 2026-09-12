# Diagramas de automatas definitivos

Los siguientes archivos corresponden a los esquemas aprobados durante la revision del lexer. El estado `qE` de los diagramas representa terminacion por error lexico; la implementacion lanza `CompilationError` inmediatamente en lugar de consumir indefinidamente dentro de un estado trampa.

## Dispatcher principal

![Dispatcher principal](diagramas/01_dispatcher_principal.png)

## AD - Directivas

![AD directivas](diagramas/02_AD_directivas.png)

## AI - Identificadores de variables

AI reconoce `[A-Za-z][A-Za-z0-9_]*`: la primera posicion debe ser una letra ASCII y las siguientes pueden incluir letras, digitos o guion bajo.

![AI variables](diagramas/03_AI_variables.png)

## AN - Numeros

![AN numeros](diagramas/04_AN_numeros.png)

## AS - Simbolos

![AS simbolos](diagramas/05_AS_simbolos.png)

## AW - Espacios

![AW espacios](diagramas/06_AW_espacios.png)

## AE - Estadisticos de RESPONSE

![AE estadisticos](diagramas/07_AE_estadisticos.png)

## Correspondencia con el codigo

| Diagrama | Implementacion |
|---|---|
| Dispatcher q0/qP/qR | modos de `lexer.py` |
| AD | escaneo de directivas |
| AI | escaneo de VARIABLE |
| AN | escaneo de NUMBER |
| AS | escaneo de simbolos |
| AW | consumo de whitespace |
| AE | escaneo de AVG/MIN/MAX en RESPONSE |
