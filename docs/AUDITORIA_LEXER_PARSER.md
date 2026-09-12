# Auditoria lexer/parser - v1.1.0

## Resultado

**APROBADO CON ACTUALIZACION DEL CONTRATO DEL LENGUAJE.**

## Identificadores

El subautomata AI reconoce ahora identificadores completos con el patron:

`[A-Za-z][A-Za-z0-9_]*`

La actualizacion permite nombres descriptivos (`precio`, `demanda_2026`, `tasa_interes`)
y conserva compatibilidad con variables de una sola letra.

## BNF

Se elimino la redundancia entre `<DECIMAL_SIGNADO>` y `<NUMERO_SIGNADO>`. La BNF
utiliza unicamente `<NUMERO_SIGNADO>` para el patron NUMBER con signo opcional.

El parser mantiene internamente funciones distintas cuando necesita devolver un
`Decimal` o construir un nodo AST; esa diferencia de representacion no cambia la
produccion sintactica aceptada.

## Distribucion discreta

La lista discreta utiliza separadores explicitos:

`\distrib demanda{{10,0.5},{15,0.3},{20,0.2}}`

Se agregaron diagnosticos sintacticos para una coma sin par posterior y para pares
adyacentes sin coma.

## AST

Se conservan los nodos del AST y la precedencia del lenguaje. Los nombres completos de
identificadores se propagan como cadenas hasta la tabla de simbolos y posteriormente son
mapeados a nombres internos seguros por el generador.

## Pruebas

La suite final incluye pruebas especificas para identificadores multicaracter, palabras
reservadas de Python usadas como nombres DSL, identificadores con guion bajo/digitos y
separacion obligatoria de pares discretos.
