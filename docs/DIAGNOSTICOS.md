# Diagnosticos del compilador

El frontend recibe diagnosticos estructurados con codigo, fase, mensaje, linea, columna, offsets y, cuando aplica, lexema/esperado/encontrado.

## Familias

### LEX - Lexer

Codigos implementados:

`LEX001`, `LEX002`, `LEX003`, `LEX005`, `LEX006`, `LEX007`.

Cubren directivas invalidas, numeros invalidos, contenido no permitido y caracteres no reconocidos.

### SYN - Parser

Codigos implementados:

`SYN001`, `SYN010`, `SYN011`, `SYN020`, `SYN021`, `SYN022`, `SYN030`, `SYN040`, `SYN041`, `SYN042`.

Cubren estructura del programa, llaves/tokens esperados, distribuciones, separacion de pares discretos, expresiones y `response`.

### SEM - Analisis semantico

Codigos implementados:

`SEM001`, `SEM002`, `SEM003`, `SEM004`, `SEM005`, `SEM006`, `SEM010`, `SEM011`, `SEM012`, `SEM013`, `SEM014`, `SEM020`, `SEM021`, `SEM030`, `SEM040`, `SEM050`, `SEM051`, `SEM052`.

Grupos principales:

- `SEM001`-`SEM006`: variables, declaraciones, referencias y uso;
- `SEM010`-`SEM014`: distribuciones;
- `SEM020`-`SEM021`: iteraciones;
- `SEM030`: response;
- `SEM040`: limite de `\pm`;
- `SEM050`-`SEM052`: errores constantes de dominio matematico.

### GEN - Generacion/runtime NumPy

Codigos implementados:

`GEN001`, `GEN002`, `GEN003`, `GEN004`, `GEN005`, `GEN006`.

Incluyen protecciones de representacion `float64`, consistencia de ramas `\pm` y validaciones defensivas del generador.

## Contrato HTTP

Un error de cualquiera de estas fases se entrega como HTTP 422:

```json
{
  "error": "compilation_failed",
  "diagnostics": [
    {
      "code": "SEM003",
      "phase": "SEMANTIC",
      "message": "...",
      "line": 1,
      "column": 12,
      "start_offset": 11,
      "end_offset": 12,
      "lexeme": "b",
      "expected": [],
      "found": null
    }
  ]
}
```

La interfaz usa los offsets para seleccionar el fragmento relacionado con el diagnostico.
