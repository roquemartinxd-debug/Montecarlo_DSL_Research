# Casos de prueba y validacion - v1.1.0

La version final contiene **205 pruebas automatizadas**.

## Distribucion por archivo

| Archivo | Pruebas |
|---|---:|
| `test_feedback_hardening.py` | 9 |
| `test_frontend.py` | 6 |
| `test_generator.py` | 13 |
| `test_generator_contract.py` | 17 |
| `test_integration_contract.py` | 8 |
| `test_job_manager.py` | 2 |
| `test_lexer.py` | 38 |
| `test_lexer_contract.py` | 20 |
| `test_parser.py` | 21 |
| `test_parser_contract.py` | 12 |
| `test_semantic.py` | 24 |
| `test_semantic_contract.py` | 19 |
| `test_server.py` | 6 |
| `test_symbol_table_contract.py` | 10 |
| **Total** | **205** |

## Casos agregados por la revision final

`test_feedback_hardening.py` comprueba especificamente:

1. identificadores de longitud variable de extremo a extremo;
2. mapeo seguro de identificadores que coinciden con palabras reservadas de Python;
3. contrato lexical de guion bajo y digitos despues de la primera letra;
4. separadores explicitos obligatorios en `\distrib`;
5. separacion estadistica de las ramas de `\pm`;
6. presencia de MCSE e intervalo de 95 % por rama;
7. aislamiento de salida no protocolaria en stdout;
8. ausencia de creacion de trabajos cuando falla la compilacion;
9. propagacion de identificadores completos en el AST.

## Ejecucion

Desde la raiz del proyecto:

```powershell
python -m pytest
```

Resultado esperado de esta entrega: **205 pruebas aprobadas**.
