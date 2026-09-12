"""Pruebas de contrato derivadas de los automatas definitivos del lexer.

Estas pruebas no agregan nuevas reglas al lenguaje. Verifican que la
implementacion permanezca alineada con q0/qP/qR y los subautomatas
AD, AI, AN, AS, AW y AE documentados para el proyecto.
"""

import pytest

from montecarlo_dsl import CompilationError, TokenType, tokenize


def token_types(source: str) -> list[TokenType]:
    return [token.type for token in tokenize(source)]


def assert_lexical_error(source: str) -> None:
    with pytest.raises(CompilationError) as exc:
        tokenize(source)
    assert exc.value.diagnostics[0].phase.name == "LEXICAL"


def test_ai_accepts_multicharacter_ascii_identifiers():
    result = tokenize("Az")
    assert [(token.type, token.lexeme) for token in result] == [
        (TokenType.VARIABLE, "Az"),
        (TokenType.EOF, ""),
    ]


@pytest.mark.parametrize("source", [r"\MODEL", r"\Model", r"\Response{}"])
def test_ad_rejects_wrong_case_directives(source: str):
    assert_lexical_error(source)


def test_an_leaves_letter_backslash_and_comma_delimiters_for_redispatch():
    result = tokenize(r"2a 3\sqrt{a} 4,5")
    assert [(token.type, token.lexeme) for token in result] == [
        (TokenType.NUMBER, "2"),
        (TokenType.VARIABLE, "a"),
        (TokenType.NUMBER, "3"),
        (TokenType.SQRT, r"\sqrt"),
        (TokenType.LBRACE, "{"),
        (TokenType.VARIABLE, "a"),
        (TokenType.RBRACE, "}"),
        (TokenType.NUMBER, "4"),
        (TokenType.COMMA, ","),
        (TokenType.NUMBER, "5"),
        (TokenType.EOF, ""),
    ]


def test_an_rejects_non_delimiter_after_number():
    assert_lexical_error("12@")


def test_response_pending_can_reach_eof_without_duplicate_eof():
    result = tokenize(r"\response")
    assert [token.type for token in result] == [TokenType.RESPONSE, TokenType.EOF]
    assert sum(token.type is TokenType.EOF for token in result) == 1


def test_response_pending_reprocesses_a_following_directive_in_normal_mode():
    result = tokenize(r"\response \model{x=a}")
    assert [token.type for token in result] == [
        TokenType.RESPONSE,
        TokenType.MODEL,
        TokenType.LBRACE,
        TokenType.VARIABLE,
        TokenType.ASSIGN,
        TokenType.VARIABLE,
        TokenType.RBRACE,
        TokenType.EOF,
    ]


def test_aw_preserves_response_modes_across_crlf_whitespace():
    source = "\\response\r\n{\r\navg,\r\nmin\r\n}"
    assert token_types(source) == [
        TokenType.RESPONSE,
        TokenType.LBRACE,
        TokenType.AVG,
        TokenType.COMMA,
        TokenType.MIN,
        TokenType.RBRACE,
        TokenType.EOF,
    ]


def test_ae_allows_eof_as_statistic_lookahead():
    result = tokenize(r"\response{avg")
    assert [token.type for token in result] == [
        TokenType.RESPONSE,
        TokenType.LBRACE,
        TokenType.AVG,
        TokenType.EOF,
    ]


@pytest.mark.parametrize(
    "source",
    [
        r"\response{AVG}",
        r"\response{Min}",
        r"\response{average}",
        r"\response{max2}",
        r"\response{avg+}",
    ],
)
def test_ae_rejects_wrong_case_extensions_and_invalid_terminators(source: str):
    assert_lexical_error(source)


def test_rbrace_returns_from_response_mode_to_normal_mode():
    result = tokenize(r"\response{avg}x")
    assert [token.type for token in result] == [
        TokenType.RESPONSE,
        TokenType.LBRACE,
        TokenType.AVG,
        TokenType.RBRACE,
        TokenType.VARIABLE,
        TokenType.EOF,
    ]
    assert result[-2].lexeme == "x"


def test_comma_keeps_response_mode_active():
    assert_lexical_error(r"\response{avg,x}")


@pytest.mark.parametrize("source", ["", r"\response", r"\response{avg"])
def test_exactly_one_eof_is_emitted_from_each_dispatcher_mode(source: str):
    result = tokenize(source)
    assert sum(token.type is TokenType.EOF for token in result) == 1
    assert result[-1].type is TokenType.EOF
