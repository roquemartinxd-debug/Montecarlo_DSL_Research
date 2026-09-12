from decimal import Decimal

import pytest

from montecarlo_dsl import CompilationError, TokenType, tokenize


REFERENCE_MODEL = r"""\model{x=\frac{-b\pm\sqrt{b^2-4a c}}{2a}}
\normal a{10,0.5}
\distrib b{{10,0.5},{15,0.3},{20,0.2}}
\uniform c{-20,30}
\iter{1000000}
\response{avg,min,max}"""


def types(source: str):
    return [token.type for token in tokenize(source)]


def lexemes(source: str):
    return [token.lexeme for token in tokenize(source)]


def test_reference_model_tokenizes_completely():
    result = tokenize(REFERENCE_MODEL)

    assert result[-1].type is TokenType.EOF
    assert sum(token.type is TokenType.PM for token in result) == 1
    assert [token.type for token in result[-8:]] == [
        TokenType.LBRACE,
        TokenType.AVG,
        TokenType.COMMA,
        TokenType.MIN,
        TokenType.COMMA,
        TokenType.MAX,
        TokenType.RBRACE,
        TokenType.EOF,
    ]


def test_all_directives_are_recognized():
    source = r"\model \normal \distrib \uniform \iter \response{} \frac \sqrt \pm"
    result = tokenize(source)
    assert [t.type for t in result] == [
        TokenType.MODEL,
        TokenType.NORMAL,
        TokenType.DISTRIB,
        TokenType.UNIFORM,
        TokenType.ITER,
        TokenType.RESPONSE,
        TokenType.LBRACE,
        TokenType.RBRACE,
        TokenType.FRAC,
        TokenType.SQRT,
        TokenType.PM,
        TokenType.EOF,
    ]


@pytest.mark.parametrize("source", [r"\input", r"\include", r"\write", r"\modelx", r"\sqrtabc"])
def test_unknown_or_extended_directives_are_rejected(source):
    with pytest.raises(CompilationError) as exc:
        tokenize(source)
    assert exc.value.diagnostics[0].phase.name == "LEXICAL"


def test_identifiers_can_have_multiple_characters():
    result = tokenize("ac")
    assert [(t.type, t.lexeme) for t in result] == [
        (TokenType.VARIABLE, "ac"),
        (TokenType.EOF, ""),
    ]


def test_identifiers_allow_digits_and_underscore_after_initial_letter():
    result = tokenize("precio tasa_interes coef2")
    assert [(t.type, t.lexeme) for t in result] == [
        (TokenType.VARIABLE, "precio"),
        (TokenType.VARIABLE, "tasa_interes"),
        (TokenType.VARIABLE, "coef2"),
        (TokenType.EOF, ""),
    ]


def test_implicit_multiplication_input_is_not_invented_by_lexer():
    result = tokenize(r"4a c 2\sqrt{a}")
    assert [(t.type, t.lexeme) for t in result] == [
        (TokenType.NUMBER, "4"),
        (TokenType.VARIABLE, "a"),
        (TokenType.VARIABLE, "c"),
        (TokenType.NUMBER, "2"),
        (TokenType.SQRT, r"\sqrt"),
        (TokenType.LBRACE, "{"),
        (TokenType.VARIABLE, "a"),
        (TokenType.RBRACE, "}"),
        (TokenType.EOF, ""),
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0", Decimal("0")),
        ("7", Decimal("7")),
        ("123", Decimal("123")),
        ("0.5", Decimal("0.5")),
        ("12.75", Decimal("12.75")),
        (".5", Decimal(".5")),
    ],
)
def test_valid_number_forms(source, expected):
    token = tokenize(source)[0]
    assert token.type is TokenType.NUMBER
    assert token.value == expected


@pytest.mark.parametrize("source", ["01", "00", "5.", ".", "1.2.3"])
def test_invalid_number_forms_are_rejected(source):
    with pytest.raises(CompilationError):
        tokenize(source)


def test_number_signs_are_separate_tokens():
    result = tokenize("-20 a^-2")
    assert [(t.type, t.lexeme) for t in result] == [
        (TokenType.MINUS, "-"),
        (TokenType.NUMBER, "20"),
        (TokenType.VARIABLE, "a"),
        (TokenType.POWER, "^"),
        (TokenType.MINUS, "-"),
        (TokenType.NUMBER, "2"),
        (TokenType.EOF, ""),
    ]


def test_response_statistics_are_special_only_inside_response():
    normal = tokenize("avg")
    assert [t.type for t in normal] == [
        TokenType.VARIABLE,
        TokenType.EOF,
    ]
    assert normal[0].lexeme == "avg"

    response = tokenize(r"\response{avg,min,max}")
    assert [t.type for t in response] == [
        TokenType.RESPONSE,
        TokenType.LBRACE,
        TokenType.AVG,
        TokenType.COMMA,
        TokenType.MIN,
        TokenType.COMMA,
        TokenType.MAX,
        TokenType.RBRACE,
        TokenType.EOF,
    ]


def test_response_pending_allows_whitespace_before_opening_brace():
    result = tokenize("\\response  \t\n { avg, min }")
    assert [t.type for t in result] == [
        TokenType.RESPONSE,
        TokenType.LBRACE,
        TokenType.AVG,
        TokenType.COMMA,
        TokenType.MIN,
        TokenType.RBRACE,
        TokenType.EOF,
    ]


def test_response_pending_reprocesses_non_brace_in_normal_mode():
    result = tokenize(r"\response 123")
    assert [t.type for t in result] == [
        TokenType.RESPONSE,
        TokenType.NUMBER,
        TokenType.EOF,
    ]


@pytest.mark.parametrize("source", [r"\response{a}", r"\response{mean}", r"\response{avg1}"])
def test_invalid_response_contents_are_lexical_errors(source):
    with pytest.raises(CompilationError):
        tokenize(source)


def test_symbols_match_contract():
    source = "{},=+-*/^"
    assert types(source) == [
        TokenType.LBRACE,
        TokenType.RBRACE,
        TokenType.COMMA,
        TokenType.ASSIGN,
        TokenType.PLUS,
        TokenType.MINUS,
        TokenType.MULT,
        TokenType.DIV,
        TokenType.POWER,
        TokenType.EOF,
    ]


@pytest.mark.parametrize("source", ["(", ")", "[", "]", "@", ";"])
def test_forbidden_symbols_are_rejected(source):
    with pytest.raises(CompilationError):
        tokenize(source)


def test_only_one_eof_is_emitted():
    result = tokenize("a   ")
    assert sum(t.type is TokenType.EOF for t in result) == 1
    assert result[-1].type is TokenType.EOF


def test_windows_crlf_is_treated_as_one_logical_line_break():
    result = tokenize("a\r\nb")
    a, b, eof = result
    assert a.span.start.line == 1
    assert b.span.start.line == 2
    assert b.span.start.column == 1
    assert eof.span.start.line == 2


def test_error_reports_line_and_column():
    with pytest.raises(CompilationError) as exc:
        tokenize("a\n@")
    diagnostic = exc.value.diagnostics[0]
    assert diagnostic.line == 2
    assert diagnostic.column == 1
