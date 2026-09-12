from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum, auto

from .diagnostics import CompilationError, Diagnostic, DiagnosticPhase
from .source import SourcePosition, SourceSpan
from .tokens import Token, TokenType


class _Mode(Enum):
    NORMAL = auto()            # q0
    RESPONSE_PENDING = auto()  # qP
    RESPONSE = auto()          # qR


@dataclass(slots=True)
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    token_type: TokenType | None = None


_DIRECTIVES: dict[str, TokenType] = {
    "model": TokenType.MODEL,
    "normal": TokenType.NORMAL,
    "distrib": TokenType.DISTRIB,
    "uniform": TokenType.UNIFORM,
    "iter": TokenType.ITER,
    "response": TokenType.RESPONSE,
    "frac": TokenType.FRAC,
    "sqrt": TokenType.SQRT,
    "pm": TokenType.PM,
}

_SYMBOLS: dict[str, TokenType] = {
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    "=": TokenType.ASSIGN,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.MULT,
    "/": TokenType.DIV,
    "^": TokenType.POWER,
}

_STATISTICS: dict[str, TokenType] = {
    "avg": TokenType.AVG,
    "min": TokenType.MIN,
    "max": TokenType.MAX,
}

_NUMBER_DELIMITERS = set("{} ,=+-*/^\\\t\n\r")


def _is_ascii_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_identifier_continue(ch: str) -> bool:
    return _is_ascii_letter(ch) or _is_ascii_digit(ch) or ch == "_"


def _is_whitespace(ch: str) -> bool:
    return ch in {" ", "\t", "\n", "\r"}


def _build_directive_trie() -> _TrieNode:
    root = _TrieNode()
    for word, token_type in _DIRECTIVES.items():
        node = root
        for ch in word:
            node = node.children.setdefault(ch, _TrieNode())
        node.token_type = token_type
    return root


_DIRECTIVE_TRIE = _build_directive_trie()


class Lexer:
    """Lexer modular basado en q0/qP/qR y los subautomatas definidos."""

    def __init__(self, source: str):
        self.source = source
        self._offset = 0
        self._line = 1
        self._column = 1
        self._previous_was_cr = False
        self._mode = _Mode.NORMAL

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []

        while True:
            if self._at_end():
                tokens.append(self._make_eof())
                return tokens

            token = self._dispatch()
            if token is None:
                continue

            tokens.append(token)

            if token.type is TokenType.EOF:
                return tokens

    def _dispatch(self) -> Token | None:
        if self._mode is _Mode.NORMAL:
            return self._dispatch_normal()
        if self._mode is _Mode.RESPONSE_PENDING:
            return self._dispatch_response_pending()
        return self._dispatch_response()

    def _dispatch_normal(self) -> Token | None:
        ch = self._peek()

        if ch == "\\":
            token = self._scan_directive()
            if token.type is TokenType.RESPONSE:
                self._mode = _Mode.RESPONSE_PENDING
            return token

        if _is_ascii_letter(ch):
            return self._scan_variable()

        if _is_ascii_digit(ch) or ch == ".":
            return self._scan_number()

        if ch in _SYMBOLS:
            return self._scan_symbol()

        if _is_whitespace(ch):
            self._scan_whitespace()
            return None

        self._raise_invalid_character(ch)

    def _dispatch_response_pending(self) -> Token | None:
        ch = self._peek()

        if _is_whitespace(ch):
            self._scan_whitespace()
            return None

        if ch == "{":
            token = self._scan_symbol()
            self._mode = _Mode.RESPONSE
            return token

        # Reprocesa el caracter en q0 para que el parser diagnostique la llave faltante.
        self._mode = _Mode.NORMAL
        return None

    def _dispatch_response(self) -> Token | None:
        ch = self._peek()

        if _is_whitespace(ch):
            self._scan_whitespace()
            return None

        if ch in {"a", "m"}:
            return self._scan_statistic()

        if ch in {",", "}"}:
            token = self._scan_symbol()
            if token.type is TokenType.RBRACE:
                self._mode = _Mode.NORMAL
            return token

        self._raise(
            code="LEX006",
            message=(
                "Dentro de \\response solo se permiten avg, min, max, comas, "
                "espacios y la llave de cierre."
            ),
            lexeme=ch,
        )

    def _scan_directive(self) -> Token:
        start = self._position()
        self._advance()
        node = _DIRECTIVE_TRIE

        while True:
            if self._at_end():
                if node.token_type is None:
                    self._raise(
                        "LEX002",
                        "Directiva incompleta al final de la entrada.",
                        span=SourceSpan(start, self._position()),
                        lexeme=self.source[start.offset:self._offset],
                    )
                return self._token(node.token_type, start)

            ch = self._peek()

            if _is_ascii_letter(ch):
                child = node.children.get(ch)
                if child is None:
                    end = self._position_after_current()
                    self._raise(
                        "LEX001",
                        "Directiva desconocida o no permitida.",
                        span=SourceSpan(start, end),
                        lexeme=self.source[start.offset:self._offset + 1],
                    )
                self._advance()
                node = child
                continue

            if node.token_type is None:
                self._raise(
                    "LEX002",
                    "Directiva incompleta.",
                    span=SourceSpan(start, self._position()),
                    lexeme=self.source[start.offset:self._offset],
                )

            # Una directiva no puede ser el prefijo de un identificador mayor.
            # Ejemplo: ``\\model_extra`` se rechaza como directiva desconocida.
            if _is_ascii_letter(ch) or ch == "_":
                end = self._position_after_current()
                self._raise(
                    "LEX001",
                    "Directiva desconocida o no permitida.",
                    span=SourceSpan(start, end),
                    lexeme=self.source[start.offset:self._offset + 1],
                )

            return self._token(node.token_type, start)

    def _scan_variable(self) -> Token:
        start = self._position()
        self._advance()  # La primera posicion siempre es una letra ASCII.
        while not self._at_end() and _is_identifier_continue(self._peek()):
            self._advance()
        lexeme = self.source[start.offset:self._offset]
        return self._token(TokenType.VARIABLE, start, value=lexeme)

    def _scan_number(self) -> Token:
        start = self._position()
        state = "n0"

        while True:
            ch = self._peek_or_none()

            if state == "n0":
                if ch == "0":
                    self._advance()
                    state = "nCero"
                    continue
                if ch is not None and "1" <= ch <= "9":
                    self._advance()
                    state = "nEntero"
                    continue
                if ch == ".":
                    self._advance()
                    state = "nPuntoInicial"
                    continue
                self._raise_number("Inicio de numero invalido.", start)

            elif state == "nCero":
                if ch is not None and _is_ascii_digit(ch):
                    self._raise_number("No se permiten ceros iniciales en numeros enteros.", start, include_current=True)
                if ch == ".":
                    self._advance()
                    state = "nPuntoCero"
                    continue
                return self._finish_number(start, ch)

            elif state == "nEntero":
                if ch is not None and _is_ascii_digit(ch):
                    self._advance()
                    continue
                if ch == ".":
                    self._advance()
                    state = "nPuntoEntero"
                    continue
                return self._finish_number(start, ch)

            elif state in {"nPuntoInicial", "nPuntoCero", "nPuntoEntero"}:
                if ch is not None and _is_ascii_digit(ch):
                    self._advance()
                    state = "nFraccion"
                    continue
                self._raise_number("Se requiere al menos un digito despues del punto decimal.", start, include_current=ch is not None)

            elif state == "nFraccion":
                if ch is not None and _is_ascii_digit(ch):
                    self._advance()
                    continue
                if ch == ".":
                    self._raise_number("Un numero no puede contener un segundo punto decimal.", start, include_current=True)
                return self._finish_number(start, ch)

    def _finish_number(self, start: SourcePosition, lookahead: str | None) -> Token:
        if lookahead is not None and not self._is_number_delimiter(lookahead):
            self._raise_number(
                f"Caracter no permitido despues de un numero: {lookahead!r}.",
                start,
                include_current=True,
            )

        lexeme = self.source[start.offset:self._offset]
        try:
            value = Decimal(lexeme)
        except InvalidOperation:
            self._raise_number("Numero invalido.", start)

        return self._token(TokenType.NUMBER, start, value=value)

    @staticmethod
    def _is_number_delimiter(ch: str) -> bool:
        return _is_ascii_letter(ch) or ch in _NUMBER_DELIMITERS

    def _scan_symbol(self) -> Token:
        start = self._position()
        ch = self._advance()
        token_type = _SYMBOLS.get(ch)
        if token_type is None:
            self._raise(
                "LEX005",
                f"Simbolo no reconocido: {ch!r}.",
                span=SourceSpan(start, self._position()),
                lexeme=ch,
            )
        return self._token(token_type, start)

    def _scan_whitespace(self) -> None:
        while not self._at_end() and _is_whitespace(self._peek()):
            self._advance()

    def _scan_statistic(self) -> Token:
        start = self._position()
        candidates = tuple(word for word in _STATISTICS if word.startswith(self._peek()))

        for word in candidates:
            if self.source.startswith(word, self._offset):
                end_offset = self._offset + len(word)
                next_ch = self.source[end_offset] if end_offset < len(self.source) else None

                if self._valid_statistic_terminator(next_ch):
                    for _ in word:
                        self._advance()
                    return self._token(_STATISTICS[word], start)

        self._raise(
            "LEX007",
            "Estadistico desconocido. Solo se permiten avg, min y max.",
            span=SourceSpan(start, self._position_after_candidate()),
            lexeme=self._response_word_fragment(),
        )

    @staticmethod
    def _valid_statistic_terminator(ch: str | None) -> bool:
        return ch is None or ch in {",", "}"} or (ch is not None and _is_whitespace(ch))

    def _at_end(self) -> bool:
        return self._offset >= len(self.source)

    def _peek(self) -> str:
        return self.source[self._offset]

    def _peek_or_none(self) -> str | None:
        return None if self._at_end() else self.source[self._offset]

    def _position(self) -> SourcePosition:
        return SourcePosition(self._line, self._column, self._offset)

    def _position_after_current(self) -> SourcePosition:
        if self._at_end():
            return self._position()
        line, column, offset = self._line, self._column, self._offset
        ch = self.source[offset]
        if ch in {"\n", "\r"}:
            return SourcePosition(line + 1, 1, offset + 1)
        if ch == "\t":
            return SourcePosition(line, column + 1, offset + 1)
        return SourcePosition(line, column + 1, offset + 1)

    def _advance(self) -> str:
        ch = self.source[self._offset]
        self._offset += 1

        if ch == "\r":
            self._line += 1
            self._column = 1
            self._previous_was_cr = True
        elif ch == "\n":
            if not self._previous_was_cr:
                self._line += 1
            self._column = 1
            self._previous_was_cr = False
        elif ch == "\t":
            self._column += 1
            self._previous_was_cr = False
        else:
            self._column += 1
            self._previous_was_cr = False

        return ch

    def _token(self, token_type: TokenType, start: SourcePosition, value=None) -> Token:
        end = self._position()
        lexeme = self.source[start.offset:end.offset]
        return Token(token_type, lexeme, SourceSpan(start, end), value)

    def _make_eof(self) -> Token:
        pos = self._position()
        return Token(TokenType.EOF, "", SourceSpan.point(pos), None)

    def _raise_invalid_character(self, ch: str) -> None:
        self._raise(
            "LEX005",
            f"Caracter no permitido en el DSL: {ch!r}.",
            lexeme=ch,
        )

    def _raise_number(
        self,
        message: str,
        start: SourcePosition,
        include_current: bool = False,
    ) -> None:
        end = self._position_after_current() if include_current and not self._at_end() else self._position()
        self._raise(
            "LEX003",
            message,
            span=SourceSpan(start, end),
            lexeme=self.source[start.offset:end.offset],
        )

    def _response_word_fragment(self) -> str:
        end = self._offset
        while end < len(self.source) and _is_ascii_letter(self.source[end]):
            end += 1
        if end == self._offset:
            end = min(self._offset + 1, len(self.source))
        return self.source[self._offset:end]

    def _position_after_candidate(self) -> SourcePosition:
        end = self._offset
        while end < len(self.source) and _is_ascii_letter(self.source[end]):
            end += 1
        return SourcePosition(self._line, self._column + (end - self._offset), end)

    def _raise(
        self,
        code: str,
        message: str,
        *,
        span: SourceSpan | None = None,
        lexeme: str | None = None,
    ) -> None:
        if span is None:
            start = self._position()
            span = SourceSpan(start, self._position_after_current())
        raise CompilationError(
            Diagnostic(
                code=code,
                phase=DiagnosticPhase.LEXICAL,
                message=message,
                span=span,
                lexeme=lexeme,
            )
        )


def tokenize(source: str) -> list[Token]:
    return Lexer(source).tokenize()
