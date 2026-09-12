from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from typing import TypeAlias

from .source import SourceSpan


class TokenType(Enum):
    MODEL = auto()
    NORMAL = auto()
    DISTRIB = auto()
    UNIFORM = auto()
    ITER = auto()
    RESPONSE = auto()
    FRAC = auto()
    SQRT = auto()
    PM = auto()

    AVG = auto()
    MIN = auto()
    MAX = auto()

    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    ASSIGN = auto()
    PLUS = auto()
    MINUS = auto()
    MULT = auto()
    DIV = auto()
    POWER = auto()

    VARIABLE = auto()
    NUMBER = auto()
    EOF = auto()


TokenValue: TypeAlias = Decimal | str | None


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    lexeme: str
    span: SourceSpan
    value: TokenValue = None
