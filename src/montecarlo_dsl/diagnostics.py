from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable

from .source import SourceSpan


class DiagnosticPhase(Enum):
    LEXICAL = auto()
    SYNTACTIC = auto()
    SEMANTIC = auto()
    GENERATION = auto()
    RUNTIME = auto()


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    phase: DiagnosticPhase
    message: str
    span: SourceSpan
    lexeme: str | None = None
    expected: tuple[str, ...] = ()
    found: str | None = None

    @property
    def line(self) -> int:
        return self.span.start.line

    @property
    def column(self) -> int:
        return self.span.start.column

    def format(self) -> str:
        return f"{self.code} [{self.phase.name}] L{self.line}:C{self.column}: {self.message}"


class CompilationError(Exception):
    """Error de compilacion con uno o mas diagnosticos estructurados."""

    def __init__(self, diagnostics: Diagnostic | Iterable[Diagnostic]):
        if isinstance(diagnostics, Diagnostic):
            items = (diagnostics,)
        else:
            items = tuple(diagnostics)
        if not items:
            raise ValueError("CompilationError requires at least one diagnostic")
        self.diagnostics = items
        super().__init__("\n".join(item.format() for item in items))
