from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourcePosition:
    """Posicion 1-based para linea/columna y 0-based para offset."""

    line: int
    column: int
    offset: int


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Rango de fuente semiabierto [start, end)."""

    start: SourcePosition
    end: SourcePosition

    @classmethod
    def point(cls, position: SourcePosition) -> "SourceSpan":
        return cls(position, position)
