from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Iterator, Mapping

from .ast import DistributionNode
from .source import SourceSpan


class SymbolKind(Enum):
    OUTPUT_VARIABLE = auto()
    RANDOM_VARIABLE = auto()


class DistributionKind(Enum):
    NONE = auto()
    NORMAL = auto()
    DISCRETE = auto()
    UNIFORM = auto()


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    kind: SymbolKind
    declaration_span: SourceSpan
    distribution_kind: DistributionKind = DistributionKind.NONE
    distribution: DistributionNode | None = None
    references: tuple[SourceSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class SymbolTable:
    """Vista inmutable de los simbolos validados del programa."""

    _symbols: Mapping[str, Symbol]

    @classmethod
    def from_symbols(cls, symbols: Mapping[str, Symbol]) -> "SymbolTable":
        return cls(MappingProxyType(dict(symbols)))

    def __getitem__(self, name: str) -> Symbol:
        return self._symbols[name]

    def get(self, name: str) -> Symbol | None:
        return self._symbols.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._symbols

    def __iter__(self) -> Iterator[str]:
        return iter(self._symbols)

    def __len__(self) -> int:
        return len(self._symbols)

    def items(self):
        return self._symbols.items()

    def values(self):
        return self._symbols.values()
