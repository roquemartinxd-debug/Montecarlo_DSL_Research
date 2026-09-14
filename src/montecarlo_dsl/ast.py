from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto

from .source import SourceSpan


@dataclass(frozen=True, slots=True)
class ASTNode:
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class ExpressionNode(ASTNode):
    pass


class UnaryOperator(Enum):
    PLUS = auto()
    MINUS = auto()


class BinaryOperator(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()


class Statistic(Enum):
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    VAR = "var"
    STD = "std"
    COUNT = "count"
    VALID_RATE = "valid_rate"
    DISCARD_RATE = "discard_rate"
    P05 = "p05"
    P50 = "p50"
    P95 = "p95"


@dataclass(frozen=True, slots=True)
class NumberLiteralNode(ExpressionNode):
    lexeme: str
    value: Decimal


@dataclass(frozen=True, slots=True)
class VariableReferenceNode(ExpressionNode):
    name: str


@dataclass(frozen=True, slots=True)
class UnaryOperationNode(ExpressionNode):
    operator: UnaryOperator
    operand: ExpressionNode


@dataclass(frozen=True, slots=True)
class BinaryOperationNode(ExpressionNode):
    operator: BinaryOperator
    left: ExpressionNode
    right: ExpressionNode
    implicit: bool = False


@dataclass(frozen=True, slots=True)
class PlusMinusNode(ExpressionNode):
    left: ExpressionNode
    right: ExpressionNode


@dataclass(frozen=True, slots=True)
class PowerNode(ExpressionNode):
    base: ExpressionNode
    exponent: NumberLiteralNode | UnaryOperationNode


@dataclass(frozen=True, slots=True)
class FractionNode(ExpressionNode):
    numerator: ExpressionNode
    denominator: ExpressionNode


@dataclass(frozen=True, slots=True)
class SqrtNode(ExpressionNode):
    expression: ExpressionNode


@dataclass(frozen=True, slots=True)
class ModelDeclarationNode(ASTNode):
    output_name: str
    expression: ExpressionNode


@dataclass(frozen=True, slots=True)
class ProbabilityPairNode(ASTNode):
    value: Decimal
    probability: Decimal


@dataclass(frozen=True, slots=True)
class DistributionNode(ASTNode):
    variable: str


@dataclass(frozen=True, slots=True)
class NormalDistributionNode(DistributionNode):
    mean: Decimal
    stddev: Decimal


@dataclass(frozen=True, slots=True)
class DiscreteDistributionNode(DistributionNode):
    pairs: tuple[ProbabilityPairNode, ...]


@dataclass(frozen=True, slots=True)
class UniformDistributionNode(DistributionNode):
    minimum: Decimal
    maximum: Decimal


@dataclass(frozen=True, slots=True)
class IterationsNode(ASTNode):
    lexeme: str
    value: Decimal


@dataclass(frozen=True, slots=True)
class ResponseNode(ASTNode):
    statistics: tuple[Statistic, ...]


@dataclass(frozen=True, slots=True)
class ProgramNode(ASTNode):
    model: ModelDeclarationNode
    distributions: tuple[DistributionNode, ...]
    iterations: IterationsNode
    response: ResponseNode
