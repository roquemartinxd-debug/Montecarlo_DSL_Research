from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

from .ast import (
    BinaryOperationNode,
    BinaryOperator,
    DiscreteDistributionNode,
    DistributionNode,
    ExpressionNode,
    FractionNode,
    NormalDistributionNode,
    NumberLiteralNode,
    PlusMinusNode,
    PowerNode,
    ProgramNode,
    SqrtNode,
    Statistic,
    UnaryOperationNode,
    UnaryOperator,
    UniformDistributionNode,
    VariableReferenceNode,
)
from .diagnostics import CompilationError, Diagnostic, DiagnosticPhase
from .parser import parse
from .source import SourceSpan
from .symbols import DistributionKind, Symbol, SymbolKind, SymbolTable


_PROBABILITY_TOLERANCE = Decimal("1e-12")
_MAX_ITERATIONS = 1_000_000


@dataclass(frozen=True, slots=True)
class ValidatedProgram:
    ast: ProgramNode
    symbol_table: SymbolTable
    output_variable: str
    random_variables: tuple[str, ...]
    iterations: int
    requested_statistics: tuple[Statistic, ...]
    has_plus_minus: bool


@dataclass(slots=True)
class _MutableSymbol:
    name: str
    kind: SymbolKind
    declaration_span: SourceSpan
    distribution_kind: DistributionKind = DistributionKind.NONE
    distribution: DistributionNode | None = None
    references: list[SourceSpan] | None = None

    def __post_init__(self) -> None:
        if self.references is None:
            self.references = []

    def freeze(self) -> Symbol:
        assert self.references is not None
        return Symbol(
            name=self.name,
            kind=self.kind,
            declaration_span=self.declaration_span,
            distribution_kind=self.distribution_kind,
            distribution=self.distribution,
            references=tuple(self.references),
        )


class SemanticAnalyzer:
    """Analisis semantico en tres pasadas y construccion de tabla de simbolos."""

    def __init__(self, program: ProgramNode):
        self.program = program
        self.diagnostics: list[Diagnostic] = []
        self._symbols: dict[str, _MutableSymbol] = {}
        self._distribution_nodes: dict[str, list[DistributionNode]] = {}
        self._used_random_variables: set[str] = set()
        self._plus_minus_count = 0

    def analyze(self) -> ValidatedProgram:
        self._register_declarations()
        self._validate_expression_and_references(self.program.model.expression)
        self._validate_global_rules()

        if self.diagnostics:
            raise CompilationError(self.diagnostics)

        frozen_symbols = {
            name: symbol.freeze()
            for name, symbol in self._symbols.items()
        }
        table = SymbolTable.from_symbols(frozen_symbols)
        random_variables = tuple(
            distribution.variable for distribution in self.program.distributions
        )

        return ValidatedProgram(
            ast=self.program,
            symbol_table=table,
            output_variable=self.program.model.output_name,
            random_variables=random_variables,
            iterations=int(self.program.iterations.value),
            requested_statistics=self.program.response.statistics,
            has_plus_minus=self._plus_minus_count == 1,
        )

    def _register_declarations(self) -> None:
        output = self.program.model.output_name
        self._symbols[output] = _MutableSymbol(
            name=output,
            kind=SymbolKind.OUTPUT_VARIABLE,
            declaration_span=self.program.model.span,
        )

        for distribution in self.program.distributions:
            name = distribution.variable
            self._distribution_nodes.setdefault(name, []).append(distribution)

            if name == output:
                self._add(
                    "SEM001",
                    f"La variable resultado '{name}' no puede declararse como variable aleatoria.",
                    distribution.span,
                )
                continue

            existing = self._symbols.get(name)
            if existing is not None and existing.kind is SymbolKind.RANDOM_VARIABLE:
                self._add(
                    "SEM004",
                    f"La variable '{name}' no puede tener mas de una distribucion.",
                    distribution.span,
                )
                continue

            self._symbols[name] = _MutableSymbol(
                name=name,
                kind=SymbolKind.RANDOM_VARIABLE,
                declaration_span=distribution.span,
                distribution_kind=self._distribution_kind(distribution),
                distribution=distribution,
            )

        if not any(symbol.kind is SymbolKind.RANDOM_VARIABLE for symbol in self._symbols.values()):
            self._add(
                "SEM006",
                "El modelo debe declarar al menos una variable aleatoria valida.",
                self.program.model.span,
            )

    def _validate_expression_and_references(self, node: ExpressionNode) -> None:
        if isinstance(node, VariableReferenceNode):
            self._register_reference(node)
            return

        if isinstance(node, NumberLiteralNode):
            return

        if isinstance(node, UnaryOperationNode):
            self._validate_expression_and_references(node.operand)
            return

        if isinstance(node, BinaryOperationNode):
            self._validate_expression_and_references(node.left)
            self._validate_expression_and_references(node.right)
            if node.operator is BinaryOperator.DIVIDE:
                denominator = self._constant_value(node.right)
                if denominator == 0:
                    self._add(
                        "SEM050",
                        "El denominador de una division constante no puede ser cero.",
                        node.right.span,
                    )
            return

        if isinstance(node, PlusMinusNode):
            self._plus_minus_count += 1
            self._validate_expression_and_references(node.left)
            self._validate_expression_and_references(node.right)
            return

        if isinstance(node, PowerNode):
            self._validate_expression_and_references(node.base)
            self._validate_expression_and_references(node.exponent)
            base = self._constant_value(node.base)
            exponent = self._constant_value(node.exponent)
            if base == 0 and exponent is not None and exponent < 0:
                self._add(
                    "SEM052",
                    "La potencia constante 0 elevada a un exponente negativo no esta definida.",
                    node.span,
                )
            return

        if isinstance(node, FractionNode):
            self._validate_expression_and_references(node.numerator)
            self._validate_expression_and_references(node.denominator)
            denominator = self._constant_value(node.denominator)
            if denominator == 0:
                self._add(
                    "SEM050",
                    "El denominador constante de \\frac no puede ser cero.",
                    node.denominator.span,
                )
            return

        if isinstance(node, SqrtNode):
            self._validate_expression_and_references(node.expression)
            value = self._constant_value(node.expression)
            if value is not None and value < 0:
                self._add(
                    "SEM051",
                    "El argumento constante de \\sqrt no puede ser negativo.",
                    node.expression.span,
                )
            return

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    def _register_reference(self, node: VariableReferenceNode) -> None:
        name = node.name
        output = self.program.model.output_name

        if name == output:
            self._add(
                "SEM002",
                f"La variable resultado '{name}' no puede aparecer en su propia expresion.",
                node.span,
            )
            symbol = self._symbols.get(name)
            if symbol is not None:
                assert symbol.references is not None
                symbol.references.append(node.span)
            return

        symbol = self._symbols.get(name)
        if symbol is None or symbol.kind is not SymbolKind.RANDOM_VARIABLE:
            self._add(
                "SEM003",
                f"La variable '{name}' se utiliza en el modelo pero no tiene una distribucion.",
                node.span,
            )
            return

        self._used_random_variables.add(name)
        assert symbol.references is not None
        symbol.references.append(node.span)

    def _validate_global_rules(self) -> None:
        self._validate_unused_distributions()
        self._validate_distributions()
        self._validate_iterations()
        self._validate_response()
        self._validate_plus_minus()

    def _validate_unused_distributions(self) -> None:
        output = self.program.model.output_name
        reported: set[str] = set()

        for distribution in self.program.distributions:
            name = distribution.variable
            if name == output or name in reported:
                continue
            if name not in self._used_random_variables:
                self._add(
                    "SEM005",
                    f"La variable '{name}' tiene una distribucion pero no se utiliza en el modelo.",
                    distribution.span,
                )
                reported.add(name)

    def _validate_distributions(self) -> None:
        for distribution in self.program.distributions:
            if isinstance(distribution, NormalDistributionNode):
                if distribution.stddev <= 0:
                    self._add(
                        "SEM010",
                        f"La desviacion estandar de '{distribution.variable}' debe ser mayor que cero.",
                        distribution.span,
                    )
                continue

            if isinstance(distribution, UniformDistributionNode):
                if distribution.minimum >= distribution.maximum:
                    self._add(
                        "SEM011",
                        f"En \\uniform para '{distribution.variable}', el minimo debe ser menor que el maximo.",
                        distribution.span,
                    )
                continue

            if isinstance(distribution, DiscreteDistributionNode):
                self._validate_discrete_distribution(distribution)
                continue

            raise TypeError(f"Unsupported distribution node: {type(distribution).__name__}")

    def _validate_discrete_distribution(self, distribution: DiscreteDistributionNode) -> None:
        total = Decimal(0)
        seen_values: set[Decimal] = set()

        for pair in distribution.pairs:
            probability = pair.probability
            total += probability

            if probability < 0 or probability > 1:
                self._add(
                    "SEM012",
                    f"La probabilidad {probability} de '{distribution.variable}' debe estar entre 0 y 1.",
                    pair.span,
                )

            if pair.value in seen_values:
                self._add(
                    "SEM014",
                    f"El valor discreto {pair.value} esta repetido para '{distribution.variable}'.",
                    pair.span,
                )
            else:
                seen_values.add(pair.value)

        if abs(total - Decimal(1)) > _PROBABILITY_TOLERANCE:
            self._add(
                "SEM013",
                f"Las probabilidades de '{distribution.variable}' suman {total}; deben sumar 1.",
                distribution.span,
            )

    def _validate_iterations(self) -> None:
        iterations = self.program.iterations

        if "." in iterations.lexeme:
            self._add(
                "SEM020",
                "\\iter debe escribirse como un entero sin parte decimal.",
                iterations.span,
            )
            return

        value = iterations.value
        if value < 1 or value > _MAX_ITERATIONS:
            self._add(
                "SEM021",
                f"\\iter debe estar entre 1 y {_MAX_ITERATIONS}.",
                iterations.span,
            )

    def _validate_response(self) -> None:
        seen: set[Statistic] = set()
        duplicates: list[str] = []

        for statistic in self.program.response.statistics:
            if statistic in seen and statistic.value not in duplicates:
                duplicates.append(statistic.value)
            seen.add(statistic)

        if duplicates:
            self._add(
                "SEM030",
                "\\response no permite estadisticos repetidos: " + ", ".join(duplicates) + ".",
                self.program.response.span,
            )

    def _validate_plus_minus(self) -> None:
        if self._plus_minus_count > 1:
            self._add(
                "SEM040",
                "El modelo puede contener como maximo un operador \\pm.",
                self.program.model.expression.span,
            )

    def _constant_value(self, node: ExpressionNode) -> Decimal | None:
        if isinstance(node, NumberLiteralNode):
            return node.value

        if isinstance(node, VariableReferenceNode):
            return None

        if isinstance(node, UnaryOperationNode):
            operand = self._constant_value(node.operand)
            if operand is None:
                return None
            return operand if node.operator is UnaryOperator.PLUS else -operand

        if isinstance(node, BinaryOperationNode):
            left = self._constant_value(node.left)
            right = self._constant_value(node.right)
            if left is None or right is None:
                return None
            if node.operator is BinaryOperator.ADD:
                return left + right
            if node.operator is BinaryOperator.SUBTRACT:
                return left - right
            if node.operator is BinaryOperator.MULTIPLY:
                return left * right
            if node.operator is BinaryOperator.DIVIDE:
                if right == 0:
                    return None
                return left / right

        if isinstance(node, PlusMinusNode):
            return None

        if isinstance(node, PowerNode):
            base = self._constant_value(node.base)
            exponent = self._constant_value(node.exponent)
            if base is None or exponent is None:
                return None
            if exponent != exponent.to_integral_value():
                return None
            try:
                return base ** int(exponent)
            except (InvalidOperation, ZeroDivisionError):
                return None

        if isinstance(node, FractionNode):
            numerator = self._constant_value(node.numerator)
            denominator = self._constant_value(node.denominator)
            if numerator is None or denominator is None or denominator == 0:
                return None
            return numerator / denominator

        if isinstance(node, SqrtNode):
            value = self._constant_value(node.expression)
            if value is None or value < 0:
                return None
            try:
                return value.sqrt()
            except InvalidOperation:
                return None

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    @staticmethod
    def _distribution_kind(distribution: DistributionNode) -> DistributionKind:
        if isinstance(distribution, NormalDistributionNode):
            return DistributionKind.NORMAL
        if isinstance(distribution, DiscreteDistributionNode):
            return DistributionKind.DISCRETE
        if isinstance(distribution, UniformDistributionNode):
            return DistributionKind.UNIFORM
        raise TypeError(f"Unsupported distribution node: {type(distribution).__name__}")

    def _add(self, code: str, message: str, span: SourceSpan) -> None:
        self.diagnostics.append(
            Diagnostic(
                code=code,
                phase=DiagnosticPhase.SEMANTIC,
                message=message,
                span=span,
            )
        )


def analyze(program: ProgramNode) -> ValidatedProgram:
    return SemanticAnalyzer(program).analyze()


def analyze_source(source: str) -> ValidatedProgram:
    return analyze(parse(source))
