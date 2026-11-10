from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import math
import time

from .ast import (
    BinaryOperationNode,
    BinaryOperator,
    ExpressionNode,
    FractionNode,
    ModelDeclarationNode,
    NumberLiteralNode,
    PlusMinusNode,
    PowerNode,
    ProgramNode,
    SqrtNode,
    UnaryOperationNode,
    UnaryOperator,
    VariableReferenceNode,
)
from .semantic import ValidatedProgram


def _is_representable_float64(value: Decimal) -> bool:
    """Comprueba que un Decimal pueda convertirse a float64 finito sin underflow no nulo."""
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return False
    if not math.isfinite(converted):
        return False
    if value != 0 and converted == 0.0:
        return False
    return True


def count_ast_nodes(node: ExpressionNode) -> int:
    """Cuenta el numero total de nodos en el AST de una expresion."""
    if isinstance(node, (NumberLiteralNode, VariableReferenceNode)):
        return 1
    if isinstance(node, UnaryOperationNode):
        return 1 + count_ast_nodes(node.operand)
    if isinstance(node, (BinaryOperationNode, PlusMinusNode)):
        return 1 + count_ast_nodes(node.left) + count_ast_nodes(node.right)
    if isinstance(node, PowerNode):
        return 1 + count_ast_nodes(node.base) + count_ast_nodes(node.exponent)
    if isinstance(node, FractionNode):
        return 1 + count_ast_nodes(node.numerator) + count_ast_nodes(node.denominator)
    if isinstance(node, SqrtNode):
        return 1 + count_ast_nodes(node.expression)
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


@dataclass(frozen=True, slots=True)
class OptimizationMetrics:
    """Metricas verificables de la fase de optimizacion del compilador (BC06)."""

    enabled: bool
    ast_nodes_before: int
    ast_nodes_after: int
    constant_folds: int
    algebraic_simplifications: int
    strength_reductions: int
    optimization_time_seconds: float

    @property
    def nodes_reduced(self) -> int:
        return self.ast_nodes_before - self.ast_nodes_after

    @property
    def total_transformations(self) -> int:
        return (
            self.constant_folds
            + self.algebraic_simplifications
            + self.strength_reductions
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "enabled": self.enabled,
            "ast_nodes_before": self.ast_nodes_before,
            "ast_nodes_after": self.ast_nodes_after,
            "nodes_reduced": self.nodes_reduced,
            "constant_folds": self.constant_folds,
            "algebraic_simplifications": self.algebraic_simplifications,
            "strength_reductions": self.strength_reductions,
            "total_transformations": self.total_transformations,
            "optimization_time_seconds": self.optimization_time_seconds,
        }


class ASTOptimizer:
    """Optimizador de AST seguro bajo la semantica numerica IEEE-754 del DSL.

    Realiza constant folding de operaciones binarias entre literales despues de
    convertir cada operando a float64, preservando el orden de redondeo del runtime
    NumPy. Tambien elimina el signo unario positivo y la doble negacion, que son
    transformaciones bit-preserving para valores float64.

    No elimina identidades dinamicas (por ejemplo ``x + 0`` o ``x * 1``), ni pliega
    ``sqrt`` o potencias mediante aritmetica escalar Python: esas transformaciones
    pueden alterar ceros con signo, NaN, redondeo o diferencias entre NumPy ufuncs
    y libm.
    """

    def __init__(self) -> None:
        self.constant_folds = 0
        self.algebraic_simplifications = 0
        self.strength_reductions = 0

    def optimize_program(
        self,
        program: ValidatedProgram,
        *,
        enabled: bool = True,
    ) -> tuple[ValidatedProgram, OptimizationMetrics]:
        started = time.perf_counter()
        original_expr = program.ast.model.expression
        nodes_before = count_ast_nodes(original_expr)

        if not enabled:
            elapsed = time.perf_counter() - started
            return program, OptimizationMetrics(
                enabled=False,
                ast_nodes_before=nodes_before,
                ast_nodes_after=nodes_before,
                constant_folds=0,
                algebraic_simplifications=0,
                strength_reductions=0,
                optimization_time_seconds=elapsed,
            )

        self.constant_folds = 0
        self.algebraic_simplifications = 0
        self.strength_reductions = 0

        optimized_expr = self.optimize_expression(original_expr)
        nodes_after = count_ast_nodes(optimized_expr)
        elapsed = time.perf_counter() - started

        optimized_model = ModelDeclarationNode(
            span=program.ast.model.span,
            output_name=program.ast.model.output_name,
            expression=optimized_expr,
        )
        optimized_ast = replace(program.ast, model=optimized_model)
        optimized_program = replace(program, ast=optimized_ast)

        return optimized_program, OptimizationMetrics(
            enabled=True,
            ast_nodes_before=nodes_before,
            ast_nodes_after=nodes_after,
            constant_folds=self.constant_folds,
            algebraic_simplifications=self.algebraic_simplifications,
            strength_reductions=self.strength_reductions,
            optimization_time_seconds=elapsed,
        )

    def optimize_expression(self, node: ExpressionNode) -> ExpressionNode:
        if isinstance(node, (NumberLiteralNode, VariableReferenceNode)):
            return node

        if isinstance(node, UnaryOperationNode):
            operand = self.optimize_expression(node.operand)
            if isinstance(operand, NumberLiteralNode) and _is_representable_float64(operand.value):
                try:
                    source_value = float(operand.value)
                except (OverflowError, ValueError):
                    source_value = math.inf
                if math.isfinite(source_value):
                    folded_value = (
                        source_value
                        if node.operator is UnaryOperator.PLUS
                        else -source_value
                    )
                    self.constant_folds += 1
                    return self._make_float64_literal(folded_value, node.span)

            # Unary plus is an identity operation; two sign-bit flips restore the
            # original IEEE-754 representation, including signed zero.
            if node.operator is UnaryOperator.PLUS:
                self.algebraic_simplifications += 1
                return replace(operand, span=node.span)
            if (
                node.operator is UnaryOperator.MINUS
                and isinstance(operand, UnaryOperationNode)
                and operand.operator is UnaryOperator.MINUS
            ):
                self.algebraic_simplifications += 1
                return replace(operand.operand, span=node.span)
            return replace(node, operand=operand)

        if isinstance(node, BinaryOperationNode):
            left = self.optimize_expression(node.left)
            right = self.optimize_expression(node.right)
            if isinstance(left, NumberLiteralNode) and isinstance(right, NumberLiteralNode):
                folded = self._fold_binary_constants(node.operator, left.value, right.value)
                if folded is not None:
                    self.constant_folds += 1
                    return self._make_float64_literal(folded, node.span)
            # Do not eliminate dynamic identities such as x + 0, x * 1 or x / 1:
            # they can change signed-zero or NaN behavior under IEEE-754.
            return replace(node, left=left, right=right)

        if isinstance(node, FractionNode):
            numerator = self.optimize_expression(node.numerator)
            denominator = self.optimize_expression(node.denominator)
            if isinstance(numerator, NumberLiteralNode) and isinstance(denominator, NumberLiteralNode):
                folded = self._fold_binary_constants(
                    BinaryOperator.DIVIDE,
                    numerator.value,
                    denominator.value,
                )
                if folded is not None:
                    self.constant_folds += 1
                    return self._make_float64_literal(folded, node.span)
            return replace(node, numerator=numerator, denominator=denominator)

        if isinstance(node, PowerNode):
            # NumPy's power ufunc is not replaced by Python's scalar pow operation:
            # retaining the node avoids claiming bitwise equivalence across libm/ufuncs.
            return replace(
                node,
                base=self.optimize_expression(node.base),
                exponent=self.optimize_expression(node.exponent),
            )

        if isinstance(node, SqrtNode):
            # Keep sqrt in the NumPy runtime so the optimized and unoptimized paths
            # use the same float64 ufunc implementation.
            return replace(node, expression=self.optimize_expression(node.expression))

        if isinstance(node, PlusMinusNode):
            return replace(
                node,
                left=self.optimize_expression(node.left),
                right=self.optimize_expression(node.right),
            )

        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    @staticmethod
    def _fold_binary_constants(
        operator: BinaryOperator,
        left: Decimal,
        right: Decimal,
    ) -> Decimal | None:
        # Convert each literal to float64 before applying the operation, matching
        # the generated NumPy runtime's operand representation and rounding order.
        if not _is_representable_float64(left) or not _is_representable_float64(right):
            return None
        try:
            left64 = float(left)
            right64 = float(right)
            if operator is BinaryOperator.ADD:
                result = left64 + right64
            elif operator is BinaryOperator.SUBTRACT:
                result = left64 - right64
            elif operator is BinaryOperator.MULTIPLY:
                result = left64 * right64
            elif operator is BinaryOperator.DIVIDE:
                if right64 == 0.0:
                    return None
                result = left64 / right64
            else:
                return None
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
        if not math.isfinite(result):
            return None
        return Decimal.from_float(result)

    @staticmethod
    def _make_float64_literal(value: float, span) -> NumberLiteralNode:
        return NumberLiteralNode(
            span=span,
            lexeme=repr(float(value)),
            value=Decimal.from_float(float(value)),
        )


def optimize_program(
    program: ValidatedProgram,
    *,
    enabled: bool = True,
) -> tuple[ValidatedProgram, OptimizationMetrics]:
    """Ejecuta la fase de optimizacion del AST sobre un programa validado."""
    return ASTOptimizer().optimize_program(program, enabled=enabled)
