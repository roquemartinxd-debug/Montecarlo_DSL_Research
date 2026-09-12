"""Pruebas de contrato para la gramatica sintactica definitiva del DSL.

Estas pruebas fijan decisiones que ya forman parte de la implementacion actual:
orden del programa, llaves como agrupacion, precedencia, multiplicacion implicita,
exponentes numericos con signo y estadisticos de RESPONSE.
"""

import pytest

from montecarlo_dsl import CompilationError, parse
from montecarlo_dsl.ast import (
    BinaryOperationNode,
    BinaryOperator,
    FractionNode,
    PlusMinusNode,
    PowerNode,
    Statistic,
    UnaryOperationNode,
    UnaryOperator,
)


def program_with_expression(expression: str) -> str:
    return rf"""\model{{x={expression}}}
\normal a{{0,1}}
\normal b{{0,1}}
\normal c{{0,1}}
\iter{{10}}
\response{{avg}}"""


def assert_syntactic_error(source: str) -> None:
    with pytest.raises(CompilationError) as exc:
        parse(source)
    assert exc.value.diagnostics[0].phase.name == "SYNTACTIC"


def test_program_order_requires_iterations_before_response():
    source = r"""\model{x=a}
\normal a{0,1}
\response{avg}
\iter{10}"""
    assert_syntactic_error(source)


def test_discrete_distribution_requires_at_least_one_probability_pair():
    source = r"""\model{x=a}
\distrib a{}
\iter{10}
\response{avg}"""
    with pytest.raises(CompilationError) as exc:
        parse(source)
    assert exc.value.diagnostics[0].code == "SYN020"


def test_response_requires_at_least_one_statistic():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{}"""
    with pytest.raises(CompilationError) as exc:
        parse(source)
    assert exc.value.diagnostics[0].code == "SYN030"


def test_signed_uniform_bounds_are_syntactically_valid():
    source = r"""\model{x=a}
\uniform a{-20,+30}
\iter{10}
\response{avg}"""
    program = parse(source)
    distribution = program.distributions[0]
    assert distribution.minimum == -20
    assert distribution.maximum == 30


def test_pm_shares_sum_precedence_and_product_binds_more_tightly():
    expression = parse(program_with_expression(r"a+b\pm c*d")).model.expression

    assert isinstance(expression, PlusMinusNode)
    assert isinstance(expression.left, BinaryOperationNode)
    assert expression.left.operator is BinaryOperator.ADD
    assert isinstance(expression.right, BinaryOperationNode)
    assert expression.right.operator is BinaryOperator.MULTIPLY


def test_sum_level_operators_are_left_associative_including_pm():
    expression = parse(program_with_expression(r"a\pm b+c")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.ADD
    assert isinstance(expression.left, PlusMinusNode)


def test_unary_plus_binds_below_power():
    expression = parse(program_with_expression("+a^2")).model.expression

    assert isinstance(expression, UnaryOperationNode)
    assert expression.operator is UnaryOperator.PLUS
    assert isinstance(expression.operand, PowerNode)


@pytest.mark.parametrize("expression", ["a^b", "a^{2}"])
def test_power_exponent_must_be_a_signed_or_unsigned_number(expression: str):
    assert_syntactic_error(program_with_expression(expression))


def test_explicit_multiplication_allows_number_on_right_even_though_implicit_does_not():
    expression = parse(program_with_expression("a*2")).model.expression
    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.MULTIPLY
    assert expression.implicit is False


def test_fraction_can_be_an_implicit_factor():
    expression = parse(program_with_expression(r"2\frac{a}{b}")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.MULTIPLY
    assert expression.implicit is True
    assert isinstance(expression.right, FractionNode)


def test_response_uses_statistic_tokens_not_general_variables():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{min,max}"""
    program = parse(source)
    assert program.response.statistics == (Statistic.MIN, Statistic.MAX)
