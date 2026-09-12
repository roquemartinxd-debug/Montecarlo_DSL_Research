from decimal import Decimal

import pytest

from montecarlo_dsl import CompilationError, parse
from montecarlo_dsl.ast import (
    BinaryOperationNode,
    BinaryOperator,
    DiscreteDistributionNode,
    FractionNode,
    NormalDistributionNode,
    NumberLiteralNode,
    PlusMinusNode,
    PowerNode,
    SqrtNode,
    Statistic,
    UnaryOperationNode,
    UnaryOperator,
    UniformDistributionNode,
    VariableReferenceNode,
)


REFERENCE_MODEL = r"""\model{x=\frac{-b\pm\sqrt{b^2-4a c}}{2a}}
\normal a{10,0.5}
\distrib b{{10,0.5},{15,0.3},{20,0.2}}
\uniform c{-20,30}
\iter{1000000}
\response{avg,min,max}"""


def program_with_expression(expression: str) -> str:
    return rf"""\model{{x={expression}}}
\normal a{{0,1}}
\normal b{{0,1}}
\normal c{{0,1}}
\iter{{10}}
\response{{avg}}"""


def test_reference_model_builds_expected_program_shape():
    program = parse(REFERENCE_MODEL)

    assert program.model.output_name == "x"
    assert len(program.distributions) == 3
    assert isinstance(program.distributions[0], NormalDistributionNode)
    assert isinstance(program.distributions[1], DiscreteDistributionNode)
    assert isinstance(program.distributions[2], UniformDistributionNode)
    assert program.iterations.lexeme == "1000000"
    assert program.iterations.value == Decimal("1000000")
    assert program.response.statistics == (Statistic.AVG, Statistic.MIN, Statistic.MAX)

    expression = program.model.expression
    assert isinstance(expression, FractionNode)
    assert isinstance(expression.numerator, PlusMinusNode)
    assert isinstance(expression.numerator.left, UnaryOperationNode)
    assert expression.numerator.left.operator is UnaryOperator.MINUS
    assert isinstance(expression.numerator.right, SqrtNode)

    denominator = expression.denominator
    assert isinstance(denominator, BinaryOperationNode)
    assert denominator.operator is BinaryOperator.MULTIPLY
    assert denominator.implicit is True


def test_operator_precedence_multiplies_before_addition():
    expression = parse(program_with_expression("a+b*c")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.ADD
    assert isinstance(expression.left, VariableReferenceNode)
    assert isinstance(expression.right, BinaryOperationNode)
    assert expression.right.operator is BinaryOperator.MULTIPLY
    assert expression.right.implicit is False


def test_subtraction_is_left_associative():
    expression = parse(program_with_expression("a-b-c")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.SUBTRACT
    assert isinstance(expression.left, BinaryOperationNode)
    assert expression.left.operator is BinaryOperator.SUBTRACT
    assert isinstance(expression.right, VariableReferenceNode)
    assert expression.right.name == "c"


def test_division_is_left_associative():
    expression = parse(program_with_expression("a/b/c")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.DIVIDE
    assert isinstance(expression.left, BinaryOperationNode)
    assert expression.left.operator is BinaryOperator.DIVIDE


def test_power_has_higher_precedence_than_unary_minus():
    expression = parse(program_with_expression("-b^2")).model.expression

    assert isinstance(expression, UnaryOperationNode)
    assert expression.operator is UnaryOperator.MINUS
    assert isinstance(expression.operand, PowerNode)
    assert isinstance(expression.operand.base, VariableReferenceNode)
    assert expression.operand.base.name == "b"


def test_group_can_change_unary_power_binding_without_parentheses():
    expression = parse(program_with_expression("{-b}^2")).model.expression

    assert isinstance(expression, PowerNode)
    assert isinstance(expression.base, UnaryOperationNode)
    assert expression.base.operator is UnaryOperator.MINUS


def test_signed_negative_exponent_is_preserved_in_ast():
    expression = parse(program_with_expression("a^-2")).model.expression

    assert isinstance(expression, PowerNode)
    assert isinstance(expression.exponent, UnaryOperationNode)
    assert expression.exponent.operator is UnaryOperator.MINUS
    assert isinstance(expression.exponent.operand, NumberLiteralNode)
    assert expression.exponent.operand.value == Decimal("2")


def test_implicit_multiplication_chain_is_constructed_by_parser():
    expression = parse(program_with_expression("4a c")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.MULTIPLY
    assert expression.implicit is True
    assert isinstance(expression.left, BinaryOperationNode)
    assert expression.left.operator is BinaryOperator.MULTIPLY
    assert expression.left.implicit is True
    assert isinstance(expression.left.left, NumberLiteralNode)
    assert expression.left.left.value == Decimal("4")
    assert isinstance(expression.left.right, VariableReferenceNode)
    assert expression.left.right.name == "a"
    assert isinstance(expression.right, VariableReferenceNode)
    assert expression.right.name == "c"


def test_implicit_multiplication_before_sqrt():
    expression = parse(program_with_expression(r"2\sqrt{a}")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.MULTIPLY
    assert expression.implicit is True
    assert isinstance(expression.left, NumberLiteralNode)
    assert isinstance(expression.right, SqrtNode)


def test_fraction_and_sqrt_are_recursive():
    expression = parse(program_with_expression(r"\frac{a+\sqrt{b}}{c}")).model.expression

    assert isinstance(expression, FractionNode)
    assert isinstance(expression.numerator, BinaryOperationNode)
    assert expression.numerator.operator is BinaryOperator.ADD
    assert isinstance(expression.numerator.right, SqrtNode)


def test_grouping_uses_braces_and_disappears_as_separate_ast_node():
    expression = parse(program_with_expression("{a+b}c")).model.expression

    assert isinstance(expression, BinaryOperationNode)
    assert expression.operator is BinaryOperator.MULTIPLY
    assert expression.implicit is True
    assert isinstance(expression.left, BinaryOperationNode)
    assert expression.left.operator is BinaryOperator.ADD


def test_signed_distribution_parameters_are_parsed_as_decimals():
    source = r"""\model{x=a}
\normal a{-10,+0.5}
\iter{10}
\response{avg}"""
    program = parse(source)
    distribution = program.distributions[0]

    assert isinstance(distribution, NormalDistributionNode)
    assert distribution.mean == Decimal("-10")
    assert distribution.stddev == Decimal("0.5")


def test_discrete_distribution_accepts_negative_values_but_not_signed_probabilities():
    valid = r"""\model{x=a}
\distrib a{{-10,0.5},{20,0.5}}
\iter{10}
\response{avg}"""
    distribution = parse(valid).distributions[0]
    assert isinstance(distribution, DiscreteDistributionNode)
    assert distribution.pairs[0].value == Decimal("-10")

    invalid = r"""\model{x=a}
\distrib a{{10,-0.5},{20,0.5}}
\iter{10}
\response{avg}"""
    with pytest.raises(CompilationError) as exc:
        parse(invalid)
    assert exc.value.diagnostics[0].phase.name == "SYNTACTIC"


def test_decimal_iterations_are_preserved_for_later_semantic_validation():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10.0}
\response{avg}"""
    iterations = parse(source).iterations
    assert iterations.lexeme == "10.0"
    assert iterations.value == Decimal("10.0")


def test_response_duplicates_are_syntactically_allowed_for_semantic_phase():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg,avg}"""
    program = parse(source)
    assert program.response.statistics == (Statistic.AVG, Statistic.AVG)


def test_missing_response_opening_brace_is_reported_by_parser():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response avg"""
    with pytest.raises(CompilationError) as exc:
        parse(source)

    diagnostic = exc.value.diagnostics[0]
    assert diagnostic.phase.name == "SYNTACTIC"
    assert diagnostic.expected == ("LBRACE",)


def test_at_least_one_distribution_is_required():
    source = r"""\model{x=a}
\iter{10}
\response{avg}"""
    with pytest.raises(CompilationError) as exc:
        parse(source)
    assert exc.value.diagnostics[0].code == "SYN010"


def test_identifier_may_contain_trailing_digits_but_adjacent_number_is_not_implicit_factor():
    parsed = parse(program_with_expression("a2")).model.expression
    assert isinstance(parsed, VariableReferenceNode)
    assert parsed.name == "a2"

    with pytest.raises(CompilationError):
        parse(program_with_expression("a 2"))


def test_adjacent_numbers_are_not_implicit_multiplication():
    with pytest.raises(CompilationError):
        parse(program_with_expression("2 3"))


def test_trailing_tokens_after_response_are_rejected():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg}a"""
    with pytest.raises(CompilationError) as exc:
        parse(source)
    diagnostic = exc.value.diagnostics[0]
    assert diagnostic.phase.name == "SYNTACTIC"
    assert diagnostic.expected == ("EOF",)


def test_parser_reports_source_location_for_missing_model_brace():
    source = r"""\model{x=a
\normal a{0,1}
\iter{10}
\response{avg}"""
    with pytest.raises(CompilationError) as exc:
        parse(source)
    diagnostic = exc.value.diagnostics[0]
    assert diagnostic.phase.name == "SYNTACTIC"
    assert diagnostic.line == 2
    assert diagnostic.column == 1
