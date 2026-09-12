from decimal import Decimal

import pytest

from montecarlo_dsl import CompilationError, analyze_source
from montecarlo_dsl.ast import Statistic
from montecarlo_dsl.symbols import DistributionKind, SymbolKind


REFERENCE_MODEL = r"""\model{x=\frac{-b\pm\sqrt{b^2-4a c}}{2a}}
\normal a{10,0.5}
\distrib b{{10,0.5},{15,0.3},{20,0.2}}
\uniform c{-20,30}
\iter{1000000}
\response{avg,min,max}"""


def codes(source: str) -> list[str]:
    with pytest.raises(CompilationError) as exc:
        analyze_source(source)
    assert all(d.phase.name == "SEMANTIC" for d in exc.value.diagnostics)
    return [d.code for d in exc.value.diagnostics]


def test_reference_model_is_semantically_valid_and_builds_symbol_table():
    validated = analyze_source(REFERENCE_MODEL)

    assert validated.output_variable == "x"
    assert validated.random_variables == ("a", "b", "c")
    assert validated.iterations == 1_000_000
    assert validated.requested_statistics == (Statistic.AVG, Statistic.MIN, Statistic.MAX)
    assert validated.has_plus_minus is True

    output = validated.symbol_table["x"]
    assert output.kind is SymbolKind.OUTPUT_VARIABLE
    assert output.distribution_kind is DistributionKind.NONE

    a = validated.symbol_table["a"]
    assert a.kind is SymbolKind.RANDOM_VARIABLE
    assert a.distribution_kind is DistributionKind.NORMAL
    assert len(a.references) == 2

    b = validated.symbol_table["b"]
    assert b.distribution_kind is DistributionKind.DISCRETE
    assert len(b.references) == 2

    c = validated.symbol_table["c"]
    assert c.distribution_kind is DistributionKind.UNIFORM
    assert len(c.references) == 1


def test_output_variable_cannot_be_random():
    source = r"""\model{x=a}
\normal x{0,1}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM001" in codes(source)


def test_output_variable_cannot_reference_itself():
    source = r"""\model{x=x+a}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM002" in codes(source)


def test_used_variable_requires_distribution():
    source = r"""\model{x=a+b}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM003" in codes(source)


def test_variable_cannot_have_two_distributions():
    source = r"""\model{x=a}
\normal a{0,1}
\uniform a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM004" in codes(source)


def test_distribution_for_unused_variable_is_rejected():
    source = r"""\model{x=a}
\normal a{0,1}
\uniform b{0,1}
\iter{10}
\response{avg}"""
    assert "SEM005" in codes(source)


def test_at_least_one_valid_random_variable_is_required_after_output_conflict():
    source = r"""\model{x=1}
\normal x{0,1}
\iter{10}
\response{avg}"""
    found = codes(source)
    assert "SEM001" in found
    assert "SEM006" in found


def test_normal_standard_deviation_must_be_positive():
    for stddev in ("0", "-1"):
        source = rf"""\model{{x=a}}
\normal a{{0,{stddev}}}
\iter{{10}}
\response{{avg}}"""
        assert "SEM010" in codes(source)


def test_uniform_minimum_must_be_less_than_maximum():
    for bounds in ("1,1", "2,1"):
        source = rf"""\model{{x=a}}
\uniform a{{{bounds}}}
\iter{{10}}
\response{{avg}}"""
        assert "SEM011" in codes(source)


def test_discrete_probability_must_be_between_zero_and_one():
    source = r"""\model{x=a}
\distrib a{{1,1.2},{2,0}}
\iter{10}
\response{avg}"""
    found = codes(source)
    assert "SEM012" in found
    assert "SEM013" in found


def test_discrete_probabilities_must_sum_to_one():
    source = r"""\model{x=a}
\distrib a{{1,0.4},{2,0.5}}
\iter{10}
\response{avg}"""
    assert "SEM013" in codes(source)


def test_discrete_probability_sum_accepts_exact_decimal_total():
    source = r"""\model{x=a}
\distrib a{{1,0.1},{2,0.2},{3,0.7}}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    distribution = validated.symbol_table["a"].distribution
    assert distribution is not None


def test_discrete_values_cannot_repeat():
    source = r"""\model{x=a}
\distrib a{{-1,0.5},{-1,0.5}}
\iter{10}
\response{avg}"""
    assert "SEM014" in codes(source)


def test_iterations_must_be_written_without_decimal_part():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10.0}
\response{avg}"""
    assert "SEM020" in codes(source)


@pytest.mark.parametrize("value", ["0", "1000001"])
def test_iterations_must_be_within_allowed_range(value: str):
    source = rf"""\model{{x=a}}
\normal a{{0,1}}
\iter{{{value}}}
\response{{avg}}"""
    assert "SEM021" in codes(source)


def test_response_statistics_cannot_repeat():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg,min,avg}"""
    assert "SEM030" in codes(source)


def test_model_can_have_at_most_one_plus_minus():
    source = r"""\model{x=a\pm b\pm c}
\normal a{0,1}
\normal b{0,1}
\normal c{0,1}
\iter{10}
\response{avg}"""
    assert "SEM040" in codes(source)


def test_constant_zero_fraction_denominator_is_rejected():
    source = r"""\model{x=\frac{a}{2-2}}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM050" in codes(source)


def test_constant_zero_binary_divisor_is_rejected():
    source = r"""\model{x=a/{3-3}}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM050" in codes(source)


def test_constant_negative_sqrt_is_rejected():
    source = r"""\model{x=a+\sqrt{1-2}}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM051" in codes(source)


def test_zero_to_negative_power_is_rejected():
    source = r"""\model{x=a+0^-1}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM052" in codes(source)


def test_runtime_dependent_domain_errors_are_not_semantic_errors():
    source = r"""\model{x=\frac{\sqrt{a}}{b}}
\normal a{0,1}
\normal b{0,1}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.random_variables == ("a", "b")


def test_multiple_independent_semantic_errors_are_collected_together():
    source = r"""\model{x=a+b}
\normal a{0,0}
\uniform a{2,1}
\uniform c{5,5}
\iter{10.0}
\response{avg,avg}"""
    found = codes(source)

    assert "SEM003" in found
    assert "SEM004" in found
    assert "SEM005" in found
    assert "SEM010" in found
    assert "SEM011" in found
    assert "SEM020" in found
    assert "SEM030" in found
