from decimal import Decimal

import pytest

from montecarlo_dsl import CompilationError, analyze_source
from montecarlo_dsl.ast import Statistic


def semantic_diagnostics(source: str):
    with pytest.raises(CompilationError) as exc:
        analyze_source(source)
    diagnostics = exc.value.diagnostics
    assert diagnostics
    assert all(d.phase.name == "SEMANTIC" for d in diagnostics)
    return diagnostics


def semantic_codes(source: str) -> list[str]:
    return [d.code for d in semantic_diagnostics(source)]


def test_semantic_contract_accepts_iteration_limits():
    for iterations in ("1", "1000000"):
        source = rf"""\model{{x=a}}
\normal a{{0,1}}
\iter{{{iterations}}}
\response{{avg}}"""
        validated = analyze_source(source)
        assert validated.iterations == int(iterations)


def test_semantic_contract_rejects_fractional_iterations_even_when_number_token_is_valid():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{.5}
\response{avg}"""
    assert "SEM020" in semantic_codes(source)


def test_semantic_contract_accepts_tiny_positive_standard_deviation():
    source = r"""\model{x=a}
\normal a{0,0.000001}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.symbol_table["a"].distribution is not None


def test_semantic_contract_accepts_uniform_with_negative_bounds():
    source = r"""\model{x=a}
\uniform a{-20,-1}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.random_variables == ("a",)


def test_semantic_contract_accepts_discrete_probability_endpoints_zero_and_one():
    source = r"""\model{x=a}
\distrib a{{1,1},{2,0}}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    distribution = validated.symbol_table["a"].distribution
    assert distribution is not None


def test_semantic_contract_probability_sum_within_tolerance_is_accepted():
    source = r"""\model{x=a}
\distrib a{{1,0.5},{2,0.4999999999995}}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.random_variables == ("a",)


def test_semantic_contract_probability_sum_outside_tolerance_is_rejected():
    source = r"""\model{x=a}
\distrib a{{1,0.5},{2,0.499999999998}}
\iter{10}
\response{avg}"""
    assert "SEM013" in semantic_codes(source)


def test_semantic_contract_decimal_equivalent_discrete_values_are_duplicates():
    source = r"""\model{x=a}
\distrib a{{1,0.5},{1.0,0.5}}
\iter{10}
\response{avg}"""
    assert "SEM014" in semantic_codes(source)


def test_semantic_contract_all_declared_random_variables_must_be_used():
    source = r"""\model{x=a}
\normal a{0,1}
\uniform b{0,1}
\distrib c{{1,1}}
\iter{10}
\response{avg}"""
    diagnostics = semantic_diagnostics(source)
    unused = [d for d in diagnostics if d.code == "SEM005"]
    assert len(unused) == 2
    assert "'b'" in unused[0].message
    assert "'c'" in unused[1].message


def test_semantic_contract_output_conflict_does_not_hide_other_valid_random_variable():
    source = r"""\model{x=a}
\normal x{0,1}
\normal a{0,1}
\iter{10}
\response{avg}"""
    found = semantic_codes(source)
    assert "SEM001" in found
    assert "SEM006" not in found


def test_semantic_contract_statistics_order_is_preserved():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{max,avg,min}"""
    validated = analyze_source(source)
    assert validated.requested_statistics == (
        Statistic.MAX,
        Statistic.AVG,
        Statistic.MIN,
    )


def test_semantic_contract_single_plus_minus_sets_validated_flag():
    source = r"""\model{x=a\pm b}
\normal a{0,1}
\normal b{0,1}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.has_plus_minus is True


def test_semantic_contract_model_without_plus_minus_clears_validated_flag():
    source = r"""\model{x=a+b}
\normal a{0,1}
\normal b{0,1}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.has_plus_minus is False


def test_semantic_contract_nested_constant_zero_divisor_is_rejected():
    source = r"""\model{x=a/{\sqrt{4}-2}}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM050" in semantic_codes(source)


def test_semantic_contract_unary_negative_constant_sqrt_is_rejected():
    source = r"""\model{x=a+\sqrt{-1}}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM051" in semantic_codes(source)


def test_semantic_contract_zero_to_negative_integer_power_is_rejected():
    source = r"""\model{x=a+0^-2}
\normal a{0,1}
\iter{10}
\response{avg}"""
    assert "SEM052" in semantic_codes(source)


def test_semantic_contract_negative_base_fractional_power_is_deferred_to_runtime():
    source = r"""\model{x=a+{-1}^0.5}
\normal a{0,1}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    assert validated.random_variables == ("a",)


def test_semantic_contract_diagnostic_spans_point_to_the_offending_constructs():
    source = r"""\model{x=a+b}
\normal a{0,0}
\iter{10.0}
\response{avg}"""
    diagnostics = semantic_diagnostics(source)
    by_code = {d.code: d for d in diagnostics}

    assert by_code["SEM003"].span.start.line == 1
    assert by_code["SEM010"].span.start.line == 2
    assert by_code["SEM020"].span.start.line == 3


def test_semantic_contract_validated_program_uses_decimal_source_values_without_float_rounding():
    source = r"""\model{x=a}
\normal a{0.1,0.2}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    distribution = validated.symbol_table["a"].distribution
    assert distribution is not None
    assert distribution.mean == Decimal("0.1")
    assert distribution.stddev == Decimal("0.2")
