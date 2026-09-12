from dataclasses import FrozenInstanceError

import pytest

from montecarlo_dsl import analyze_source
from montecarlo_dsl.ast import (
    DiscreteDistributionNode,
    NormalDistributionNode,
    UniformDistributionNode,
)
from montecarlo_dsl.symbols import DistributionKind, SymbolKind


SOURCE = r"""\model{x=a+a+b+c}
\normal a{10,0.5}
\distrib b{{10,0.5},{20,0.5}}
\uniform c{-20,30}
\iter{100}
\response{avg,min,max}"""


def validated():
    return analyze_source(SOURCE)


def test_symbol_table_contract_contains_output_and_random_variables_only():
    table = validated().symbol_table
    assert len(table) == 4
    assert tuple(table) == ("x", "a", "b", "c")
    assert all(name in table for name in ("x", "a", "b", "c"))
    assert "z" not in table


def test_symbol_table_contract_get_and_index_behave_as_mapping():
    table = validated().symbol_table
    assert table.get("a") is table["a"]
    assert table.get("missing") is None
    with pytest.raises(KeyError):
        _ = table["missing"]


def test_symbol_table_contract_output_symbol_has_no_distribution():
    output = validated().symbol_table["x"]
    assert output.kind is SymbolKind.OUTPUT_VARIABLE
    assert output.distribution_kind is DistributionKind.NONE
    assert output.distribution is None
    assert output.declaration_span.start.line == 1


def test_symbol_table_contract_random_symbols_keep_distribution_kind_and_node():
    table = validated().symbol_table

    assert table["a"].kind is SymbolKind.RANDOM_VARIABLE
    assert table["a"].distribution_kind is DistributionKind.NORMAL
    assert isinstance(table["a"].distribution, NormalDistributionNode)

    assert table["b"].distribution_kind is DistributionKind.DISCRETE
    assert isinstance(table["b"].distribution, DiscreteDistributionNode)

    assert table["c"].distribution_kind is DistributionKind.UNIFORM
    assert isinstance(table["c"].distribution, UniformDistributionNode)


def test_symbol_table_contract_declaration_spans_preserve_source_lines():
    table = validated().symbol_table
    assert table["a"].declaration_span.start.line == 2
    assert table["b"].declaration_span.start.line == 3
    assert table["c"].declaration_span.start.line == 4


def test_symbol_table_contract_reference_spans_are_recorded_in_expression_order():
    table = validated().symbol_table
    assert len(table["a"].references) == 2
    assert len(table["b"].references) == 1
    assert len(table["c"].references) == 1
    assert all(span.start.line == 1 for span in table["a"].references)
    assert table["a"].references[0].start.column < table["a"].references[1].start.column


def test_symbol_table_contract_mapping_is_immutable():
    table = validated().symbol_table
    with pytest.raises(TypeError):
        table._symbols["new"] = table["a"]  # type: ignore[index]


def test_symbol_table_contract_symbols_are_frozen():
    symbol = validated().symbol_table["a"]
    with pytest.raises(FrozenInstanceError):
        symbol.name = "z"  # type: ignore[misc]


def test_symbol_table_contract_items_and_values_expose_the_validated_entries():
    table = validated().symbol_table
    assert [name for name, _ in table.items()] == ["x", "a", "b", "c"]
    assert [symbol.name for symbol in table.values()] == ["x", "a", "b", "c"]


def test_validated_program_contract_preserves_random_variable_declaration_order():
    program = validated()
    assert program.output_variable == "x"
    assert program.random_variables == ("a", "b", "c")
