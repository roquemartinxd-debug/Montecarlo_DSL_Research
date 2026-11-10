from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from montecarlo_dsl import (
    ASTOptimizer,
    GeneratorConfig,
    analyze_source,
    compile_source,
    count_ast_nodes,
    optimize_program,
)
from montecarlo_dsl.ast import NumberLiteralNode, UnaryOperationNode, VariableReferenceNode


PREFIX = "MC_DSL_EVENT:"


def _run_compiled(tmp_path: Path, source: str, *, seed: int, optimize: bool):
    compiled = compile_source(
        source,
        GeneratorConfig(
            seed=seed,
            batch_size=100,
            logical_block_size=100,
            max_workers=2,
            histogram_bins=20,
            optimize=optimize,
        ),
    )
    script_path = tmp_path / f"sim_opt_{optimize}.py"
    script_path.write_text(compiled.script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    events = [
        json.loads(line[len(PREFIX):])
        for line in proc.stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    return compiled, events[-1]


def test_constant_folding_reduces_pure_constant_subtrees():
    source = r"""\model{y = x + {2 + 3 * 4} - \frac{10}{2} + \sqrt{16} + 2^3}
\normal x{5, 1}
\iter{100}
\response{avg}"""
    program = analyze_source(source)
    optimized, metrics = optimize_program(program, enabled=True)

    assert metrics.enabled is True
    assert metrics.constant_folds >= 3
    assert metrics.ast_nodes_after < metrics.ast_nodes_before
    assert metrics.nodes_reduced > 0


def test_only_bit_preserving_unary_simplifications_are_applied():
    source = r"""\model{y = 1 * {+x} + 0 - 0 + \frac{x^1}{1} + {-{-x}}}
\normal x{10, 2}
\iter{100}
\response{avg}"""
    program = analyze_source(source)
    optimized, metrics = optimize_program(program, enabled=True)

    assert metrics.algebraic_simplifications >= 2
    assert metrics.strength_reductions == 0
    assert metrics.ast_nodes_after < metrics.ast_nodes_before


def test_dynamic_float_identities_are_not_removed():
    source = r"""\model{y = 1 * {x + 0}^1 / 1}
\normal x{10, 2}
\iter{100}
\response{avg}"""
    program = analyze_source(source)
    optimized, metrics = optimize_program(program, enabled=True)
    assert not isinstance(optimized.ast.model.expression, VariableReferenceNode)
    assert count_ast_nodes(optimized.ast.model.expression) == metrics.ast_nodes_before


def test_dynamic_negation_and_identity_operations_are_preserved():
    source = r"""\model{y = {-1} * x + {0 - z} + x / {-1}}
\normal x{2, 1}
\uniform z{1, 3}
\iter{100}
\response{avg}"""
    program = analyze_source(source)
    optimized, metrics = optimize_program(program, enabled=True)
    assert metrics.algebraic_simplifications == 0


def test_optimizer_does_not_eliminate_potentially_nonfinite_zero_multiplication(tmp_path: Path):
    # Si x < 0, \sqrt{x} produce NaN; 0 * \sqrt{x} debe seguir produciendo NaN
    # para respetar nonfinite_policy = discard_and_report.
    source = r"""\model{y = a + 0 * \sqrt{x}}
\normal a{10, 1}
\uniform x{-5, 5}
\iter{400}
\response{avg,count,valid_rate,discard_rate}"""
    _, final_unopt = _run_compiled(tmp_path, source, seed=42, optimize=False)
    _, final_opt = _run_compiled(tmp_path, source, seed=42, optimize=True)

    assert final_opt["discarded_results"] > 0
    assert final_opt["valid_results"] == final_unopt["valid_results"]
    assert final_opt["discarded_results"] == final_unopt["discarded_results"]
    assert final_opt["stats"] == final_unopt["stats"]
    assert final_opt["diagnostics"] == final_unopt["diagnostics"]
    assert final_opt["histogram"] == final_unopt["histogram"]
    assert final_opt["branches"] == final_unopt["branches"]
    assert final_opt["paired"] == final_unopt["paired"]


@pytest.mark.parametrize(
    "source",
    [
        r"""\model{y = {2 + 3} * x + \frac{8}{2} * {z + 0} - 1 * x^1}
\normal x{10, 2}
\uniform z{-4, 6}
\iter{400}
\response{avg,var,std,min,max,p05,p50,p95}""",
        r"""\model{y = \frac{-b \pm \sqrt{b^2 - {4 * 1} * a * c}}{2 * 1 * a} + {10 - 10}}
\normal a{10, 0.5}
\distrib b{{10, 0.5}, {15, 0.3}, {20, 0.2}}
\uniform c{-20, 30}
\iter{400}
\response{avg,var,std,count,valid_rate,discard_rate,p05,p50,p95}""",
    ],
)
def test_differential_equivalence_between_unoptimized_and_optimized(
    tmp_path: Path,
    source: str,
):
    c_unopt, final_unopt = _run_compiled(tmp_path, source, seed=909, optimize=False)
    c_opt, final_opt = _run_compiled(tmp_path, source, seed=909, optimize=True)

    assert c_opt.optimization_metrics is not None
    assert c_opt.optimization_metrics.nodes_reduced > 0
    assert final_opt["valid_results"] == final_unopt["valid_results"]
    assert final_opt["discarded_results"] == final_unopt["discarded_results"]
    assert final_opt["stats"] == final_unopt["stats"]


def test_constant_folding_uses_float64_rounding_not_decimal_rounding(tmp_path: Path):
    source = r"""\model{y = 0.1 + 0.2 + x}
\normal x{0, 1}
\iter{1000}
\response{avg,var,min,max,p50}"""
    _, final_unopt = _run_compiled(tmp_path, source, seed=1701, optimize=False)
    _, final_opt = _run_compiled(tmp_path, source, seed=1701, optimize=True)
    assert final_opt["stats"] == final_unopt["stats"]


def test_optimizer_does_not_remove_signed_zero_sensitive_addition(tmp_path: Path):
    source = r"""\model{y = 0 + x}
\normal x{0, 1}
\iter{1000}
\response{avg,var,min,max,p50}"""
    compiled = compile_source(source, GeneratorConfig(seed=1702, optimize=True))
    assert " + " in compiled.script
