from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from montecarlo_dsl import (
    CompilationError,
    GeneratorConfig,
    NumpyExpressionVisitor,
    analyze_source,
    generate_python_script,
    generate_source,
)


def run_generated(
    tmp_path: Path,
    source: str,
    *,
    seed: int = 123,
    batch_size: int = 20,
    max_workers: int = 2,
    histogram_bins: int = 12,
):
    script = generate_source(
        source,
        GeneratorConfig(
            seed=seed,
            batch_size=batch_size,
            max_workers=max_workers,
            histogram_bins=histogram_bins,
        ),
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    script_path = tmp_path / "generated_contract.py"
    script_path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    prefix = "MC_DSL_EVENT:"
    messages = [
        json.loads(line[len(prefix):])
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    return script, completed, messages


def generation_error(source: str):
    validated = analyze_source(source)
    with pytest.raises(CompilationError) as exc:
        generate_python_script(validated, GeneratorConfig(seed=1))
    diagnostics = exc.value.diagnostics
    assert diagnostics
    assert all(d.phase.name == "GENERATION" for d in diagnostics)
    return diagnostics


def test_generator_contract_unary_plus_and_minus_are_preserved_by_visitor():
    source = r"""\model{x=-a+{+a}}
\normal a{0,1}
\iter{10}
\response{avg}"""
    validated = analyze_source(source)
    branch = NumpyExpressionVisitor().visit(validated.ast.model.expression).branches[0]
    assert "-(a)" in branch
    assert "+(a)" in branch


def test_generator_contract_implicit_and_explicit_multiplication_have_same_runtime_result(tmp_path):
    implicit = r"""\model{x=2a}
\normal a{0,1}
\iter{80}
\response{avg,min,max}"""
    explicit = r"""\model{x=2*a}
\normal a{0,1}
\iter{80}
\response{avg,min,max}"""

    _, p1, m1 = run_generated(tmp_path / "implicit", implicit, seed=991)
    _, p2, m2 = run_generated(tmp_path / "explicit", explicit, seed=991)
    assert p1.returncode == 0, p1.stderr
    assert p2.returncode == 0, p2.stderr
    assert m1[-1]["stats"] == m2[-1]["stats"]
    assert m1[-1]["histogram"] == m2[-1]["histogram"]


def test_generator_contract_plus_minus_propagates_two_branches_through_parent_expression(tmp_path):
    source = r"""\model{x=2{a\pm1}}
\normal a{0,1}
\iter{40}
\response{avg,min,max}"""
    _, completed, messages = run_generated(tmp_path, source, seed=21)
    assert completed.returncode == 0, completed.stderr
    assert messages[0]["has_plus_minus"] is True
    assert messages[-1]["valid_results"] == 80
    assert messages[-1]["discarded_results"] == 0
    assert [branch["id"] for branch in messages[-1]["branches"]] == ["plus", "minus"]
    assert all(branch["valid_results"] == 40 for branch in messages[-1]["branches"])


def test_generator_contract_dynamic_division_by_zero_discards_only_invalid_results(tmp_path):
    source = r"""\model{x=1/a}
\distrib a{{0,0.5},{1,0.5}}
\iter{120}
\response{avg,min,max}"""
    _, completed, messages = run_generated(tmp_path, source, seed=1, batch_size=30)
    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert 0 < final["valid_results"] < 120
    assert final["valid_results"] + final["discarded_results"] == 120
    assert final["stats"] == {"avg": 1.0, "min": 1.0, "max": 1.0}


def test_generator_contract_negative_base_fractional_power_is_resolved_at_runtime(tmp_path):
    source = r"""\model{x=a+{-1}^0.5}
\normal a{0,1}
\iter{30}
\response{avg}"""
    _, completed, messages = run_generated(tmp_path, source, seed=4)
    assert completed.returncode == 1
    assert messages[-1]["type"] == "error"
    assert "sin ningun resultado" in messages[-1]["message"]


def test_generator_contract_partial_last_batch_is_planned_exactly(tmp_path):
    source = r"""\model{x=a}
\normal a{0,1}
\iter{53}
\response{avg}"""
    _, completed, messages = run_generated(tmp_path, source, seed=2, batch_size=20)
    assert completed.returncode == 0, completed.stderr
    assert messages[0]["batch_count"] == 3
    batch_messages = [m for m in messages if m["type"] == "batch"]
    assert sorted(m["batch"]["iterations"] for m in batch_messages) == [13, 20, 20]
    assert messages[-1]["processed_iterations"] == 53


def test_generator_contract_worker_count_never_exceeds_number_of_batches(tmp_path):
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg}"""
    _, completed, messages = run_generated(
        tmp_path, source, seed=9, batch_size=100, max_workers=4
    )
    assert completed.returncode == 0, completed.stderr
    assert messages[0]["batch_count"] == 1
    assert messages[0]["workers"] == 1


def test_generator_contract_requested_statistics_preserve_source_order(tmp_path):
    source = r"""\model{x=a}
\normal a{0,1}
\iter{30}
\response{max,avg,min}"""
    _, completed, messages = run_generated(tmp_path, source, seed=3)
    assert completed.returncode == 0, completed.stderr
    assert messages[0]["requested_statistics"] == ["max", "avg", "min"]
    assert list(messages[-1]["stats"]) == ["max", "avg", "min"]


def test_generator_contract_constant_sample_distribution_produces_exact_histogram_count(tmp_path):
    source = r"""\model{x=a}
\distrib a{{2,1}}
\iter{50}
\response{avg,min,max}"""
    _, completed, messages = run_generated(tmp_path, source, seed=11, histogram_bins=8)
    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert final["stats"] == {"avg": 2.0, "min": 2.0, "max": 2.0}
    assert final["histogram"]["total_count"] == 50
    assert sum(b["count"] for b in final["histogram"]["bins"]) == 50
    assert len(final["histogram"]["bins"]) == 8


def test_generator_contract_each_batch_event_histogram_matches_accumulated_valid_results(tmp_path):
    source = r"""\model{x=\sqrt{a}}
\uniform a{-1,1}
\iter{90}
\response{avg}"""
    _, completed, messages = run_generated(tmp_path, source, seed=15, batch_size=30)
    assert completed.returncode == 0, completed.stderr
    batch_messages = [m for m in messages if m["type"] == "batch"]
    assert batch_messages
    for message in batch_messages:
        assert message["histogram"]["total_count"] == message["valid_results"]
        assert message["valid_results"] + message["discarded_results"] == (
            message["processed_iterations"]
        )


def test_generator_contract_generated_script_includes_windows_process_safety_hook():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg}"""
    script = generate_source(source, GeneratorConfig(seed=1))
    assert "multiprocessing.freeze_support()" in script
    assert 'if __name__ == "__main__":' in script


def test_generator_contract_default_seed_is_nonnegative_128_bit_integer():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg}"""
    script = generate_source(source)
    match = re.search(r"^MASTER_SEED = (\d+)$", script, flags=re.MULTILINE)
    assert match is not None
    seed = int(match.group(1))
    assert 0 <= seed < (1 << 128)


def test_generator_contract_nonzero_decimal_that_underflows_float64_is_rejected():
    tiny = "0." + ("0" * 399) + "1"
    source = rf"""\model{{x=a+{tiny}}}
\normal a{{0,1}}
\iter{{10}}
\response{{avg}}"""
    diagnostics = generation_error(source)
    assert diagnostics[0].code == "GEN004"


def test_generator_contract_tiny_positive_normal_stddev_cannot_collapse_to_zero():
    tiny = "0." + ("0" * 399) + "1"
    source = rf"""\model{{x=a}}
\normal a{{0,{tiny}}}
\iter{{10}}
\response{{avg}}"""
    diagnostics = generation_error(source)
    assert diagnostics[0].code == "GEN004"


def test_generator_contract_uniform_bounds_must_remain_ordered_after_float64_conversion():
    upper = "1." + ("0" * 399) + "1"
    source = rf"""\model{{x=a}}
\uniform a{{1,{upper}}}
\iter{{10}}
\response{{avg}}"""
    diagnostics = generation_error(source)
    assert diagnostics[0].code == "GEN005"


def test_generator_contract_discrete_values_must_remain_distinct_after_float64_conversion():
    second = "1." + ("0" * 399) + "1"
    source = rf"""\model{{x=a}}
\distrib a{{{{1,0.5}},{{{second},0.5}}}}
\iter{{10}}
\response{{avg}}"""
    diagnostics = generation_error(source)
    assert diagnostics[0].code == "GEN006"


def test_generator_contract_valid_extreme_but_representable_small_float_is_accepted():
    tiny = "0." + ("0" * 299) + "1"
    source = rf"""\model{{x=a+{tiny}}}
\normal a{{0,1}}
\iter{{10}}
\response{{avg}}"""
    script = generate_source(source, GeneratorConfig(seed=1))
    assert "1e-300" in script
