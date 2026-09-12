from __future__ import annotations

import json
from pathlib import Path
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


REFERENCE_MODEL_SMALL = r"""\model{x=\frac{-b\pm\sqrt{b^2-4a c}}{2a}}
\normal a{10,0.5}
\distrib b{{10,0.5},{15,0.3},{20,0.2}}
\uniform c{-20,30}
\iter{400}
\response{avg,min,max}"""


def run_generated(tmp_path: Path, source: str, *, seed: int = 12345, batch_size: int = 50):
    script = generate_source(
        source,
        GeneratorConfig(seed=seed, batch_size=batch_size, max_workers=2, histogram_bins=30),
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "generated_simulation.py"
    path.write_text(script, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(path)],
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


def test_generator_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        GeneratorConfig(batch_size=0)
    with pytest.raises(ValueError):
        GeneratorConfig(max_workers=0)
    with pytest.raises(ValueError):
        GeneratorConfig(histogram_bins=0)
    with pytest.raises(ValueError):
        GeneratorConfig(seed=-1)


def test_expression_visitor_generates_two_branches_for_plus_minus():
    validated = analyze_source(
        r"""\model{x=a\pm1}
\normal a{0,1}
\iter{10}
\response{avg}"""
    )
    generated = NumpyExpressionVisitor().visit(validated.ast.model.expression)

    assert len(generated.branches) == 2
    assert "+" in generated.branches[0]
    assert "-" in generated.branches[1]


def test_expression_visitor_preserves_numpy_operations():
    validated = analyze_source(
        r"""\model{x=\frac{\sqrt{a^2}}{2a}}
\normal a{1,1}
\iter{10}
\response{avg}"""
    )
    branch = NumpyExpressionVisitor().visit(validated.ast.model.expression).branches[0]

    assert "np.sqrt" in branch
    assert "np.power" in branch
    assert "/" in branch
    assert "*" in branch


def test_generated_script_is_standalone_and_uses_required_runtime_components():
    validated = analyze_source(REFERENCE_MODEL_SMALL)
    script = generate_python_script(
        validated,
        GeneratorConfig(seed=99, batch_size=100, max_workers=4, histogram_bins=30),
    )

    compile(script, "<generated>", "exec")
    assert "ProcessPoolExecutor" in script
    assert "np.random.default_rng" in script
    assert "BRANCH_IDS = ('plus', 'minus')" in script
    assert "BRANCH_COUNT = len(BRANCH_IDS)" in script
    assert "HISTOGRAM_BINS = 30" in script
    assert "MASTER_SEED = 99" in script
    assert "json.dumps" in script
    assert "eval(" not in script
    assert "exec(" not in script


def test_generated_normal_model_executes_and_emits_exact_histogram(tmp_path):
    source = r"""\model{x=a}
\normal a{0,1}
\iter{200}
\response{avg,min,max}"""
    _, completed, messages = run_generated(tmp_path, source, seed=7, batch_size=50)

    assert completed.returncode == 0, completed.stderr
    assert messages[0]["type"] == "start"
    assert messages[0]["seed"] == "7"
    assert messages[0]["batch_count"] == 4
    assert messages[0]["workers"] == 2
    assert messages[-1]["type"] == "complete"

    final = messages[-1]
    assert final["processed_iterations"] == 200
    assert final["valid_results"] == 200
    assert final["discarded_results"] == 0
    assert final["histogram"]["total_count"] == 200
    assert sum(item["count"] for item in final["histogram"]["bins"]) == 200
    assert len(final["histogram"]["bins"]) == 30
    assert set(final["stats"]) == {"avg", "min", "max"}


def test_generated_plus_minus_model_counts_two_results_per_iteration(tmp_path):
    source = r"""\model{x=a\pm1}
\normal a{0,1}
\iter{120}
\response{avg,min,max}"""
    _, completed, messages = run_generated(tmp_path, source, seed=31, batch_size=40)

    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert final["type"] == "complete"
    assert final["candidate_results"] == 240
    assert final["valid_results"] == 240
    assert final["discarded_results"] == 0
    assert final["histogram"] is None
    assert [branch["id"] for branch in final["branches"]] == ["plus", "minus"]
    for branch in final["branches"]:
        assert branch["valid_results"] == 120
        assert branch["discarded_results"] == 0
        assert branch["histogram"]["total_count"] == 120
        assert sum(item["count"] for item in branch["histogram"]["bins"]) == 120
        assert branch["diagnostics"]["mcse_mean"] is not None


def test_runtime_dependent_invalid_results_are_discarded(tmp_path):
    source = r"""\model{x=\sqrt{a}}
\uniform a{-1,1}
\iter{300}
\response{avg,min,max}"""
    _, completed, messages = run_generated(tmp_path, source, seed=123, batch_size=60)

    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert 0 < final["valid_results"] < 300
    assert final["valid_results"] + final["discarded_results"] == 300
    assert final["histogram"]["total_count"] == final["valid_results"]


def test_response_controls_public_stats_but_histogram_remains_exact(tmp_path):
    source = r"""\model{x=a}
\uniform a{1,2}
\iter{80}
\response{avg}"""
    _, completed, messages = run_generated(tmp_path, source, seed=5, batch_size=20)

    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert set(final["stats"]) == {"avg"}
    assert final["histogram"]["data_min"] is not None
    assert final["histogram"]["data_max"] is not None
    assert final["histogram"]["total_count"] == 80


def test_same_seed_is_reproducible_despite_parallel_completion_order(tmp_path):
    source = r"""\model{x=a+b}
\normal a{0,1}
\uniform b{-2,3}
\iter{240}
\response{avg,min,max}"""

    _, first_process, first = run_generated(tmp_path / "first", source, seed=777, batch_size=30)
    _, second_process, second = run_generated(tmp_path / "second", source, seed=777, batch_size=30)

    assert first_process.returncode == 0, first_process.stderr
    assert second_process.returncode == 0, second_process.stderr

    first_final = first[-1]
    second_final = second[-1]
    assert first_final["type"] == "complete"
    assert second_final["type"] == "complete"
    assert first_final["stats"] == second_final["stats"]
    assert first_final["histogram"] == second_final["histogram"]
    assert first_final["valid_results"] == second_final["valid_results"]
    assert first_final["discarded_results"] == second_final["discarded_results"]


def test_discrete_distribution_is_generated_and_executes(tmp_path):
    source = r"""\model{x=a}
\distrib a{{-1,0.25},{2,0.75}}
\iter{100}
\response{min,max}"""
    script, completed, messages = run_generated(tmp_path, source, seed=42, batch_size=25)

    assert "rng.choice" in script
    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert final["stats"]["min"] == -1.0
    assert final["stats"]["max"] == 2.0
    assert set(final["stats"]) == {"min", "max"}


def test_reference_quadratic_model_executes_with_filtered_results(tmp_path):
    _, completed, messages = run_generated(
        tmp_path,
        REFERENCE_MODEL_SMALL,
        seed=1234,
        batch_size=100,
    )

    assert completed.returncode == 0, completed.stderr
    final = messages[-1]
    assert final["type"] == "complete"
    assert final["processed_iterations"] == 400
    assert 0 < final["valid_results"] <= 800
    assert final["valid_results"] + final["discarded_results"] == 800
    assert final["candidate_results"] == 800
    assert final["histogram"] is None
    assert len(final["branches"]) == 2
    for branch in final["branches"]:
        assert branch["valid_results"] + branch["discarded_results"] == 400
        assert branch["histogram"]["total_count"] == branch["valid_results"]


def test_all_runtime_results_invalid_emit_error_event(tmp_path):
    source = r"""\model{x=\sqrt{a}}
\uniform a{-2,-1}
\iter{60}
\response{avg}"""
    _, completed, messages = run_generated(tmp_path, source, seed=2, batch_size=20)

    assert completed.returncode == 1
    assert messages[-1]["type"] == "error"
    assert "sin ningun resultado" in messages[-1]["message"]


def test_unrepresentable_float64_literal_is_generation_error():
    huge = "9" * 400
    source = rf"""\model{{x=a+{huge}}}
\normal a{{0,1}}
\iter{{10}}
\response{{avg}}"""
    validated = analyze_source(source)

    with pytest.raises(CompilationError) as exc:
        generate_python_script(validated, GeneratorConfig(seed=1))

    assert exc.value.diagnostics[0].phase.name == "GENERATION"
    assert exc.value.diagnostics[0].code == "GEN001"
