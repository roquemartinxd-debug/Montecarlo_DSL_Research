from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from montecarlo_dsl import CompilationError, analyze_source, parse, tokenize
from montecarlo_dsl.ast import VariableReferenceNode
from montecarlo_dsl.compiler import compile_source
from montecarlo_dsl.generator import GeneratorConfig, generate_source
from montecarlo_dsl.job_manager import Job, JobStatus
from montecarlo_dsl.server import create_app
from montecarlo_dsl.job_manager import JobManager
from montecarlo_dsl.tokens import TokenType


PREFIX = "MC_DSL_EVENT:"


def _run_generated(tmp_path: Path, source: str, seed: int = 77):
    script = generate_source(
        source,
        GeneratorConfig(seed=seed, batch_size=25, max_workers=2, histogram_bins=12),
    )
    path = tmp_path / "generated.py"
    path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    events = [
        json.loads(line[len(PREFIX):])
        for line in completed.stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    return completed, events


def test_multicharacter_identifiers_work_end_to_end(tmp_path: Path) -> None:
    source = r"""\model{resultado=precio-demanda}
\normal precio{100,4}
\uniform demanda{20,30}
\iter{100}
\response{avg,min,max}"""
    completed, events = _run_generated(tmp_path, source)
    assert completed.returncode == 0, completed.stderr
    assert events[-1]["type"] == "complete"
    assert events[-1]["valid_results"] == 100


def test_identifiers_that_are_python_keywords_are_mangled_in_generated_script() -> None:
    source = r"""\model{resultado=class+for}
\normal class{1,0.1}
\normal for{2,0.1}
\iter{10}
\response{avg}"""
    script = compile_source(source, GeneratorConfig(seed=1)).script
    compile(script, "<generated>", "exec")
    assert "_mc_var_0" in script and "_mc_var_1" in script
    assert "class = rng." not in script
    assert "for = rng." not in script


def test_identifier_lexical_contract_supports_underscore_and_digits() -> None:
    tokens = tokenize("tasa_interes2")
    assert [(token.type, token.lexeme) for token in tokens] == [
        (TokenType.VARIABLE, "tasa_interes2"),
        (TokenType.EOF, ""),
    ]


def test_discrete_pairs_require_explicit_external_separator() -> None:
    valid = r"""\model{resultado=demanda}
\distrib demanda{{10,0.5},{20,0.5}}
\iter{10}
\response{avg}"""
    parse(valid)

    invalid = valid.replace("},{", "}{")
    with pytest.raises(CompilationError) as exc:
        parse(invalid)
    assert exc.value.diagnostics[0].code == "SYN022"


def test_plus_minus_keeps_branches_statistically_separate(tmp_path: Path) -> None:
    source = r"""\model{resultado=base\pm5}
\normal base{10,1}
\iter{100}
\response{avg,min,max}"""
    completed, events = _run_generated(tmp_path, source, seed=12)
    assert completed.returncode == 0, completed.stderr
    final = events[-1]
    assert final["candidate_results"] == 200
    assert final["histogram"] is None
    plus, minus = final["branches"]
    assert plus["id"] == "plus" and minus["id"] == "minus"
    assert plus["valid_results"] == 100
    assert minus["valid_results"] == 100
    assert plus["stats"]["avg"] > minus["stats"]["avg"]
    assert plus["histogram"]["total_count"] == 100
    assert minus["histogram"]["total_count"] == 100


def test_each_branch_reports_monte_carlo_standard_error_and_ci95(tmp_path: Path) -> None:
    source = r"""\model{resultado=base}
\normal base{10,2}
\iter{100}
\response{avg}"""
    completed, events = _run_generated(tmp_path, source, seed=22)
    assert completed.returncode == 0, completed.stderr
    diagnostics = events[-1]["branches"][0]["diagnostics"]
    assert diagnostics["n"] == 100
    assert diagnostics["sample_stddev"] > 0
    assert diagnostics["mcse_mean"] > 0
    assert len(diagnostics["ci95_mean"]) == 2
    assert diagnostics["ci95_mean"][0] < diagnostics["ci95_mean"][1]


@pytest.mark.asyncio
async def test_non_protocol_stdout_cannot_corrupt_structured_event_channel() -> None:
    script = f'''import json\nprint("diagnostico libre", flush=True)\nprint("{PREFIX}" + json.dumps({{"type":"start","iterations":1}}), flush=True)\nprint("otra salida libre", flush=True)\nprint("{PREFIX}" + json.dumps({{"type":"complete","processed_iterations":1,"total_iterations":1,"done":True}}), flush=True)\n'''
    job = Job(job_id="noise", iterations=1, script=script, seed=1)
    await job.run()
    assert job.status is JobStatus.COMPLETED
    assert [event["type"] for event in job.hub.history] == ["start", "complete"]
    assert "diagnostico libre" in job.stdout_diagnostics_tail
    assert "otra salida libre" in job.stdout_diagnostics_tail


def test_compilation_failure_never_creates_runtime_job() -> None:
    manager = JobManager(batch_size=10, max_workers=1)
    with TestClient(create_app(manager)) as client:
        response = client.post(
            "/api/jobs",
            json={"source": r"\model{resultado=desconocida}\normal entrada{0,1}\iter{10}\response{avg}"},
        )
    assert response.status_code == 422
    assert response.json()["error"] == "compilation_failed"
    assert manager.jobs == {}


def test_long_identifier_is_a_single_ast_reference() -> None:
    source = r"""\model{resultado=tasa_interes}
\normal tasa_interes{0,1}
\iter{10}
\response{avg}"""
    program = analyze_source(source)
    expression = program.ast.model.expression
    assert isinstance(expression, VariableReferenceNode)
    assert expression.name == "tasa_interes"
