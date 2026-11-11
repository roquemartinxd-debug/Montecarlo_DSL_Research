"""Regression tests for crash-safe and resumable v1.3.1 experiment campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from benchmarks.campaign_checkpoint import (
    CampaignCheckpoint,
    _write_json_atomic,
    simulation_identity,
)
from benchmarks import run_v1_3_experiments as runner
from montecarlo_dsl.generator import GeneratorConfig


@pytest.fixture
def campaign_identity() -> dict:
    return {"source_tree_sha256": "fixed-for-test", "confirmatory_reps": 30,
            "pilot_reps": 10, "ablation_reps": 10, "python": "3.11"}


def test_checkpoint_persists_each_completed_run_and_reuses_it(tmp_path: Path, campaign_identity: dict) -> None:
    directory = tmp_path / "checkpoints"
    case = {"tag": "exp0_test_r1", "kind": "subprocess_memory_probe", "source_sha256": "abc"}
    store = CampaignCheckpoint(directory, campaign_identity, resume=False)
    result = {"total_seconds": 1.25, "sha256": "fake", "execution_seconds": 0.9}
    store.save(case, result)
    assert store.created == 1
    assert store.load(case) == result
    assert store.reused == 1

    after_restart = CampaignCheckpoint(directory, campaign_identity, resume=True)
    assert after_restart.load(case) == result
    assert after_restart.created == 0
    assert after_restart.reused == 1
    assert after_restart.summary()["mixed_measurement_sessions"] is True


def test_resume_rejects_changed_parameters(tmp_path: Path, campaign_identity: dict) -> None:
    directory = tmp_path / "checkpoint"
    CampaignCheckpoint(directory, campaign_identity, resume=False)
    with pytest.raises(RuntimeError, match="different code/version"):
        CampaignCheckpoint(directory, {**campaign_identity, "confirmatory_reps": 31}, resume=True)
    with pytest.raises(RuntimeError, match="Use --resume"):
        CampaignCheckpoint(directory, campaign_identity, resume=False)


def test_resume_requires_existing_checkpoint(tmp_path: Path, campaign_identity: dict) -> None:
    with pytest.raises(RuntimeError, match="No checkpoint found"):
        CampaignCheckpoint(tmp_path / "missing", campaign_identity, resume=True)


def test_different_config_or_fitted_weights_is_a_different_record() -> None:
    from montecarlo_dsl.cost_model import InterpretableASTCostModel
    base = GeneratorConfig(seed=10)
    changed = GeneratorConfig(seed=11)
    model_a = InterpretableASTCostModel()
    model_b = InterpretableASTCostModel(weights_seconds_per_sample=(1e-6, 2e-8, 3e-8, 4e-8))
    a = simulation_identity(tag="run", kind="simulation", source="A", config=base, cost_model=model_a)
    b = simulation_identity(tag="run", kind="simulation", source="A", config=changed, cost_model=model_a)
    c = simulation_identity(tag="run", kind="simulation", source="A", config=base, cost_model=model_b)
    assert a != b and a != c


def test_modified_checkpoint_record_is_rejected(tmp_path: Path, campaign_identity: dict) -> None:
    store = CampaignCheckpoint(tmp_path, campaign_identity, resume=False)
    case = {"tag": "exp5_test", "kind": "subprocess_memory_probe"}
    store.save(case, {"total_seconds": 1.25})
    record_path = store._record_path(case)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["result"]["total_seconds"] = 99999
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupted"):
        store.load(case)


def test_json_replace_failure_preserves_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "state.json"
    _write_json_atomic(path, {"status": "original"})
    def break_replace(a, b):
        raise OSError("simulated power interruption")
    monkeypatch.setattr(os, "replace", break_replace)
    with pytest.raises(OSError, match="simulated"):
        _write_json_atomic(path, {"status": "new"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "original"}
    assert list(tmp_path.glob(".checkpoint_*.tmp")) == []


def test_resume_real_simulation_with_compiled_data(tmp_path: Path, campaign_identity: dict,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    source = runner.source_with_iterations("normal_mean", 2_000)
    config = GeneratorConfig(seed=101, max_workers=1, scheduler_mode="fixed", max_iterations=2_000)
    path = tmp_path / "checkpoint"
    first_store = CampaignCheckpoint(path, campaign_identity, resume=False)
    monkeypatch.setattr(runner, "_CHECKPOINT", first_store)
    with tempfile.TemporaryDirectory() as work:
        first = runner.run_simulation(source, config, temp_dir=Path(work), tag="exp0_resume_test")
        assert first_store.created == 1
        counter_after_run = runner._MODULE_COUNTER
        monkeypatch.setattr(runner, "_CHECKPOINT", CampaignCheckpoint(path, campaign_identity, resume=True))
        second = runner.run_simulation(source, config, temp_dir=Path(work), tag="exp0_resume_test")
        assert runner._MODULE_COUNTER == counter_after_run
        assert second["sha256"] == first["sha256"]
        assert second["total_seconds"] == first["total_seconds"]
        assert second["compiled_metrics"] == first["compiled_metrics"]
        assert second["script_bytes"] == first["script_bytes"]
        assert runner._CHECKPOINT.reused == 1


def test_warmups_are_repeated_after_resume(tmp_path: Path, campaign_identity: dict,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    source = runner.source_with_iterations("normal_mean", 1_000)
    config = GeneratorConfig(seed=101, max_workers=1, scheduler_mode="fixed", max_iterations=1_000)
    store = CampaignCheckpoint(tmp_path, campaign_identity, resume=False)
    monkeypatch.setattr(runner, "_CHECKPOINT", store)
    with tempfile.TemporaryDirectory() as work:
        runner.run_simulation(source, config, temp_dir=Path(work), tag="warmup_normal_mean")
        runner.run_simulation(source, config, temp_dir=Path(work), tag="warmup_normal_mean")
    assert store.created == store.reused == 0
    assert store.warmups == 2


def test_memory_subprocess_probe_is_resume_safe(tmp_path: Path, campaign_identity: dict,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    source = runner.source_with_iterations("normal_mean", 2_000)
    config = GeneratorConfig(seed=101, max_workers=1, scheduler_mode="fixed", max_iterations=2_000)
    path = tmp_path / "checkpoints"
    first_store = CampaignCheckpoint(path, campaign_identity, resume=False)
    monkeypatch.setattr(runner, "_CHECKPOINT", first_store)
    with tempfile.TemporaryDirectory() as work:
        first = runner._run_subprocess_memory_probe(
            source, config, temp_dir=Path(work), tag="exp8_probe_resume"
        )
        assert first["complete_event"]["valid_results"] == 2_000
        assert first_store.created == 1
        monkeypatch.setattr(runner, "_CHECKPOINT", CampaignCheckpoint(path, campaign_identity, resume=True))
        monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not rerun")))
        second = runner._run_subprocess_memory_probe(
            source, config, temp_dir=Path(work), tag="exp8_probe_resume"
        )
        assert second == first


def test_fitted_model_tuple_survives_json_restart(tmp_path: Path, campaign_identity: dict) -> None:
    from montecarlo_dsl.cost_model import InterpretableASTCostModel
    model = InterpretableASTCostModel(weights_seconds_per_sample=(1e-6, 2e-8, 3e-8, 4e-8))
    case = simulation_identity(tag="exp8_model_r1", kind="subprocess_memory_probe", source="X",
                               config=GeneratorConfig(seed=101), cost_model=model)
    assert isinstance(case["cost_model"]["weights_seconds_per_sample"], list)
    initial = CampaignCheckpoint(tmp_path, campaign_identity, resume=False)
    initial.save(case, {"total_seconds": 2.5})
    restarted = CampaignCheckpoint(tmp_path, campaign_identity, resume=True)
    assert restarted.load(case) == {"total_seconds": 2.5}


def test_real_simulation_fitted_model_can_resume(tmp_path: Path, campaign_identity: dict,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    from montecarlo_dsl.cost_model import InterpretableASTCostModel
    source = runner.source_with_iterations("normal_mean", 1_000)
    config = GeneratorConfig(seed=101, max_workers=1, scheduler_mode="fixed", max_iterations=1_000)
    model = InterpretableASTCostModel(weights_seconds_per_sample=(1e-6, 2e-8, 3e-8, 4e-8))
    store = CampaignCheckpoint(tmp_path / "checkpoint", campaign_identity, resume=False)
    monkeypatch.setattr(runner, "_CHECKPOINT", store)
    with tempfile.TemporaryDirectory() as work:
        first = runner.run_simulation(source, config, temp_dir=Path(work),
                                      tag="exp0_normal_mean_r1", cost_model=model)
        assert store.created == 1
        monkeypatch.setattr(runner, "_CHECKPOINT",
                            CampaignCheckpoint(store.directory, campaign_identity, resume=True))
        second = runner.run_simulation(source, config, temp_dir=Path(work),
                                       tag="exp0_normal_mean_r1", cost_model=model)
        assert first["sha256"] == second["sha256"]
        assert first["total_seconds"] == second["total_seconds"]


def test_main_interrupted_then_resume_without_remeasuring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real --resume orchestration, not only the cache data structure."""
    from montecarlo_dsl.cost_model import InterpretableASTCostModel
    source = runner.source_with_iterations("normal_mean", 1_000)
    config = GeneratorConfig(seed=101, max_workers=1, scheduler_mode="fixed", max_iterations=1_000)
    fitted = InterpretableASTCostModel()
    outputs = tmp_path / "results"
    ckpt = tmp_path / "checkpoint"
    monkeypatch.setattr(runner, "OUTPUT_DIR", outputs)
    raised = {"once": False}

    def fake_exp5(temp_dir, *, reps):
        runner.run_simulation(source, config, temp_dir=temp_dir, tag="exp5_restart_test")
        return [], [], fitted

    def fake_exp0(temp_dir, *, reps, cost_model):
        runner.run_simulation(source, config, temp_dir=temp_dir,
                              tag="exp0_restart_test", cost_model=cost_model)
        if not raised["once"]:
            raised["once"] = True
            raise RuntimeError("simulated interruption")
        return []

    monkeypatch.setattr(runner, "run_exp5_ast_cost_generalization", fake_exp5)
    monkeypatch.setattr(runner, "run_exp0_v1_2_vs_v1_3_baseline_comparison", fake_exp0)
    for name in (
        "run_exp1_functional_equivalence", "run_exp2_performance_and_scalability",
        "run_exp3_batch_adaptation", "run_exp4_schedule_invariance_reproducibility",
        "run_exp6_compiler_optimization", "run_exp7_statistical_validation_25seeds",
        "run_exp8_memory_consumption", "run_exp9_ablation_study",
    ):
        monkeypatch.setattr(runner, name, lambda *a, **kw: None)

    options = ["run_v1_3_experiments.py", "--confirmatory-reps", "1",
               "--pilot-reps", "1", "--ablation-reps", "1",
               "--checkpoint-dir", str(ckpt), "--output-dir", str(outputs)]
    monkeypatch.setattr(runner.sys, "argv", options)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.main()
    assert json.loads((ckpt / "campaign_status.json").read_text())["state"] == "interrupted"
    assert len(list((ckpt / "runs").rglob("*.json"))) == 2

    monkeypatch.setattr(runner.sys, "argv", options + ["--resume"])
    assert runner.main() == 0
    status = json.loads((ckpt / "campaign_status.json").read_text())
    assert status["state"] == "completed"
    assert status["new_simulations"] == 0
    assert status["reused_simulations"] == 2
    assert json.loads((outputs / "environment.json").read_text())["checkpoint_provenance"]["mixed_measurement_sessions"]


def test_campaign_runner_direct_script_invocation_resolves_benchmarks_package() -> None:
    """The documented PowerShell command must work without setting PYTHONPATH."""
    project_root = Path(runner.__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(project_root / "benchmarks" / "run_v1_3_experiments.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--resume" in completed.stdout
    assert "--confirmatory-reps" in completed.stdout
