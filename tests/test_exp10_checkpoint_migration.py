from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_exp10_checkpoint_frontier_fix.py"
SPEC = importlib.util.spec_from_file_location("exp10_migration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_crlf_tree_reconstruction_uses_recorded_generator_hash(tmp_path: Path) -> None:
    root = tmp_path
    _write(root / "pyproject.toml", b"project = 'test'\r\n")
    _write(root / "requirements-aws-experiment.txt", b"numpy==2.4.6\r\n")
    _write(root / "benchmarks/run_v1_3_experiments.py", b"# runner\r\n")
    _write(root / "benchmarks/campaign_checkpoint.py", b"# checkpoint\r\n")
    _write(root / "benchmarks/models/model.dsl", b"model\r\n")
    _write(root / "src/montecarlo_dsl/generator.py", b"fixed generator\r\n")
    _write(root / "src/montecarlo_dsl/runtime.py", b"runtime\r\n")

    hashes = MIGRATION._source_tree_hashes(root)
    old_generator_hash = "a" * 64
    expected_old_tree = dict(hashes)
    expected_old_tree[MIGRATION.GENERATOR_RELATIVE] = old_generator_hash

    from benchmarks.campaign_checkpoint import _digest

    old_tree_digest = _digest(expected_old_tree)
    assert MIGRATION._reconstruct_previous_tree_digest(root, old_generator_hash) == old_tree_digest
    assert MIGRATION._normalized_generator_sha256(root) == hashlib.sha256(b"fixed generator\n").hexdigest()


def test_reconstruction_rejects_changes_outside_generator(tmp_path: Path) -> None:
    root = tmp_path
    _write(root / "pyproject.toml", b"project = 'test'\n")
    _write(root / "requirements-aws-experiment.txt", b"numpy==2.4.6\n")
    _write(root / "benchmarks/run_v1_3_experiments.py", b"# runner\n")
    _write(root / "benchmarks/campaign_checkpoint.py", b"# checkpoint\n")
    _write(root / "benchmarks/models/model.dsl", b"model\n")
    _write(root / "src/montecarlo_dsl/generator.py", b"fixed generator\n")
    _write(root / "src/montecarlo_dsl/runtime.py", b"runtime\n")

    hashes = MIGRATION._source_tree_hashes(root)
    old_generator_hash = "b" * 64
    expected_old_tree = dict(hashes)
    expected_old_tree[MIGRATION.GENERATOR_RELATIVE] = old_generator_hash

    from benchmarks.campaign_checkpoint import _digest

    old_tree_digest = _digest(expected_old_tree)
    _write(root / "src/montecarlo_dsl/runtime.py", b"changed runtime\n")

    assert MIGRATION._reconstruct_previous_tree_digest(root, old_generator_hash) != old_tree_digest
