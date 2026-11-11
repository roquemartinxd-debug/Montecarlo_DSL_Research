#!/usr/bin/env python3
"""Safely migrate the Windows EXP-10 exploratory checkpoint after the bounded-frontier fix.

This migration is line-ending aware. It verifies that every source input except
`generator.py` is byte-identical to the source tree recorded by the checkpoint,
then verifies that the current generator normalizes to the reviewed fixed source.
The manifest is updated with the *actual local byte hashes*, so `--resume` uses
the same fingerprint algorithm as the campaign runner on Windows (CRLF) or Linux (LF).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from benchmarks.campaign_checkpoint import _digest, _write_json_atomic, source_tree_digest

# Fingerprints observed in the user's actual Windows checkpoint and reviewed hotfix.
EXPECTED_OLD_GENERATOR_SHA256 = "bf3d67fb65119bed27af42f43072c3cf55fa2a19e70a74b7c85145a023b0a26a"
EXPECTED_FIXED_GENERATOR_LF_SHA256 = "0f94f15aa7064f9df9745ed1ce57927db4c76c47f692b10a7602d99a1fdd46f7"
EXPECTED_WEIGHTS = [
    2.2583574403375308e-07,
    1.7661118774571211e-08,
    8.9818060129072783e-08,
    5.5e-09,
]
MODELS = ("normal_mean", "linear_combination", "heavy_expression", "quadratic_pm")
LEVELS = (100_000, 250_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000)
CONFIGS = ("serial_in_process", "fixed_pool_w1", "fixed_pool_w2", "fixed_pool_w4", "adaptive_hybrid")
EXPECTED_TAGS = {
    f"exp10_exploratory_{model}_{iterations}_{config}_r{rep}"
    for model in MODELS
    for iterations in LEVELS
    for config in CONFIGS
    for rep in range(1, 11)
}
GENERATOR_RELATIVE = "src/montecarlo_dsl/generator.py"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_tree_hashes(root: Path) -> dict[str, str]:
    """Mirror campaign_checkpoint.source_tree_digest, returning per-file hashes."""
    files = [
        root / "pyproject.toml",
        root / "requirements-aws-experiment.txt",
        root / "benchmarks" / "run_v1_3_experiments.py",
        root / "benchmarks" / "campaign_checkpoint.py",
    ]
    files.extend((root / "src" / "montecarlo_dsl").rglob("*.py"))
    files.extend((root / "benchmarks" / "models").rglob("*.dsl"))
    contents: dict[str, str] = {}
    for path in sorted(files, key=lambda p: p.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(f"Missing campaign input: {path}")
        contents[path.relative_to(root).as_posix()] = _sha256_file(path)
    if GENERATOR_RELATIVE not in contents:
        raise RuntimeError(f"Campaign generator is missing: {GENERATOR_RELATIVE}")
    return contents


def _reconstruct_previous_tree_digest(root: Path, previous_generator_sha256: str) -> str:
    """Reconstruct the checkpoint's source tree using current files except old generator hash."""
    contents = _source_tree_hashes(root)
    contents[GENERATOR_RELATIVE] = previous_generator_sha256
    return _digest(contents)


def _normalized_generator_sha256(root: Path) -> str:
    """Normalize CRLF to LF only for comparison with the reviewed canonical generator."""
    data = (root / GENERATOR_RELATIVE).read_bytes()
    return _sha256_bytes(data.replace(b"\r\n", b"\n"))


def _resolve_manifest(checkpoint_arg: Path) -> Path:
    candidate = checkpoint_arg.expanduser().resolve()
    if candidate.name == "manifest.json":
        return candidate
    if (candidate / "manifest.json").is_file():
        return candidate / "manifest.json"
    nested = candidate / "break-even_exploratory" / "manifest.json"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"No EXP-10 checkpoint manifest found under: {candidate}")


def migrate(checkpoint_arg: Path) -> int:
    manifest_path = _resolve_manifest(checkpoint_arg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})

    if manifest.get("schema") != 1:
        raise RuntimeError("Unsupported checkpoint schema; refusing migration.")
    previous_tree = identity.get("source_tree_sha256")
    previous_generator = identity.get("generator_sha256")
    if not isinstance(previous_tree, str) or len(previous_tree) != 64:
        raise RuntimeError("Checkpoint source-tree fingerprint is missing or invalid.")
    if previous_generator != EXPECTED_OLD_GENERATOR_SHA256:
        raise RuntimeError("Checkpoint generator fingerprint is not the expected pre-fix Windows version.")
    if identity.get("study") != "break-even" or identity.get("phase") != "exploratory":
        raise RuntimeError("Only the Windows EXP-10 exploratory checkpoint can be migrated.")
    if identity.get("platform_label") != "windows-2core" or identity.get("physical_cores_declared") != 2:
        raise RuntimeError("Checkpoint is not the expected Windows 2-core campaign.")
    if identity.get("campaign_label") != "windows-v1.3.1-exp10-exploratory":
        raise RuntimeError("Campaign label does not match the expected exploratory run.")
    if identity.get("experiment_profile") != "targeted-windows-2core":
        raise RuntimeError("Experiment profile does not match the expected Windows profile.")
    if identity.get("targeted_protocol_version") != "1.0.0":
        raise RuntimeError("Targeted experiment protocol version does not match the expected campaign.")
    if identity.get("exploration_reps") != 10 or identity.get("confirmatory_reps") != 30:
        raise RuntimeError("Exploration/confirmation repetition counts do not match the expected campaign.")
    targeted_script = ROOT / "benchmarks" / "run_v1_3_targeted_experiments.py"
    if _sha256_file(targeted_script) != identity.get("targeted_script_sha256"):
        raise RuntimeError("Targeted experiment script differs from the version used by the checkpoint.")
    if identity.get("cost_model_weights_seconds_per_sample") != EXPECTED_WEIGHTS:
        raise RuntimeError("Cost-model weights differ from the recorded Windows campaign.")
    if not str(identity.get("output_dir", "")).replace("\\", "/").endswith(
        "benchmarks/results/targeted/windows_exp10_exploratory"
    ):
        raise RuntimeError("Output directory does not match the expected EXP-10 exploratory output.")

    # Prove that all campaign inputs other than generator.py are byte-identical
    # to those recorded at checkpoint time. This is CRLF-safe and does not guess
    # a platform-independent source-tree hash.
    reconstructed_previous_tree = _reconstruct_previous_tree_digest(ROOT, previous_generator)
    if reconstructed_previous_tree != previous_tree:
        raise RuntimeError(
            "The current campaign inputs differ from the checkpoint source tree beyond generator.py; "
            "refusing to migrate existing measurements."
        )

    normalized_generator = _normalized_generator_sha256(ROOT)
    if normalized_generator != EXPECTED_FIXED_GENERATOR_LF_SHA256:
        raise RuntimeError(
            "Current generator does not normalize to the reviewed bounded-frontier hotfix."
        )

    # Store the exact local fingerprints used by source_tree_digest() on this OS.
    current_generator = _sha256_file(ROOT / GENERATOR_RELATIVE)
    current_tree = source_tree_digest(ROOT)

    runs_dir = manifest_path.parent / "runs"
    records = sorted(runs_dir.rglob("*.json")) if runs_dir.is_dir() else []
    if len(records) != len(EXPECTED_TAGS):
        raise RuntimeError(
            f"Expected exactly {len(EXPECTED_TAGS)} completed EXP-10 records; found {len(records)}. "
            "No manifest changes were made."
        )

    seen_tags: set[str] = set()
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        run_identity = record.get("identity", {})
        tag = run_identity.get("tag")
        if record.get("schema") != 1 or run_identity.get("kind") != "simulation":
            raise RuntimeError(f"Unexpected checkpoint record: {record_path}")
        if tag not in EXPECTED_TAGS or tag in seen_tags:
            raise RuntimeError(f"Unexpected or duplicate simulation tag: {tag!r}")
        if record.get("result_sha256") != _digest(record.get("result")):
            raise RuntimeError(f"Checkpoint result checksum mismatch: {record_path}")
        seen_tags.add(tag)

    missing = EXPECTED_TAGS - seen_tags
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(f"Checkpoint set is incomplete; missing tags include: {sample}")

    old_fingerprint = manifest.get("fingerprint")
    new_identity = dict(identity)
    new_identity["source_tree_sha256"] = current_tree
    new_identity["generator_sha256"] = current_generator
    manifest["identity"] = new_identity
    manifest["fingerprint"] = _digest(new_identity)
    migrations = list(manifest.get("migrations", []))
    migrations.append({
        "migration": "exp10-bounded-contiguous-task-frontier-fix-windows-crlf-aware",
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": manifest["fingerprint"],
        "old_source_tree_sha256": previous_tree,
        "old_generator_sha256": previous_generator,
        "new_source_tree_sha256": current_tree,
        "new_generator_sha256": current_generator,
        "normalized_fixed_generator_sha256": normalized_generator,
        "verified_records": len(records),
        "note": (
            "Verified that all source-tree inputs other than generator.py are byte-identical to the "
            "checkpoint source. The current generator normalizes to the reviewed bounded-frontier fix. "
            "Manifest fingerprints were updated using the exact local bytes (including Windows CRLF) "
            "so campaign resume computes the same identity. Existing simulation records were not modified."
        ),
    })
    manifest["migrations"] = migrations
    _write_json_atomic(manifest_path, manifest)
    print(f"MIGRATION PASS: {len(records)} checkpoint records verified.")
    print(f"Previous source tree reconstructed: {reconstructed_previous_tree}")
    print(f"Fixed generator normalized SHA-256: {normalized_generator}")
    print(f"New local source tree SHA-256: {current_tree}")
    print(f"New local generator SHA-256: {current_generator}")
    print(f"Manifest updated: {manifest_path}")
    print("Previously completed 1,400 exploratory simulations remain reusable.")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, required=True,
        help="Parent checkpoint directory or the break-even_exploratory directory containing manifest.json",
    )
    args = parser.parse_args()
    migrate(args.checkpoint_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
