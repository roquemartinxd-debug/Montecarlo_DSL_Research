"""Crash-safe, per-simulation checkpoints for MC DSL v1.3.1 experiments.

The files are an audit/recovery artefact, not a replacement for the published
CSV tables. A checkpoint can be resumed only with the same code, model files,
Python/dependency environment, machine and experiment repetition parameters.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any
from uuid import uuid4


CHECKPOINT_SCHEMA = 1


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(obj: Any) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Commit a complete JSON record with an atomic rename on the same volume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".checkpoint_", suffix=".tmp", delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def source_tree_digest(root: Path) -> str:
    """Hash everything that can change compilation, models or experimental protocol."""
    files = [root / "pyproject.toml", root / "requirements-aws-experiment.txt",
             root / "benchmarks" / "run_v1_3_experiments.py",
             root / "benchmarks" / "campaign_checkpoint.py"]
    files.extend((root / "src" / "montecarlo_dsl").rglob("*.py"))
    files.extend((root / "benchmarks" / "models").rglob("*.dsl"))
    contents: dict[str, str] = {}
    for path in sorted(files, key=lambda p: p.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(f"Missing campaign input: {path}")
        contents[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _digest(contents)


def campaign_identity(root: Path, *, confirmatory_reps: int, pilot_reps: int,
                      ablation_reps: int, campaign_label: str = "local-v1.3.1",
                      experiment_profile: str = "standard", instance_type: str | None = None,
                      cloud_region: str | None = None, output_dir: str | None = None) -> dict[str, Any]:
    """Treat results gathered under different experimental conditions as different campaigns."""
    from importlib import metadata
    import numpy
    import scipy

    return {
        "source_tree_sha256": source_tree_digest(root),
        "confirmatory_reps": int(confirmatory_reps),
        "pilot_reps": int(pilot_reps),
        "ablation_reps": int(ablation_reps),
        "campaign_label": str(campaign_label),
        "experiment_profile": str(experiment_profile),
        "instance_type": instance_type,
        "cloud_region": cloud_region,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "fastapi": metadata.version("fastapi"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "checkpoint_schema": CHECKPOINT_SCHEMA,
    }


def simulation_identity(*, tag: str, kind: str, source: str, config: Any,
                        cost_model: Any) -> dict[str, Any]:
    """Never confuse different seeds, compiler configurations or fitted weights."""
    if not is_dataclass(config):
        raise TypeError("GeneratorConfig must be a dataclass")
    if cost_model is not None and not is_dataclass(cost_model):
        raise TypeError("Cost model must be a dataclass")
    # JSON has arrays, not tuples. Normalize here so the in-memory identity
    # matches exactly what is loaded after a restart (including fitted weights).
    return json.loads(_stable_json({
        "tag": str(tag),
        "kind": str(kind),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "config": asdict(config),
        "cost_model": asdict(cost_model) if cost_model is not None else None,
    }))


class CampaignCheckpoint:
    def __init__(self, directory: Path, identity: dict[str, Any], *, resume: bool,
                 progress_every: int = 1):
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.identity = identity
        self.fingerprint = _digest(identity)
        self.session_id = uuid4().hex
        self.created = 0
        self.reused = 0
        self.warmups = 0
        self.progress_every = max(1, int(progress_every))
        self.seen_session_ids: set[str] = {self.session_id}
        self.directory.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.is_file():
            if not resume:
                raise RuntimeError(
                    f"Checkpoint already exists at {self.directory}. Use --resume "
                    "for this exact campaign or select a NEW --checkpoint-dir."
                )
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Unreadable checkpoint manifest: {self.manifest_path}") from exc
            if manifest.get("schema") != CHECKPOINT_SCHEMA or manifest.get("fingerprint") != self.fingerprint or manifest.get("identity") != identity:
                raise RuntimeError(
                    "Checkpoint belongs to a different code/version, machine, environment "
                    "or repetition configuration. Do NOT mix runs; use a NEW --checkpoint-dir. "
                    f"Existing: {self.manifest_path}"
                )
        elif resume:
            raise RuntimeError(
                f"No checkpoint found at {self.manifest_path}; remove --resume to start a fresh campaign."
            )
        else:
            if any(self.directory.iterdir()):
                raise RuntimeError(f"Checkpoint directory is not empty: {self.directory}")
            _write_json_atomic(self.manifest_path, {
                "schema": CHECKPOINT_SCHEMA,
                "fingerprint": self.fingerprint,
                "identity": identity,
                "created_at_utc": _utc_now(),
                "description": "Per-successful-simulation checkpoints; performance values measured in their original session",
            })

        self._started_at = _utc_now()
        print(
            f"[checkpoint] {'RESUME' if resume else 'NEW'} "
            f"directory={self.directory} session={self.session_id[:8]}", flush=True,
        )

    def _record_path(self, identity: dict[str, Any]) -> Path:
        key = _digest(identity)
        return self.directory / "runs" / key[:2] / f"{key}.json"

    def load(self, identity: dict[str, Any]) -> dict[str, Any] | None:
        path = self._record_path(identity)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (record.get("schema") != CHECKPOINT_SCHEMA
                    or record.get("identity") != identity
                    or record.get("result_sha256") != _digest(record["result"])):
                raise ValueError("Identity or result checksum mismatch")
            result = record["result"]
            if identity["kind"] == "simulation":
                from hashlib import sha256
                # Verify the statistical event was recorded intact, independently
                # from the checkpoint's outer JSON digest.
                if result["sha256"] != sha256(_stable_json({
                    field: result["complete_event"].get(field) for field in (
                        "processed_iterations", "total_iterations", "candidate_results",
                        "valid_results", "discarded_results", "branches", "paired",
                        "stats", "histogram", "diagnostics",
                    )
                }).encode("utf-8")).hexdigest():
                    raise ValueError("Simulation result hash mismatch")
            self.reused += 1
            self.seen_session_ids.add(record.get("session_id", "unknown"))
            self._progress("REUSED", identity["tag"])
            self.save_status("running")
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Checkpoint record is corrupted: {path}. "
                "Inspect/move the individual record before retrying."
            ) from exc

    def save(self, identity: dict[str, Any], result: dict[str, Any]) -> None:
        path = self._record_path(identity)
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite a completed checkpoint: {path}")
        record = {
            "schema": CHECKPOINT_SCHEMA,
            "identity": identity,
            "result": result,
            "result_sha256": _digest(result),
            "session_id": self.session_id,
            "saved_at_utc": _utc_now(),
        }
        _write_json_atomic(path, record)
        self.created += 1
        self._progress("SAVED", identity["tag"])
        self.save_status("running")

    def mark_warmup(self, tag: str) -> None:
        self.warmups += 1
        print(f"[warmup] {tag} (runs again after resume)", flush=True)

    def _progress(self, status: str, tag: str) -> None:
        total = self.created + self.reused
        if total <= 1 or total % self.progress_every == 0:
            print(f"[checkpoint] {status:<6} new={self.created} reused={self.reused} "
                  f"total={total} last={tag}", flush=True)

    def summary(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.fingerprint,
            "checkpoint_directory": str(self.directory),
            "session_id": self.session_id,
            "session_started_at_utc": self._started_at,
            "new_simulations": self.created,
            "reused_simulations": self.reused,
            "warmups_reexecuted": self.warmups,
            "measurement_session_ids_observed": sorted(self.seen_session_ids),
            "mixed_measurement_sessions": len(self.seen_session_ids) > 1,
        }

    def save_status(self, state: str) -> None:
        _write_json_atomic(self.directory / "campaign_status.json", {
            "state": state,
            "updated_at_utc": _utc_now(),
            **self.summary(),
        })
