"""Regression tests for the version-specific RNG smoke-test controls."""

from __future__ import annotations

from argparse import Namespace
import csv
from pathlib import Path

from benchmarks import rng_reproducibility as rng


def test_rng_controls_distinguish_physical_invariance_and_legacy_sensitivity(
    monkeypatch, tmp_path: Path
):
    calls: list[dict] = []

    def fake_run_dsl(**kwargs):
        calls.append(kwargs)
        if kwargs.get("seed_strategy", "philox_logical_block") == "batch_index":
            # Legacy stream is intentionally sensitive to physical partition size.
            sample = kwargs["seed"] * 1000 + kwargs["batch_size"]
        else:
            sample = kwargs["seed"] * 1000  # Physical batches do not affect samples.
        return {"stats": {"avg": float(sample), "var": 1.0}}, 0.001

    monkeypatch.setattr(rng, "run_dsl", fake_run_dsl)
    args = Namespace(
        control_model="normal_mean", seed=101, iterations=100_000,
        batch_size=25_000, histogram_bins=30, timeout=30.0,
    )

    rows = rng.run_controls(args, tmp_path)
    by_name = {}
    for row in rows:
        by_name.setdefault(row["control"], []).append(row)

    assert set(by_name) == {
        "seed_sensitivity",
        "physical_batch_size_invariance",
        "legacy_batch_size_sensitivity",
    }
    assert len(by_name["seed_sensitivity"]) == 2
    assert len(by_name["physical_batch_size_invariance"]) == 3
    assert len(by_name["legacy_batch_size_sensitivity"]) == 3
    assert all(row["pass"] for row in rows)
    assert all(
        row["seed_strategy"] == "philox_logical_block"
        for row in by_name["physical_batch_size_invariance"]
    )
    assert all(
        row["seed_strategy"] == "batch_index"
        for row in by_name["legacy_batch_size_sensitivity"]
    )
    assert all(
        call.get("seed_strategy") == "batch_index"
        and call["scheduler_mode"] == "fixed"
        for call in calls[-3:]
    )

    with (tmp_path / "rng_reproducibility_controls.csv").open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    assert len(data) == 8
    assert data[-1]["control"] == "legacy_batch_size_sensitivity"


def test_rng_control_expectations_fail_when_batch_relations_are_reversed(monkeypatch, tmp_path):
    def fake_run_dsl(**kwargs):
        if kwargs.get("seed_strategy", "philox_logical_block") == "batch_index":
            sample = kwargs["seed"]  # Unwanted legacy invariance.
        else:
            sample = kwargs["seed"] + kwargs["batch_size"]  # Unwanted physical sensitivity.
        return {"stats": {"avg": float(sample), "var": 1.0}}, 0.001

    monkeypatch.setattr(rng, "run_dsl", fake_run_dsl)
    args = Namespace(
        control_model="normal_mean", seed=101, iterations=100_000,
        batch_size=25_000, histogram_bins=30, timeout=30.0,
    )
    rows = rng.run_controls(args, tmp_path)
    assert all(not row["pass"] for row in rows if row["control"] != "seed_sensitivity")
    assert all(row["pass"] for row in rows if row["control"] == "seed_sensitivity")
