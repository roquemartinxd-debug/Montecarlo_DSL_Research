from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from montecarlo_dsl import (
    BoundedStreamingHistogram,
    GeneratorConfig,
    compact_weighted_sketch,
    generate_source,
    weighted_quantiles,
)


PREFIX = "MC_DSL_EVENT:"


def _canonical_hash(complete_event: dict) -> str:
    payload = {
        "processed_iterations": complete_event.get("processed_iterations"),
        "total_iterations": complete_event.get("total_iterations"),
        "candidate_results": complete_event.get("candidate_results"),
        "valid_results": complete_event.get("valid_results"),
        "discarded_results": complete_event.get("discarded_results"),
        "branches": complete_event.get("branches"),
        "paired": complete_event.get("paired"),
        "stats": complete_event.get("stats"),
        "histogram": complete_event.get("histogram"),
        "diagnostics": complete_event.get("diagnostics"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_sim(tmp_path: Path, name: str, source: str, config: GeneratorConfig) -> dict:
    script = generate_source(source, config)
    script_path = tmp_path / f"{name}.py"
    script_path.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    events = [
        json.loads(line[len(PREFIX):])
        for line in proc.stdout.splitlines()
        if line.startswith(PREFIX)
    ]
    assert events[-1]["type"] == "complete"
    return events[-1]


def test_reproducibility_is_invariant_to_workers_physical_batch_size_and_completion_order(
    tmp_path: Path,
):
    source = r"""\model{y = 2x + 3z - \frac{w}{2}}
\normal x{10, 2}
\uniform z{-5, 15}
\distrib w{{2, 0.5}, {4, 0.3}, {8, 0.2}}
\iter{600}
\response{avg,var,std,min,max,count,valid_rate,discard_rate,p05,p50,p95}"""

    configs = [
        ("fixed_b100_w1", GeneratorConfig(seed=555, batch_size=100, logical_block_size=100, max_workers=1, scheduler_mode="fixed")),
        ("fixed_b200_w2", GeneratorConfig(seed=555, batch_size=200, logical_block_size=100, max_workers=2, scheduler_mode="fixed")),
        ("unaligned_b137_w1", GeneratorConfig(seed=555, batch_size=137, logical_block_size=100, max_workers=1, scheduler_mode="fixed")),
        ("ast_only_w2", GeneratorConfig(seed=555, batch_size=200, logical_block_size=100, max_workers=2, scheduler_mode="ast_only")),
        ("hybrid_w2", GeneratorConfig(seed=555, batch_size=200, logical_block_size=100, max_workers=2, scheduler_mode="hybrid")),
        ("reverse_order", GeneratorConfig(seed=555, batch_size=200, logical_block_size=100, max_workers=2, scheduler_mode="hybrid", task_completion_order="reverse")),
        ("shuffled_order", GeneratorConfig(seed=555, batch_size=200, logical_block_size=100, max_workers=2, scheduler_mode="hybrid", task_completion_order="shuffled")),
    ]

    hashes = set()
    for name, cfg in configs:
        final = _run_sim(tmp_path, name, source, cfg)
        hashes.add(_canonical_hash(final))

    assert len(hashes) == 1, f"Expected 1 canonical hash across all schedules, got {len(hashes)}"


def test_plus_minus_and_nonfinite_model_reproducibility_across_schedules(tmp_path: Path):
    source = r"""\model{y = x \pm \sqrt{z}}
\normal x{10, 2}
\uniform z{-4, 12}
\iter{500}
\response{avg,var,std,count,valid_rate,discard_rate,p05,p50,p95}"""

    c1 = _run_sim(
        tmp_path,
        "pm_fixed_b100",
        source,
        GeneratorConfig(seed=777, batch_size=100, logical_block_size=100, max_workers=1, scheduler_mode="fixed"),
    )
    c2 = _run_sim(
        tmp_path,
        "pm_hybrid_b250_rev",
        source,
        GeneratorConfig(
            seed=777,
            batch_size=250,
            logical_block_size=100,
            max_workers=2,
            scheduler_mode="hybrid",
            task_completion_order="reverse",
        ),
    )
    assert _canonical_hash(c1) == _canonical_hash(c2)


def test_weighted_quantiles_incorporate_weights_and_compaction_discards_blocks():
    centers = np.array([10.0, 10.1, 50.0, 100.0], dtype=np.float64)
    weights = np.array([480, 470, 30, 20], dtype=np.int64)
    wq = weighted_quantiles(centers, weights, (0.05, 0.50, 0.95))
    assert wq is not None
    assert wq[1] == pytest.approx(10.0, abs=1.0)

    c_comp, w_comp = compact_weighted_sketch(
        np.linspace(0.0, 10.0, 5000),
        np.ones(5000, dtype=np.int64),
        0.0,
        10.0,
        target_bins=512,
    )
    assert c_comp.size <= 512
    assert int(np.sum(w_comp)) == 5000
    assert float(c_comp[0]) == pytest.approx(0.0)
    assert float(c_comp[-1]) == pytest.approx(10.0)


def test_bounded_streaming_histogram_conserves_counts_extrema_and_discards_merged_blocks():
    rng = np.random.default_rng(123)
    samples = rng.normal(loc=50.0, scale=10.0, size=20_000)

    hist = BoundedStreamingHistogram(
        bins=30,
        fine_bins=128,
        max_quantile_samples_per_block=128,
        max_global_points=512,
    )
    for b in range(20):
        hist.add_block(b, samples[b * 1000 : (b + 1) * 1000])

    assert len(hist._pending_blocks) == 0
    assert hist._merged_centers.size <= 1024

    counts, edges, dmin, dmax = hist.build_histogram()
    assert counts is not None and edges is not None
    assert int(np.sum(counts)) == 20_000
    assert dmin == pytest.approx(float(np.min(samples)))
    assert dmax == pytest.approx(float(np.max(samples)))

    q = hist.quantiles()
    exact_q05, exact_q50, exact_q95 = np.quantile(samples, [0.05, 0.50, 0.95])
    assert q["p05"] == pytest.approx(float(exact_q05), abs=0.25)
    assert q["p50"] == pytest.approx(float(exact_q50), abs=0.25)
    assert q["p95"] == pytest.approx(float(exact_q95), abs=0.25)


def test_bounded_streaming_histogram_quantiles_across_multimodal_skewed_discrete_and_heavy_tailed():
    rng = np.random.default_rng(2026)
    n = 40_000
    block_size = 2_000

    # 1. Multimodal (mezcla bimodal bien separada)
    bimodal = np.where(
        rng.random(n) < 0.45,
        rng.normal(-15.0, 1.5, size=n),
        rng.normal(25.0, 2.5, size=n),
    )
    # 2. Asimetrica positiva (log-normal / cuadratica sesgada)
    skewed = np.exp(rng.normal(1.5, 0.65, size=n))
    # 3. Discreta pura (escalonada en 5 soportes)
    discrete = rng.choice(
        np.array([2.0, 5.0, 12.0, 30.0, 80.0]),
        size=n,
        p=np.array([0.10, 0.25, 0.35, 0.20, 0.10]),
    )
    # 4. Colas pesadas (t de Student gl=4 desplazada)
    heavy_tailed = 50.0 + 5.0 * rng.standard_t(df=4, size=n)

    for dist_name, samples, tol in (
        ("bimodal", bimodal, 0.35),
        ("skewed", skewed, 0.25),
        ("discrete", discrete, 1e-6),
        ("heavy_tailed", heavy_tailed, 0.45),
    ):
        hist = BoundedStreamingHistogram(bins=30, fine_bins=256, max_global_points=2048)
        for b in range(n // block_size):
            hist.add_block(b, samples[b * block_size : (b + 1) * block_size])
        assert len(hist._pending_blocks) == 0
        q = hist.quantiles()
        ex05, ex50, ex95 = np.quantile(samples, [0.05, 0.50, 0.95])
        assert q["p05"] == pytest.approx(float(ex05), abs=tol), f"{dist_name} p05 mismatch"
        assert q["p50"] == pytest.approx(float(ex50), abs=tol), f"{dist_name} p50 mismatch"
        assert q["p95"] == pytest.approx(float(ex95), abs=tol), f"{dist_name} p95 mismatch"


def test_bounded_memory_mode_reduces_peak_memory_and_preserves_weighted_quantiles_and_paired_o1(
    tmp_path: Path,
):
    source = r"""\model{y = x \pm z^2}
\normal x{10, 2}
\uniform z{0, 5}
\iter{100000}
\response{avg,var,std,min,max,p05,p50,p95}"""

    retain_final = _run_sim(
        tmp_path,
        "mem_retain_pm",
        source,
        GeneratorConfig(
            seed=101,
            batch_size=25_000,
            logical_block_size=5_000,
            max_workers=1,
            scheduler_mode="hybrid",
            memory_mode="retain_all",
        ),
    )
    bounded_final = _run_sim(
        tmp_path,
        "mem_bounded_pm",
        source,
        GeneratorConfig(
            seed=101,
            batch_size=25_000,
            logical_block_size=5_000,
            max_workers=1,
            scheduler_mode="hybrid",
            memory_mode="bounded",
        ),
    )

    for b_ret, b_bnd in zip(retain_final["branches"], bounded_final["branches"]):
        for exact_key in ("avg", "var", "std", "min", "max"):
            assert b_bnd["stats"][exact_key] == b_ret["stats"][exact_key]
        for q_key in ("p05", "p50", "p95"):
            assert b_bnd["stats"][q_key] == pytest.approx(b_ret["stats"][q_key], abs=0.15)
        assert b_bnd["diagnostics"] == b_ret["diagnostics"]
        assert b_bnd["histogram"]["total_count"] == 100_000

    # Verificamos tambien la rama pareada (delta y correlacion O(1))
    p_ret = retain_final["paired"]
    p_bnd = bounded_final["paired"]
    assert p_ret is not None and p_bnd is not None
    assert p_bnd["correlation_plus_minus_mean"] == pytest.approx(
        p_ret["correlation_plus_minus_mean"], abs=1e-12
    )
    assert p_bnd["histogram"]["total_count"] == p_bnd["valid_results"]
    assert p_ret["histogram"]["total_count"] == p_ret["valid_results"]
    for q_key in ("p05", "p50", "p95"):
        assert p_bnd["stats"][q_key] == pytest.approx(p_ret["stats"][q_key], abs=0.15)

    meta_bnd = bounded_final["metadata"]
    meta_ret = retain_final["metadata"]
    assert meta_bnd["retained_coordinator_payload_bytes"] < meta_ret["retained_coordinator_payload_bytes"] * 0.25
    assert meta_bnd["peak_retained_coordinator_bytes"] < meta_ret["peak_retained_coordinator_bytes"] * 0.25
    assert meta_bnd["coordinator_peak_rss_bytes"] > 0
    assert meta_bnd["aggregate_peak_rss_bytes"] >= meta_bnd["coordinator_peak_rss_bytes"]
