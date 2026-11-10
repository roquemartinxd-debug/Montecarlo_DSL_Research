from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from montecarlo_dsl import GeneratorConfig, generate_source, parse, tokenize
from montecarlo_dsl.ast import Statistic
from montecarlo_dsl.statistics import RunningStats, merge_many, wilson_interval


def _run_generated(tmp_path: Path, source: str, *, seed: int = 123, batch_size: int = 50):
    script = generate_source(
        source,
        GeneratorConfig(seed=seed, batch_size=batch_size, max_workers=2, histogram_bins=20),
    )
    script_path = tmp_path / "generated_v12.py"
    script_path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    prefix = "MC_DSL_EVENT:"
    events = [
        json.loads(line[len(prefix):])
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    return completed, events


def test_running_stats_matches_numpy_and_merges_batches():
    values = np.array([1.0, 2.0, 3.0, 10.0, -5.0, 7.5], dtype=np.float64)
    whole = RunningStats.from_values(values)
    left = RunningStats.from_values(values[:3])
    right = RunningStats.from_values(values[3:])
    merged = merge_many([left, right])

    assert whole.n == len(values)
    assert whole.mean == pytest.approx(float(np.mean(values)))
    assert whole.sample_variance == pytest.approx(float(np.var(values, ddof=1)))
    assert whole.sample_stddev == pytest.approx(float(np.std(values, ddof=1)))
    assert merged.mean == pytest.approx(whole.mean)
    assert merged.sample_variance == pytest.approx(whole.sample_variance)


def test_wilson_interval_is_bounded_and_contains_observed_rate():
    interval = wilson_interval(80, 100)
    assert interval is not None
    assert 0.0 <= interval[0] <= interval[1] <= 1.0
    assert interval[0] < 0.80 < interval[1]


def test_parser_accepts_extended_statistical_response_tokens():
    source = r"""\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg,var,std,count,valid_rate,discard_rate,p05,p50,p95}"""
    program = parse(source)
    assert program.response.statistics == (
        Statistic.AVG,
        Statistic.VAR,
        Statistic.STD,
        Statistic.COUNT,
        Statistic.VALID_RATE,
        Statistic.DISCARD_RATE,
        Statistic.P05,
        Statistic.P50,
        Statistic.P95,
    )
    token_types = [token.type.name for token in tokenize(r"\response{p05,p50,p95}")]
    assert token_types[:5] == ["RESPONSE", "LBRACE", "P05", "COMMA", "P50"]


def test_generated_runtime_reports_extended_statistics_and_reproducibility_metadata(tmp_path):
    source = r"""\model{x=a}
\normal a{10,2}
\iter{200}
\response{avg,var,std,count,valid_rate,discard_rate,p05,p50,p95}"""
    completed, events = _run_generated(tmp_path, source, seed=99, batch_size=40)
    assert completed.returncode == 0, completed.stderr
    start = events[0]
    final = events[-1]

    assert start["statistical_core_version"] == "1.3.1"
    assert start["metadata"]["nonfinite_policy"] == "discard_and_report"
    assert start["metadata"]["rng_bit_generator"]
    assert final["metadata"]["seed_strategy"].startswith("numpy.SeedSequence")
    assert set(final["stats"]) == {
        "avg",
        "var",
        "std",
        "count",
        "valid_rate",
        "discard_rate",
        "p05",
        "p50",
        "p95",
    }
    assert final["stats"]["count"] == 200
    assert final["stats"]["valid_rate"] == pytest.approx(1.0)
    assert final["stats"]["discard_rate"] == pytest.approx(0.0)
    assert final["diagnostics"]["sample_variance"] is not None
    assert final["diagnostics"]["valid_rate_ci95_wilson"] is not None


def test_plus_minus_runtime_reports_paired_delta_statistics(tmp_path):
    source = r"""\model{x=a\pm1}
\normal a{0,1}
\iter{120}
\response{avg,std,count}"""
    completed, events = _run_generated(tmp_path, source, seed=13, batch_size=30)
    assert completed.returncode == 0, completed.stderr
    final = events[-1]
    paired = final["paired"]
    assert paired is not None
    assert paired["candidate_results"] == 120
    assert paired["valid_results"] == 120
    assert paired["stats"]["count"] == 120
    assert paired["stats"]["avg"] == pytest.approx(2.0)
    assert paired["diagnostics"]["mcse_mean"] == pytest.approx(0.0)
    assert paired["histogram"]["total_count"] == paired["valid_results"]


def test_bounded_streaming_histogram_rejects_unbounded_out_of_order_buffer():
    from montecarlo_dsl.statistics import BlockHistogramSummary, BoundedStreamingHistogram

    sketch = BoundedStreamingHistogram(max_pending_blocks=2)
    summary = BlockHistogramSummary.from_array(np.array([1.0, 2.0]))
    sketch.add_summary(2, summary)
    sketch.add_summary(3, summary)
    with pytest.raises(RuntimeError, match="max_pending_blocks"):
        sketch.add_summary(4, summary)


def test_bounded_streaming_histogram_drains_pending_blocks_when_frontier_arrives():
    from montecarlo_dsl.statistics import BlockHistogramSummary, BoundedStreamingHistogram

    sketch = BoundedStreamingHistogram(max_pending_blocks=2)
    sketch.add_summary(1, BlockHistogramSummary.from_array(np.array([2.0])))
    sketch.add_summary(0, BlockHistogramSummary.from_array(np.array([1.0])))
    assert sketch.total_count == 2
    assert len(sketch._pending_blocks) == 0
