from montecarlo_dsl.statistics import RunningStats
import numpy as np


def calculate_statistics(values):
    values = np.asarray(values, dtype=float)

    total = len(values)

    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]

    discarded = total - len(valid_values)

    stats = RunningStats.from_values(valid_values)

    if len(valid_values) > 0:
        percentiles = np.percentile(
            valid_values,
            [5, 50, 95]
        )

        p05 = float(percentiles[0])
        p50 = float(percentiles[1])
        p95 = float(percentiles[2])

    else:
        p05 = None
        p50 = None
        p95 = None

    return {
        "mean": stats.mean if stats.n > 0 else None,
        "variance": stats.sample_variance,
        "std": stats.sample_stddev,
        "count": stats.n,
        "valid_rate": stats.n / total if total else 0,
        "discard_rate": discarded / total if total else 0,
        "sample_min": stats.sample_min,
        "sample_max": stats.sample_max,
        "mcse": stats.mcse_mean,
        "relative_mcse": stats.relative_mcse(),
        "p05": p05,
        "p50": p50,
        "p95": p95,
    }