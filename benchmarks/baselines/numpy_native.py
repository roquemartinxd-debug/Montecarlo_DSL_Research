from __future__ import annotations

import math
import numpy as np


def calculate_statistics(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    total = values.size

    valid_mask = np.isfinite(values)
    valid_values = values[valid_mask]

    valid_count = valid_values.size
    discarded = total - valid_count

    if valid_count == 0:
        return {
            "mean": None,
            "variance": None,
            "std": None,
            "count": 0,
            "valid_rate": 0.0,
            "discard_rate": (
                discarded / total
                if total
                else 0.0
            ),
            "sample_min": None,
            "sample_max": None,
            "mcse": None,
            "relative_mcse": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }

    mean = float(
        np.mean(valid_values)
    )

    if valid_count > 1:

        variance = float(
            np.var(
                valid_values,
                ddof=1,
            )
        )

        std = float(
            np.std(
                valid_values,
                ddof=1,
            )
        )

        mcse = (
            std
            / math.sqrt(valid_count)
        )

    else:
        variance = None
        std = None
        mcse = None

    percentiles = np.percentile(
        valid_values,
        [5, 50, 95],
    )

    relative_mcse = None

    if (
        mcse is not None
        and mean != 0.0
    ):
        relative_mcse = abs(
            mcse / mean
        )

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "count": int(valid_count),
        "valid_rate":
            valid_count / total,
        "discard_rate":
            discarded / total,
        "sample_min":
            float(np.min(valid_values)),
        "sample_max":
            float(np.max(valid_values)),
        "mcse": mcse,
        "relative_mcse":
            relative_mcse,
        "p05":
            float(percentiles[0]),
        "p50":
            float(percentiles[1]),
        "p95":
            float(percentiles[2]),
    }


def run_normal_mean(
    seed=101,
    iterations=100000,
):

    rng = np.random.default_rng(seed)

    x = rng.normal(
        loc=10,
        scale=2,
        size=iterations,
    )

    return calculate_statistics(x)


def run_uniform_mean(
    seed=101,
    iterations=100000,
):

    rng = np.random.default_rng(seed)

    x = rng.uniform(
        low=0,
        high=10,
        size=iterations,
    )

    return calculate_statistics(x)


def run_discrete_mean(
    seed=101,
    iterations=100000,
):

    rng = np.random.default_rng(seed)

    x = rng.choice(
        [10, 20, 30],
        size=iterations,
        p=[0.5, 0.3, 0.2],
    )

    return calculate_statistics(x)


def run_linear_combination(
    seed=101,
    iterations=100000,
):

    rng = np.random.default_rng(seed)

    x = rng.normal(
        loc=10,
        scale=2,
        size=iterations,
    )

    z = rng.uniform(
        low=0,
        high=6,
        size=iterations,
    )

    y = 2 * x + 3 * z

    return calculate_statistics(y)