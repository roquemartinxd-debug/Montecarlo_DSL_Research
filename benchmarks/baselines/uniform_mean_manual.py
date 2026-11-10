import time
import numpy as np

from benchmarks.baselines.common_statistics import calculate_statistics


def run(seed=101, iterations=100000):

    start = time.perf_counter()

    rng = np.random.default_rng(seed)

    x = rng.uniform(
        low=0,
        high=10,
        size=iterations
    )

    y = x

    stats = calculate_statistics(y)

    elapsed = time.perf_counter() - start

    stats.update(
        {
            "model": "uniform_mean",
            "implementation": "python_manual",
            "seed": seed,
            "iterations": iterations,
            "execution_time_seconds": elapsed,
        }
    )

    return stats


if __name__ == "__main__":

    result = run()

    for key, value in result.items():
        print(f"{key}: {value}")