import time
import numpy as np

from benchmarks.baselines.common_statistics import calculate_statistics


def run(seed=101, iterations=100000):

    start = time.perf_counter()

    rng = np.random.default_rng(seed)

    values = np.array(
        [10, 20, 30]
    )

    probabilities = np.array(
        [0.5, 0.3, 0.2]
    )

    x = rng.choice(
        values,
        size=iterations,
        p=probabilities
    )

    y = x

    stats = calculate_statistics(y)

    elapsed = time.perf_counter() - start

    stats.update(
        {
            "model": "discrete_mean",
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