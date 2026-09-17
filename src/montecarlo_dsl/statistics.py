from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RunningStats:
    """Acumulador numericamente estable para muestras finitas.

    Usa la actualizacion incremental de Welford y la regla de combinacion
    de Chan-Golub-LeVeque para unir estados producidos por lotes o workers.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    sample_min: float | None = None
    sample_max: float | None = None

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "RunningStats":
        state = cls()
        for value in values:
            state = state.update(float(value))
        return state

    def update(self, value: float) -> "RunningStats":
        if not math.isfinite(value):
            return self
        n = self.n + 1
        delta = value - self.mean
        mean = self.mean + delta / n
        delta2 = value - mean
        m2 = self.m2 + delta * delta2
        sample_min = value if self.sample_min is None else min(self.sample_min, value)
        sample_max = value if self.sample_max is None else max(self.sample_max, value)
        return RunningStats(n=n, mean=mean, m2=m2, sample_min=sample_min, sample_max=sample_max)

    def merge(self, other: "RunningStats") -> "RunningStats":
        if other.n == 0:
            return self
        if self.n == 0:
            return other
        n = self.n + other.n
        delta = other.mean - self.mean
        mean = self.mean + delta * other.n / n
        m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / n
        sample_min = min(self.sample_min, other.sample_min)  # type: ignore[arg-type]
        sample_max = max(self.sample_max, other.sample_max)  # type: ignore[arg-type]
        return RunningStats(n=n, mean=mean, m2=m2, sample_min=sample_min, sample_max=sample_max)

    @property
    def sample_variance(self) -> float | None:
        if self.n <= 1:
            return None
        return max(0.0, self.m2 / (self.n - 1))

    @property
    def sample_stddev(self) -> float | None:
        variance = self.sample_variance
        return None if variance is None else math.sqrt(variance)

    @property
    def mcse_mean(self) -> float | None:
        stddev = self.sample_stddev
        if stddev is None or self.n <= 0:
            return None
        return stddev / math.sqrt(self.n)

    def asymptotic_ci_mean(self, z: float = 1.96) -> list[float] | None:
        mcse = self.mcse_mean
        if mcse is None:
            return None
        return [self.mean - z * mcse, self.mean + z * mcse]

    def relative_mcse(self) -> float | None:
        mcse = self.mcse_mean
        if mcse is None or self.mean == 0:
            return None
        return abs(mcse / self.mean)


def merge_many(states: Iterable[RunningStats]) -> RunningStats:
    result = RunningStats()
    for state in states:
        result = result.merge(state)
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> list[float] | None:
    """Intervalo de Wilson para una proporcion binomial.

    Devuelve None cuando no hay ensayos. No usa SciPy para mantener el paquete
    ligero y reproducible en el entorno local del proyecto.
    """

    if trials <= 0:
        return None
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denom
    return [max(0.0, center - half), min(1.0, center + half)]
