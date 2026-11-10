from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


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

    @classmethod
    def from_array(cls, values: np.ndarray) -> "RunningStats":
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        if n == 0:
            return cls()
        mean = float(np.mean(arr, dtype=np.float64))
        centered = arr - mean
        m2 = float(np.sum(centered * centered, dtype=np.float64))
        return cls(
            n=n,
            mean=mean,
            m2=m2,
            sample_min=float(np.min(arr)),
            sample_max=float(np.max(arr)),
        )

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


def weighted_quantiles(
    centers: np.ndarray,
    weights: np.ndarray,
    probs: Sequence[float] = (0.05, 0.50, 0.95),
) -> list[float] | None:
    """Calcula cuantiles ponderados incorporando explicitamente los pesos enteros de cada punto/bin.

    Fusiona primero centros identicos (masas puntuales / distribuciones discretas). Si el boceto
    corresponde a un soporte discreto pequeno (<= 32 valores unicos), evalua la funcion cuantil
    escalonada exacta; en caso continuo o mixto, interpola sobre la CDF empirica centrada en el peso.
    """
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if c.size == 0 or w.size == 0 or c.size != w.size:
        return None
    positive = w > 0
    if not np.any(positive):
        return None
    c = c[positive]
    w = w[positive]

    order = np.argsort(c, kind="mergesort")
    c_sorted = c[order]
    w_sorted = w[order]

    uniq_mask = np.empty(c_sorted.size, dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = c_sorted[1:] != c_sorted[:-1]
    if not np.all(uniq_mask):
        idx = np.flatnonzero(uniq_mask)
        c_sorted = c_sorted[idx]
        w_sorted = np.add.reduceat(w_sorted, idx)

    if c_sorted.size == 1:
        return [float(c_sorted[0]) for _ in probs]

    cum_w = np.cumsum(w_sorted, dtype=np.float64)
    total_w = float(cum_w[-1])
    if total_w <= 0:
        return None

    if c_sorted.size <= 32 and float(np.max(w_sorted)) / total_w >= 0.05:
        p_arr = np.asarray(probs, dtype=np.float64)
        step_idx = np.searchsorted(cum_w / total_w, p_arr, side="left")
        step_idx = np.clip(step_idx, 0, c_sorted.size - 1)
        return [float(c_sorted[int(i)]) for i in step_idx]

    cdf = (cum_w - 0.5 * w_sorted) / total_w
    interpolated = np.interp(np.asarray(probs, dtype=np.float64), cdf, c_sorted)
    return [float(val) for val in interpolated]


def compact_weighted_sketch(
    centers: np.ndarray,
    weights: np.ndarray,
    data_min: float,
    data_max: float,
    *,
    target_bins: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Comprime un conjunto de puntos ponderados a a lo sumo ``target_bins`` centroides equiprobables.

    Conserva de forma exacta:
    - La suma entera de pesos: ``sum(new_weights) == sum(weights)``.
    - Los soportes discretos cuando el numero de centros unicos es ``<= target_bins``.
    - Los extremos observados ``data_min`` y ``data_max``.
    """
    c = np.asarray(centers, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.int64).reshape(-1)
    if c.size == 0 or data_min == data_max:
        return c, w

    order = np.argsort(c, kind="mergesort")
    c_sorted = c[order]
    w_sorted = w[order]
    uniq_mask = np.empty(c_sorted.size, dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = c_sorted[1:] != c_sorted[:-1]
    if not np.all(uniq_mask):
        idx = np.flatnonzero(uniq_mask)
        c_sorted = c_sorted[idx]
        w_sorted = np.add.reduceat(w_sorted, idx)

    if c_sorted.size <= target_bins:
        return c_sorted, w_sorted

    k_int = max(1, int(target_bins) - 2)
    c_int = c_sorted[1:-1]
    w_int = w_sorted[1:-1]
    cw_int = np.cumsum(w_int, dtype=np.float64)
    total_int_w = max(1.0, float(cw_int[-1]))
    bucket_ids = np.minimum(k_int - 1, ((cw_int - 0.5 * w_int) * k_int / total_int_w).astype(np.int64))
    change = np.empty(bucket_ids.size, dtype=bool)
    change[0] = True
    change[1:] = bucket_ids[1:] != bucket_ids[:-1]
    starts = np.flatnonzero(change)
    new_w_int = np.add.reduceat(w_int, starts)
    new_s_int = np.add.reduceat(c_int * w_int.astype(np.float64), starts)
    new_c_int = new_s_int / new_w_int

    out_c = np.empty(new_c_int.size + 2, dtype=np.float64)
    out_w = np.empty(new_w_int.size + 2, dtype=np.int64)
    out_c[0] = float(data_min)
    out_w[0] = int(w_sorted[0])
    out_c[1:-1] = new_c_int
    out_w[1:-1] = new_w_int
    out_c[-1] = float(data_max)
    out_w[-1] = int(w_sorted[-1])
    return out_c, out_w


@dataclass(frozen=True, slots=True)
class BlockHistogramSummary:
    """Resumen compacto ponderado por centroides equiprobables para un bloque logico o tarea fisica."""

    count: int
    data_min: float | None
    data_max: float | None
    centers: np.ndarray
    weights: np.ndarray

    @property
    def quantile_sample(self) -> np.ndarray:
        return self.centers

    @property
    def nbytes(self) -> int:
        return int(self.centers.nbytes + self.weights.nbytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "min": self.data_min,
            "max": self.data_max,
            "centers": self.centers,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockHistogramSummary":
        return cls(
            count=int(payload["count"]),
            data_min=payload.get("min", payload.get("data_min")),
            data_max=payload.get("max", payload.get("data_max")),
            centers=np.asarray(payload["centers"], dtype=np.float64).reshape(-1),
            weights=np.asarray(payload["weights"], dtype=np.int64).reshape(-1),
        )

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        *,
        fine_bins: int = 256,
        max_quantile_samples: int = 256,
        precomputed_min: float | None = None,
        precomputed_max: float | None = None,
    ) -> "BlockHistogramSummary":
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if precomputed_min is None or precomputed_max is None:
            arr = arr[np.isfinite(arr)]
        n = int(arr.size)
        if n == 0:
            empty = np.empty(0, dtype=np.float64)
            return cls(
                count=0,
                data_min=None,
                data_max=None,
                centers=empty,
                weights=np.empty(0, dtype=np.int64),
            )

        vmin = float(np.min(arr)) if precomputed_min is None else float(precomputed_min)
        vmax = float(np.max(arr)) if precomputed_max is None else float(precomputed_max)

        if n <= fine_bins or vmin == vmax:
            centers = arr.copy()
            weights = np.ones(n, dtype=np.int64)
        else:
            s = np.sort(arr, kind="mergesort")
            uniq_mask = np.empty(n, dtype=bool)
            uniq_mask[0] = True
            uniq_mask[1:] = s[1:] != s[:-1]
            if int(np.sum(uniq_mask)) <= fine_bins:
                u_idx = np.flatnonzero(uniq_mask)
                centers = s[u_idx].astype(np.float64, copy=False)
                weights = np.diff(np.append(u_idx, n)).astype(np.int64, copy=False)
            else:
                interior = s[1:-1]
                k_int = max(1, int(fine_bins) - 2)
                idx = np.linspace(0, interior.size, k_int + 1, dtype=np.int64)
                counts = np.diff(idx)
                nonzero = counts > 0
                starts = idx[:-1][nonzero]
                sums = np.add.reduceat(interior, starts)
                c_int = sums / counts[nonzero]
                w_int = counts[nonzero]
                centers = np.empty(c_int.size + 2, dtype=np.float64)
                weights = np.empty(w_int.size + 2, dtype=np.int64)
                centers[0] = vmin
                weights[0] = 1
                centers[1:-1] = c_int
                weights[1:-1] = w_int
                centers[-1] = vmax
                weights[-1] = 1

        return cls(
            count=n,
            data_min=vmin,
            data_max=vmax,
            centers=centers,
            weights=weights,
        )


class BoundedStreamingHistogram:
    """Acumulador unificado de histogramas y cuantiles para el runtime y el nucleo estadistico.

    Soporta dos modos de operacion bajo una misma interfaz:
    - ``exact_retention=False`` (por defecto): fusiona los bloques en orden contiguo
      ``next_block_index = 0, 1, 2, ...`` dentro de un boceto ponderado de capacidad fija
      ``max_global_points`` y descarta los bloques ya fusionados, logrando memoria O(K_max) = O(1).
    - ``exact_retention=True``: conserva los arreglos de muestras por indice de bloque para
      evaluar cuantiles e histogramas exactos cuando ``memory_mode == 'retain_all'`` o cuando
      el numero total de iteraciones es pequeno (``<= 65,536``).
    """

    def __init__(
        self,
        bins: int = 30,
        *,
        fine_bins: int = 256,
        max_quantile_samples_per_block: int = 256,
        max_global_points: int = 2048,
        max_pending_blocks: int = 64,
        exact_retention: bool = False,
    ) -> None:
        if bins <= 0:
            raise ValueError("bins must be positive")
        if max_pending_blocks <= 0:
            raise ValueError("max_pending_blocks must be positive")
        self.bins = bins
        self.fine_bins = fine_bins
        self.max_quantile_samples_per_block = max_quantile_samples_per_block
        self.max_global_points = max_global_points
        self.max_pending_blocks = int(max_pending_blocks)
        self.exact_retention = exact_retention

        self._pending_blocks: dict[int, BlockHistogramSummary] = {}
        self._next_block_to_merge: int = 0
        self._merged_count: int = 0
        self._merged_min: float | None = None
        self._merged_max: float | None = None
        self._merged_centers: np.ndarray = np.empty(0, dtype=np.float64)
        self._merged_weights: np.ndarray = np.empty(0, dtype=np.int64)

        self._exact_chunks: dict[int, np.ndarray] = {}
        self._exact_count: int = 0
        self._exact_min: float | None = None
        self._exact_max: float | None = None

    @property
    def total_count(self) -> int:
        if self.exact_retention:
            return self._exact_count
        self._advance_contiguous_frontier()
        pending_count = sum(item.count for item in self._pending_blocks.values())
        return self._merged_count + pending_count

    def add_block(
        self,
        block_index: int,
        values: np.ndarray,
        *,
        precomputed_min: float | None = None,
        precomputed_max: float | None = None,
    ) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if precomputed_min is None or precomputed_max is None:
            arr = arr[np.isfinite(arr)]
        if self.exact_retention:
            idx = int(block_index)
            if arr.size == 0:
                return
            self._exact_chunks[idx] = arr
            self._exact_count += int(arr.size)
            vmin = float(np.min(arr)) if precomputed_min is None else float(precomputed_min)
            vmax = float(np.max(arr)) if precomputed_max is None else float(precomputed_max)
            self._exact_min = vmin if self._exact_min is None else min(self._exact_min, vmin)
            self._exact_max = vmax if self._exact_max is None else max(self._exact_max, vmax)
            return

        summary = BlockHistogramSummary.from_array(
            arr,
            fine_bins=self.fine_bins,
            max_quantile_samples=self.max_quantile_samples_per_block,
            precomputed_min=precomputed_min,
            precomputed_max=precomputed_max,
        )
        self.add_summary(block_index, summary)

    def add_summary(self, block_index: int, summary: BlockHistogramSummary | dict[str, Any]) -> None:
        if isinstance(summary, dict):
            summary = BlockHistogramSummary.from_dict(summary)
        if self.exact_retention:
            if summary.count > 0 and summary.weights.size == summary.centers.size:
                if np.all(summary.weights == 1):
                    self.add_block(
                        block_index,
                        summary.centers,
                        precomputed_min=summary.data_min,
                        precomputed_max=summary.data_max,
                    )
                    return
            self.exact_retention = False

        idx = int(block_index)
        if idx < self._next_block_to_merge:
            return
        if (
            idx != self._next_block_to_merge
            and idx not in self._pending_blocks
            and len(self._pending_blocks) >= self.max_pending_blocks
        ):
            raise RuntimeError(
                "BoundedStreamingHistogram exceeded max_pending_blocks; "
                "consume logical blocks in canonical order or increase the explicit bound."
            )
        self._pending_blocks[idx] = summary
        self._advance_contiguous_frontier()

    def _advance_contiguous_frontier(self) -> None:
        while self._next_block_to_merge in self._pending_blocks:
            summary = self._pending_blocks.pop(self._next_block_to_merge)
            self._next_block_to_merge += 1
            if summary.count <= 0 or summary.data_min is None or summary.data_max is None:
                continue
            self._merged_count += summary.count
            self._merged_min = (
                summary.data_min
                if self._merged_min is None
                else min(self._merged_min, summary.data_min)
            )
            self._merged_max = (
                summary.data_max
                if self._merged_max is None
                else max(self._merged_max, summary.data_max)
            )
            if self._merged_centers.size == 0:
                self._merged_centers = summary.centers.copy()
                self._merged_weights = summary.weights.copy()
            else:
                self._merged_centers = np.concatenate([self._merged_centers, summary.centers])
                self._merged_weights = np.concatenate([self._merged_weights, summary.weights])

            if self._merged_centers.size > self.max_global_points * 2:
                self._merged_centers, self._merged_weights = compact_weighted_sketch(
                    self._merged_centers,
                    self._merged_weights,
                    self._merged_min,
                    self._merged_max,
                    target_bins=self.max_global_points,
                )

    def _all_active_centers_and_weights(self) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
        self._advance_contiguous_frontier()
        centers_list = []
        weights_list = []
        dmin = self._merged_min
        dmax = self._merged_max

        if self._merged_centers.size > 0:
            centers_list.append(self._merged_centers)
            weights_list.append(self._merged_weights)

        for idx in sorted(self._pending_blocks):
            item = self._pending_blocks[idx]
            if item.count > 0 and item.data_min is not None and item.data_max is not None:
                centers_list.append(item.centers)
                weights_list.append(item.weights)
                dmin = item.data_min if dmin is None else min(dmin, item.data_min)
                dmax = item.data_max if dmax is None else max(dmax, item.data_max)

        if not centers_list or dmin is None or dmax is None:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return empty_f, empty_i, None, None

        return (
            np.concatenate(centers_list),
            np.concatenate(weights_list),
            dmin,
            dmax,
        )

    @staticmethod
    def _histogram_edges_for_range(data_min: float, data_max: float, bins: int) -> np.ndarray:
        if data_min == data_max:
            padding = max(abs(data_min) * 0.01, 0.5)
            low = data_min - padding
            high = data_max + padding
        else:
            low = data_min
            high = data_max
        return np.linspace(low, high, bins + 1, dtype=np.float64)

    def build_histogram(
        self,
        *,
        override_min: float | None = None,
        override_max: float | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float | None]:
        if self.exact_retention:
            if self._exact_count <= 0 or not self._exact_chunks:
                return None, None, None, None
            data_min = self._exact_min if override_min is None else float(override_min)
            data_max = self._exact_max if override_max is None else float(override_max)
            if data_min is None or data_max is None:
                return None, None, None, None
            edges = self._histogram_edges_for_range(data_min, data_max, self.bins)
            counts = np.zeros(self.bins, dtype=np.int64)
            for idx in sorted(self._exact_chunks):
                chunk = self._exact_chunks[idx]
                if chunk.size > 0:
                    c, _ = np.histogram(chunk, bins=edges)
                    counts += c.astype(np.int64, copy=False)
            return counts, edges, data_min, data_max

        centers, weights, dmin, dmax = self._all_active_centers_and_weights()
        data_min = dmin if override_min is None else float(override_min)
        data_max = dmax if override_max is None else float(override_max)
        if centers.size == 0 or data_min is None or data_max is None:
            return None, None, None, None

        edges = self._histogram_edges_for_range(data_min, data_max, self.bins)
        counts, _ = np.histogram(centers, bins=edges, weights=weights)
        return counts.astype(np.int64, copy=False), edges, data_min, data_max

    def quantiles(self, probs: Sequence[float] = (0.05, 0.50, 0.95)) -> dict[str, float | None]:
        if self.exact_retention:
            if self._exact_count <= 0 or not self._exact_chunks:
                return {"p05": None, "p50": None, "p95": None}
            ordered = [self._exact_chunks[k] for k in sorted(self._exact_chunks) if self._exact_chunks[k].size > 0]
            if not ordered:
                return {"p05": None, "p50": None, "p95": None}
            values = np.concatenate(ordered)
            if values.size == 0:
                return {"p05": None, "p50": None, "p95": None}
            q_vals = np.quantile(values, probs)
            return {
                "p05": float(q_vals[0]),
                "p50": float(q_vals[1]),
                "p95": float(q_vals[2]),
            }

        centers, weights, _, _ = self._all_active_centers_and_weights()
        q = weighted_quantiles(centers, weights, probs)
        if q is None:
            return {"p05": None, "p50": None, "p95": None}
        return {"p05": q[0], "p50": q[1], "p95": q[2]}

    def retained_bytes(self) -> int:
        total = int(self._merged_centers.nbytes + self._merged_weights.nbytes)
        for item in self._pending_blocks.values():
            total += item.nbytes
        for arr in self._exact_chunks.values():
            total += int(arr.nbytes)
        return total

    def clear(self) -> None:
        self._pending_blocks.clear()
        self._merged_centers = np.empty(0, dtype=np.float64)
        self._merged_weights = np.empty(0, dtype=np.int64)
        self._merged_count = 0
        self._merged_min = None
        self._merged_max = None
        self._exact_chunks.clear()
        self._exact_count = 0
        self._exact_min = None
        self._exact_max = None


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
