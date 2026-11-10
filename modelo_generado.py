# Generado por montecarlo_dsl v1.3.0.
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
import platform
import random
import resource
import sys
import time
import traceback
from typing import Any, Iterable, Sequence

import numpy as np

PROTOCOL_VERSION = 2
STATISTICAL_CORE_VERSION = "1.2.0"
RUNTIME_VERSION = "1.3.0"
EVENT_PREFIX = "MC_DSL_EVENT:"
MASTER_SEED = 101
TOTAL_ITERATIONS = 1000000
BATCH_SIZE = 25000
LOGICAL_BLOCK_SIZE = 5000
EFF_LOGICAL_BLOCK_SIZE = min(LOGICAL_BLOCK_SIZE, TOTAL_ITERATIONS)
LOGICAL_BLOCK_COUNT = (TOTAL_ITERATIONS + EFF_LOGICAL_BLOCK_SIZE - 1) // EFF_LOGICAL_BLOCK_SIZE
MAX_WORKERS = 4
HISTOGRAM_BINS = 30
SCHEDULER_MODE = 'hybrid'
SEED_STRATEGY = 'philox_logical_block'
MEMORY_MODE = 'bounded'
TASK_COMPLETION_ORDER = 'as_completed'
OPTIMIZER_ENABLED = True
OPTIMIZATION_METRICS = {'enabled': True, 'ast_nodes_before': 17, 'ast_nodes_after': 17, 'nodes_reduced': 0, 'constant_folds': 0, 'algebraic_simplifications': 0, 'strength_reductions': 0, 'total_transformations': 0, 'optimization_time_seconds': 0.00012114600031054579}
AST_COST_FEATURES = {'num_random_variables': 3, 'num_normal_vars': 1, 'num_uniform_vars': 1, 'num_discrete_vars': 1, 'discrete_support_total': 3, 'num_add_sub': 3, 'num_mul': 3, 'num_div_frac': 1, 'num_power': 1, 'num_sqrt': 1, 'num_unary': 1, 'ast_node_count': 17, 'ast_depth': 7, 'branch_count': 2, 'requested_stats_count': 3, 'requires_quantiles': False, 'estimated_bytes_per_sample': 88, 'distribution_cost_index': 6.65, 'arithmetic_cost_index': 9.6, 'transcendental_cost_index': 5.6, 'weighted_operation_score': 22.85}
COST_MODEL_WEIGHTS = (1.2624926371329765e-07, 7.282681154485559e-09, 6.007353201571716e-08, 6.867690673466813e-09)
PRIOR_COST_MODEL_WEIGHTS = (6.5e-08, 2.8e-08, 2.6e-08, 2.2e-08)
STATIC_PER_SAMPLE_SECONDS = 7.898440685129256e-07
SERIAL_TASK_OVERHEAD_SECONDS = 0.00035
PARALLEL_POOL_SPAWN_SECONDS = 0.18
PARALLEL_TASK_IPC_SECONDS = 0.006
PRIOR_EQUIVALENT_SAMPLES = 1500
ESTIMATED_BYTES_PER_SAMPLE = 88
STATIC_SCHEDULE_TASKS = [(0, 125000, 0, 125000), (1, 125000, 125000, 250000), (2, 125000, 250000, 375000), (3, 125000, 375000, 500000), (4, 125000, 500000, 625000), (5, 125000, 625000, 750000), (6, 125000, 750000, 875000), (7, 60000, 875000, 935000), (8, 60000, 935000, 995000), (9, 5000, 995000, 1000000)]
STATIC_EFFECTIVE_WORKERS = 2
STATIC_USE_PROCESS_POOL = True
REQUESTED_STATISTICS = ('avg', 'min', 'max')
REQUIRES_QUANTILES = False
BRANCH_IDS = ('plus', 'minus')
BRANCH_LABELS = ('Rama +', 'Rama -')
BRANCH_COUNT = len(BRANCH_IDS)
EXACT_SAMPLE_RETENTION_LIMIT = 65_536
FINE_HISTOGRAM_BINS = 256
MAX_GLOBAL_SKETCH_POINTS = 2048

# Implementacion unificada inyectada directamente desde montecarlo_dsl.statistics
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
        exact_retention: bool = False,
    ) -> None:
        if bins <= 0:
            raise ValueError("bins must be positive")
        self.bins = bins
        self.fine_bins = fine_bins
        self.max_quantile_samples_per_block = max_quantile_samples_per_block
        self.max_global_points = max_global_points
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


def _process_peak_rss_bytes():
    """Obtiene el pico real de memoria residente (ru_maxrss) del proceso actual en bytes."""
    try:
        ru = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return ru if sys.platform == "darwin" else ru * 1024
    except Exception:
        return 0


def _emit(payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(EVENT_PREFIX + encoded, flush=True)


def _batch_seed_sequence(batch_index):
    return np.random.SeedSequence([MASTER_SEED, int(batch_index)])


def _variable_rng(logical_block_index, var_index):
    ss = np.random.SeedSequence([MASTER_SEED, int(logical_block_index), int(var_index)])
    return np.random.default_rng(np.random.Philox(ss))


def _empty_stats_state():
    return {"n": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None}


def _stats_from_values(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(values.size)
    if n == 0:
        return _empty_stats_state()
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    m2 = float(np.sum(centered * centered, dtype=np.float64))
    return {
        "n": n,
        "mean": mean,
        "m2": m2,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _merge_stats(left, right):
    if right["n"] == 0:
        return dict(left)
    if left["n"] == 0:
        return dict(right)
    n_left = int(left["n"])
    n_right = int(right["n"])
    n = n_left + n_right
    delta = right["mean"] - left["mean"]
    mean = left["mean"] + delta * n_right / n
    m2 = left["m2"] + right["m2"] + delta * delta * n_left * n_right / n
    return {
        "n": n,
        "mean": mean,
        "m2": m2,
        "min": min(left["min"], right["min"]),
        "max": max(left["max"], right["max"]),
    }


def _stats_from_state(state):
    merged = state["canonical_stats"]
    for key in sorted(state["batch_stats"]):
        merged = _merge_stats(merged, state["batch_stats"][key])
    return merged


def _sample_variance(stats_state):
    n = stats_state["n"]
    if n <= 1:
        return None
    return max(0.0, stats_state["m2"] / (n - 1))


def _sample_stddev(stats_state):
    variance = _sample_variance(stats_state)
    return None if variance is None else math.sqrt(variance)


def _mcse_mean(stats_state):
    stddev = _sample_stddev(stats_state)
    if stddev is None or stats_state["n"] <= 0:
        return None
    return stddev / math.sqrt(stats_state["n"])


def _ci95_mean(stats_state):
    mcse = _mcse_mean(stats_state)
    if mcse is None:
        return None
    mean = stats_state["mean"]
    return [mean - 1.96 * mcse, mean + 1.96 * mcse]


def _wilson_interval(successes, trials, z=1.96):
    return wilson_interval(successes, trials, z=z)


def _weighted_quantiles(centers, weights):
    q = weighted_quantiles(centers, weights, (0.05, 0.50, 0.95))
    if q is None:
        return {"p05": None, "p50": None, "p95": None}
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _compact_weighted_sketch(centers, weights, data_min, data_max, target_bins=MAX_GLOBAL_SKETCH_POINTS):
    return compact_weighted_sketch(
        centers,
        weights,
        data_min,
        data_max,
        target_bins=target_bins,
    )


def _summarize_block_values(values, stats_state):
    n = int(stats_state["n"])
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "count": 0,
            "min": None,
            "max": None,
            "centers": empty,
            "weights": np.empty(0, dtype=np.int64),
        }
    summary_obj = BlockHistogramSummary.from_array(
        values,
        fine_bins=FINE_HISTOGRAM_BINS,
        max_quantile_samples=FINE_HISTOGRAM_BINS,
        precomputed_min=stats_state["min"],
        precomputed_max=stats_state["max"],
    )
    return summary_obj.to_dict()


def _use_compact_block_summaries():
    return MEMORY_MODE == "bounded" and TOTAL_ITERATIONS > EXACT_SAMPLE_RETENTION_LIMIT


def _make_streaming_histogram():
    return BoundedStreamingHistogram(
        bins=HISTOGRAM_BINS,
        fine_bins=FINE_HISTOGRAM_BINS,
        max_quantile_samples_per_block=FINE_HISTOGRAM_BINS,
        max_global_points=MAX_GLOBAL_SKETCH_POINTS,
        exact_retention=not _use_compact_block_summaries(),
    )


def _quantile_payload(sketch, compute_now=True):
    if not REQUIRES_QUANTILES or not compute_now or sketch is None:
        return {"p05": None, "p50": None, "p95": None}
    return sketch.quantiles((0.05, 0.50, 0.95))


def _stats_payload(stats_state, candidate_results, sketch, compute_quantiles=True):
    n = int(stats_state["n"])
    variance = _sample_variance(stats_state)
    stddev = _sample_stddev(stats_state)
    quantiles = _quantile_payload(sketch, compute_now=compute_quantiles)
    valid_rate = n / candidate_results if candidate_results else None
    discard_rate = 1.0 - valid_rate if valid_rate is not None else None
    available = {
        "avg": stats_state["mean"] if n else None,
        "min": stats_state["min"],
        "max": stats_state["max"],
        "var": variance,
        "std": stddev,
        "count": n,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "p05": quantiles["p05"],
        "p50": quantiles["p50"],
        "p95": quantiles["p95"],
    }
    return {name: available[name] for name in REQUESTED_STATISTICS}


def _numerical_diagnostics(stats_state, candidate_results):
    n = int(stats_state["n"])
    discarded = max(0, int(candidate_results) - n)
    valid_rate = n / candidate_results if candidate_results else None
    discard_rate = discarded / candidate_results if candidate_results else None
    stddev = _sample_stddev(stats_state)
    mcse = _mcse_mean(stats_state)
    ci95 = _ci95_mean(stats_state)
    relative = abs(mcse / stats_state["mean"]) if mcse is not None and stats_state["mean"] != 0 else None
    return {
        "n": n,
        "candidate_results": int(candidate_results),
        "sample_min": stats_state["min"],
        "sample_max": stats_state["max"],
        "sample_variance": _sample_variance(stats_state),
        "sample_stddev": stddev,
        "mcse_mean": mcse,
        "ci95_mean": ci95,
        "ci95_mean_type": "normal_approximation",
        "relative_mcse": relative,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "valid_rate_ci95_wilson": _wilson_interval(n, candidate_results),
        "discard_rate_ci95_wilson": _wilson_interval(discarded, candidate_results),
        "nonfinite_policy": "discard_and_report",
    }


def _histogram_payload(counts, edges, data_min, data_max):
    if counts is None or edges is None or data_min is None or data_max is None:
        return {"data_min": None, "data_max": None, "total_count": 0, "bins": []}
    bins = [
        {"x0": float(edges[i]), "x1": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]
    return {
        "data_min": float(data_min),
        "data_max": float(data_max),
        "total_count": int(np.sum(counts, dtype=np.int64)),
        "bins": bins,
    }


def _coerce_result_array(values, iterations):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == iterations:
        return array
    if array.size == 1:
        return np.full(iterations, float(array[0]), dtype=np.float64)
    raise ValueError(
        f"La expresion del modelo produjo {array.size} valores para {iterations} iteraciones."
    )


def _evaluate_variables_philox(logical_block_index, iterations):
    _mc_var_0 = _variable_rng(logical_block_index, 0).normal(loc=10.0, scale=0.5, size=iterations).astype(np.float64, copy=False)
    _mc_var_1 = _variable_rng(logical_block_index, 1).choice(np.array([10.0, 15.0, 20.0], dtype=np.float64), size=iterations, p=np.array([0.5, 0.3, 0.2], dtype=np.float64))
    _mc_var_2 = _variable_rng(logical_block_index, 2).uniform(low=-20.0, high=30.0, size=iterations).astype(np.float64, copy=False)
    branch_values = []
    branch_raw_values = []
    with np.errstate(all="ignore"):
        _raw_0 = _coerce_result_array(((((((-(_mc_var_1))) + (np.sqrt((((np.power((_mc_var_1), (2.0))) - (((((4.0) * (_mc_var_0))) * (_mc_var_2))))))))) / (((2.0) * (_mc_var_0))))), iterations)
        branch_raw_values.append(_raw_0)
        _finite_0 = _raw_0[np.isfinite(_raw_0)]
        branch_values.append(_finite_0)
        _raw_1 = _coerce_result_array(((((((-(_mc_var_1))) - (np.sqrt((((np.power((_mc_var_1), (2.0))) - (((((4.0) * (_mc_var_0))) * (_mc_var_2))))))))) / (((2.0) * (_mc_var_0))))), iterations)
        branch_raw_values.append(_raw_1)
        _finite_1 = _raw_1[np.isfinite(_raw_1)]
        branch_values.append(_finite_1)
    return branch_raw_values, branch_values


def _evaluate_variables_legacy(batch_index, iterations):
    rng = np.random.default_rng(_batch_seed_sequence(batch_index))
    _mc_var_0 = rng.normal(loc=10.0, scale=0.5, size=iterations).astype(np.float64, copy=False)
    _mc_var_1 = rng.choice(np.array([10.0, 15.0, 20.0], dtype=np.float64), size=iterations, p=np.array([0.5, 0.3, 0.2], dtype=np.float64))
    _mc_var_2 = rng.uniform(low=-20.0, high=30.0, size=iterations).astype(np.float64, copy=False)
    branch_values = []
    branch_raw_values = []
    with np.errstate(all="ignore"):
        _raw_0 = _coerce_result_array(((((((-(_mc_var_1))) + (np.sqrt((((np.power((_mc_var_1), (2.0))) - (((((4.0) * (_mc_var_0))) * (_mc_var_2))))))))) / (((2.0) * (_mc_var_0))))), iterations)
        branch_raw_values.append(_raw_0)
        _finite_0 = _raw_0[np.isfinite(_raw_0)]
        branch_values.append(_finite_0)
        _raw_1 = _coerce_result_array(((((((-(_mc_var_1))) - (np.sqrt((((np.power((_mc_var_1), (2.0))) - (((((4.0) * (_mc_var_0))) * (_mc_var_2))))))))) / (((2.0) * (_mc_var_0))))), iterations)
        branch_raw_values.append(_raw_1)
        _finite_1 = _raw_1[np.isfinite(_raw_1)]
        branch_values.append(_finite_1)
    return branch_raw_values, branch_values


def _build_branch_and_paired_records(branch_raw_values, branch_values, iterations, compact_summary=False, compute_batch_quantiles=True):
    branch_results = []
    for branch_index, values in enumerate(branch_values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid_results = int(values.size)
        discarded_results = int(iterations - valid_results)
        stats_state = _stats_from_values(values)
        summary = _summarize_block_values(values, stats_state) if compact_summary else None
        if REQUIRES_QUANTILES and compute_batch_quantiles:
            if summary is not None:
                batch_quantiles = _weighted_quantiles(summary["centers"], summary["weights"])
            elif valid_results > 0:
                q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}
        branch_results.append({
            "id": BRANCH_IDS[branch_index],
            "label": BRANCH_LABELS[branch_index],
            "values": None if compact_summary else values,
            "summary": summary,
            "batch_quantiles": batch_quantiles,
            "candidate_results": int(iterations),
            "valid_results": valid_results,
            "discarded_results": discarded_results,
            "stats_state": stats_state,
            "avg": stats_state["mean"] if valid_results else None,
            "min": stats_state["min"],
            "max": stats_state["max"],
        })

    paired = None
    if BRANCH_COUNT == 2:
        mask_plus = np.isfinite(branch_raw_values[0])
        mask_minus = np.isfinite(branch_raw_values[1])
        mask_pair = mask_plus & mask_minus
        delta_values = branch_raw_values[0][mask_pair] - branch_raw_values[1][mask_pair]
        paired_stats = _stats_from_values(delta_values)
        correlation = None
        if paired_stats["n"] > 1:
            plus_pair = branch_raw_values[0][mask_pair]
            minus_pair = branch_raw_values[1][mask_pair]
            corr_matrix = np.corrcoef(plus_pair, minus_pair)
            candidate_corr = float(corr_matrix[0, 1])
            correlation = candidate_corr if math.isfinite(candidate_corr) else None
        summary = _summarize_block_values(delta_values, paired_stats) if compact_summary else None
        if REQUIRES_QUANTILES and compute_batch_quantiles:
            if summary is not None:
                batch_quantiles = _weighted_quantiles(summary["centers"], summary["weights"])
            elif delta_values.size > 0:
                q05, q50, q95 = np.quantile(delta_values, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}
        paired = {
            "id": "plus_minus_delta",
            "label": "Diferencia pareada: Rama + menos Rama -",
            "candidate_results": int(iterations),
            "valid_results": int(delta_values.size),
            "discarded_results": int(iterations - delta_values.size),
            "stats_state": paired_stats,
            "correlation_plus_minus": correlation,
            "values": None if compact_summary else delta_values,
            "summary": summary,
            "batch_quantiles": batch_quantiles,
        }
    return branch_results, paired


def _merge_logical_blocks_into_task_summary(logical_blocks, task_iterations):
    ordered_keys = sorted(logical_blocks)
    branch_results = []
    for b_idx in range(BRANCH_COUNT):
        stats_state = _empty_stats_state()
        valid_total = 0
        c_parts = []
        w_parts = []
        v_parts = []
        for k in ordered_keys:
            lb_b = logical_blocks[k]["branches"][b_idx]
            stats_state = _merge_stats(stats_state, lb_b["stats_state"])
            valid_total += lb_b["valid_results"]
            if REQUIRES_QUANTILES and lb_b["valid_results"] > 0:
                if lb_b["summary"] is not None:
                    c_parts.append(lb_b["summary"]["centers"])
                    w_parts.append(lb_b["summary"]["weights"])
                elif lb_b["values"] is not None:
                    v_parts.append(lb_b["values"])

        if REQUIRES_QUANTILES and valid_total > 0:
            if c_parts:
                batch_quantiles = _weighted_quantiles(np.concatenate(c_parts), np.concatenate(w_parts))
            elif v_parts:
                vals = np.concatenate(v_parts)
                q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}

        branch_results.append({
            "id": BRANCH_IDS[b_idx],
            "label": BRANCH_LABELS[b_idx],
            "values": None,
            "summary": None,
            "batch_quantiles": batch_quantiles,
            "candidate_results": int(task_iterations),
            "valid_results": int(valid_total),
            "discarded_results": int(task_iterations - valid_total),
            "stats_state": stats_state,
            "avg": stats_state["mean"] if valid_total else None,
            "min": stats_state["min"],
            "max": stats_state["max"],
        })

    paired = None
    if BRANCH_COUNT == 2:
        p_stats = _empty_stats_state()
        p_valid = 0
        corrs = []
        c_parts = []
        w_parts = []
        v_parts = []
        for k in ordered_keys:
            lb_p = logical_blocks[k]["paired"]
            p_stats = _merge_stats(p_stats, lb_p["stats_state"])
            p_valid += lb_p["valid_results"]
            if lb_p["correlation_plus_minus"] is not None:
                corrs.append(lb_p["correlation_plus_minus"])
            if REQUIRES_QUANTILES and lb_p["valid_results"] > 0:
                if lb_p["summary"] is not None:
                    c_parts.append(lb_p["summary"]["centers"])
                    w_parts.append(lb_p["summary"]["weights"])
                elif lb_p["values"] is not None:
                    v_parts.append(lb_p["values"])

        if REQUIRES_QUANTILES and p_valid > 0:
            if c_parts:
                batch_quantiles = _weighted_quantiles(np.concatenate(c_parts), np.concatenate(w_parts))
            elif v_parts:
                vals = np.concatenate(v_parts)
                q05, q50, q95 = np.quantile(vals, [0.05, 0.50, 0.95])
                batch_quantiles = {"p05": float(q05), "p50": float(q50), "p95": float(q95)}
            else:
                batch_quantiles = {"p05": None, "p50": None, "p95": None}
        else:
            batch_quantiles = {"p05": None, "p50": None, "p95": None}

        paired = {
            "id": "plus_minus_delta",
            "label": "Diferencia pareada: Rama + menos Rama -",
            "candidate_results": int(task_iterations),
            "valid_results": int(p_valid),
            "discarded_results": int(task_iterations - p_valid),
            "stats_state": p_stats,
            "correlation_plus_minus": float(math.fsum(corrs) / len(corrs)) if corrs else None,
            "values": None,
            "summary": None,
            "batch_quantiles": batch_quantiles,
        }
    return branch_results, paired


def _simulate_batch(batch_index, iterations, start_sample=None, end_sample=None):
    compact = _use_compact_block_summaries()
    if SEED_STRATEGY == "batch_index":
        raw_vals, fin_vals = _evaluate_variables_legacy(batch_index, iterations)
        branch_results, paired = _build_branch_and_paired_records(
            raw_vals, fin_vals, iterations, compact_summary=compact
        )
        return {
            "batch_index": int(batch_index),
            "iterations": int(iterations),
            "start_sample": int(batch_index) * int(iterations),
            "end_sample": (int(batch_index) + 1) * int(iterations),
            "aligned": True,
            "worker_pid": os.getpid(),
            "worker_peak_rss_bytes": _process_peak_rss_bytes(),
            "branches": branch_results,
            "paired": paired,
            "logical_blocks": {
                int(batch_index): {
                    "branches": branch_results,
                    "paired": paired,
                }
            },
        }

    if start_sample is None or end_sample is None:
        start_sample = int(batch_index) * BATCH_SIZE
        end_sample = min(TOTAL_ITERATIONS, start_sample + int(iterations))
    task_iterations = int(end_sample - start_sample)
    first_block = start_sample // EFF_LOGICAL_BLOCK_SIZE
    last_block = (end_sample - 1) // EFF_LOGICAL_BLOCK_SIZE

    aligned = (
        start_sample == first_block * EFF_LOGICAL_BLOCK_SIZE
        and end_sample == min(TOTAL_ITERATIONS, (last_block + 1) * EFF_LOGICAL_BLOCK_SIZE)
    )

    logical_blocks = {}

    if aligned:
        single_block = (first_block == last_block)
        for b in range(first_block, last_block + 1):
            b_start = b * EFF_LOGICAL_BLOCK_SIZE
            b_end = min(TOTAL_ITERATIONS, (b + 1) * EFF_LOGICAL_BLOCK_SIZE)
            b_iters = b_end - b_start
            b_raw, b_fin = _evaluate_variables_philox(b, b_iters)
            lb_branches, lb_paired = _build_branch_and_paired_records(
                b_raw,
                b_fin,
                b_iters,
                compact_summary=compact,
                compute_batch_quantiles=single_block,
            )
            logical_blocks[int(b)] = {
                "branches": lb_branches,
                "paired": lb_paired,
            }

        if single_block:
            only_lb = logical_blocks[int(first_block)]
            return {
                "batch_index": int(batch_index),
                "iterations": task_iterations,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "aligned": True,
                "worker_pid": os.getpid(),
                "worker_peak_rss_bytes": _process_peak_rss_bytes(),
                "branches": only_lb["branches"],
                "paired": only_lb["paired"],
                "logical_blocks": logical_blocks,
            }

        task_branches, task_paired = _merge_logical_blocks_into_task_summary(
            logical_blocks, task_iterations
        )
        return {
            "batch_index": int(batch_index),
            "iterations": task_iterations,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "aligned": True,
            "worker_pid": os.getpid(),
            "worker_peak_rss_bytes": _process_peak_rss_bytes(),
            "branches": task_branches,
            "paired": task_paired,
            "logical_blocks": logical_blocks,
        }

    task_raw_per_branch = [[] for _ in range(BRANCH_COUNT)]
    task_fin_per_branch = [[] for _ in range(BRANCH_COUNT)]

    for b in range(first_block, last_block + 1):
        b_start = b * EFF_LOGICAL_BLOCK_SIZE
        b_end = min(TOTAL_ITERATIONS, (b + 1) * EFF_LOGICAL_BLOCK_SIZE)
        b_iters = b_end - b_start
        b_raw, b_fin = _evaluate_variables_philox(b, b_iters)

        if start_sample < b_end <= end_sample:
            lb_branches, lb_paired = _build_branch_and_paired_records(
                b_raw, b_fin, b_iters, compact_summary=compact
            )
            logical_blocks[int(b)] = {
                "branches": lb_branches,
                "paired": lb_paired,
            }

        slice_lo = max(start_sample, b_start) - b_start
        slice_hi = min(end_sample, b_end) - b_start
        for idx in range(BRANCH_COUNT):
            sub_raw = b_raw[idx][slice_lo:slice_hi]
            task_raw_per_branch[idx].append(sub_raw)
            task_fin_per_branch[idx].append(sub_raw[np.isfinite(sub_raw)])

    merged_raw = [np.concatenate(parts) for parts in task_raw_per_branch]
    merged_fin = [np.concatenate(parts) for parts in task_fin_per_branch]
    task_branches, task_paired = _build_branch_and_paired_records(
        merged_raw, merged_fin, task_iterations, compact_summary=compact
    )
    return {
        "batch_index": int(batch_index),
        "iterations": task_iterations,
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "aligned": False,
        "worker_pid": os.getpid(),
        "worker_peak_rss_bytes": _process_peak_rss_bytes(),
        "branches": task_branches,
        "paired": task_paired,
        "logical_blocks": logical_blocks,
    }


def _fixed_batch_plan():
    batches = []
    offset = 0
    batch_index = 0
    while offset < TOTAL_ITERATIONS:
        size = min(BATCH_SIZE, TOTAL_ITERATIONS - offset)
        batches.append((batch_index, size, offset, offset + size))
        batch_index += 1
        offset += size
    return batches


def _plan_after_pilot(pilot_size, pilot_elapsed):
    """Calibra el coste con el bloque piloto y construye el plan adaptativo unificado."""
    if "montecarlo_dsl.scheduler" in sys.modules and "montecarlo_dsl.cost_model" in sys.modules:
        _cm_mod = sys.modules["montecarlo_dsl.cost_model"]
        _sc_mod = sys.modules["montecarlo_dsl.scheduler"]
        cost_model = _cm_mod.InterpretableASTCostModel(
            weights_seconds_per_sample=COST_MODEL_WEIGHTS,
            prior_weights_seconds_per_sample=PRIOR_COST_MODEL_WEIGHTS,
            serial_task_overhead_seconds=SERIAL_TASK_OVERHEAD_SECONDS,
            parallel_pool_spawn_seconds=PARALLEL_POOL_SPAWN_SECONDS,
            parallel_task_ipc_seconds=PARALLEL_TASK_IPC_SECONDS,
            prior_equivalent_samples=PRIOR_EQUIVALENT_SAMPLES,
        )
        features = _cm_mod.ASTCostFeatures(
            num_random_variables=AST_COST_FEATURES["num_random_variables"],
            num_normal_vars=AST_COST_FEATURES["num_normal_vars"],
            num_uniform_vars=AST_COST_FEATURES["num_uniform_vars"],
            num_discrete_vars=AST_COST_FEATURES["num_discrete_vars"],
            discrete_support_total=AST_COST_FEATURES["discrete_support_total"],
            num_add_sub=AST_COST_FEATURES["num_add_sub"],
            num_mul=AST_COST_FEATURES["num_mul"],
            num_div_frac=AST_COST_FEATURES["num_div_frac"],
            num_power=AST_COST_FEATURES["num_power"],
            num_sqrt=AST_COST_FEATURES["num_sqrt"],
            num_unary=AST_COST_FEATURES["num_unary"],
            ast_node_count=AST_COST_FEATURES["ast_node_count"],
            ast_depth=AST_COST_FEATURES["ast_depth"],
            branch_count=AST_COST_FEATURES["branch_count"],
            requested_stats_count=AST_COST_FEATURES["requested_stats_count"],
            requires_quantiles=AST_COST_FEATURES["requires_quantiles"],
            estimated_bytes_per_sample=AST_COST_FEATURES["estimated_bytes_per_sample"],
        )
        scheduler = _sc_mod.AdaptiveMonteCarloScheduler(cost_model)
        plan = scheduler.build_plan(
            features,
            total_iterations=TOTAL_ITERATIONS,
            batch_size=BATCH_SIZE,
            logical_block_size=LOGICAL_BLOCK_SIZE,
            max_workers=MAX_WORKERS,
            scheduler_mode=SCHEDULER_MODE,
            pilot_iterations=pilot_size,
            pilot_elapsed_seconds=pilot_elapsed,
            start_logical_block=1,
            initial_task_index=1,
        )
        return (
            plan.task_tuples(),
            plan.effective_workers,
            plan.use_process_pool,
            plan.cost_estimate.pilot_per_sample_seconds,
            plan.cost_estimate.effective_per_sample_seconds,
        )

    block_call_overhead = SERIAL_TASK_OVERHEAD_SECONDS * (1.6 if BRANCH_COUNT == 2 else 1.0)
    net_pilot = max(1.0e-6, float(pilot_elapsed) - block_call_overhead)
    pilot_per_sample = max(5.0e-9, net_pilot / float(pilot_size))
    if SCHEDULER_MODE == "pilot_only":
        effective_per_sample = pilot_per_sample
    else:
        alpha = min(0.88, max(0.15, float(pilot_size) / float(pilot_size + PRIOR_EQUIVALENT_SAMPLES)))
        effective_per_sample = (1.0 - alpha) * STATIC_PER_SAMPLE_SECONDS + alpha * pilot_per_sample

    remaining_blocks = max(0, LOGICAL_BLOCK_COUNT - 1)
    remaining_iterations = max(0, TOTAL_ITERATIONS - EFF_LOGICAL_BLOCK_SIZE)
    if remaining_blocks == 0:
        return [], 1, False, pilot_per_sample, effective_per_sample

    cpu_cap = os.cpu_count() or 1
    max_viable_workers = min(MAX_WORKERS, cpu_cap, remaining_blocks)
    memory_budget = 32 * 1024 * 1024
    max_samples_per_task = max(
        EFF_LOGICAL_BLOCK_SIZE,
        memory_budget // max(8, ESTIMATED_BYTES_PER_SAMPLE),
    )
    max_blocks_by_memory = max(1, min(remaining_blocks, max_samples_per_task // EFF_LOGICAL_BLOCK_SIZE))

    target_serial_tasks = min(remaining_blocks, 4 if remaining_iterations <= 250_000 else 8)
    serial_blocks = min(
        max_blocks_by_memory,
        max(1, math.ceil(remaining_blocks / max(1, target_serial_tasks))),
    )
    serial_tasks = math.ceil(remaining_blocks / serial_blocks)
    pure_work = float(remaining_iterations) * float(effective_per_sample)
    best_seconds = pure_work + serial_tasks * SERIAL_TASK_OVERHEAD_SECONDS
    best_workers = 1
    best_blocks_per_task = serial_blocks

    for w in range(2, max_viable_workers + 1):
        target_parallel_tasks = min(remaining_blocks, max(2 * w, min(4 * w, remaining_blocks)))
        cand_blocks = min(max_blocks_by_memory, max(1, remaining_blocks // target_parallel_tasks))
        num_tasks = math.ceil(remaining_blocks / cand_blocks)
        efficiency = max(0.55, 1.0 - 0.08 * (w - 1))
        imbalance_tail = 0.25 * (cand_blocks * EFF_LOGICAL_BLOCK_SIZE * effective_per_sample)
        cand_seconds = (
            PARALLEL_POOL_SPAWN_SECONDS
            + pure_work / (w * efficiency)
            + num_tasks * PARALLEL_TASK_IPC_SECONDS
            + imbalance_tail
        )
        if cand_seconds * 1.12 < best_seconds:
            best_seconds = cand_seconds
            best_workers = w
            best_blocks_per_task = cand_blocks

    tasks = []
    b = 1
    idx = 1
    tail_threshold = (
        max(1, LOGICAL_BLOCK_COUNT - 2 * best_workers * max(1, best_blocks_per_task // 2))
        if best_workers > 1 and best_blocks_per_task > 1
        else LOGICAL_BLOCK_COUNT
    )
    while b < LOGICAL_BLOCK_COUNT:
        step = (
            max(1, best_blocks_per_task // 2)
            if b >= tail_threshold and best_workers > 1
            else best_blocks_per_task
        )
        next_b = min(LOGICAL_BLOCK_COUNT, b + step)
        s_lo = b * EFF_LOGICAL_BLOCK_SIZE
        s_hi = min(TOTAL_ITERATIONS, next_b * EFF_LOGICAL_BLOCK_SIZE)
        tasks.append((idx, s_hi - s_lo, s_lo, s_hi))
        b = next_b
        idx += 1

    return tasks, best_workers, (best_workers > 1), pilot_per_sample, effective_per_sample


def _neumaier_add(current_sum, current_comp, value):
    val = float(value)
    t = current_sum + val
    if abs(current_sum) >= abs(val):
        current_comp += (current_sum - t) + val
    else:
        current_comp += (val - t) + current_sum
    return t, current_comp


def _new_branch_state(branch_id, label):
    return {
        "id": branch_id,
        "label": label,
        "progress_sketch": None,
        "progress_step": 0,
        "canonical_sketch": _make_streaming_histogram(),
        "next_canonical_block": 0,
        "canonical_stats": _empty_stats_state(),
        "candidate_results": 0,
        "valid_results": 0,
        "discarded_results": 0,
        "stats_state": _empty_stats_state(),
        "batch_stats": {},
        "histogram_counts": None,
        "histogram_edges": None,
    }


def _new_paired_state():
    return {
        "id": "plus_minus_delta",
        "label": "Diferencia pareada: Rama + menos Rama -",
        "progress_sketch": None,
        "progress_step": 0,
        "progress_corr_sum": 0.0,
        "progress_corr_comp": 0.0,
        "progress_corr_count": 0,
        "canonical_sketch": _make_streaming_histogram(),
        "next_canonical_block": 0,
        "canonical_stats": _empty_stats_state(),
        "candidate_results": 0,
        "valid_results": 0,
        "discarded_results": 0,
        "stats_state": _empty_stats_state(),
        "batch_stats": {},
        "correlations": {},
        "corr_sum": 0.0,
        "corr_comp": 0.0,
        "corr_count": 0,
        "next_corr_block": 0,
    }


def _advance_state_frontier(state):
    """Avanza la frontera contigua de bloques logicos y libera los bloques ya reducidos (O(1) memoria)."""
    if MEMORY_MODE != "bounded":
        return
    while state["next_canonical_block"] in state["batch_stats"]:
        b_idx = state["next_canonical_block"]
        b_stats = state["batch_stats"].pop(b_idx)
        state["canonical_stats"] = _merge_stats(state["canonical_stats"], b_stats)
        state["next_canonical_block"] = b_idx + 1


def _advance_paired_frontier(paired_state):
    """Avanza la frontera contigua de estadisticos y correlaciones pareadas, liberando cada entrada procesada (O(1))."""
    if paired_state is None:
        return
    _advance_state_frontier(paired_state)
    if MEMORY_MODE != "bounded":
        return
    while paired_state["next_corr_block"] in paired_state["correlations"]:
        c_idx = paired_state["next_corr_block"]
        corr_val = paired_state["correlations"].pop(c_idx)
        if corr_val is not None:
            paired_state["corr_sum"], paired_state["corr_comp"] = _neumaier_add(
                paired_state["corr_sum"],
                paired_state["corr_comp"],
                corr_val,
            )
            paired_state["corr_count"] += 1
        paired_state["next_corr_block"] = c_idx + 1


def _canonical_correlation_mean(paired_state):
    if paired_state is None:
        return None
    c_sum = float(paired_state["corr_sum"])
    c_comp = float(paired_state["corr_comp"])
    c_count = int(paired_state["corr_count"])
    for k in sorted(paired_state["correlations"]):
        val = paired_state["correlations"][k]
        if val is not None:
            c_sum, c_comp = _neumaier_add(c_sum, c_comp, val)
            c_count += 1
    if c_count <= 0:
        return None
    return float((c_sum + c_comp) / c_count)


def _progress_correlation_mean(paired_state):
    if paired_state is None or paired_state["progress_corr_count"] <= 0:
        return None
    return float(
        (paired_state["progress_corr_sum"] + paired_state["progress_corr_comp"])
        / paired_state["progress_corr_count"]
    )


def _batch_item_stats(batch_result):
    n = int(batch_result["stats_state"]["n"])
    cand = int(batch_result["candidate_results"])
    variance = _sample_variance(batch_result["stats_state"])
    stddev = _sample_stddev(batch_result["stats_state"])
    quantiles = batch_result.get("batch_quantiles") or {"p05": None, "p50": None, "p95": None}
    valid_rate = n / cand if cand else None
    discard_rate = 1.0 - valid_rate if valid_rate is not None else None
    available = {
        "avg": batch_result["stats_state"]["mean"] if n else None,
        "min": batch_result["stats_state"]["min"],
        "max": batch_result["stats_state"]["max"],
        "var": variance,
        "std": stddev,
        "count": n,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "p05": quantiles["p05"],
        "p50": quantiles["p50"],
        "p95": quantiles["p95"],
    }
    return {name: available[name] for name in REQUESTED_STATISTICS}


def _branch_payload(state, batch_result=None, is_final=False):
    stats_state = _stats_from_state(state) if is_final else state["stats_state"]
    candidate_results = int(state["candidate_results"])
    active_sketch = (
        state["canonical_sketch"]
        if (is_final or state["progress_sketch"] is None)
        else state["progress_sketch"]
    )
    if is_final and state["valid_results"] > 0:
        counts, edges, _, _ = state["canonical_sketch"].build_histogram(
            override_min=stats_state["min"],
            override_max=stats_state["max"],
        )
    else:
        counts = state["histogram_counts"]
        edges = state["histogram_edges"]
    payload = {
        "id": state["id"],
        "label": state["label"],
        "candidate_results": candidate_results,
        "valid_results": int(state["valid_results"]),
        "discarded_results": int(state["discarded_results"]),
        "stats": _stats_payload(
            stats_state,
            candidate_results,
            active_sketch,
            compute_quantiles=is_final or (MEMORY_MODE == "retain_all" and SEED_STRATEGY == "batch_index") or TOTAL_ITERATIONS <= EXACT_SAMPLE_RETENTION_LIMIT,
        ),
        "diagnostics": _numerical_diagnostics(stats_state, candidate_results),
        "histogram": _histogram_payload(
            counts,
            edges,
            stats_state["min"],
            stats_state["max"],
        ),
    }
    if batch_result is not None:
        payload["batch"] = {
            "candidate_results": batch_result["candidate_results"],
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _batch_item_stats(batch_result),
        }
    return payload


def _paired_payload(state, batch_result=None, is_final=False):
    if state is None:
        return None
    stats_state = _stats_from_state(state) if is_final else state["stats_state"]
    candidate_results = int(state["candidate_results"])
    active_sketch = (
        state["canonical_sketch"]
        if (is_final or state["progress_sketch"] is None)
        else state["progress_sketch"]
    )
    corr_mean = _canonical_correlation_mean(state) if is_final else _progress_correlation_mean(state)
    payload = {
        "id": state["id"],
        "label": state["label"],
        "candidate_results": candidate_results,
        "valid_results": int(state["valid_results"]),
        "discarded_results": int(state["discarded_results"]),
        "stats": _stats_payload(
            stats_state,
            candidate_results,
            active_sketch,
            compute_quantiles=is_final or (MEMORY_MODE == "retain_all" and SEED_STRATEGY == "batch_index") or TOTAL_ITERATIONS <= EXACT_SAMPLE_RETENTION_LIMIT,
        ),
        "diagnostics": _numerical_diagnostics(stats_state, candidate_results),
        "correlation_plus_minus_mean": corr_mean,
    }
    if batch_result is not None:
        payload["batch"] = {
            "candidate_results": batch_result["candidate_results"],
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _batch_item_stats(batch_result),
            "correlation_plus_minus": batch_result["correlation_plus_minus"],
        }
    return payload


def _retained_coordinator_payload_bytes(branch_states, paired_state, pending_logical_blocks=None):
    """Contabiliza los bytes de carga util retenidos en las estructuras del coordinador + bloque de trabajo."""
    total_bytes = 0
    for state in branch_states.values():
        total_bytes += int(state["canonical_sketch"].retained_bytes())
        if state["progress_sketch"] is not None:
            total_bytes += int(state["progress_sketch"].retained_bytes())
        total_bytes += len(state["batch_stats"]) * 64
    if paired_state is not None:
        total_bytes += int(paired_state["canonical_sketch"].retained_bytes())
        if paired_state["progress_sketch"] is not None:
            total_bytes += int(paired_state["progress_sketch"].retained_bytes())
        total_bytes += len(paired_state["batch_stats"]) * 64
        total_bytes += len(paired_state["correlations"]) * 32
    if pending_logical_blocks:
        for lb_data in pending_logical_blocks.values():
            for lb_b in lb_data["branches"]:
                if lb_b["summary"] is not None:
                    total_bytes += int(lb_b["summary"]["centers"].nbytes + lb_b["summary"]["weights"].nbytes)
                elif lb_b["values"] is not None:
                    total_bytes += int(lb_b["values"].nbytes)
            if lb_data.get("paired") is not None:
                lb_p = lb_data["paired"]
                if lb_p["summary"] is not None:
                    total_bytes += int(lb_p["summary"]["centers"].nbytes + lb_p["summary"]["weights"].nbytes)
                elif lb_p["values"] is not None:
                    total_bytes += int(lb_p["values"].nbytes)
    working_block = EFF_LOGICAL_BLOCK_SIZE if MEMORY_MODE == "bounded" else BATCH_SIZE
    total_bytes += int(working_block * ESTIMATED_BYTES_PER_SAMPLE)
    return total_bytes


def _metadata(worker_count, batch_count, runtime_extra=None):
    if SEED_STRATEGY == "philox_logical_block":
        rng_name = np.random.default_rng(np.random.Philox(np.random.SeedSequence([MASTER_SEED, 0, 0]))).bit_generator.__class__.__name__
        seed_desc = "numpy.SeedSequence([MASTER_SEED, logical_block_index, var_index]) -> Philox"
    else:
        rng_name = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 0])).bit_generator.__class__.__name__
        seed_desc = "numpy.SeedSequence([MASTER_SEED, batch_index])"
    meta = {
        "statistical_core_version": STATISTICAL_CORE_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "rng_bit_generator": rng_name,
        "seed_strategy": seed_desc,
        "scheduler_mode": SCHEDULER_MODE,
        "logical_block_size": int(EFF_LOGICAL_BLOCK_SIZE),
        "logical_block_count": int(LOGICAL_BLOCK_COUNT),
        "memory_mode": MEMORY_MODE,
        "optimizer_enabled": OPTIMIZER_ENABLED,
        "optimization_metrics": OPTIMIZATION_METRICS,
        "ast_cost_features": AST_COST_FEATURES,
        "cost_model_weights_seconds_per_sample": list(COST_MODEL_WEIGHTS),
        "static_per_sample_seconds": STATIC_PER_SAMPLE_SECONDS,
        "nonfinite_policy": "discard_and_report",
        "ci95_mean_type": "normal_approximation",
        "batch_count": int(batch_count),
        "worker_count": int(worker_count),
    }
    if runtime_extra:
        meta.update(runtime_extra)
    return meta


def _ingest_canonical_logical_block(lb_index, lb_data, branch_states, paired_state):
    for lb_branch in lb_data["branches"]:
        b_state = branch_states[lb_branch["id"]]
        b_state["batch_stats"][lb_index] = lb_branch["stats_state"]
        if lb_branch["valid_results"] > 0:
            if lb_branch["summary"] is not None:
                b_state["canonical_sketch"].add_summary(lb_index, lb_branch["summary"])
            elif lb_branch["values"] is not None:
                b_state["canonical_sketch"].add_block(
                    lb_index,
                    lb_branch["values"],
                    precomputed_min=lb_branch["min"],
                    precomputed_max=lb_branch["max"],
                )

    if paired_state is not None and lb_data["paired"] is not None:
        lb_paired = lb_data["paired"]
        paired_state["batch_stats"][lb_index] = lb_paired["stats_state"]
        paired_state["correlations"][lb_index] = lb_paired["correlation_plus_minus"]
        if lb_paired["valid_results"] > 0:
            if lb_paired["summary"] is not None:
                paired_state["canonical_sketch"].add_summary(lb_index, lb_paired["summary"])
            elif lb_paired["values"] is not None:
                paired_state["canonical_sketch"].add_block(
                    lb_index,
                    lb_paired["values"],
                    precomputed_min=lb_paired["stats_state"]["min"],
                    precomputed_max=lb_paired["stats_state"]["max"],
                )


def _ingest_task_result(result, branch_states, paired_state, pending_logical_blocks, block_covered_samples):
    is_aligned = bool(result.get("aligned", True))

    # 1. Acumulador de progreso: incorpora unicamente los resultados de la tarea fisica completada
    for branch_result in result["branches"]:
        state = branch_states[branch_result["id"]]
        state["candidate_results"] += branch_result["candidate_results"]
        state["valid_results"] += branch_result["valid_results"]
        state["discarded_results"] += branch_result["discarded_results"]
        state["stats_state"] = _merge_stats(state["stats_state"], branch_result["stats_state"])
        if not is_aligned and branch_result["valid_results"] > 0:
            if state["progress_sketch"] is None:
                state["progress_sketch"] = _make_streaming_histogram()
            step = state["progress_step"]
            if branch_result["summary"] is not None:
                state["progress_sketch"].add_summary(step, branch_result["summary"])
            elif branch_result["values"] is not None:
                state["progress_sketch"].add_block(
                    step,
                    branch_result["values"],
                    precomputed_min=branch_result["min"],
                    precomputed_max=branch_result["max"],
                )
            state["progress_step"] = step + 1

    if paired_state is not None and result["paired"] is not None:
        paired = result["paired"]
        paired_state["candidate_results"] += paired["candidate_results"]
        paired_state["valid_results"] += paired["valid_results"]
        paired_state["discarded_results"] += paired["discarded_results"]
        paired_state["stats_state"] = _merge_stats(paired_state["stats_state"], paired["stats_state"])
        if paired["correlation_plus_minus"] is not None:
            paired_state["progress_corr_sum"], paired_state["progress_corr_comp"] = _neumaier_add(
                paired_state["progress_corr_sum"],
                paired_state["progress_corr_comp"],
                paired["correlation_plus_minus"],
            )
            paired_state["progress_corr_count"] += 1
        if not is_aligned and paired["valid_results"] > 0:
            if paired_state["progress_sketch"] is None:
                paired_state["progress_sketch"] = _make_streaming_histogram()
            p_step = paired_state["progress_step"]
            if paired["summary"] is not None:
                paired_state["progress_sketch"].add_summary(p_step, paired["summary"])
            elif paired["values"] is not None:
                paired_state["progress_sketch"].add_block(
                    p_step,
                    paired["values"],
                    precomputed_min=paired["stats_state"]["min"],
                    precomputed_max=paired["stats_state"]["max"],
                )
            paired_state["progress_step"] = p_step + 1

    # 2. Acumulador canonico: incorpora bloques logicos cuando su cobertura esta completa
    new_lb_indices = sorted(result["logical_blocks"])
    if SEED_STRATEGY == "batch_index" or is_aligned:
        for lb_index in new_lb_indices:
            _ingest_canonical_logical_block(
                int(lb_index),
                result["logical_blocks"][lb_index],
                branch_states,
                paired_state,
            )
    else:
        for lb_index, lb_data in result["logical_blocks"].items():
            pending_logical_blocks[int(lb_index)] = lb_data
        start_s = int(result["start_sample"])
        end_s = int(result["end_sample"])
        first_b = start_s // EFF_LOGICAL_BLOCK_SIZE
        last_b = (end_s - 1) // EFF_LOGICAL_BLOCK_SIZE
        for b in range(first_b, last_b + 1):
            b_start = b * EFF_LOGICAL_BLOCK_SIZE
            b_end = min(TOTAL_ITERATIONS, (b + 1) * EFF_LOGICAL_BLOCK_SIZE)
            overlap = max(0, min(end_s, b_end) - max(start_s, b_start))
            block_covered_samples[b] = block_covered_samples.get(b, 0) + overlap

        for lb_index in sorted(list(pending_logical_blocks.keys())):
            b_start = lb_index * EFF_LOGICAL_BLOCK_SIZE
            b_end = min(TOTAL_ITERATIONS, (lb_index + 1) * EFF_LOGICAL_BLOCK_SIZE)
            b_iters = b_end - b_start
            if block_covered_samples.get(lb_index, 0) >= b_iters:
                lb_data = pending_logical_blocks.pop(lb_index)
                block_covered_samples.pop(lb_index, None)
                _ingest_canonical_logical_block(lb_index, lb_data, branch_states, paired_state)

    for b_state in branch_states.values():
        _advance_state_frontier(b_state)
    if paired_state is not None:
        _advance_paired_frontier(paired_state)

    # 3. Actualizacion incremental del histograma de progreso (garantiza total_count == valid_results)
    for branch_index, branch_result in enumerate(result["branches"]):
        state = branch_states[branch_result["id"]]
        if state["valid_results"] <= 0:
            continue
        data_min = state["stats_state"]["min"]
        data_max = state["stats_state"]["max"]
        if not is_aligned or state["progress_sketch"] is not None:
            active_sketch = state["progress_sketch"] or state["canonical_sketch"]
            state["histogram_counts"], state["histogram_edges"], _, _ = active_sketch.build_histogram(
                override_min=data_min,
                override_max=data_max,
            )
            continue

        edges = state["histogram_edges"]
        range_changed = (
            edges is None
            or branch_result["min"] is None
            or float(edges[0]) > data_min
            or float(edges[-1]) < data_max
            or (data_min != data_max and (float(edges[0]) != data_min or float(edges[-1]) != data_max))
        )
        if range_changed or not new_lb_indices:
            state["histogram_counts"], state["histogram_edges"], _, _ = state["canonical_sketch"].build_histogram(
                override_min=data_min,
                override_max=data_max,
            )
        else:
            for lb_index in new_lb_indices:
                lb_branch = result["logical_blocks"][lb_index]["branches"][branch_index]
                if lb_branch["valid_results"] <= 0:
                    continue
                if lb_branch["summary"] is not None:
                    c, _ = np.histogram(
                        lb_branch["summary"]["centers"],
                        bins=state["histogram_edges"],
                        weights=lb_branch["summary"]["weights"],
                    )
                else:
                    c, _ = np.histogram(lb_branch["values"], bins=state["histogram_edges"])
                state["histogram_counts"] += c.astype(np.int64, copy=False)


def _emit_batch_event(result, completed_batches, total_batches, processed_iterations, branch_states, paired_state):
    batch_branch_lookup = {branch["id"]: branch for branch in result["branches"]}
    branches_payload = [
        _branch_payload(state, batch_branch_lookup[state["id"]], is_final=False)
        for state in branch_states.values()
    ]
    valid_total = sum(branch["valid_results"] for branch in branches_payload)
    discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
    paired_batch = result["paired"] if result["paired"] is not None else None

    _emit({
        "type": "batch",
        "batch_index": result["batch_index"],
        "completed_batches": completed_batches,
        "batch_count": total_batches,
        "processed_iterations": processed_iterations,
        "total_iterations": TOTAL_ITERATIONS,
        "progress": processed_iterations / TOTAL_ITERATIONS,
        "candidate_results": processed_iterations * BRANCH_COUNT,
        "valid_results": valid_total,
        "discarded_results": discarded_total,
        "batch": {"iterations": result["iterations"]},
        "branches": branches_payload,
        "paired": _paired_payload(paired_state, paired_batch, is_final=False),
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
        "done": False,
    })


def _order_completed_results(results_list):
    if TASK_COMPLETION_ORDER == "reverse":
        return list(reversed(results_list))
    if TASK_COMPLETION_ORDER == "shuffled":
        shuffled = list(results_list)
        rng = random.Random(MASTER_SEED ^ 0xA5A5)
        rng.shuffle(shuffled)
        return shuffled
    return results_list


def _run():
    _t_sim_start = time.perf_counter()
    coordinator_pid = os.getpid()
    worker_rss_by_pid = {}
    branch_states = {
        branch_id: _new_branch_state(branch_id, label)
        for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
    }
    paired_state = _new_paired_state() if BRANCH_COUNT == 2 else None
    pending_logical_blocks = {}
    block_covered_samples = {}
    processed_iterations = 0
    completed_batches = 0
    peak_retained_coordinator_bytes = 0

    pilot_elapsed = None
    pilot_per_sample = None
    effective_per_sample = STATIC_PER_SAMPLE_SECONDS
    pilot_result = None

    use_fixed_partition = (
        SCHEDULER_MODE == "fixed"
        or SEED_STRATEGY == "batch_index"
        or BATCH_SIZE < LOGICAL_BLOCK_SIZE
    )

    if use_fixed_partition:
        if SCHEDULER_MODE == "fixed" and SEED_STRATEGY != "batch_index" and BATCH_SIZE >= LOGICAL_BLOCK_SIZE:
            batches = list(STATIC_SCHEDULE_TASKS)
            worker_count = STATIC_EFFECTIVE_WORKERS
            use_pool = STATIC_USE_PROCESS_POOL
        else:
            batches = _fixed_batch_plan()
            worker_count = min(MAX_WORKERS, len(batches))
            use_pool = (worker_count > 1 and TOTAL_ITERATIONS >= LOGICAL_BLOCK_SIZE) or (
                SCHEDULER_MODE == "fixed" and TOTAL_ITERATIONS >= LOGICAL_BLOCK_SIZE
            )
        remaining_batches = batches
    elif SCHEDULER_MODE == "ast_only":
        remaining_batches = list(STATIC_SCHEDULE_TASKS)
        worker_count = STATIC_EFFECTIVE_WORKERS
        use_pool = STATIC_USE_PROCESS_POOL
        batches = remaining_batches
    else:
        pilot_size = EFF_LOGICAL_BLOCK_SIZE
        t0 = time.perf_counter()
        pilot_result = _simulate_batch(0, pilot_size, 0, pilot_size)
        pilot_elapsed = time.perf_counter() - t0
        (
            remaining_batches,
            worker_count,
            use_pool,
            pilot_per_sample,
            effective_per_sample,
        ) = _plan_after_pilot(pilot_size, pilot_elapsed)
        batches = [(0, pilot_size, 0, pilot_size)] + remaining_batches

    total_batches = len(batches)
    runtime_extra = {
        "pilot_elapsed_seconds": pilot_elapsed,
        "pilot_per_sample_seconds": pilot_per_sample,
        "effective_per_sample_seconds": effective_per_sample,
        "use_process_pool": bool(use_pool),
    }

    _emit({
        "type": "start",
        "protocol_version": PROTOCOL_VERSION,
        "statistical_core_version": STATISTICAL_CORE_VERSION,
        "seed": str(MASTER_SEED),
        "iterations": TOTAL_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "batch_count": total_batches,
        "workers": worker_count,
        "histogram_bins": HISTOGRAM_BINS,
        "requested_statistics": list(REQUESTED_STATISTICS),
        "branches": [
            {"id": branch_id, "label": label}
            for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
        ],
        "has_plus_minus": BRANCH_COUNT == 2,
        "metadata": _metadata(worker_count, total_batches, runtime_extra),
    })

    def _record_and_emit(result):
        nonlocal completed_batches, processed_iterations, peak_retained_coordinator_bytes
        completed_batches += 1
        processed_iterations += result["iterations"]
        w_pid = int(result.get("worker_pid", coordinator_pid))
        w_rss = int(result.get("worker_peak_rss_bytes", 0))
        if w_pid != coordinator_pid and w_rss > 0:
            worker_rss_by_pid[w_pid] = max(worker_rss_by_pid.get(w_pid, 0), w_rss)
        _ingest_task_result(
            result,
            branch_states,
            paired_state,
            pending_logical_blocks,
            block_covered_samples,
        )
        cur_ret = _retained_coordinator_payload_bytes(
            branch_states, paired_state, pending_logical_blocks
        )
        if cur_ret > peak_retained_coordinator_bytes:
            peak_retained_coordinator_bytes = cur_ret
        _emit_batch_event(
            result,
            completed_batches,
            total_batches,
            processed_iterations,
            branch_states,
            paired_state,
        )

    if pilot_result is not None:
        _record_and_emit(pilot_result)

    if remaining_batches:
        if use_pool:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_simulate_batch, b_idx, size, s_lo, s_hi): b_idx
                    for b_idx, size, s_lo, s_hi in remaining_batches
                }
                if TASK_COMPLETION_ORDER == "as_completed":
                    for future in as_completed(futures):
                        _record_and_emit(future.result())
                else:
                    collected = [f.result() for f in futures]
                    collected.sort(key=lambda item: item["batch_index"])
                    for result in _order_completed_results(collected):
                        _record_and_emit(result)
        else:
            ordered_tasks = _order_completed_results(remaining_batches)
            for b_idx, size, s_lo, s_hi in ordered_tasks:
                result = _simulate_batch(b_idx, size, s_lo, s_hi)
                _record_and_emit(result)

    for lb_index in sorted(list(pending_logical_blocks.keys())):
        lb_data = pending_logical_blocks.pop(lb_index)
        block_covered_samples.pop(lb_index, None)
        _ingest_canonical_logical_block(lb_index, lb_data, branch_states, paired_state)

    for b_state in branch_states.values():
        if b_state["progress_sketch"] is not None:
            b_state["progress_sketch"].clear()
            b_state["progress_sketch"] = None
        _advance_state_frontier(b_state)
    if paired_state is not None:
        if paired_state["progress_sketch"] is not None:
            paired_state["progress_sketch"].clear()
            paired_state["progress_sketch"] = None
        _advance_paired_frontier(paired_state)

    branches_payload = [_branch_payload(state, is_final=True) for state in branch_states.values()]
    valid_total = sum(branch["valid_results"] for branch in branches_payload)
    discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
    if valid_total == 0:
        raise RuntimeError("La simulacion termino sin ningun resultado numerico finito.")

    final_retained_bytes = _retained_coordinator_payload_bytes(
        branch_states, paired_state, pending_logical_blocks
    )
    peak_retained_coordinator_bytes = max(peak_retained_coordinator_bytes, final_retained_bytes)
    coordinator_peak_rss = _process_peak_rss_bytes()
    workers_peak_rss = int(sum(worker_rss_by_pid.values()))
    aggregate_peak_rss = int(coordinator_peak_rss + workers_peak_rss)

    runtime_extra["retained_coordinator_payload_bytes"] = int(final_retained_bytes)
    runtime_extra["peak_retained_coordinator_bytes"] = int(peak_retained_coordinator_bytes)
    runtime_extra["peak_memory_bytes"] = int(peak_retained_coordinator_bytes)
    runtime_extra["coordinator_peak_rss_bytes"] = int(coordinator_peak_rss)
    runtime_extra["workers_peak_rss_bytes"] = int(workers_peak_rss)
    runtime_extra["aggregate_peak_rss_bytes"] = int(aggregate_peak_rss)
    runtime_extra["simulation_compute_seconds"] = time.perf_counter() - _t_sim_start

    _emit({
        "type": "complete",
        "processed_iterations": processed_iterations,
        "total_iterations": TOTAL_ITERATIONS,
        "progress": 1.0,
        "candidate_results": TOTAL_ITERATIONS * BRANCH_COUNT,
        "valid_results": valid_total,
        "discarded_results": discarded_total,
        "branches": branches_payload,
        "paired": _paired_payload(paired_state, is_final=True),
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
        "metadata": _metadata(worker_count, total_batches, runtime_extra),
        "done": True,
    })


def main():
    multiprocessing.freeze_support()
    try:
        _run()
    except Exception as exc:
        _emit({
            "type": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "done": True,
        })
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
