# Generado por montecarlo_dsl v1.2.0.
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import platform
import sys
import traceback

import numpy as np

PROTOCOL_VERSION = 2
STATISTICAL_CORE_VERSION = "1.2.0"
EVENT_PREFIX = "MC_DSL_EVENT:"
MASTER_SEED = 202
TOTAL_ITERATIONS = 100000
BATCH_SIZE = 20000
MAX_WORKERS = 2
HISTOGRAM_BINS = 30
REQUESTED_STATISTICS = ('avg', 'var', 'std', 'count', 'valid_rate', 'discard_rate', 'p05', 'p50', 'p95')
BRANCH_IDS = ('main',)
BRANCH_LABELS = ('Resultado',)
BRANCH_COUNT = len(BRANCH_IDS)


# El runtime v1.2.0 calcula estadisticos de media/varianza con estados por lote
# y combinacion tipo Chan-Golub-LeVeque. Las medias se calculan sobre resultados
# finitos; los NaN/inf se reportan por separado como descartes.
def _emit(payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(EVENT_PREFIX + encoded, flush=True)


def _batch_seed_sequence(batch_index):
    return np.random.SeedSequence([MASTER_SEED, batch_index])


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


def _stats_from_batch_states(batch_states):
    state = _empty_stats_state()
    for batch_index in sorted(batch_states):
        state = _merge_stats(state, batch_states[batch_index])
    return state


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
    if trials <= 0:
        return None
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * trials)) / trials) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _quantile_payload(chunks):
    if not chunks:
        return {"p05": None, "p50": None, "p95": None}
    values = np.concatenate(chunks)
    if values.size == 0:
        return {"p05": None, "p50": None, "p95": None}
    q05, q50, q95 = np.quantile(values, [0.05, 0.50, 0.95])
    return {"p05": float(q05), "p50": float(q50), "p95": float(q95)}


def _stats_payload(stats_state, candidate_results, chunks):
    n = int(stats_state["n"])
    variance = _sample_variance(stats_state)
    stddev = _sample_stddev(stats_state)
    quantiles = _quantile_payload(chunks)
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


def _histogram_edges(data_min, data_max):
    if data_min == data_max:
        padding = max(abs(data_min) * 0.01, 0.5)
        low = data_min - padding
        high = data_max + padding
    else:
        low = data_min
        high = data_max
    return np.linspace(low, high, HISTOGRAM_BINS + 1, dtype=np.float64)


def _recompute_histogram(chunks, data_min, data_max):
    edges = _histogram_edges(data_min, data_max)
    counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    for chunk in chunks:
        chunk_counts, _ = np.histogram(chunk, bins=edges)
        counts += chunk_counts.astype(np.int64, copy=False)
    return counts, edges


def _coerce_result_array(values, iterations):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == iterations:
        return array
    if array.size == 1:
        return np.full(iterations, float(array[0]), dtype=np.float64)
    raise ValueError(
        f"La expresion del modelo produjo {array.size} valores para {iterations} iteraciones."
    )


def _simulate_batch(batch_index, iterations):
    rng = np.random.default_rng(_batch_seed_sequence(batch_index))
    _mc_var_0 = rng.normal(loc=10.0, scale=2.0, size=iterations).astype(np.float64, copy=False)
    _mc_var_1 = rng.uniform(low=0.0, high=6.0, size=iterations).astype(np.float64, copy=False)

    branch_values = []
    branch_raw_values = []
    with np.errstate(all="ignore"):
        _raw_0 = _coerce_result_array((((((2.0) * (_mc_var_0))) + (((3.0) * (_mc_var_1))))), iterations)
        branch_raw_values.append(_raw_0)
        _finite_0 = _raw_0[np.isfinite(_raw_0)]
        branch_values.append(_finite_0)

    branch_results = []
    for branch_index, values in enumerate(branch_values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid_results = int(values.size)
        discarded_results = int(iterations - valid_results)
        stats_state = _stats_from_values(values)
        branch_results.append({
            "id": BRANCH_IDS[branch_index],
            "label": BRANCH_LABELS[branch_index],
            "values": values,
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
        paired = {
            "id": "plus_minus_delta",
            "label": "Diferencia pareada: Rama + menos Rama -",
            "candidate_results": int(iterations),
            "valid_results": int(delta_values.size),
            "discarded_results": int(iterations - delta_values.size),
            "stats_state": paired_stats,
            "correlation_plus_minus": correlation,
            "values": delta_values,
        }

    return {
        "batch_index": int(batch_index),
        "iterations": int(iterations),
        "branches": branch_results,
        "paired": paired,
    }


def _batch_plan():
    batches = []
    remaining = TOTAL_ITERATIONS
    batch_index = 0
    while remaining > 0:
        size = min(BATCH_SIZE, remaining)
        batches.append((batch_index, size))
        batch_index += 1
        remaining -= size
    return batches


def _new_branch_state(branch_id, label):
    return {
        "id": branch_id,
        "label": label,
        "all_chunks": [],
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
        "all_chunks": [],
        "candidate_results": 0,
        "valid_results": 0,
        "discarded_results": 0,
        "stats_state": _empty_stats_state(),
        "batch_stats": {},
        "correlations": {},
    }


def _branch_payload(state, batch_result=None):
    stats_state = _stats_from_batch_states(state["batch_stats"])
    candidate_results = int(state["candidate_results"])
    payload = {
        "id": state["id"],
        "label": state["label"],
        "candidate_results": candidate_results,
        "valid_results": int(state["valid_results"]),
        "discarded_results": int(state["discarded_results"]),
        "stats": _stats_payload(stats_state, candidate_results, state["all_chunks"]),
        "diagnostics": _numerical_diagnostics(stats_state, candidate_results),
        "histogram": _histogram_payload(
            state["histogram_counts"],
            state["histogram_edges"],
            stats_state["min"],
            stats_state["max"],
        ),
    }
    if batch_result is not None:
        payload["batch"] = {
            "candidate_results": batch_result["candidate_results"],
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _stats_payload(
                batch_result["stats_state"],
                batch_result["candidate_results"],
                [batch_result["values"]],
            ),
        }
    return payload


def _paired_payload(state, batch_result=None):
    if state is None:
        return None
    stats_state = _stats_from_batch_states(state["batch_stats"])
    candidate_results = int(state["candidate_results"])
    correlation_values = [value for value in state["correlations"].values() if value is not None]
    payload = {
        "id": state["id"],
        "label": state["label"],
        "candidate_results": candidate_results,
        "valid_results": int(state["valid_results"]),
        "discarded_results": int(state["discarded_results"]),
        "stats": _stats_payload(stats_state, candidate_results, state["all_chunks"]),
        "diagnostics": _numerical_diagnostics(stats_state, candidate_results),
        "correlation_plus_minus_mean": float(math.fsum(correlation_values) / len(correlation_values)) if correlation_values else None,
    }
    if batch_result is not None:
        payload["batch"] = {
            "candidate_results": batch_result["candidate_results"],
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _stats_payload(
                batch_result["stats_state"],
                batch_result["candidate_results"],
                [batch_result["values"]],
            ),
            "correlation_plus_minus": batch_result["correlation_plus_minus"],
        }
    return payload


def _metadata(worker_count, batch_count):
    rng_name = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, 0])).bit_generator.__class__.__name__
    return {
        "statistical_core_version": STATISTICAL_CORE_VERSION,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "rng_bit_generator": rng_name,
        "seed_strategy": "numpy.SeedSequence([MASTER_SEED, batch_index])",
        "nonfinite_policy": "discard_and_report",
        "ci95_mean_type": "normal_approximation",
        "batch_count": int(batch_count),
        "worker_count": int(worker_count),
    }


def _run():
    batches = _batch_plan()
    worker_count = min(MAX_WORKERS, len(batches))

    _emit({
        "type": "start",
        "protocol_version": PROTOCOL_VERSION,
        "statistical_core_version": STATISTICAL_CORE_VERSION,
        "seed": str(MASTER_SEED),
        "iterations": TOTAL_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "batch_count": len(batches),
        "workers": worker_count,
        "histogram_bins": HISTOGRAM_BINS,
        "requested_statistics": list(REQUESTED_STATISTICS),
        "branches": [
            {"id": branch_id, "label": label}
            for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
        ],
        "has_plus_minus": BRANCH_COUNT == 2,
        "metadata": _metadata(worker_count, len(batches)),
    })

    branch_states = {
        branch_id: _new_branch_state(branch_id, label)
        for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
    }
    paired_state = _new_paired_state() if BRANCH_COUNT == 2 else None
    processed_iterations = 0
    completed_batches = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_simulate_batch, batch_index, size): batch_index
            for batch_index, size in batches
        }

        for future in as_completed(futures):
            result = future.result()
            completed_batches += 1
            processed_iterations += result["iterations"]
            batch_branch_lookup = {branch["id"]: branch for branch in result["branches"]}

            for branch_result in result["branches"]:
                state = branch_states[branch_result["id"]]
                previous_min = state["stats_state"]["min"]
                previous_max = state["stats_state"]["max"]
                state["candidate_results"] += branch_result["candidate_results"]
                state["valid_results"] += branch_result["valid_results"]
                state["discarded_results"] += branch_result["discarded_results"]
                state["stats_state"] = _merge_stats(state["stats_state"], branch_result["stats_state"])
                state["batch_stats"][result["batch_index"]] = branch_result["stats_state"]

                values = branch_result["values"]

                if branch_result["valid_results"]:
                    state["all_chunks"].append(values)
                    data_min = state["stats_state"]["min"]
                    data_max = state["stats_state"]["max"]
                    range_changed = data_min != previous_min or data_max != previous_max
                    if state["histogram_counts"] is None or range_changed:
                        state["histogram_counts"], state["histogram_edges"] = _recompute_histogram(
                            state["all_chunks"], data_min, data_max
                        )
                    else:
                        new_counts, _ = np.histogram(values, bins=state["histogram_edges"])
                        state["histogram_counts"] += new_counts.astype(np.int64, copy=False)

            if paired_state is not None and result["paired"] is not None:
                paired = result["paired"]
                paired_state["candidate_results"] += paired["candidate_results"]
                paired_state["valid_results"] += paired["valid_results"]
                paired_state["discarded_results"] += paired["discarded_results"]
                paired_state["stats_state"] = _merge_stats(paired_state["stats_state"], paired["stats_state"])
                paired_state["batch_stats"][result["batch_index"]] = paired["stats_state"]
                paired_state["correlations"][result["batch_index"]] = paired["correlation_plus_minus"]
                if paired["valid_results"]:
                    paired_state["all_chunks"].append(paired["values"])

            branches_payload = [
                _branch_payload(state, batch_branch_lookup[state["id"]])
                for state in branch_states.values()
            ]
            valid_total = sum(branch["valid_results"] for branch in branches_payload)
            discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
            paired_batch = result["paired"] if result["paired"] is not None else None

            _emit({
                "type": "batch",
                "batch_index": result["batch_index"],
                "completed_batches": completed_batches,
                "batch_count": len(batches),
                "processed_iterations": processed_iterations,
                "total_iterations": TOTAL_ITERATIONS,
                "progress": processed_iterations / TOTAL_ITERATIONS,
                "candidate_results": processed_iterations * BRANCH_COUNT,
                "valid_results": valid_total,
                "discarded_results": discarded_total,
                "batch": {"iterations": result["iterations"]},
                "branches": branches_payload,
                "paired": _paired_payload(paired_state, paired_batch),
                "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
                "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
                "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
                "done": False,
            })

    branches_payload = [_branch_payload(state) for state in branch_states.values()]
    valid_total = sum(branch["valid_results"] for branch in branches_payload)
    discarded_total = sum(branch["discarded_results"] for branch in branches_payload)
    if valid_total == 0:
        raise RuntimeError("La simulacion termino sin ningun resultado numerico finito.")

    _emit({
        "type": "complete",
        "processed_iterations": processed_iterations,
        "total_iterations": TOTAL_ITERATIONS,
        "progress": 1.0,
        "candidate_results": TOTAL_ITERATIONS * BRANCH_COUNT,
        "valid_results": valid_total,
        "discarded_results": discarded_total,
        "branches": branches_payload,
        "paired": _paired_payload(paired_state),
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
        "metadata": _metadata(worker_count, len(batches)),
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
