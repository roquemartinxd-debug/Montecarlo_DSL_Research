# Generado por montecarlo_dsl.
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import sys
import traceback

import numpy as np

PROTOCOL_VERSION = 2
EVENT_PREFIX = "MC_DSL_EVENT:"
MASTER_SEED = 12345
TOTAL_ITERATIONS = 1000000
BATCH_SIZE = 25000
MAX_WORKERS = 4
HISTOGRAM_BINS = 30
REQUESTED_STATISTICS = ('avg', 'min', 'max',)
BRANCH_IDS = ('plus', 'minus')
BRANCH_LABELS = ('Rama +', 'Rama -')
BRANCH_COUNT = len(BRANCH_IDS)


def _emit(payload):
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(EVENT_PREFIX + encoded, flush=True)


def _batch_seed(batch_index):
    material = f"{MASTER_SEED}:{batch_index}".encode("ascii")
    digest = hashlib.blake2b(material, digest_size=16).digest()
    return int.from_bytes(digest, "big")


def _stats_payload(average, minimum, maximum):
    available = {"avg": average, "min": minimum, "max": maximum}
    return {name: available[name] for name in REQUESTED_STATISTICS}


def _numerical_diagnostics(valid_results, total_sum, total_sum_sq):
    if valid_results <= 0:
        return {
            "n": 0,
            "sample_stddev": None,
            "mcse_mean": None,
            "ci95_mean": None,
            "relative_mcse": None,
        }

    mean = total_sum / valid_results
    if valid_results > 1:
        numerator = total_sum_sq - (total_sum * total_sum) / valid_results
        variance = max(0.0, numerator / (valid_results - 1))
        sample_stddev = math.sqrt(variance)
        mcse = sample_stddev / math.sqrt(valid_results)
        ci95 = [mean - 1.96 * mcse, mean + 1.96 * mcse]
        relative = abs(mcse / mean) if mean != 0 else None
    else:
        sample_stddev = None
        mcse = None
        ci95 = None
        relative = None

    return {
        "n": int(valid_results),
        "sample_stddev": sample_stddev,
        "mcse_mean": mcse,
        "ci95_mean": ci95,
        "relative_mcse": relative,
    }


def _histogram_payload(counts, edges, data_min, data_max):
    if counts is None or edges is None:
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


def _simulate_batch(batch_index, iterations):
    rng = np.random.default_rng(_batch_seed(batch_index))
    _mc_var_0 = rng.normal(loc=10.0, scale=0.5, size=iterations).astype(np.float64, copy=False)
    _mc_var_1 = rng.choice(np.array([10.0, 15.0, 20.0], dtype=np.float64), size=iterations, p=np.array([0.5, 0.3, 0.2], dtype=np.float64))
    _mc_var_2 = rng.uniform(low=-20.0, high=30.0, size=iterations).astype(np.float64, copy=False)

    branch_values = []
    with np.errstate(all="ignore"):
        _result_0 = np.asarray((((((-(_mc_var_1))) + (np.sqrt((((np.power((_mc_var_1), (2.0))) - (((((4.0) * (_mc_var_0))) * (_mc_var_2))))))))) / (((2.0) * (_mc_var_0)))), dtype=np.float64).reshape(-1)
        _finite_0 = _result_0[np.isfinite(_result_0)]
        branch_values.append(_finite_0)
        _result_1 = np.asarray((((((-(_mc_var_1))) - (np.sqrt((((np.power((_mc_var_1), (2.0))) - (((((4.0) * (_mc_var_0))) * (_mc_var_2))))))))) / (((2.0) * (_mc_var_0)))), dtype=np.float64).reshape(-1)
        _finite_1 = _result_1[np.isfinite(_result_1)]
        branch_values.append(_finite_1)

    branch_results = []
    for branch_index, values in enumerate(branch_values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        valid_results = int(values.size)
        discarded_results = int(iterations - valid_results)
        if valid_results:
            batch_sum = float(np.sum(values, dtype=np.float64))
            batch_sum_sq = float(np.sum(values * values, dtype=np.float64))
            batch_min = float(np.min(values))
            batch_max = float(np.max(values))
            batch_avg = batch_sum / valid_results
        else:
            batch_sum = 0.0
            batch_sum_sq = 0.0
            batch_min = None
            batch_max = None
            batch_avg = None

        branch_results.append({
            "id": BRANCH_IDS[branch_index],
            "label": BRANCH_LABELS[branch_index],
            "values": values,
            "valid_results": valid_results,
            "discarded_results": discarded_results,
            "sum": batch_sum,
            "sum_sq": batch_sum_sq,
            "avg": batch_avg,
            "min": batch_min,
            "max": batch_max,
        })

    return {
        "batch_index": int(batch_index),
        "iterations": int(iterations),
        "branches": branch_results,
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
        "batch_sums": {},
        "batch_sums_sq": {},
        "valid_results": 0,
        "discarded_results": 0,
        "global_min": None,
        "global_max": None,
        "histogram_counts": None,
        "histogram_edges": None,
    }


def _branch_payload(state, batch_result=None):
    valid = state["valid_results"]
    total_sum = math.fsum(state["batch_sums"][index] for index in sorted(state["batch_sums"]))
    total_sum_sq = math.fsum(
        state["batch_sums_sq"][index] for index in sorted(state["batch_sums_sq"])
    )
    average = total_sum / valid if valid else None
    payload = {
        "id": state["id"],
        "label": state["label"],
        "valid_results": valid,
        "discarded_results": state["discarded_results"],
        "stats": _stats_payload(average, state["global_min"], state["global_max"]),
        "diagnostics": _numerical_diagnostics(valid, total_sum, total_sum_sq),
        "histogram": _histogram_payload(
            state["histogram_counts"],
            state["histogram_edges"],
            state["global_min"],
            state["global_max"],
        ),
    }
    if batch_result is not None:
        payload["batch"] = {
            "valid_results": batch_result["valid_results"],
            "discarded_results": batch_result["discarded_results"],
            "stats": _stats_payload(
                batch_result["avg"], batch_result["min"], batch_result["max"]
            ),
        }
    return payload


def _run():
    batches = _batch_plan()
    worker_count = min(MAX_WORKERS, len(batches))

    _emit({
        "type": "start",
        "protocol_version": PROTOCOL_VERSION,
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
    })

    branch_states = {
        branch_id: _new_branch_state(branch_id, label)
        for branch_id, label in zip(BRANCH_IDS, BRANCH_LABELS)
    }
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
                state["valid_results"] += branch_result["valid_results"]
                state["discarded_results"] += branch_result["discarded_results"]
                state["batch_sums"][result["batch_index"]] = branch_result["sum"]
                state["batch_sums_sq"][result["batch_index"]] = branch_result["sum_sq"]

                values = branch_result["values"]
                previous_min = state["global_min"]
                previous_max = state["global_max"]

                if branch_result["valid_results"]:
                    state["all_chunks"].append(values)
                    batch_min = branch_result["min"]
                    batch_max = branch_result["max"]
                    state["global_min"] = (
                        batch_min if state["global_min"] is None else min(state["global_min"], batch_min)
                    )
                    state["global_max"] = (
                        batch_max if state["global_max"] is None else max(state["global_max"], batch_max)
                    )

                    range_changed = (
                        state["global_min"] != previous_min or state["global_max"] != previous_max
                    )
                    if state["histogram_counts"] is None or range_changed:
                        state["histogram_counts"], state["histogram_edges"] = _recompute_histogram(
                            state["all_chunks"], state["global_min"], state["global_max"]
                        )
                    else:
                        new_counts, _ = np.histogram(values, bins=state["histogram_edges"])
                        state["histogram_counts"] += new_counts.astype(np.int64, copy=False)

            branches_payload = [
                _branch_payload(state, batch_branch_lookup[state["id"]])
                for state in branch_states.values()
            ]
            valid_total = sum(branch["valid_results"] for branch in branches_payload)
            discarded_total = sum(branch["discarded_results"] for branch in branches_payload)

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
        "stats": branches_payload[0]["stats"] if BRANCH_COUNT == 1 else None,
        "histogram": branches_payload[0]["histogram"] if BRANCH_COUNT == 1 else None,
        "diagnostics": branches_payload[0]["diagnostics"] if BRANCH_COUNT == 1 else None,
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
