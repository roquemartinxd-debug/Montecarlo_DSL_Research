from pathlib import Path
import csv
import json
import math
import subprocess
import sys
import tempfile

from montecarlo_dsl.compiler import compile_source
from montecarlo_dsl.generator import GeneratorConfig

EVENT_PREFIX = "MC_DSL_EVENT:"


ROOT = Path(__file__).resolve().parents[1]

MODEL = (
    ROOT
    / "benchmarks"
    / "models"
    / "robustness"
    / "sqrt_partial_invalid.dsl"
)

OUTPUT = (
    ROOT
    / "benchmarks"
    / "results"
    / "final"
    / "v1.2.0"
    / "nonfinite_validation.csv"
)

SEEDS = [101, 202, 303, 404, 505]

THEORETICAL_VALID_RATE = 0.5
THEORETICAL_MEAN = (2.0 / 3.0) * math.sqrt(10.0)
THEORETICAL_VARIANCE = 5.0 / 9.0


def run_seed(seed: int):

    source = MODEL.read_text(encoding="utf-8")

    compiled = compile_source(
        source,
        GeneratorConfig(
            seed=seed,
            batch_size=25_000,
            max_workers=2,
            histogram_bins=30,
        ),
    )

    with tempfile.TemporaryDirectory() as temp_dir:

        script_path = Path(temp_dir) / "simulation.py"
        script_path.write_text(
            compiled.script,
            encoding="utf-8",
        )

        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for seed {seed}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )

    complete = None

    for line in process.stdout.splitlines():

        if not line.startswith(EVENT_PREFIX):
            continue

        event = json.loads(
            line[len(EVENT_PREFIX):]
        )

        if event.get("type") == "complete":
            complete = event

    if complete is None:
        raise RuntimeError(
            f"No complete event found for seed {seed}"
        )

    stats = complete["stats"]
    diagnostics = complete["diagnostics"]
    metadata = complete["metadata"]

    candidate = complete["candidate_results"]
    valid = complete["valid_results"]
    discarded = complete["discarded_results"]

    valid_rate = stats["valid_rate"]
    discard_rate = stats["discard_rate"]

    estimated_mean = stats["avg"]
    estimated_variance = stats["var"]

    mcse = diagnostics["mcse_mean"]

    mean_abs_error = abs(
        estimated_mean - THEORETICAL_MEAN
    )

    mean_error_mcse = (
        mean_abs_error / mcse
        if mcse > 0
        else float("inf")
    )

    ci_low, ci_high = (
        diagnostics["valid_rate_ci95_wilson"]
    )

    valid_rate_contains_theory = (
        ci_low
        <= THEORETICAL_VALID_RATE
        <= ci_high
    )

    within_3_mcse = mean_error_mcse <= 3.0

    accounting_ok = (
        valid + discarded == candidate
        and stats["count"] == valid
    )

    policy_ok = (
        metadata["nonfinite_policy"]
        == "discard_and_report"
    )

    passed = (
        valid_rate_contains_theory
        and within_3_mcse
        and accounting_ok
        and policy_ok
    )

    return {
        "model": "sqrt_partial_invalid",
        "seed": seed,
        "iterations": complete["total_iterations"],
        "candidate_results": candidate,
        "valid_results": valid,
        "discarded_results": discarded,
        "valid_rate": valid_rate,
        "discard_rate": discard_rate,
        "theoretical_valid_rate": THEORETICAL_VALID_RATE,
        "valid_rate_ci95_low": ci_low,
        "valid_rate_ci95_high": ci_high,
        "valid_rate_contains_theory":
            valid_rate_contains_theory,
        "estimated_mean": estimated_mean,
        "theoretical_valid_mean":
            THEORETICAL_MEAN,
        "mean_absolute_error":
            mean_abs_error,
        "mcse_mean": mcse,
        "mean_error_in_mcse":
            mean_error_mcse,
        "within_3_mcse":
            within_3_mcse,
        "estimated_variance":
            estimated_variance,
        "theoretical_valid_variance":
            THEORETICAL_VARIANCE,
        "variance_absolute_error":
            abs(
                estimated_variance
                - THEORETICAL_VARIANCE
            ),
        "accounting_ok":
            accounting_ok,
        "nonfinite_policy":
            metadata["nonfinite_policy"],
        "policy_ok":
            policy_ok,
        "statistical_core_version":
            metadata["statistical_core_version"],
        "rng_bit_generator":
            metadata["rng_bit_generator"],
        "seed_strategy":
            metadata["seed_strategy"],
        "passed":
            passed,
    }


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for seed in SEEDS:

        print(f"Running seed {seed}...")

        result = run_seed(seed)

        rows.append(result)

        print(
            f"seed={seed} "
            f"valid_rate={result['valid_rate']:.5f} "
            f"mean={result['estimated_mean']:.6f} "
            f"mean_error_mcse="
            f"{result['mean_error_in_mcse']:.3f} "
            f"passed={result['passed']}"
        )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {OUTPUT}")

    if not all(row["passed"] for row in rows):
        raise SystemExit(
            "One or more validation runs failed."
        )

    print("ALL NONFINITE VALIDATION RUNS PASSED")


if __name__ == "__main__":
    main()