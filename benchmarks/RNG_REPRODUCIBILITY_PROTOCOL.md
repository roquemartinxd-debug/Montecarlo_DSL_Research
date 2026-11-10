# RNG reproducibility: v1.3.1 and legacy controls

## Research question

With a fixed DSL model, master seed, total iteration count, logical block size, and software
version, does changing the physical batch size or worker count change the final numerical output?

## Strategies must not be confused

- **v1.3.1 default (`philox_logical_block`):** each logical block has a deterministic Philox
  substream derived from `SeedSequence([MASTER_SEED, logical_block_index, var_index])`.
  The default logical block size is 5,000; it remains fixed across the comparison.
  Physical task boundaries and worker counts must not change the canonical final numerical result.
- **Legacy emulation (`batch_index`):** each physical batch has a seed derived from
  `SeedSequence([MASTER_SEED, batch_index])`. Changing physical batch size changes the
  random stream segmentation, and consequently is expected to change the result.
  This control uses v1.3.1 code configured in legacy mode; it does **not** execute
  a frozen historical v1.2.0 program.

## Main experiment

Defaults:

- models: `normal_mean`, `uniform_mean`, `discrete_mean`, `linear_combination`
- seed: `101`
- iterations: `1,000,000`
- physical batch size: `25,000`
- logical block size: `5,000`
- workers: `1, 2, 4`
- repetitions: `10` per model/worker configuration
- histogram bins: `30`

Each run generates a program, executes it in a new Python subprocess, canonicalizes
numerical final output, and computes its SHA-256. Timing and execution metadata
(including worker count) are excluded from the hash.

**Main success criterion:** for each model, all hashes under workers 1/2/4 and
all repetitions must be identical.

## Controls

1. **Seed sensitivity:** under default Philox, seeds 101 and 102 must produce different hashes.
2. **Physical batch-size invariance (v1.3.1):** under default Philox with logical block size
   fixed at 5,000, batch sizes 20,000, 25,000, and 50,000 must produce exactly one hash.
3. **Legacy batch-index sensitivity:** under explicit `seed_strategy="batch_index"` and
   `scheduler_mode="fixed"`, batch sizes 20,000, 25,000, and 50,000 must produce more
   than one hash.

This is a deterministic functional check, not evidence of statistically significant
speedup or cross-platform bitwise reproducibility.

## Official execution

```powershell
python benchmarks/rng_reproducibility.py
```

Results are written to:

```text
benchmarks/results/rng_reproducibility/v1.3.1/
```

Outputs include `rng_reproducibility_raw.csv`, `rng_reproducibility_summary.csv`,
`rng_reproducibility_controls.csv`, `environment.json`, and `raw/*.json`.
The controls CSV now includes the `seed_strategy` column to distinguish old
and new RNG semantics.

## Windows smoke test

```powershell
python benchmarks/rng_reproducibility.py --models normal_mean --iterations 100000 --repetitions 2 --output-dir .pytest_tmp/rng_smoke
```

The smoke test should report PASS for model hashes, seed sensitivity, physical
batch-size invariance, legacy batch sensitivity, and overall status.
