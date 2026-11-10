# MC DSL v1.3.1 - Validation Status After EXP-10 to EXP-13

Date: 2026-11-10
Branch: paper1-v1.3.1-candidate
Status: Scientific validation audit - not yet a public release

## 1. Source provenance

### Main Ubuntu campaign snapshot

Source tree SHA-256:
57d8d62a2a18ec4933a4433026ac90b63134008cf6444970b2d6482d60329793

Generator SHA-256:
535e258b28371a5103a38032c6ef51f4ec81cc891fe6ab17660dcb98add95737

Scheduler SHA-256:
9904a4484a7c76b382b5f936a666fa6d21de30b7bc398656687270413f338dfd

Cost model SHA-256:
4c1f8d662948897648bd341b412eb583075be0b23c550158e455f4e73e1f88f9

Optimizer SHA-256:
b24c960c20723e586c519c62e413e5cd042fe7b3cef52a89a4f106970faae0be

Campaign metadata:
git_branch=unknown
git_commit=unknown
git_dirty=false

### Supplementary hotfix snapshot

Source tree SHA-256:
622c226bd70ab7875b9bbd80eebfef75b4484c328bdb5ff8b616f2d6d9e18787

Generator SHA-256:
0f94f15aa7064f9df9745ed1ce57927db4c76c47f692b10a7602d99a1fdd46f7

Scheduler, cost model and optimizer hashes remain unchanged.

The two generator snapshots must not be described as identical.
The main campaign used the earlier generator; EXP-10 to EXP-13
used the subsequent incremental-ingestion revision.

## 2. Completed experimental campaigns

### Main v1.3.1 Ubuntu campaign

- 2,865 new simulations.
- Zero reused simulations.
- One measurement session.
- 1,050 performance observations.
- 320 ablation observations.
- 14 AST cost generalization observations.
- 84 schedule invariance observations.
- 175 statistical validation observations.
- 22 memory comparison observations.
- 10 quantile/histogram fidelity observations.

Environment: Ubuntu 24.04, Python 3.11.9,
NumPy 2.4.6, AWS c7i.2xlarge, four physical cores.

### EXP-10 - Ubuntu scheduler

- 4,200 confirmatory raw measurements.
- 140 summary rows.
- 28 model/iteration cells.
- Five configurations.
- 30 paired repetitions.

Source tree: supplementary hotfix snapshot.

### EXP-11 - Ubuntu memory

- 330 raw measurements.
- 82 summary rows.
- 72 fidelity comparisons.
- 330 structural-bound records.
- 12 log-log slope records.

### EXP-12 - Windows scheduler

- 140 AST calibration observations.
- 14 AST calibration summary rows.
- 24 process-pool cost probes.
- 5,040 confirmatory raw measurements.
- 168 confirmatory summary rows.
- 28 cells and six configurations.
- 30 paired repetitions per cell.

Environment: Windows Server 2022, Python 3.11.9,
NumPy 2.4.6, AWS c7i.2xlarge, four physical cores.

### EXP-13 - Windows memory

- 230 raw measurements.
- 82 summary rows.
- 72 fidelity comparisons.
- 230 structural-bound records.
- 12 log-log slope records.

The campaign includes bounded/retain-all comparisons,
multiple worker configurations and ordering stress tests.

## 3. Revalidation checklist status

Legend:
[x] Evidence available and requirement covered.
[~] Partial evidence or methodological qualification remains.
[ ] Evidence still required.

### Scheduler and cost model

- [x] Main AST cost generalization: eight training models
  and six holdout models, with ten observations per model.
- [x] RQ3 ablation: 320 observations and 32 summary rows.
- [x] EXP-10 Ubuntu scheduler confirmation.
- [x] EXP-12 Windows scheduler confirmation and calibration.
- [x] Result hashes checked across paired scheduler configurations.

### Memory and statistical fidelity

- [x] EXP-11 Ubuntu memory study.
- [x] EXP-13 Windows memory study.
- [x] Measurements include retained payload, process-tree RSS
  sampling, coordinator/worker metrics and structural bounds.
- [x] max_in_flight_tasks is recorded.
- [x] Quantile and histogram fidelity includes paired_delta.
- [~] The exact requested memory/order matrix is not identical
  to the original checklist. The targeted campaigns use
  as_completed, reverse, shuffled and delayed_first; do not
  relabel as_completed as forward.
- [~] RSS is sampled at 10 ms. It is not an exact guarantee
  of the absolute process-tree peak or unique physical memory.
- [~] Do not claim mathematical O(1) complexity from slopes
  or sampled RSS alone.

### Optimizer differential validation

- [x] Float64 constant-folding behavior tested.
- [x] Dynamic identity preservation tested.
- [x] Non-finite propagation tested.
- [x] Decimal rounding case 0.1 + 0.2 tested.
- [~] Explicit optimizer differential coverage for arithmetic
  division-by-zero and overflow still requires review.

### Automated testing and CI

- [x] 242 tests passed in three independent Linux groups
  using Python 3.13.5 and NumPy 2.3.5.
- [x] Windows focused tests and EXP-13 preflight/smoke
  were reported successful after Git installation.
- [~] Preserve the successful Windows focused-test log.
  The original packaged log records the earlier Git failure.
- [ ] Run the complete suite in the target Windows
  Python 3.11 environment and preserve its log.
- [ ] Confirm actual GitHub Actions runs on Ubuntu and Windows.

### Publication artifacts

- [x] Main campaign CSV outputs exist.
- [x] Main campaign tables and figures were generated.
- [ ] Reconcile all tables and figures with the manuscript.
- [ ] Update the manuscript's historical v1.3.0 warning
  and distinguish the two v1.3.1 source snapshots.
- [ ] Verify every DOI and bibliographic reference against
  its primary source.
- [ ] Prepare a reproducibility package with source snapshots,
  manifests, dependencies and campaign artifacts.

## 4. Scientific interpretation constraints

1. Do not pool absolute Ubuntu and Windows timings.
2. Compare platform-specific results and disclose hardware,
   operating system and runtime differences.
3. Do not claim that all campaigns used an identical generator.
4. Preserve the main campaign's unknown Git commit as
   a provenance limitation.
5. Distinguish retained payload, sampled process-tree RSS
   and the sum of historical per-process RSS maxima.
6. Do not describe log-log slopes as proof of O(1).
7. Do not attribute the main campaign's measurements to
   the supplementary hotfix source snapshot.

## 5. Release decision

MC DSL v1.3.1 remains a release candidate.

No public tag or release should be created until:
- the remaining test/CI evidence is reviewed;
- the manuscript and references are reconciled;
- the two experimental source snapshots are documented;
- the final source archive and reproducibility package
  have been verified.

This document supplements, but does not erase,
the original revalidation checklist.
