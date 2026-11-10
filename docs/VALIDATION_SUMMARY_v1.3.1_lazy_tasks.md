# Validation summary — v1.3.1 lazy task generation

## Scope

This patch addresses two items from the v1.3.1 review:

1. The automated contract test for `statistical_core_version` expected `1.2.0` although the generated runtime correctly declares `1.3.1`.
2. Fixed and adaptive scheduling plans materialized physical-task descriptors proportional to the number of tasks. The runtime also created full task lists and index maps before submission.

## Implemented changes

- Updated the version contract assertion to expect `1.3.1`.
- Added `LazyPhysicalTaskSequence`, a re-iterable task sequence storing a factory and an integer task count. Iteration and positive indexing/slicing do not materialize the full sequence. Explicit reverse slicing and `task_tuples()` remain opt-in materialization operations for compatibility.
- Added `iter_runtime_schedule_tasks()` and `count_runtime_schedule_tasks()`. The latter calculates the physical task count arithmetically without enumerating tasks.
- Changed fixed and adaptive scheduler plans to retain only constant-size generation parameters and the task count.
- Changed the generated runtime to produce fixed/adaptive tasks on demand. It no longer builds `task_list`, `canonical_tasks`, `canonical_task_indices`, `submission_order_indices`, or `pending_tasks_by_index`.
- Bounded task submission/reordering to windows of at most `2 * worker_count`; futures and completed results share that same in-flight bound.
- Preserved canonical result ingestion. For large bounded-memory serial runs, task ingestion remains canonical and is still lazy.
- Added regression tests for fixed and adaptive plans, one-million-task generated-script size, 100-million-task lazy planning, task-count/iterator equivalence, and generated `fixed`/`ast_only` runtime modes.

## Automated test results

All 242 collected tests passed when run in three independent groups on Linux with Python 3.13.5 and NumPy 2.3.5:

| Group | Test files | Passed |
|---|---|---:|
| 1 | Scheduler, feedback, frontend, generator, generator contract, integration contract, job manager | 70 |
| 2 | Lexer, lexer contract, optimizer, schedule-independent reproducibility | 73 |
| 3 | Parser, parser contract, semantic, semantic contract, server, statistics core, symbol table contract | 99 |
| **Total** | **18 test modules** | **242** |

A single monolithic pytest invocation exceeded the review environment's time limit. The 242 tests were therefore verified in the three groups above. Actual Windows CI and revalidation in the target PowerShell/Python 3.11 environment remain pending.

## Lazy-plan memory microcheck

A local `tracemalloc` check constructed a fixed plan for 100,000,000 one-sample tasks. The returned plan reported `task_count=100000000`, used `LazyPhysicalTaskSequence`, and showed 1,776 bytes of peak Python allocations during plan construction in that run.

This microcheck measures Python allocations for plan construction only. It is not a measurement of the full runtime's RSS or a proof of total process memory complexity. Runtime memory experiments must still be repeated with fixed maximum task size, worker count, and in-flight window.

## Still required before publication

- Run the complete suite in the target PowerShell/Python 3.11 environment and save the log.
- Confirm the CI workflow on Linux and Windows runners.
- Re-run the v1.3.1 experimental campaign; do not reuse v1.3.0 tables as v1.3.1 results.
- Re-measure retained payload and process/worker RSS at increasing iteration counts and under forward, reverse, and shuffled task submission.
- Verify all references and regenerate tables/figures from the new CSV outputs.
