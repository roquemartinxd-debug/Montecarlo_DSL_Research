# EXP-10 bounded reorder-buffer hotfix (v1.3.1)

## Symptom

A serial `ast_only` bounded warm-up at 5,000,000 iterations failed with:

`RuntimeError: Bounded coordinator exceeded the logical-block reorder buffer.`

The physical task was contiguous and submitted canonically, but contained 125 logical blocks while the coordinator's pending-block limit was 64. The old implementation inserted all summaries before advancing the canonical frontier and falsely treated an in-order task as out-of-order buffering.

## Fix

When a task is aligned with the current canonical frontier and its logical block indices are contiguous, the coordinator streams those blocks through the canonical reducer one at a time. It advances the frontier and releases each block before ingesting the next. Non-contiguous and out-of-order blocks remain subject to the bounded reorder limit.

The scheduler policy, fitted cost-model weights, worker selection, RNG strategy, logical block size, and task plan are unchanged.

## Windows checkpoint migration

The migration helper is CRLF-aware. The original migration helper expected an LF source-tree fingerprint and did not match the actual Windows checkpoint. The corrected helper reconstructs the old source tree by replacing only the current generator hash with the generator hash recorded in the checkpoint. It requires all other campaign inputs to match byte-for-byte, checks the normalized fixed generator, validates all 1,400 expected checkpoint tags and their result checksums, and updates the manifest with the actual local raw-byte hashes.

See `docs/EXP10_CHECKPOINT_MIGRATION_WINDOWS_v1.3.1.md` for the exact procedure. Do not edit the manifest manually.

## Resume

After the corrected migration prints `MIGRATION PASS`, resume with the exact same EXP-10 arguments and directories, adding `--resume`. The previously completed 1,400 measurements remain reusable; the failed 5M warm-up is run again because warm-ups are not saved as measurements.
