# EXP-10 checkpoint migration on Windows (CRLF-aware)

The first migration helper expected source-tree fingerprints generated from an LF checkout. The actual Windows checkpoint recorded a different raw-byte source-tree fingerprint because the campaign fingerprint hashes exact file bytes. The checkpoint itself records:

- old source tree: `3f24390cd58f4079387068730b8eb678965110d337347144b406c7f1f78d0af6`
- old generator: `bf3d67fb65119bed27af42f43072c3cf55fa2a19e70a74b7c85145a023b0a26a`

The corrected generator, after CRLF-to-LF normalization, must hash to:

`0f94f15aa7064f9df9745ed1ce57927db4c76c47f692b10a7602d99a1fdd46f7`

The revised migration script does not assume a platform-independent source-tree hash. It reconstructs the previous tree by hashing the current campaign inputs and replacing only the current generator hash with the generator hash recorded in the checkpoint. That reconstructed digest must equal the checkpoint's original source-tree digest. This proves that all other campaign inputs are byte-identical.

It then verifies the normalized current generator against the reviewed fix, verifies the exact Windows EXP-10 campaign identity, validates all 1,400 expected checkpoint tags and each result checksum, and updates the manifest using the actual current raw-byte hashes. This makes the manifest consistent with the same fingerprint algorithm used by `--resume` on Windows.

Run from the project root, after applying the reorder-buffer hotfix and passing focused tests:

```powershell
python scripts/migrate_exp10_checkpoint_frontier_fix.py --checkpoint-dir benchmarks/checkpoints/targeted/windows_exp10_exploratory
```

Do not edit `manifest.json` manually. The migration updates only the manifest; it does not rewrite the 1,400 checkpoint records or the CSV measurements.

The migration also verifies the separately fingerprinted targeted-experiment runner and the recorded repetition counts before changing the manifest.
