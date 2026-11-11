from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Montecarlo_DSL_Research"
EXCLUDED_DIRS = {
    ".git", ".venv", ".pytest_cache", ".pytest_tmp", "__pycache__",
    ".mypy_cache", ".ruff_cache", "dist",
}
EXCLUDED_PREFIXES = {
    "benchmarks/results", "benchmarks/checkpoints", "docs/tables",
    "docs/figures", "logs",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def should_include(relative: Path) -> bool:
    parts = relative.parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    posix = relative.as_posix()
    if any(posix == prefix or posix.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES):
        return False
    if relative.suffix in EXCLUDED_SUFFIXES:
        return False
    if relative.name in {".DS_Store", "Thumbs.db"}:
        return False
    return True


def create_source_bundle(output: Path) -> tuple[int, int]:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    count = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if not should_include(relative):
                continue
            archive_name = (Path(PROJECT_NAME) / relative).as_posix()
            archive.write(path, archive_name)
            count += 1
    return count, output.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a clean MC DSL source ZIP for the AWS campaign (excludes local data/checkpoints)."
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "dist" / "Montecarlo_DSL_Research_v1.3.1_aws_source.zip",
    )
    args = parser.parse_args()
    count, size = create_source_bundle(args.output)
    print(f"AWS source bundle: {args.output.expanduser().resolve()}")
    print(f"Files included: {count}")
    print(f"Archive size: {size / (1024 * 1024):.2f} MiB")
    print("Excluded: .venv, .git, logs, checkpoints, prior result CSVs, generated tables/figures and caches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
