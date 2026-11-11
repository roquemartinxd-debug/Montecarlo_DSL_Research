"""Regression checks for Windows-style spawn in the in-process benchmark runner."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generated_benchmark_module_can_be_imported_by_spawn_workers(tmp_path: Path) -> None:
    """A spawned worker must unpickle tasks from the temporary generated module.

    Even on Linux/macOS, enforce Windows' spawn semantics in a child driver.
    The driver is a file (rather than -c) so spawn can safely import __main__.
    """
    driver = tmp_path / "run_spawn_benchmark.py"
    driver.write_text(
        textwrap.dedent(
            """
            import multiprocessing
            import sys
            import tempfile
            from pathlib import Path

            from benchmarks.run_v1_3_experiments import run_simulation, source_with_iterations
            from montecarlo_dsl.generator import GeneratorConfig

            def main():
                multiprocessing.set_start_method("spawn", force=True)
                source = source_with_iterations("normal_mean", 10000)
                config = GeneratorConfig(
                    batch_size=2500,
                    max_workers=2,
                    seed=101,
                    scheduler_mode="fixed",
                    seed_strategy="batch_index",
                    memory_mode="retain_all",
                    optimize=False,
                    max_iterations=10000,
                )
                original_path = list(sys.path)
                with tempfile.TemporaryDirectory(prefix="mcdsl_spawn_regression_") as tmp:
                    for _ in range(2):
                        result = run_simulation(source, config, temp_dir=Path(tmp))
                        assert result["complete_event"]["valid_results"] == 10000
                        assert result["start_event"]["metadata"]["use_process_pool"] is True
                        assert list(Path(tmp).glob("mcdsl_gen_*.py")) == []
                        assert sys.path == original_path
                print("SPAWN_REGRESSION_PASS", flush=True)

            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(PROJECT_ROOT), str(PROJECT_ROOT / "src"), env.get("PYTHONPATH")])
    )
    proc = subprocess.run(
        [sys.executable, str(driver)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SPAWN_REGRESSION_PASS" in proc.stdout
