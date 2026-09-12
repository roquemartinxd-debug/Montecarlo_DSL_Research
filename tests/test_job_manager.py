from __future__ import annotations

import asyncio

import pytest

from montecarlo_dsl.job_manager import JobManager, JobStatus
from montecarlo_dsl.compiler import compile_program
from montecarlo_dsl.generator import GeneratorConfig
from montecarlo_dsl.semantic import analyze_source


SOURCE = r"""
\model{x=a}
\normal a{0,1}
\iter{10}
\response{avg}
""".strip()


@pytest.mark.asyncio
async def test_manager_allows_only_one_active_job(monkeypatch) -> None:
    manager = JobManager(batch_size=10, max_workers=1)
    program = analyze_source(SOURCE)

    release = asyncio.Event()

    async def blocked_run(self):
        self.status = JobStatus.RUNNING
        await release.wait()
        self.status = JobStatus.COMPLETED
        await self.hub.publish({"type": "complete", "done": True})

    monkeypatch.setattr("montecarlo_dsl.job_manager.Job.run", blocked_run)

    first = await manager.create_job(compile_program(program, GeneratorConfig(seed=1, batch_size=10, max_workers=1)))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="another_job_is_active"):
        await manager.create_job(compile_program(program, GeneratorConfig(seed=2, batch_size=10, max_workers=1)))

    release.set()
    assert first.task is not None
    await first.task

    second = await manager.create_job(compile_program(program, GeneratorConfig(seed=2, batch_size=10, max_workers=1)))
    release.set()
    assert second.task is not None
    await second.task
    assert second.status is JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_immediate_cancel_marks_job_cancelled_and_releases_slot() -> None:
    manager = JobManager(batch_size=10, max_workers=1)
    program = analyze_source(SOURCE)

    first = await manager.create_job(compile_program(program, GeneratorConfig(seed=1, batch_size=10, max_workers=1)))
    cancelled = await manager.cancel(first.job_id)
    assert cancelled is first
    assert first.status is JobStatus.CANCELLED
    assert first.hub.last_event() is not None
    assert first.hub.last_event()["type"] == "cancelled"

    second = await manager.create_job(compile_program(program, GeneratorConfig(seed=2, batch_size=10, max_workers=1)))
    assert second.job_id != first.job_id
    assert manager.get(first.job_id) is None
    await manager.cancel(second.job_id)
