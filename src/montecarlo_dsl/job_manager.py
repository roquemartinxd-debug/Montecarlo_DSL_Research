from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any
from uuid import uuid4

from .compiler import CompiledSimulation


class JobStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}


_TERMINAL_EVENT_TYPES = {"complete", "error", "cancelled"}
_ALLOWED_EVENT_TYPES = {"start", "batch", "complete", "error"}
_EVENT_PREFIX = "MC_DSL_EVENT:"


@dataclass(slots=True)
class JobEventHub:
    history: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, Any] | None]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False

    async def publish(self, event: dict[str, Any]) -> None:
        async with self.lock:
            if self.closed:
                return
            self.history.append(event)
            queues = tuple(self.subscribers)
            if event.get("type") in _TERMINAL_EVENT_TYPES:
                self.closed = True
        for queue in queues:
            queue.put_nowait(event)
            if self.closed:
                queue.put_nowait(None)

    async def subscribe(self) -> tuple[list[dict[str, Any]], asyncio.Queue[dict[str, Any] | None] | None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        async with self.lock:
            replay = list(self.history)
            if self.closed:
                return replay, None
            self.subscribers.add(queue)
            return replay, queue

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any] | None] | None) -> None:
        if queue is None:
            return
        async with self.lock:
            self.subscribers.discard(queue)

    def last_event(self) -> dict[str, Any] | None:
        return self.history[-1] if self.history else None


@dataclass(slots=True)
class Job:
    job_id: str
    iterations: int
    script: str
    seed: int
    hub: JobEventHub = field(default_factory=JobEventHub)
    status: JobStatus = JobStatus.CREATED
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None
    temp_dir: Path | None = None
    stderr_tail: str = ""
    stdout_diagnostics_tail: str = ""
    cancel_requested: bool = False

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    async def run(self) -> None:
        if self.status is not JobStatus.CREATED:
            return
        if self.cancel_requested:
            self.status = JobStatus.CANCELLED
            await self.hub.publish({
                "type": "cancelled",
                "job_id": self.job_id,
                "message": "La simulacion fue cancelada antes de iniciar.",
                "done": True,
            })
            return
        self.status = JobStatus.RUNNING
        saw_complete = False
        saw_error = False
        terminal_event: dict[str, Any] | None = None

        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"montecarlo_{self.job_id[:8]}_"))
            script_path = self.temp_dir / "generated_job.py"
            script_path.write_text(self.script, encoding="utf-8", newline="\n")
            self.script = ""

            process_kwargs: dict[str, Any] = {
                "cwd": str(self.temp_dir),
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if os.name == "nt":
                process_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_kwargs["start_new_session"] = True

            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-u",
                str(script_path),
                **process_kwargs,
            )

            assert self.process.stdout is not None
            assert self.process.stderr is not None
            stderr_task = asyncio.create_task(self._read_stderr(self.process.stderr))

            async for raw_line in self.process.stdout:
                if self.cancel_requested:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                event = self._parse_event(line)
                if event is None:
                    self._remember_stdout_diagnostic(line)
                    continue
                event_type = event["type"]
                if event_type == "complete":
                    saw_complete = True
                    terminal_event = event
                    # El estado terminal se publica despues de confirmar el subproceso.
                    continue
                if event_type == "error":
                    saw_error = True
                    terminal_event = event
                    continue
                await self.hub.publish(event)

            return_code = await self.process.wait()
            await stderr_task

            if self.cancel_requested:
                self.status = JobStatus.CANCELLED
                await self.hub.publish(
                    {
                        "type": "cancelled",
                        "job_id": self.job_id,
                        "message": "La simulacion fue cancelada por el usuario.",
                        "done": True,
                    }
                )
                return

            if saw_error:
                self.status = JobStatus.FAILED
                assert terminal_event is not None
                await self.hub.publish(terminal_event)
                return

            if return_code != 0 or not saw_complete:
                self.status = JobStatus.FAILED
                await self.hub.publish(
                    {
                        "type": "error",
                        "error_type": "SubprocessProtocolError",
                        "message": "El proceso de simulacion termino de forma inesperada.",
                        "done": True,
                    }
                )
                return

            self.status = JobStatus.COMPLETED
            assert terminal_event is not None
            await self.hub.publish(terminal_event)

        except asyncio.CancelledError:
            self.cancel_requested = True
            await self._terminate_process_tree()
            self.status = JobStatus.CANCELLED
            await self.hub.publish(
                {
                    "type": "cancelled",
                    "job_id": self.job_id,
                    "message": "La simulacion fue cancelada.",
                    "done": True,
                }
            )
            raise
        except Exception as exc:
            self.status = JobStatus.FAILED
            await self._terminate_process_tree()
            await self.hub.publish(
                {
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "message": "El backend no pudo ejecutar la simulacion.",
                    "done": True,
                }
            )
        finally:
            self.process = None
            self._cleanup_temp_dir()

    async def cancel(self) -> None:
        if self.terminal:
            return
        self.cancel_requested = True
        await self._terminate_process_tree()
        if self.task is not None and self.task is not asyncio.current_task():
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=5.0)
            except asyncio.TimeoutError:
                self.task.cancel()
            except asyncio.CancelledError:
                pass

    async def _read_stderr(self, stream: asyncio.StreamReader) -> None:
        # Se limita stderr para diagnostico interno del servidor.
        chunks: list[str] = []
        total = 0
        max_chars = 16_384
        while True:
            data = await stream.read(4096)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            chunks.append(text)
            total += len(text)
            while total > max_chars and chunks:
                removed = chunks.pop(0)
                total -= len(removed)
        self.stderr_tail = "".join(chunks)[-max_chars:]

    @staticmethod
    def _parse_event(line: str) -> dict[str, Any] | None:
        if not line.startswith(_EVENT_PREFIX):
            return None
        payload = line[len(_EVENT_PREFIX):]
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("El subproceso emitio un evento con JSON invalido.") from exc
        if not isinstance(event, dict) or event.get("type") not in _ALLOWED_EVENT_TYPES:
            raise RuntimeError("El subproceso emitio un evento fuera del protocolo permitido.")
        return event

    def _remember_stdout_diagnostic(self, line: str) -> None:
        # Cualquier salida que no use el prefijo reservado queda fuera del protocolo
        # estructurado y no puede interferir con los eventos consumidos por FastAPI.
        combined = (self.stdout_diagnostics_tail + "\n" + line).strip()
        self.stdout_diagnostics_tail = combined[-16_384:]

    async def _terminate_process_tree(self) -> None:
        process = self.process
        if process is None or process.returncode is not None:
            return

        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    def _cleanup_temp_dir(self) -> None:
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None


class JobManager:
    """Gestiona un unico trabajo Monte Carlo activo a la vez."""

    def __init__(self, *, batch_size: int = 25_000, max_workers: int = 4, histogram_bins: int = 30):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.histogram_bins = histogram_bins
        self.jobs: dict[str, Job] = {}
        self._active_job_id: str | None = None
        self._lock = asyncio.Lock()

    async def create_job(self, compiled: CompiledSimulation) -> Job:
        async with self._lock:
            active = self.active_job
            if active is not None and not active.terminal:
                raise RuntimeError("another_job_is_active")

            job = Job(
                job_id=uuid4().hex,
                iterations=compiled.iterations,
                script=compiled.script,
                seed=compiled.seed,
            )
            self.jobs = {job.job_id: job}
            self._active_job_id = job.job_id
            job.task = asyncio.create_task(self._run_and_release(job))
            return job

    async def _run_and_release(self, job: Job) -> None:
        try:
            await job.run()
        finally:
            async with self._lock:
                if self._active_job_id == job.job_id:
                    self._active_job_id = None

    @property
    def active_job(self) -> Job | None:
        if self._active_job_id is None:
            return None
        return self.jobs.get(self._active_job_id)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    async def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        await job.cancel()
        return job

    async def shutdown(self) -> None:
        jobs = tuple(self.jobs.values())
        for job in jobs:
            if not job.terminal:
                await job.cancel()
