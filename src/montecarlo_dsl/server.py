from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .api_models import (
    CompilationFailurePayload,
    CreateJobRequest,
    CreateJobResponse,
    DiagnosticPayload,
    JobStatusResponse,
)
from .compiler import compile_source
from .diagnostics import CompilationError, Diagnostic
from .generator import GeneratorConfig
from .job_manager import JobManager


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _diagnostic_payload(diagnostic: Diagnostic) -> DiagnosticPayload:
    return DiagnosticPayload(
        code=diagnostic.code,
        phase=diagnostic.phase.name,
        message=diagnostic.message,
        line=diagnostic.line,
        column=diagnostic.column,
        start_offset=diagnostic.span.start.offset,
        end_offset=diagnostic.span.end.offset,
        lexeme=diagnostic.lexeme,
        expected=list(diagnostic.expected),
        found=diagnostic.found,
    )


def create_app(manager: JobManager | None = None) -> FastAPI:
    job_manager = manager or JobManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.job_manager = job_manager
        yield
        await job_manager.shutdown()

    app = FastAPI(
        title="Monte Carlo DSL Backend",
        version="1.1.0",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def frontend():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/jobs", response_model=CreateJobResponse, status_code=202)
    async def create_job(payload: CreateJobRequest, request: Request):
        seed = int(payload.seed) if payload.seed is not None else None
        manager: JobManager = request.app.state.job_manager

        try:
            # Las fases LEX/SYN/SEM/GEN concluyen completamente antes de crear
            # el trabajo de ejecucion. Cualquier fallo de compilacion usa HTTP 422.
            compiled = compile_source(
                payload.source,
                GeneratorConfig(
                    batch_size=manager.batch_size,
                    max_workers=manager.max_workers,
                    histogram_bins=manager.histogram_bins,
                    seed=seed,
                ),
            )
            job = await manager.create_job(compiled)
        except CompilationError as exc:
            body = CompilationFailurePayload(
                diagnostics=[_diagnostic_payload(item) for item in exc.diagnostics]
            )
            return JSONResponse(status_code=422, content=body.model_dump())
        except RuntimeError as exc:
            if str(exc) == "another_job_is_active":
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "job_already_running",
                        "message": "Ya existe una simulacion activa. Cancelala o espera a que termine.",
                    },
                )
            raise

        return CreateJobResponse(
            job_id=job.job_id,
            status=job.status.value,
            seed=str(job.seed),
            iterations=compiled.iterations,
            websocket_path=f"/ws/jobs/{job.job_id}",
        )

    @app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
    async def job_status(job_id: str, request: Request):
        manager: JobManager = request.app.state.job_manager
        job = manager.get(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "job_not_found"})
        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status.value,
            seed=str(job.seed),
            iterations=job.iterations,
            terminal=job.terminal,
            last_event=job.hub.last_event(),
        )

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request):
        manager: JobManager = request.app.state.job_manager
        job = await manager.cancel(job_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "job_not_found"})
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "terminal": job.terminal,
        }

    @app.websocket("/ws/jobs/{job_id}")
    async def job_events(websocket: WebSocket, job_id: str):
        manager: JobManager = websocket.app.state.job_manager
        job = manager.get(job_id)
        await websocket.accept()
        if job is None:
            await websocket.close(code=4404, reason="job_not_found")
            return

        replay, queue = await job.hub.subscribe()
        try:
            for event in replay:
                await websocket.send_json(event)
            if queue is None:
                await websocket.close(code=1000)
                return

            while True:
                event = await queue.get()
                if event is None:
                    break
                await websocket.send_json(event)
            await websocket.close(code=1000)
        except WebSocketDisconnect:
            pass
        finally:
            await job.hub.unsubscribe(queue)

    return app


app = create_app()
