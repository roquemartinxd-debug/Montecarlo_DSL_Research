from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from montecarlo_dsl.job_manager import Job, JobManager, JobStatus
from montecarlo_dsl.server import create_app


VALID_SOURCE = r"""
\model{x=a}
\normal a{0,1}
\iter{100}
\response{avg,min,max}
""".strip()

RUNTIME_ERROR_SOURCE = r"""
\model{x=\sqrt{a}}
\uniform a{-2,-1}
\iter{100}
\response{avg}
""".strip()


def test_generation_phase_diagnostics_use_same_422_contract() -> None:
    tiny = "0." + "0" * 400 + "1"
    source = rf"\model{{x=a}}\normal a{{{tiny},1}}\iter{{1}}\response{{avg}}"

    with TestClient(create_app(JobManager())) as client:
        response = client.post("/api/jobs", json={"source": source})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "compilation_failed"
    assert body["diagnostics"][0]["phase"] == "GENERATION"
    assert body["diagnostics"][0]["code"] == "GEN004"


def test_complete_event_is_synchronized_with_terminal_job_status() -> None:
    manager = JobManager(batch_size=100, max_workers=1, histogram_bins=10)
    with TestClient(create_app(manager)) as client:
        created = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": "9"})
        assert created.status_code == 202
        job = created.json()

        with client.websocket_connect(job["websocket_path"]) as websocket:
            while True:
                event = websocket.receive_json()
                if event["type"] == "complete":
                    break

        status = client.get(f"/api/jobs/{job['job_id']}")
        assert status.status_code == 200
        payload = status.json()
        assert payload["status"] == "completed"
        assert payload["terminal"] is True
        assert payload["last_event"]["type"] == "complete"


def test_runtime_error_event_is_synchronized_with_failed_job_status() -> None:
    manager = JobManager(batch_size=100, max_workers=1, histogram_bins=10)
    with TestClient(create_app(manager)) as client:
        created = client.post(
            "/api/jobs",
            json={"source": RUNTIME_ERROR_SOURCE, "seed": "10"},
        )
        assert created.status_code == 202
        job = created.json()

        with client.websocket_connect(job["websocket_path"]) as websocket:
            while True:
                event = websocket.receive_json()
                if event["type"] == "error":
                    assert event["error_type"] == "RuntimeError"
                    break

        payload = client.get(f"/api/jobs/{job['job_id']}").json()
        assert payload["status"] == "failed"
        assert payload["terminal"] is True
        assert payload["last_event"]["type"] == "error"


def test_http_layer_returns_409_while_another_job_is_active(monkeypatch) -> None:
    release = asyncio.Event()

    async def blocked_run(self: Job) -> None:
        self.status = JobStatus.RUNNING
        await self.hub.publish({"type": "start", "iterations": self.iterations})
        await release.wait()
        self.status = JobStatus.COMPLETED
        await self.hub.publish({"type": "complete", "done": True})

    monkeypatch.setattr(Job, "run", blocked_run)
    manager = JobManager(batch_size=100, max_workers=1)

    with TestClient(create_app(manager)) as client:
        first = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": "1"})
        assert first.status_code == 202

        second = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": "2"})
        assert second.status_code == 409
        assert second.json()["error"] == "job_already_running"

        client.portal.call(release.set)
        job = manager.get(first.json()["job_id"])
        assert job is not None and job.task is not None
        client.portal.call(lambda: asyncio.shield(job.task))


def test_cancel_endpoint_and_websocket_share_cancelled_terminal_state(monkeypatch) -> None:
    async def cancellable_run(self: Job) -> None:
        self.status = JobStatus.RUNNING
        await self.hub.publish({"type": "start", "iterations": self.iterations})
        while not self.cancel_requested:
            await asyncio.sleep(0.005)
        self.status = JobStatus.CANCELLED
        await self.hub.publish(
            {
                "type": "cancelled",
                "job_id": self.job_id,
                "message": "cancelled in contract test",
                "done": True,
            }
        )

    monkeypatch.setattr(Job, "run", cancellable_run)
    manager = JobManager(batch_size=100, max_workers=1)

    with TestClient(create_app(manager)) as client:
        created = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": "3"})
        assert created.status_code == 202
        job = created.json()

        with client.websocket_connect(job["websocket_path"]) as websocket:
            assert websocket.receive_json()["type"] == "start"
            cancelled = client.post(f"/api/jobs/{job['job_id']}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert cancelled.json()["terminal"] is True
            assert websocket.receive_json()["type"] == "cancelled"

        status = client.get(f"/api/jobs/{job['job_id']}").json()
        assert status["status"] == "cancelled"
        assert status["terminal"] is True
        assert status["last_event"]["type"] == "cancelled"


def test_unknown_websocket_job_closes_with_application_404_code() -> None:
    with TestClient(create_app(JobManager())) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/jobs/not-found") as websocket:
                websocket.receive_json()

    assert exc_info.value.code == 4404


def test_seed_accepts_full_128_bit_range_and_rejects_overflow() -> None:
    max_seed = str((1 << 128) - 1)
    too_large = str(1 << 128)

    manager = JobManager(batch_size=100, max_workers=1)
    with TestClient(create_app(manager)) as client:
        accepted = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": max_seed})
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        client.post(f"/api/jobs/{job_id}/cancel")

        rejected = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": too_large})
        assert rejected.status_code == 422


def test_start_event_advertises_protocol_and_requested_statistics() -> None:
    manager = JobManager(batch_size=100, max_workers=1, histogram_bins=10)
    with TestClient(create_app(manager)) as client:
        created = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": "11"}).json()
        with client.websocket_connect(created["websocket_path"]) as websocket:
            start = websocket.receive_json()
            assert start["type"] == "start"
            assert start["protocol_version"] == 2
            assert start["requested_statistics"] == ["avg", "min", "max"]
            assert start["seed"] == "11"

            while True:
                event = websocket.receive_json()
                if event["type"] in {"complete", "error", "cancelled"}:
                    break
