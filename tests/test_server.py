from __future__ import annotations

import time

from fastapi.testclient import TestClient

from montecarlo_dsl.job_manager import JobManager
from montecarlo_dsl.server import create_app


VALID_SOURCE = r"""
\model{x=a}
\normal a{0,1}
\iter{100}
\response{avg,min,max}
""".strip()

INVALID_SOURCE = r"""
\model{x=a}
\normal b{0,1}
\iter{10}
\response{avg}
""".strip()


def wait_for_terminal(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["terminal"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("job did not reach a terminal state")


def test_health_endpoint() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_compilation_failure_returns_structured_diagnostics() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        response = client.post("/api/jobs", json={"source": INVALID_SOURCE})
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "compilation_failed"
        assert body["diagnostics"]
        diagnostic = body["diagnostics"][0]
        assert diagnostic["code"].startswith("SEM")
        assert diagnostic["phase"] == "SEMANTIC"
        assert diagnostic["line"] >= 1
        assert diagnostic["column"] >= 1


def test_seed_validation_is_strict() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        response = client.post(
            "/api/jobs",
            json={"source": VALID_SOURCE, "seed": "not-a-number"},
        )
        assert response.status_code == 422


def test_job_streams_json_events_and_completes() -> None:
    manager = JobManager(batch_size=100, max_workers=1, histogram_bins=10)
    with TestClient(create_app(manager)) as client:
        created = client.post(
            "/api/jobs",
            json={"source": VALID_SOURCE, "seed": "12345"},
        )
        assert created.status_code == 202
        job = created.json()
        assert job["seed"] == "12345"
        assert job["iterations"] == 100

        event_types: list[str] = []
        final_event = None
        with client.websocket_connect(job["websocket_path"]) as websocket:
            while True:
                event = websocket.receive_json()
                event_types.append(event["type"])
                if event["type"] in {"complete", "error", "cancelled"}:
                    final_event = event
                    break

        assert event_types[0] == "start"
        assert "batch" in event_types
        assert event_types[-1] == "complete"
        assert final_event is not None
        assert final_event["processed_iterations"] == 100
        assert final_event["valid_results"] == 100
        assert final_event["histogram"]["total_count"] == 100

        status = wait_for_terminal(client, job["job_id"])
        assert status["status"] == "completed"
        assert status["last_event"]["type"] == "complete"

        internal = manager.get(job["job_id"])
        assert internal is not None
        assert internal.temp_dir is None
        assert internal.script == ""


def test_late_websocket_subscriber_receives_event_history() -> None:
    manager = JobManager(batch_size=100, max_workers=1, histogram_bins=10)
    with TestClient(create_app(manager)) as client:
        created = client.post("/api/jobs", json={"source": VALID_SOURCE, "seed": "7"})
        assert created.status_code == 202
        job = created.json()
        status = wait_for_terminal(client, job["job_id"])
        assert status["status"] == "completed"

        received: list[str] = []
        with client.websocket_connect(job["websocket_path"]) as websocket:
            while True:
                try:
                    event = websocket.receive_json()
                except Exception:
                    break
                received.append(event["type"])
                if event["type"] == "complete":
                    break

        assert received[0] == "start"
        assert received[-1] == "complete"


def test_unknown_job_endpoints_return_404() -> None:
    with TestClient(create_app(JobManager())) as client:
        response = client.get("/api/jobs/not-found")
        assert response.status_code == 404
        response = client.post("/api/jobs/not-found/cancel")
        assert response.status_code == 404
