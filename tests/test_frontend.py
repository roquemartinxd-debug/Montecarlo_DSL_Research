from __future__ import annotations

from fastapi.testclient import TestClient

from montecarlo_dsl.job_manager import JobManager
from montecarlo_dsl.server import create_app


def test_frontend_is_served_from_same_fastapi_origin() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="sourceEditor"' in response.text
        assert 'id="histogram"' in response.text
        assert '/static/vendor/d3.v3.min.js' in response.text
        assert '/static/app.js' in response.text
        assert 'https://' not in response.text


def test_frontend_static_assets_are_local_and_available() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        for path in (
            "/static/styles.css",
            "/static/app.js",
            "/static/vendor/d3.v3.min.js",
        ):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.content


def test_frontend_javascript_uses_http_jobs_and_websocket_stream() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        script = client.get("/static/app.js").text
        assert 'fetch("/api/jobs"' in script
        assert 'new WebSocket(' in script
        assert 'event.type === "batch"' in script
        assert 'renderHistogram(branch.histogram, branch.stats || {})' in script
        assert 'event.type === "cancelled"' in script


def test_favicon_does_not_generate_a_404() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        response = client.get("/favicon.ico")
        assert response.status_code == 204


def test_frontend_reconciles_by_polling_if_websocket_closes_early() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        script = client.get("/static/app.js").text
        assert "function scheduleReconcile" in script
        assert "window.setTimeout" in script
        assert 'fetch("/api/jobs/" + encodeURIComponent(state.jobId))' in script
        assert "scheduleReconcile(500)" in script
        assert "scheduleReconcile(1000)" in script

def test_frontend_allows_optional_reproducibility_seed() -> None:
    with TestClient(create_app(JobManager(batch_size=100, max_workers=1))) as client:
        html = client.get("/").text
        script = client.get("/static/app.js").text
        assert 'id="seedInput"' in html
        assert 'placeholder="Vacío = automática"' in html
        assert 'var requestedSeed = seedInput.value.trim();' in script
        assert 'if (requestedSeed) requestBody.seed = requestedSeed;' in script
        assert 'body: JSON.stringify(requestBody)' in script

