import httpx
import structlog

from minddrill.main import app


async def test_request_logs_carry_request_id() -> None:
    transport = httpx.ASGITransport(app=app)
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as captured:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/health")

    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    assert request_id

    request_logs = [entry for entry in captured if "request_id" in entry]
    assert request_logs
    assert all(entry["request_id"] == request_id for entry in request_logs)
    assert {"request_start", "request_end"} <= {
        entry["event"] for entry in request_logs
    }
