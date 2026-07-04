import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import httpx
import jwt
import pytest

from minddrill.auth.security import JWT_ALGORITHM
from minddrill.config import get_settings
from minddrill.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _register(client: httpx.AsyncClient, username: str, password: str) -> httpx.Response:
    return await client.post(
        "/api/v1/auth/register", json={"username": username, "password": password}
    )


async def _login(client: httpx.AsyncClient, username: str, password: str) -> httpx.Response:
    return await client.post(
        "/api/v1/auth/login", json={"username": username, "password": password}
    )


def _make_token(*, sub: str, exp_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "exp": now + exp_delta}
    return jwt.encode(payload, get_settings().jwt_secret, algorithm=JWT_ALGORITHM)


async def test_register_creates_user(client: httpx.AsyncClient) -> None:
    response = await _register(client, "alice", "hunter2")

    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "alice"
    assert uuid.UUID(body["user_id"])


async def test_register_duplicate_username_returns_409(client: httpx.AsyncClient) -> None:
    await _register(client, "bob", "hunter2")
    response = await _register(client, "bob", "different")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


async def test_register_password_too_long_returns_400(client: httpx.AsyncClient) -> None:
    response = await _register(client, "longpass", "x" * 73)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


async def test_register_concurrent_duplicate_returns_409(client: httpx.AsyncClient) -> None:
    responses = await asyncio.gather(
        _register(client, "racer", "hunter2"),
        _register(client, "racer", "hunter2"),
    )

    statuses = sorted(r.status_code for r in responses)
    assert statuses == [201, 409]


async def test_login_correct_password_returns_token(client: httpx.AsyncClient) -> None:
    await _register(client, "carol", "hunter2")
    response = await _login(client, "carol", "hunter2")

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 3600
    assert body["access_token"]


async def test_login_wrong_password_returns_401(client: httpx.AsyncClient) -> None:
    await _register(client, "dave", "hunter2")
    response = await _login(client, "dave", "wrong")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_login_unknown_username_returns_401(client: httpx.AsyncClient) -> None:
    response = await _login(client, "nobody", "hunter2")

    assert response.status_code == 401


async def test_protected_route_without_token_returns_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/_whoami")

    assert response.status_code == 401


async def test_protected_route_with_valid_token_returns_current_user(
    client: httpx.AsyncClient,
) -> None:
    await _register(client, "erin", "hunter2")
    login_response = await _login(client, "erin", "hunter2")
    token = login_response.json()["access_token"]

    response = await client.get(
        "/api/v1/_whoami", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "erin"


async def test_protected_route_with_expired_token_returns_401(client: httpx.AsyncClient) -> None:
    register_response = await _register(client, "frank", "hunter2")
    user_id = register_response.json()["user_id"]
    token = _make_token(sub=user_id, exp_delta=timedelta(seconds=-1))

    response = await client.get(
        "/api/v1/_whoami", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 401


async def test_protected_route_with_tampered_token_returns_401(client: httpx.AsyncClient) -> None:
    await _register(client, "grace", "hunter2")
    login_response = await _login(client, "grace", "hunter2")
    token = login_response.json()["access_token"]
    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")

    response = await client.get(
        "/api/v1/_whoami", headers={"Authorization": f"Bearer {tampered}"}
    )

    assert response.status_code == 401


async def test_protected_route_with_unknown_user_in_token_returns_401(
    client: httpx.AsyncClient,
) -> None:
    token = _make_token(sub=str(uuid.uuid4()), exp_delta=timedelta(hours=1))

    response = await client.get(
        "/api/v1/_whoami", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 401
