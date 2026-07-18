"""Sessions, short-term memory, and the streaming /chat path.

Memory here is conversation history only — /chat does not retrieve, so no
documents are ingested in these tests.
"""

import httpx

from minddrill.main import app
from minddrill.models.message import Message
from minddrill.providers.failover import get_providers
from minddrill.sessions.memory import trim_history
from tests.conftest import parse_sse, register_user


async def _create_session(client: httpx.AsyncClient, headers: dict) -> str:
    resp = await client.post("/api/v1/sessions", json={"title": "t"}, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


async def _chat(
    client: httpx.AsyncClient, headers: dict, session_id: str, message: str
):
    return await client.post(
        "/api/v1/chat",
        json={"message": message, "session_id": session_id},
        headers=headers,
    )


# --- memory trim (pure unit) -----------------------------------------------


def test_trim_drops_oldest_over_budget():
    history = [
        Message(role="user", content="a", token_count=10),
        Message(role="assistant", content="b", token_count=10),
        Message(role="user", content="c", token_count=10),
    ]
    trimmed = trim_history(history, budget=25)
    # Newest two fit in 25; the oldest is dropped. Order stays chronological.
    assert [m.content for m in trimmed] == ["b", "c"]


def test_trim_keeps_newest_even_if_over_budget():
    history = [Message(role="user", content="big", token_count=100)]
    assert trim_history(history, budget=10) == history


# --- history order ----------------------------------------------------------


async def test_history_returns_messages_in_order(client: httpx.AsyncClient) -> None:
    _, headers = await register_user(client, "historian")
    session_id = await _create_session(client, headers)

    r1 = await _chat(client, headers, session_id, "first")
    assert r1.status_code == 200, r1.text
    r2 = await _chat(client, headers, session_id, "second")
    assert r2.status_code == 200, r2.text

    resp = await client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
    assert resp.status_code == 200, resp.text
    messages = resp.json()["messages"]
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [m["content"] for m in messages][::2] == ["first", "second"]


# --- streaming + persistence ------------------------------------------------


async def test_chat_streams_and_persists_both(client: httpx.AsyncClient) -> None:
    _, headers = await register_user(client, "chatter")
    session_id = await _create_session(client, headers)

    resp = await _chat(client, headers, session_id, "hello")
    assert resp.status_code == 200, resp.text
    names = [name for name, _ in parse_sse(resp.text)]
    assert "status" in names
    assert "token" in names
    assert "done" in names

    resp = await client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
    messages = resp.json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "hello"
    assert messages[1]["content"]  # assistant reply persisted


# --- cross-user isolation ---------------------------------------------------


async def test_other_users_session_is_404(client: httpx.AsyncClient) -> None:
    _, owner_headers = await register_user(client, "owner")
    session_id = await _create_session(client, owner_headers)

    _, intruder_headers = await register_user(client, "intruder")

    read = await client.get(
        f"/api/v1/sessions/{session_id}/messages", headers=intruder_headers
    )
    assert read.status_code == 404
    assert read.json()["error"]["code"] == "not_found"

    chat = await _chat(client, intruder_headers, session_id, "hi")
    assert chat.status_code == 404


# --- failed generation persists no assistant message -----------------------


class _MidFailProvider:
    """Yields one token, then raises — a mid-stream provider failure."""

    def __init__(self) -> None:
        self.last_usage = {"input_tokens": 1, "output_tokens": 0}

    async def stream(self, messages, **kwargs):
        yield "partial "
        raise RuntimeError("boom")


async def test_failed_generation_does_not_persist_assistant(
    client: httpx.AsyncClient,
) -> None:
    _, headers = await register_user(client, "unlucky")
    session_id = await _create_session(client, headers)

    app.dependency_overrides[get_providers] = lambda: [_MidFailProvider()]
    try:
        resp = await _chat(client, headers, session_id, "hello")
    finally:
        app.dependency_overrides.pop(get_providers, None)

    assert resp.status_code == 200, resp.text
    names = [name for name, _ in parse_sse(resp.text)]
    assert "error" in names
    assert "done" not in names

    resp = await client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
    messages = resp.json()["messages"]
    # The user turn stays; no assistant turn is written on a failed generation.
    assert [m["role"] for m in messages] == ["user"]


class _FailBeforeFirstToken:
    """Raises before yielding any token — fails during failover, before the stream."""

    async def stream(self, messages, **kwargs):
        raise RuntimeError("429 quota exceeded")
        yield  # pragma: no cover


async def test_chat_all_providers_down_returns_503(client: httpx.AsyncClient) -> None:
    _, headers = await register_user(client, "alldown_chatter")
    session_id = await _create_session(client, headers)

    app.dependency_overrides[get_providers] = lambda: [
        _FailBeforeFirstToken(),
        _FailBeforeFirstToken(),
    ]
    try:
        resp = await _chat(client, headers, session_id, "hello")
    finally:
        app.dependency_overrides.pop(get_providers, None)

    # Failover resolves before the stream opens, so this is a plain JSON 503.
    assert resp.status_code == 503
    assert not resp.headers["content-type"].startswith("text/event-stream")
    assert resp.json()["error"]["code"] == "providers_unavailable"
