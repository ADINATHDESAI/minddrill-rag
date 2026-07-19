"""The agent loop over /chat: tool calls in the SSE stream, guardrails, errors.

/chat routes through `create_agent`; these tests script the model with a fake
that returns canned `AIMessage`s (tool calls and answers) so the loop runs
without a real provider.
"""

import uuid
from types import SimpleNamespace

import httpx
from langchain_core.messages import AIMessage

from minddrill.agent import loop as agent_loop
from minddrill.agent.model import get_agent_model
from minddrill.db.session import SessionLocal
from minddrill.main import app
from minddrill.models.chunk import Chunk
from minddrill.models.document import Document
from tests.conftest import FakeEmbedder, FakeToolModel, parse_sse, register_user


async def _session(client: httpx.AsyncClient, headers: dict) -> str:
    resp = await client.post("/api/v1/sessions", json={"title": "t"}, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()["session_id"]


async def _chat(client, headers, session_id, message):
    return await client.post(
        "/api/v1/chat",
        json={"message": message, "session_id": session_id},
        headers=headers,
    )


def _use_model(model) -> None:
    app.dependency_overrides[get_agent_model] = lambda: model


def _tool_call(name, args, call_id):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


async def test_agent_calls_tool_then_answers(client: httpx.AsyncClient) -> None:
    _, headers = await register_user(client, "agent_calc")
    session_id = await _session(client, headers)

    _use_model(
        FakeToolModel(
            responses=[
                _tool_call("calculator", {"expression": "2 + 2"}, "c1"),
                AIMessage(content="The answer is 4."),
            ]
        )
    )
    resp = await _chat(client, headers, session_id, "what is 2 + 2?")
    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    names = [n for n, _ in events]

    # The stream shows the tool call, then its result, then tokens, then done.
    assert names.index("tool_call") < names.index("tool_result")
    assert names.index("tool_result") < names.index("token")
    assert "done" in names

    call = next(d for n, d in events if n == "tool_call")
    assert call["tool"] == "calculator"
    assert call["args"] == {"expression": "2 + 2"}
    result = next(d for n, d in events if n == "tool_result")
    assert result["result"] == "4"

    # The assistant turn is persisted after a clean completion.
    msgs = (
        await client.get(f"/api/v1/sessions/{session_id}/messages", headers=headers)
    ).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "The answer is 4."


async def test_kb_search_emits_sources_for_citations(
    client: httpx.AsyncClient,
) -> None:
    user_id, headers = await register_user(client, "agent_kb")
    session_id = await _session(client, headers)

    content = "The mascot of the Bluejays is a bird named Ace."
    embedder = FakeEmbedder()
    async with SessionLocal() as db:
        doc = Document(
            user_id=uuid.UUID(user_id),
            source_type="pdf",
            source_uri="x",
            content_hash="h-kb",
            status="done",
        )
        db.add(doc)
        await db.flush()
        db.add(
            Chunk(
                document_id=doc.id,
                user_id=uuid.UUID(user_id),
                chunk_index=0,
                content=content,
                token_count=10,
                embedding=await embedder.embed_query(content),
            )
        )
        await db.commit()

    _use_model(
        FakeToolModel(
            responses=[
                _tool_call("knowledge_base_search", {"query": "mascot bird"}, "k1"),
                AIMessage(content="The mascot is Ace [1]."),
            ]
        )
    )
    resp = await _chat(client, headers, session_id, "who is the mascot?")
    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    names = [n for n, _ in events]

    # The KB hit must produce a sources event so the [1] marker resolves, and the
    # answer is reported grounded.
    assert "sources" in names
    srcs = next(d for n, d in events if n == "sources")["sources"]
    assert srcs and srcs[0]["id"] == 1
    done = next(d for n, d in events if n == "done")
    assert done["grounded"] is True


async def test_step_limit_stops_a_runaway_loop(
    client: httpx.AsyncClient, monkeypatch
) -> None:
    _, headers = await register_user(client, "agent_runaway")
    session_id = await _session(client, headers)

    # A model that never stops calling a tool would loop forever without a cap.
    monkeypatch.setattr(
        agent_loop,
        "get_settings",
        lambda: SimpleNamespace(memory_token_budget=2000, agent_max_steps=2),
    )
    _use_model(
        FakeToolModel(
            responses=[
                _tool_call("calculator", {"expression": "1 + 1"}, f"c{i}")
                for i in range(10)
            ]
        )
    )
    resp = await _chat(client, headers, session_id, "loop forever")
    assert resp.status_code == 200, resp.text
    names = [n for n, _ in parse_sse(resp.text)]

    # The guardrail halts at the model-call limit and still ends the stream.
    assert names.count("tool_call") <= 2
    assert "done" in names


async def test_bad_tool_input_is_handled_not_crashed(
    client: httpx.AsyncClient,
) -> None:
    _, headers = await register_user(client, "agent_badinput")
    session_id = await _session(client, headers)

    # `calculator` requires `expression`; calling it with none fails schema
    # validation. The loop must surface that as a tool result, not a 500.
    _use_model(
        FakeToolModel(
            responses=[
                _tool_call("calculator", {}, "c1"),
                AIMessage(content="Sorry, I could not compute that."),
            ]
        )
    )
    resp = await _chat(client, headers, session_id, "calculate nothing")
    assert resp.status_code == 200, resp.text
    events = parse_sse(resp.text)
    names = [n for n, _ in events]
    assert "tool_result" in names
    assert "error" not in names
    assert "done" in names
