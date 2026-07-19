"""Agent tools: safe calculator, weather, and per-user knowledge-base search.

Tool inputs are the machine-consumed path, so they are validated; these tests
exercise correct results, rejection of unsafe input, and the isolation boundary
on `knowledge_base_search` (it must only ever see the current user's chunks).
"""

import httpx

from minddrill.agent import tools as agent_tools
from minddrill.agent.tools import _calculator, _weather, build_tools, calculate
from minddrill.models.chunk import Chunk
from minddrill.models.document import Document
from minddrill.models.user import User

_SECRET = "The launch code for Project Zephyr is alpha seven seven."


# --- calculator -------------------------------------------------------------


def test_calculator_evaluates_arithmetic():
    assert calculate("2 * (3 + 4)") == 14
    assert calculate("2 ** 10") == 1024
    assert _calculator("10 / 4") == "2.5"


def test_calculator_rejects_non_arithmetic():
    # Names, calls, and attribute access are not arithmetic — never evaluated.
    assert _calculator("__import__('os').system('ls')").startswith("error")
    assert _calculator("a + 1").startswith("error")
    assert _calculator("2 +").startswith("error")


def test_calculator_rejects_oversized_power():
    # A huge exponent would pin a CPU thread building a giant int — rejected.
    assert _calculator("2 ** 100000000").startswith("error")
    assert _calculator("10 ** 9 ** 9").startswith("error")


# --- weather ----------------------------------------------------------------


async def test_weather_formats_current(monkeypatch):
    async def fake_fetch(latitude, longitude):
        return {"temperature": 21.0, "windspeed": 5.0}

    monkeypatch.setattr(agent_tools, "_fetch_weather", fake_fetch)
    result = await _weather(52.52, 13.41)
    assert "21.0" in result and "5.0" in result


async def test_weather_handles_failure(monkeypatch):
    async def boom(latitude, longitude):
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(agent_tools, "_fetch_weather", boom)
    result = await _weather(0.0, 0.0)
    assert result.startswith("error")


# --- knowledge base search isolation ----------------------------------------


def _kb_tool(tools):
    return next(t for t in tools if t.name == "knowledge_base_search")


async def _seed_chunk(session, user_id, embedder, content):
    doc = Document(
        user_id=user_id,
        source_type="pdf",
        source_uri="x",
        content_hash=f"h-{user_id}",
        status="done",
    )
    session.add(doc)
    await session.flush()
    session.add(
        Chunk(
            document_id=doc.id,
            user_id=user_id,
            chunk_index=0,
            content=content,
            token_count=10,
            embedding=await embedder.embed_query(content),
        )
    )


async def test_kb_search_is_scoped_to_current_user(db_session, embedder, reranker):
    alice = User(username="alice_tool", password_hash="x")
    bob = User(username="bob_tool", password_hash="x")
    db_session.add_all([alice, bob])
    await db_session.flush()
    await _seed_chunk(db_session, alice.id, embedder, _SECRET)
    await db_session.commit()

    query = "launch code for Project Zephyr"
    owner = await _kb_tool(
        build_tools(alice.id, db_session, embedder, reranker)
    ).ainvoke({"query": query})
    intruder = await _kb_tool(
        build_tools(bob.id, db_session, embedder, reranker)
    ).ainvoke({"query": query})

    assert "alpha seven seven" in owner  # owner retrieves their own chunk
    assert "alpha seven seven" not in intruder  # never crosses the tenant boundary


def test_kb_search_schema_hides_user_id():
    # The model-visible schema must expose `query` only — user_id is injected.
    from minddrill.agent.tools import KnowledgeBaseSearchInput

    assert list(KnowledgeBaseSearchInput.model_fields) == ["query"]
