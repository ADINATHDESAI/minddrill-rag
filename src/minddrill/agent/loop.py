"""Streaming /chat through the agent loop.

`create_agent` (LangChain on LangGraph) drives the tool loop; the linear RAG path
stays hand-written in `rag/retrieve.py`. Loop-level guardrails are middleware:
`ModelCallLimitMiddleware` caps the steps (ends gracefully with the best answer so
far) and `ModelFallbackMiddleware` gives Gemini → OpenRouter failover.

The LangGraph event stream is mapped onto the typed SSE protocol: assistant
tokens → `token`, an AIMessage tool call → `tool_call`, a returned ToolMessage →
`tool_result`, and a knowledge-base hit → `sources` (so citation markers
resolve). The user turn is persisted before generating; the assistant turn only
after a clean completion, on a fresh session decoupled from request teardown.
"""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.agent.tools import build_tools
from minddrill.config import get_settings
from minddrill.db.session import SessionLocal
from minddrill.models.message import Message
from minddrill.models.session import ChatSession
from minddrill.rag import events
from minddrill.rag.embedder import Embedder
from minddrill.rag.reranker import Reranker
from minddrill.sessions.memory import estimate_tokens, load_history, trim_history

log = structlog.get_logger(__name__)

_AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use a tool when it helps: search the user's "
    "knowledge base for questions about their documents, do arithmetic with the "
    "calculator, or look up weather. Otherwise answer directly and concisely."
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return "" if content is None else str(content)


async def run_agent_chat(
    session: AsyncSession,
    chat_session: ChatSession,
    message: str,
    *,
    model: BaseChatModel,
    fallback_model: BaseChatModel | None,
    embedder: Embedder,
    reranker: Reranker,
) -> AsyncIterator[dict]:
    """Persist the user turn, build the agent, return its SSE event stream."""
    start = time.perf_counter()

    session.add(
        Message(
            session_id=chat_session.id,
            role="user",
            content=message,
            token_count=estimate_tokens(message),
        )
    )
    chat_session.last_active_at = func.now()
    await session.commit()

    settings = get_settings()
    history = await load_history(session, chat_session.id)
    trimmed = trim_history(history, settings.memory_token_budget)
    input_messages = [{"role": m.role, "content": m.content} for m in trimmed]

    sources_sink: list[dict] = []
    tools = build_tools(chat_session.user_id, session, embedder, reranker, sources_sink)
    middleware: list[Any] = [
        ModelCallLimitMiddleware(
            run_limit=settings.agent_max_steps, exit_behavior="end"
        )
    ]
    if fallback_model is not None:
        middleware.insert(0, ModelFallbackMiddleware(fallback_model))

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=_AGENT_SYSTEM_PROMPT,
        middleware=middleware,
    )
    return _stream_agent(chat_session.id, start, agent, input_messages, sources_sink)


async def _stream_agent(
    session_id: uuid.UUID,
    start: float,
    agent: Any,
    input_messages: list[dict],
    sources_sink: list[dict],
) -> AsyncIterator[dict]:
    yield events.status("generating")
    parts: list[str] = []
    ttft_ms: int | None = None
    usage: dict[str, int] | None = None
    grounded = False

    stream = agent.astream(
        {"messages": input_messages}, stream_mode=["updates", "messages"]
    )
    try:
        async for mode, data in stream:
            if mode == "messages":
                msg, _meta = data
                if not isinstance(msg, (AIMessage, AIMessageChunk)):
                    continue
                text = _content_text(msg.content)
                if text:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - start) * 1000)
                    parts.append(text)
                    yield events.token(text)
                meta = getattr(msg, "usage_metadata", None)
                if meta:
                    usage = usage or {"input_tokens": 0, "output_tokens": 0}
                    usage["input_tokens"] += meta.get("input_tokens", 0)
                    usage["output_tokens"] += meta.get("output_tokens", 0)
            else:  # "updates": node outputs carry tool calls and tool returns
                for update in (data or {}).values():
                    if not isinstance(update, dict):
                        continue
                    for m in update.get("messages", []):
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for call in m.tool_calls:
                                yield events.tool_call(
                                    call["name"], call.get("args", {})
                                )
                        elif isinstance(m, ToolMessage):
                            yield events.tool_result(m.name, _content_text(m.content))
                            # A KB hit filled the sink: emit the sources its
                            # citation markers point at, then reset for any
                            # later search in the same run.
                            if sources_sink:
                                grounded = True
                                yield events.sources(list(sources_sink))
                                sources_sink.clear()
    except Exception as exc:  # failure after the stream opened
        log.warning("agent.stream_error", session_id=str(session_id), error=str(exc))
        yield events.error("internal_error", "generation failed mid-stream")
        return
    finally:
        await stream.aclose()

    text = "".join(parts)
    if text:
        try:
            async with SessionLocal() as write:
                write.add(
                    Message(
                        session_id=session_id,
                        role="assistant",
                        content=text,
                        token_count=estimate_tokens(text),
                    )
                )
                await write.commit()
        except Exception as exc:
            log.warning(
                "agent.persist_error", session_id=str(session_id), error=str(exc)
            )
            yield events.error("internal_error", "failed to persist reply")
            return

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info(
        "agent.answered",
        session_id=str(session_id),
        ttft_ms=ttft_ms or 0,
        latency_ms=latency_ms,
    )
    yield events.done(
        usage or {"input_tokens": 0, "output_tokens": 0},
        ttft_ms or 0,
        latency_ms,
        grounded=grounded,
    )
