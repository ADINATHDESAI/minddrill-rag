"""Agent tools: a registry of name + Pydantic input schema + function.

Tool inputs are the machine-consumed path, so they are validated with Pydantic
(the `args_schema` on each tool). `knowledge_base_search`'s model-visible schema
is `query` only: `user_id` is never a tool parameter but a server-side closure
value threaded into both retrieval arms, so the model can neither read nor forge
it.
"""

import ast
import operator
import uuid
from collections.abc import Callable

import httpx
import structlog
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.config import get_settings
from minddrill.rag.embedder import Embedder
from minddrill.rag.reranker import Reranker
from minddrill.rag.retrieve import Retriever

log = structlog.get_logger(__name__)

_KB_CANDIDATES = 20  # fused candidates handed to the reranker
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_TIMEOUT = 8.0


# --- calculator -------------------------------------------------------------

_BIN_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
# Ceilings on exponentiation: `2 ** 2 ** 30` would otherwise pin a CPU thread and
# exhaust memory building a giant integer, a trivial denial-of-service input.
_MAX_EXPONENT = 100
_MAX_POW_BASE = 1e6


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if type(node.op) is ast.Pow and (
            abs(right) > _MAX_EXPONENT or abs(left) > _MAX_POW_BASE
        ):
            raise ValueError("exponent or base too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only arithmetic over numbers is allowed")


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression with a whitelisted AST walk (no eval)."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '2 * (3 + 4)'")


def _calculator(expression: str) -> str:
    try:
        return str(calculate(expression))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
        return f"error: {exc}"


calculator_tool = StructuredTool.from_function(
    func=_calculator,
    name="calculator",
    description="Evaluate a basic arithmetic expression. Numbers and + - * / // % ** only.",
    args_schema=CalculatorInput,
)


# --- weather ----------------------------------------------------------------


class WeatherInput(BaseModel):
    latitude: float
    longitude: float


async def _fetch_weather(latitude: float, longitude: float) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
    }
    async with httpx.AsyncClient(timeout=_WEATHER_TIMEOUT) as client:
        resp = await client.get(_WEATHER_URL, params=params)
        resp.raise_for_status()
        return resp.json().get("current_weather", {})


async def _weather(latitude: float, longitude: float) -> str:
    try:
        current = await _fetch_weather(latitude, longitude)
    except (httpx.HTTPError, ValueError) as exc:
        return f"error: weather lookup failed ({exc})"
    if not current:
        return "error: no current weather available for that location"
    return f"{current.get('temperature')}°C, wind {current.get('windspeed')} km/h"


weather_tool = StructuredTool.from_function(
    coroutine=_weather,
    name="weather",
    description="Get current weather for a latitude/longitude via Open-Meteo.",
    args_schema=WeatherInput,
)


# --- knowledge base search (per-user, isolation-critical) -------------------


class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(description="What to look up in the user's documents.")


def _knowledge_base_search(
    user_id: uuid.UUID,
    session: AsyncSession,
    embedder: Embedder,
    reranker: Reranker,
    sources_sink: list[dict] | None,
) -> Callable:
    """Return a KB-search coroutine bound to one user's isolation boundary.

    The `[i]` markers in the returned text reference the source rows pushed onto
    `sources_sink`, which the caller emits as an SSE `sources` event so the
    citations resolve.
    """

    async def search(query: str) -> str:
        query_vec = await embedder.embed_query(query)
        # Both arms of hybrid_search filter WHERE user_id = :user_id — the real
        # multi-tenant boundary. The injected user_id is never model-visible.
        chunks = await Retriever(session).hybrid_search(
            query, query_vec, user_id, _KB_CANDIDATES
        )
        if not chunks:
            return "No matching documents found in your knowledge base."

        settings = get_settings()
        scored = await reranker.rerank(query, chunks, settings.rerank_top_n)
        if settings.rerank_enabled and (
            not scored or scored[0][1] < settings.grounding_min_score
        ):
            return "No sufficiently relevant documents found in your knowledge base."

        log.info("agent.kb_search", user_id=str(user_id), hits=len(scored))
        if sources_sink is not None:
            sources_sink.extend(
                {
                    "id": i,
                    "chunk_id": str(c.id),
                    "document_id": str(c.document_id),
                    "score": score,
                }
                for i, (c, score) in enumerate(scored, start=1)
            )
        return "\n\n".join(
            f"[{i}] {c.content}" for i, (c, _) in enumerate(scored, start=1)
        )

    return search


def build_tools(
    user_id: uuid.UUID,
    session: AsyncSession,
    embedder: Embedder,
    reranker: Reranker,
    sources_sink: list[dict] | None = None,
) -> list[BaseTool]:
    """Assemble the agent's tools, binding user-scoped tools to this request.

    `knowledge_base_search` pushes the sources behind its citation markers onto
    `sources_sink` when provided.
    """
    kb_tool = StructuredTool.from_function(
        coroutine=_knowledge_base_search(
            user_id, session, embedder, reranker, sources_sink
        ),
        name="knowledge_base_search",
        description="Search the user's own ingested documents for relevant passages.",
        args_schema=KnowledgeBaseSearchInput,
    )
    return [calculator_tool, weather_tool, kb_tool]
