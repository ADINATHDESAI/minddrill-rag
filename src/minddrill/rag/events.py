"""Typed SSE event builders.

The transport (sse-starlette) expects `{"event": name, "data": str}` dicts. These
helpers are the only place that shape is constructed, so a future WebSocket swap
touches this module and nothing in the query logic.
"""

import json
from typing import Any


def _event(name: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": name, "data": json.dumps(data)}


def status(state: str) -> dict[str, str]:
    return _event("status", {"state": state})


def sources(items: list[dict[str, Any]]) -> dict[str, str]:
    return _event("sources", {"sources": items})


def token(text: str) -> dict[str, str]:
    return _event("token", {"text": text})


def tool_call(tool: str, args: dict[str, Any]) -> dict[str, str]:
    return _event("tool_call", {"tool": tool, "args": args})


def tool_result(tool: str, result: Any) -> dict[str, str]:
    return _event("tool_result", {"tool": tool, "result": result})


def decline(reason: str) -> dict[str, str]:
    return _event("decline", {"reason": reason})


def done(
    usage: dict[str, int], ttft_ms: int, latency_ms: int, grounded: bool
) -> dict[str, str]:
    return _event(
        "done",
        {
            "usage": usage,
            "ttft_ms": ttft_ms,
            "latency_ms": latency_ms,
            "grounded": grounded,
        },
    )


def error(code: str, message: str) -> dict[str, str]:
    return _event("error", {"code": code, "message": message})
