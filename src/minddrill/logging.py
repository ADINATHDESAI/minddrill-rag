"""Structured JSON logging.

`configure_logging` sets up structlog to render JSON via stdlib logging, so any
`logging.getLogger` call elsewhere (e.g. from libraries) also comes out as JSON.
`RequestIDMiddleware` generates a per-request id, binds it to structlog's
contextvars (so every log line emitted while handling the request carries it),
and echoes it back as the `X-Request-ID` header for client-side correlation.
"""

import logging
import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


def configure_logging(log_level: str = "INFO") -> None:
    """Configure stdlib logging + structlog to emit one JSON object per line."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "minddrill") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Binds a request id to structlog context and logs at the request seam."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        # Also stashed on request.state: BaseHTTPMiddleware's contextvars don't
        # reliably reach code running inside an SSE generator's own task, so
        # streaming endpoints read the id from state rather than structlog.
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        logger = get_logger("minddrill.request")
        start = time.perf_counter()
        logger.info("request_start", method=request.method, path=request.url.path)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "request_error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
