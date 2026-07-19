"""FastAPI application entrypoint.

Routers for auth, ingestion, chat, etc. mount onto `create_app()` per slice.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from minddrill.auth.deps import get_current_user
from minddrill.auth.router import router as auth_router
from minddrill.config import get_settings
from minddrill.logging import RequestIDMiddleware, configure_logging
from minddrill.models.user import User
from minddrill.observability import get_langfuse
from minddrill.rag.reranker import warm_reranker
from minddrill.rag.router import router as rag_router
from minddrill.sessions.router import router as sessions_router

# Maps HTTPException.status_code -> the API spec's error `code` string.
_ERROR_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    429: "rate_limited",
    500: "internal_error",
    503: "providers_unavailable",
}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Load the cross-encoder once at startup so the first query doesn't eat the
    # model-load latency. Off the loop — construction is blocking.
    await asyncio.to_thread(warm_reranker)
    yield
    # Traces are batched client-side; flush so the last requests' traces are
    # sent rather than dropped on process exit.
    get_langfuse().flush()


def create_app() -> FastAPI:
    configure_logging(get_settings().log_level)

    app = FastAPI(title="MindDrill", lifespan=_lifespan)
    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        code = _ERROR_CODES.get(exc.status_code, "internal_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Exists solely to exercise `get_current_user` end to end; superseded by real
    # protected endpoints in later slices.
    @app.get("/api/v1/_whoami")
    async def whoami(current_user: User = Depends(get_current_user)) -> dict[str, str]:
        return {"user_id": str(current_user.id), "username": current_user.username}

    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(rag_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")

    return app


app = create_app()
