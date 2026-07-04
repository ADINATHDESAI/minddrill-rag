"""FastAPI application entrypoint.

Slice 0 is skeleton only: a single liveness route. Routers for auth, ingestion,
chat, etc. mount onto `create_app()` in later slices.
"""

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="MindDrill")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
