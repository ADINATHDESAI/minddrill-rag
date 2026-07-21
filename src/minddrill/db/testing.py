"""Test-database resolution and schema bootstrap, shared by pytest and eval/run_eval.py.

Both entry points must run against `<db>_test`, never the database `DATABASE_URL`
points the running app at. `resolve_test_database_url` derives that name and
refuses to return anything that doesn't end in `_test`; callers must set
`DATABASE_URL` to its result (and clear `get_settings`'s cache) before importing
anything that constructs an engine.
"""

from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import text

from minddrill.config import get_settings


def resolve_test_database_url() -> str:
    """Derive the test DB URL from DATABASE_URL by suffixing the db name with `_test`.

    Never used verbatim: this exists so tests (and the offline eval) can't be
    pointed at a real database by accident. Aborts loudly if the resolved name
    doesn't end in `_test`.
    """
    base_url = get_settings().database_url
    parts = urlsplit(base_url)
    db_name = parts.path.lstrip("/")
    if not db_name:
        raise RuntimeError(
            f"DATABASE_URL has no database name to derive a test database from: {base_url!r}"
        )
    test_name = f"{db_name}_test"
    test_url = urlunsplit(
        (parts.scheme, parts.netloc, f"/{test_name}", parts.query, parts.fragment)
    )

    resolved_name = urlsplit(test_url).path.lstrip("/")
    if not resolved_name.endswith("_test"):
        raise RuntimeError(
            f"Refusing to run against database {resolved_name!r}: it does not end "
            "in '_test'. Tests and the offline eval must never run against a "
            "non-test database."
        )
    return test_url


async def init_test_schema() -> None:
    """Create the vector/pg_search extensions, all tables, and the BM25 index.

    Must run against an engine already bound to the resolved `_test` URL. The
    BM25 index is a `USING bm25` index `create_all` can't emit, so it's created
    here to mirror the migration.
    """
    # Imported lazily: these modules construct their engine/tables at import
    # time from the *current* DATABASE_URL, so callers must set that env var to
    # the resolved test URL before this function (and these imports) run.
    from minddrill.db.session import Base, engine
    from minddrill.models import chunk as _chunk  # noqa: F401  registers Chunk
    from minddrill.models import document as _document  # noqa: F401  registers Document
    from minddrill.models import (  # noqa: F401  registers IngestionJob
        ingestion_job as _ingestion_job,
    )
    from minddrill.models import message as _message  # noqa: F401  registers Message
    from minddrill.models import (  # noqa: F401  registers ChatSession
        session as _session_model,
    )
    from minddrill.models import user as _user  # noqa: F401  registers User

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_search"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks "
                "USING bm25 (id, content, user_id) WITH (key_field='id')"
            )
        )
    # Callers may run this in a throwaway event loop (asyncio.run at import
    # time); dispose the pool so no connection is reused across loops.
    await engine.dispose()
