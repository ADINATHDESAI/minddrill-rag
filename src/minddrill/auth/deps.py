"""Auth dependencies.

Every data endpoint depends on `get_current_user` so reads can be scoped to the
caller (per-user isolation, CLAUDE.md). Slice 0 ships the stub only — real JWT
decode/verify lands in slice 1.
"""


async def get_current_user() -> str:
    """Resolve the authenticated user from the request. STUB (slice 1)."""
    raise NotImplementedError("Authentication is implemented in slice 1")
