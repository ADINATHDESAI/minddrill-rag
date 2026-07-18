"""Session and chat endpoints.

Sessions are conversations owned by the caller. `/chat` streams a reply grounded
in the session's recent turns (short-term memory) — it does not retrieve
documents. Every read is scoped to `current_user`; another user's session is
indistinguishable from a missing one (404).
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from minddrill.auth.deps import get_current_user
from minddrill.db.session import get_session
from minddrill.models.message import Message
from minddrill.models.session import ChatSession
from minddrill.models.user import User
from minddrill.providers.base import LLMProvider
from minddrill.providers.failover import ProvidersUnavailable, get_providers
from minddrill.sessions.chat import run_chat
from minddrill.sessions.schemas import (
    ChatRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    MessageOut,
    MessagesResponse,
)

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

router = APIRouter(tags=["sessions"])


async def _owned_session(
    session_id: UUID, current_user: User, session: AsyncSession
) -> ChatSession:
    chat_session = await session.scalar(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == current_user.id
        )
    )
    if chat_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="session not found"
        )
    return chat_session


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    body: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> CreateSessionResponse:
    chat_session = ChatSession(user_id=current_user.id, title=body.title)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return CreateSessionResponse(
        session_id=chat_session.id, created_at=chat_session.created_at
    )


@router.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
async def get_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> MessagesResponse:
    await _owned_session(session_id, current_user, session)
    rows = await session.scalars(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at, Message.id)
    )
    return MessagesResponse(
        session_id=session_id,
        messages=[
            MessageOut(role=m.role, content=m.content, created_at=m.created_at)
            for m in rows
        ],
    )


@router.post("/chat")
async def chat(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    providers: list[LLMProvider] = Depends(get_providers),
) -> EventSourceResponse:
    chat_session = await _owned_session(body.session_id, current_user, session)
    # Memory and provider failover resolve here, before the stream opens: an
    # all-providers-down failure returns a plain 503, never a half-open stream.
    try:
        generator = await run_chat(session, chat_session, body.message, providers)
    except ProvidersUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="all inference providers failed",
        )
    return EventSourceResponse(generator, headers=_SSE_HEADERS)
