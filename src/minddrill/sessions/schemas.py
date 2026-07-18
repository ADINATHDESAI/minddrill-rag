"""Request/response bodies for the session and chat endpoints (docs/03_API_SPEC.md)."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    title: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: UUID
    created_at: datetime


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class MessagesResponse(BaseModel):
    session_id: UUID
    messages: list[MessageOut]


class ChatRequest(BaseModel):
    message: str
    session_id: UUID
