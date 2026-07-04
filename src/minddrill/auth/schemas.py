"""Request/response bodies for the auth endpoints (docs/03_API_SPEC.md)."""

from uuid import UUID

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    username: str
    password: str


class RegisterResponse(BaseModel):
    user_id: UUID
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
