"""Auth endpoints — register & login (docs/03_API_SPEC.md "Auth")."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from minddrill.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from minddrill.auth.security import (
    ACCESS_TOKEN_EXPIRE_SECONDS,
    BCRYPT_MAX_PASSWORD_BYTES,
    create_access_token,
    hash_password,
    verify_password,
)
from minddrill.db.session import get_session
from minddrill.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest, session: AsyncSession = Depends(get_session)
) -> RegisterResponse:
    if len(body.password.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"password must be at most {BCRYPT_MAX_PASSWORD_BYTES} bytes",
        )

    existing = await session.scalar(select(User).where(User.username == body.username))
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="username already taken")

    user = User(username=body.username, password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="username already taken")
    await session.refresh(user)
    return RegisterResponse(user_id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest, session: AsyncSession = Depends(get_session)
) -> TokenResponse:
    user = await session.scalar(select(User).where(User.username == body.username))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid username or password"
        )

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, expires_in=ACCESS_TOKEN_EXPIRE_SECONDS)
