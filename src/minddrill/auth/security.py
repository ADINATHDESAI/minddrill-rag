"""Password hashing and JWT encode/decode (specs/slice-1-auth.md).

HS256, secret from settings, payload `{sub: user_id, exp}`, 1 hour expiry.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

import jwt
from passlib.context import CryptContext

from minddrill.config import get_settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_SECONDS = 3600
BCRYPT_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def create_access_token(user_id: UUID) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "exp": now + timedelta(seconds=ACCESS_TOKEN_EXPIRE_SECONDS),
    }
    return jwt.encode(payload, get_settings().jwt_secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT. Raises `jwt.PyJWTError` on expired/tampered/malformed tokens."""
    return jwt.decode(token, get_settings().jwt_secret, algorithms=[JWT_ALGORITHM])
