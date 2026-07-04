# Slice 1 — Auth (username/password -> JWT)

Tables:
- users(id uuid pk, username text unique, password_hash text, created_at)
  via an alembic migration.

Endpoints (see docs/03_API_SPEC.md "Auth"):
- POST /auth/register {username, password} -> 201 {user_id, username}.
  409 if username taken.
- POST /auth/login {username, password} -> 200 {access_token, token_type,
  expires_in}. 401 on bad credentials.

Security:
- Hash passwords with passlib bcrypt. Never store plaintext.
- JWT: HS256, secret from settings, payload {sub: user_id, exp}. 1 hour expiry.

Dependency:
- get_current_user: read Authorization: Bearer <jwt>, decode, load user,
  return it. 401 on missing/invalid/expired. Replaces the slice-0 stub.

Tests:
- register creates a user; duplicate username -> 409.
- login with right password -> token; wrong password -> 401.
- a protected test route: no token -> 401; valid token -> 200 and correct user.

What can go wrong: expired token, tampered token, unknown user in token,
missing Authorization header.