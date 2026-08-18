"""Password hashing and JWT issuing/verification.

Kept free of database and HTTP concerns so it can be unit-tested directly.
"""
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.core.config import SecuritySettings
from app.core.exceptions import AuthenticationError

# bcrypt truncates silently at 72 bytes; callers validate before reaching here,
# but the constant is defined once so both layers agree.
BCRYPT_MAX_BYTES = 72


def hash_password(password: str, cfg: SecuritySettings) -> str:
    if len(password.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=cfg.bcrypt_rounds)
    ).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        # Malformed hash in the row — treat as a failed login, not a 500.
        return False


def create_access_token(
    subject: uuid.UUID, role: str, cfg: SecuritySettings
) -> tuple[str, datetime]:
    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=cfg.access_token_expire_minutes)
    claims: dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "iat": now,
        "exp": expires_at,
        "jti": str(uuid.uuid4()),
    }
    token = jwt.encode(
        claims, cfg.secret_key.get_secret_value(), algorithm=cfg.algorithm
    )
    return token, expires_at


def decode_access_token(token: str, cfg: SecuritySettings) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            cfg.secret_key.get_secret_value(),
            algorithms=[cfg.algorithm],
        )
    except JWTError as exc:
        raise AuthenticationError("Invalid or expired token") from exc


def subject_from_token(token: str, cfg: SecuritySettings) -> uuid.UUID:
    payload = decode_access_token(token, cfg)
    raw: str | None = payload.get("sub")
    if not raw:
        raise AuthenticationError("Token is missing a subject claim")
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        # A syntactically valid token carrying a non-UUID subject would otherwise
        # raise deep inside the repository as an unhandled 500.
        raise AuthenticationError("Token subject is not a valid identifier") from exc
