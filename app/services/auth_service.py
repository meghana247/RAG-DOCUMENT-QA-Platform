"""Registration, login and token issuing."""
import logging
import uuid
from datetime import datetime

from app.core.config import SecuritySettings
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)

# A bcrypt hash of a throwaway value, used to keep the failed-login path's timing
# indistinguishable from the success path. Without it, "unknown email" returns
# noticeably faster than "wrong password", which leaks which accounts exist.
_DUMMY_HASH = "$2b$12$C6UzMDM.H6dfI/f/IKcEe.rTS.pXxdc3EflIVLGrfsHTfCcYVSDMe"


class AuthService:
    def __init__(self, repository: UserRepository, config: SecuritySettings) -> None:
        self._repository = repository
        self._config = config

    @staticmethod
    def _normalize_email(email: str) -> str:
        # Emails are case-insensitive in practice; storing them lowercase stops
        # "A@b.com" and "a@b.com" becoming two accounts.
        return email.strip().lower()

    async def register(self, email: str, password: str, role: str = "user") -> User:
        normalized = self._normalize_email(email)
        if await self._repository.get_by_email(normalized):
            raise ConflictError("An account with this email already exists")

        user = User(
            email=normalized,
            password_hash=hash_password(password, self._config),
            role=role,
        )
        await self._repository.add(user)
        await self._repository.session.commit()
        logger.info("user registered", extra={"user_id": str(user.id)})
        return user

    async def authenticate(self, email: str, password: str) -> tuple[str, datetime, User]:
        user: User | None = await self._repository.get_by_email(
            self._normalize_email(email)
        )

        if user is None:
            verify_password(password, _DUMMY_HASH)
            raise AuthenticationError("Invalid email or password")

        if not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("This account has been deactivated")

        token, expires_at = create_access_token(user.id, user.role, self._config)
        logger.info("user authenticated", extra={"user_id": str(user.id)})
        return token, expires_at, user

    async def get_active_user(self, user_id: uuid.UUID) -> User:
        user = await self._repository.get_by_id(user_id)
        if user is None or not user.is_active:
            # Covers tokens still inside their validity window for a user who has
            # since been deleted or deactivated.
            raise AuthenticationError("Account no longer active")
        return user
