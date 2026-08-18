"""FastAPI dependency providers.

Routers depend on services, never on repositories or sessions directly. That
keeps HTTP concerns out of business logic and makes a router testable by
overriding one dependency.
"""
# No `from __future__ import annotations` here: PEP 563 turns the annotations on
# the class-based `Pagination` dependency into strings that FastAPI cannot
# resolve back to `Query(...)`, which fails at app construction.
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.container import Container, get_container
from app.core.database import get_db
from app.core.exceptions import AuthorizationError
from app.core.logging import user_id_var
from app.core.security import subject_from_token
from app.models.user import User
from app.repositories.chat_repository import ChatRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

# auto_error=False so a missing header surfaces as our own 401 envelope rather
# than FastAPI's default shape, keeping every error response identical.
_bearer = HTTPBearer(auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db)]
AppSettings = Annotated[Settings, Depends(get_settings)]


def container_dep() -> Container:
    return get_container()


AppContainer = Annotated[Container, Depends(container_dep)]


# --------------------------------------------------------------------- services


def auth_service(session: DbSession, settings: AppSettings) -> AuthService:
    return AuthService(UserRepository(session), settings.security)


def document_service(
    session: DbSession, container: AppContainer, settings: AppSettings
) -> DocumentService:
    return DocumentService(DocumentRepository(session), container.pipeline, settings)


def chat_service(
    session: DbSession, container: AppContainer, settings: AppSettings
) -> ChatService:
    return ChatService(
        ChatRepository(session),
        DocumentRepository(session),
        container.pipeline,
        settings,
    )


AuthServiceDep = Annotated[AuthService, Depends(auth_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(document_service)]
ChatServiceDep = Annotated[ChatService, Depends(chat_service)]


# ------------------------------------------------------------------------ auth


async def current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    service: AuthServiceDep,
    settings: AppSettings,
) -> User:
    from app.core.exceptions import AuthenticationError

    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Missing bearer token")

    user_id = subject_from_token(credentials.credentials, settings.security)
    user = await service.get_active_user(user_id)

    # Every subsequent log line in this request carries the user id.
    user_id_var.set(str(user.id))
    return user


CurrentUser = Annotated[User, Depends(current_user)]


async def admin_user(user: CurrentUser) -> User:
    if not user.is_admin:
        raise AuthorizationError("This endpoint requires an administrator account")
    return user


AdminUser = Annotated[User, Depends(admin_user)]


# ------------------------------------------------------------------ pagination


class Pagination:
    """Offset pagination with a hard ceiling.

    The original list endpoints returned every row. One user with 10,000 chunks
    of chat history would have produced a response large enough to time out.
    """

    def __init__(
        self,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> None:
        self.limit = limit
        self.offset = offset


PaginationDep = Annotated[Pagination, Depends(Pagination)]


# ----------------------------------------------------------- background session


async def background_session() -> AsyncIterator[AsyncSession]:
    """A session independent of any request.

    Background ingestion outlives the request that scheduled it, so it cannot
    borrow the request's session — that session is closed as soon as the response
    is sent.
    """
    from app.core.database import SessionFactory

    async with SessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
