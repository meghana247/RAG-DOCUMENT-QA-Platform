import uuid

from sqlalchemy import func, select

from app.models.chat import ChatMessage, ChatSession
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository):
    async def get_session(
        self, session_id: uuid.UUID, user_id: uuid.UUID
    ) -> ChatSession | None:
        result = await self.session.execute(
            select(ChatSession).where(
                ChatSession.id == session_id, ChatSession.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def add_session(self, chat_session: ChatSession) -> ChatSession:
        self.session.add(chat_session)
        await self.session.flush()
        return chat_session

    async def list_sessions(
        self, user_id: uuid.UUID, limit: int, offset: int
    ) -> list[ChatSession]:
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_sessions(self, user_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count(ChatSession.id)).where(ChatSession.user_id == user_id)
        )
        return result.scalar_one()

    async def delete_session(self, chat_session: ChatSession) -> None:
        await self.session.delete(chat_session)

    async def add_message(self, message: ChatMessage) -> ChatMessage:
        self.session.add(message)
        await self.session.flush()
        return message

    async def list_messages(
        self, session_id: uuid.UUID, limit: int, offset: int
    ) -> list[ChatMessage]:
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_messages(self, session_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count(ChatMessage.id)).where(ChatMessage.session_id == session_id)
        )
        return result.scalar_one()

    async def recent_history(self, session_id: uuid.UUID, limit: int) -> list[ChatMessage]:
        """The most recent `limit` messages, returned oldest-first.

        The original code paired `ORDER BY created_at ASC` with `LIMIT 10`, which
        returns the *first* ten messages ever sent — so after ten messages the
        model stopped seeing recent context entirely. Ordering descending selects
        the recent window; the reversal restores chronological order for the prompt.
        """
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))
