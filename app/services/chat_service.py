"""Chat orchestration around the RAG pipeline."""
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

from app.core.config import Settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, DocumentStatus
from app.rag.pipeline import RagPipeline
from app.rag.types import RagAnswer, RetrievedChunk
from app.repositories.chat_repository import ChatRepository
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000

# Persists the assistant message once a streamed answer is complete.
StreamFinalizer = Callable[[str], Awaitable[ChatMessage]]


class ChatService:
    def __init__(
        self,
        chat_repository: ChatRepository,
        document_repository: DocumentRepository,
        pipeline: RagPipeline,
        settings: Settings,
    ) -> None:
        self._chats = chat_repository
        self._documents = document_repository
        self._pipeline = pipeline
        self._settings = settings

    async def create_session(
        self, user_id: uuid.UUID, document_id: uuid.UUID | None, title: str
    ) -> ChatSession:
        if document_id is not None:
            document: Document | None = await self._documents.get(document_id, user_id)
            if document is None:
                raise NotFoundError("Document not found")
            if document.status != DocumentStatus.READY.value:
                # Scoping a session to a document that has no chunks yet would
                # produce confidently empty answers.
                raise ValidationError(
                    f"Document is not ready for questions (status: {document.status})"
                )

        chat_session = ChatSession(
            user_id=user_id, document_id=document_id, title=title[:255]
        )
        await self._chats.add_session(chat_session)
        await self._chats.session.commit()
        return chat_session

    async def get_session(self, session_id: uuid.UUID, user_id: uuid.UUID) -> ChatSession:
        chat_session = await self._chats.get_session(session_id, user_id)
        if chat_session is None:
            # 404 rather than 403 for a session owned by someone else: confirming
            # that an id exists is itself a leak.
            raise NotFoundError("Chat session not found")
        return chat_session

    async def list_sessions(
        self, user_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[ChatSession], int]:
        return (
            await self._chats.list_sessions(user_id, limit, offset),
            await self._chats.count_sessions(user_id),
        )

    async def list_messages(
        self, session_id: uuid.UUID, user_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[ChatMessage], int]:
        await self.get_session(session_id, user_id)
        return (
            await self._chats.list_messages(session_id, limit, offset),
            await self._chats.count_messages(session_id),
        )

    async def delete_session(self, session_id: uuid.UUID, user_id: uuid.UUID) -> None:
        chat_session = await self.get_session(session_id, user_id)
        await self._chats.delete_session(chat_session)
        await self._chats.session.commit()

    def _validate(self, content: str) -> str:
        stripped = content.strip()
        if not stripped:
            raise ValidationError("Message cannot be empty")
        if len(stripped) > MAX_MESSAGE_CHARS:
            raise ValidationError(
                f"Message is too long ({len(stripped)} characters, max {MAX_MESSAGE_CHARS})"
            )
        return stripped

    async def _load_history(self, session_id: uuid.UUID) -> list[dict[str, str]]:
        """Recent turns, oldest-first, excluding the message being answered.

        Called *before* the new user message is persisted. The original code saved
        the question first and then appended it again after loading history, so the
        model received every question twice.
        """
        turns = self._settings.generation.history_turns * 2
        messages = await self._chats.recent_history(session_id, turns)
        return [{"role": m.role, "content": m.content} for m in messages]

    async def send_message(
        self, session_id: uuid.UUID, user_id: uuid.UUID, content: str
    ) -> tuple[ChatMessage, RagAnswer]:
        chat_session = await self.get_session(session_id, user_id)
        question = self._validate(content)

        history = await self._load_history(chat_session.id)

        # Persist and commit the question before the model call. The alternative
        # holds a transaction open for the whole generation, tying up a connection
        # for seconds; this way a failed generation still leaves the question
        # recorded, which is what a user expects to see on reload.
        await self._chats.add_message(
            ChatMessage(session_id=chat_session.id, role="user", content=question)
        )
        await self._chats.session.commit()

        started = time.perf_counter()
        answer = await self._pipeline.answer(
            self._chats.session,
            query=question,
            user_id=user_id,
            document_id=chat_session.document_id,
            history=history,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        message = ChatMessage(
            session_id=chat_session.id,
            role="assistant",
            content=answer.answer,
            sources=[source.to_citation() for source in answer.sources],
            latency_ms=latency_ms,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
        )
        await self._chats.add_message(message)
        await self._chats.session.commit()

        return message, answer

    async def stream_message(
        self, session_id: uuid.UUID, user_id: uuid.UUID, content: str
    ) -> tuple[list[RetrievedChunk], AsyncIterator[str], StreamFinalizer]:
        """Retrieve, then hand back a token stream and a finalizer.

        The assistant message can only be written once the stream is exhausted, so
        persistence is returned as a callable rather than done here.
        """
        chat_session = await self.get_session(session_id, user_id)
        question = self._validate(content)

        history = await self._load_history(chat_session.id)
        await self._chats.add_message(
            ChatMessage(session_id=chat_session.id, role="user", content=question)
        )
        await self._chats.session.commit()

        started = time.perf_counter()
        sources, stream = await self._pipeline.stream_answer(
            self._chats.session,
            query=question,
            user_id=user_id,
            document_id=chat_session.document_id,
            history=history,
        )

        async def finalize(full_answer: str) -> ChatMessage:
            message = ChatMessage(
                session_id=chat_session.id,
                role="assistant",
                content=full_answer,
                sources=[source.to_citation() for source in sources],
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            await self._chats.add_message(message)
            await self._chats.session.commit()
            return message

        return sources, stream, finalize
