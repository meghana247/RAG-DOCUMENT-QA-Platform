import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.document import Document, DocumentChunk
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository):
    async def get(
        self, document_id: uuid.UUID, user_id: uuid.UUID | None = None
    ) -> Document | None:
        """Fetch a document, scoped to an owner unless `user_id` is omitted.

        Callers pass `user_id=None` only for admin paths; every user-facing path
        must scope, or one tenant can read another's documents by guessing an id.
        """
        query = select(Document).where(Document.id == document_id)
        if user_id is not None:
            query = query.where(Document.user_id == user_id)
        return (await self.session.execute(query)).scalar_one_or_none()

    async def get_by_hash(self, user_id: uuid.UUID, content_hash: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(
                Document.user_id == user_id, Document.content_hash == content_hash
            )
        )
        return result.scalar_one_or_none()

    async def add(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def list_for_user(
        self, user_id: uuid.UUID, limit: int, offset: int
    ) -> list[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_for_user(self, user_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count(Document.id)).where(Document.user_id == user_id)
        )
        return result.scalar_one()

    async def list_all(self, limit: int, offset: int) -> list[Document]:
        result = await self.session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def count_all(self) -> int:
        return (await self.session.execute(select(func.count(Document.id)))).scalar_one()

    async def remove(self, document: Document) -> None:
        # Chunks go with it via ON DELETE CASCADE.
        await self.session.delete(document)

    async def bulk_insert_chunks(self, rows: list[dict]) -> None:
        """Insert chunk rows in one statement.

        A per-row `session.add()` loop issues one INSERT per chunk; a 300-page PDF
        becomes ~1500 round trips. This is a single multi-values INSERT.

        `on_conflict_do_nothing` makes a retried ingest idempotent instead of
        failing on the (document_id, chunk_index) unique constraint.
        """
        if not rows:
            return
        statement = pg_insert(DocumentChunk).values(rows)
        await self.session.execute(
            statement.on_conflict_do_nothing(constraint="uq_chunk_document_index")
        )

    async def delete_chunks(self, document_id: uuid.UUID) -> None:
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )

    async def count_chunks_for_user(self, user_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.user_id == user_id)
        )
        return result.scalar_one()
