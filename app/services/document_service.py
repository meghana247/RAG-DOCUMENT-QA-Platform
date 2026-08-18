"""Document ingestion.

Ingestion is split into a fast transactional part (record the upload) and a slow
CPU-bound part (extract, chunk, embed). The router commits the first and schedules
the second, so an upload responds in milliseconds instead of holding the request —
and a database transaction — open for the length of an embedding run.

The scheduling boundary is deliberately a single call, `process_document`, taking
only a document id. Moving from in-process background tasks to a real queue
(Celery, arq, RQ) is then a change of caller, not a change of logic.
"""
import hashlib
import logging
import uuid

from app.core.config import Settings
from app.core.exceptions import (
    DocumentProcessingError,
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.models.document import Document, DocumentStatus
from app.rag.extraction import SUPPORTED_EXTENSIONS
from app.rag.pipeline import RagPipeline
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        pipeline: RagPipeline,
        settings: Settings,
    ) -> None:
        self._repository = repository
        self._pipeline = pipeline
        self._settings = settings

    @staticmethod
    def _validate_extension(filename: str) -> None:
        suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedMediaTypeError(
                f"Unsupported file type '{suffix or filename}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    async def register_upload(
        self, *, user_id: uuid.UUID, filename: str, content_type: str, content: bytes
    ) -> tuple[Document, bool]:
        """Record an upload and return (document, is_new).

        Returns the existing row when the same bytes have already been ingested,
        so a duplicate upload costs nothing rather than re-embedding the file.
        """
        self._validate_extension(filename)

        if len(content) > self._settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"File is {len(content) // 1024}KB; the limit is "
                f"{self._settings.max_upload_bytes // 1024}KB"
            )
        if not content:
            raise DocumentProcessingError("The uploaded file is empty")

        content_hash = hashlib.sha256(content).hexdigest()

        if existing := await self._repository.get_by_hash(user_id, content_hash):
            logger.info(
                "duplicate upload ignored",
                extra={"document_id": str(existing.id), "status": existing.status},
            )
            if existing.status == DocumentStatus.FAILED.value:
                # Let a previously failed ingest be retried.
                existing.status = DocumentStatus.PENDING.value
                existing.error_message = None
                await self._repository.session.commit()
                return existing, True
            return existing, False

        document = Document(
            user_id=user_id,
            filename=filename[:255],
            content_type=content_type or "application/octet-stream",
            size_bytes=len(content),
            content_hash=content_hash,
            status=DocumentStatus.PENDING.value,
        )
        await self._repository.add(document)
        await self._repository.session.commit()
        return document, True

    async def process_document(self, document_id: uuid.UUID, content: bytes) -> Document:
        """Extract, chunk, embed and store. Safe to retry.

        Runs outside the request's transaction. Failures are recorded on the row
        rather than raised into the void, so a client polling status sees *why*
        ingestion failed instead of a document stuck on "pending" forever.
        """
        document = await self._repository.get(document_id)
        if document is None:
            raise NotFoundError("Document not found")

        document.status = DocumentStatus.PROCESSING.value
        await self._repository.session.commit()

        try:
            chunks, page_count = self._pipeline.chunk_document(
                content, document.filename, document.content_type
            )
            if not chunks:
                raise DocumentProcessingError("The document produced no usable text")

            embeddings = await self._pipeline.embed_chunks(chunks)

            # Idempotency: clear any chunks from a previous attempt before writing.
            await self._repository.delete_chunks(document.id)
            await self._repository.bulk_insert_chunks(
                [
                    {
                        "id": uuid.uuid4(),
                        "document_id": document.id,
                        "user_id": document.user_id,
                        "content": chunk.content,
                        "chunk_index": chunk.index,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "char_count": chunk.char_count,
                        "embedding": embedding,
                    }
                    for chunk, embedding in zip(chunks, embeddings, strict=False)
                ]
            )

            document.total_chunks = len(chunks)
            document.page_count = page_count
            document.status = DocumentStatus.READY.value
            document.error_message = None
            await self._repository.session.commit()

            logger.info(
                "document ingested",
                extra={
                    "document_id": str(document.id),
                    "chunks": len(chunks),
                    "pages": page_count,
                    "bytes": document.size_bytes,
                },
            )
            return document

        except Exception as exc:
            await self._repository.session.rollback()
            document = await self._repository.get(document_id)
            if document is not None:
                document.status = DocumentStatus.FAILED.value
                # Bounded: the column is Text, but an enormous driver traceback
                # in an API response is noise.
                document.error_message = str(exc)[:1000]
                await self._repository.session.commit()
            logger.error(
                "document ingestion failed",
                exc_info=exc,
                extra={"document_id": str(document_id)},
            )
            raise

    async def get(self, document_id: uuid.UUID, user_id: uuid.UUID) -> Document:
        document = await self._repository.get(document_id, user_id)
        if document is None:
            raise NotFoundError("Document not found")
        return document

    async def list_for_user(
        self, user_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        return (
            await self._repository.list_for_user(user_id, limit, offset),
            await self._repository.count_for_user(user_id),
        )

    async def delete(
        self, document_id: uuid.UUID, user_id: uuid.UUID, is_admin: bool = False
    ) -> None:
        document = await self._repository.get(
            document_id, None if is_admin else user_id
        )
        if document is None:
            raise NotFoundError("Document not found")
        await self._repository.remove(document)
        await self._repository.session.commit()
        logger.info("document deleted", extra={"document_id": str(document_id)})
