import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, status

from app.core.config import Settings
from app.core.database import SessionFactory
from app.core.dependencies import (
    AppSettings,
    CurrentUser,
    DocumentServiceDep,
    PaginationDep,
)
from app.core.exceptions import PayloadTooLargeError
from app.core.logging import request_id_var
from app.repositories.document_repository import DocumentRepository
from app.schemas.common import Page
from app.schemas.document import DocumentResponse, UploadAccepted

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_READ_CHUNK = 1024 * 1024  # 1MB


async def _read_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, aborting once it exceeds the limit.

    `await file.read()` with no argument buffers the entire body first and only
    then allows a size check — by which point a hostile or mistaken 2GB upload has
    already been paid for in memory.
    """
    buffer = bytearray()
    while segment := await file.read(_READ_CHUNK):
        buffer.extend(segment)
        if len(buffer) > max_bytes:
            raise PayloadTooLargeError(
                f"File exceeds the {max_bytes // (1024 * 1024)}MB limit"
            )
    return bytes(buffer)


async def _ingest_in_background(
    document_id: uuid.UUID, content: bytes, settings: Settings, request_id: str
) -> None:
    """Run ingestion on its own session, outside the request lifecycle.

    In-process background tasks do not survive a restart or a crash: a document
    left in `processing` should be re-uploaded, and the duplicate-hash path makes
    that cheap. A durable queue (Celery/arq) is the next step if that matters —
    only this function changes.
    """
    from app.core.container import get_container
    from app.services.document_service import DocumentService

    request_id_var.set(request_id)
    container = get_container()

    async with SessionFactory() as session:
        service = DocumentService(DocumentRepository(session), container.pipeline, settings)
        try:
            await service.process_document(document_id, content)
        except Exception:
            # Already logged and recorded on the row by the service; swallowed here
            # so the task does not surface as an unhandled exception in the loop.
            logger.warning(
                "background ingestion did not complete",
                extra={"document_id": str(document_id)},
            )


@router.post(
    "/upload", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED
)
async def upload_document(
    background: BackgroundTasks,
    service: DocumentServiceDep,
    settings: AppSettings,
    user: CurrentUser,
    file: UploadFile = File(...),
) -> UploadAccepted:
    """Accept a document for ingestion.

    Returns 202: extraction, chunking and embedding happen after the response.
    Poll `GET /documents/{id}` until `status` is `ready`.
    """
    content = await _read_bounded(file, settings.max_upload_bytes)

    document, is_new = await service.register_upload(
        user_id=user.id,
        filename=file.filename or "untitled",
        content_type=file.content_type or "",
        content=content,
    )

    if is_new:
        background.add_task(
            _ingest_in_background,
            document.id,
            content,
            settings,
            request_id_var.get() or "",
        )
        message = "Upload accepted. Poll the document until its status is 'ready'."
    else:
        message = "This file has already been ingested."

    return UploadAccepted(
        document=DocumentResponse.model_validate(document),
        duplicate=not is_new,
        message=message,
    )


@router.get("", response_model=Page[DocumentResponse])
async def list_documents(
    service: DocumentServiceDep, user: CurrentUser, page: PaginationDep
) -> Page[DocumentResponse]:
    documents, total = await service.list_for_user(user.id, page.limit, page.offset)
    return Page[DocumentResponse](
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID, service: DocumentServiceDep, user: CurrentUser
) -> DocumentResponse:
    document = await service.get(document_id, user.id)
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, service: DocumentServiceDep, user: CurrentUser
) -> None:
    await service.delete(document_id, user.id, is_admin=user.is_admin)
