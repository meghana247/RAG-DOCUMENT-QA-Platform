from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: UUID
    user_id: UUID
    filename: str
    content_type: str
    size_bytes: int
    status: str
    error_message: str | None = None
    total_chunks: int
    page_count: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class UploadAccepted(BaseModel):
    """Returned by the upload endpoint.

    Ingestion runs in the background, so the response describes a document that
    is not yet queryable. `duplicate` tells the client that the bytes were already
    ingested and no work was scheduled.
    """

    document: DocumentResponse
    duplicate: bool = False
    message: str
