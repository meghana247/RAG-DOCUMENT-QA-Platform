from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    document_id: UUID | None = Field(
        default=None,
        description="Restrict retrieval to one document. Omit to search all of the user's documents.",
    )
    title: str = "New Chat"


class ChatSessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    document_id: UUID | None
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    """One retrieved passage the answer drew on.

    `chunk_index` is kept alongside `page_number` because plain-text and DOCX
    sources have no pages, so it is the only stable locator for those.
    """

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    content: str
    score: float


class ChatMessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    sources: list[Citation] = []
    latency_ms: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AnswerResponse(BaseModel):
    """The assistant message plus per-request retrieval metadata."""

    message: ChatMessageResponse
    grounded: bool = Field(
        description="False when nothing cleared the relevance threshold, meaning the "
        "answer is a refusal rather than a document-grounded response."
    )
    search_query: str = Field(
        description="The query actually used for retrieval, after follow-up rewriting."
    )
    timings: dict[str, float] | None = None
