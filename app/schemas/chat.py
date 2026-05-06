from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional, List


class ChatSessionCreate(BaseModel):
    document_id: Optional[UUID] = None
    title: str = "New Chat"


class ChatSessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    document_id: Optional[UUID]
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str


class ChatMessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    sources: List[dict]
    created_at: datetime

    model_config = {"from_attributes": True}
