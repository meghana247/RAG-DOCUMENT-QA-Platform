from pydantic import BaseModel
from uuid import UUID
from datetime import datetime


class DocumentResponse(BaseModel):
    id: UUID
    user_id: UUID
    filename: str
    content_type: str
    total_chunks: int
    created_at: datetime

    model_config = {"from_attributes": True}
