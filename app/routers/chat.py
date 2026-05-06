import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.database import get_db
from app.schemas.chat import ChatSessionCreate, ChatSessionResponse, MessageCreate, ChatMessageResponse
from app.services.rag_service import create_session, answer_query, save_message
from app.models.chat import ChatSession, ChatMessage
from app.models.document import Document
from app.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def new_session(
    body: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.document_id:
        result = await db.execute(
            select(Document).where(Document.id == body.document_id, Document.user_id == current_user.id)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Document not found")
    return await create_session(db, current_user.id, body.document_id, body.title)


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
    )
    return result.scalars().all()


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
    session_id: str,
    body: MessageCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await save_message(db, session.id, "user", body.content)
    answer, sources = await answer_query(db, session, body.content, current_user.id)
    return await save_message(db, session.id, "assistant", answer, sources)


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == uuid.UUID(session_id))
        .order_by(ChatMessage.created_at.asc())
    )
    return result.scalars().all()


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == uuid.UUID(session_id),
            ChatSession.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()
