import uuid
from typing import Optional
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import DocumentChunk, Document
from app.models.chat import ChatSession, ChatMessage
from app.services.embedding_service import generate_embedding
from app.config import settings

client = Groq(api_key=settings.groq_api_key)


async def retrieve_chunks(
    db: AsyncSession,
    query_embedding: list,
    user_id: uuid.UUID,
    document_id: Optional[uuid.UUID] = None,
) -> list[DocumentChunk]:
    query = (
        select(DocumentChunk)
        .where(DocumentChunk.user_id == user_id)
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(settings.top_k_chunks)
    )
    if document_id:
        query = query.where(DocumentChunk.document_id == document_id)
    result = await db.execute(query)
    return result.scalars().all()


async def _get_filename(db: AsyncSession, doc_id: uuid.UUID) -> str:
    result = await db.execute(select(Document.filename).where(Document.id == doc_id))
    return result.scalar_one_or_none() or "unknown"


async def answer_query(
    db: AsyncSession,
    session: ChatSession,
    user_message: str,
    user_id: uuid.UUID,
) -> tuple[str, list]:
    query_embedding = generate_embedding(user_message)
    chunks = await retrieve_chunks(db, query_embedding, user_id, document_id=session.document_id)

    context_parts, sources = [], []
    for chunk in chunks:
        filename = await _get_filename(db, chunk.document_id)
        context_parts.append(f"[Source: {filename}, chunk {chunk.chunk_index}]\n{chunk.content}")
        sources.append({
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "filename": filename,
            "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
            "chunk_index": chunk.chunk_index,
        })

    context = "\n\n---\n\n".join(context_parts)

    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .limit(10)
    )
    history = history_result.scalars().all()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based on the provided document context. "
                "Always cite the source file and chunk index when referencing information. "
                "If the answer is not in the context, say so clearly.\n\n"
                f"Context:\n{context}"
            ),
        }
    ]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content, sources


async def create_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    document_id: Optional[uuid.UUID],
    title: str,
) -> ChatSession:
    session = ChatSession(user_id=user_id, document_id=document_id, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def save_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    sources: list = None,
) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content, sources=sources or [])
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg
