import io
import uuid
from typing import List
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document, DocumentChunk
from app.services.embedding_service import generate_embeddings
from app.config import settings


def extract_text(content: bytes, content_type: str, filename: str) -> str:
    if content_type == "application/pdf" or filename.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or filename.endswith(".docx"):
        from docx import Document as DocxDoc
        doc = DocxDoc(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    return content.decode("utf-8", errors="replace")


def chunk_text(text: str) -> List[str]:
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + settings.chunk_size])
        start += settings.chunk_size - settings.chunk_overlap
    return [c for c in chunks if c.strip()]


async def process_document(db: AsyncSession, file: UploadFile, user_id: uuid.UUID) -> Document:
    content = await file.read()
    text = extract_text(content, file.content_type or "", file.filename or "")
    chunks = chunk_text(text)

    doc = Document(
        user_id=user_id,
        filename=file.filename,
        content_type=file.content_type or "text/plain",
        total_chunks=len(chunks),
    )
    db.add(doc)
    await db.flush()

    embeddings = generate_embeddings(chunks)
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        db.add(DocumentChunk(
            document_id=doc.id,
            user_id=user_id,
            content=chunk,
            chunk_index=i,
            embedding=emb,
        ))

    await db.commit()
    await db.refresh(doc)
    return doc


async def get_user_documents(db: AsyncSession, user_id: uuid.UUID) -> List[Document]:
    result = await db.execute(
        select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


async def delete_document(db: AsyncSession, doc_id: uuid.UUID, user_id: uuid.UUID, is_admin: bool = False) -> bool:
    query = select(Document).where(Document.id == doc_id)
    if not is_admin:
        query = query.where(Document.user_id == user_id)
    result = await db.execute(query)
    doc = result.scalar_one_or_none()
    if not doc:
        return False
    await db.delete(doc)
    await db.commit()
    return True
