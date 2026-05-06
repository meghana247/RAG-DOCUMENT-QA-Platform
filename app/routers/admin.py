from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.database import get_db
from app.schemas.auth import UserResponse
from app.schemas.document import DocumentResponse
from app.dependencies import require_admin
from app.models.user import User
from app.models.document import Document

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=List[UserResponse])
async def list_all_users(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


@router.get("/documents", response_model=List[DocumentResponse])
async def list_all_documents(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin),
):
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()
