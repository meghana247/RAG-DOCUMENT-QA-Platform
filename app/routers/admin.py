from fastapi import APIRouter

from app.core.dependencies import AdminUser, DbSession, PaginationDep
from app.repositories.document_repository import DocumentRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import UserResponse
from app.schemas.common import Page
from app.schemas.document import DocumentResponse

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=Page[UserResponse])
async def list_users(
    session: DbSession, _: AdminUser, page: PaginationDep
) -> Page[UserResponse]:
    repository = UserRepository(session)
    users = await repository.list_all(page.limit, page.offset)
    return Page[UserResponse](
        items=[UserResponse.model_validate(u) for u in users],
        total=await repository.count(),
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/documents", response_model=Page[DocumentResponse])
async def list_documents(
    session: DbSession, _: AdminUser, page: PaginationDep
) -> Page[DocumentResponse]:
    repository = DocumentRepository(session)
    documents = await repository.list_all(page.limit, page.offset)
    return Page[DocumentResponse](
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=await repository.count_all(),
        limit=page.limit,
        offset=page.offset,
    )
