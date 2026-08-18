from fastapi import APIRouter, status

from app.core.dependencies import AuthServiceDep, CurrentUser
from app.schemas.auth import Token, UserLogin, UserRegister, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister, service: AuthServiceDep) -> UserResponse:
    user = await service.register(body.email, body.password)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=Token)
async def login(body: UserLogin, service: AuthServiceDep) -> Token:
    token, expires_at, _ = await service.authenticate(body.email, body.password)
    return Token(access_token=token, expires_at=expires_at)


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)
