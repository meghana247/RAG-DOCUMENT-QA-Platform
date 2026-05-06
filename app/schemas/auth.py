from pydantic import BaseModel, EmailStr, field_validator
from uuid import UUID
from datetime import datetime


def _password_within_bcrypt_limit(v: str) -> str:
    if len(v.encode("utf-8")) > 72:
        raise ValueError("password must be at most 72 bytes")
    return v


class UserRegister(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_within_bcrypt_limit(cls, v: str) -> str:
        return _password_within_bcrypt_limit(v)


class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_within_bcrypt_limit(cls, v: str) -> str:
        return _password_within_bcrypt_limit(v)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
