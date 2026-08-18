from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator

# bcrypt only considers the first 72 bytes of input; anything longer is silently
# truncated, so a 100-character password would be no stronger than its first 72.
BCRYPT_MAX_BYTES = 72
MIN_PASSWORD_LENGTH = 8


def _validate_password(value: str) -> str:
    if len(value) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(value.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
    return value


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(description=f"{MIN_PASSWORD_LENGTH}-{BCRYPT_MAX_BYTES} characters")

    @field_validator("password")
    @classmethod
    def check_password(cls, value: str) -> str:
        return _validate_password(value)


class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def check_password_length(cls, value: str) -> str:
        # Login only guards the upper bound: rejecting a short password here would
        # tell an attacker that the stored password is longer than what they tried.
        if len(value.encode("utf-8")) > BCRYPT_MAX_BYTES:
            raise ValueError(f"password must be at most {BCRYPT_MAX_BYTES} bytes")
        return value


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime


class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
