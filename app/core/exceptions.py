"""Domain exceptions and the handlers that turn them into HTTP responses.

Services raise these; they know nothing about HTTP. Only `register_exception_handlers`
maps them to status codes, which keeps the service layer transport-agnostic and
testable without a client.
"""
import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base for all expected failures."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred"

    def __init__(self, message: str | None = None, **context: Any) -> None:
        self.message = message or self.message
        self.context = context
        super().__init__(self.message)


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "Resource not found"


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "Resource already exists"


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"
    message = "Invalid input"


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthenticated"
    message = "Invalid or expired credentials"


class AuthorizationError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"
    message = "Insufficient permissions"


class PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "payload_too_large"
    message = "Upload exceeds the maximum allowed size"


class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"
    message = "Unsupported file type"


class DocumentProcessingError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "document_processing_failed"
    message = "Could not extract text from the document"


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Too many requests"


class UpstreamError(AppError):
    """A dependency we do not control failed (LLM provider, etc.)."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "upstream_error"
    message = "An upstream service is unavailable"


def _body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {"error": {"code": code, "message": message}}
    if extra:
        payload["error"]["details"] = extra
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        # 4xx is caller error and expected; 5xx is ours and needs a stack trace.
        if exc.status_code >= 500:
            logger.error(
                "request failed: %s", exc.code, exc_info=exc, extra={"path": request.url.path}
            )
        else:
            logger.info(
                "request rejected: %s", exc.code, extra={"path": request.url.path, **exc.context}
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.code, exc.message, **exc.context),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_body("validation_error", "Request validation failed", fields=exc.errors()),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _db_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        # Never leak SQL or connection strings to the caller.
        logger.error("database error", exc_info=exc, extra={"path": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_body("database_unavailable", "The service is temporarily unavailable"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_body("internal_error", "An unexpected error occurred"),
        )
