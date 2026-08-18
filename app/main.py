"""Application factory and wiring."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.container import build_container
from app.core.database import check_connection, dispose_engine
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.routers import admin, auth, chat, documents, health
from app.routers.health import VERSION

logger = logging.getLogger(__name__)

# Probes must never be rate limited — a limiter that throttles the health check
# takes the instance out of service under exactly the load it should survive.
_RATE_LIMIT_EXEMPT = frozenset({"/health", "/health/ready", "/docs", "/openapi.json"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    configure_logging(
        settings.observability.log_level, settings.observability.log_format
    )
    logger.info(
        "starting application",
        extra={
            "version": VERSION,
            "environment": settings.environment,
            "embedding_model": settings.embedding.model_name,
            "retrieval_strategy": settings.retrieval.strategy,
            "rerank_enabled": settings.retrieval.rerank_enabled,
        },
    )

    if not await check_connection():
        # Fail loudly at boot rather than serving 500s. Schema creation is *not*
        # done here: it belongs to `alembic upgrade head`, run as a deploy step.
        # Doing it in the lifespan means every replica races to create the same
        # tables and indexes on every restart.
        logger.error("database unreachable at startup")

    container = build_container(settings)
    await container.startup()

    try:
        yield
    finally:
        logger.info("shutting down")
        await container.shutdown()
        await dispose_engine()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="RAG Document Q&A Platform",
        description=(
            "Multi-tenant retrieval-augmented question answering over uploaded "
            "documents. Hybrid dense + lexical retrieval, reciprocal rank fusion, "
            "MMR diversification and cross-encoder reranking, with per-user data "
            "isolation enforced at the query level."
        ),
        version=VERSION,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        # Interactive docs expose every endpoint and schema; useful in
        # development, unnecessary surface area in production.
    )
    app.state.settings = settings

    register_exception_handlers(app)

    # Order matters: middleware added last runs first. Rate limiting should reject
    # before any per-request work happens, and the request id must be set before
    # the limiter logs anything.
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.rate_limit_per_minute,
        exempt_paths=_RATE_LIMIT_EXEMPT,
    )
    app.add_middleware(RequestContextMiddleware)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["X-Request-ID"],
        )

    prefix = settings.api_prefix.rstrip("/")
    for module in (health, auth, documents, chat, admin):
        app.include_router(module.router, prefix=prefix)

    return app


app = create_app()
