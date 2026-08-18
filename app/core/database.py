"""Engine, session factory and the request-scoped session dependency."""
import logging
from collections.abc import AsyncIterator

from sqlalchemy import URL, make_url, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import DatabaseSettings, get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _build_url(cfg: DatabaseSettings) -> URL:
    url = make_url(cfg.url)
    if cfg.disable_statement_cache:
        # SQLAlchemy's own prepared-statement cache. Must be off behind a
        # transaction-pooling proxy, which may hand the same session a
        # different backend connection between statements.
        url = url.update_query_dict({"prepared_statement_cache_size": "0"}, append=False)
    return url


def create_engine(cfg: DatabaseSettings) -> AsyncEngine:
    kwargs: dict = {
        "echo": cfg.echo,
        "pool_pre_ping": cfg.pool_pre_ping,
        "future": True,
    }

    if cfg.pool_size == 0:
        # No pooling: one connection per checkout, closed on release. Correct for
        # serverless, where each instance is short-lived and a pool per instance
        # multiplies into the database's connection limit.
        kwargs["poolclass"] = NullPool
    else:
        kwargs.update(
            pool_size=cfg.pool_size,
            max_overflow=cfg.max_overflow,
            pool_recycle=cfg.pool_recycle_seconds,
        )

    connect_args: dict = {}
    if cfg.disable_statement_cache:
        connect_args["statement_cache_size"] = 0
    if connect_args:
        kwargs["connect_args"] = connect_args

    return create_async_engine(_build_url(cfg), **kwargs)


_settings = get_settings()
engine: AsyncEngine = create_engine(_settings.db)

SessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """Request-scoped session.

    Rolls back on any exception so a failed request cannot leave a transaction
    half-applied, and never commits implicitly — services commit deliberately.
    """
    async with SessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def check_connection() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("database health probe failed", exc_info=True)
        return False


async def dispose_engine() -> None:
    await engine.dispose()
