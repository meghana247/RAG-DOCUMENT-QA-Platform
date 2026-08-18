"""Repository base.

Repositories own queries; they never commit. Transaction boundaries belong to
the service layer, so one service call can span several repositories and still
be atomic.
"""
from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
