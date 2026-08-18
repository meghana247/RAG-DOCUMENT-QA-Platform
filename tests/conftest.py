"""Test configuration.

Settings are read once at import and cached, so the environment has to be in
place before anything under `app.` is imported — hence the module-level setup
rather than a fixture.
"""
import os

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault(
    "SECURITY__SECRET_KEY", "test-secret-key-that-is-long-enough-to-pass-validation"
)
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/ragtest"
)
os.environ.setdefault("OBSERVABILITY__LOG_FORMAT", "console")

import pytest  # noqa: E402

from app.core.config import Settings  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings()
