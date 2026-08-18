"""Application assembly and configuration guards.

`create_app()` is called directly rather than through TestClient: the lifespan
loads the embedding model, which these tests neither need nor should pay for.
"""
import pytest
from pydantic import ValidationError

from app.core.config import RetrievalSettings, SecuritySettings, Settings
from app.main import create_app


def _routes(app) -> set[str]:
    return {route.path for route in app.routes if hasattr(route, "path")}


def test_expected_routes_are_registered(settings):
    paths = _routes(create_app(settings))
    for expected in (
        "/health",
        "/health/ready",
        "/auth/register",
        "/auth/login",
        "/auth/me",
        "/documents/upload",
        "/documents/{document_id}",
        "/chat/sessions",
        "/chat/sessions/{session_id}/messages",
        "/chat/sessions/{session_id}/messages/stream",
        "/admin/users",
    ):
        assert expected in paths, f"missing route {expected}"


def test_api_prefix_is_applied(settings):
    settings = settings.model_copy(update={"api_prefix": "/api/v1"})
    assert "/api/v1/auth/login" in _routes(create_app(settings))


def test_docs_are_disabled_in_production(settings):
    production = settings.model_copy(update={"environment": "production"})
    assert create_app(production).docs_url is None
    assert create_app(settings).docs_url == "/docs"


class TestConfigGuards:
    def test_short_secret_key_is_rejected(self):
        with pytest.raises(ValidationError):
            SecuritySettings(secret_key="too-short")

    def test_placeholder_secret_key_is_rejected(self):
        # The value shipped in .env.example must never reach a deployment.
        with pytest.raises(ValidationError):
            SecuritySettings(secret_key="change-this-to-a-long-random-secret-key")

    def test_sync_database_driver_is_rejected(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host/db")
        with pytest.raises(ValidationError):
            Settings()

    def test_candidate_k_below_top_k_is_rejected(self, monkeypatch):
        # Over-fetching is what gives MMR and the reranker anything to reorder.
        monkeypatch.setenv("RETRIEVAL__TOP_K", "20")
        monkeypatch.setenv("RETRIEVAL__CANDIDATE_K", "5")
        with pytest.raises(ValidationError):
            Settings()

    def test_missing_api_key_is_rejected(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr(
            "app.core.config.Settings.model_config",
            {**Settings.model_config, "env_file": None},
        )
        with pytest.raises(ValidationError):
            Settings()

    def test_legacy_flat_env_vars_still_work(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        assert Settings().generation.model == "llama-3.1-8b-instant"

    def test_retrieval_defaults_are_sane(self):
        config = RetrievalSettings()
        assert config.candidate_k > config.top_k
        assert 0.0 <= config.mmr_lambda <= 1.0
        assert config.hnsw_ef_search >= config.candidate_k
