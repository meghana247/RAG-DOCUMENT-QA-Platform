"""Application settings.

Grouped into nested models so each subsystem owns its own knobs. Env vars use
`__` as the nesting delimiter, e.g. `RAG__TOP_K=8`, `DB__POOL_SIZE=10`.
Flat legacy names (DATABASE_URL, GROQ_API_KEY, SECRET_KEY) are still honoured.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_async_driver(url: str) -> str:
    if not url.startswith("postgresql+asyncpg://"):
        raise ValueError(
            "Database URL must use the asyncpg driver: postgresql+asyncpg://... "
            f"(got {url.split('://', 1)[0]}://)"
        )
    return url


def _validate_secret_key(raw: str) -> str:
    if len(raw) < 32:
        raise ValueError(
            "Secret key must be at least 32 characters. Generate one with: "
            "python -c 'import secrets;print(secrets.token_urlsafe(48))'"
        )
    if "change" in raw.lower() or raw.lower().startswith("your-"):
        raise ValueError("Secret key is still a placeholder value")
    return raw


class DatabaseSettings(BaseSettings):
    url: str = "postgresql+asyncpg://postgres:postgres@db:5432/ragdb"

    # Pooling. Set pool_size=0 to use NullPool — required behind a serverless
    # platform or an external pooler (pgbouncer/Neon), where holding a pool per
    # instance exhausts the server's connection limit.
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle_seconds: int = 1800
    pool_pre_ping: bool = True

    # asyncpg prepares statements by default, which breaks under pgbouncer in
    # transaction pooling mode. Disable when connecting through a pooler.
    disable_statement_cache: bool = False

    echo: bool = False

    @field_validator("url")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        return _validate_async_driver(v)


class SecuritySettings(BaseSettings):
    secret_key: SecretStr = SecretStr("")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    bcrypt_rounds: int = 12

    @field_validator("secret_key")
    @classmethod
    def _reject_weak_key(cls, v: SecretStr) -> SecretStr:
        _validate_secret_key(v.get_secret_value())
        return v


class EmbeddingSettings(BaseSettings):
    # `model_` is a protected prefix in pydantic; opting out keeps the readable
    # EMBEDDING__MODEL_NAME env var instead of renaming the field.
    model_config = SettingsConfigDict(protected_namespaces=())

    model_name: str = "all-MiniLM-L6-v2"
    dim: int = 384
    batch_size: int = 32

    # Some models (e5, bge, nomic) require asymmetric prefixes on queries vs
    # passages. Getting these wrong silently degrades recall, so they are
    # explicit rather than inferred from the model name.
    query_prefix: str = ""
    passage_prefix: str = ""

    # Query embeddings are cached in-process; ingestion embeddings are not
    # (they are seen once).
    query_cache_size: int = 2048

    # Cap on how many chunks are embedded in one ingest call, to bound peak
    # memory on small containers.
    max_chunks_per_document: int = 5000


class ChunkingSettings(BaseSettings):
    # Sized in characters. ~4 chars/token for English, so 1200 chars is roughly
    # 300 tokens — large enough to hold a coherent idea, small enough that 8 of
    # them fit a context budget comfortably.
    chunk_size: int = 1200
    chunk_overlap: int = 180
    min_chunk_size: int = 100


class RetrievalSettings(BaseSettings):
    # Final number of chunks handed to the generator.
    top_k: int = 6

    # How many candidates each retriever fetches before fusion. Over-fetching
    # is what makes reranking and MMR useful — with candidate_k == top_k there
    # is nothing left to reorder.
    candidate_k: int = 40

    strategy: Literal["vector", "keyword", "hybrid"] = "hybrid"

    # Reciprocal Rank Fusion constant. 60 is the value from the original RRF
    # paper and is not sensitive.
    rrf_k: int = 60

    # HNSW search breadth. Raised above the default 40 because every query is
    # filtered by user_id; a narrow search can walk the graph into other
    # tenants' vectors and return fewer than top_k rows for this tenant.
    hnsw_ef_search: int = 200

    # Maximal Marginal Relevance: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7

    # Cosine similarity floor. Chunks below this never reach the prompt, so an
    # off-topic question yields an honest "not in your documents" instead of a
    # confident answer built from the least-bad matches.
    min_similarity: float = 0.25

    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20


class GenerationSettings(BaseSettings):
    api_key: SecretStr = SecretStr("")
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_retries: int = 2

    # Character budget for retrieved context in the prompt. Chunks are dropped
    # from the tail once this is hit, so a large top_k cannot silently blow the
    # model's context window.
    context_char_budget: int = 12000

    # Number of prior turns replayed into the prompt.
    history_turns: int = 6

    # Rewrite follow-up questions into standalone ones before retrieval.
    # Costs one extra (small) LLM call per message when history exists.
    contextualize_queries: bool = True
    contextualize_model: str = "llama-3.1-8b-instant"


class ObservabilitySettings(BaseSettings):
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Attach per-stage retrieval timings to chat responses. Useful in staging,
    # noisy in production.
    expose_debug_timings: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = "development"
    api_prefix: str = ""
    cors_origins: list[str] = Field(default_factory=list)

    # Max upload size enforced in the router before the body is buffered.
    max_upload_bytes: int = 25 * 1024 * 1024

    rate_limit_per_minute: int = 60

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    # --- legacy flat env vars, kept so existing .env files keep working ---
    database_url: str | None = None
    secret_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    groq_model: str | None = None

    @model_validator(mode="after")
    def _apply_legacy_aliases(self) -> "Settings":
        # Validators do not re-run on assignment, so each aliased value is
        # checked explicitly here. Without this a legacy DATABASE_URL carrying a
        # sync driver reached create_async_engine and failed with an opaque error.
        if self.database_url:
            self.db.url = _validate_async_driver(self.database_url)
        if self.secret_key:
            _validate_secret_key(self.secret_key.get_secret_value())
            self.security.secret_key = self.secret_key
        if self.groq_api_key:
            self.generation.api_key = self.groq_api_key
        if self.groq_model:
            self.generation.model = self.groq_model

        if not self.generation.api_key.get_secret_value():
            raise ValueError("GROQ_API_KEY (or GENERATION__API_KEY) is required")

        if self.retrieval.candidate_k < self.retrieval.top_k:
            raise ValueError("retrieval.candidate_k must be >= retrieval.top_k")

        return self

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
