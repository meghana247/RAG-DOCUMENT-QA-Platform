from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:postgres@db:5432/ragdb"
    secret_key: str = "change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_chunks: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
