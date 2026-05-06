from sentence_transformers import SentenceTransformer
from app.config import settings

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def generate_embedding(text: str) -> list[float]:
    return get_model().encode(text, normalize_embeddings=True).tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()
