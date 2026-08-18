"""Composition root.

Everything expensive — the embedding model, the cross-encoder, the HTTP client —
is built exactly once here and shared. Nothing else in the codebase constructs
these, so there is a single place to look when swapping an implementation, and no
risk of a second copy of a 400MB model being loaded by an unrelated module.
"""
from __future__ import annotations

import logging

from groq import AsyncGroq

from app.core.config import Settings, get_settings
from app.rag.chunking import RecursiveChunker
from app.rag.contextualizer import QueryContextualizer
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.generation import GroqGenerator, build_client
from app.rag.pipeline import RagPipeline
from app.rag.reranking import CrossEncoderReranker, NoOpReranker
from app.rag.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

        self.groq_client: AsyncGroq = build_client(self.settings.generation)
        self.embedder = SentenceTransformerEmbedder(self.settings.embedding)
        self.reranker = (
            CrossEncoderReranker(self.settings.retrieval)
            if self.settings.retrieval.rerank_enabled
            else NoOpReranker()
        )
        self.retriever = HybridRetriever(self.settings.retrieval)
        self.generator = GroqGenerator(self.groq_client, self.settings.generation)
        self.contextualizer = QueryContextualizer(
            self.groq_client, self.settings.generation
        )
        self.chunker = RecursiveChunker(self.settings.chunking)

        self.pipeline = RagPipeline(
            settings=self.settings,
            embedder=self.embedder,
            retriever=self.retriever,
            reranker=self.reranker,
            generator=self.generator,
            contextualizer=self.contextualizer,
            chunker=self.chunker,
        )

    async def startup(self) -> None:
        """Load models before the first request rather than during it."""
        await self.embedder.warmup()
        if hasattr(self.reranker, "warmup"):
            await self.reranker.warmup()

    async def shutdown(self) -> None:
        await self.groq_client.close()


_container: Container | None = None


def build_container(settings: Settings | None = None) -> Container:
    global _container
    _container = Container(settings)
    return _container


def get_container() -> Container:
    if _container is None:
        raise RuntimeError(
            "Container not initialised. It is built in the application lifespan; "
            "tests should call build_container() explicitly."
        )
    return _container
