"""Structural interfaces for the swappable pieces of the pipeline.

These are `Protocol`s, not base classes: an implementation only has to match the
shape. That keeps a test double a five-line class, and means swapping the
embedder for a hosted API is a config change plus one new module, with no edits
to the pipeline.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.types import Chunk, RetrievedChunk, TextBlock


@runtime_checkable
class Chunker(Protocol):
    def split(self, blocks: list[TextBlock]) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(
        self,
        db: AsyncSession,
        *,
        query: str,
        query_embedding: list[float],
        user_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
        limit: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Generator(Protocol):
    async def generate(
        self,
        *,
        query: str,
        context: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> tuple[str, int | None, int | None]: ...

    def stream(
        self,
        *,
        query: str,
        context: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> AsyncIterator[str]: ...
