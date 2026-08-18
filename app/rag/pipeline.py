"""The RAG pipeline orchestrator.

Holds the *sequence*; every step behind a protocol so it can be replaced or
stubbed. Retrieval order is deliberate:

    contextualise → embed → hybrid retrieve (over-fetch)
      → relevance floor → MMR (diversify) → cross-encoder rerank → top-k

Over-fetching first is what gives the later stages something to work with: the
floor can only remove noise it was shown, MMR can only diversify across
candidates it has, and the reranker can only promote a passage that was
retrieved. Diversification runs before reranking so the cross-encoder spends its
budget on distinct passages rather than five copies of the same paragraph.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.rag.chunking import RecursiveChunker
from app.rag.contextualizer import QueryContextualizer
from app.rag.extraction import extract_blocks
from app.rag.protocols import Embedder, Generator, Reranker, Retriever
from app.rag.retrieval import apply_mmr, apply_relevance_floor
from app.rag.types import Chunk, RagAnswer, RetrievedChunk, StageTimings

logger = logging.getLogger(__name__)


class _Clock:
    """Millisecond stopwatch. `perf_counter` so it is immune to clock changes."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = (now - self._start) * 1000.0
        self._start = now
        return elapsed


class RagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        contextualizer: QueryContextualizer,
        chunker: RecursiveChunker | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._contextualizer = contextualizer
        self._chunker = chunker or RecursiveChunker(settings.chunking)

    # ------------------------------------------------------------------ ingest

    def chunk_document(
        self, content: bytes, filename: str, content_type: str
    ) -> tuple[list[Chunk], int | None]:
        """Extract and chunk. Pure CPU, no I/O — safe to call before any writes."""
        blocks = extract_blocks(content, filename, content_type)
        chunks = self._chunker.split(blocks)
        pages = [b.page_number for b in blocks if b.page_number]
        return chunks, (max(pages) if pages else None)

    async def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return await self._embedder.embed_passages([chunk.content for chunk in chunks])

    # ------------------------------------------------------------------- query

    async def _retrieve(
        self,
        db: AsyncSession,
        *,
        search_query: str,
        query_embedding: list[float],
        user_id: uuid.UUID,
        document_id: uuid.UUID | None,
        timings: StageTimings,
        clock: _Clock,
    ) -> list[RetrievedChunk]:
        config = self._settings.retrieval

        candidates = await self._retriever.retrieve(
            db,
            query=search_query,
            query_embedding=query_embedding,
            user_id=user_id,
            document_id=document_id,
            limit=config.candidate_k,
        )
        candidates = apply_relevance_floor(candidates, config.min_similarity)
        timings.retrieve_ms = clock.lap()

        if not candidates:
            return []

        if config.mmr_enabled:
            # Diversify over a window wider than top_k so the reranker still has
            # choices, but narrower than the full candidate set so MMR's
            # pairwise comparisons stay cheap.
            window = min(len(candidates), max(config.rerank_candidates, config.top_k * 3))
            candidates = apply_mmr(candidates, query_embedding, window, config.mmr_lambda)

        selected = await self._reranker.rerank(search_query, candidates, config.top_k)
        timings.rerank_ms = clock.lap()
        return selected

    async def answer(
        self,
        db: AsyncSession,
        *,
        query: str,
        user_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> RagAnswer:
        history = history or []
        timings = StageTimings()
        clock = _Clock()
        total = _Clock()

        search_query = await self._contextualizer.contextualize(query, history)
        timings.contextualize_ms = clock.lap()

        query_embedding = await self._embedder.embed_query(search_query)
        timings.embed_ms = clock.lap()

        selected = await self._retrieve(
            db,
            search_query=search_query,
            query_embedding=query_embedding,
            user_id=user_id,
            document_id=document_id,
            timings=timings,
            clock=clock,
        )

        # The prompt is built from the chunks that fit the budget; those same
        # chunks are what gets cited back, so [n] always resolves.
        answer_text, prompt_tokens, completion_tokens = await self._generator.generate(
            query=query, context=selected, history=history
        )
        timings.generate_ms = clock.lap()
        timings.total_ms = total.lap()

        logger.info(
            "rag query served",
            extra={
                "chunks_used": len(selected),
                "grounded": bool(selected),
                "rewritten": search_query != query,
                **timings.as_dict(),
            },
        )

        return RagAnswer(
            answer=answer_text,
            sources=selected,
            timings=timings,
            search_query=search_query,
            grounded=bool(selected),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream_answer(
        self,
        db: AsyncSession,
        *,
        query: str,
        user_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[list[RetrievedChunk], AsyncIterator[str]]:
        """Retrieve eagerly, then hand back a token iterator.

        Sources are resolved before streaming starts so the transport can emit a
        `sources` event first — a client that only learns its citations after the
        answer has finished rendering cannot show them inline.
        """
        history = history or []
        timings = StageTimings()
        clock = _Clock()

        search_query = await self._contextualizer.contextualize(query, history)
        timings.contextualize_ms = clock.lap()

        query_embedding = await self._embedder.embed_query(search_query)
        timings.embed_ms = clock.lap()

        selected = await self._retrieve(
            db,
            search_query=search_query,
            query_embedding=query_embedding,
            user_id=user_id,
            document_id=document_id,
            timings=timings,
            clock=clock,
        )

        stream = self._generator.stream(query=query, context=selected, history=history)
        return selected, stream
