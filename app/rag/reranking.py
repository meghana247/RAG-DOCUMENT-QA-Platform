"""Cross-encoder reranking.

The retriever scores the query and each chunk independently — their embeddings
never meet, which is what makes ANN search fast. A cross-encoder instead reads
the pair together and can judge whether the chunk actually answers *this*
question. It is far too slow to run over a corpus, which is why it runs last,
over the ~20 candidates retrieval already narrowed to.

This is a quality optimisation, not a correctness requirement: if the model fails
to load, retrieval order is kept and the request still succeeds.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from anyio import to_thread

from app.core.config import RetrievalSettings
from app.rag.types import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Satisfies the `Reranker` protocol."""

    def __init__(self, config: RetrievalSettings, max_concurrency: int = 2) -> None:
        self._config = config
        self._model: CrossEncoder | None = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._unavailable = False

    async def warmup(self) -> None:
        if not self._config.rerank_enabled:
            return
        model = await self._get_model()
        if model is not None:
            await to_thread.run_sync(lambda: model.predict([("warmup", "warmup")]))
            logger.info("reranker ready", extra={"model": self._config.rerank_model})

    async def _get_model(self) -> CrossEncoder | None:
        if self._unavailable:
            return None
        if self._model is not None:
            return self._model

        async with self._load_lock:
            if self._model is not None:
                return self._model
            if self._unavailable:
                return None

            def _load() -> CrossEncoder:
                from sentence_transformers import CrossEncoder

                return CrossEncoder(self._config.rerank_model)

            try:
                self._model = await to_thread.run_sync(_load)
            except Exception:
                # Degrade instead of failing every chat request.
                self._unavailable = True
                logger.warning(
                    "reranker unavailable, falling back to fusion order",
                    exc_info=True,
                    extra={"model": self._config.rerank_model},
                )
                return None
            return self._model

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not self._config.rerank_enabled or len(candidates) <= 1:
            return candidates[:top_k]

        model = await self._get_model()
        if model is None:
            return candidates[:top_k]

        # Only the strongest candidates are worth the cross-encoder's time; the
        # tail is unlikely to be promoted into the top-k.
        window = candidates[: self._config.rerank_candidates]
        overflow = candidates[self._config.rerank_candidates :]
        pairs = [(query, candidate.content) for candidate in window]

        def _predict() -> list[float]:
            return [float(score) for score in model.predict(pairs, show_progress_bar=False)]

        try:
            async with self._semaphore:
                scores = await to_thread.run_sync(_predict)
        except Exception:
            logger.warning("reranking failed, keeping fusion order", exc_info=True)
            return candidates[:top_k]

        for candidate, score in zip(window, scores, strict=False):
            candidate.rerank_score = score
            candidate.final_score = score

        window.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
        return (window + overflow)[:top_k]


class NoOpReranker:
    """Used when reranking is disabled, so the pipeline needs no branch."""

    async def warmup(self) -> None:  # pragma: no cover - trivial
        return

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_k]
