"""Embedding provider.

Two problems with the original version, both invisible until the service had
more than one concurrent user:

  * `model.encode()` is CPU-bound and was called directly from `async def`
    handlers, so every embedding call blocked the event loop and stalled all
    other requests for its duration.
  * The model was loaded on first use inside a request, making one unlucky
    request pay a multi-second import-and-load cost.

Encoding now runs in a worker thread behind a concurrency limit, the model is
loaded at startup, and query embeddings are cached (chat traffic repeats
questions far more than ingestion repeats passages).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from anyio import to_thread

from app.core.config import EmbeddingSettings

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class _LruCache:
    """Small bounded cache. `functools.lru_cache` cannot wrap a coroutine."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(0, capacity)
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> list[float] | None:
        if self._capacity == 0:
            return None
        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: list[float]) -> None:
        if self._capacity == 0:
            return
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)


class SentenceTransformerEmbedder:
    """Local embedding model. Satisfies the `Embedder` protocol."""

    def __init__(self, config: EmbeddingSettings, max_concurrency: int = 2) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._load_lock = asyncio.Lock()
        # Each encode call already parallelises across cores internally; letting
        # many run at once oversubscribes the CPU and makes every one slower.
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._cache = _LruCache(config.query_cache_size)

    @property
    def dimension(self) -> int:
        return self._config.dim

    async def warmup(self) -> None:
        """Load the model and run one encode so the first real request is fast."""
        model = await self._get_model()
        await to_thread.run_sync(lambda: model.encode(["warmup"]))
        logger.info("embedder ready", extra={"model": self._config.model_name})

    async def _get_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:  # another coroutine won the race
                return self._model
            logger.info("loading embedding model", extra={"model": self._config.model_name})

            def _load() -> SentenceTransformer:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self._config.model_name)
                reported = model.get_sentence_embedding_dimension()
                if reported != self._config.dim:
                    # Mismatched dimensions would be rejected by the pgvector
                    # column at insert time, or silently corrupt an existing
                    # index. Fail at startup instead.
                    raise ValueError(
                        f"Model '{self._config.model_name}' emits {reported}-dim vectors "
                        f"but EMBEDDING__DIM is {self._config.dim}. Changing models "
                        f"requires a migration to resize the embedding column and "
                        f"re-embedding every existing chunk."
                    )
                return model

            self._model = await to_thread.run_sync(_load)
            return self._model

    async def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        model = await self._get_model()

        def _run() -> list[list[float]]:
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # lets cosine distance reduce to a dot product
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in vectors]

        async with self._semaphore:
            return await to_thread.run_sync(_run)

    async def embed_query(self, text: str) -> list[float]:
        prepared = f"{self._config.query_prefix}{text.strip()}"
        key = hashlib.sha256(prepared.encode("utf-8")).hexdigest()

        if (cached := self._cache.get(key)) is not None:
            return cached

        vectors = await self._encode([prepared], batch_size=1)
        self._cache.put(key, vectors[0])
        return vectors[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > self._config.max_chunks_per_document:
            raise ValueError(
                f"Document produced {len(texts)} chunks, exceeding the limit of "
                f"{self._config.max_chunks_per_document}"
            )
        prepared = [f"{self._config.passage_prefix}{t}" for t in texts]
        return await self._encode(prepared, batch_size=self._config.batch_size)

    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._cache.hits, "misses": self._cache.misses}
