"""Hybrid retrieval: dense vectors + Postgres full-text, fused with RRF.

Why hybrid. A pure embedding search is strong on paraphrase and weak on exact
tokens — invoice numbers, error codes, surnames, API names, acronyms. Those are
precisely the terms users type. Postgres full-text search is the opposite. Fusing
the two rankings recovers both behaviours without adding a search service.

Why one SQL statement. `AsyncSession` is not safe for concurrent use, so running
the two legs as separate awaits would serialise into two round trips anyway.
Expressing both legs plus the fusion as CTEs lets Postgres do the work in one
trip, and the planner can use the HNSW and GIN indexes independently.

Reciprocal Rank Fusion combines by *rank*, not score, which matters because
cosine similarity (0..1) and `ts_rank_cd` (unbounded) are not comparable
quantities. Attempting to blend them with weights requires per-corpus tuning;
RRF needs none.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import RetrievalSettings
from app.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

# Selected for every candidate regardless of which leg found it.
_PROJECTION = """
    c.id            AS chunk_id,
    c.document_id   AS document_id,
    c.content       AS content,
    c.chunk_index   AS chunk_index,
    c.page_number   AS page_number,
    c.section       AS section,
    c.embedding     AS embedding,
    d.filename      AS filename
"""

# Both legs are bounded by :candidate_k and filtered by tenant. The
# `CAST(:document_id AS uuid)` form is required rather than bare `:document_id`
# so asyncpg can infer the parameter type when the value is NULL.
_TENANT_FILTER = """
    c.user_id = CAST(:user_id AS uuid)
    AND (
        CAST(:document_id AS uuid) IS NULL
        OR c.document_id = CAST(:document_id AS uuid)
    )
"""

_HYBRID_SQL = f"""
WITH vector_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY c.embedding <=> CAST(:query_vector AS vector)) AS rank,
        1 - (c.embedding <=> CAST(:query_vector AS vector)) AS similarity
    FROM document_chunks c
    WHERE {_TENANT_FILTER}
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:query_vector AS vector)
    LIMIT :candidate_k
),
keyword_hits AS (
    SELECT
        c.id,
        ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.content_tsv, q.query) DESC) AS rank,
        ts_rank_cd(c.content_tsv, q.query) AS score
    FROM document_chunks c
    CROSS JOIN websearch_to_tsquery('english', :query_text) AS q(query)
    WHERE {_TENANT_FILTER}
      AND c.content_tsv @@ q.query
    ORDER BY ts_rank_cd(c.content_tsv, q.query) DESC
    LIMIT :candidate_k
),
fused AS (
    SELECT
        COALESCE(v.id, k.id) AS id,
        COALESCE(1.0 / (:rrf_k + v.rank), 0.0)
            + COALESCE(1.0 / (:rrf_k + k.rank), 0.0) AS fusion_score,
        v.similarity AS vector_score,
        k.score      AS keyword_score
    FROM vector_hits v
    FULL OUTER JOIN keyword_hits k ON v.id = k.id
)
SELECT {_PROJECTION},
       f.fusion_score, f.vector_score, f.keyword_score
FROM fused f
JOIN document_chunks c ON c.id = f.id
JOIN documents d       ON d.id = c.document_id
ORDER BY f.fusion_score DESC
LIMIT :candidate_k
"""

_VECTOR_SQL = f"""
SELECT {_PROJECTION},
       1 - (c.embedding <=> CAST(:query_vector AS vector)) AS vector_score,
       NULL::real AS keyword_score,
       NULL::real AS fusion_score
FROM document_chunks c
JOIN documents d ON d.id = c.document_id
WHERE {_TENANT_FILTER}
  AND c.embedding IS NOT NULL
ORDER BY c.embedding <=> CAST(:query_vector AS vector)
LIMIT :candidate_k
"""

_KEYWORD_SQL = f"""
SELECT {_PROJECTION},
       NULL::real AS vector_score,
       ts_rank_cd(c.content_tsv, q.query) AS keyword_score,
       NULL::real AS fusion_score
FROM document_chunks c
JOIN documents d ON d.id = c.document_id
CROSS JOIN websearch_to_tsquery('english', :query_text) AS q(query)
WHERE {_TENANT_FILTER}
  AND c.content_tsv @@ q.query
ORDER BY ts_rank_cd(c.content_tsv, q.query) DESC
LIMIT :candidate_k
"""


def _to_vector_literal(vector: list[float]) -> str:
    """pgvector's text input format.

    Passed as a string plus an explicit cast so the query works whether or not
    the pgvector asyncpg codec happens to be registered on this connection.
    """
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"


def _parse_embedding(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        # Arrives as '[0.1,0.2,...]' when no codec is registered.
        return [float(part) for part in raw.strip("[]").split(",") if part]
    return [float(value) for value in raw]


class HybridRetriever:
    """Satisfies the `Retriever` protocol."""

    def __init__(self, config: RetrievalSettings) -> None:
        self._config = config

    async def retrieve(
        self,
        db: AsyncSession,
        *,
        query: str,
        query_embedding: list[float],
        user_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        candidate_k = max(limit or self._config.candidate_k, self._config.top_k)
        strategy = self._config.strategy

        params: dict[str, Any] = {
            "user_id": str(user_id),
            "document_id": str(document_id) if document_id else None,
            "candidate_k": candidate_k,
            "query_vector": _to_vector_literal(query_embedding),
            "query_text": query,
            "rrf_k": self._config.rrf_k,
        }

        if strategy == "vector":
            sql = _VECTOR_SQL
        elif strategy == "keyword":
            sql = _KEYWORD_SQL
        else:
            sql = _HYBRID_SQL

        # HNSW is approximate: it explores `ef_search` neighbours and only then
        # applies the tenant filter, so a low value can return fewer rows than
        # requested for a user whose chunks are sparse in the graph. Raising it
        # trades a little latency for recall. SET LOCAL is scoped to this
        # transaction, so it cannot leak into other requests.
        if strategy != "keyword":
            await db.execute(
                text(f"SET LOCAL hnsw.ef_search = {int(self._config.hnsw_ef_search)}")
            )

        rows = (await db.execute(text(sql), params)).mappings().all()

        candidates = [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                filename=row["filename"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                page_number=row["page_number"],
                section=row["section"],
                vector_score=(
                    float(row["vector_score"]) if row["vector_score"] is not None else None
                ),
                keyword_score=(
                    float(row["keyword_score"]) if row["keyword_score"] is not None else None
                ),
                fusion_score=(
                    float(row["fusion_score"]) if row["fusion_score"] is not None else None
                ),
                embedding=_parse_embedding(row["embedding"]),
            )
            for row in rows
        ]

        # A chunk found only by the keyword leg has no cosine score from SQL.
        # Fill it in so the relevance floor can be applied uniformly — the
        # embeddings are already in hand, so this costs one dot product each.
        _backfill_vector_scores(candidates, query_embedding)

        for candidate in candidates:
            candidate.final_score = candidate.fusion_score or candidate.vector_score or 0.0

        logger.debug(
            "retrieval candidates",
            extra={
                "strategy": strategy,
                "candidates": len(candidates),
                "vector_hits": sum(1 for c in candidates if c.vector_score is not None),
                "keyword_hits": sum(1 for c in candidates if c.keyword_score is not None),
            },
        )
        return candidates


def _backfill_vector_scores(
    candidates: list[RetrievedChunk], query_embedding: list[float]
) -> None:
    missing = [c for c in candidates if c.vector_score is None and c.embedding]
    if not missing:
        return
    import numpy as np

    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = np.asarray([c.embedding for c in missing], dtype=np.float32)
    # Vectors are stored L2-normalised, so the dot product *is* cosine similarity.
    for candidate, score in zip(missing, matrix @ query, strict=False):
        candidate.vector_score = float(score)


def apply_relevance_floor(
    candidates: list[RetrievedChunk], min_similarity: float
) -> list[RetrievedChunk]:
    """Drop weak matches so an off-topic question gets an honest refusal.

    Lexical hits are exempt. A chunk that literally contains the user's rare term
    is relevant even when its embedding is distant — rescuing exactly that case
    is the reason the keyword leg exists, so applying a cosine floor to it would
    undo the benefit.
    """
    kept = [
        candidate
        for candidate in candidates
        if candidate.keyword_score is not None
        or (candidate.vector_score or 0.0) >= min_similarity
    ]
    if len(kept) < len(candidates):
        logger.debug(
            "relevance floor applied",
            extra={"dropped": len(candidates) - len(kept), "threshold": min_similarity},
        )
    return kept


def apply_mmr(
    candidates: list[RetrievedChunk], query_embedding: list[float], k: int, lambda_: float
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance re-selection.

    Overlapping chunks — and chunk overlap guarantees some — otherwise fill the
    top-k with near-duplicates, so the model sees the same sentence five times
    and the other four slots of context are wasted. MMR penalises a candidate by
    its similarity to what has already been picked.
    """
    if k <= 0:
        return []

    usable = [c for c in candidates if c.embedding]
    if len(usable) <= 1:
        return candidates[:k]

    # k == 1 still runs the selection below rather than returning candidates[0]:
    # that shortcut would assume the caller pre-sorted by relevance.

    import numpy as np

    matrix = np.asarray([c.embedding for c in usable], dtype=np.float32)
    query = np.asarray(query_embedding, dtype=np.float32)
    relevance = matrix @ query
    similarity = matrix @ matrix.T

    selected: list[int] = [int(np.argmax(relevance))]
    remaining = set(range(len(usable))) - set(selected)

    while len(selected) < min(k, len(usable)) and remaining:
        best_index, best_score = None, -np.inf
        for index in remaining:
            redundancy = float(np.max(similarity[index, selected]))
            score = lambda_ * float(relevance[index]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected.append(best_index)
        remaining.discard(best_index)

    picked = [usable[i] for i in selected]

    # Candidates with no stored embedding cannot participate; append them after
    # the diversified set rather than discarding them.
    if len(picked) < k:
        picked.extend([c for c in candidates if not c.embedding][: k - len(picked)])
    return picked
