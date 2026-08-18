"""Value objects passed between RAG stages.

Every stage takes and returns one of these rather than ORM rows, so stages can
be unit-tested without a database and swapped without touching their callers.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextBlock:
    """A unit of text as it appeared in the source document.

    Extraction produces these; the chunker consumes them. Keeping page/section
    here is what lets a citation say "page 7" rather than "chunk 42".
    """

    text: str
    page_number: int | None = None
    section: str | None = None


@dataclass(slots=True)
class Chunk:
    """A chunk ready to be embedded and stored."""

    content: str
    index: int
    page_number: int | None = None
    section: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval, carrying every score that touched it.

    Scores are kept separately instead of collapsed into one number so that a
    bad ranking can be diagnosed after the fact — you can see whether the vector
    leg, the keyword leg, or the reranker was responsible.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None

    vector_score: float | None = None       # cosine similarity, 0..1
    keyword_score: float | None = None      # ts_rank_cd, unbounded
    fusion_score: float | None = None       # reciprocal rank fusion
    rerank_score: float | None = None       # cross-encoder logit
    final_score: float = 0.0

    embedding: list[float] | None = None    # retained only for MMR

    def to_citation(self) -> dict[str, Any]:
        """Compact form persisted on the message and returned to the client."""
        preview = self.content[:240] + ("..." if len(self.content) > 240 else "")
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section": self.section,
            "content": preview,
            "score": round(self.final_score, 4),
        }


@dataclass(slots=True)
class StageTimings:
    """Wall-clock per stage. Logged on every query; surfaced to the API only
    when observability.expose_debug_timings is on."""

    contextualize_ms: float = 0.0
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "contextualize_ms": round(self.contextualize_ms, 1),
            "embed_ms": round(self.embed_ms, 1),
            "retrieve_ms": round(self.retrieve_ms, 1),
            "rerank_ms": round(self.rerank_ms, 1),
            "generate_ms": round(self.generate_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass(slots=True)
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    timings: StageTimings = field(default_factory=StageTimings)
    search_query: str = ""          # post-contextualisation query actually used
    grounded: bool = True           # False when nothing cleared the score floor
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
