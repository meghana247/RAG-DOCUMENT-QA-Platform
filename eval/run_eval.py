"""Compare retrieval configurations against the golden set.

    python -m eval.run_eval --user-email you@example.com

Requires a database with documents already ingested, and a golden set whose
questions are answerable from them. The point is not the absolute numbers — they
depend entirely on your corpus — but the *comparison between rows*: whether
hybrid beats vector-only here, whether reranking earns its latency, whether a
larger candidate_k pays for itself.

Run this before and after any retrieval change. A change that improves nDCG on
five questions and drops it on twenty is not an improvement.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import time
from typing import Any

from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.core.container import build_container
from app.core.database import SessionFactory
from app.models.user import User
from app.rag.retrieval import HybridRetriever, apply_mmr, apply_relevance_floor
from app.rag.types import RetrievedChunk
from eval.metrics import EvalResult, aggregate

GOLDEN_SET = pathlib.Path(__file__).parent / "golden_set.json"


def _is_relevant(chunk: RetrievedChunk, question: dict[str, Any]) -> bool:
    needles = question.get("must_contain", [])
    if not needles:
        return False
    haystack = chunk.content.lower()
    hits = [needle.lower() in haystack for needle in needles]
    return all(hits) if question.get("match") == "all" else any(hits)


# Each row isolates one variable against the baseline, so a difference in the
# output table has exactly one cause.
CONFIGURATIONS: list[tuple[str, dict[str, Any]]] = [
    ("vector only", {"strategy": "vector", "mmr_enabled": False, "rerank_enabled": False}),
    ("keyword only", {"strategy": "keyword", "mmr_enabled": False, "rerank_enabled": False}),
    ("hybrid (RRF)", {"strategy": "hybrid", "mmr_enabled": False, "rerank_enabled": False}),
    ("hybrid + MMR", {"strategy": "hybrid", "mmr_enabled": True, "rerank_enabled": False}),
    ("hybrid + MMR + rerank", {"strategy": "hybrid", "mmr_enabled": True, "rerank_enabled": True}),
]


async def evaluate(
    name: str, settings: Settings, overrides: dict[str, Any], user_id, questions: list[dict]
) -> EvalResult:
    retrieval = settings.retrieval.model_copy(update=overrides)
    container = build_container(settings)
    retriever = HybridRetriever(retrieval)

    per_question: list[tuple[list[bool], int]] = []
    latencies: list[float] = []

    async with SessionFactory() as session:
        for question in questions:
            started = time.perf_counter()

            embedding = await container.embedder.embed_query(question["question"])
            candidates = await retriever.retrieve(
                session,
                query=question["question"],
                query_embedding=embedding,
                user_id=user_id,
                limit=retrieval.candidate_k,
            )
            candidates = apply_relevance_floor(candidates, retrieval.min_similarity)

            if retrieval.mmr_enabled and candidates:
                candidates = apply_mmr(
                    candidates, embedding, retrieval.rerank_candidates, retrieval.mmr_lambda
                )
            if retrieval.rerank_enabled and candidates:
                candidates = await container.reranker.rerank(
                    question["question"], candidates, retrieval.top_k
                )
            selected = candidates[: retrieval.top_k]
            latencies.append((time.perf_counter() - started) * 1000)

            if question.get("expect_no_results"):
                # Scored inverted: returning nothing is the correct behaviour, so
                # an empty result set counts as a hit.
                per_question.append(([not selected], 1))
                continue

            relevance = [_is_relevant(chunk, question) for chunk in selected]
            per_question.append((relevance, max(1, sum(relevance))))

    return aggregate(name, per_question, latencies, settings.retrieval.top_k)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-email", required=True, help="Whose corpus to search")
    parser.add_argument("--golden-set", type=pathlib.Path, default=GOLDEN_SET)
    args = parser.parse_args()

    settings = get_settings()
    payload = json.loads(args.golden_set.read_text())
    questions = payload["questions"]

    async with SessionFactory() as session:
        user = (
            await session.execute(select(User).where(User.email == args.user_email.lower()))
        ).scalar_one_or_none()

    if user is None:
        raise SystemExit(f"No user with email {args.user_email}")

    print(f"\n{len(questions)} questions, top_k={settings.retrieval.top_k}\n")
    print(EvalResult.header())
    print("-" * 96)

    for name, overrides in CONFIGURATIONS:
        result = await evaluate(name, settings, overrides, user.id, questions)
        print(result.as_row())

    print(
        "\nhit@k: any relevant passage retrieved | recall@k: share of relevant found"
        "\nMRR: rank of first hit | nDCG: rank-weighted quality"
        "\nRead down the columns: each row changes one variable from the row above.\n"
    )


if __name__ == "__main__":
    asyncio.run(main())
