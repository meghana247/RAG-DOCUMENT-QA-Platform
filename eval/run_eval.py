"""Retrieval evaluation against a fixed corpus and a labelled golden set.

    python -m eval.run_eval                # index eval/corpus if needed, then score
    python -m eval.run_eval --reingest     # rebuild the index first (after chunking changes)

Needs only a migrated Postgres with pgvector (DATABASE_URL). It indexes the PDFs in
eval/corpus under a dedicated evaluation user through the real ingestion path,
checks that every label in the golden set is satisfiable by some indexed chunk,
then runs each retrieval configuration through `RagPipeline.retrieve` — the same
code production calls — and writes a JSON report.

No LLM is called. Retrieval is scored on its own, because a generator can only
answer from the passages it is given: a retrieval miss caps answer quality no
matter how good the model is.

Read the table down the rows: each configuration changes one thing from the row
above, so a difference between adjacent rows has one cause.
"""

from __future__ import annotations

import os

# Retrieval never calls the LLM or issues tokens, but Settings validates these
# at import time. Placeholders let the evaluation run in CI without real secrets.
os.environ.setdefault("GROQ_API_KEY", "eval-placeholder-not-used")
os.environ.setdefault("SECURITY__SECRET_KEY", "eval-placeholder-secret-key-long-enough-to-validate")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import mimetypes  # noqa: E402
import pathlib  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from typing import Any  # noqa: E402

from sqlalchemy import delete, select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.database import SessionFactory  # noqa: E402
from app.models.document import Document, DocumentChunk, DocumentStatus  # noqa: E402
from app.models.user import User  # noqa: E402
from app.rag.chunking import RecursiveChunker  # noqa: E402
from app.rag.contextualizer import QueryContextualizer  # noqa: E402
from app.rag.embeddings import SentenceTransformerEmbedder  # noqa: E402
from app.rag.generation import GroqGenerator, build_client  # noqa: E402
from app.rag.pipeline import RagPipeline  # noqa: E402
from app.rag.reranking import CrossEncoderReranker, NoOpReranker  # noqa: E402
from app.rag.retrieval import HybridRetriever  # noqa: E402
from app.repositories.document_repository import DocumentRepository  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from eval.labels import Question, judge_ranking, load_golden_set, unmatched_evidence  # noqa: E402
from eval.metrics import EvalResult, aggregate  # noqa: E402

EVAL_DIR = pathlib.Path(__file__).parent
GOLDEN_SET = EVAL_DIR / "golden_set.json"
CORPUS_DIR = EVAL_DIR / "corpus"
DEFAULT_OUTPUT = EVAL_DIR / "results" / "retrieval.json"

# A reserved, non-routable domain: this account can never receive mail or log in
# (its password hash is not a valid bcrypt string).
EVAL_USER_EMAIL = "retrieval-eval@eval.invalid"

# Each row isolates one variable against the row above it.
CONFIGURATIONS: list[tuple[str, dict[str, Any]]] = [
    ("vector only", {"strategy": "vector", "mmr_enabled": False, "rerank_enabled": False}),
    ("keyword only", {"strategy": "keyword", "mmr_enabled": False, "rerank_enabled": False}),
    ("hybrid (RRF)", {"strategy": "hybrid", "mmr_enabled": False, "rerank_enabled": False}),
    ("hybrid + MMR", {"strategy": "hybrid", "mmr_enabled": True, "rerank_enabled": False}),
    ("hybrid + MMR + rerank", {"strategy": "hybrid", "mmr_enabled": True, "rerank_enabled": True}),
]

logger = logging.getLogger("eval")


# --------------------------------------------------------------------- corpus


def corpus_files(corpus_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}
    )


async def _eval_user(session: AsyncSession, *, reset: bool) -> uuid.UUID:
    existing = (
        await session.execute(select(User).where(User.email == EVAL_USER_EMAIL))
    ).scalar_one_or_none()

    if existing is not None and reset:
        # Documents and chunks go with the user via ON DELETE CASCADE.
        await session.execute(delete(User).where(User.id == existing.id))
        await session.commit()
        existing = None

    if existing is not None:
        return existing.id

    user = User(email=EVAL_USER_EMAIL, password_hash="!disabled", role="user", is_active=False)
    session.add(user)
    await session.commit()
    return user.id


async def index_corpus(
    pipeline: RagPipeline, settings: Settings, corpus_dir: pathlib.Path, *, reset: bool
) -> uuid.UUID:
    """Ingest every corpus file through DocumentService, skipping ones already indexed.

    Ingestion is content-addressed, so re-running is cheap. Pass reset after
    changing chunking or the embedding model, otherwise the old chunks are scored.
    """
    files = corpus_files(corpus_dir)
    if not files:
        raise SystemExit(f"No corpus files found in {corpus_dir}")

    async with SessionFactory() as session:
        user_id = await _eval_user(session, reset=reset)
        service = DocumentService(DocumentRepository(session), pipeline, settings)

        for path in files:
            content = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            document, _ = await service.register_upload(
                user_id=user_id, filename=path.name, content_type=content_type, content=content
            )
            if document.status == DocumentStatus.READY.value:
                continue
            started = time.perf_counter()
            document = await service.process_document(document.id, content)
            print(
                f"  indexed {path.name}: {document.total_chunks} chunks, "
                f"{document.page_count} pages in {time.perf_counter() - started:.1f}s"
            )

        # Documents left behind from an older corpus would add distractors that
        # the golden set knows nothing about.
        stale = (
            (
                await session.execute(
                    select(Document.filename).where(
                        Document.user_id == user_id,
                        Document.filename.not_in([p.name for p in files]),
                    )
                )
            )
            .scalars()
            .all()
        )
        if stale:
            raise SystemExit(
                f"Evaluation index holds documents not in the corpus: {stale}. Re-run with --reingest."
            )

    return user_id


async def _indexed_chunks(user_id: uuid.UUID) -> list[tuple[str, str]]:  # (content, filename)
    async with SessionFactory() as session:
        rows = await session.execute(
            select(DocumentChunk.content, Document.filename)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.user_id == user_id)
        )
        return list(rows.tuples())


# ------------------------------------------------------------------ evaluation


def _pipeline_for(
    settings: Settings,
    overrides: dict[str, Any],
    *,
    embedder: SentenceTransformerEmbedder,
    rerankers: dict[str, CrossEncoderReranker],
    shared: dict[str, Any],
) -> tuple[RagPipeline, Settings]:
    retrieval = settings.retrieval.model_copy(update=overrides)
    variant = settings.model_copy(update={"retrieval": retrieval})

    if retrieval.rerank_enabled:
        # One cross-encoder per distinct model, loaded once and reused.
        reranker: CrossEncoderReranker | NoOpReranker = rerankers.setdefault(
            retrieval.rerank_model, CrossEncoderReranker(retrieval)
        )
    else:
        reranker = NoOpReranker()

    pipeline = RagPipeline(
        settings=variant,
        embedder=embedder,
        retriever=HybridRetriever(retrieval),
        reranker=reranker,
        generator=shared["generator"],
        contextualizer=shared["contextualizer"],
        chunker=shared["chunker"],
    )
    return pipeline, variant


async def evaluate(
    name: str,
    pipeline: RagPipeline,
    settings: Settings,
    user_id: uuid.UUID,
    questions: list[Question],
) -> tuple[EvalResult, list[dict[str, Any]]]:
    top_k = settings.retrieval.top_k
    per_question: list[tuple[list[bool], int]] = []
    refusals: list[bool] = []
    latencies: list[float] = []
    details: list[dict[str, Any]] = []

    async with SessionFactory() as session:
        for question in questions:
            started = time.perf_counter()
            selected = await pipeline.retrieve(session, query=question.question, user_id=user_id)
            latencies.append((time.perf_counter() - started) * 1000)
            # End the transaction so SET LOCAL hnsw.ef_search cannot carry over.
            await session.rollback()

            ranked = [(chunk.content, chunk.filename) for chunk in selected]
            if question.answerable:
                relevance = judge_ranking(ranked, question)
                per_question.append((relevance, len(question.evidence)))
                outcome: dict[str, Any] = {
                    "hit": any(relevance),
                    "evidence_found": sum(relevance),
                    "evidence_total": len(question.evidence),
                    "first_relevant_rank": next(
                        (i for i, r in enumerate(relevance, start=1) if r), None
                    ),
                }
            else:
                refused = not selected
                refusals.append(refused)
                outcome = {"refused": refused}

            details.append(
                {
                    "id": question.id,
                    "category": question.category,
                    **outcome,
                    "retrieved": [
                        {
                            "filename": c.filename,
                            "chunk_index": c.chunk_index,
                            "page": c.page_number,
                        }
                        for c in selected
                    ],
                }
            )

    result = aggregate(name, per_question, latencies, top_k, refusals)
    return result, details


def _by_category(details: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        groups[row["category"]].append(row)

    summary = {}
    for category, rows in sorted(groups.items()):
        if category == "unanswerable":
            summary[category] = {
                "questions": len(rows),
                "refusal_rate": round(sum(r["refused"] for r in rows) / len(rows), 4),
            }
        else:
            summary[category] = {
                "questions": len(rows),
                "hit_rate": round(sum(r["hit"] for r in rows) / len(rows), 4),
                "recall": round(
                    sum(r["evidence_found"] / r["evidence_total"] for r in rows) / len(rows), 4
                ),
            }
    return summary


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=EVAL_DIR,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ------------------------------------------------------------------------ main


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--golden-set", type=pathlib.Path, default=GOLDEN_SET)
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS_DIR)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reingest", action="store_true", help="Drop and rebuild the evaluation index first"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    questions = load_golden_set(args.golden_set)

    embedder = SentenceTransformerEmbedder(settings.embedding)
    await embedder.warmup()
    groq_client = build_client(settings.generation)  # never called; satisfies the constructor
    shared = {
        "generator": GroqGenerator(groq_client, settings.generation),
        "contextualizer": QueryContextualizer(groq_client, settings.generation),
        "chunker": RecursiveChunker(settings.chunking),
    }
    rerankers: dict[str, CrossEncoderReranker] = {}

    try:
        ingest_pipeline, _ = _pipeline_for(
            settings,
            {"rerank_enabled": False},
            embedder=embedder,
            rerankers=rerankers,
            shared=shared,
        )
        print(f"Indexing corpus from {args.corpus} ...")
        user_id = await index_corpus(ingest_pipeline, settings, args.corpus, reset=args.reingest)

        corpus = await _indexed_chunks(user_id)
        if problems := unmatched_evidence(questions, corpus):
            raise SystemExit(
                "These golden-set labels match no indexed chunk, so they could never be "
                "retrieved. Fix the label (or the chunking) before trusting any score:\n  "
                + "\n  ".join(problems)
            )

        # Embed every question once up front so the first configuration does not
        # pay the embedding cost and later ones read it from cache.
        for question in questions:
            await embedder.embed_query(question.question)

        answerable = sum(q.answerable for q in questions)
        print(
            f"\n{len(questions)} questions ({answerable} answerable, "
            f"{len(questions) - answerable} unanswerable) over {len(corpus)} chunks, "
            f"top_k={settings.retrieval.top_k}\n"
        )
        print(EvalResult.header())
        print("-" * 98)

        report_rows = []
        for name, overrides in CONFIGURATIONS:
            pipeline, variant = _pipeline_for(
                settings, overrides, embedder=embedder, rerankers=rerankers, shared=shared
            )
            result, details = await evaluate(name, pipeline, variant, user_id, questions)
            print(result.as_row())
            report_rows.append(
                {
                    "configuration": name,
                    "overrides": overrides,
                    "metrics": result.as_dict(),
                    "by_category": _by_category(details),
                    "questions": details,
                }
            )
    finally:
        await groq_client.close()

    print(
        "\nhit@k: question had a relevant passage in the top k | recall@k: share of a question's"
        "\nevidence found | MRR: 1/rank of first relevant passage | nDCG: rank-weighted quality"
        "\nrefusal: unanswerable questions where retrieval correctly returned nothing"
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "golden_set": args.golden_set.name,
        "corpus": [p.name for p in corpus_files(args.corpus)],
        "indexed_chunks": len(corpus),
        "questions": len(questions),
        "settings": {
            "embedding_model": settings.embedding.model_name,
            "rerank_model": settings.retrieval.rerank_model,
            "chunking": settings.chunking.model_dump(),
            "retrieval": settings.retrieval.model_dump(
                exclude={"strategy", "mmr_enabled", "rerank_enabled", "rerank_model"}
            ),
        },
        "results": report_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
