"""Answer-quality evaluation.

Generates an answer to every golden-set question with the production pipeline,
then grades each one with an LLM judge for correctness against the reference
answer and faithfulness to the retrieved passages.

    GROQ_API_KEY=... python -m eval.answer_eval
"""

from __future__ import annotations

import os

os.environ.setdefault("GENERATION__MAX_RETRIES", "8")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import uuid  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from statistics import mean  # noqa: E402
from typing import Any  # noqa: E402

from groq import AsyncGroq  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.container import Container  # noqa: E402
from app.core.database import SessionFactory  # noqa: E402
from eval.judge import (  # noqa: E402
    CORRECTNESS_SCORE,
    REFUSAL_REFERENCE,
    citations_valid,
    judge,
)
from eval.labels import Question, load_golden_set  # noqa: E402
from eval.run_eval import (  # noqa: E402
    CORPUS_DIR,
    EVAL_DIR,
    GOLDEN_SET,
    PLACEHOLDER_GROQ_KEY,
    index_corpus,
)

DEFAULT_OUTPUT = EVAL_DIR / "results" / "answers.json"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"


async def grade(
    container: Container,
    judge_client: AsyncGroq,
    judge_model: str,
    user_id: uuid.UUID,
    question: Question,
) -> dict[str, Any]:
    async with SessionFactory() as session:
        result = await container.pipeline.answer(session, query=question.question, user_id=user_id)

    verdict = await judge(
        judge_client,
        judge_model,
        question=question.question,
        reference=question.answer or REFUSAL_REFERENCE,
        passages=[chunk.content for chunk in result.sources],
        answer=result.answer,
    )
    return {
        "id": question.id,
        "category": question.category,
        "correct": verdict.correct,
        "faithful": verdict.faithful,
        "citations_valid": (
            citations_valid(result.answer, len(result.sources)) if result.sources else None
        ),
        "reason": verdict.reason,
        "answer": result.answer,
        "sources": [
            {"filename": c.filename, "page": c.page_number, "chunk_index": c.chunk_index}
            for c in result.sources
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    cited = [r["citations_valid"] for r in rows if r["citations_valid"] is not None]

    by_category: dict[str, list[float]] = defaultdict(list)
    for row in answerable:
        by_category[row["category"]].append(CORRECTNESS_SCORE[row["correct"]])

    return {
        "questions": len(rows),
        "correctness": round(mean(CORRECTNESS_SCORE[r["correct"]] for r in answerable), 4),
        "fully_correct": round(mean(r["correct"] == "yes" for r in answerable), 4),
        "faithfulness": round(mean(r["faithful"] for r in rows), 4),
        "citation_validity": round(mean(cited), 4) if cited else None,
        "refusal_rate": (
            round(mean(r["correct"] == "yes" for r in unanswerable), 4) if unanswerable else None
        ),
        "correctness_by_category": {
            category: round(mean(scores), 4) for category, scores in sorted(by_category.items())
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=pathlib.Path, default=GOLDEN_SET)
    parser.add_argument("--corpus", type=pathlib.Path, default=CORPUS_DIR)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    settings = get_settings()
    api_key = settings.generation.api_key.get_secret_value()
    if api_key == PLACEHOLDER_GROQ_KEY:
        raise SystemExit("Set GROQ_API_KEY: answer evaluation calls the language model.")

    questions = load_golden_set(args.golden_set)
    container = Container(settings)
    judge_client = AsyncGroq(api_key=api_key, max_retries=8)

    try:
        await container.startup()
        user_id = await index_corpus(container.pipeline, settings, args.corpus, reset=False)

        rows = []
        for number, question in enumerate(questions, start=1):
            row = await grade(container, judge_client, args.judge_model, user_id, question)
            rows.append(row)
            print(f"[{number:>2}/{len(questions)}] {row['correct']:<7} {question.id}")
    finally:
        await container.shutdown()
        await judge_client.close()

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generation_model": settings.generation.model,
        "judge_model": args.judge_model,
        "summary": summary,
        "questions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
