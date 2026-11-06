"""LLM-as-judge grading of generated answers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, cast

from groq import AsyncGroq

Correctness = Literal["yes", "partial", "no"]

CORRECTNESS_SCORE: dict[str, float] = {"yes": 1.0, "partial": 0.5, "no": 0.0}

REFUSAL_REFERENCE = (
    "The documents do not contain this information. A correct answer says so "
    "and does not answer from general knowledge."
)

_PROMPT = """You grade answers produced by a document question-answering system.

Question: {question}

Reference answer: {reference}

Passages the system was given:
{passages}

System answer: {answer}

Return a JSON object with exactly these keys:
- "correct": "yes" if the system answer conveys the reference answer, "partial" if it \
is incomplete or partly wrong, "no" otherwise. Extra detail that is supported by the \
passages does not reduce the grade.
- "faithful": true if every factual claim in the system answer is supported by the \
passages, false otherwise. An answer that only says the information is not available \
is faithful.
- "reason": one short sentence explaining the grade."""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Verdict:
    correct: Correctness
    faithful: bool
    reason: str


def parse_verdict(raw: str) -> Verdict:
    data = json.loads(raw)
    correct = str(data["correct"]).lower()
    if correct not in CORRECTNESS_SCORE or not isinstance(data["faithful"], bool):
        raise ValueError(f"Malformed verdict: {raw}")
    return Verdict(cast(Correctness, correct), data["faithful"], str(data.get("reason", "")))


def citations_valid(answer: str, source_count: int) -> bool:
    cited = {int(n) for n in _CITATION.findall(answer)}
    return bool(cited) and all(1 <= n <= source_count for n in cited)


async def judge(
    client: AsyncGroq,
    model: str,
    *,
    question: str,
    reference: str,
    passages: list[str],
    answer: str,
) -> Verdict:
    rendered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": _PROMPT.format(
                    question=question,
                    reference=reference,
                    passages=rendered or "(none)",
                    answer=answer,
                ),
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return parse_verdict(response.choices[0].message.content or "")
