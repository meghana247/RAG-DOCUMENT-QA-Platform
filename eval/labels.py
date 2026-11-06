"""Golden-set loading and relevance judgement.

Kept free of database and model code so the judging rules — the part that decides
every number the evaluation reports — can be unit-tested directly.

A question carries *evidence items*, each a fact a correct answer needs. A chunk
satisfies an item when it comes from the item's document (if one is named) and
contains every listed substring. An item may instead list `any_of` alternatives,
for a fact the corpus states in more than one place or wording (a definition in
prose and again in a summary table, say); any one alternative satisfies it. Relevance is judged per evidence item, not per
chunk: chunks overlap, so one fact often appears in two neighbouring chunks, and
counting both as separate "relevant passages" would penalise MMR for doing its job
of dropping the near-duplicate.
"""

from __future__ import annotations

import json
import pathlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE = re.compile(r"\s+")

CATEGORIES = frozenset(
    {"exact-identifier", "acronym", "definition", "paraphrase", "multi-hop", "unanswerable"}
)


def normalize_for_match(text: str) -> str:
    """Case-, whitespace- and ligature-insensitive form used for label matching.

    PDF extraction leaves ligatures ("ﬁ") and stray spaces inside words
    ("c onfidently"). Removing all whitespace after NFKC folding makes a label
    survive both, at a negligible risk of matching across a word boundary.
    """
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text)).lower()


@dataclass(frozen=True)
class Match:
    contains: tuple[str, ...]
    document: str | None = None

    def matches(self, content: str, filename: str) -> bool:
        if self.document is not None and filename != self.document:
            return False
        haystack = normalize_for_match(content)
        return all(normalize_for_match(needle) in haystack for needle in self.contains)


@dataclass(frozen=True)
class Evidence:
    """One fact a correct answer needs; satisfied by any of its alternatives."""

    alternatives: tuple[Match, ...]

    def matches(self, content: str, filename: str) -> bool:
        return any(option.matches(content, filename) for option in self.alternatives)

    def describe(self) -> str:
        return " | ".join(str(list(option.contains)) for option in self.alternatives)


def _parse_match(raw: dict[str, Any]) -> Match:
    return Match(contains=tuple(raw["contains"]), document=raw.get("document"))


def _parse_evidence(raw: dict[str, Any]) -> Evidence:
    options = raw.get("any_of", [raw])
    return Evidence(alternatives=tuple(_parse_match(option) for option in options))


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    answer: str | None = None

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_golden_set(path: pathlib.Path) -> list[Question]:
    payload: dict[str, Any] = json.loads(path.read_text())
    questions: list[Question] = []
    seen: set[str] = set()

    for raw in payload["questions"]:
        qid = raw["id"]
        if qid in seen:
            raise ValueError(f"Duplicate question id {qid!r}")
        seen.add(qid)

        category = raw["category"]
        if category not in CATEGORIES:
            raise ValueError(f"{qid}: unknown category {category!r}")

        evidence = tuple(_parse_evidence(item) for item in raw.get("evidence", []))
        if any(
            not item.alternatives or any(not option.contains for option in item.alternatives)
            for item in evidence
        ):
            raise ValueError(f"{qid}: an evidence item has no substrings to match")
        if (category == "unanswerable") != (not evidence):
            raise ValueError(
                f"{qid}: unanswerable questions must have no evidence, and all others must have some"
            )

        answer = raw.get("answer")
        if evidence and not answer:
            raise ValueError(f"{qid}: answerable questions need a reference answer")

        questions.append(Question(qid, category, raw["question"], evidence, answer))

    return questions


def judge_ranking(ranked: list[tuple[str, str]], question: Question) -> list[bool]:
    """Mark each ranked (content, filename) as relevant or not, in rank order.

    A position is relevant when it covers an evidence item no earlier position
    has covered. That keeps every metric bounded by the number of evidence items,
    so a perfect retrieval scores exactly 1.0 and duplicates earn nothing.
    """
    covered: set[int] = set()
    relevance: list[bool] = []
    for content, filename in ranked:
        new = {
            index
            for index, item in enumerate(question.evidence)
            if index not in covered and item.matches(content, filename)
        }
        covered |= new
        relevance.append(bool(new))
    return relevance


def unmatched_evidence(questions: list[Question], corpus: list[tuple[str, str]]) -> list[str]:
    """Evidence items that no chunk in the indexed corpus satisfies.

    A label like that can never be retrieved, so it would score as a retrieval
    failure when it is really a labelling (or chunking) error. The evaluation
    refuses to run rather than report a number built on it.
    """
    problems = []
    for question in questions:
        for index, item in enumerate(question.evidence):
            if not any(item.matches(content, filename) for content, filename in corpus):
                problems.append(f"{question.id} evidence[{index}] {item.describe()}")
    return problems
