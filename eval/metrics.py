"""Retrieval metrics.

Tuning retrieval by reading a few answers and forming an impression is how a
change that helps three questions and quietly breaks twenty gets shipped. These
are the standard measures; all take a per-question list of booleans marking which
retrieved positions were relevant, in rank order.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def hit_rate(relevance: list[bool]) -> float:
    """Did anything relevant appear at all? The floor a RAG system must clear —
    the generator cannot answer from context it never received."""
    return 1.0 if any(relevance) else 0.0


def recall_at_k(relevance: list[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(relevance[:k]) / total_relevant


def precision_at_k(relevance: list[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(relevance[:k]) / k


def reciprocal_rank(relevance: list[bool]) -> float:
    """1/rank of the first relevant hit.

    Position matters more than raw presence: models attend unevenly across a
    prompt, so a passage ranked first is worth more than the same passage ranked
    sixth.
    """
    for index, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevance: list[bool], total_relevant: int, k: int) -> float:
    """Normalised discounted cumulative gain — rank-sensitive, unlike recall."""
    dcg = sum(
        (1.0 / math.log2(index + 1))
        for index, is_relevant in enumerate(relevance[:k], start=1)
        if is_relevant
    )
    ideal_hits = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


@dataclass
class EvalResult:
    """Ranking metrics are averaged over answerable questions only.

    Unanswerable questions have nothing to rank, so folding them into hit rate
    or precision would mix two different behaviours into one number. They are
    reported on their own as `refusal_rate`: the share where retrieval correctly
    returned nothing.
    """

    name: str
    questions: int
    hit_rate: float
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    p50_latency_ms: float
    p95_latency_ms: float
    unanswerable: int = 0
    refusal_rate: float = 0.0

    def as_row(self) -> str:
        return (
            f"{self.name:<24} {self.hit_rate:>7.3f} {self.recall_at_k:>9.3f} "
            f"{self.precision_at_k:>8.3f} {self.mrr:>7.3f} {self.ndcg:>7.3f} "
            f"{self.refusal_rate:>8.3f} {self.p50_latency_ms:>8.0f} {self.p95_latency_ms:>8.0f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'configuration':<24} {'hit@k':>7} {'recall@k':>9} "
            f"{'prec@k':>8} {'MRR':>7} {'nDCG':>7} {'refusal':>8} {'p50 ms':>8} {'p95 ms':>8}"
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in asdict(self).items()
        }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def aggregate(
    name: str,
    per_question: list[tuple[list[bool], int]],
    latencies: list[float],
    k: int,
    refusals: list[bool] | None = None,
) -> EvalResult:
    refusals = refusals or []
    refusal_rate = (sum(refusals) / len(refusals)) if refusals else 0.0

    if not per_question:
        return EvalResult(name, 0, 0, 0, 0, 0, 0, 0, 0, len(refusals), refusal_rate)

    count = len(per_question)
    return EvalResult(
        name=name,
        questions=count,
        hit_rate=sum(hit_rate(r) for r, _ in per_question) / count,
        recall_at_k=sum(recall_at_k(r, t, k) for r, t in per_question) / count,
        precision_at_k=sum(precision_at_k(r, k) for r, _ in per_question) / count,
        mrr=sum(reciprocal_rank(r) for r, _ in per_question) / count,
        ndcg=sum(ndcg_at_k(r, t, k) for r, t in per_question) / count,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        unanswerable=len(refusals),
        refusal_rate=refusal_rate,
    )
