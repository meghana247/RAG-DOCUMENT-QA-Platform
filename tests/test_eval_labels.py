"""Relevance judging for the retrieval evaluation.

These rules decide every number the evaluation reports, so they are tested as
carefully as the metrics themselves.
"""

import json
import pathlib

import pytest

from eval.labels import (
    Evidence,
    Match,
    Question,
    judge_ranking,
    load_golden_set,
    normalize_for_match,
    unmatched_evidence,
)
from eval.metrics import aggregate

GOLDEN_SET = pathlib.Path(__file__).parent.parent / "eval" / "golden_set.json"


def _evidence(*needles: str, document: str | None = None) -> Evidence:
    return Evidence(alternatives=(Match(contains=needles, document=document),))


def _question(*evidence: Evidence, category: str = "definition") -> Question:
    return Question(id="q", category=category, question="?", evidence=evidence)


class TestMatching:
    def test_ignores_case_whitespace_and_ligatures(self):
        # pypdf output: a ligature, a stray space inside a word, a line break.
        extracted = "GAI systems generate and c onﬁdently\npresent erroneous content"
        assert (
            normalize_for_match(extracted).find(
                normalize_for_match("Confidently present erroneous")
            )
            >= 0
        )

    def test_all_substrings_must_be_present(self):
        item = _evidence("Partial (Tier 1)", "Adaptive (Tier 4)")
        assert item.matches("Partial (Tier 1) ... Adaptive (Tier 4)", "csf.pdf")
        assert not item.matches("Partial (Tier 1) only", "csf.pdf")

    def test_document_restriction(self):
        item = _evidence("GOVERN", document="rmf.pdf")
        assert item.matches("GOVERN function", "rmf.pdf")
        assert not item.matches("GOVERN function", "csf.pdf")

    def test_any_alternative_satisfies_the_item(self):
        item = Evidence(
            alternatives=(
                Match(contains=("test, evaluation, verification, and validation",)),
                Match(contains=("Evaluation, Verification, and Validation (TEVV)",)),
            )
        )
        assert item.matches("... Evaluation, Verification, and Validation (TEVV) tasks", "x")


class TestJudgeRanking:
    def test_duplicate_chunks_for_one_fact_count_once(self):
        # Overlapping chunks repeat a sentence. Crediting both would let a
        # retriever that returns near-duplicates outscore one that diversifies.
        question = _question(_evidence("residual risk"))
        ranked = [("residual risk A", "f"), ("residual risk B", "f"), ("other", "f")]
        assert judge_ranking(ranked, question) == [True, False, False]

    def test_multi_hop_needs_each_fact(self):
        question = _question(_evidence("Current Profile"), _evidence("Target Profile"))
        relevance = judge_ranking([("Current Profile", "f"), ("noise", "f")], question)
        assert relevance == [True, False]
        assert sum(relevance) / len(question.evidence) == 0.5

    def test_one_chunk_can_cover_two_facts(self):
        question = _question(_evidence("Current Profile"), _evidence("Target Profile"))
        assert judge_ranking([("Current Profile and Target Profile", "f")], question) == [True]

    def test_perfect_retrieval_scores_one(self):
        question = _question(_evidence("a"), _evidence("b"))
        relevance = judge_ranking([("a", "f"), ("b", "f")], question)
        result = aggregate("x", [(relevance, len(question.evidence))], [1.0], k=6)
        assert result.recall_at_k == 1.0
        assert result.ndcg == 1.0
        assert result.mrr == 1.0


class TestRefusals:
    def test_unanswerable_questions_are_kept_out_of_ranking_metrics(self):
        result = aggregate("x", [([True], 1)], latencies=[1.0], k=6, refusals=[True, False])
        assert result.hit_rate == 1.0
        assert result.refusal_rate == 0.5
        assert result.unanswerable == 2


class TestGoldenSetFile:
    def test_the_committed_golden_set_loads(self):
        questions = load_golden_set(GOLDEN_SET)
        assert len(questions) >= 30
        assert {q.category for q in questions} >= {"paraphrase", "multi-hop", "unanswerable"}

    def test_unanswerable_must_have_no_evidence(self, tmp_path):
        path = tmp_path / "golden.json"
        path.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "id": "a",
                            "category": "unanswerable",
                            "question": "?",
                            "evidence": [{"contains": ["x"]}],
                        }
                    ]
                }
            )
        )
        with pytest.raises(ValueError, match="unanswerable"):
            load_golden_set(path)

    def test_duplicate_ids_are_rejected(self, tmp_path):
        path = tmp_path / "golden.json"
        row = {
            "id": "a",
            "category": "definition",
            "question": "?",
            "answer": "x",
            "evidence": [{"contains": ["x"]}],
        }
        path.write_text(json.dumps({"questions": [row, row]}))
        with pytest.raises(ValueError, match="Duplicate"):
            load_golden_set(path)


def test_unmatched_evidence_reports_unsatisfiable_labels():
    question = _question(_evidence("present"), _evidence("absent"))
    problems = unmatched_evidence([question], [("present here", "f")])
    assert len(problems) == 1
    assert "absent" in problems[0]
