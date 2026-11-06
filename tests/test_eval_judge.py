import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from eval.check import failures
from eval.judge import Verdict, citations_valid, judge
from eval.metrics import summarize_answers


def test_verdict_rejects_unknown_grades():
    with pytest.raises(ValidationError):
        Verdict(correct="maybe", faithful=True, reason="")


async def test_judge_numbers_passages_like_the_generator():
    seen = {}

    def fake_llm(inputs):
        seen.update(inputs)
        return Verdict(correct="yes", faithful=True, reason="ok")

    verdict = await judge(
        RunnableLambda(fake_llm),
        question="q",
        reference="r",
        passages=["first", "second"],
        answer="a [1]",
    )
    assert verdict.correct == "yes"
    assert seen["passages"] == "[1] first\n\n[2] second"


@pytest.mark.parametrize(
    ("answer", "sources", "expected"),
    [
        ("Tiers range from Partial [1] to Adaptive [2][3].", 3, True),
        ("Cites a passage that was never shown [4].", 3, False),
        ("No citations at all.", 3, False),
    ],
)
def test_citations_valid(answer, sources, expected):
    assert citations_valid(answer, sources) is expected


def test_summarize_separates_refusals_from_correctness():
    rows = [
        {"category": "definition", "correct": "yes", "faithful": True, "citations_valid": True},
        {
            "category": "paraphrase",
            "correct": "partial",
            "faithful": False,
            "citations_valid": False,
        },
        {"category": "unanswerable", "correct": "yes", "faithful": True, "citations_valid": None},
    ]
    summary = summarize_answers(rows)
    assert summary["correctness"] == 0.75
    assert summary["fully_correct"] == 0.5
    assert summary["citation_validity"] == 0.5
    assert summary["refusal_rate"] == 1.0
    assert summary["correctness_by_category"] == {"definition": 1.0, "paraphrase": 0.5}


def test_threshold_failures():
    assert failures({"mrr": 0.54, "ndcg": 0.4}, {"mrr": 0.5, "ndcg": 0.5}) == ["ndcg 0.4 < 0.5"]
    assert failures({}, {"mrr": 0.5}) == ["mrr None < 0.5"]
