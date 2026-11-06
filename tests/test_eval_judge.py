import pytest

from eval.answer_eval import summarize
from eval.judge import citations_valid, parse_verdict


def test_parse_verdict():
    verdict = parse_verdict('{"correct": "Partial", "faithful": true, "reason": "misses a tier"}')
    assert (verdict.correct, verdict.faithful) == ("partial", True)


@pytest.mark.parametrize(
    "raw",
    ['{"correct": "maybe", "faithful": true}', '{"correct": "yes", "faithful": "yes"}', "not json"],
)
def test_parse_verdict_rejects_malformed_output(raw):
    with pytest.raises(ValueError):
        parse_verdict(raw)


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
    summary = summarize(rows)
    assert summary["correctness"] == 0.75
    assert summary["fully_correct"] == 0.5
    assert summary["citation_validity"] == 0.5
    assert summary["refusal_rate"] == 1.0
    assert summary["correctness_by_category"] == {"definition": 1.0, "paraphrase": 0.5}
