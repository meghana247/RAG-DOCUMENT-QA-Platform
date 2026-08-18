"""Metric correctness. Wrong metrics are worse than none — they justify bad changes."""
from eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_reciprocal_rank_rewards_earlier_hits():
    assert reciprocal_rank([True, False, False]) == 1.0
    assert reciprocal_rank([False, True, False]) == 0.5
    assert reciprocal_rank([False, False, False]) == 0.0


def test_recall_counts_only_the_first_k():
    assert recall_at_k([False, False, True], total_relevant=1, k=2) == 0.0
    assert recall_at_k([False, False, True], total_relevant=1, k=3) == 1.0


def test_precision_is_hits_over_k():
    assert precision_at_k([True, False, True, False], k=4) == 0.5


def test_ndcg_is_rank_sensitive_where_recall_is_not():
    early = [True, False, False, False]
    late = [False, False, False, True]

    # Recall cannot tell these apart; nDCG can, which is why it is the metric to
    # watch when evaluating a reranker.
    assert recall_at_k(early, 1, 4) == recall_at_k(late, 1, 4)
    assert ndcg_at_k(early, 1, 4) > ndcg_at_k(late, 1, 4)


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([True, True, False], total_relevant=2, k=3) == 1.0


def test_empty_and_degenerate_inputs():
    assert recall_at_k([], 0, 5) == 0.0
    assert ndcg_at_k([], 0, 5) == 0.0
    assert precision_at_k([True], 0) == 0.0
