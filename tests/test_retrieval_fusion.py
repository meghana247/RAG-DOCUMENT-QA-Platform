"""Pure ranking logic: relevance floor and MMR.

Exercised without a database — these are the parts most likely to silently
degrade answer quality, so they are tested directly.
"""
import uuid

from app.rag.retrieval import _to_vector_literal, apply_mmr, apply_relevance_floor
from app.rag.types import RetrievedChunk


def _chunk(
    content: str = "text",
    vector: float | None = None,
    keyword: float | None = None,
    embedding: list[float] | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="f.pdf",
        content=content,
        chunk_index=0,
        vector_score=vector,
        keyword_score=keyword,
        embedding=embedding,
    )


class TestRelevanceFloor:
    def test_weak_semantic_matches_are_dropped(self):
        candidates = [_chunk(vector=0.8), _chunk(vector=0.1)]
        kept = apply_relevance_floor(candidates, 0.25)
        assert len(kept) == 1
        assert kept[0].vector_score == 0.8

    def test_lexical_matches_survive_a_low_cosine(self):
        """The reason hybrid search exists.

        A chunk containing the user's exact rare term is relevant even when its
        embedding is distant; applying a cosine floor to it would undo the
        keyword leg's whole contribution.
        """
        candidates = [_chunk(vector=0.05, keyword=0.9)]
        assert len(apply_relevance_floor(candidates, 0.25)) == 1

    def test_everything_can_be_filtered_out(self):
        # An empty result is what triggers the honest refusal path.
        assert apply_relevance_floor([_chunk(vector=0.01)], 0.25) == []

    def test_missing_scores_are_treated_as_zero(self):
        assert apply_relevance_floor([_chunk()], 0.25) == []


class TestMmr:
    def test_near_duplicates_are_not_all_selected(self):
        """Chunk overlap produces passages that are equally relevant and nearly
        identical. Without MMR they fill the top-k and the model sees one fact
        five times; the distinct passage that would have answered the rest of
        the question never reaches the prompt.

        Geometry matters here: the duplicates are equally relevant to the query
        (0.9) but identical to each other, while the distinct passage is just as
        relevant and only 0.81 similar to them. Tested at the production
        lambda of 0.7.
        """
        query = [1.0, 0.0, 0.0]
        duplicates = [
            _chunk(content=f"dup{i}", vector=0.9, embedding=[0.9, 0.436, 0.0])
            for i in range(3)
        ]
        distinct = _chunk(content="other", vector=0.9, embedding=[0.9, 0.0, 0.436])

        selected = apply_mmr(duplicates + [distinct], query, k=2, lambda_=0.7)

        assert len(selected) == 2
        assert "other" in [c.content for c in selected], "MMR kept only redundant passages"

    def test_lambda_one_is_pure_relevance(self):
        query = [1.0, 0.0]
        best = _chunk(content="best", embedding=[1.0, 0.0])
        worse = _chunk(content="worse", embedding=[0.0, 1.0])

        # Deliberately passed least-relevant-first: selection must rank by
        # relevance, not trust the input order.
        selected = apply_mmr([worse, best], query, k=1, lambda_=1.0)
        assert selected[0].content == "best"

    def test_zero_k_returns_nothing(self):
        assert apply_mmr([_chunk(embedding=[1.0, 0.0])], [1.0, 0.0], k=0, lambda_=0.7) == []

    def test_returns_early_when_nothing_to_diversify(self):
        single = [_chunk(embedding=[1.0, 0.0])]
        assert apply_mmr(single, [1.0, 0.0], k=5, lambda_=0.7) == single

    def test_candidates_without_embeddings_are_not_lost(self):
        with_embedding = _chunk(content="a", embedding=[1.0, 0.0])
        without = _chunk(content="b")
        selected = apply_mmr([with_embedding, without], [1.0, 0.0], k=2, lambda_=0.7)
        assert {c.content for c in selected} == {"a", "b"}


def test_vector_literal_is_pgvector_text_format():
    assert _to_vector_literal([0.5, -0.25, 0.0]) == "[0.5,-0.25,0]"
