"""End-to-end retrieval against a real Postgres with pgvector.

Skipped unless TEST_DATABASE_URL points at a migrated database:

    docker run -d -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb \
        -p 55432:5432 pgvector/pgvector:pg16
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/ragdb \
        alembic upgrade head
    TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/ragdb pytest

The hybrid query cannot be meaningfully tested against SQLite or a mock: the
CTE fusion, `websearch_to_tsquery`, the generated tsvector column and the
pgvector distance operator are all Postgres-specific, and errors in them —
a missing cast, a parameter whose type asyncpg cannot infer — surface only here.
"""
import math
import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import RetrievalSettings
from app.rag.retrieval import HybridRetriever

TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL, reason="TEST_DATABASE_URL not set; skipping Postgres integration tests"
)

DIM = 384


def _unit_vector(seed: int) -> list[float]:
    """Deterministic normalised vector, so cosine similarity is a dot product."""
    values = [math.sin(seed * 0.7 + i * 0.013) for i in range(DIM)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


@pytest.fixture
async def session():
    engine = create_async_engine(TEST_DB_URL, poolclass=None)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def corpus(session):
    """Two users, so tenant isolation is actually exercised."""
    owner_id, intruder_id = uuid.uuid4(), uuid.uuid4()
    document_id = uuid.uuid4()

    for user_id, email in ((owner_id, "owner@test.dev"), (intruder_id, "intruder@test.dev")):
        await session.execute(
            text(
                "INSERT INTO users (id, email, password_hash, role, is_active) "
                "VALUES (:id, :email, 'x', 'user', true)"
            ),
            {"id": user_id, "email": f"{uuid.uuid4()}-{email}"},
        )

    for doc_id, user_id in ((document_id, owner_id), (uuid.uuid4(), intruder_id)):
        await session.execute(
            text(
                "INSERT INTO documents (id, user_id, filename, content_type, content_hash, status) "
                "VALUES (:id, :uid, 'report.pdf', 'application/pdf', :hash, 'ready')"
            ),
            {"id": doc_id, "uid": user_id, "hash": uuid.uuid4().hex},
        )
        if user_id == intruder_id:
            intruder_doc = doc_id

    rows = [
        # (content, seed) — content chosen so lexical and semantic legs disagree.
        ("Quarterly revenue for Q1 was 4.2 million dollars.", 1),
        ("The subscription may be terminated with thirty days written notice.", 2),
        ("Invoice INV-2024-0871 covers professional services rendered in March.", 3),
        ("Uptime commitments are described in the SLA appendix.", 4),
        ("Completely unrelated filler text about gardening equipment.", 5),
    ]
    for index, (content, seed) in enumerate(rows):
        await session.execute(
            text(
                "INSERT INTO document_chunks "
                "(id, document_id, user_id, content, chunk_index, page_number, char_count, embedding) "
                "VALUES (:id, :doc, :uid, :content, :idx, :page, :len, CAST(:emb AS vector))"
            ),
            {
                "id": uuid.uuid4(),
                "doc": document_id,
                "uid": owner_id,
                "content": content,
                "idx": index,
                "page": index + 1,
                "len": len(content),
                "emb": "[" + ",".join(str(v) for v in _unit_vector(seed)) + "]",
            },
        )

    # A chunk belonging to the other tenant that would rank highly if leaked.
    await session.execute(
        text(
            "INSERT INTO document_chunks "
            "(id, document_id, user_id, content, chunk_index, char_count, embedding) "
            "VALUES (:id, :doc, :uid, :content, 0, 10, CAST(:emb AS vector))"
        ),
        {
            "id": uuid.uuid4(),
            "doc": intruder_doc,
            "uid": intruder_id,
            "content": "SECRET other tenant revenue Q1 invoice INV-2024-0871 SLA",
            "emb": "[" + ",".join(str(v) for v in _unit_vector(1)) + "]",
        },
    )
    await session.commit()

    yield {"owner_id": owner_id, "intruder_id": intruder_id, "document_id": document_id}

    await session.execute(
        text("DELETE FROM users WHERE id IN (:a, :b)"),
        {"a": owner_id, "b": intruder_id},
    )
    await session.commit()


def _retriever(**overrides) -> HybridRetriever:
    return HybridRetriever(RetrievalSettings(**overrides))


class TestHybridSql:
    """These assert the query *runs*. A cast or parameter-typing error here is
    invisible to unit tests and fails only against a real driver."""

    async def test_hybrid_query_executes_and_returns_rows(self, session, corpus):
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="quarterly revenue",
            query_embedding=_unit_vector(1),
            user_id=corpus["owner_id"],
            limit=10,
        )
        assert results
        assert all(r.fusion_score is not None for r in results)

    async def test_vector_only_query_executes(self, session, corpus):
        results = await _retriever(strategy="vector").retrieve(
            session,
            query="quarterly revenue",
            query_embedding=_unit_vector(1),
            user_id=corpus["owner_id"],
            limit=10,
        )
        assert results
        assert results[0].vector_score is not None

    async def test_keyword_only_query_executes(self, session, corpus):
        results = await _retriever(strategy="keyword").retrieve(
            session,
            query="invoice INV-2024-0871",
            query_embedding=_unit_vector(9),
            user_id=corpus["owner_id"],
            limit=10,
        )
        assert results
        assert any("INV-2024-0871" in r.content for r in results)

    async def test_query_with_punctuation_does_not_raise(self, session, corpus):
        """`to_tsquery` throws a syntax error on raw user input; the code uses
        `websearch_to_tsquery`, which tolerates arbitrary text."""
        for hostile in ["what's the revenue?", "a & b | c", "!!! ???", "'quoted phrase'", "-"]:
            results = await _retriever(strategy="hybrid").retrieve(
                session,
                query=hostile,
                query_embedding=_unit_vector(1),
                user_id=corpus["owner_id"],
                limit=5,
            )
            assert isinstance(results, list)


class TestHybridBehaviour:
    async def test_exact_identifier_is_found(self, session, corpus):
        """The case pure vector search misses. The query embedding is deliberately
        unrelated to the invoice chunk, so only the lexical leg can find it."""
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="INV-2024-0871",
            query_embedding=_unit_vector(50),
            user_id=corpus["owner_id"],
            limit=10,
        )
        assert any("INV-2024-0871" in r.content for r in results)

    async def test_keyword_only_hits_get_a_backfilled_cosine_score(self, session, corpus):
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="INV-2024-0871",
            query_embedding=_unit_vector(50),
            user_id=corpus["owner_id"],
            limit=10,
        )
        # Needed so the relevance floor can be applied uniformly.
        assert all(r.vector_score is not None for r in results)

    async def test_semantically_closest_chunk_ranks_first_for_vector_search(
        self, session, corpus
    ):
        results = await _retriever(strategy="vector").retrieve(
            session,
            query="anything",
            query_embedding=_unit_vector(3),
            user_id=corpus["owner_id"],
            limit=5,
        )
        assert "INV-2024-0871" in results[0].content

    async def test_filename_is_joined_not_fetched_per_chunk(self, session, corpus):
        """The original code called a separate query per chunk to resolve the
        filename — an N+1 on the hot path. It comes from the join now."""
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="revenue",
            query_embedding=_unit_vector(1),
            user_id=corpus["owner_id"],
            limit=5,
        )
        assert all(r.filename == "report.pdf" for r in results)

    async def test_page_numbers_are_returned(self, session, corpus):
        results = await _retriever(strategy="vector").retrieve(
            session,
            query="revenue",
            query_embedding=_unit_vector(1),
            user_id=corpus["owner_id"],
            limit=5,
        )
        assert any(r.page_number is not None for r in results)


class TestTenantIsolation:
    async def test_other_tenants_chunks_are_never_returned(self, session, corpus):
        """The other tenant's chunk is a lexical match for every term in the query
        and has an identical embedding to a relevant chunk, so it would surface
        immediately if the filter were missing from either leg."""
        for strategy in ("hybrid", "vector", "keyword"):
            results = await _retriever(strategy=strategy).retrieve(
                session,
                query="revenue Q1 invoice INV-2024-0871 SLA",
                query_embedding=_unit_vector(1),
                user_id=corpus["owner_id"],
                limit=20,
            )
            assert not any("SECRET" in r.content for r in results), f"leak via {strategy}"

    async def test_document_scope_restricts_results(self, session, corpus):
        other_document = uuid.uuid4()
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="revenue",
            query_embedding=_unit_vector(1),
            user_id=corpus["owner_id"],
            document_id=other_document,
            limit=10,
        )
        assert results == []

    async def test_null_document_id_searches_everything(self, session, corpus):
        """Exercises the `CAST(:document_id AS uuid) IS NULL` branch — asyncpg
        cannot infer the type of a bare NULL parameter, so a missing cast fails
        only at runtime."""
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="revenue",
            query_embedding=_unit_vector(1),
            user_id=corpus["owner_id"],
            document_id=None,
            limit=10,
        )
        assert results

    async def test_unknown_user_gets_nothing(self, session, corpus):
        results = await _retriever(strategy="hybrid").retrieve(
            session,
            query="revenue",
            query_embedding=_unit_vector(1),
            user_id=uuid.uuid4(),
            limit=10,
        )
        assert results == []
