# RAG Document Q&A Platform

Multi-tenant retrieval-augmented question answering over uploaded documents.
FastAPI, Postgres + pgvector, hybrid dense/lexical retrieval, and Groq for
generation.

Upload a PDF, DOCX or TXT, ask questions in natural language, get answers grounded
in the document with citations that point at a **page**. Every user's data is
isolated at the query level.

---

## What makes the retrieval good

The naive pipeline — chunk every 500 characters, embed, take the top 5 by cosine
distance — fails in specific, predictable ways. Each stage below exists to fix one
of them.

```
question
   │
   ├─ contextualise ......... rewrite follow-ups into standalone queries
   ├─ embed ................. cached, off the event loop
   │
   ├─ retrieve (one SQL round trip)
   │     ├─ dense leg ....... pgvector HNSW, cosine
   │     ├─ lexical leg ..... Postgres FTS, ts_rank_cd
   │     └─ fuse ............ reciprocal rank fusion
   │
   ├─ relevance floor ....... drop weak matches → honest refusal
   ├─ MMR ................... diversify away near-duplicates
   ├─ rerank ................ cross-encoder over ~20 survivors
   │
   └─ generate .............. numbered citations, bounded context
```

| Stage | Problem it solves |
|---|---|
| **HNSW index** | Without an ANN index every query is a sequential scan computing exact cosine distance over every chunk the user owns. Measured on 20k chunks: **33ms → 17ms**, and the gap grows linearly because the scan is O(n). |
| **Hybrid retrieval** | Embeddings are weak on exact tokens — invoice numbers, error codes, surnames, acronyms — which is precisely what users type. Postgres FTS is strong there and weak on paraphrase. Fusing both recovers each. |
| **Reciprocal Rank Fusion** | Cosine similarity (0–1) and `ts_rank_cd` (unbounded) are not comparable, so blending them with weights needs per-corpus tuning. RRF combines by *rank* and needs none. |
| **Relevance floor** | Top-k always returns k results, so an off-topic question yields the least-bad matches and a confident hallucination. Below-threshold results are dropped; empty context short-circuits to a refusal and skips the LLM call entirely. |
| **MMR** | Chunk overlap guarantees near-duplicates. Without diversification the top-k fills with the same paragraph and the other slots are wasted. |
| **Cross-encoder rerank** | Retrieval scores query and chunk independently. A cross-encoder reads the pair together — too slow for a corpus, ideal over 20 candidates. Degrades gracefully to fusion order if the model is unavailable. |
| **Structure-aware chunking** | Character slicing splits mid-word and mid-sentence, so a chunk's embedding describes two half-thoughts and matches neither. Chunks now land on paragraph/sentence boundaries, never span a page, and carry page and section provenance. |
| **Query contextualisation** | "What about the second one?" carries no retrievable signal. Rewritten into a standalone query using recent turns before retrieval. |

Retrieval is tunable entirely through environment variables — see `.env.example`.
`RETRIEVAL__STRATEGY=vector` reproduces the naive behaviour for comparison.

---

## Measuring changes

Retrieval quality cannot be judged by reading a few answers. `eval/` scores the
pipeline against three public-domain NIST publications (AI RMF 1.0, the Generative
AI Profile, CSF 2.0) and 36 labelled questions: exact identifiers, acronyms,
definitions, paraphrases, multi-hop questions, and unanswerable ones.

```bash
python -m eval.run_eval        # indexes eval/corpus itself; needs only DATABASE_URL
```

| Configuration | hit@6 | recall@6 | MRR | nDCG | refusal | p50 ms |
|---|---|---|---|---|---|---|
| vector only | 0.581 | 0.532 | 0.399 | 0.426 | 0.80 | 11 |
| keyword only | 0.226 | 0.226 | 0.210 | 0.214 | 1.00 | 4 |
| hybrid (RRF) | 0.645 | 0.597 | 0.455 | 0.484 | 0.80 | 10 |
| hybrid + MMR | 0.645 | 0.597 | 0.396 | 0.442 | 0.80 | 13 |
| **hybrid + MMR + rerank** (default) | **0.645** | **0.613** | **0.539** | **0.540** | **0.80** | 569 |

Each row changes one variable from the row above, so a difference has one cause.
Hybrid retrieval beats either leg alone, and the cross-encoder lifts MRR from 0.40
to 0.54 by moving the right passage higher. Keyword-only search is the weakest
link: it requires every question word to match. See [`eval/README.md`](eval/README.md)
for the corpus, labelling rules and analysis.

Run it before and after any retrieval change. A change that improves five
questions and quietly breaks twenty is not an improvement.

---

## Architecture

```
app/
├── core/                  cross-cutting infrastructure
│   ├── config.py          nested settings, validated at startup
│   ├── container.py       composition root — models loaded once
│   ├── database.py        engine, session factory, pooling
│   ├── dependencies.py    FastAPI providers
│   ├── exceptions.py      domain errors → HTTP, one response envelope
│   ├── logging.py         JSON logs with request/user correlation
│   ├── middleware.py      request IDs, access logs, rate limiting
│   └── security.py        hashing, JWT
├── rag/                   the pipeline — no HTTP, no ORM
│   ├── protocols.py       Chunker / Embedder / Retriever / Reranker / Generator
│   ├── types.py           value objects passed between stages
│   ├── extraction.py      PDF/DOCX/TXT → blocks with page + section
│   ├── chunking.py        recursive, boundary-aware
│   ├── embeddings.py      threadpooled, LRU-cached
│   ├── retrieval.py       hybrid SQL, RRF, MMR, relevance floor
│   ├── reranking.py       cross-encoder, optional
│   ├── contextualizer.py  follow-up rewriting
│   ├── generation.py      prompt assembly, citations, streaming
│   └── pipeline.py        orchestration + per-stage timings
├── repositories/          queries only, never commit
├── services/              business logic, owns transactions
├── routers/               HTTP only
├── schemas/               request/response contracts
└── models/                SQLAlchemy tables
migrations/                alembic — schema is never created at startup
tests/                     82 tests; retrieval integration hits real Postgres
eval/                      retrieval metrics + golden set
```

**Dependencies point inward.** Routers depend on services, services on
repositories and the pipeline, the pipeline on protocols. `app/rag/` imports no
HTTP framework and no router; it can be driven from a script, a test, or a queue
worker.

**Stages are protocols, not base classes.** Swapping the local embedder for a
hosted API is one new module implementing `Embedder` plus one line in
`container.py`. A test double is five lines.

**Transactions belong to services.** Repositories flush, never commit, so one
service call spanning several repositories is atomic.

---

## Running it

```bash
cp .env.example .env
# set SECURITY__SECRET_KEY and GROQ_API_KEY

docker compose up --build -d
docker compose exec api alembic upgrade head
```

API at `http://localhost:8000`, docs at `/docs` (disabled when
`ENVIRONMENT=production`).

Migrations are a deploy step, never a startup side effect: `create_all()` at boot
races across replicas, cannot express an index build or a backfill, and offers no
rollback.

```bash
make test     # pytest
make lint     # ruff + mypy
make eval     # score retrieval
make help     # everything else
```

### Local development without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
alembic upgrade head
uvicorn app.main:app --reload
```

### Running the integration tests

The hybrid query cannot be tested against SQLite or a mock — CTE fusion,
`websearch_to_tsquery`, the generated tsvector column and pgvector's distance
operator are all Postgres-specific:

```bash
docker run -d -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ragdb \
    -p 55432:5432 pgvector/pgvector:pg16
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/ragdb \
    alembic upgrade head
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/ragdb \
    pytest
```

---

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/auth/register` | |
| `POST` | `/auth/login` | Returns a bearer token and its expiry |
| `GET` | `/auth/me` | |
| `POST` | `/documents/upload` | **202** — ingestion is asynchronous |
| `GET` | `/documents/{id}` | Poll until `status` is `ready` |
| `GET` | `/documents` | Paginated |
| `DELETE` | `/documents/{id}` | |
| `POST` | `/chat/sessions` | Optionally scoped to one document |
| `POST` | `/chat/sessions/{id}/messages` | Answer + citations + timings |
| `POST` | `/chat/sessions/{id}/messages/stream` | SSE: `sources`, `token`…, `done` |
| `GET` | `/chat/sessions/{id}/messages` | Paginated |
| `GET` | `/admin/users`, `/admin/documents` | Admin role required |
| `GET` | `/health` | Liveness — touches no dependency |
| `GET` | `/health/ready` | Readiness — probes the database |

Upload returns 202 because extraction, chunking and embedding take seconds; doing
that inside the request holds a database transaction open for its duration. The
same bytes uploaded twice are deduplicated by SHA-256 and cost nothing.

Liveness deliberately does not check the database: a liveness probe that fails on
a database blip restarts a healthy process and turns a recoverable outage into a
crash loop.

All errors share one envelope:

```json
{ "error": { "code": "not_found", "message": "Document not found" } }
```

---

## Operational notes

**Scaling.** Each worker loads its own copy of the embedding model, so scale with
replicas and keep `--workers 1`. Embedding and reranking run in a bounded
threadpool — they are CPU-bound, and running them on the event loop stalls every
concurrent request.

**Serverless / external pooler.** Set `DB__POOL_SIZE=0` (NullPool) and
`DB__DISABLE_STATEMENT_CACHE=true` when connecting through pgbouncer, Neon's
pooled endpoint, or from a serverless platform.

**Rate limiting** is in-process, so the effective limit is `replicas × limit`. It
protects one instance from a runaway client; move the counters to Redis before
relying on it for abuse prevention.

**Background ingestion** uses in-process tasks, which do not survive a restart. A
document stuck in `processing` should be re-uploaded — the duplicate-hash path
makes that cheap. Swapping in Celery or arq changes one function,
`_ingest_in_background`.

**Changing the embedding model** requires a migration to resize the `vector`
column and a full re-embed of every chunk. Startup fails fast if the model's
dimension disagrees with `EMBEDDING__DIM` rather than corrupting the index.

**Changing chunk size** requires re-ingesting documents; existing chunks are not
rewritten.

---

## Configuration

Every knob is an environment variable, grouped with `__` nesting
(`RETRIEVAL__TOP_K`, `DB__POOL_SIZE`). Flat legacy names — `DATABASE_URL`,
`GROQ_API_KEY`, `SECRET_KEY` — are still honoured. See `.env.example` for the
annotated list.

Startup fails on a secret key under 32 characters or one still holding a
placeholder, on a sync database driver, and on `candidate_k < top_k`. Failing at
boot beats discovering it from production traffic.

---

## Stack

FastAPI · Postgres 16 + pgvector (HNSW) · SQLAlchemy 2.0 async · Alembic ·
sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) ·
`cross-encoder/ms-marco-MiniLM-L-6-v2` · Groq (`llama-3.3-70b-versatile`) ·
JWT + bcrypt · Docker
