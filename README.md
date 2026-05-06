# RAG Document Q&A Platform

A production-ready, multi-tenant **Retrieval-Augmented Generation (RAG)** platform that lets users upload documents and have intelligent conversations with them — powered by vector similarity search, semantic embeddings, and Groq's blazing-fast LLM inference.

---

## What It Does

Upload a PDF, DOCX, or TXT file, ask questions in natural language, and get accurate answers grounded in your document — with source citations pointing to the exact chunk the answer came from. Every user's data is fully isolated; no one can access another user's documents or chat history.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Client / Postman                      │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTPS REST
┌────────────────────────────▼────────────────────────────────┐
│                    FastAPI Application                       │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Auth   │  │ Documents │  │   Chat   │  │  Admin   │  │
│  │  Router  │  │  Router   │  │  Router  │  │  Router  │  │
│  └──────────┘  └─────┬─────┘  └────┬─────┘  └──────────┘  │
│                       │             │                        │
│          ┌────────────▼─────────────▼──────────┐            │
│          │           Service Layer              │            │
│          │  ┌────────────┐  ┌───────────────┐  │            │
│          │  │  Document  │  │  RAG Service  │  │            │
│          │  │  Service   │  │  (Retrieval + │  │            │
│          │  │ (Extract → │  │   Generation) │  │            │
│          │  │  Chunk →   │  └──────┬────────┘  │            │
│          │  │  Embed)    │         │            │            │
│          │  └────────────┘         │            │            │
│          └─────────────────────────────────────┘             │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼──────────┐   ┌─────────▼──────────┐
│   PostgreSQL 16     │   │     Groq API        │
│   + pgvector        │   │  (LLaMA 3.3 70B)   │
│                     │   │                     │
│  • users            │   │  Fast LLM inference │
│  • documents        │   │  with chat history  │
│  • document_chunks  │   │  and source context │
│    (384-dim vectors)│   └─────────────────────┘
│  • chat_sessions    │
│  • chat_messages    │
└─────────────────────┘
```

---

## Key Features

| Feature | Details |
|---|---|
| **Multi-tenant isolation** | Every query, chunk, and session is scoped to the authenticated user — strict per-user data separation enforced at the database level |
| **Semantic search** | Documents are chunked and embedded using `all-MiniLM-L6-v2`; queries use cosine similarity via pgvector to retrieve the top-K most relevant chunks |
| **Cited answers** | Every LLM response includes source metadata — filename, chunk index, and a content preview — so answers are always traceable |
| **Conversational memory** | The last 10 messages of a chat session are included in the LLM prompt, enabling multi-turn conversations |
| **Multi-format support** | Parses PDF, DOCX, and plain text files out of the box |
| **JWT authentication** | Stateless token-based auth with role-based access control (`user` / `admin`) |
| **Admin panel** | Admins can view all users and all documents across the platform |
| **Docker-first** | Single `docker compose up` spins up the full stack — app + pgvector database |
| **Async throughout** | Built on async SQLAlchemy + asyncpg for non-blocking I/O end to end |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI 0.115 |
| **Language** | Python 3.11 |
| **Database** | PostgreSQL 16 + pgvector extension |
| **ORM** | SQLAlchemy 2.0 (async) |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dimensions) |
| **LLM** | Groq API — `llama-3.3-70b-versatile` |
| **Auth** | JWT (python-jose) + bcrypt password hashing |
| **Document parsing** | pypdf, python-docx |
| **Containerization** | Docker + Docker Compose |
| **ASGI Server** | Uvicorn |

---

## How RAG Works in This Project

```
User uploads document
        │
        ▼
 Text Extraction (PDF / DOCX / TXT)
        │
        ▼
 Chunking  ──── 500 chars per chunk, 50-char overlap
        │
        ▼
 Embedding  ──── all-MiniLM-L6-v2 → 384-dim vectors
        │
        ▼
 Stored in PostgreSQL (pgvector)
        │
        │   User asks a question
        │         │
        │         ▼
        │   Query is embedded
        │         │
        │         ▼
        └──► Cosine similarity search → Top 5 chunks retrieved
                  │
                  ▼
         Chunks + chat history injected into LLM prompt
                  │
                  ▼
         Groq LLaMA 3.3 70B generates a cited answer
                  │
                  ▼
         Response + sources saved to DB and returned
```

---

## Project Structure

```
rag-document-qa-platform/
├── app/
│   ├── main.py                  # FastAPI app + lifespan (DB init)
│   ├── config.py                # Pydantic settings from .env
│   ├── database.py              # Async SQLAlchemy engine + session
│   ├── dependencies.py          # JWT token → current user dependency
│   ├── models/
│   │   ├── user.py              # User model (id, email, role, password_hash)
│   │   ├── document.py          # Document + DocumentChunk (with vector column)
│   │   └── chat.py              # ChatSession + ChatMessage (with sources JSON)
│   ├── routers/
│   │   ├── auth.py              # /auth/register, /auth/login, /auth/me
│   │   ├── documents.py         # /documents/upload, GET /, DELETE /{id}
│   │   ├── chat.py              # /chat/sessions CRUD + /messages
│   │   └── admin.py             # /admin/users, /admin/documents
│   ├── services/
│   │   ├── auth_service.py      # Password hashing, JWT creation
│   │   ├── document_service.py  # Extract → chunk → embed → store
│   │   ├── embedding_service.py # SentenceTransformer wrapper (lazy-loaded)
│   │   └── rag_service.py       # Vector retrieval + Groq LLM call
│   └── schemas/
│       ├── auth.py              # Request/response Pydantic models for auth
│       ├── document.py          # Document schemas
│       └── chat.py              # Chat session + message schemas
├── Dockerfile                   # Bakes in the embedding model at build time
├── docker-compose.yml           # App + pgvector services
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- A free [Groq API key](https://console.groq.com/)

### 1. Clone the repo

```bash
git clone https://github.com/meghana247/RAG-DOCUMENT-QA-Platform.git
cd RAG-DOCUMENT-QA-Platform
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/ragdb
SECRET_KEY=your-long-random-secret-key
GROQ_API_KEY=your-groq-api-key-here
```

### 3. Start the stack

```bash
docker compose up --build
```

> The first build downloads the `all-MiniLM-L6-v2` embedding model into the image so the first request isn't slow.

The API is now live at **`http://localhost:8000`**

---

## API Reference

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/auth/register` | Create a new user account |
| `POST` | `/auth/login` | Login and receive a JWT token |
| `GET` | `/auth/me` | Get the current user's profile |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload a PDF, DOCX, or TXT file |
| `GET` | `/documents/` | List all documents for the current user |
| `DELETE` | `/documents/{id}` | Delete a document and all its chunks |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat/sessions` | Create a new chat session (optionally scoped to a document) |
| `GET` | `/chat/sessions` | List all sessions for the current user |
| `POST` | `/chat/sessions/{id}/messages` | Send a message and get an AI answer with sources |
| `GET` | `/chat/sessions/{id}/messages` | Retrieve full message history for a session |
| `DELETE` | `/chat/sessions/{id}` | Delete a session and its messages |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/users` | List all registered users |
| `GET` | `/admin/documents` | List all documents across all users |

---

## End-to-End Walkthrough (curl)

```bash
# 1. Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "secret123"}'

# 2. Login → grab token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "secret123"}' | jq -r .access_token)

# 3. Upload a document
DOC_ID=$(curl -s -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@report.pdf" | jq -r .id)

# 4. Start a chat session scoped to that document
SESSION_ID=$(curl -s -X POST http://localhost:8000/chat/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"title\": \"Report Analysis\", \"document_id\": \"$DOC_ID\"}" | jq -r .id)

# 5. Ask a question
curl -s -X POST http://localhost:8000/chat/sessions/$SESSION_ID/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "What are the key findings in this report?"}' | jq
```

---

## Configuration Reference

All settings are controlled via environment variables (or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL connection string |
| `SECRET_KEY` | *(required)* | JWT signing secret |
| `GROQ_API_KEY` | *(required)* | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `EMBEDDING_DIM` | `384` | Vector dimension (must match model) |
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K_CHUNKS` | `5` | Number of chunks retrieved per query |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT expiry window |

---

## Design Decisions

**Why pgvector instead of a dedicated vector DB?**
Keeping vectors in PostgreSQL alongside relational data means no cross-service joins, simpler transactions, and one less infrastructure dependency. For most document Q&A workloads, pgvector's cosine similarity performance is more than sufficient.

**Why Groq?**
Groq's LPU inference is significantly faster than GPU-based providers for the same model, making the chat feel responsive without streaming complexity.

**Why chunk overlap?**
A 50-character overlap ensures that a sentence spanning a chunk boundary isn't cut off from its context in retrieval.

**Why bake the embedding model into the Docker image?**
The first request would otherwise trigger a ~90MB model download inside the container. Baking it in trades a larger image for instant startup.

---

## License

MIT
