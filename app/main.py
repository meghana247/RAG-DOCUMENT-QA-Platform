from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.database import init_db
from app.routers import auth, documents, chat, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="RAG Document Q&A Platform",
    description="Multi-tenant RAG-based document question answering with per-user data isolation",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
