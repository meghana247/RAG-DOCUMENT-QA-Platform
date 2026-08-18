# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder ----
# Wheels are built in a stage that has a compiler; the runtime image gets none of
# it. build-essential alone is ~200MB that has no business in production.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CPU-only torch. The default wheel pulls the full CUDA runtime — several GB of
# GPU libraries that cannot be used on a CPU host.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ---------------------------------------------------------------- runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# Running as root inside a container means a container escape starts as root.
RUN useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Bake the models into the image. Downloading them on first request makes one
# unlucky user wait ~30s, and makes the container depend on HuggingFace being
# reachable at runtime — hence TRANSFORMERS_OFFLINE above.
USER app
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

# No --reload: it runs a file watcher and a supervisor subprocess, and reloads on
# any file change. Workers are left at 1 because each one loads its own copy of
# the embedding model — scale with replicas, not workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
