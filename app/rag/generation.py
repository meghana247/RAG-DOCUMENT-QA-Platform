"""Answer generation against retrieved context.

Fixes carried over from the original implementation:

  * The Groq client was synchronous and called from an `async def` handler, so
    every generation blocked the event loop for its full duration — seconds, on
    a 70B model. It is now `AsyncGroq`.
  * History was loaded with `ORDER BY created_at ASC LIMIT 10`, which returns the
    *oldest* ten messages. Past turn ten, the model was permanently reading the
    start of the conversation and never the recent turns.
  * The current question was appended after a history slice that already
    contained it, so the model saw the question twice.
  * The whole retrieved context was interpolated into the system prompt with no
    size bound, so a larger top_k could silently overflow the context window.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from groq import APIConnectionError, APIStatusError, AsyncGroq, RateLimitError

from app.core.config import GenerationSettings
from app.core.exceptions import UpstreamError
from app.rag.types import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a document question-answering assistant. Answer using \
only the numbered context passages below.

Rules:
- Cite the passages you used with bracketed numbers, e.g. [1] or [2][3]. Cite the \
specific passage each claim came from, not every passage.
- If the context does not contain the answer, say exactly what is missing. Do not \
supply the answer from general knowledge.
- If passages conflict, report the conflict rather than silently choosing one.
- Quote exact figures, names, dates, and identifiers verbatim from the context.
- Be concise. Do not restate the question.

Context passages:
{context}"""

_NO_CONTEXT_ANSWER = (
    "I could not find anything relevant to that in your uploaded documents. "
    "Try rephrasing the question, or upload a document that covers it."
)


def build_context_block(
    chunks: list[RetrievedChunk], char_budget: int
) -> tuple[str, list[RetrievedChunk]]:
    """Render passages as a numbered, attributed block within a size budget.

    Returns the rendered text and the chunks that actually fit, so the citation
    numbers the model is shown always line up with the sources stored on the
    message. Returning the full list here while dropping some from the prompt
    would make `[4]` point at a passage the model never saw.
    """
    parts: list[str] = []
    included: list[RetrievedChunk] = []
    used = 0

    for position, chunk in enumerate(chunks, start=1):
        location = f"page {chunk.page_number}" if chunk.page_number else f"chunk {chunk.chunk_index}"
        if chunk.section:
            location += f", section '{chunk.section}'"
        rendered = f"[{position}] {chunk.filename} ({location})\n{chunk.content}"

        if used + len(rendered) > char_budget and included:
            logger.info(
                "context truncated to fit budget",
                extra={"included": len(included), "available": len(chunks)},
            )
            break

        parts.append(rendered)
        included.append(chunk)
        used += len(rendered)

    return "\n\n---\n\n".join(parts), included


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]],
    config: GenerationSettings,
) -> tuple[list[dict[str, str]], list[RetrievedChunk]]:
    context, included = build_context_block(chunks, config.context_char_budget)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(context=context)}
    ]

    # `history` must exclude the current question — the caller is responsible for
    # that, and this slice takes the most recent turns rather than the oldest.
    turns = config.history_turns * 2  # a "turn" is one user + one assistant message
    for message in history[-turns:]:
        if message.get("role") in ("user", "assistant") and message.get("content"):
            messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": query})
    return messages, included


class GroqGenerator:
    """Satisfies the `Generator` protocol."""

    def __init__(self, client: AsyncGroq, config: GenerationSettings) -> None:
        self._client = client
        self._config = config

    async def generate(
        self,
        *,
        query: str,
        context: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> tuple[str, int | None, int | None]:
        if not context:
            # No relevant context means there is nothing to ground an answer in.
            # Returning the refusal directly saves a paid call that could only
            # produce a hallucination.
            return _NO_CONTEXT_ANSWER, None, None

        messages, _ = build_messages(query, context, history, self._config)

        try:
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                timeout=self._config.timeout_seconds,
            )
        except RateLimitError as exc:
            raise UpstreamError("The language model is rate limited; retry shortly") from exc
        except (APIConnectionError, APIStatusError) as exc:
            logger.error("generation failed", exc_info=exc)
            raise UpstreamError("The language model is currently unavailable") from exc

        usage = response.usage
        return (
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    async def stream(
        self,
        *,
        query: str,
        context: list[RetrievedChunk],
        history: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        if not context:
            yield _NO_CONTEXT_ANSWER
            return

        messages, _ = build_messages(query, context, history, self._config)

        try:
            stream = await self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                timeout=self._config.timeout_seconds,
                stream=True,
            )
            async for event in stream:
                if delta := event.choices[0].delta.content:
                    yield delta
        except RateLimitError as exc:
            raise UpstreamError("The language model is rate limited; retry shortly") from exc
        except (APIConnectionError, APIStatusError) as exc:
            logger.error("streaming generation failed", exc_info=exc)
            raise UpstreamError("The language model is currently unavailable") from exc


def build_client(config: GenerationSettings) -> AsyncGroq:
    return AsyncGroq(
        api_key=config.api_key.get_secret_value(),
        max_retries=config.max_retries,
        timeout=config.timeout_seconds,
    )
