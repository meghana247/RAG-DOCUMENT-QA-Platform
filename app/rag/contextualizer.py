"""Follow-up query rewriting.

Retrieval sees only the current message, so a conversational follow-up — "what
about the second one?", "why?", "and in 2023?" — carries almost no retrievable
signal and returns whatever the bare pronoun happens to embed near. Rewriting it
into a standalone question using recent turns is one of the largest single wins
available to a chat-shaped RAG system.

Uses a small fast model; the rewrite is on the critical path, so it is capped
tightly and falls back to the original question on any failure or timeout.
"""
from __future__ import annotations

import logging

from groq import AsyncGroq

from app.core.config import GenerationSettings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Rewrite the user's latest message into a single standalone search query "
    "that makes sense without the conversation history. Resolve pronouns and "
    "implicit references using the history. Preserve every proper noun, "
    "identifier, and number exactly as written. If the message is already "
    "self-contained, return it unchanged. Reply with the query only — no "
    "preamble, quotes, or explanation."
)

# A rewrite longer than this is the model ignoring instructions and explaining
# itself; the original question is safer than a paragraph of commentary.
_MAX_REWRITE_CHARS = 400


class QueryContextualizer:
    def __init__(self, client: AsyncGroq, config: GenerationSettings) -> None:
        self._client = client
        self._config = config

    async def contextualize(self, query: str, history: list[dict[str, str]]) -> str:
        if not self._config.contextualize_queries or not history:
            return query

        transcript = "\n".join(
            f"{turn['role']}: {turn['content'][:500]}" for turn in history[-4:]
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._config.contextualize_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Conversation:\n{transcript}\n\nLatest message: {query}",
                    },
                ],
                temperature=0.0,
                max_tokens=128,
                timeout=10.0,
            )
        except Exception:
            # Never fail a chat because the optional rewrite step failed.
            logger.warning("query contextualisation failed, using original", exc_info=True)
            return query

        rewritten = (response.choices[0].message.content or "").strip().strip('"')

        if not rewritten or len(rewritten) > _MAX_REWRITE_CHARS:
            return query

        if rewritten != query:
            logger.info(
                "query rewritten", extra={"original": query, "rewritten": rewritten}
            )
        return rewritten
