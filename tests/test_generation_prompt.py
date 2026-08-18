"""Prompt assembly.

Each test here pins one of the defects in the original implementation. They are
cheap, deterministic, and would have caught bugs that were otherwise invisible —
a duplicated question or a stale history window degrades answers without ever
raising an error.
"""
import uuid

from app.core.config import GenerationSettings
from app.rag.generation import _NO_CONTEXT_ANSWER, build_context_block, build_messages
from app.rag.types import RetrievedChunk


def _chunk(content: str, filename: str = "report.pdf", page: int | None = 3) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        content=content,
        chunk_index=7,
        page_number=page,
        final_score=0.9,
    )


class TestContextBlock:
    def test_passages_are_numbered_from_one(self):
        rendered, included = build_context_block([_chunk("alpha"), _chunk("beta")], 10_000)
        assert "[1] report.pdf (page 3)" in rendered
        assert "[2] report.pdf (page 3)" in rendered
        assert len(included) == 2

    def test_falls_back_to_chunk_index_without_a_page(self):
        rendered, _ = build_context_block([_chunk("alpha", page=None)], 10_000)
        # TXT and DOCX sources have no pages, so the chunk index is the only locator.
        assert "chunk 7" in rendered

    def test_budget_truncates_and_included_matches_what_was_rendered(self):
        """The returned list must match the passages actually in the prompt.

        If the budget drops a passage but the caller still stores every source,
        a citation like [4] points at text the model never saw.
        """
        chunks = [_chunk("x" * 400) for _ in range(10)]
        rendered, included = build_context_block(chunks, 1000)

        assert len(included) < len(chunks)
        for position, _ in enumerate(included, start=1):
            assert f"[{position}]" in rendered
        assert f"[{len(included) + 1}]" not in rendered

    def test_at_least_one_passage_survives_an_impossible_budget(self):
        _, included = build_context_block([_chunk("y" * 5000)], 10)
        assert len(included) == 1

    def test_empty_context_renders_empty(self):
        rendered, included = build_context_block([], 1000)
        assert rendered == ""
        assert included == []


class TestMessageAssembly:
    def _config(self, **kwargs) -> GenerationSettings:
        return GenerationSettings(api_key="k", history_turns=2, **kwargs)

    def test_question_appears_exactly_once(self):
        """The original code saved the user message, then loaded history (which
        now contained it), then appended the question again — so the model was
        shown every question twice."""
        history = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
        ]
        messages, _ = build_messages("what is the total?", [_chunk("data")], history, self._config())

        occurrences = sum(1 for m in messages if m["content"] == "what is the total?")
        assert occurrences == 1
        assert messages[-1] == {"role": "user", "content": "what is the total?"}

    def test_history_window_keeps_the_most_recent_turns(self):
        """`ORDER BY created_at ASC LIMIT 10` returned the *oldest* ten messages,
        so past turn ten the model never saw recent context again."""
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(20)
        ]
        messages, _ = build_messages("now", [_chunk("data")], history, self._config())

        contents = [m["content"] for m in messages]
        assert "msg19" in contents, "recent history was dropped"
        assert "msg0" not in contents, "stale history was kept"

    def test_history_turns_bounds_the_window(self):
        history = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        messages, _ = build_messages("now", [_chunk("d")], history, self._config())
        # system + (history_turns * 2) + current question
        assert len(messages) == 1 + 4 + 1

    def test_system_prompt_carries_the_context(self):
        messages, _ = build_messages("q", [_chunk("the revenue was 42")], [], self._config())
        assert messages[0]["role"] == "system"
        assert "the revenue was 42" in messages[0]["content"]

    def test_malformed_history_entries_are_skipped(self):
        history = [
            {"role": "system", "content": "injected"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "kept"},
        ]
        messages, _ = build_messages("q", [_chunk("d")], history, self._config())
        contents = [m["content"] for m in messages]

        # Only the leading system prompt may have role=system; a history entry
        # claiming that role would let stored text override the instructions.
        assert sum(1 for m in messages if m["role"] == "system") == 1
        assert "injected" not in contents
        assert "kept" in contents


def test_refusal_text_is_explicit():
    assert "could not find" in _NO_CONTEXT_ANSWER.lower()
