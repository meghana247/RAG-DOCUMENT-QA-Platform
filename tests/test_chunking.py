"""Chunker behaviour.

These assert the properties that retrieval quality depends on: chunks land on
real boundaries, overlap exists, and page attribution survives.
"""
import pytest

from app.core.config import ChunkingSettings
from app.rag.chunking import RecursiveChunker, _tail_overlap
from app.rag.types import TextBlock


@pytest.fixture
def chunker() -> RecursiveChunker:
    return RecursiveChunker(
        ChunkingSettings(chunk_size=200, chunk_overlap=40, min_chunk_size=50)
    )


def test_short_document_is_one_chunk(chunker):
    chunks = chunker.split([TextBlock(text="A single short paragraph.")])
    assert len(chunks) == 1
    assert chunks[0].content == "A single short paragraph."
    assert chunks[0].index == 0


def test_chunks_respect_the_size_limit(chunker):
    text = " ".join(f"sentence number {i}." for i in range(200))
    chunks = chunker.split([TextBlock(text=text)])

    assert len(chunks) > 1
    # Allowed a small margin: a chunk is packed up to the limit and the final
    # separator-bearing piece may cross it slightly rather than be split mid-word.
    assert all(len(c.content) <= 260 for c in chunks)


def test_words_are_not_split_across_chunks(chunker):
    """The original 500-char slicing cut mid-word, which corrupts embeddings."""
    text = " ".join("indistinguishable" for _ in range(60))
    chunks = chunker.split([TextBlock(text=text)])

    for chunk in chunks:
        for word in chunk.content.split():
            assert word == "indistinguishable", f"split word found: {word!r}"


def test_consecutive_chunks_overlap(chunker):
    sentences = [f"Fact {i} is that thing number {i} exists and matters." for i in range(40)]
    chunks = chunker.split([TextBlock(text=" ".join(sentences))])

    assert len(chunks) >= 2
    # The tail of one chunk should reappear at the head of the next, so a fact
    # spanning the boundary is retrievable from either side.
    first_tail = chunks[0].content.split()[-3:]
    assert any(word in chunks[1].content for word in first_tail)


def test_page_numbers_are_preserved(chunker):
    blocks = [
        TextBlock(text="Page one content. " * 20, page_number=1),
        TextBlock(text="Page two content. " * 20, page_number=2),
    ]
    chunks = chunker.split(blocks)

    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}
    assert all(c.page_number is not None for c in chunks)


def test_chunks_never_span_two_pages(chunker):
    blocks = [
        TextBlock(text="Short page one.", page_number=1),
        TextBlock(text="Short page two.", page_number=2),
    ]
    chunks = chunker.split(blocks)

    # Both blocks are under min_chunk_size, but merging them would attribute
    # page-two text to page one.
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_sections_are_preserved_and_not_merged(chunker):
    blocks = [
        TextBlock(text="Intro text.", section="Introduction"),
        TextBlock(text="Method text.", section="Method"),
    ]
    chunks = chunker.split(blocks)
    assert [c.section for c in chunks] == ["Introduction", "Method"]


def test_indices_are_dense_and_ordered(chunker):
    text = " ".join(f"paragraph {i} body text here." for i in range(80))
    chunks = chunker.split([TextBlock(text=text)])
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_runt_chunks_are_merged_within_a_page(chunker):
    blocks = [
        TextBlock(text="Main body of the section. " * 8, page_number=1),
        TextBlock(text="Fig 1.", page_number=1),
    ]
    chunks = chunker.split(blocks)
    assert all(len(c.content) >= 50 for c in chunks), "a runt chunk survived"


def test_empty_and_whitespace_input_yields_nothing(chunker):
    assert chunker.split([]) == []
    assert chunker.split([TextBlock(text="   \n\n  ")]) == []


def test_text_with_no_separators_still_chunks(chunker):
    """Pathological input — minified data with no spaces — must not loop forever."""
    chunks = chunker.split([TextBlock(text="x" * 1000)])
    assert len(chunks) >= 4
    assert "".join(c.content for c in chunks).count("x") >= 1000 - 40 * len(chunks)


def test_tail_overlap_starts_on_a_word_boundary():
    assert _tail_overlap("alpha beta gamma delta", 12) == "gamma delta"
    assert _tail_overlap("short", 0) == ""
