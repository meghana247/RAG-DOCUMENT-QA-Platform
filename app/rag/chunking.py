"""Structure-aware recursive chunking.

The original implementation collapsed the document to a single whitespace-joined
string and cut it every 500 characters. That splits mid-word and mid-sentence, so
a chunk's embedding describes two half-thoughts and matches neither cleanly. It
also discarded page numbers, making real citations impossible.

This chunker instead:
  * splits on the strongest available boundary (paragraph > line > sentence > word),
  * packs adjacent pieces up to the target size so chunks are uniformly full,
  * never merges across a page or section boundary,
  * carries overlap at a word boundary so a sentence spanning two chunks is
    retrievable from either.
"""
from __future__ import annotations

from app.core.config import ChunkingSettings
from app.rag.types import Chunk, TextBlock

# Ordered strongest to weakest. "" is the last resort: a hard cut, only reached
# by pathological input such as minified data with no spaces.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


def _split_recursive(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Break `text` into pieces of at most `size`, preferring early separators."""
    if len(text) <= size:
        return [text] if text.strip() else []

    separator, *rest = separators
    remaining = tuple(rest)

    if separator == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    if separator not in text:
        return _split_recursive(text, size, remaining)

    pieces: list[str] = []
    for part in text.split(separator):
        if not part:
            continue
        # The separator must go back on every piece, including a plain space.
        # `str.split` discards it, and `_pack` concatenates pieces directly — so
        # dropping it here joins "the cat" into "thecat" and destroys the text.
        # A trailing separator on the final piece is removed by `_pack`'s strip.
        candidate = part + separator
        if len(candidate) <= size:
            pieces.append(candidate)
        else:
            pieces.extend(_split_recursive(candidate, size, remaining))
    return [p for p in pieces if p.strip()]


def _tail_overlap(text: str, overlap: int) -> str:
    """Last ~`overlap` characters of `text`, trimmed to a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    # Cutting at the first space avoids starting the overlap mid-word; if the
    # tail has no space at all, keep it whole rather than return nothing.
    return tail[space + 1 :] if space != -1 else tail


class RecursiveChunker:
    """Default chunker. Stateless and cheap — safe to construct per request."""

    def __init__(self, config: ChunkingSettings) -> None:
        self._size = config.chunk_size
        self._overlap = min(config.chunk_overlap, config.chunk_size // 2)
        self._min_size = config.min_chunk_size

    def split(self, blocks: list[TextBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0

        for block in blocks:
            pieces = _split_recursive(block.text, self._size, _SEPARATORS)
            for content in self._pack(pieces):
                chunks.append(
                    Chunk(
                        content=content,
                        index=index,
                        page_number=block.page_number,
                        section=block.section,
                    )
                )
                index += 1

        return self._merge_runts(chunks)

    def _pack(self, pieces: list[str]) -> list[str]:
        """Greedily combine pieces up to the target size, adding overlap."""
        packed: list[str] = []
        buffer = ""

        for piece in pieces:
            if not buffer:
                buffer = piece
                continue
            if len(buffer) + len(piece) <= self._size:
                buffer += piece
                continue

            packed.append(buffer.strip())
            overlap = _tail_overlap(buffer, self._overlap)
            # Only carry overlap when it leaves room for real new content,
            # otherwise consecutive chunks become near-duplicates.
            buffer = (overlap + piece) if len(overlap) + len(piece) <= self._size else piece

        if buffer.strip():
            packed.append(buffer.strip())
        return packed

    def _merge_runts(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fold undersized chunks into their neighbour.

        A 30-character chunk ("Appendix B") embeds to noise and pollutes results.
        Merging is only legal within the same page and section, so provenance
        stays truthful.
        """
        if not chunks:
            return []

        merged: list[Chunk] = []
        for chunk in chunks:
            previous: Chunk | None = merged[-1] if merged else None
            fits = (
                previous is not None
                and len(chunk.content) < self._min_size
                and previous.page_number == chunk.page_number
                and previous.section == chunk.section
                and len(previous.content) + len(chunk.content) + 1 <= self._size
            )
            if fits:
                previous.content = f"{previous.content}\n{chunk.content}"
            else:
                merged.append(chunk)

        # Reindex so chunk_index stays a dense 0..n-1 sequence, which the
        # (document_id, chunk_index) unique constraint relies on.
        for position, chunk in enumerate(merged):
            chunk.index = position
        return merged
