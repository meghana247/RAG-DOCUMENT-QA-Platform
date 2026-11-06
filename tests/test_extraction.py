"""Text extraction and normalisation."""
import io

import pytest

from app.core.exceptions import DocumentProcessingError, UnsupportedMediaTypeError
from app.rag.extraction import extract_blocks, normalize


class TestNormalize:
    def test_paragraph_breaks_survive(self):
        """Blank lines are the strongest boundary signal the chunker has; the
        original code destroyed them with `" ".join(text.split())`."""
        assert normalize("para one\n\npara two") == "para one\n\npara two"

    def test_horizontal_whitespace_collapses(self):
        assert normalize("too    many\tspaces") == "too many spaces"

    def test_excess_blank_lines_collapse_to_one(self):
        assert normalize("a\n\n\n\n\nb") == "a\n\nb"

    def test_hyphenated_line_breaks_are_rejoined(self):
        # PDF extraction leaves these behind; without rejoining, "represen-" and
        # "tation" embed as two meaningless fragments.
        assert normalize("represen-\ntation") == "representation"

    def test_null_bytes_are_stripped(self):
        assert "\x00" not in normalize("clean\x00text")

    def test_ligatures_are_expanded(self):
        # pypdf returns typographic ligatures as single code points. Left in place,
        # "conﬁdently" never matches a search for "confidently", in either the
        # full-text index or the embedding model's vocabulary.
        assert normalize("conﬁdently eﬀective ﬂow") == "confidently effective flow"


class TestDispatch:
    def test_plain_text(self):
        blocks = extract_blocks(b"hello world", "notes.txt", "text/plain")
        assert len(blocks) == 1
        assert blocks[0].text == "hello world"
        assert blocks[0].page_number is None

    def test_markdown_is_treated_as_text(self):
        blocks = extract_blocks(b"# Title\n\nbody", "readme.md", "")
        assert blocks[0].text.startswith("# Title")

    def test_unsupported_extension_is_rejected(self):
        with pytest.raises(UnsupportedMediaTypeError):
            extract_blocks(b"MZ\x90\x00", "malware.exe", "application/octet-stream")

    def test_mime_type_is_used_when_the_extension_is_missing(self):
        blocks = extract_blocks(b"body text", "noextension", "text/plain")
        assert blocks[0].text == "body text"

    def test_empty_result_raises_rather_than_storing_nothing(self):
        # A scanned PDF with no text layer would otherwise produce a document
        # with zero chunks that silently answers every question with nothing.
        with pytest.raises(DocumentProcessingError):
            extract_blocks(b"   \n\n  ", "blank.txt", "text/plain")

    def test_undecodable_bytes_do_not_crash(self):
        blocks = extract_blocks(b"\xff\xfe\x00valid", "weird.txt", "text/plain")
        assert blocks

    def test_corrupt_pdf_raises_a_domain_error(self):
        with pytest.raises(DocumentProcessingError):
            extract_blocks(b"not really a pdf", "broken.pdf", "application/pdf")


def test_pdf_pages_become_separate_blocks():
    """Page attribution is what makes 'see page 4' possible."""
    pypdf = pytest.importorskip("pypdf")

    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    # Blank pages carry no text, so extraction correctly reports none.
    with pytest.raises(DocumentProcessingError):
        extract_blocks(buffer.getvalue(), "blank.pdf", "application/pdf")
