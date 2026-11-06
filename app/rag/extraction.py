"""Document text extraction that preserves provenance.

The point of returning `TextBlock`s rather than one flat string is that page and
heading information survives into the chunk rows, so answers can cite a page and
the chunker can avoid merging across a section boundary.
"""
from __future__ import annotations

import io
import logging
import re
import unicodedata
from collections.abc import Callable

from app.core.exceptions import DocumentProcessingError, UnsupportedMediaTypeError
from app.rag.types import TextBlock

logger = logging.getLogger(__name__)

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Runs of spaces/tabs collapse, but blank lines survive — they are the strongest
# paragraph signal the chunker has.
_HORIZONTAL_WS = re.compile(r"[ \t\x0b\f\r]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
# Ligatures and soft hyphens that PDF extraction commonly leaves behind.
_SOFT_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def normalize(text: str) -> str:
    # NFKC folds compatibility characters to their plain form — chiefly the
    # ligatures (ﬁ, ﬂ, ﬀ) that PDF extraction emits. Without it a word like
    # "conﬁdently" is invisible to a search for "confidently": Postgres stems it
    # as a different token, and the embedding tokenizer sees an unknown symbol.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "").replace("­", "")
    text = _SOFT_HYPHEN_BREAK.sub(r"\1\2", text)
    text = _HORIZONTAL_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _EXCESS_NEWLINES.sub("\n\n", text).strip()


def _extract_pdf(content: bytes) -> list[TextBlock]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content))
    except PdfReadError as exc:
        raise DocumentProcessingError("The PDF could not be parsed") from exc

    if reader.is_encrypted:
        # An empty-password decrypt covers the common "protected but not
        # password-locked" case; anything else is a hard failure.
        try:
            if reader.decrypt("") == 0:
                raise DocumentProcessingError("The PDF is password protected")
        except (NotImplementedError, PdfReadError) as exc:
            raise DocumentProcessingError("The PDF is password protected") from exc

    blocks: list[TextBlock] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # One malformed page should not lose the other 200.
            logger.warning("failed to extract PDF page", extra={"page": page_no})
            continue
        cleaned = normalize(raw)
        if cleaned:
            blocks.append(TextBlock(text=cleaned, page_number=page_no))
    return blocks


# Word's built-in heading styles are named "Heading 1".."Heading 9"; Title too.
_HEADING_STYLE = re.compile(r"^(Heading\s*\d+|Title|Subtitle)$", re.IGNORECASE)


def _extract_docx(content: bytes) -> list[TextBlock]:
    from docx import Document as DocxDocument

    try:
        document = DocxDocument(io.BytesIO(content))
    except Exception as exc:
        raise DocumentProcessingError("The DOCX file could not be parsed") from exc

    blocks: list[TextBlock] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = normalize("\n\n".join(buffer))
            if text:
                blocks.append(TextBlock(text=text, section=current_section))
            buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "") if paragraph.style else ""
        if _HEADING_STYLE.match(style):
            # A heading closes the previous section and names the next one, so
            # chunks never straddle two unrelated topics.
            flush()
            current_section = text[:255]
            continue
        buffer.append(text)

    flush()

    # Tables carry real content in most business documents; row-per-line keeps
    # cell alignment readable to the model.
    for table in document.tables:
        rows = [
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
            if any(cell.text.strip() for cell in row.cells)
        ]
        if rows:
            blocks.append(TextBlock(text=normalize("\n".join(rows)), section="table"))

    return blocks


def _extract_text(content: bytes) -> list[TextBlock]:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            decoded = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        decoded = content.decode("utf-8", errors="replace")

    cleaned = normalize(decoded)
    return [TextBlock(text=cleaned)] if cleaned else []


_EXTRACTORS: dict[str, Callable[[bytes], list[TextBlock]]] = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_text,
    ".md": _extract_text,
}

SUPPORTED_EXTENSIONS = frozenset(_EXTRACTORS)


def extract_blocks(content: bytes, filename: str, content_type: str = "") -> list[TextBlock]:
    """Dispatch on extension, falling back to the declared MIME type.

    Extension wins because browsers and HTTP clients disagree wildly about DOCX
    and Markdown MIME types, while the filename is supplied by the user and is
    usually right.
    """
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    extractor = _EXTRACTORS.get(suffix)

    if extractor is None:
        if content_type == PDF_MIME:
            extractor = _extract_pdf
        elif content_type == DOCX_MIME:
            extractor = _extract_docx
        elif content_type.startswith("text/"):
            extractor = _extract_text

    if extractor is None:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{suffix or content_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    blocks = extractor(content)
    if not blocks:
        # Almost always a scanned PDF with no text layer. Say so, rather than
        # storing a document with zero chunks that silently answers nothing.
        raise DocumentProcessingError(
            "No extractable text found. If this is a scanned document, it needs OCR first."
        )
    return blocks
