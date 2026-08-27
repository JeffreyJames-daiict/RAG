"""Internal dataclasses for the PDF parsing layer."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ParsedBlock:
    """A single text block extracted from a page.

    PyMuPDF returns text in block-level granularity. Each block contains
    text from a bounded rectangular region on the page.
    """
    text: str
    bbox: tuple  # (x0, y0, x1, y1) — bounding box on the page
    block_type: int = 0  # 0 = text, 1 = image (we only process text)
    block_no: int = 0


@dataclass
class ParsedPage:
    """All text content extracted from a single PDF page."""
    page_number: int        # 1-indexed
    text: str               # Full concatenated text of the page
    blocks: List[ParsedBlock] = field(default_factory=list)
    headings: List[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Structured extraction result from a PDF document.

    Returned by PDFParser.parse() — carries all pages and document-level
    PDF metadata (title, author, etc.) from the PDF header.
    """
    pages: List[ParsedPage] = field(default_factory=list)
    pdf_metadata: Dict[str, Any] = field(default_factory=dict)
    parser_version: str = "pymupdf-v1.0"

    @property
    def full_text(self) -> str:
        """Concatenated text of all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def num_pages(self) -> int:
        return len(self.pages)
