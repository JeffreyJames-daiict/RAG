"""PDF text extraction using PyMuPDF (fitz).

Opens PDFs entirely in memory — no temp files — and returns a structured
ParsedDocument with per-page text, block information, and PDF-level metadata.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pymupdf as fitz  # PyMuPDF (replaces deprecated 'import fitz')

from rag.parsing.models import ParsedBlock, ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Heuristic: flag lines that look like headings (bold, short, title-case or ALL CAPS)
_HEADING_RE = re.compile(
    r"^(?:[A-Z][A-Z\s\d:,\-]{2,60}|[A-Z][a-z].{2,60})$"
)


def _is_heading(text: str) -> bool:
    """Return True if the text fragment looks like a section heading."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    return bool(_HEADING_RE.match(stripped))


class PDFParser:
    """Extract structured text from PDF bytes using PyMuPDF.

    Usage::

        parser = PDFParser()
        parsed = parser.parse(pdf_bytes, filename="report.pdf")
        for page in parsed.pages:
            print(f"Page {page.page_number}: {len(page.text)} chars")
    """

    PARSER_VERSION = "pymupdf-v1.0"

    def __init__(self, extract_images: bool = False):
        """
        Args:
            extract_images: If True, annotate image blocks in metadata
                            (text is still extracted from surrounding text blocks).
        """
        self.extract_images = extract_images

    def parse(self, content: bytes, filename: Optional[str] = None) -> ParsedDocument:
        """Parse a PDF from raw bytes and return a structured ParsedDocument.

        Args:
            content:  Raw PDF bytes (e.g., read from file or fetched from S3).
            filename: Optional original filename for error reporting.

        Returns:
            ParsedDocument with one ParsedPage per PDF page.

        Raises:
            ValueError: If the content is empty or not a valid PDF.
            RuntimeError: If PyMuPDF fails to open the document.
        """
        if not content:
            raise ValueError(f"PDF content is empty{f' for {filename}' if filename else ''}.")

        label = filename or "<in-memory>"
        logger.info("Parsing PDF '%s' (%d bytes)", label, len(content))

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise RuntimeError(f"PyMuPDF failed to open '{label}': {exc}") from exc

        # Extract PDF document-level metadata
        raw_meta = doc.metadata or {}
        pdf_metadata = {
            "title": raw_meta.get("title", ""),
            "author": raw_meta.get("author", ""),
            "subject": raw_meta.get("subject", ""),
            "creator": raw_meta.get("creator", ""),
            "num_pages": doc.page_count,
            "filename": label,
        }

        pages: list[ParsedPage] = []

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed

            # Extract text in "blocks" mode — returns list of block tuples:
            # (x0, y0, x1, y1, "text", block_no, block_type)
            raw_blocks = page.get_text("blocks")

            parsed_blocks: list[ParsedBlock] = []
            headings: list[str] = []
            page_lines: list[str] = []

            for block in raw_blocks:
                x0, y0, x1, y1, block_text, block_no, block_type = block

                # block_type 1 = image block — skip text extraction, optionally annotate
                if block_type == 1:
                    continue

                clean_text = block_text.strip()
                if not clean_text:
                    continue

                parsed_blocks.append(
                    ParsedBlock(
                        text=clean_text,
                        bbox=(x0, y0, x1, y1),
                        block_type=block_type,
                        block_no=block_no,
                    )
                )
                page_lines.append(clean_text)

                # Detect headings heuristically
                first_line = clean_text.split("\n")[0].strip()
                if _is_heading(first_line):
                    headings.append(first_line)

            full_page_text = "\n".join(page_lines)

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=full_page_text,
                    blocks=parsed_blocks,
                    headings=headings,
                )
            )

            logger.debug(
                "Page %d/%d: %d chars, %d blocks, %d headings",
                page_number,
                doc.page_count,
                len(full_page_text),
                len(parsed_blocks),
                len(headings),
            )

        doc.close()

        logger.info(
            "Finished parsing '%s': %d pages, %d total chars",
            label,
            len(pages),
            sum(len(p.text) for p in pages),
        )

        return ParsedDocument(
            pages=pages,
            pdf_metadata=pdf_metadata,
            parser_version=self.PARSER_VERSION,
        )
