"""Parsing package — PDF text extraction, table detection, and multimodal chunking.

Exports:
    PDFParser            — extracts structured text and tables from PDF bytes
    RecursiveTextChunker — token-aware text chunker
    TableChunker         — table-aware chunker with row grouping and context injection
    TableValidator       — structural and consistency validator for tables
    TableSerializer      — hybrid/markdown/row-grouped/json table serializers
    generate_table_summary — table overview summary generator
    ChunkingConfig       — dataclass to configure text chunk_size, overlap, encoding
    ParsedDocument       — canonical structured parse result
    ParsedPage           — per-page extraction result
    DocumentElement      — base element (HeadingElement, TextElement, TableElement)
    TableElement         — structured table element
    HeadingElement       — section heading element
    TextElement          — plain text element
"""

from rag.parsing.chunker import ChunkingConfig, RecursiveTextChunker
from rag.parsing.models import (
    DocumentElement,
    ElementType,
    HeadingElement,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    TableElement,
    TableValidationIssue,
    TextElement,
)
from rag.parsing.pdf_parser import PDFParser
from rag.parsing.table_chunker import TableChunker
from rag.parsing.table_processor import (
    TableSerializer,
    TableValidator,
    generate_table_summary,
)

__all__ = [
    "PDFParser",
    "RecursiveTextChunker",
    "TableChunker",
    "TableValidator",
    "TableSerializer",
    "generate_table_summary",
    "ChunkingConfig",
    "ParsedDocument",
    "ParsedPage",
    "ParsedBlock",
    "DocumentElement",
    "HeadingElement",
    "TextElement",
    "TableElement",
    "TableValidationIssue",
    "ElementType",
]
