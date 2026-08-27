"""Parsing package — PDF text extraction and recursive chunking.

Exports:
    PDFParser       — extracts structured text from PDF bytes using PyMuPDF
    RecursiveTextChunker — token-aware recursive chunker
    ChunkingConfig  — dataclass to configure chunk_size, overlap, encoding
    ParsedDocument  — structured parse result (pages + metadata)
    ParsedPage      — per-page extraction result
"""

from rag.parsing.chunker import ChunkingConfig, RecursiveTextChunker
from rag.parsing.models import ParsedDocument, ParsedPage
from rag.parsing.pdf_parser import PDFParser

__all__ = [
    "PDFParser",
    "RecursiveTextChunker",
    "ChunkingConfig",
    "ParsedDocument",
    "ParsedPage",
]
