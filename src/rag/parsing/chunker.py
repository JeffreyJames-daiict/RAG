"""Recursive, token-aware text chunker for parsed PDF content.

Splits document text into overlapping chunks that respect natural language
boundaries (paragraphs → sentences → words) while staying within a configurable
token budget. Persists chunk metadata (page number, section heading, token count).
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tiktoken

from rag.core.models import ChunkSchema
from rag.parsing.models import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

# Separator hierarchy: try to split on larger units first
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]


@dataclass
class ChunkingConfig:
    """Configuration for the recursive text chunker.

    Attributes:
        chunk_size:        Maximum number of tokens per chunk.
        chunk_overlap:     Number of tokens to overlap between adjacent chunks.
        chunking_version:  Version tag stored on each Chunk row.
        encoding_model:    tiktoken encoding to use for token counting.
                           Defaults to 'cl100k_base' (GPT-4 / text-embedding-3).
    """
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_version: str = "v1.0"
    encoding_model: str = "cl100k_base"


class RecursiveTextChunker:
    """Splits a ParsedDocument into token-bounded, overlap-aware chunks.

    Strategy mirrors LangChain's RecursiveCharacterTextSplitter but is
    self-contained and token-aware via tiktoken — so chunk_size means
    *tokens*, not characters.

    Usage::

        config = ChunkingConfig(chunk_size=512, chunk_overlap=64)
        chunker = RecursiveTextChunker(config)
        chunks = chunker.chunk(parsed_doc, document_id="uuid-here", version=1)
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        try:
            self._enc = tiktoken.get_encoding(self.config.encoding_model)
        except Exception as exc:
            logger.warning("tiktoken encoding '%s' not found, falling back to cl100k_base: %s",
                           self.config.encoding_model, exc)
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,
        version: int,
    ) -> List[ChunkSchema]:
        """Chunk a ParsedDocument into a list of ChunkSchema objects.

        Args:
            parsed_doc:   Output of PDFParser.parse().
            document_id:  UUID string of the parent document row.
            version:      Active version number of the document.

        Returns:
            Ordered list of ChunkSchema ready to be inserted into `chunks`.
        """
        chunks: List[ChunkSchema] = []
        chunk_index = 0

        for page in parsed_doc.pages:
            if not page.text.strip():
                continue

            # Determine current section heading (last heading seen on this page)
            section = page.headings[-1] if page.headings else None

            # Split this page's text into token-bounded segments
            page_chunks = self._split_text(page.text)

            for raw_text in page_chunks:
                clean = raw_text.strip()
                if not clean:
                    continue

                token_count = len(self._enc.encode(clean))
                metadata: Dict[str, Any] = {
                    "chunk_type": "text",
                    "token_count": token_count,
                    "page": page.page_number,
                    "num_pages": parsed_doc.num_pages,
                    "parser_version": parsed_doc.parser_version,
                    "content_source": getattr(page, "content_source", "native"),
                    "ocr_applied": getattr(page, "ocr_applied", False),
                    "ocr_confidence": getattr(page, "ocr_confidence", None),
                    "quality_score": getattr(page, "quality_score", None),
                    "ocr_provider": getattr(page, "ocr_provider", None),
                    "preprocessing_profile": getattr(page, "preprocessing_profile", None),
                    "ocr_attempt": getattr(page, "ocr_attempt", 1),
                    "language": getattr(page, "detected_language", "eng"),
                    "language_confidence": getattr(page, "language_confidence", 1.0),
                }

                chunks.append(
                    ChunkSchema(
                        chunk_id=uuid.uuid4(),
                        document_id=uuid.UUID(document_id),
                        version=version,
                        chunk_index=chunk_index,
                        text=clean,
                        section=section,
                        page=page.page_number,
                        metadata=metadata,
                        chunking_version=self.config.chunking_version,
                    )
                )
                chunk_index += 1

        logger.info(
            "Chunked document %s v%d → %d chunks (size=%d tok, overlap=%d tok)",
            document_id, version, len(chunks),
            self.config.chunk_size, self.config.chunk_overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split_text(self, text: str) -> List[str]:
        """Recursively split text on decreasing separator granularity."""
        return self._recursive_split(text, _SEPARATORS)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text, trying each separator in priority order."""
        # Base case: text already fits in one chunk
        if self._token_len(text) <= self.config.chunk_size:
            return [text]

        separator = separators[0] if separators else ""
        remaining_separators = separators[1:]

        parts = text.split(separator) if separator else list(text)

        # Merge parts into groups that fit within chunk_size, with overlap
        return self._merge_splits(parts, separator, remaining_separators)

    def _merge_splits(
        self,
        splits: List[str],
        separator: str,
        remaining_separators: List[str],
    ) -> List[str]:
        """Merge consecutive splits into token-bounded chunks with overlap."""
        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for part in splits:
            part_tokens = self._token_len(part)

            # If a single part exceeds chunk_size, recurse on it
            if part_tokens > self.config.chunk_size:
                if current_parts:
                    built = separator.join(current_parts)
                    if built.strip():
                        chunks.append(built)
                    current_parts = []
                    current_tokens = 0

                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks)
                continue

            sep_tokens = self._token_len(separator) if current_parts else 0
            total = current_tokens + sep_tokens + part_tokens

            if total > self.config.chunk_size and current_parts:
                # Flush current window
                built = separator.join(current_parts)
                if built.strip():
                    chunks.append(built)

                # Preserve overlap: remove leading parts until we're under overlap budget
                while current_parts and current_tokens > self.config.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_tokens -= self._token_len(removed)
                    if current_parts:
                        current_tokens -= self._token_len(separator)

            current_parts.append(part)
            current_tokens += (self._token_len(separator) if len(current_parts) > 1 else 0) + part_tokens

        # Flush remaining
        if current_parts:
            built = separator.join(current_parts)
            if built.strip():
                chunks.append(built)

        return chunks
