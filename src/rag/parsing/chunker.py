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
from typing import Any, Dict, List, Optional, Tuple

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

    @staticmethod
    def _compute_heading_level(heading: str) -> int:
        """Infer hierarchical level (1=section, 2=subsection, 3=subsubsection) from heading text."""
        clean = heading.strip()
        num_match = re.match(r"^(?:section|appendix|chapter)?\s*(\d+(?:\.\d+)*)", clean, re.IGNORECASE)
        if num_match:
            digits = num_match.group(1).rstrip(".").split(".")
            return min(len(digits), 4)
        if clean.lower().startswith("subsection"):
            return 2
        if clean.lower().startswith("subsubsection"):
            return 3
        return 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,
        version: int,
        normalized_doc: Optional[Any] = None,
    ) -> List[ChunkSchema]:
        """Chunk a ParsedDocument (or NormalizedDocument) into a list of ChunkSchema objects.

        Args:
            parsed_doc:      Output of PDFParser.parse().
            document_id:     UUID string of the parent document row.
            version:         Active version number of the document.
            normalized_doc:  Optional NormalizedDocument output from ContentNormalizer.

        Returns:
            Ordered list of ChunkSchema ready to be inserted into `chunks`.
        """
        chunks: List[ChunkSchema] = []
        chunk_index = 0

        # Build lookup for normalized page blocks if available
        norm_pages_map = {}
        if normalized_doc and hasattr(normalized_doc, "pages"):
            for np in normalized_doc.pages:
                norm_pages_map[np.page_number] = np

        # Document-wide hierarchical heading stack: [(level, heading_text), ...]
        heading_stack: List[Tuple[int, str]] = []

        for page in parsed_doc.pages:
            norm_page = norm_pages_map.get(page.page_number)

            # Determine text to chunk
            if norm_page and norm_page.normalized_text.strip():
                text_to_chunk = norm_page.normalized_text
            else:
                text_to_chunk = page.text

            if not text_to_chunk.strip():
                continue

            # Update hierarchical heading stack if this page has headings
            if page.headings:
                for h in page.headings:
                    clean_h = h.strip()
                    if not clean_h:
                        continue
                    lvl = self._compute_heading_level(clean_h)
                    # Pop headings at the same or deeper level
                    while heading_stack and heading_stack[-1][0] >= lvl:
                        heading_stack.pop()
                    heading_stack.append((lvl, clean_h))

            # Derive section, subsection, and breadcrumb path
            section_path = [h for _, h in heading_stack]
            primary_section = heading_stack[0][1] if heading_stack else None
            subsection = heading_stack[1][1] if len(heading_stack) > 1 else None
            section_breadcrumb = " > ".join(section_path) if section_path else None

            # Split this page's text into token-bounded segments
            page_chunks = self._split_text(text_to_chunk)

            # Collect normalization audit info if available
            norm_actions = []
            quality_flags = []
            if norm_page:
                for b in norm_page.active_blocks:
                    for act in getattr(b, "normalization_actions", []):
                        if act not in norm_actions:
                            norm_actions.append(act)
                    for flag in getattr(b, "quality_flags", []):
                        if flag not in quality_flags:
                            quality_flags.append(flag)

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
                    "section": primary_section,
                    "subsection": subsection,
                    "section_path": section_path,
                    "breadcrumb": section_breadcrumb,
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

                if norm_actions:
                    metadata["normalization_actions"] = norm_actions
                if quality_flags:
                    metadata["quality_flags"] = quality_flags

                chunks.append(
                    ChunkSchema(
                        chunk_id=uuid.uuid4(),
                        document_id=uuid.UUID(document_id),
                        version=version,
                        chunk_index=chunk_index,
                        text=clean,
                        section=section_breadcrumb,
                        page=page.page_number,
                        metadata=metadata,
                        chunking_version=self.config.chunking_version,
                    )
                )
                chunk_index += 1

        logger.info(
            "Chunked document %s v%d → %d chunks (size=%d tok, overlap=%d tok, normalized=%s)",
            document_id, version, len(chunks),
            self.config.chunk_size, self.config.chunk_overlap,
            bool(normalized_doc),
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
