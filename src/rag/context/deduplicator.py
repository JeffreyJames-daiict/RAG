"""Deduplication engine for candidate retrieval chunks.

Eliminates:
- Duplicate chunk IDs
- Exact SHA-256 text duplicates
- Near-duplicate chunks with excessive token Jaccard overlap (> 0.85)
- Repeated parent/child or duplicate table content
Preserves distinct chunks belonging to the same document.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rag.retrieval.models import RetrievedChunkResult


def _tokenize_words(text: str) -> Set[str]:
    """Extract lowercased alphanumeric word tokens for Jaccard calculation."""
    return set(re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower()))


def _compute_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Calculate token Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection) / float(union) if union > 0 else 0.0


def _text_hash(text: str) -> str:
    """Normalized content hash for exact deduplication."""
    norm = " ".join(text.strip().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class ContextDeduplicator:
    """Filters redundant and near-duplicate chunks while preserving distinct evidence."""

    def __init__(self, overlap_threshold: float = 0.85):
        self.overlap_threshold = overlap_threshold

    def deduplicate(
        self,
        chunks: List[Any],
    ) -> Tuple[List[Any], List[str]]:
        """Filter duplicate and near-duplicate chunks.

        Accepts items that have `.chunk_id`, `.text`, and optional `.document_id`
        (e.g., RetrievedChunkResult or ContextChunk).

        Returns:
            Tuple of (kept_chunks, dropped_warnings)
        """
        kept: List[Any] = []
        warnings: List[str] = []

        seen_chunk_ids: Set[str] = set()
        seen_text_hashes: Set[str] = set()
        seen_parent_ids: Set[str] = set()

        # Token cache for Jaccard comparisons: list of (chunk_id, doc_id, token_set, text_len)
        admitted_tokens: List[Tuple[str, str, Set[str], int]] = []

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None) or str(id(chunk))
            doc_id = getattr(chunk, "document_id", "")
            text = getattr(chunk, "text", "") or ""

            # 1. Exact chunk_id check
            if chunk_id in seen_chunk_ids:
                warnings.append(f"Dropped duplicate chunk ID: {chunk_id}")
                continue

            # 2. Exact text hash check
            thash = _text_hash(text)
            if thash in seen_text_hashes:
                warnings.append(f"Dropped exact text duplicate: {chunk_id}")
                continue

            # 3. Repeated parent check
            meta = getattr(chunk, "metadata", {}) or getattr(chunk, "chunk_metadata", {}) or {}
            parent_id = meta.get("parent_chunk_id") or meta.get("parent_id")
            if parent_id and parent_id in seen_parent_ids:
                # If a chunk is marked with parent_id and the full parent is already admitted
                warnings.append(f"Dropped chunk with already-admitted parent: {chunk_id} (parent: {parent_id})")
                continue

            # 4. Near-duplicate overlap check (token Jaccard)
            tokens = _tokenize_words(text)
            is_near_duplicate = False

            if len(tokens) >= 10:
                for admitted_id, admitted_doc, admitted_toks, admitted_len in admitted_tokens:
                    # Focus overlap check within same document or cross-document if high similarity
                    overlap = _compute_jaccard(tokens, admitted_toks)
                    if overlap >= self.overlap_threshold:
                        is_near_duplicate = True
                        warnings.append(
                            f"Dropped near-duplicate chunk {chunk_id} (Jaccard {overlap:.2f} with {admitted_id})"
                        )
                        break

                    # Substring containment check: if one chunk is nearly entirely contained in another
                    if doc_id == admitted_doc and (len(text) < admitted_len * 0.9 or admitted_len < len(text) * 0.9):
                        shorter, longer = (text, chunk) if len(text) < admitted_len else (chunk, text)
                        # If shorter is 90% contained in longer
                        if isinstance(shorter, str) and shorter in text:
                            is_near_duplicate = True
                            warnings.append(f"Dropped contained substring chunk {chunk_id} with {admitted_id}")
                            break

            if is_near_duplicate:
                continue

            # Admit chunk
            seen_chunk_ids.add(chunk_id)
            seen_text_hashes.add(thash)
            if parent_id:
                seen_parent_ids.add(parent_id)
            admitted_tokens.append((chunk_id, doc_id, tokens, len(text)))
            kept.append(chunk)

        return kept, warnings
