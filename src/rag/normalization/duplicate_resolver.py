"""Source-aware duplicate content detector and resolver.

Resolves content conflicts between extraction sources (Native PDF text, OCR text,
and structured TableElement extractions) based on quality, fidelity, and source priority.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.normalization.models import ExtractionMethod, NormalizedContentBlock, NormalizedContentType
from rag.parsing.models import TableElement

logger = logging.getLogger(__name__)


@dataclass
class DuplicateResolutionDecision:
    """Record of a duplicate content resolution decision for provenance auditing."""
    selected_block_id: str
    suppressed_block_id: str
    selected_source: str
    suppressed_source: str
    similarity_score: float
    reason: str


class DuplicateResolver:
    """Detects and resolves duplicate content across different extraction mechanisms on a page."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def resolve_page_duplicates(
        self,
        blocks: List[NormalizedContentBlock],
        page_tables: Optional[List[TableElement]] = None,
    ) -> Tuple[List[NormalizedContentBlock], List[DuplicateResolutionDecision]]:
        """Resolve duplicate content across blocks on a single page.

        1. Suppress text blocks that duplicate extracted TableElements.
        2. Suppress duplicate OCR blocks when clean native PDF text is already present.
        3. Suppress duplicate native blocks when OCR quality is demonstrably superior.

        Returns:
            Tuple of (all_blocks_with_updated_suppression_state, list_of_decisions).
        """
        decisions: List[DuplicateResolutionDecision] = []
        if not blocks:
            return [], []

        # 1. Check against structured tables on this page
        if page_tables:
            for block in blocks:
                if block.content_type == NormalizedContentType.TABLE or block.is_suppressed:
                    continue

                for table in page_tables:
                    sim = self._calculate_table_text_similarity(block.normalized_text, table)
                    if sim >= 0.70:  # High overlap with table content
                        block.is_suppressed = True
                        block.suppression_reason = (
                            f"Duplicate of structured TableElement '{table.table_id}' "
                            f"(similarity={sim:.2f})"
                        )
                        decision = DuplicateResolutionDecision(
                            selected_block_id=table.table_id or table.element_id,
                            suppressed_block_id=block.block_id,
                            selected_source="table_extraction",
                            suppressed_source=block.extraction_method.value,
                            similarity_score=sim,
                            reason="Structured table extraction preferred over unformatted plain text",
                        )
                        decisions.append(decision)
                        block.provenance["duplicate_resolution"] = {
                            "suppressed_by": table.table_id,
                            "similarity": round(sim, 3),
                            "reason": decision.reason,
                        }
                        break

        # 2. Pairwise comparison between active blocks (e.g. Native vs OCR)
        active_blocks = [b for b in blocks if not b.is_suppressed]
        for i in range(len(active_blocks)):
            b1 = active_blocks[i]
            if b1.is_suppressed:
                continue

            for j in range(i + 1, len(active_blocks)):
                b2 = active_blocks[j]
                if b2.is_suppressed:
                    continue

                # Only compare if they originate from different extraction methods or are near identical
                sim = self._calculate_text_similarity(b1.normalized_text, b2.normalized_text)
                if sim >= self.similarity_threshold:
                    selected, suppressed, reason = self._select_winner(b1, b2, sim)
                    suppressed.is_suppressed = True
                    suppressed.suppression_reason = reason
                    dec = DuplicateResolutionDecision(
                        selected_block_id=selected.block_id,
                        suppressed_block_id=suppressed.block_id,
                        selected_source=selected.extraction_method.value,
                        suppressed_source=suppressed.extraction_method.value,
                        similarity_score=sim,
                        reason=reason,
                    )
                    decisions.append(dec)
                    suppressed.provenance["duplicate_resolution"] = {
                        "suppressed_by": selected.block_id,
                        "similarity": round(sim, 3),
                        "reason": reason,
                    }

        return blocks, decisions

    def _select_winner(
        self,
        b1: NormalizedContentBlock,
        b2: NormalizedContentBlock,
        sim: float,
    ) -> Tuple[NormalizedContentBlock, NormalizedContentBlock, str]:
        """Select higher fidelity block between two duplicate candidates."""
        # Preference order: Table > Native PDF (if clean) > OCR
        # Check if one is native and one is OCR
        is_b1_native = b1.extraction_method == ExtractionMethod.NATIVE_PDF
        is_b2_native = b2.extraction_method == ExtractionMethod.NATIVE_PDF
        is_b1_ocr = b1.extraction_method == ExtractionMethod.OCR
        is_b2_ocr = b2.extraction_method == ExtractionMethod.OCR

        if is_b1_native and is_b2_ocr:
            if b1.confidence_score >= 0.70:
                return b1, b2, f"Native PDF text preferred over OCR (similarity={sim:.2f})"
            else:
                return b2, b1, f"OCR text preferred due to low native text quality ({b1.confidence_score:.2f})"

        if is_b2_native and is_b1_ocr:
            if b2.confidence_score >= 0.70:
                return b2, b1, f"Native PDF text preferred over OCR (similarity={sim:.2f})"
            else:
                return b1, b2, f"OCR text preferred due to low native text quality ({b2.confidence_score:.2f})"

        # If both same source, select the one with higher confidence score or more content
        if b1.confidence_score >= b2.confidence_score:
            return b1, b2, f"Higher confidence score ({b1.confidence_score:.2f} >= {b2.confidence_score:.2f})"
        else:
            return b2, b1, f"Higher confidence score ({b2.confidence_score:.2f} > {b1.confidence_score:.2f})"

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Compute token-level Jaccard similarity between two text strings."""
        if not text1 or not text2:
            return 0.0

        words1 = set(re.findall(r"\w+", text1.lower()))
        words2 = set(re.findall(r"\w+", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_table_text_similarity(self, text: str, table: TableElement) -> float:
        """Estimate what fraction of words in `text` are present in table cells/headers."""
        if not text:
            return 0.0

        text_words = set(re.findall(r"\w+", text.lower()))
        if not text_words:
            return 0.0

        table_words: Set[str] = set()
        for h in table.headers:
            table_words.update(re.findall(r"\w+", h.lower()))
        for row in table.rows:
            for cell in row:
                table_words.update(re.findall(r"\w+", cell.lower()))

        if not table_words:
            return 0.0

        intersection = len(text_words.intersection(table_words))
        # If over 70% of the block's words are verbatim inside the table, it's a duplicate
        return intersection / len(text_words)
