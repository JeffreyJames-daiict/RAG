"""Safe merging strategy for mixed PDF pages containing both native and OCR content."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

from rag.ocr.models import OCRBlock

if TYPE_CHECKING:
    from rag.parsing.models import ParsedBlock

logger = logging.getLogger(__name__)


def _bbox_overlap_ratio(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Compute overlap of bbox1 covered by bbox2."""
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return intersection / area1 if area1 > 0 else 0.0


def _text_overlap_similarity(s1: str, s2: str) -> float:
    """Rough Jaccard word similarity between two text snippets."""
    words1 = set(s1.lower().split())
    words2 = set(s2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


class MixedContentMerger:
    """Merges native extracted text blocks and OCR recognized blocks safely without duplication."""

    def __init__(self, iou_overlap_threshold: float = 0.30, text_similarity_threshold: float = 0.60):
        self.iou_overlap_threshold = iou_overlap_threshold
        self.text_similarity_threshold = text_similarity_threshold

    def merge_blocks(
        self,
        native_blocks: List[ParsedBlock],
        ocr_blocks: List[OCRBlock],
        page_number: int,
    ) -> List[ParsedBlock]:
        """Combine native text blocks with non-overlapping OCR blocks.

        Native text blocks are always prioritized over OCR blocks.

        Args:
            native_blocks: High-quality blocks extracted directly from PDF text layer.
            ocr_blocks: Text blocks recognized by OCR from rendered page image.
            page_number: Current page number.

        Returns:
            Unified, non-duplicate list of ParsedBlock objects in vertical reading order.
        """
        from rag.parsing.models import ParsedBlock

        if not ocr_blocks:
            return native_blocks
        if not native_blocks:
            # Convert all OCR blocks to ParsedBlocks
            return [
                ParsedBlock(
                    text=b.text,
                    bbox=b.bbox,
                    block_type=b.block_type,
                    block_no=b.block_no,
                )
                for b in ocr_blocks
                if b.text.strip()
            ]

        accepted_ocr_blocks: List[ParsedBlock] = []

        for ocr_b in ocr_blocks:
            ocr_text = ocr_b.text.strip()
            if not ocr_text:
                continue

            # Check spatial and semantic overlap against all native blocks
            is_duplicate = False
            for nat_b in native_blocks:
                spatial_overlap = _bbox_overlap_ratio(ocr_b.bbox, nat_b.bbox)
                text_sim = _text_overlap_similarity(ocr_text, nat_b.text)

                if spatial_overlap >= self.iou_overlap_threshold or text_sim >= self.text_similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        "Suppressed duplicate OCR block on page %d (overlap=%.2f, sim=%.2f): '%s'",
                        page_number, spatial_overlap, text_sim, ocr_text[:40],
                    )
                    break

            if not is_duplicate:
                accepted_ocr_blocks.append(
                    ParsedBlock(
                        text=ocr_text,
                        bbox=ocr_b.bbox,
                        block_type=ocr_b.block_type,
                        block_no=len(native_blocks) + len(accepted_ocr_blocks) + 1,
                    )
                )

        # Merge and sort all blocks in vertical reading sequence (y0)
        combined = native_blocks + accepted_ocr_blocks
        combined.sort(key=lambda b: (b.bbox[1] if b.bbox else 0.0))
        return combined
