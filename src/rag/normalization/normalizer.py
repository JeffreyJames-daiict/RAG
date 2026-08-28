"""Orchestrator for the Content Normalization Layer.

Transforms raw ParsedDocument extraction outputs (Native text, OCR text,
and multi-table elements) into a canonical, normalized, chunk-ready NormalizedDocument.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from rag.core.config import NormalizationConfig, get_settings
from rag.normalization.duplicate_resolver import DuplicateResolver
from rag.normalization.header_footer_detector import HeaderFooterDetector
from rag.normalization.models import (
    ExtractionMethod,
    NormalizedContentBlock,
    NormalizedContentType,
    NormalizedDocument,
    NormalizedPage,
)
from rag.normalization.ocr_normalizer import OCRNormalizer
from rag.normalization.text_normalizer import TextNormalizer
from rag.parsing.models import DocumentElement, HeadingElement, ParsedDocument, ParsedPage, TableElement, TextElement

logger = logging.getLogger(__name__)


class ContentNormalizer:
    """Orchestrates comprehensive, non-destructive document normalization."""

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or get_settings().normalization

        self.text_normalizer = TextNormalizer(
            unicode_form=self.config.unicode_normalization_form,
            repair_soft_hyphens=self.config.repair_soft_hyphens,
            remove_standalone_page_numbers=self.config.remove_standalone_page_numbers,
        )
        self.header_footer_detector = HeaderFooterDetector(
            header_repeat_threshold=self.config.header_repeat_threshold,
            footer_repeat_threshold=self.config.footer_repeat_threshold,
        )
        self.duplicate_resolver = DuplicateResolver(
            similarity_threshold=self.config.duplicate_similarity_threshold,
        )
        self.ocr_normalizer = OCRNormalizer(
            confidence_threshold=self.config.ocr_confidence_threshold,
            max_symbol_ratio=self.config.max_symbol_ratio,
            excessive_repetition_threshold=self.config.excessive_repetition_threshold,
        )

    def normalize(
        self,
        parsed_doc: ParsedDocument,
        document_id: str = "",
        filename: str = "",
    ) -> NormalizedDocument:
        """Transform a ParsedDocument into a canonical NormalizedDocument ready for chunking."""
        if not self.config.enabled:
            logger.info("Normalization disabled by configuration. Passing through raw content.")
            return self._build_passthrough_document(parsed_doc, document_id, filename)

        logger.info(
            "Starting normalization for '%s' (%d pages, %d tables)",
            filename or document_id, len(parsed_doc.pages), len(parsed_doc.tables),
        )

        doc_id = document_id or str(uuid4())
        normalized_doc = NormalizedDocument(
            document_id=doc_id,
            filename=filename,
            tables=parsed_doc.tables,
            primary_language=parsed_doc.primary_language,
            languages_detected=parsed_doc.languages_detected,
        )

        # ── Step 1: Document-Level Header / Footer Analysis ───────────────────
        page_texts = [p.text for p in parsed_doc.pages]
        hf_detection = self.header_footer_detector.analyze_document(page_texts)

        total_blocks_processed = 0
        total_blocks_dropped = 0
        total_duplicates_detected = 0
        total_headers_removed = 0
        total_footers_removed = 0
        total_low_conf_blocks = 0
        total_quality_flags = 0

        # ── Step 2: Page-by-Page Processing ──────────────────────────────────
        for page in parsed_doc.pages:
            norm_page = self._process_page(
                page=page,
                doc_id=doc_id,
                hf_detection=hf_detection,
            )
            normalized_doc.pages.append(norm_page)

            # Accumulate statistics
            for b in norm_page.blocks:
                total_blocks_processed += 1
                if b.is_suppressed:
                    total_duplicates_detected += 1
                if "low_confidence" in b.quality_flags:
                    total_low_conf_blocks += 1
                total_quality_flags += len(b.quality_flags)

            total_headers_removed += norm_page.metadata.get("headers_removed_count", 0)
            total_footers_removed += norm_page.metadata.get("footers_removed_count", 0)
            total_blocks_dropped += norm_page.metadata.get("blocks_dropped_count", 0)

        active_blocks_count = sum(len(p.active_blocks) for p in normalized_doc.pages)

        normalized_doc.normalization_diagnostics = {
            "total_pages": len(normalized_doc.pages),
            "blocks_processed": total_blocks_processed,
            "blocks_active": active_blocks_count,
            "blocks_dropped": total_blocks_dropped,
            "duplicates_detected": total_duplicates_detected,
            "headers_removed": total_headers_removed,
            "footers_removed": total_footers_removed,
            "low_confidence_blocks": total_low_conf_blocks,
            "quality_flags_count": total_quality_flags,
        }

        logger.info(
            "Normalization completed for '%s': %d pages → %d active blocks (%d duplicates suppressed, %d headers stripped)",
            filename or doc_id, len(normalized_doc.pages), active_blocks_count,
            total_duplicates_detected, total_headers_removed,
        )

        return normalized_doc

    def _process_page(
        self,
        page: ParsedPage,
        doc_id: str,
        hf_detection: Any,
    ) -> NormalizedPage:
        """Normalize all elements on a single page with error resilience."""
        norm_page = NormalizedPage(
            page_number=page.page_number,
            raw_text=page.text,
            primary_language=page.detected_language,
            language_confidence=page.language_confidence,
            content_source=page.content_source,
            ocr_applied=page.ocr_applied,
            ocr_confidence=page.ocr_confidence,
            quality_score=page.quality_score,
            tables=page.tables,
        )

        # 1. Clean raw page text with header/footer removal if enabled
        page_clean_text = page.text
        rem_headers: List[str] = []
        rem_footers: List[str] = []

        if self.config.remove_headers or self.config.remove_footers:
            page_clean_text, rem_headers, rem_footers = self.header_footer_detector.strip_headers_and_footers(
                page_text=page.text,
                detection=hf_detection,
            )

        norm_page.metadata["headers_removed_count"] = len(rem_headers)
        norm_page.metadata["footers_removed_count"] = len(rem_footers)
        if rem_headers:
            norm_page.metadata["removed_headers"] = rem_headers
        if rem_footers:
            norm_page.metadata["removed_footers"] = rem_footers

        # 2. Convert elements into NormalizedContentBlocks
        raw_blocks: List[NormalizedContentBlock] = []

        # If rich DocumentElements exist on page, use them
        if page.elements:
            for elem in page.elements:
                b = self._convert_element_to_block(elem, page, doc_id)
                if b is not None:
                    raw_blocks.append(b)
        elif page_clean_text.strip():
            # Fallback: create paragraph-level blocks from page_clean_text
            paragraphs = [p.strip() for p in page_clean_text.split("\n\n") if p.strip()]
            for idx, p_text in enumerate(paragraphs):
                b = NormalizedContentBlock(
                    document_id=doc_id,
                    page_number=page.page_number,
                    content_type=NormalizedContentType.TEXT,
                    source_type=page.content_source,
                    original_text=p_text,
                    normalized_text=p_text,
                    extraction_method=(
                        ExtractionMethod.OCR if page.ocr_applied else ExtractionMethod.NATIVE_PDF
                    ),
                    language=page.detected_language,
                    language_confidence=page.language_confidence,
                    confidence_score=page.ocr_confidence or page.quality_score or 1.0,
                )
                raw_blocks.append(b)

        # 3. Apply Text & OCR Normalization to each block
        valid_blocks: List[NormalizedContentBlock] = []
        blocks_dropped = 0

        for block in raw_blocks:
            try:
                # Normalize text content (unless it's a table with separate serialization)
                if block.content_type != NormalizedContentType.TABLE:
                    norm_text, actions = self.text_normalizer.normalize(block.original_text)
                    block.normalized_text = norm_text
                    for act in actions:
                        block.add_action(act)

                # OCR-specific checks and flags
                if block.extraction_method == ExtractionMethod.OCR or page.ocr_applied:
                    block = self.ocr_normalizer.process_block(block)

                # Validation checks: drop empty or sub-minimum noise
                if block.content_type != NormalizedContentType.TABLE:
                    words = block.normalized_text.split()
                    if (
                        len(block.normalized_text) < self.config.min_text_length
                        or len(words) < self.config.min_word_count
                    ):
                        # Allow headings to be shorter if they pass heading heuristics
                        if block.content_type == NormalizedContentType.HEADING and len(block.normalized_text) >= 3:
                            valid_blocks.append(block)
                        else:
                            blocks_dropped += 1
                            continue
                    else:
                        valid_blocks.append(block)
                else:
                    valid_blocks.append(block)

            except Exception as exc:
                logger.warning(
                    "Error normalizing block on page %d of %s: %s",
                    page.page_number, doc_id, exc,
                )
                block.add_flag("normalization_error")
                valid_blocks.append(block)

        norm_page.metadata["blocks_dropped_count"] = blocks_dropped

        # 4. Duplicate Resolution across sources on this page
        resolved_blocks, decisions = self.duplicate_resolver.resolve_page_duplicates(
            blocks=valid_blocks,
            page_tables=page.tables,
        )

        norm_page.blocks = resolved_blocks
        norm_page.normalized_text = "\n\n".join(
            b.normalized_text for b in norm_page.active_blocks if b.normalized_text
        )

        return norm_page

    def _convert_element_to_block(
        self,
        elem: DocumentElement,
        page: ParsedPage,
        doc_id: str,
    ) -> Optional[NormalizedContentBlock]:
        """Convert a DocumentElement into a NormalizedContentBlock."""
        if isinstance(elem, HeadingElement):
            return NormalizedContentBlock(
                document_id=doc_id,
                page_number=elem.page_number,
                content_type=NormalizedContentType.HEADING,
                source_type=elem.content_source,
                original_text=elem.content,
                normalized_text=elem.content,
                extraction_method=(
                    ExtractionMethod.OCR if elem.ocr_applied else ExtractionMethod.NATIVE_PDF
                ),
                language=page.detected_language,
                language_confidence=page.language_confidence,
                confidence_score=elem.ocr_confidence or elem.quality_score or 1.0,
                metadata={"bbox": elem.bbox, "section": elem.section, "level": elem.level},
            )
        elif isinstance(elem, TextElement):
            return NormalizedContentBlock(
                document_id=doc_id,
                page_number=elem.page_number,
                content_type=NormalizedContentType.TEXT,
                source_type=elem.content_source,
                original_text=elem.content,
                normalized_text=elem.content,
                extraction_method=(
                    ExtractionMethod.OCR if elem.ocr_applied else ExtractionMethod.NATIVE_PDF
                ),
                language=page.detected_language,
                language_confidence=page.language_confidence,
                confidence_score=elem.ocr_confidence or elem.quality_score or 1.0,
                metadata={"bbox": elem.bbox, "section": elem.section},
            )
        elif isinstance(elem, TableElement):
            # For tables, preserve the table representation and reference
            raw_text = elem.raw_markdown or ""
            if not raw_text and elem.rows:
                raw_text = f"Table: {elem.caption or elem.table_id} ({elem.num_rows} rows x {elem.num_cols} cols)"
            return NormalizedContentBlock(
                document_id=doc_id,
                page_number=elem.page_number,
                content_type=NormalizedContentType.TABLE,
                source_type="native",
                original_text=raw_text,
                normalized_text=raw_text,
                extraction_method=ExtractionMethod.TABLE_EXTRACTION,
                language=page.detected_language,
                language_confidence=page.language_confidence,
                confidence_score=1.0,
                table_element=elem,
                metadata={
                    "table_id": elem.table_id,
                    "num_rows": elem.num_rows,
                    "num_cols": elem.num_cols,
                    "validation_status": elem.validation_status,
                },
            )
        return None

    def _build_passthrough_document(
        self,
        parsed_doc: ParsedDocument,
        document_id: str,
        filename: str,
    ) -> NormalizedDocument:
        """Create a passthrough NormalizedDocument when normalization is disabled."""
        doc_id = document_id or str(uuid4())
        norm_doc = NormalizedDocument(
            document_id=doc_id,
            filename=filename,
            tables=parsed_doc.tables,
            primary_language=parsed_doc.primary_language,
            languages_detected=parsed_doc.languages_detected,
        )
        for page in parsed_doc.pages:
            norm_page = NormalizedPage(
                page_number=page.page_number,
                raw_text=page.text,
                normalized_text=page.text,
                primary_language=page.detected_language,
                language_confidence=page.language_confidence,
                content_source=page.content_source,
                ocr_applied=page.ocr_applied,
                ocr_confidence=page.ocr_confidence,
                quality_score=page.quality_score,
                tables=page.tables,
            )
            for elem in page.elements:
                b = self._convert_element_to_block(elem, page, doc_id)
                if b:
                    norm_page.blocks.append(b)
            norm_doc.pages.append(norm_page)
        return norm_doc
