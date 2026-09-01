"""PDF text, OCR, and multi-table extraction using PyMuPDF (fitz) and modular OCR providers.

Opens PDFs entirely in memory — no temp files — and returns a structured
canonical ParsedDocument with per-page text, multi-page stitched TableElements,
HeadingElements, and TextElements with bounding-box collision subtraction,
page quality classification, and image-based OCR extraction.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import pymupdf as fitz  # PyMuPDF

from rag.assets.extractor import AssetExtractor
from rag.assets.models import ExtractedVisualCandidate
from rag.core.config import AssetConfig, OCRConfig, TableConfig, get_settings
from rag.ocr import (
    BaseOCRProvider,
    ContentSource,
    MixedContentMerger,
    OCRBlock,
    OCRPageResult,
    OCRProviderRegistry,
    OCRQualityEvaluator,
    PageClassificationResult,
    PageQualityAnalyzer,
)
from rag.parsing.models import (
    DocumentElement,
    FigureElement,
    HeadingElement,
    ParsedBlock,
    ParsedDocument,
    ParsedPage,
    TableElement,
    TextElement,
)
from rag.parsing.table_processor import (
    TableContinuationStitcher,
    TableHeaderAndBoundaryDetector,
    TableValidator,
)

logger = logging.getLogger(__name__)

# Heuristic: flag lines that look like headings (numbered 1.2, capitalized titles, short)
_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9\s:,\-&/\(\)]{2,80})$"
)


def _is_heading(text: str) -> bool:
    """Return True if the text fragment looks like a section heading."""
    stripped = text.strip()
    if not stripped or len(stripped) > 120 or stripped.endswith(".") or stripped.endswith(":"):
        return False
    return bool(_HEADING_RE.match(stripped))


def _bbox_overlap_ratio(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate the overlap ratio of bbox1 covered by bbox2."""
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    return intersection / area1 if area1 > 0 else 0.0


def _is_inside_table(block_bbox: Tuple[float, float, float, float], table_bboxes: List[Tuple[float, float, float, float]]) -> bool:
    """Check if a text block bounding box lies substantially inside any table bounding box."""
    for t_bbox in table_bboxes:
        if _bbox_overlap_ratio(block_bbox, t_bbox) > 0.4:
            return True
        cx = (block_bbox[0] + block_bbox[2]) / 2.0
        cy = (block_bbox[1] + block_bbox[3]) / 2.0
        if t_bbox[0] <= cx <= t_bbox[2] and t_bbox[1] <= cy <= t_bbox[3]:
            return True
    return False


class PDFParser:
    """Extract structured text, discrete multi-table elements, and OCR content from PDF bytes."""

    PARSER_VERSION = "pymupdf-v2.0"

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True,
        table_config: Optional[TableConfig] = None,
        ocr_config: Optional[OCRConfig] = None,
        asset_config: Optional[AssetConfig] = None,
        ocr_provider: Optional[BaseOCRProvider] = None,
    ):
        """
        Args:
            extract_images: If True, annotate image blocks in metadata.
            extract_tables: If True, detect and extract tables as TableElements.
            table_config:   Optional TableConfig model (defaults to settings.table).
            ocr_config:     Optional OCRConfig model (defaults to settings.ocr).
            asset_config:   Optional AssetConfig model (defaults to settings.asset).
            ocr_provider:   Optional explicit OCR provider instance.
        """
        settings = get_settings()
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.table_config = table_config or settings.table
        self.ocr_config = ocr_config or getattr(settings, "ocr", OCRConfig())
        self.asset_config = asset_config or getattr(settings, "asset", AssetConfig())

        # OCR Subsystem Components
        self.page_analyzer = PageQualityAnalyzer(
            min_native_text_quality=self.ocr_config.min_native_text_quality,
            default_language=self.ocr_config.language,
        )
        self.ocr_evaluator = OCRQualityEvaluator(
            min_confidence=self.ocr_config.min_confidence,
            max_retries=self.ocr_config.max_retries,
            default_preprocessing=self.ocr_config.default_preprocessing,
        )
        self.mixed_merger = MixedContentMerger(
            iou_overlap_threshold=self.ocr_config.iou_overlap_threshold
        )
        self.asset_extractor = AssetExtractor(config=self.asset_config)

        if ocr_provider is not None:
            self.ocr_provider = ocr_provider
        else:
            try:
                self.ocr_provider = OCRProviderRegistry.get_provider(self.ocr_config.provider)
            except Exception as e:
                logger.warning("Failed to load OCR provider '%s': %s. Falling back to mock.", self.ocr_config.provider, e)
                self.ocr_provider = OCRProviderRegistry.get_provider("mock")

    def _render_page_image(self, page: fitz.Page) -> Image.Image:
        """Render a PyMuPDF page to PIL Image at configured DPI."""
        pix = page.get_pixmap(dpi=self.ocr_config.render_dpi)
        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    def parse(self, content: bytes, filename: Optional[str] = None) -> ParsedDocument:
        """Parse a PDF from raw bytes and return a structured canonical ParsedDocument."""
        if not content:
            raise ValueError(f"PDF content is empty{f' for {filename}' if filename else ''}.")

        label = filename or "<in-memory>"
        doc_slug = Path(label).stem.replace(" ", "_").replace("-", "_")
        logger.info(
            "Parsing PDF '%s' (%d bytes, extract_tables=%s, ocr_enabled=%s)",
            label, len(content), self.extract_tables, self.ocr_config.enabled,
        )

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise RuntimeError(f"PyMuPDF failed to open '{label}': {exc}") from exc

        raw_meta = doc.metadata or {}
        pdf_metadata = {
            "title": raw_meta.get("title", ""),
            "author": raw_meta.get("author", ""),
            "subject": raw_meta.get("subject", ""),
            "creator": raw_meta.get("creator", ""),
            "num_pages": doc.page_count,
            "filename": label,
        }
        doc_title = raw_meta.get("title") or Path(label).stem

        raw_detected_tables: List[TableElement] = []
        page_text_blocks_map: Dict[int, List[ParsedBlock]] = {}
        page_headings_map: Dict[int, List[str]] = {}
        page_text_lines_map: Dict[int, List[str]] = {}
        page_elements_map: Dict[int, List[DocumentElement]] = {}
        page_figures_map: Dict[int, List[FigureElement]] = {}
        page_provenance_map: Dict[int, Dict[str, Any]] = {}
        current_section: Optional[str] = None
        tables_failed_count = 0

        # Diagnostics counters
        ocr_diag = {
            "pages_native": 0,
            "pages_scanned": 0,
            "pages_mixed": 0,
            "ocr_pages_processed": 0,
            "ocr_pages_failed": 0,
        }

        # ── Step 1: Detect Tables, Evaluate Page Quality, and Extract Blocks ──
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed

            # 1a. Detect Tables with Header Candidate Validation & BBox Refinement
            page_raw_tables_data = []
            table_bboxes: List[Tuple[float, float, float, float]] = []

            if self.extract_tables:
                try:
                    tabs = page.find_tables()
                    for t_idx, tab in enumerate(tabs.tables):
                        raw_bbox = tuple(tab.bbox)
                        extracted_matrix = tab.extract()
                        if not extracted_matrix:
                            continue

                        headers, rows, refined_bbox, _ = TableHeaderAndBoundaryDetector.detect_header_and_trim(
                            raw_matrix=extracted_matrix,
                            tab_rows=getattr(tab, "rows", None),
                            tab_bbox=raw_bbox,
                        )

                        table_bboxes.append(refined_bbox)
                        page_raw_tables_data.append((t_idx, headers, rows, refined_bbox))

                except Exception as tab_exc:
                    tables_failed_count += 1
                    logger.warning("Table detection warning on page %d of '%s': %s", page_number, label, tab_exc)

            # 1b. Evaluate Page Quality & Layout Classification
            raw_native_blocks = page.get_text("blocks")
            native_text = page.get_text("text")

            classification = self.page_analyzer.analyze_page(
                page=page,
                page_number=page_number,
                native_text=native_text,
                native_blocks=raw_native_blocks,
            )

            # Determine whether to execute OCR
            use_ocr = self.ocr_config.enabled and classification.ocr_required
            page_source = classification.source_type.value
            ocr_applied = False
            ocr_conf: Optional[float] = None
            ocr_quality: Optional[float] = None
            ocr_profile: Optional[str] = None
            ocr_attempt = 1

            if classification.source_type == ContentSource.NATIVE:
                ocr_diag["pages_native"] += 1
            elif classification.source_type == ContentSource.SCANNED:
                ocr_diag["pages_scanned"] += 1
            elif classification.source_type == ContentSource.MIXED:
                ocr_diag["pages_mixed"] += 1

            # Prepare native text blocks (excluding table bounding box overlap)
            native_parsed_blocks: List[ParsedBlock] = []
            for block_no, block in enumerate(raw_native_blocks):
                x0, y0, x1, y1, block_text, b_no, block_type = block
                block_bbox = (x0, y0, x1, y1)

                if block_type == 1:
                    continue

                clean_text = block_text.strip()
                if not clean_text:
                    continue

                if _is_inside_table(block_bbox, table_bboxes):
                    logger.debug("Skipping text block %d on page %d (inside table bbox)", block_no, page_number)
                    continue

                native_parsed_blocks.append(
                    ParsedBlock(
                        text=clean_text,
                        bbox=block_bbox,
                        block_type=block_type,
                        block_no=b_no,
                    )
                )

            final_page_blocks: List[ParsedBlock] = []

            # 1c. Execute OCR if required
            if use_ocr:
                ocr_diag["ocr_pages_processed"] += 1
                logger.info(
                    "OCR_STARTED page %d of '%s' (type=%s, lang=%s, reason='%s')",
                    page_number, label, classification.source_type.value,
                    classification.detected_language, classification.reason,
                )
                try:
                    target_lang = classification.detected_language or self.ocr_config.language
                    page_image = self._render_page_image(page)
                    ocr_result = self.ocr_evaluator.process_with_retry(
                        provider=self.ocr_provider,
                        raw_image=page_image,
                        page_number=page_number,
                        language=target_lang,
                    )
                    ocr_applied = True
                    ocr_conf = ocr_result.confidence
                    ocr_quality = ocr_result.quality_score
                    ocr_profile = ocr_result.preprocessing_profile
                    ocr_attempt = ocr_result.attempt

                    # Scale OCR coordinates from rendered image pixel space back to PDF point space if needed
                    # PyMuPDF standard 72 pt/in vs render_dpi
                    scale_factor = 72.0 / float(self.ocr_config.render_dpi)
                    scaled_ocr_blocks: List[OCRBlock] = []
                    for ob in ocr_result.blocks:
                        scaled_bbox = (
                            ob.bbox[0] * scale_factor,
                            ob.bbox[1] * scale_factor,
                            ob.bbox[2] * scale_factor,
                            ob.bbox[3] * scale_factor,
                        )
                        # Skip OCR blocks that land inside detected table bounding boxes
                        if not _is_inside_table(scaled_bbox, table_bboxes):
                            scaled_ocr_blocks.append(
                                OCRBlock(
                                    text=ob.text,
                                    bbox=scaled_bbox,
                                    confidence=ob.confidence,
                                    block_type=ob.block_type,
                                    block_no=ob.block_no,
                                    lines=ob.lines,
                                )
                            )

                    # Merge or assign blocks depending on classification
                    if classification.source_type == ContentSource.SCANNED:
                        # Full replacement: convert OCR blocks to ParsedBlocks
                        for idx, ob in enumerate(scaled_ocr_blocks):
                            final_page_blocks.append(
                                ParsedBlock(
                                    text=ob.text,
                                    bbox=ob.bbox,
                                    block_type=ob.block_type,
                                    block_no=idx,
                                )
                            )
                    elif classification.source_type == ContentSource.MIXED:
                        # Non-duplicating spatial and semantic merge
                        final_page_blocks = self.mixed_merger.merge(
                            native_blocks=native_parsed_blocks,
                            ocr_blocks=scaled_ocr_blocks,
                        )
                    else:
                        final_page_blocks = list(native_parsed_blocks)

                    logger.info(
                        "OCR_COMPLETED page %d: %d blocks extracted (conf=%.2f, quality=%.2f, profile=%s)",
                        page_number, len(scaled_ocr_blocks), ocr_conf or 0.0, ocr_quality or 0.0, ocr_profile,
                    )

                    # Re-detect language from the actual OCR output text.
                    # For scanned pages the pre-OCR classification ran on empty native text
                    # (empty string), so classification.language_confidence is 0.0.
                    # Now that we have real text from the OCR engine, re-run detection on it
                    # to get a meaningful language + confidence for the provenance record.
                    if final_page_blocks:
                        ocr_page_text = " ".join(
                            b.text for b in final_page_blocks if b.text and b.text.strip()
                        )
                        if ocr_page_text.strip():
                            post_ocr_lang = self.page_analyzer.language_detector.detect_language(ocr_page_text)
                            detected_lang_final = post_ocr_lang.language
                            lang_conf_final = post_ocr_lang.confidence
                        else:
                            detected_lang_final = classification.detected_language
                            lang_conf_final = classification.language_confidence
                    else:
                        detected_lang_final = classification.detected_language
                        lang_conf_final = classification.language_confidence

                except Exception as ocr_exc:
                    ocr_diag["ocr_pages_failed"] += 1
                    logger.exception("OCR failed on page %d of '%s': %s", page_number, label, ocr_exc)
                    final_page_blocks = list(native_parsed_blocks)
                    detected_lang_final = classification.detected_language
                    lang_conf_final = classification.language_confidence

            else:
                final_page_blocks = list(native_parsed_blocks)
                detected_lang_final = classification.detected_language
                lang_conf_final = classification.language_confidence

            page_provenance_map[page_number] = {
                "content_source": page_source,
                "ocr_applied": ocr_applied,
                "ocr_confidence": ocr_conf,
                "quality_score": ocr_quality if ocr_applied else classification.native_text_quality,
                "ocr_provider": self.ocr_provider.name if ocr_applied else None,
                "preprocessing_profile": ocr_profile,
                "ocr_attempt": ocr_attempt,
                "detected_language": detected_lang_final,
                "language_confidence": lang_conf_final,
            }

            # 1d. Assemble DocumentElements and Sort into Natural Reading Sequence
            parsed_blocks: List[ParsedBlock] = []
            headings: List[str] = []
            page_text_lines: List[str] = []
            page_elements: List[DocumentElement] = []

            items_to_sort = []
            for b_idx, block_obj in enumerate(final_page_blocks):
                items_to_sort.append(("text", block_obj.bbox[1] if block_obj.bbox else 0.0, block_obj))

            for t_idx, headers, rows, refined_bbox in page_raw_tables_data:
                items_to_sort.append(("table", refined_bbox[1], (t_idx, headers, rows, refined_bbox)))

            items_to_sort.sort(key=lambda x: x[1])

            for item in items_to_sort:
                kind = item[0]

                if kind == "text":
                    _, _, block_obj = item
                    clean_text = block_obj.text.strip()
                    if not clean_text:
                        continue

                    parsed_blocks.append(block_obj)
                    page_text_lines.append(clean_text)

                    first_line = clean_text.split("\n")[0].strip()
                    if _is_heading(first_line):
                        headings.append(first_line)
                        current_section = first_line
                        h_elem = HeadingElement(
                            element_id=f"{doc_slug}_p{page_number}_h{len(headings)}",
                            page_number=page_number,
                            bbox=block_obj.bbox,
                            section=current_section,
                            content=first_line,
                            content_source=page_source,
                            ocr_applied=ocr_applied,
                            ocr_confidence=ocr_conf,
                            quality_score=ocr_quality,
                        )
                        page_elements.append(h_elem)
                    else:
                        t_elem = TextElement(
                            element_id=f"{doc_slug}_p{page_number}_txt{len(page_elements)+1}",
                            page_number=page_number,
                            bbox=block_obj.bbox,
                            section=current_section,
                            content=clean_text,
                            content_source=page_source,
                            ocr_applied=ocr_applied,
                            ocr_confidence=ocr_conf,
                            quality_score=ocr_quality,
                        )
                        page_elements.append(t_elem)

                elif kind == "table":
                    _, _, (t_idx, headers, rows, refined_bbox) = item
                    table_id = f"{doc_slug}_p{page_number}_t{t_idx+1}"
                    table_elem = TableElement(
                        element_id=table_id,
                        table_id=table_id,
                        page_number=page_number,
                        page_start=page_number,
                        page_end=page_number,
                        bbox=refined_bbox,
                        section=current_section,
                        headers=headers,
                        rows=rows,
                        context_hierarchy=[doc_title] + ([current_section] if current_section else []),
                    )
                    if self.table_config.validation_enabled:
                        table_elem = TableValidator.validate(
                            table_elem,
                            enable_hierarchical_headers=self.table_config.hierarchical_headers_enabled,
                            enable_cell_inheritance=self.table_config.cell_inheritance_enabled,
                        )
                    raw_detected_tables.append(table_elem)

            page_text_blocks_map[page_number] = parsed_blocks
            page_headings_map[page_number] = headings
            page_text_lines_map[page_number] = page_text_lines
            page_elements_map[page_number] = page_elements

            # 1e. Extract Visual Candidates (Figures, Diagrams, Architecture Images)
            page_figures: List[FigureElement] = []
            if self.asset_config.enabled:
                try:
                    candidates: List[ExtractedVisualCandidate] = []
                    # 1. Digital embedded images
                    if classification.source_type in (ContentSource.NATIVE, ContentSource.MIXED):
                        digital_cands = self.asset_extractor.extract_from_digital_page(
                            doc=doc,
                            page_idx=page_idx,
                            document_id=doc_slug,
                        )
                        candidates.extend(digital_cands)

                    # 2. Scanned figure regions
                    if classification.source_type == ContentSource.SCANNED or (not candidates and classification.source_type == ContentSource.MIXED):
                        page_img = self._render_page_image(page)
                        scanned_cands = self.asset_extractor.extract_from_scanned_page_image(
                            page_image=page_img,
                            page_number=page_number,
                            document_id=doc_slug,
                            page_rect_pts=(float(page.rect.width), float(page.rect.height)),
                        )
                        candidates.extend(scanned_cands)

                    for cand in candidates:
                        fig_elem = FigureElement(
                            element_id=f"{doc_slug}_p{page_number}_fig{len(page_figures)+1}",
                            asset_id=cand.asset_id,
                            page_number=page_number,
                            bbox=cand.bbox,
                            width=cand.width,
                            height=cand.height,
                            mime_type=cand.mime_type,
                            image_bytes=cand.image_bytes,
                            content_source=page_source,
                            metadata={"source_type": cand.source_type},
                        )
                        page_figures.append(fig_elem)

                except Exception as asset_exc:
                    logger.warning("Visual asset extraction warning on page %d of '%s': %s", page_number, label, asset_exc)

            page_figures_map[page_number] = page_figures

        doc.close()

        # ── Step 2: Multi-Page Continuation Stitching ─────────────────────────
        total_detected_tables = len(raw_detected_tables)
        if self.table_config.multipage_continuation_enabled and total_detected_tables > 1:
            final_tables = TableContinuationStitcher.stitch_multipage_tables(
                raw_detected_tables,
                confidence_threshold=self.table_config.continuation_confidence_threshold,
            )
        else:
            final_tables = raw_detected_tables

        stitched_count = sum(1 for t in final_tables if t.is_continuation)

        # ── Step 3: Assemble ParsedPages and Global Elements List ────────────
        pages: List[ParsedPage] = []
        all_elements: List[DocumentElement] = []
        all_figures: List[FigureElement] = []

        for page_number in range(1, doc_metadata_pages := pdf_metadata["num_pages"] + 1):
            page_txt_elements = page_elements_map.get(page_number, [])
            page_tabs = [t for t in final_tables if t.page_start == page_number or t.page_number == page_number]
            page_figs = page_figures_map.get(page_number, [])
            all_figures.extend(page_figs)

            combined_page_elements: List[DocumentElement] = page_txt_elements + page_tabs + page_figs
            combined_page_elements.sort(key=lambda e: (e.bbox[1] if e.bbox else 0.0))

            full_page_text = "\n".join(page_text_lines_map.get(page_number, []))
            prov = page_provenance_map.get(page_number, {})

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=full_page_text,
                    blocks=page_text_blocks_map.get(page_number, []),
                    headings=page_headings_map.get(page_number, []),
                    tables=page_tabs,
                    figures=page_figs,
                    elements=combined_page_elements,
                    content_source=prov.get("content_source", "native"),
                    ocr_applied=prov.get("ocr_applied", False),
                    ocr_confidence=prov.get("ocr_confidence"),
                    quality_score=prov.get("quality_score"),
                    ocr_provider=prov.get("ocr_provider"),
                    preprocessing_profile=prov.get("preprocessing_profile"),
                    ocr_attempt=prov.get("ocr_attempt", 1),
                    detected_language=prov.get("detected_language", "eng"),
                    language_confidence=prov.get("language_confidence", 1.0),
                )
            )
            all_elements.extend(combined_page_elements)

        # ── Step 4: Aggregate Diagnostics & Language ─────────────────────────
        table_diagnostics = {
            "tables_detected": total_detected_tables,
            "tables_stitched": stitched_count,
            "tables_processed": len(final_tables),
            "tables_warning": sum(1 for t in final_tables if t.validation_status == "warning"),
            "tables_failed": tables_failed_count,
        }

        asset_diagnostics = {
            "assets_extracted": len(all_figures),
            "assets_classified": len(all_figures),
            "figures_detected": len(all_figures),
            "diagrams_detected": 0,
            "ocr_applied": 0,
            "vision_applied": 0,
        }

        detected_langs = [p.detected_language for p in pages if getattr(p, "detected_language", None)]
        primary_lang = max(set(detected_langs), key=detected_langs.count) if detected_langs else "eng"
        unique_langs = list(dict.fromkeys(detected_langs))

        logger.info(
            "Finished parsing '%s': %d pages (lang=%s, native=%d, scanned=%d, mixed=%d, ocr_processed=%d), %d tables, %d figures",
            label, len(pages), primary_lang, ocr_diag["pages_native"], ocr_diag["pages_scanned"], ocr_diag["pages_mixed"],
            ocr_diag["ocr_pages_processed"], len(final_tables), len(all_figures),
        )

        return ParsedDocument(
            pages=pages,
            elements=all_elements,
            tables=final_tables,
            figures=all_figures,
            pdf_metadata=pdf_metadata,
            parser_version=self.PARSER_VERSION,
            primary_language=primary_lang,
            languages_detected=unique_langs,
            table_diagnostics=table_diagnostics,
            ocr_diagnostics=ocr_diag,
            asset_diagnostics=asset_diagnostics,
        )
