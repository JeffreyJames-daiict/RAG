"""PDF text and multi-table extraction using PyMuPDF (fitz).

Opens PDFs entirely in memory — no temp files — and returns a structured
canonical ParsedDocument with per-page text, multi-page stitched TableElements,
HeadingElements, and TextElements with bounding-box collision subtraction.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf as fitz  # PyMuPDF

from rag.core.config import TableConfig, get_settings
from rag.parsing.models import (
    DocumentElement,
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
    """Extract structured text and discrete multi-table elements from PDF bytes using PyMuPDF."""

    PARSER_VERSION = "pymupdf-v2.0"

    def __init__(
        self,
        extract_images: bool = False,
        extract_tables: bool = True,
        table_config: Optional[TableConfig] = None,
    ):
        """
        Args:
            extract_images: If True, annotate image blocks in metadata.
            extract_tables: If True, detect and extract tables as TableElements.
            table_config:   Optional TableConfig model (defaults to settings.table).
        """
        settings = get_settings()
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.table_config = table_config or settings.table

    def parse(self, content: bytes, filename: Optional[str] = None) -> ParsedDocument:
        """Parse a PDF from raw bytes and return a structured canonical ParsedDocument."""
        if not content:
            raise ValueError(f"PDF content is empty{f' for {filename}' if filename else ''}.")

        label = filename or "<in-memory>"
        doc_slug = Path(label).stem.replace(" ", "_").replace("-", "_")
        logger.info("Parsing PDF '%s' (%d bytes, extract_tables=%s)", label, len(content), self.extract_tables)

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
        current_section: Optional[str] = None
        tables_failed_count = 0

        # ── Step 1: Detect Tables and Extract Text per Page ──────────────────
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

            # 1b. Extract Text Blocks with Refined Table BBox Subtraction
            raw_blocks = page.get_text("blocks")
            parsed_blocks: List[ParsedBlock] = []
            headings: List[str] = []
            page_text_lines: List[str] = []
            page_elements: List[DocumentElement] = []

            # Combine text items and table items into unified vertical reading order
            items_to_sort = []

            for block_no, block in enumerate(raw_blocks):
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

                items_to_sort.append(("text", y0, block_bbox, clean_text, block_no, b_no, block_type))

            for t_idx, headers, rows, refined_bbox in page_raw_tables_data:
                items_to_sort.append(("table", refined_bbox[1], refined_bbox, (t_idx, headers, rows), None, None, None))

            items_to_sort.sort(key=lambda x: x[1])

            # Process items in vertical order to track accurate section headers
            for item in items_to_sort:
                kind = item[0]

                if kind == "text":
                    _, _, block_bbox, clean_text, block_no, b_no, block_type = item
                    parsed_blocks.append(
                        ParsedBlock(
                            text=clean_text,
                            bbox=block_bbox,
                            block_type=block_type,
                            block_no=b_no,
                        )
                    )
                    page_text_lines.append(clean_text)

                    first_line = clean_text.split("\n")[0].strip()
                    if _is_heading(first_line):
                        headings.append(first_line)
                        current_section = first_line
                        h_elem = HeadingElement(
                            element_id=f"{doc_slug}_p{page_number}_h{len(headings)}",
                            page_number=page_number,
                            bbox=block_bbox,
                            section=current_section,
                            content=first_line,
                        )
                        page_elements.append(h_elem)
                    else:
                        t_elem = TextElement(
                            element_id=f"{doc_slug}_p{page_number}_txt{block_no+1}",
                            page_number=page_number,
                            bbox=block_bbox,
                            section=current_section,
                            content=clean_text,
                        )
                        page_elements.append(t_elem)

                elif kind == "table":
                    _, _, refined_bbox, (t_idx, headers, rows), _, _, _ = item
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

        for page_number in range(1, doc_metadata_pages := pdf_metadata["num_pages"] + 1):
            page_txt_elements = page_elements_map.get(page_number, [])
            page_tabs = [t for t in final_tables if t.page_start == page_number or t.page_number == page_number]

            combined_page_elements: List[DocumentElement] = page_txt_elements + page_tabs
            combined_page_elements.sort(key=lambda e: (e.bbox[1] if e.bbox else 0.0))

            full_page_text = "\n".join(page_text_lines_map.get(page_number, []))

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=full_page_text,
                    blocks=page_text_blocks_map.get(page_number, []),
                    headings=page_headings_map.get(page_number, []),
                    tables=page_tabs,
                    elements=combined_page_elements,
                )
            )
            all_elements.extend(combined_page_elements)

        # ── Step 4: Aggregate Table Health Diagnostics ───────────────────────
        table_diagnostics = {
            "tables_detected": total_detected_tables,
            "tables_stitched": stitched_count,
            "tables_processed": len(final_tables),
            "tables_warning": sum(1 for t in final_tables if t.validation_status == "warning"),
            "tables_failed": tables_failed_count,
        }

        logger.info(
            "Finished parsing '%s': %d pages, %d tables processed (%d multi-page stitched, %d warnings)",
            label, len(pages), len(final_tables), stitched_count, table_diagnostics["tables_warning"],
        )

        return ParsedDocument(
            pages=pages,
            elements=all_elements,
            tables=final_tables,
            pdf_metadata=pdf_metadata,
            parser_version=self.PARSER_VERSION,
            table_diagnostics=table_diagnostics,
        )
