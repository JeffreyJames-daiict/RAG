"""Advanced table normalization, validation, continuation stitching, header detection, and semantic serialization engine.

Provides:
- TableHeaderAndBoundaryDetector: detects authentic table headers, trims non-table leading paragraphs, and refines bboxes
- HierarchicalHeaderNormalizer: flattens multi-tier header rows into unambiguous composite headers
- CellInheritanceProcessor: forward-fills empty hierarchy cells in merged category structures
- TableValidator: structural integrity and anomaly checking (non-fatal warnings)
- TableContinuationStitcher: multi-signal cross-page table continuation detector and stitcher
- TableSerializer: hybrid, markdown, row_based, and json serializers
- generate_table_summary: deterministic token-bounded table schema summary generator
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from rag.parsing.models import TableElement, TableValidationIssue

logger = logging.getLogger(__name__)


# ── 1. Table Header and Boundary Detector ─────────────────────────────────────

class TableHeaderAndBoundaryDetector:
    """Detects authentic table headers, trims non-table leading paragraphs, and refines bboxes."""

    @classmethod
    def score_header_candidate(cls, row: List[str], num_cols: int, row_idx: int = 0) -> float:
        """Score candidate header row based on column density, conciseness, non-numeric labels, and position."""
        if not row:
            return -1.0

        clean_cells = [str(c or "").strip() for c in row]
        non_empty = [c for c in clean_cells if c]

        if not non_empty:
            return -1.0

        # If table has >= 2 columns but candidate row has only 1 populated cell -> banner/paragraph
        if len(non_empty) < 2 and num_cols >= 2:
            return -1.0

        avg_len = sum(len(c) for c in non_empty) / len(non_empty)
        max_len = max(len(c) for c in non_empty)

        # Severe penalty for paragraph-like prose or multi-sentence text
        if avg_len > 60 or max_len > 120:
            return -1.0
        if any("\n\n" in c or len(c.split(". ")) > 2 for c in non_empty):
            return -1.0

        # Severe penalty if row looks like data rows (numbers, dates, currency, checkmarks only)
        data_cell_count = sum(1 for c in non_empty if any(ch.isdigit() or ch in "$€£¥%✓—" for ch in c))
        data_ratio = data_cell_count / len(non_empty)
        if data_ratio > 0.35 and len(non_empty) > 1:
            return -1.0

        unique_ratio = len(set(non_empty)) / len(non_empty)
        if unique_ratio < 0.5 and len(non_empty) > 2:
            # Low uniqueness: repeated identical text across columns
            return -1.0

        populated_ratio = len(non_empty) / num_cols if num_cols else 0.0
        conciseness = 1.0 - min(avg_len / 40.0, 1.0)
        pos_factor = max(0.5, 1.0 - (0.08 * row_idx))

        score = ((populated_ratio * 0.45) + (unique_ratio * 0.35) + (conciseness * 0.20)) * pos_factor
        return score

    @classmethod
    def detect_header_and_trim(
        cls,
        raw_matrix: List[List[str]],
        tab_rows: Optional[Any] = None,
        tab_bbox: Optional[Tuple[float, float, float, float]] = None,
        max_scan_rows: int = 6,
    ) -> Tuple[List[str], List[List[str]], Tuple[float, float, float, float], int]:
        """Find the true header row, trim leading document rows, and compute refined bounding box.

        Returns:
            Tuple of (headers, data_rows, refined_bbox, header_row_index).
        """
        if not raw_matrix:
            return [], [], (tab_bbox or (0.0, 0.0, 0.0, 0.0)), 0

        num_cols = max(len(r) for r in raw_matrix)
        best_row_idx = 0
        best_score = -1.0

        scan_limit = min(max_scan_rows, len(raw_matrix))
        for idx in range(scan_limit):
            score = cls.score_header_candidate(raw_matrix[idx], num_cols, row_idx=idx)
            if score > best_score:
                best_score = score
                best_row_idx = idx

        if best_score < 0.25:
            # No valid header found in candidate scan -> fallback to synthetic Column_1..N
            headers = [f"Column_{i+1}" for i in range(num_cols)]
            data_rows = [r for r in raw_matrix if any(str(c or "").strip() for c in r)]
            refined_bbox = tab_bbox or (0.0, 0.0, 0.0, 0.0)
            return headers, data_rows, refined_bbox, -1

        # Clean header strings
        raw_header_row = raw_matrix[best_row_idx]
        headers = []
        for i, c in enumerate(raw_header_row):
            clean_c = re.sub(r"\s+", " ", str(c or "").strip())
            if not clean_c:
                clean_c = f"Column_{i+1}"
            headers.append(clean_c)

        # Extract data rows after the header
        raw_data_rows = raw_matrix[best_row_idx + 1:]
        data_rows = []
        for r in raw_data_rows:
            clean_r = [re.sub(r"\s+", " ", str(c or "").strip()) for c in r]
            if any(clean_r):
                data_rows.append(clean_r)

        # Refine bounding box if tab_rows is available
        if tab_rows and len(tab_rows) > best_row_idx and tab_bbox:
            top_y = tab_rows[best_row_idx].bbox[1]
            last_row_idx = min(best_row_idx + len(data_rows), len(tab_rows) - 1)
            bottom_y = tab_rows[last_row_idx].bbox[3]
            refined_bbox = (tab_bbox[0], top_y, tab_bbox[2], bottom_y)
        else:
            refined_bbox = tab_bbox or (0.0, 0.0, 0.0, 0.0)

        logger.info(
            "TableHeaderAndBoundaryDetector: header detected at row %d (score=%.2f) -> %s",
            best_row_idx, best_score, headers,
        )
        return headers, data_rows, refined_bbox, best_row_idx


# ── 2. Hierarchical Header Normalizer ─────────────────────────────────────────

class HierarchicalHeaderNormalizer:
    """Detects and flattens multi-tier/hierarchical header rows into composite column names."""

    @classmethod
    def normalize_headers(cls, headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]], bool]:
        """Check if the first row of data is actually a secondary header row.

        Returns:
            Tuple of (normalized_headers, remaining_rows, was_multi_tier).
        """
        if not headers or not rows:
            return headers, rows, False

        # Heuristics for 2-tier header:
        # Top tier MUST have empty/spanned cells (e.g. ["Revenue", "", "Expenses", ""])
        # while some headers are populated. If all headers are populated or all are synthetic Column_N,
        # it is not a multi-tier header.
        has_spanned_top = (
            any(not (h or "").strip() for h in headers)
            and any(bool((h or "").strip()) for h in headers)
        )
        if not has_spanned_top:
            return headers, rows, False

        candidate_sub_header = rows[0]
        all_short_in_sub = all(len((c or "").strip()) < 35 for c in candidate_sub_header)
        if all_short_in_sub and len(candidate_sub_header) == len(headers):
            # Forward-fill top tier headers across empty spans
            filled_top = []
            curr_top = ""
            for h in headers:
                clean_h = (h or "").strip()
                if clean_h:
                    curr_top = clean_h
                filled_top.append(curr_top)

            # Combine top tier + sub tier
            composite_headers = []
            for top_h, sub_h in zip(filled_top, candidate_sub_header):
                clean_sub = (sub_h or "").strip()
                if top_h and clean_sub and top_h != clean_sub:
                    composite_headers.append(f"{top_h} > {clean_sub}")
                else:
                    composite_headers.append(clean_sub or top_h or "Column")

            logger.info("HierarchicalHeaderNormalizer: detected 2-tier header -> %s", composite_headers)
            return composite_headers, rows[1:], True

        return headers, rows, False


# ── 3. Cell Inheritance / Forward-Fill Processor ──────────────────────────────

class CellInheritanceProcessor:
    """Forward-fills empty cells in hierarchical category/spanning columns."""

    @staticmethod
    def apply_inheritance(headers: List[str], rows: List[List[str]]) -> List[List[str]]:
        """Forward-fill empty values in the first (grouping) column when structural evidence supports it."""
        if not rows or len(rows) < 2:
            return rows

        # Only apply if column 0 has a mix of populated and empty cells
        col0_vals = [(r[0] if r else "").strip() for r in rows]
        has_empties = any(not v for v in col0_vals)
        has_values = any(bool(v) for v in col0_vals)

        if not (has_empties and has_values):
            return rows

        # Verify that subsequent columns on empty col0 rows actually have data
        inherited_rows: List[List[str]] = []
        current_category = ""

        for row in rows:
            if not row:
                continue
            new_row = list(row)
            col0_val = (new_row[0] or "").strip()

            if col0_val:
                current_category = col0_val
            elif current_category and any((c or "").strip() for c in new_row[1:]):
                new_row[0] = current_category

            inherited_rows.append(new_row)

        return inherited_rows


# ── 4. Table Validator ────────────────────────────────────────────────────────

class TableValidator:
    """Validates structural and content integrity of an extracted table."""

    @staticmethod
    def validate(
        table: TableElement,
        enable_hierarchical_headers: bool = True,
        enable_cell_inheritance: bool = True,
    ) -> TableElement:
        """Run validation rules, multi-tier header normalization, and cell inheritance."""
        issues: List[TableValidationIssue] = []

        # Rule 0: Header quality validation (reject long paragraphs or low-uniqueness headers)
        if table.headers:
            avg_len = sum(len(h) for h in table.headers) / len(table.headers)
            max_len = max(len(h) for h in table.headers)
            is_paragraph_header = avg_len > 80 or max_len > 140 or any("\n\n" in h for h in table.headers)
            is_repeated_header = len(set(table.headers)) == 1 and len(table.headers) > 1

            if is_paragraph_header or is_repeated_header:
                issues.append(
                    TableValidationIssue(
                        issue_type="INVALID_PROSE_HEADER_REPLACED",
                        message=f"Rejected invalid prose/repeated header: '{table.headers[0][:50]}...'",
                        severity="warning",
                    )
                )
                max_cols = max(len(table.headers), max((len(r) for r in table.rows), default=0))
                table.headers = [f"Column_{i+1}" for i in range(max_cols)]
                table.num_cols = max_cols

        # Optional: Hierarchical 2-tier header detection (run before fallback column naming)
        if enable_hierarchical_headers and table.headers and table.rows:
            new_headers, remaining_rows, was_multi = HierarchicalHeaderNormalizer.normalize_headers(
                table.headers, table.rows
            )
            if was_multi:
                table.headers = new_headers
                table.rows = remaining_rows
                table.num_cols = len(new_headers)
                issues.append(
                    TableValidationIssue(
                        issue_type="HIERARCHICAL_HEADER_NORMALIZED",
                        message=f"Normalized 2-tier header into composite column names: {new_headers}",
                        severity="info",
                    )
                )

        # Rule 1: Check for empty headers or missing header list
        if not table.headers:
            issues.append(
                TableValidationIssue(
                    issue_type="MISSING_HEADERS",
                    message="Table has no detected column headers; generated default column names.",
                    severity="warning",
                )
            )
            max_cols = max((len(r) for r in table.rows), default=0)
            table.headers = [f"Column_{i+1}" for i in range(max_cols)]
            table.num_cols = max_cols
        else:
            cleaned_headers = []
            for i, h in enumerate(table.headers):
                clean_h = re.sub(r"\s+", " ", (h or "").strip())
                if not clean_h:
                    clean_h = f"Column_{i+1}"
                    issues.append(
                        TableValidationIssue(
                            issue_type="EMPTY_HEADER_CELL",
                            message=f"Header at column index {i+1} was empty; renamed to {clean_h}.",
                            severity="warning",
                            col_idx=i,
                        )
                    )
                cleaned_headers.append(clean_h)
            table.headers = cleaned_headers
            table.num_cols = len(cleaned_headers)

        # Rule 2: Check for empty table
        if not table.rows:
            issues.append(
                TableValidationIssue(
                    issue_type="EMPTY_TABLE",
                    message="Table contains 0 data rows.",
                    severity="warning",
                )
            )
            table.validation_issues = issues
            table.validation_status = "warning"
            table.num_rows = 0
            return table

        # Optional: Cell inheritance forward-fill
        if enable_cell_inheritance and table.rows:
            table.rows = CellInheritanceProcessor.apply_inheritance(table.headers, table.rows)

        expected_cols = table.num_cols
        normalized_rows: List[List[str]] = []

        # Rule 3: Check row lengths and normalize
        for r_idx, raw_row in enumerate(table.rows):
            clean_row = [re.sub(r"\s+", " ", (c or "").strip()) for c in raw_row]

            if len(clean_row) != expected_cols:
                issues.append(
                    TableValidationIssue(
                        issue_type="COLUMN_COUNT_MISMATCH",
                        message=(
                            f"Row {r_idx+1} has {len(clean_row)} columns, "
                            f"expected {expected_cols}."
                        ),
                        severity="warning",
                        row_idx=r_idx,
                    )
                )
                if len(clean_row) < expected_cols:
                    clean_row.extend([""] * (expected_cols - len(clean_row)))
                else:
                    clean_row = clean_row[:expected_cols]

            normalized_rows.append(clean_row)

        table.rows = normalized_rows
        table.num_rows = len(normalized_rows)
        table.validation_issues = issues
        table.validation_status = "warning" if any(i.severity == "warning" for i in issues) else "valid"

        return table


# ── 5. Multi-Page Table Continuation Stitcher ─────────────────────────────────

class TableContinuationStitcher:
    """Detects multi-page table fragments continuing across consecutive pages and merges them."""

    @classmethod
    def stitch_multipage_tables(
        cls,
        tables: List[TableElement],
        confidence_threshold: float = 0.85,
    ) -> List[TableElement]:
        """Examine list of tables in document order and merge confident continuation fragments."""
        if len(tables) < 2:
            return tables

        stitched: List[TableElement] = []
        skip_indices = set()

        for i in range(len(tables)):
            if i in skip_indices:
                continue

            current_table = tables[i]

            # Look ahead for continuations on subsequent pages
            for j in range(i + 1, len(tables)):
                if j in skip_indices:
                    continue

                next_table = tables[j]
                if next_table.page_number != current_table.page_end + 1:
                    break

                is_continuation, confidence = cls._evaluate_continuation(current_table, next_table)

                if is_continuation and confidence >= confidence_threshold:
                    logger.info(
                        "TableContinuationStitcher: merging table '%s' (pg %d) into '%s' (pg %d..%d) [confidence=%.2f]",
                        next_table.table_id, next_table.page_number,
                        current_table.table_id, current_table.page_start, current_table.page_end,
                        confidence,
                    )
                    current_table.rows.extend(next_table.rows)
                    current_table.num_rows = len(current_table.rows)
                    current_table.page_end = next_table.page_number
                    current_table.is_continuation = True
                    current_table.continuation_confidence = confidence

                    skip_indices.add(j)
                else:
                    break

            stitched.append(current_table)

        return stitched

    @classmethod
    def _evaluate_continuation(cls, table_a: TableElement, table_b: TableElement) -> Tuple[bool, float]:
        """Compute multi-factor confidence score for table continuation across pages."""
        score = 0.0

        # Signal 1: Column count match (30% weight)
        if table_a.num_cols == table_b.num_cols and table_a.num_cols > 0:
            score += 0.30
        else:
            return False, 0.0

        # Signal 2: Header similarity (35% weight)
        headers_a_str = " | ".join(h.lower().strip() for h in table_a.headers)
        headers_b_str = " | ".join(h.lower().strip() for h in table_b.headers)

        if headers_a_str == headers_b_str and headers_a_str:
            score += 0.35
        else:
            similarity = SequenceMatcher(None, headers_a_str, headers_b_str).ratio()
            score += 0.35 * similarity

        # Signal 3: Section continuity (20% weight)
        if table_a.section and table_b.section and table_a.section == table_b.section:
            score += 0.20
        elif not table_b.section or table_b.section == table_a.section:
            score += 0.15

        # Signal 4: Geometry (Table A near bottom, Table B near top of page) (15% weight)
        if table_a.bbox and table_b.bbox:
            a_bottom = table_a.bbox[3]
            b_top = table_b.bbox[1]
            if a_bottom > 450 and b_top < 200:
                score += 0.15
            elif b_top < 200:
                score += 0.10
        else:
            score += 0.10

        return score >= 0.80, round(score, 4)


# ── 6. Table Serializer ───────────────────────────────────────────────────────

class TableSerializer:
    """Serializes structured table elements into text formats optimized for LLM/RAG."""

    @staticmethod
    def to_markdown(table: TableElement, rows_subset: Optional[List[List[str]]] = None) -> str:
        """Render table or row subset as a standard Markdown table."""
        headers = table.headers or [f"Col_{i+1}" for i in range(table.num_cols)]
        rows = rows_subset if rows_subset is not None else table.rows

        if not headers:
            return ""

        escaped_headers = [h.replace("|", "\\|") for h in headers]
        header_line = "| " + " | ".join(escaped_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"

        row_lines = []
        for r in rows:
            padded_row = r + [""] * (len(headers) - len(r))
            escaped_cells = [cell.replace("|", "\\|") for cell in padded_row[:len(headers)]]
            row_lines.append("| " + " | ".join(escaped_cells) + " |")

        return "\n".join([header_line, sep_line] + row_lines)

    @staticmethod
    def to_row_key_value(headers: List[str], row: List[str], row_num: Optional[int] = None) -> str:
        """Render a single row as key-value pairs."""
        lines = []
        if row_num is not None:
            lines.append(f"• Row {row_num}:")
        for i, val in enumerate(row):
            col_name = headers[i] if i < len(headers) else f"Column_{i+1}"
            lines.append(f"  {col_name}: {val or '(empty)'}")
        return "\n".join(lines)

    @classmethod
    def to_hybrid(
        cls,
        table: TableElement,
        rows_subset: List[List[str]],
        row_start: int,
        total_rows: int,
        document_title: Optional[str] = None,
        column_slice_label: Optional[str] = None,
    ) -> str:
        """Generate high-density hybrid representation: Context + Schema + Markdown Table + Key-Values."""
        parts: List[str] = []

        # 1. Context Breadcrumb
        context_parts = []
        if document_title:
            context_parts.append(f"Document: {document_title}")
        if table.page_start != table.page_end:
            context_parts.append(f"Pages {table.page_start}–{table.page_end}")
        else:
            context_parts.append(f"Page {table.page_number}")
        if table.context_hierarchy:
            context_parts.append(" > ".join(table.context_hierarchy))
        elif table.section:
            context_parts.append(f"Section: {table.section}")
        if table.caption:
            context_parts.append(f"Table: {table.caption}")
        else:
            context_parts.append(f"Table: {table.table_id}")

        if context_parts:
            parts.append(f"[{' | '.join(context_parts)}]")

        # 2. Preceding descriptive context
        if table.preceding_text:
            parts.append(f"Context: {table.preceding_text}")

        # 3. Table Overview & Row/Column Range
        row_end = row_start + len(rows_subset) - 1
        range_desc = f"(Rows {row_start}–{row_end} of {total_rows}"
        if column_slice_label:
            range_desc += f", {column_slice_label}"
        range_desc += ")"

        parts.append(f"Table: {table.caption or table.table_id} {range_desc}")
        parts.append(f"Columns: {', '.join(table.headers)}")

        # 4. Markdown View
        md_view = cls.to_markdown(table, rows_subset=rows_subset)
        if md_view:
            parts.append("\n" + md_view)

        # 5. Row-by-Row Key-Value View
        kv_lines = ["\nDetailed Row Records:"]
        for idx, row in enumerate(rows_subset):
            current_row_num = row_start + idx
            kv_lines.append(cls.to_row_key_value(table.headers, row, row_num=current_row_num))
        parts.append("\n".join(kv_lines))

        return "\n".join(parts)

    @classmethod
    def to_row_grouped(
        cls,
        table: TableElement,
        rows_subset: List[List[str]],
        row_start: int,
        total_rows: int,
        document_title: Optional[str] = None,
    ) -> str:
        """Render row-grouped key-value format."""
        parts: List[str] = []
        header_desc = []
        if document_title:
            header_desc.append(f"Document: {document_title}")
        if table.section:
            header_desc.append(f"Section: {table.section}")
        header_desc.append(f"Table: {table.caption or table.table_id}")
        parts.append(f"[{' | '.join(header_desc)}]")

        row_end = row_start + len(rows_subset) - 1
        parts.append(f"Rows {row_start}–{row_end} of {total_rows}")
        parts.append(f"Columns: {', '.join(table.headers)}\n")

        for idx, row in enumerate(rows_subset):
            parts.append(cls.to_row_key_value(table.headers, row, row_num=row_start + idx))
            parts.append("")

        return "\n".join(parts)

    @classmethod
    def to_json(
        cls,
        table: TableElement,
        rows_subset: Optional[List[List[str]]] = None,
    ) -> str:
        """Render table rows as JSON string."""
        rows = rows_subset if rows_subset is not None else table.rows
        records = []
        for row in rows:
            rec = {
                table.headers[i] if i < len(table.headers) else f"Column_{i+1}": val
                for i, val in enumerate(row)
            }
            records.append(rec)
        return json.dumps(records, indent=2)


# ── 7. Token-Safe Table Summary Generator ─────────────────────────────────────

def generate_table_summary(
    table: TableElement,
    document_title: Optional[str] = None,
    max_tokens: int = 512,
    encoding_model: str = "cl100k_base",
) -> str:
    """Generate a deterministic, token-budget-safe summary chunk for the entire table."""
    try:
        enc = tiktoken.get_encoding(encoding_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    parts = []

    # Title & Location
    breadcrumb = []
    if document_title:
        breadcrumb.append(f"Document: {document_title}")
    if table.section:
        breadcrumb.append(f"Section: {table.section}")
    if table.page_start != table.page_end:
        breadcrumb.append(f"Pages {table.page_start}–{table.page_end} (Multi-Page Table)")
    else:
        breadcrumb.append(f"Page {table.page_number}")
    parts.append(f"[{' | '.join(breadcrumb)}]")

    table_name = table.caption or table.table_id
    parts.append(f"Table Overview: {table_name}")

    if table.preceding_text:
        parts.append(f"Description / Context: {table.preceding_text}")

    parts.append(f"Schema: {table.num_rows} rows × {table.num_cols} columns")
    parts.append(f"Columns: {', '.join(table.headers)}")

    if table.rows:
        sample_count = min(3, len(table.rows))
        sample_rows = table.rows[:sample_count]
        sample_parts = [f"\nSample Data (First {sample_count} rows):"]
        for i, r in enumerate(sample_rows, 1):
            sample_parts.append(TableSerializer.to_row_key_value(table.headers, r, row_num=i))

        candidate_full = "\n".join(parts + sample_parts)
        if len(enc.encode(candidate_full)) <= max_tokens:
            parts.extend(sample_parts)

    if table.validation_issues:
        issues_summary = ", ".join(i.issue_type for i in table.validation_issues[:3])
        parts.append(f"Quality Notes: {issues_summary}")

    result = "\n".join(parts)

    tokens = enc.encode(result)
    if len(tokens) > max_tokens:
        result = enc.decode(tokens[:max_tokens])

    return result
