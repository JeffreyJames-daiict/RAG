"""Comprehensive test suite for advanced multi-table RAG features:
- Multi-table isolation (6 tables across 3 pages)
- Multi-page table continuation stitching
- Hierarchical 2-tier header normalization
- Empty cell inheritance forward-fill
- Wide table column slicing with repeated entity anchors
- Bounded parent table retrieval with neighbor windows
- Failure isolation and diagnostic reporting
"""

from __future__ import annotations

import uuid
from typing import List

import pytest

from rag.core.config import TableConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Base, Chunk, Document
from rag.parsing.models import (
    DocumentElement,
    ElementType,
    ParsedDocument,
    ParsedPage,
    TableElement,
    TextElement,
)
from rag.parsing.table_chunker import TableChunker
from rag.parsing.table_processor import (
    CellInheritanceProcessor,
    HierarchicalHeaderNormalizer,
    TableContinuationStitcher,
    TableSerializer,
    TableValidator,
    generate_table_summary,
)
from rag.retrieval import RetrievalFilter, TableAwareRetriever


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-Table Isolation Test (6 tables across 3 pages)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTableIsolation:
    def test_six_tables_across_three_pages_completely_isolated(self):
        doc_id = str(uuid.uuid4())
        chunker = TableChunker(table_config=TableConfig(summary_enabled=False))

        tables: List[TableElement] = []
        for p in [1, 2, 3]:
            for t in [1, 2]:
                table_id = f"doc_p{p}_t{t}"
                headers = [f"P{p}T{t}_Col1", f"P{p}T{t}_Col2"]
                rows = [[f"P{p}T{t}_R1C1", f"P{p}T{t}_R1C2"], [f"P{p}T{t}_R2C1", f"P{p}T{t}_R2C2"]]
                elem = TableElement(
                    element_id=table_id,
                    table_id=table_id,
                    page_number=p,
                    section=f"Section Page {p}",
                    headers=headers,
                    rows=rows,
                )
                tables.append(elem)

        all_chunks = []
        for t in tables:
            chunks = chunker.chunk_table(t, document_id=doc_id, version=1, starting_chunk_index=len(all_chunks))
            all_chunks.extend(chunks)

        # 6 tables * 1 chunk each = 6 chunks
        assert len(all_chunks) == 6
        table_ids = [c.metadata["table_id"] for c in all_chunks]
        assert len(set(table_ids)) == 6  # Zero duplicate table IDs

        # Ensure no cross-table row leakage
        for idx, c in enumerate(all_chunks):
            expected_p = (idx // 2) + 1
            expected_t = (idx % 2) + 1
            expected_prefix = f"P{expected_p}T{expected_t}"
            assert expected_prefix in c.text
            # Should NOT contain other tables' prefixes
            for other_p in [1, 2, 3]:
                for other_t in [1, 2]:
                    if (other_p, other_t) != (expected_p, expected_t):
                        assert f"P{other_p}T{other_t}_R1C1" not in c.text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Page Continuation Stitching Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTableContinuationStitching:
    def test_stitch_contiguous_tables_across_pages(self):
        # Table 1 on page 1 bottom
        t1 = TableElement(
            element_id="doc_p1_t1",
            table_id="doc_p1_t1",
            page_number=1,
            page_start=1,
            page_end=1,
            bbox=(50, 500, 550, 750),  # near bottom
            section="Financials",
            caption="Balance Sheet",
            headers=["Item", "2023", "2024"],
            rows=[["Cash", "$100", "$150"], ["Assets", "$500", "$600"]],
        )
        # Table 2 on page 2 top (same headers & section)
        t2 = TableElement(
            element_id="doc_p2_t1",
            table_id="doc_p2_t1",
            page_number=2,
            page_start=2,
            page_end=2,
            bbox=(50, 50, 550, 300),   # near top
            section="Financials",
            caption="Balance Sheet",
            headers=["Item", "2023", "2024"],
            rows=[["Liabilities", "$200", "$250"], ["Equity", "$300", "$350"]],
        )

        stitched = TableContinuationStitcher.stitch_multipage_tables([t1, t2], confidence_threshold=0.85)

        assert len(stitched) == 1
        merged = stitched[0]
        assert merged.table_id == "doc_p1_t1"
        assert merged.page_start == 1
        assert merged.page_end == 2
        assert merged.is_continuation is True
        assert merged.num_rows == 4
        assert [r[0] for r in merged.rows] == ["Cash", "Assets", "Liabilities", "Equity"]

    def test_do_not_stitch_unrelated_tables_on_adjacent_pages(self):
        t1 = TableElement(
            element_id="doc_p1_t1",
            table_id="doc_p1_t1",
            page_number=1,
            headers=["Product", "Price"],
            rows=[["Mug", "$15"]],
            section="Shop Catalog",
        )
        t2 = TableElement(
            element_id="doc_p2_t1",
            table_id="doc_p2_t1",
            page_number=2,
            headers=["Employee", "Department", "Salary"],  # different columns & count
            rows=[["Alice", "Eng", "$100k"]],
            section="Internal Staff",
        )

        stitched = TableContinuationStitcher.stitch_multipage_tables([t1, t2], confidence_threshold=0.85)
        assert len(stitched) == 2  # Kept strictly independent


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hierarchical Header Normalization Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHierarchicalHeaders:
    def test_normalize_two_tier_headers(self):
        raw_headers = ["Revenue", "", "Expenses", ""]
        raw_rows = [
            ["2023", "2024", "2023", "2024"],  # Sub-header tier
            ["$10M", "$15M", "$6M", "$8M"],     # Real data row
        ]

        normalized_headers, remaining_rows, was_multi = HierarchicalHeaderNormalizer.normalize_headers(
            raw_headers, raw_rows
        )

        assert was_multi is True
        assert normalized_headers == [
            "Revenue > 2023",
            "Revenue > 2024",
            "Expenses > 2023",
            "Expenses > 2024",
        ]
        assert len(remaining_rows) == 1
        assert remaining_rows[0] == ["$10M", "$15M", "$6M", "$8M"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Empty Cell Inheritance / Forward-Fill Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCellInheritance:
    def test_forward_fill_hierarchical_category_cells(self):
        headers = ["Category", "Product", "Price"]
        rows = [
            ["Electronics", "Laptop", "$1000"],
            ["", "Mouse", "$25"],
            ["", "Keyboard", "$50"],
            ["Furniture", "Desk", "$200"],
            ["", "Chair", "$100"],
        ]

        filled_rows = CellInheritanceProcessor.apply_inheritance(headers, rows)

        assert filled_rows[0] == ["Electronics", "Laptop", "$1000"]
        assert filled_rows[1] == ["Electronics", "Mouse", "$25"]
        assert filled_rows[2] == ["Electronics", "Keyboard", "$50"]
        assert filled_rows[3] == ["Furniture", "Desk", "$200"]
        assert filled_rows[4] == ["Furniture", "Chair", "$100"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Wide Table Column Slicing Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWideTableSlicing:
    def test_wide_table_sliced_with_anchor_column_repeated(self):
        config = TableConfig(
            wide_table_threshold=5,
            summary_enabled=False,
        )
        chunker = TableChunker(table_config=config, max_tokens_per_chunk=2000)

        # 12 columns table
        headers = ["Company"] + [f"Metric_{i}" for i in range(1, 12)]
        rows = [["Acme Corp"] + [f"val_{i}" for i in range(1, 12)]]

        table = TableElement(
            element_id="wide_t1",
            table_id="wide_t1",
            page_number=1,
            headers=headers,
            rows=rows,
        )

        doc_id = str(uuid.uuid4())
        chunks = chunker.chunk_table(table, document_id=doc_id, version=1)

        # Should be sliced into multiple column groups
        assert len(chunks) > 1
        for c in chunks:
            # Every slice must contain the anchor column header and value
            assert "Company" in c.text
            assert "Acme Corp" in c.text
            assert "column_slice" in c.metadata


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bounded Parent Table Expansion Retrieval Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedParentExpansion:
    @pytest.fixture
    def setup_retriever_large_table(self):
        db = DatabaseManager("sqlite:///:memory:")
        from rag.core.migrate import ensure_raw_schema
        ensure_raw_schema(db.engine)
        Base.metadata.create_all(db.engine)
        doc_id = str(uuid.uuid4())

        with db.session() as s:
            doc = Document(
                document_id=doc_id,
                tenant_id="default",
                source="LOCAL_FILE",
                source_uri="/docs/big_table.pdf",
                filename="big_table.pdf",
                mime_type="application/pdf",
                content_hash="hash_big",
                version=1,
                status="CHUNKED",
            )
            s.add(doc)

            # Insert summary chunk (chunk #0)
            summary_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                version=1,
                chunk_index=0,
                text="Table Overview: 100 Products Catalog",
                chunk_metadata={"chunk_type": "table_summary", "table_id": "catalog_t1", "parent_id": "catalog_t1"},
            )
            s.add(summary_chunk)

            # Insert 10 row chunks (each spanning 10 rows: 1..10, 11..20, ... 91..100)
            for i in range(1, 11):
                r_start = (i - 1) * 10 + 1
                r_end = i * 10
                c = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc_id,
                    version=1,
                    chunk_index=i,
                    text=f"Table: Catalog (Rows {r_start}-{r_end})\n• Product {r_start}: Details...",
                    chunk_metadata={
                        "chunk_type": "table",
                        "table_id": "catalog_t1",
                        "parent_id": "catalog_t1",
                        "row_start": r_start,
                        "row_end": r_end,
                    },
                )
                s.add(c)

        return TableAwareRetriever(db), doc_id

    def test_bounded_neighbor_expansion(self, setup_retriever_large_table):
        retriever, doc_id = setup_retriever_large_table

        # Target chunk is #5 (Rows 41-50), with neighbor_window=1 -> should return #4, #5, #6 + summary
        expanded = retriever.expand_parent_table(
            table_id="catalog_t1",
            document_id=doc_id,
            target_chunk_index=5,
            neighbor_window=1,
        )

        # Expected: summary (#0) + #4 + #5 + #6 = 4 chunks
        assert len(expanded) == 4
        chunk_indices = [c.chunk_index for c in expanded]
        assert 0 in chunk_indices
        assert 4 in chunk_indices
        assert 5 in chunk_indices
        assert 6 in chunk_indices
        # Chunks #1, #2, #8, #9 should NOT be included
        assert 1 not in chunk_indices
        assert 9 not in chunk_indices

    def test_max_rows_ceiling_in_expansion(self, setup_retriever_large_table):
        retriever, doc_id = setup_retriever_large_table

        # Limit expansion to max 25 rows (each chunk is 10 rows -> max 2 row chunks + summary)
        expanded = retriever.expand_parent_table(
            table_id="catalog_t1",
            document_id=doc_id,
            max_rows=25,
        )
        row_chunks = [c for c in expanded if c.chunk_type == "table"]
        assert len(row_chunks) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# 7. Failure Isolation Test
# ─────────────────────────────────────────────────────────────────────────────

class TestFailureIsolation:
    def test_malformed_table_validation_warning_non_fatal(self):
        malformed_table = TableElement(
            element_id="bad_t1",
            table_id="bad_t1",
            page_number=1,
            headers=[],
            rows=[],  # empty headers and empty rows
        )
        validated = TableValidator.validate(malformed_table)

        assert validated.validation_status == "warning"
        assert len(validated.validation_issues) >= 1

        # Chunker should still handle without crashing
        chunker = TableChunker()
        chunks = chunker.chunk_table(validated, document_id=str(uuid.uuid4()), version=1)
        assert len(chunks) == 1
        assert "Empty" in chunks[0].text


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dedicated Phase 9 Regression Tests
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from rag.parsing.pdf_parser import PDFParser
from rag.parsing.table_processor import TableHeaderAndBoundaryDetector


class TestHeaderValidationAndBoundaryRegressions:
    def test_1_normal_table_with_valid_headers(self):
        raw_matrix = [
            ["Product", "Price", "Stock"],
            ["Widget A", "$10", "100"],
            ["Widget B", "$20", "50"],
        ]
        headers, data_rows, _, header_idx = TableHeaderAndBoundaryDetector.detect_header_and_trim(raw_matrix)
        assert header_idx == 0
        assert headers == ["Product", "Price", "Stock"]
        assert len(data_rows) == 2

    def test_2_long_paragraph_rejected_as_header(self):
        long_paragraph = (
            "2.3 Data & backend Style, rim/handle color, and inner color become selectable "
            "configuration options on the product (same option pattern already used for other products). "
            "Each order stores: style, rim/handle color, inner color, image, crop data."
        )
        raw_matrix = [
            [long_paragraph, None, None],
            ["Product", "Format", "Price"],
            ["Calendar", "A4", "$15"],
        ]
        headers, data_rows, _, header_idx = TableHeaderAndBoundaryDetector.detect_header_and_trim(raw_matrix)
        assert header_idx == 1  # Row 0 rejected, Row 1 selected
        assert headers == ["Product", "Format", "Price"]
        assert len(data_rows) == 1
        assert long_paragraph not in headers

    def test_3_repeated_identical_paragraph_across_columns_rejected(self):
        repeated = "Generic disclaimer repeated across all columns in header."
        raw_matrix = [
            [repeated, repeated, repeated],
            ["ID", "Name", "Score"],
            ["1", "Alice", "95"],
        ]
        headers, data_rows, _, header_idx = TableHeaderAndBoundaryDetector.detect_header_and_trim(raw_matrix)
        assert header_idx == 1  # Row 0 rejected because of 0% distinctness
        assert headers == ["ID", "Name", "Score"]

    def test_4_table_with_vertically_merged_cells_inheritance(self):
        headers = ["Product", "Format"]
        rows = [
            ["Calendar", "A4 portrait"],
            ["", "A3 portrait"],
            ["Mug", "Standard"],
            ["", "Large"],
        ]
        inherited = CellInheritanceProcessor.apply_inheritance(headers, rows)
        assert inherited[0] == ["Calendar", "A4 portrait"]
        assert inherited[1] == ["Calendar", "A3 portrait"]
        assert inherited[2] == ["Mug", "Standard"]
        assert inherited[3] == ["Mug", "Large"]

    def test_5_table_with_no_reliable_header_generates_fallbacks(self):
        # Numeric / paragraph data with no clear header
        raw_matrix = [
            ["100", "200", "300"],
            ["400", "500", "600"],
        ]
        table = TableElement(
            element_id="t_num",
            table_id="t_num",
            page_number=1,
            headers=[],
            rows=raw_matrix,
        )
        validated = TableValidator.validate(table)
        assert validated.headers == ["Column_1", "Column_2", "Column_3"]
        assert len(validated.rows) == 2

    def test_6_nearby_document_text_remains_outside_table(self):
        pdf_path = Path("data/FridaySnap_New_Requirements-1.pdf")
        if not pdf_path.exists():
            pytest.skip("FridaySnap PDF not found in data/")

        parser = PDFParser(extract_tables=True)
        parsed = parser.parse(pdf_path.read_bytes(), filename="FridaySnap_New_Requirements-1.pdf")

        # Page 2 should have text elements containing "2.3 Data & backend" and "Integrate the Allcop"
        p2 = parsed.pages[1]
        p2_text = "\n".join(elem.content for elem in p2.elements if elem.element_type.value != "table")
        assert "2.3 Data & backend" in p2_text
        assert "Integrate the Allcop" in p2_text
        assert "Six calendar types" in p2_text

    def test_7_friday_snap_page_2_regression_ground_truth(self):
        pdf_path = Path("data/FridaySnap_New_Requirements-1.pdf")
        if not pdf_path.exists():
            pytest.skip("FridaySnap PDF not found in data/")

        parser = PDFParser(extract_tables=True)
        parsed = parser.parse(pdf_path.read_bytes(), filename="FridaySnap_New_Requirements-1.pdf")

        assert len(parsed.tables) >= 1
        t = parsed.tables[0]

        # 1. Exact Ground Truth Headers
        assert t.headers == [
            "Product",
            "Format",
            "Matt",
            "Glossy (Hochglanz veredelt)",
            "Recycling paper",
        ]

        # 2. Exact 15 Data Rows
        assert len(t.rows) == 15

        # 3. Product Inheritance Verification
        # Row 7 & 8: Appointment (wall) calendar
        assert t.rows[6][0] == "Appointment (wall) calendar"
        assert t.rows[6][1] == "A4 portrait"
        assert t.rows[7][0] == "Appointment (wall) calendar"
        assert t.rows[7][1] == "A3 portrait"

        # Row 9 & 10: Family calendar
        assert t.rows[8][0] == "Family calendar"
        assert t.rows[9][0] == "Family calendar"

        # 4. No paragraph text in headers
        assert not any("2.3 Data & backend" in h for h in t.headers)
        assert not any("Allcop" in h for h in t.headers)

