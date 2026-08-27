"""Comprehensive test suite for multi-table document processing, extraction, chunking, and retrieval."""

from __future__ import annotations

import io
import uuid
from typing import List

import pytest

try:
    import pymupdf as fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from rag.core.config import TableConfig
from rag.core.database import DatabaseManager
from rag.core.db_models import Base, Chunk, Document, DocumentVersion
from rag.parsing.models import (
    DocumentElement,
    ElementType,
    HeadingElement,
    ParsedDocument,
    ParsedPage,
    TableElement,
    TextElement,
)
from rag.parsing.table_chunker import TableChunker
from rag.parsing.table_processor import (
    TableSerializer,
    TableValidator,
    generate_table_summary,
)
from rag.retrieval import RetrievalFilter, TableAwareRetriever


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_sample_table(num_rows: int = 5, num_cols: int = 3, table_id: str = "doc_p1_t1") -> TableElement:
    headers = [f"Col_{i+1}" for i in range(num_cols)]
    rows = [[f"Val_{r}_{c}" for c in range(1, num_cols + 1)] for r in range(1, num_rows + 1)]
    return TableElement(
        element_id=table_id,
        table_id=table_id,
        page_number=1,
        section="Financial Results",
        caption="Q3 Summary Table",
        headers=headers,
        rows=rows,
        context_hierarchy=["Annual Report", "Financial Results"],
        preceding_text="The following table details quarterly figures.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. TableValidator Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTableValidator:
    def test_validate_clean_table(self):
        table = _make_sample_table(num_rows=3, num_cols=3)
        validated = TableValidator.validate(table)
        assert validated.validation_status == "valid"
        assert len(validated.validation_issues) == 0
        assert validated.num_rows == 3
        assert validated.num_cols == 3

    def test_validate_missing_headers_generates_fallbacks(self):
        table = TableElement(
            element_id="t1",
            table_id="t1",
            page_number=1,
            headers=[],
            rows=[["A", "B", "C"], ["D", "E", "F"]],
        )
        validated = TableValidator.validate(table)
        assert validated.validation_status == "warning"
        assert any(i.issue_type == "MISSING_HEADERS" for i in validated.validation_issues)
        assert validated.headers == ["Column_1", "Column_2", "Column_3"]

    def test_validate_inconsistent_columns_padded(self):
        table = TableElement(
            element_id="t2",
            table_id="t2",
            page_number=1,
            headers=["Name", "Age", "City"],
            rows=[
                ["Alice", "30", "NYC"],
                ["Bob"],  # Missing 2 columns
            ],
        )
        validated = TableValidator.validate(table)
        assert validated.validation_status == "warning"
        assert any(i.issue_type == "COLUMN_COUNT_MISMATCH" for i in validated.validation_issues)
        assert validated.rows[1] == ["Bob", "", ""]  # Padded to 3 columns

    def test_validate_empty_table(self):
        table = TableElement(
            element_id="t3",
            table_id="t3",
            page_number=1,
            headers=["A", "B"],
            rows=[],
        )
        validated = TableValidator.validate(table)
        assert validated.validation_status == "warning"
        assert any(i.issue_type == "EMPTY_TABLE" for i in validated.validation_issues)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TableSerializer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTableSerializer:
    def test_to_markdown_output(self):
        table = _make_sample_table(num_rows=2, num_cols=2)
        md = TableSerializer.to_markdown(table)
        assert "| Col_1 | Col_2 |" in md
        assert "| --- | --- |" in md
        assert "| Val_1_1 | Val_1_2 |" in md

    def test_to_row_key_value(self):
        headers = ["Product", "Price"]
        row = ["Mug", "$15"]
        kv = TableSerializer.to_row_key_value(headers, row, row_num=1)
        assert "Product: Mug" in kv
        assert "Price: $15" in kv

    def test_to_hybrid_contains_context_and_details(self):
        table = _make_sample_table(num_rows=3, num_cols=2)
        hybrid = TableSerializer.to_hybrid(
            table=table,
            rows_subset=table.rows,
            row_start=1,
            total_rows=3,
            document_title="Sample Doc",
        )
        assert "Document: Sample Doc" in hybrid
        assert "Financial Results" in hybrid
        assert "Col_1" in hybrid
        assert "| Val_1_1 | Val_1_2 |" in hybrid
        assert "Detailed Row Records:" in hybrid

    def test_generate_table_summary(self):
        table = _make_sample_table(num_rows=10, num_cols=3)
        summary = generate_table_summary(table, document_title="Corporate Overview")
        assert "Corporate Overview" in summary
        assert "Schema: 10 rows × 3 columns" in summary
        assert "Col_1, Col_2, Col_3" in summary
        assert "Sample Data" in summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. TableChunker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTableChunker:
    def test_small_table_single_chunk(self):
        config = TableConfig(summary_enabled=False, max_rows_per_chunk=10)
        chunker = TableChunker(table_config=config, max_tokens_per_chunk=1000)
        table = _make_sample_table(num_rows=3, num_cols=3)
        chunks = chunker.chunk_table(table, document_id=str(uuid.uuid4()), version=1)

        assert len(chunks) == 1
        assert chunks[0].metadata["chunk_type"] == "table"
        assert chunks[0].metadata["row_start"] == 1
        assert chunks[0].metadata["row_end"] == 3
        assert chunks[0].metadata["table_id"] == table.table_id

    def test_large_table_row_grouping_with_summary(self):
        config = TableConfig(
            summary_enabled=True,
            parent_child_enabled=True,
            max_rows_per_chunk=5,
        )
        chunker = TableChunker(table_config=config, max_tokens_per_chunk=2000)
        table = _make_sample_table(num_rows=15, num_cols=3)
        doc_id = str(uuid.uuid4())
        chunks = chunker.chunk_table(table, document_id=doc_id, version=1)

        # 15 rows / 5 rows per chunk = 3 row chunks + 1 summary chunk = 4 chunks
        assert len(chunks) == 4

        # Chunk 0 should be table_summary
        assert chunks[0].metadata["chunk_type"] == "table_summary"
        assert chunks[0].metadata["parent_id"] == table.table_id

        # Chunks 1..3 should be row batches
        assert chunks[1].metadata["chunk_type"] == "table"
        assert chunks[1].metadata["row_start"] == 1
        assert chunks[1].metadata["row_end"] == 5

        assert chunks[2].metadata["row_start"] == 6
        assert chunks[2].metadata["row_end"] == 10

        assert chunks[3].metadata["row_start"] == 11
        assert chunks[3].metadata["row_end"] == 15

        # All chunks share the same parent_id
        for c in chunks:
            assert c.metadata["parent_id"] == table.table_id

    def test_multiple_tables_have_distinct_ids(self):
        chunker = TableChunker()
        t1 = _make_sample_table(num_rows=2, table_id="doc_p1_t1")
        t2 = _make_sample_table(num_rows=2, table_id="doc_p1_t2")
        doc_id = str(uuid.uuid4())

        c1 = chunker.chunk_table(t1, document_id=doc_id, version=1, starting_chunk_index=0)
        c2 = chunker.chunk_table(t2, document_id=doc_id, version=1, starting_chunk_index=len(c1))

        assert c1[0].metadata["table_id"] == "doc_p1_t1"
        assert c2[0].metadata["table_id"] == "doc_p1_t2"
        assert c1[0].metadata["table_id"] != c2[0].metadata["table_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. TableAwareRetriever Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTableAwareRetriever:
    @pytest.fixture
    def setup_retriever(self):
        db = DatabaseManager("sqlite:///:memory:")
        Base.metadata.create_all(db.engine)

        doc_id = str(uuid.uuid4())
        with db.session() as s:
            doc = Document(
                document_id=doc_id,
                tenant_id="default",
                source="LOCAL_FILE",
                source_uri="/docs/patent.pdf",
                filename="patent.pdf",
                mime_type="application/pdf",
                content_hash="hash123",
                version=1,
                status="CHUNKED",
            )
            s.add(doc)

            # Add 1 text chunk
            c_text = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                version=1,
                chunk_index=0,
                text="This document discusses enterprise patent strategy and innovations.",
                section="Introduction",
                page=1,
                chunk_metadata={"chunk_type": "text"},
            )
            s.add(c_text)

            # Add 2 table child chunks belonging to parent table 'patent_table_01'
            c_tab1 = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                version=1,
                chunk_index=1,
                text="Table: Patent Portfolio\nRows 1-5\nPatent US123: Active Assignee IBM",
                section="Patents",
                page=2,
                chunk_metadata={
                    "chunk_type": "table",
                    "table_id": "patent_table_01",
                    "parent_id": "patent_table_01",
                    "row_start": 1,
                    "row_end": 5,
                },
            )
            c_tab2 = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                version=1,
                chunk_index=2,
                text="Table: Patent Portfolio\nRows 6-10\nPatent US456: Pending Assignee Google",
                section="Patents",
                page=2,
                chunk_metadata={
                    "chunk_type": "table",
                    "table_id": "patent_table_01",
                    "parent_id": "patent_table_01",
                    "row_start": 6,
                    "row_end": 10,
                },
            )
            s.add_all([c_tab1, c_tab2])

        return TableAwareRetriever(db), doc_id

    def test_search_all_chunks(self, setup_retriever):
        retriever, doc_id = setup_retriever
        results = retriever.search(query="patent", top_k=10)
        assert len(results) >= 2

    def test_search_table_filter(self, setup_retriever):
        retriever, doc_id = setup_retriever
        results = retriever.search(query="patent", retrieval_filter=RetrievalFilter(chunk_type="table"))
        assert len(results) > 0
        for r in results:
            assert r.chunk_type == "table"

    def test_search_text_filter(self, setup_retriever):
        retriever, doc_id = setup_retriever
        results = retriever.search(query="patent", retrieval_filter=RetrievalFilter(chunk_type="text"))
        assert len(results) == 1
        assert results[0].chunk_type == "text"

    def test_expand_parent_table(self, setup_retriever):
        retriever, doc_id = setup_retriever
        siblings = retriever.expand_parent_table(table_id="patent_table_01", document_id=doc_id)
        assert len(siblings) == 2
        assert siblings[0].metadata["row_start"] == 1
        assert siblings[1].metadata["row_start"] == 6


# ─────────────────────────────────────────────────────────────────────────────
# 5. PDFParser Table Extraction & Collision Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF is not installed")
class TestPDFParserWithTables:
    def test_parse_real_pdf_with_table(self):
        from pathlib import Path
        from rag.parsing.pdf_parser import PDFParser
        pdf_path = Path("data/FridaySnap_New_Requirements-1.pdf")
        if not pdf_path.exists():
            pytest.skip("Test data PDF not present")

        parser = PDFParser(extract_tables=True)
        parsed = parser.parse(pdf_path.read_bytes(), filename="FridaySnap.pdf")

        assert parsed.num_pages >= 1
        assert len(parsed.elements) > 0
        # Elements include HeadingElement, TextElement, and TableElement
        elem_types = {e.element_type for e in parsed.elements}
        assert ElementType.TEXT in elem_types
        # If tables exist in doc, TableElement is present in elements and tables
        if parsed.tables:
            assert len(parsed.tables) > 0
            assert parsed.tables[0].table_id.startswith("FridaySnap")
            assert parsed.tables[0].num_cols >= 1

