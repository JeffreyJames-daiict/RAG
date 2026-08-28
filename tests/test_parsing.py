"""Tests for the PDF parsing and chunking layer.

Covers:
- PDFParser: in-memory parsing, page extraction, metadata, error handling
- RecursiveTextChunker: splitting, overlap, token limits, per-page tracking
- Integration: parse_and_chunk() pipeline method with a real DB
"""

from __future__ import annotations

import io
import uuid
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ── Try to import optional deps gracefully ─────────────────────────────────
try:
    import pymupdf as fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

from rag.parsing.models import ParsedDocument, ParsedPage
from rag.parsing.chunker import ChunkingConfig, RecursiveTextChunker
from rag.parsing.pdf_parser import PDFParser
from rag.core.models import ChunkSchema


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_minimal_pdf(text_per_page: List[str]) -> bytes:
    """Create a minimal valid in-memory PDF with given text per page."""
    doc = fitz.open()
    for text in text_per_page:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_parsed_doc(pages_text: List[str]) -> ParsedDocument:
    """Build a synthetic ParsedDocument without invoking the real parser."""
    pages = [
        ParsedPage(page_number=i + 1, text=t, blocks=[], headings=[])
        for i, t in enumerate(pages_text)
    ]
    return ParsedDocument(pages=pages, pdf_metadata={"num_pages": len(pages)})


# ─────────────────────────────────────────────────────────────────────────────
# PDFParser tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF not installed")
class TestPDFParser:
    """Tests for rag.parsing.pdf_parser.PDFParser."""

    def setup_method(self):
        from rag.parsing.pdf_parser import PDFParser
        self.parser = PDFParser()

    def test_parse_single_page_pdf(self):
        pdf_bytes = make_minimal_pdf(["Hello World. This is a test page."])
        result = self.parser.parse(pdf_bytes, filename="test.pdf")

        assert result.num_pages == 1
        assert "Hello" in result.pages[0].text
        assert result.parser_version == PDFParser.PARSER_VERSION

    def test_parse_multi_page_pdf(self):
        texts = ["Page one content.", "Page two content.", "Page three content."]
        pdf_bytes = make_minimal_pdf(texts)
        result = self.parser.parse(pdf_bytes, filename="multi.pdf")

        assert result.num_pages == 3
        for i, page in enumerate(result.pages):
            assert page.page_number == i + 1

    def test_pdf_metadata_captured(self):
        pdf_bytes = make_minimal_pdf(["Test metadata page."])
        result = self.parser.parse(pdf_bytes)
        assert "num_pages" in result.pdf_metadata
        assert result.pdf_metadata["num_pages"] == 1

    def test_full_text_property(self):
        texts = ["First page.", "Second page."]
        pdf_bytes = make_minimal_pdf(texts)
        result = self.parser.parse(pdf_bytes)
        full = result.full_text
        assert "First page." in full
        assert "Second page." in full

    def test_empty_bytes_raises_value_error(self):
        from rag.parsing.pdf_parser import PDFParser
        parser = PDFParser()
        with pytest.raises(ValueError, match="empty"):
            parser.parse(b"", filename="empty.pdf")

    def test_invalid_bytes_raises_runtime_error(self):
        from rag.parsing.pdf_parser import PDFParser
        parser = PDFParser()
        with pytest.raises(RuntimeError, match="PyMuPDF"):
            parser.parse(b"not a pdf at all", filename="bad.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# RecursiveTextChunker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRecursiveTextChunker:
    """Tests for rag.parsing.chunker.RecursiveTextChunker."""

    def _make_chunker(self, chunk_size=512, overlap=64) -> RecursiveTextChunker:
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
        return RecursiveTextChunker(config)

    def _doc_id(self) -> str:
        return str(uuid.uuid4())

    def test_short_text_produces_single_chunk(self):
        chunker = self._make_chunker(chunk_size=512)
        doc = make_parsed_doc(["This is a short sentence that fits in one chunk."])
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert "short sentence" in chunks[0].text

    def test_long_text_produces_multiple_chunks(self):
        # Generate ~200-word text guaranteed to exceed chunk_size=50 tokens
        long_text = " ".join(["word"] * 300)
        doc = make_parsed_doc([long_text])
        chunker = self._make_chunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self):
        long_text = "\n\n".join(["Paragraph " + str(i) + ". " + "word " * 60 for i in range(5)])
        doc = make_parsed_doc([long_text])
        chunker = self._make_chunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_carries_page_number(self):
        doc = make_parsed_doc(["Page 1 text.", "Page 2 text."])
        chunker = self._make_chunker()
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        pages = {c.page for c in chunks}
        assert 1 in pages
        assert 2 in pages

    def test_chunk_metadata_has_token_count(self):
        doc = make_parsed_doc(["Hello world, this is a test sentence."])
        chunker = self._make_chunker()
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        assert len(chunks) >= 1
        for ch in chunks:
            assert "token_count" in ch.metadata
            assert ch.metadata["token_count"] > 0

    def test_returns_chunk_schema_instances(self):
        doc = make_parsed_doc(["Some text here."])
        chunker = self._make_chunker()
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        for ch in chunks:
            assert isinstance(ch, ChunkSchema)

    def test_empty_pages_are_skipped(self):
        doc = make_parsed_doc(["", "   ", "Real content here."])
        chunker = self._make_chunker()
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        for ch in chunks:
            assert ch.text.strip() != ""

    def test_version_is_preserved_on_chunks(self):
        doc = make_parsed_doc(["Some content."])
        chunker = self._make_chunker()
        doc_id = self._doc_id()
        chunks = chunker.chunk(doc, document_id=doc_id, version=3)
        for ch in chunks:
            assert ch.version == 3
            assert str(ch.document_id) == doc_id

    def test_section_heading_carried_on_chunks(self):
        page = ParsedPage(
            page_number=1,
            text="Introduction\n\n" + "word " * 50,
            headings=["Introduction"],
        )
        doc = ParsedDocument(pages=[page])
        chunker = self._make_chunker(chunk_size=50, overlap=5)
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        assert any(ch.section == "Introduction" for ch in chunks)

    def test_chunking_version_from_config(self):
        config = ChunkingConfig(chunk_size=512, chunk_overlap=64, chunking_version="v2.0")
        chunker = RecursiveTextChunker(config)
        doc = make_parsed_doc(["Test text."])
        chunks = chunker.chunk(doc, document_id=self._doc_id(), version=1)
        for ch in chunks:
            assert ch.chunking_version == "v2.0"


# ─────────────────────────────────────────────────────────────────────────────
# Integration: parse_and_chunk() via pipeline (real PDF files)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_FITZ, reason="PyMuPDF not installed")
class TestParseAndChunkIntegration:
    """Integration tests hitting a real in-memory SQLite DB with real PDF bytes."""

    def _setup_pipeline(self):
        """Create an in-memory SQLite pipeline with a mock connector."""
        from rag.core.database import DatabaseManager
        from rag.core.db_models import Base
        from rag.ingestion.pipeline import IngestionPipeline

        db_manager = DatabaseManager(db_url="sqlite:///:memory:")
        from rag.core.migrate import ensure_raw_schema
        ensure_raw_schema(db_manager.engine)
        Base.metadata.create_all(bind=db_manager.engine)
        pipeline = IngestionPipeline(db_manager=db_manager)
        return pipeline, db_manager

    def _insert_fetched_doc(self, db_manager, doc_id: str, filename: str, source_uri: str):
        """Directly insert a FETCHED document row + version row."""
        from rag.core.db_models import Document, DocumentVersion
        from datetime import datetime, timezone

        with db_manager.session() as session:
            doc = Document(
                document_id=doc_id,
                tenant_id="default",
                source="LOCAL_FILE",
                source_uri=source_uri,
                filename=filename,
                mime_type="application/pdf",
                content_hash="abc123",
                version=1,
                status="FETCHED",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(doc)
            session.flush()
            ver = DocumentVersion(
                document_id=doc_id,
                version=1,
                content_hash="abc123",
                raw_location=source_uri,
            )
            session.add(ver)

    def test_parse_and_chunk_advances_status_to_chunked(self):
        from pathlib import Path
        from rag.core.models import SourceType, DocumentStatus
        from rag.core.db_models import Document
        from sqlalchemy import select

        pipeline, db_manager = self._setup_pipeline()

        # Use an actual PDF from the data directory if available
        data_dir = Path(__file__).resolve().parent.parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))

        if not pdf_files:
            pytest.skip("No PDF files found in data/ directory for integration test.")

        pdf_path = pdf_files[0]
        doc_id = str(uuid.uuid4())

        # Seed DB with a FETCHED doc pointing at the real file
        self._insert_fetched_doc(
            db_manager, doc_id,
            filename=pdf_path.name,
            source_uri=str(pdf_path),
        )

        # Register a real LocalFileConnector
        from rag.connectors.local_file import LocalFileConnector
        connector = LocalFileConnector(base_dir=data_dir)
        pipeline.register_connector("local", connector)

        # Run parse_and_chunk
        config = ChunkingConfig(chunk_size=256, chunk_overlap=32)
        results = pipeline.parse_and_chunk("local", chunking_config=config)

        assert doc_id in results
        assert results[doc_id] > 0, "Expected at least one chunk from a real PDF."

        # Verify DB status
        with db_manager.session() as session:
            doc = session.get(Document, doc_id)
            assert doc.status == DocumentStatus.CHUNKED.value
            assert doc.error is None

    def test_parse_and_chunk_idempotent(self):
        """Running parse_and_chunk twice with force_reparse should not duplicate chunks."""
        from pathlib import Path
        from rag.core.db_models import Chunk
        from sqlalchemy import func, select

        pipeline, db_manager = self._setup_pipeline()
        data_dir = Path(__file__).resolve().parent.parent / "data"
        pdf_files = list(data_dir.glob("*.pdf"))

        if not pdf_files:
            pytest.skip("No PDF files found in data/ directory.")

        pdf_path = pdf_files[0]
        doc_id = str(uuid.uuid4())
        self._insert_fetched_doc(db_manager, doc_id, pdf_path.name, str(pdf_path))

        from rag.connectors.local_file import LocalFileConnector
        pipeline.register_connector("local", LocalFileConnector(base_dir=data_dir))

        config = ChunkingConfig(chunk_size=256, chunk_overlap=32)

        r1 = pipeline.parse_and_chunk("local", chunking_config=config)
        count_after_first = r1[doc_id]

        r2 = pipeline.parse_and_chunk("local", chunking_config=config, force_reparse=True)
        count_after_second = r2[doc_id]

        assert count_after_first == count_after_second, (
            f"Chunk count changed between runs: {count_after_first} -> {count_after_second}"
        )

        with db_manager.session() as session:
            total = session.execute(
                select(func.count()).select_from(Chunk)
                .where(Chunk.document_id == doc_id)
            ).scalar_one()
            assert total == count_after_first
