"""Comprehensive test suite for the OCR subsystem and PDF parser integration."""

from __future__ import annotations

import io
import uuid
import pytest
from PIL import Image, ImageDraw
import pymupdf as fitz

from rag.core.config import OCRConfig, TableConfig
from rag.ocr.analyzer import PageQualityAnalyzer
from rag.ocr.base import BaseOCRProvider, OCRProviderRegistry
from rag.ocr.evaluator import OCRQualityEvaluator
from rag.ocr.merger import MixedContentMerger
from rag.ocr.models import (
    ContentSource,
    OCRBlock,
    OCRPageResult,
    OCRStatus,
    PageClassificationResult,
)
from rag.ocr.preprocessor import ImagePreprocessor
from rag.ocr.providers.mock_ocr import MockOCRProvider
from rag.parsing.chunker import RecursiveTextChunker
from rag.parsing.models import ParsedBlock
from rag.parsing.pdf_parser import PDFParser
from rag.parsing.table_chunker import TableChunker


# ─────────────────────────────────────────────────────────────────────────────
# Helper to create in-memory synthetic PDFs for testing
# ─────────────────────────────────────────────────────────────────────────────

def _create_digital_pdf(text: str = "This is a clean, native digital PDF document for testing.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _create_scanned_pdf(text_on_image: str = "Scanned invoice text from image.") -> bytes:
    """Create a PDF with no text layer, containing only a rasterized bitmap image."""
    img = Image.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), text_on_image, fill=(0, 0, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_image(page.rect, stream=img_bytes.getvalue())
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _create_mixed_pdf() -> bytes:
    """Create a 2-page PDF: Page 1 is Digital Native, Page 2 is Scanned Image-only."""
    doc = fitz.open()
    # Page 1: Native Text
    p1 = doc.new_page(width=600, height=800)
    p1.insert_text((72, 72), "Page 1: Digital Executive Summary with Native Text.", fontsize=12)

    # Page 2: Image Only
    img = Image.new("RGB", (600, 800), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Page 2: Scanned Diagram and Blueprint Description", fill=(0, 0, 0))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    p2 = doc.new_page(width=600, height=800)
    p2.insert_image(p2.rect, stream=img_bytes.getvalue())

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# 1. PageQualityAnalyzer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPageQualityAnalyzer:
    def test_clean_native_text_scores_high(self):
        analyzer = PageQualityAnalyzer(min_native_text_quality=0.70)
        text = (
            "Enterprise Retrieval-Augmented Generation (RAG) Architecture "
            "Overview and Implementation Specification for Production."
        )
        score, metrics = analyzer.calculate_text_quality(text)
        assert score >= 0.85
        assert metrics["printable_ratio"] == 1.0
        assert metrics["meaningful_word_ratio"] >= 0.80

    def test_garbled_text_scores_low(self):
        analyzer = PageQualityAnalyzer(min_native_text_quality=0.70)
        garbled = "\ufffd\ufffd\x00\x01\x02 x#@!$ 9*&^% \ufffd\ufffd"
        score, metrics = analyzer.calculate_text_quality(garbled)
        assert score < 0.50

    def test_scanned_page_classification(self):
        pdf_bytes = _create_scanned_pdf("Rasterized scanned contract")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]

        analyzer = PageQualityAnalyzer(min_native_text_quality=0.70)
        classification = analyzer.analyze_page(page, page_number=1)

        assert classification.source_type == ContentSource.SCANNED
        assert classification.ocr_required is True
        assert classification.image_count >= 1
        doc.close()

    def test_digital_page_classification(self):
        pdf_bytes = _create_digital_pdf("Clean native digital document with plenty of words and paragraphs.")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]

        analyzer = PageQualityAnalyzer(min_native_text_quality=0.70)
        classification = analyzer.analyze_page(page, page_number=1)

        assert classification.source_type == ContentSource.NATIVE
        assert classification.ocr_required is False
        doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ImagePreprocessor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImagePreprocessor:
    def test_preprocessing_profiles(self):
        img = Image.new("RGB", (200, 200), color=(180, 180, 180))
        for profile in ("minimal", "standard", "aggressive"):
            processed = ImagePreprocessor.preprocess(img, profile=profile)
            assert isinstance(processed, Image.Image)
            assert processed.size == (200, 200)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OCRQualityEvaluator & Retry Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRQualityEvaluator:
    def test_evaluate_high_quality_result(self):
        evaluator = OCRQualityEvaluator(min_confidence=0.60)
        res = OCRPageResult(
            page_number=1,
            text="High quality invoice description with clear recognized text.",
            confidence=0.92,
        )
        evaluated = evaluator.evaluate_result(res)
        assert evaluated.status == OCRStatus.PASS
        assert evaluated.quality_score >= 0.75

    def test_evaluate_empty_result_fails(self):
        evaluator = OCRQualityEvaluator(min_confidence=0.60)
        res = OCRPageResult(page_number=1, text="", confidence=0.0)
        evaluated = evaluator.evaluate_result(res)
        assert evaluated.status == OCRStatus.FAILED

    def test_process_with_retry_progression(self):
        evaluator = OCRQualityEvaluator(min_confidence=0.70, max_retries=2)
        mock_provider = MockOCRProvider(name="mock", default_text="Recognized text on retry")

        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        result = evaluator.process_with_retry(
            provider=mock_provider,
            raw_image=img,
            page_number=1,
        )
        assert result.status == OCRStatus.PASS
        assert result.text == "Recognized text on retry"


# ─────────────────────────────────────────────────────────────────────────────
# 4. MixedContentMerger Tests (De-duplication)
# ─────────────────────────────────────────────────────────────────────────────

class TestMixedContentMerger:
    def test_suppresses_duplicate_overlapping_ocr_block(self):
        merger = MixedContentMerger(iou_overlap_threshold=0.30)
        native_blocks = [
            ParsedBlock(text="Quarterly Financial Report Q3", bbox=(50.0, 50.0, 300.0, 100.0), block_no=1)
        ]
        # Overlapping OCR block with identical text
        ocr_blocks = [
            OCRBlock(text="Quarterly Financial Report Q3", bbox=(50.0, 50.0, 300.0, 100.0), confidence=0.90),
            # Non-overlapping image diagram block
            OCRBlock(text="Chart 1: Operating Margin", bbox=(50.0, 400.0, 300.0, 500.0), confidence=0.92),
        ]

        merged = merger.merge_blocks(native_blocks, ocr_blocks, page_number=1)
        # Should have 2 blocks: 1 native + 1 non-overlapping OCR block (duplicate suppressed)
        assert len(merged) == 2
        assert merged[0].text == "Quarterly Financial Report Q3"
        assert merged[1].text == "Chart 1: Operating Margin"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full PDFParser & Downstream Chunking Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFParserWithOCR:
    def test_digital_pdf_bypasses_ocr(self):
        pdf_bytes = _create_digital_pdf("Digital native document content.")
        mock_provider = MockOCRProvider(default_text="SHOULD_NOT_BE_USED")

        parser = PDFParser(
            ocr_config=OCRConfig(enabled=True, min_native_text_quality=0.70),
            ocr_provider=mock_provider,
        )
        parsed = parser.parse(pdf_bytes, filename="digital.pdf")

        assert parsed.num_pages == 1
        p1 = parsed.pages[0]
        assert p1.content_source == "native"
        assert p1.ocr_applied is False
        assert "SHOULD_NOT_BE_USED" not in p1.text
        assert "Digital native document" in p1.text

    def test_scanned_pdf_triggers_ocr(self):
        pdf_bytes = _create_scanned_pdf("Scanned text on image")
        mock_provider = MockOCRProvider(default_text="Mock OCR: Scanned text recognized successfully.")

        parser = PDFParser(
            ocr_config=OCRConfig(enabled=True, min_native_text_quality=0.70),
            ocr_provider=mock_provider,
        )
        parsed = parser.parse(pdf_bytes, filename="scanned.pdf")

        assert parsed.num_pages == 1
        p1 = parsed.pages[0]
        assert p1.content_source == "scanned"
        assert p1.ocr_applied is True
        assert p1.ocr_provider == "mock"
        assert "Scanned text recognized successfully" in p1.text

        # Test downstream text chunking works on OCR output
        chunker = RecursiveTextChunker()
        chunks = chunker.chunk(parsed, document_id=str(uuid.uuid4()), version=1)
        assert len(chunks) >= 1
        assert "Scanned text recognized" in chunks[0].text

    def test_mixed_pdf_processes_each_page_independently(self):
        pdf_bytes = _create_mixed_pdf()
        mock_provider = MockOCRProvider(default_text="Mock OCR Page 2 recognized text")

        parser = PDFParser(
            ocr_config=OCRConfig(enabled=True, min_native_text_quality=0.70),
            ocr_provider=mock_provider,
        )
        parsed = parser.parse(pdf_bytes, filename="mixed.pdf")

        assert parsed.num_pages == 2
        # Page 1: Native
        assert parsed.pages[0].content_source == "native"
        assert parsed.pages[0].ocr_applied is False
        assert "Digital Executive Summary" in parsed.pages[0].text

        # Page 2: Scanned -> OCR
        assert parsed.pages[1].content_source == "scanned"
        assert parsed.pages[1].ocr_applied is True
        assert "Mock OCR Page 2 recognized text" in parsed.pages[1].text

        # Diagnostics check
        assert parsed.ocr_diagnostics["pages_native"] == 1
        assert parsed.ocr_diagnostics["pages_scanned"] == 1
        assert parsed.ocr_diagnostics["ocr_pages_processed"] == 1

    def test_database_persistence_of_scores_and_metrics(self, tmp_path):
        """Verify that quality scores, OCR metrics, and table stats are written to PostgreSQL/SQLite tables."""
        from sqlalchemy import select
        from rag.connectors.local_file import LocalFileConnector
        from rag.core.database import DatabaseManager
        from rag.core.db_models import Base, Chunk, Document, DocumentVersion
        from rag.core.migrate import ensure_raw_schema
        from rag.ingestion.pipeline import IngestionPipeline
        from rag.storage.local_storage import LocalRawStorageBackend

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pdf_bytes = _create_mixed_pdf()
        (data_dir / "mixed_test.pdf").write_bytes(pdf_bytes)

        db = DatabaseManager("sqlite:///:memory:")
        ensure_raw_schema(db.engine)
        Base.metadata.create_all(bind=db.engine)

        raw_store = LocalRawStorageBackend(base_dir=tmp_path / "raw_store")
        pipeline = IngestionPipeline(db_manager=db, raw_storage=raw_store)
        pipeline.register_connector("local", LocalFileConnector(base_dir=data_dir))

        # Ingest file
        summary = pipeline.ingest_from_connector("local")
        assert summary.new_count == 1

        # Parse & chunk
        pipeline.parse_and_chunk("local")

        with db.session() as s:
            doc = s.execute(select(Document)).scalars().first()
            assert doc is not None

            ver = s.execute(
                select(DocumentVersion).where(DocumentVersion.document_id == doc.document_id)
            ).scalars().first()
            assert ver is not None
            meta = ver.metadata_snapshot
            assert meta is not None
            assert "avg_quality_score" in meta
            assert meta["avg_quality_score"] > 0.0
            assert "ocr_diagnostics" in meta
            assert "page_quality_scores" in meta
            assert len(meta["page_quality_scores"]) == 2

            # Check normalized relational DocumentVersionMetadata table
            from rag.core.db_models import DocumentVersionMetadata, ChunkMetadata
            ver_meta_rec = s.execute(
                select(DocumentVersionMetadata).where(DocumentVersionMetadata.document_id == doc.document_id)
            ).scalars().first()
            assert ver_meta_rec is not None
            assert ver_meta_rec.primary_language == "eng"
            assert ver_meta_rec.num_pages == 2
            assert ver_meta_rec.avg_quality_score is not None

            # Check chunk metadata and normalized ChunkMetadata table
            chunks = s.execute(
                select(Chunk).where(Chunk.document_id == doc.document_id)
            ).scalars().all()
            assert len(chunks) >= 1
            for c in chunks:
                assert "content_source" in c.chunk_metadata
                assert "chunk_type" in c.chunk_metadata

            chunk_meta_recs = s.execute(
                select(ChunkMetadata).where(ChunkMetadata.document_id == doc.document_id)
            ).scalars().all()
            assert len(chunk_meta_recs) == len(chunks)
            for cm in chunk_meta_recs:
                assert cm.chunk_type in ("text", "table", "table_summary")
                assert cm.language == "eng"
                assert cm.token_count > 0

