"""Unit tests for Chunk Distribution Analyzer and Section/Subsection Hierarchical Tracking."""

import pytest

from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, ChunkMetadata, Document
from rag.core.models import DocumentStatus
from rag.parsing.chunker import ChunkingConfig, RecursiveTextChunker
from rag.parsing.models import ParsedDocument, ParsedPage
from rag.retrieval.distribution import ChunkDistributionAnalyzer


def test_chunk_distribution_from_records():
    """Verify statistics computation from raw chunk record dicts."""
    records = [
        {
            "chunk_type": "text",
            "token_count": 150,
            "page": 1,
            "section": "Introduction",
            "content_source": "native",
            "ocr_applied": False,
            "quality_score": 0.95,
            "language": "eng",
        },
        {
            "chunk_type": "text",
            "token_count": 350,
            "page": 1,
            "section": "Introduction > Overview",
            "content_source": "native",
            "ocr_applied": False,
            "quality_score": 0.90,
            "language": "eng",
        },
        {
            "chunk_type": "table",
            "token_count": 80,
            "page": 2,
            "section": "Financials",
            "content_source": "ocr",
            "ocr_applied": True,
            "ocr_confidence": 0.88,
            "quality_score": 0.85,
            "language": "eng",
        },
    ]

    report = ChunkDistributionAnalyzer.analyze_records(records, tenant_id="test_tenant")

    assert report.total_chunks == 3
    assert report.chunk_types == {"text": 2, "table": 1}
    assert report.token_stats.min == 80
    assert report.token_stats.max == 350
    assert report.token_stats.mean == round((150 + 350 + 80) / 3, 2)
    assert report.page_coverage.total_pages_covered == 2
    assert report.section_distribution.chunks_with_section == 3
    assert report.modality_distribution.ocr_applied_count == 1
    assert abs(report.modality_distribution.ocr_ratio - 0.3333) < 1e-3


def test_chunk_distribution_database():
    """Verify DB querying and distribution reporting in ChunkDistributionAnalyzer."""
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    with db.session() as session:
        doc = Document(
            document_id="doc_dist",
            tenant_id="tenant_dist",
            source="local",
            filename="data.pdf",
            source_uri="s3://data.pdf",
            mime_type="application/pdf",
            content_hash="hash_dist",
            status=DocumentStatus.INDEXED.value,
            version=1,
        )
        session.add(doc)

        c1 = Chunk(
            chunk_id="c1",
            document_id="doc_dist",
            version=1,
            chunk_index=0,
            text="First chunk",
            page=1,
            section="Header 1",
        )
        cm1 = ChunkMetadata(
            chunk_id="c1",
            document_id="doc_dist",
            version=1,
            chunk_type="text",
            token_count=100,
            language="eng",
            content_source="native",
            ocr_applied=False,
        )

        c2 = Chunk(
            chunk_id="c2",
            document_id="doc_dist",
            version=1,
            chunk_index=1,
            text="Second chunk",
            page=2,
            section="Header 2",
        )
        cm2 = ChunkMetadata(
            chunk_id="c2",
            document_id="doc_dist",
            version=1,
            chunk_type="table",
            token_count=200,
            language="eng",
            content_source="ocr",
            ocr_applied=True,
            ocr_confidence=0.92,
        )

        session.add_all([c1, cm1, c2, cm2])

    with db.session() as session:
        report = ChunkDistributionAnalyzer.analyze_db(session, tenant_id="tenant_dist")

    assert report.total_chunks == 2
    assert report.chunk_types.get("text") == 1
    assert report.chunk_types.get("table") == 1
    assert report.token_stats.count == 2
    assert report.page_coverage.total_pages_covered == 2


def test_hierarchical_section_and_subsection_breadcrumbs():
    """Verify RecursiveTextChunker produces hierarchical section, subsection, and breadcrumbs."""
    p1 = ParsedPage(
        page_number=1,
        text="Chapter 1 Introduction\n\nSome introductory content that begins the section.",
        headings=["1. Introduction", "1.1 Background"],
    )
    p2 = ParsedPage(
        page_number=2,
        text="Continuing the background discussion on page two without new headings.",
        headings=[],
    )
    p3 = ParsedPage(
        page_number=3,
        text="Methodology section with details.",
        headings=["2. Methodology"],
    )
    doc = ParsedDocument(pages=[p1, p2, p3])

    chunker = RecursiveTextChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10))
    chunks = chunker.chunk(doc, document_id="00000000-0000-0000-0000-000000000001", version=1)

    assert len(chunks) >= 3

    # First page chunks should have section "1. Introduction" and subsection "1.1 Background"
    p1_chunk = next(c for c in chunks if c.page == 1)
    assert p1_chunk.metadata["section"] == "1. Introduction"
    assert p1_chunk.metadata["subsection"] == "1.1 Background"
    assert p1_chunk.metadata["section_path"] == ["1. Introduction", "1.1 Background"]
    assert p1_chunk.section == "1. Introduction > 1.1 Background"

    # Second page chunks should inherit the active section hierarchy from page 1
    p2_chunk = next(c for c in chunks if c.page == 2)
    assert p2_chunk.metadata["section"] == "1. Introduction"
    assert p2_chunk.metadata["subsection"] == "1.1 Background"
    assert p2_chunk.section == "1. Introduction > 1.1 Background"

    # Third page chunk resets to "2. Methodology"
    p3_chunk = next(c for c in chunks if c.page == 3)
    assert p3_chunk.metadata["section"] == "2. Methodology"
    assert p3_chunk.metadata["subsection"] is None
    assert p3_chunk.section == "2. Methodology"
