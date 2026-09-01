"""End-to-end integration tests for embedding generation in the IngestionPipeline."""

import io
from pathlib import Path
import fitz  # PyMuPDF
import pytest
from sqlalchemy import select
from rag.connectors.local_file import LocalFileConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.core.models import DocumentStatus
from rag.ingestion.pipeline import IngestionPipeline


def _create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def temp_workspace(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    return docs_dir


def test_pipeline_generate_embeddings_stage(temp_workspace: Path):
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    pdf_file = temp_workspace / "report.pdf"
    pdf_file.write_bytes(_create_sample_pdf("Enterprise AI and RAG architecture for scalable search."))

    connector = LocalFileConnector(base_dir=temp_workspace, allowed_extensions={".pdf"})
    pipeline = IngestionPipeline(db_manager=db)
    pipeline.register_connector("local", connector)

    # 1. Ingest
    summary = pipeline.ingest_from_connector("local")
    assert summary.fetched_count == 1

    with db.session() as session:
        doc = session.execute(select(Document)).scalar_one()
        assert doc.status == DocumentStatus.FETCHED.value

    # 2. Parse & Chunk
    chunk_counts = pipeline.parse_and_chunk("local")
    assert sum(chunk_counts.values()) >= 1

    with db.session() as session:
        doc = session.execute(select(Document)).scalar_one()
        assert doc.status == DocumentStatus.CHUNKED.value

    # 3. Generate Embeddings
    embed_results = pipeline.generate_embeddings("local")
    assert len(embed_results) == 1
    doc_id = list(embed_results.keys())[0]
    embed_summary = embed_results[doc_id]
    assert embed_summary.embedded_count >= 1
    assert embed_summary.failed_count == 0

    # Verify status is now INDEXED and embeddings are in DB
    with db.session() as session:
        doc = session.execute(select(Document)).scalar_one()
        assert doc.status == DocumentStatus.INDEXED.value

        embs = session.execute(select(Embedding)).scalars().all()
        assert len(embs) == embed_summary.embedded_count
        assert len(embs[0].vector) == 1536
        assert embs[0].provider == "mock"


def test_pipeline_parse_and_chunk_auto_embed(temp_workspace: Path):
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    pdf_file = temp_workspace / "auto_doc.pdf"
    pdf_file.write_bytes(_create_sample_pdf("Auto embed pipeline step verification."))

    connector = LocalFileConnector(base_dir=temp_workspace, allowed_extensions={".pdf"})
    pipeline = IngestionPipeline(db_manager=db)
    pipeline.register_connector("local", connector)

    pipeline.ingest_from_connector("local")
    pipeline.parse_and_chunk("local", auto_embed=True)

    with db.session() as session:
        doc = session.execute(select(Document)).scalar_one()
        assert doc.status == DocumentStatus.INDEXED.value

        embs = session.execute(select(Embedding)).scalars().all()
        assert len(embs) >= 1


def test_pipeline_incremental_ingestion_with_embeddings(temp_workspace: Path):
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    pdf_file = temp_workspace / "versioned_doc.pdf"
    pdf_file.write_bytes(_create_sample_pdf("Version 1 content."))

    connector = LocalFileConnector(base_dir=temp_workspace, allowed_extensions={".pdf"})
    pipeline = IngestionPipeline(db_manager=db)
    pipeline.register_connector("local", connector)

    # Initial ingestion, parse, chunk, embed
    pipeline.ingest_from_connector("local")
    pipeline.parse_and_chunk("local", auto_embed=True)

    with db.session() as session:
        doc = session.execute(select(Document)).scalar_one()
        assert doc.version == 1
        assert doc.status == DocumentStatus.INDEXED.value
        embs_v1 = session.execute(select(Embedding)).scalars().all()
        assert len(embs_v1) >= 1
        v1_count = len(embs_v1)

    # Re-run on unchanged file -> zero new embeddings
    pipeline.ingest_from_connector("local")
    embed_results_unchanged = pipeline.generate_embeddings("local")
    assert len(embed_results_unchanged) == 0  # No CHUNKED docs to process

    # Update file content -> triggers new version (v2)
    pdf_file.write_bytes(_create_sample_pdf("Version 2 updated content with new sections."))
    pipeline.ingest_from_connector("local")
    pipeline.parse_and_chunk("local", auto_embed=True)

    with db.session() as session:
        doc = session.execute(select(Document)).scalar_one()
        assert doc.version == 2
        assert doc.status == DocumentStatus.INDEXED.value
        embs_total = session.execute(select(Embedding)).scalars().all()
        # Both v1 and v2 chunks have embeddings preserved
        assert len(embs_total) > v1_count
