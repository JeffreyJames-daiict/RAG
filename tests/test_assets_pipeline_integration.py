"""End-to-end integration tests for visual figure processing in IngestionPipeline."""

import io
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pytest
from sqlalchemy import select

from rag.connectors.local_file import LocalFileConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, ChunkMetadata, Document, DocumentAsset, Embedding
from rag.core.models import DocumentStatus
from rag.ingestion.pipeline import IngestionPipeline
from rag.storage.local_storage import LocalRawStorageBackend


def _create_rich_pdf_with_figure(output_path: Path) -> None:
    # Create an image for the diagram
    img = Image.new("RGB", (300, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 280, 180], fill="lightcyan", outline="navy", width=2)
    draw.text((40, 50), "API Gateway", fill="black")
    draw.text((40, 90), "Retriever Service", fill="black")
    draw.text((40, 130), "Vector Database", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    doc = fitz.open()
    page = doc.new_page(width=600, height=800)

    # Text
    page.insert_text((50, 50), "Enterprise AI Pipeline Documentation", fontsize=16)
    page.insert_text((50, 80), "Below is the architectural overview of the system.", fontsize=11)

    # Insert image
    img_rect = fitz.Rect(50, 100, 350, 300)
    page.insert_image(img_rect, stream=img_bytes)

    # Caption immediately below figure
    page.insert_text((50, 315), "Figure 1: RAG System Architecture and Ingestion Flow", fontsize=11)

    # Paragraph text
    page.insert_text((50, 350), "The system processes visual assets alongside textual chunks.", fontsize=11)

    doc.save(str(output_path))
    doc.close()


@pytest.fixture
def temp_workspace(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    raw_dir = tmp_path / "raw_storage"
    raw_dir.mkdir()
    return docs_dir, raw_dir


def test_pipeline_end_to_end_figure_processing(temp_workspace):
    docs_dir, raw_dir = temp_workspace
    pdf_path = docs_dir / "architecture_report.pdf"
    _create_rich_pdf_with_figure(pdf_path)

    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    raw_backend = LocalRawStorageBackend(base_dir=raw_dir)
    from rag.assets.processor import AssetProcessor
    from rag.core.config import AssetConfig, VisionConfig
    asset_proc = AssetProcessor(
        config=AssetConfig(vision=VisionConfig(provider="mock")),
        storage_backend=raw_backend,
    )
    pipeline = IngestionPipeline(db_manager=db, raw_storage=raw_backend, asset_processor=asset_proc)

    connector = LocalFileConnector(base_dir=docs_dir, allowed_extensions={".pdf"})
    pipeline.register_connector("local", connector)

    # 1. Ingest
    summary = pipeline.ingest_from_connector("local")
    assert summary.fetched_count == 1

    # 2. Parse & Chunk with auto-embed
    chunk_counts = pipeline.parse_and_chunk("local", auto_embed=True)
    assert sum(chunk_counts.values()) >= 1

    with db.session() as session:
        # Check Document state
        doc = session.execute(select(Document)).scalar_one()
        assert doc.status == DocumentStatus.INDEXED.value

        # Check DocumentAsset records
        assets = session.execute(select(DocumentAsset)).scalars().all()
        assert len(assets) == 1
        asset = assets[0]
        assert asset.document_id == doc.document_id
        assert asset.page_number == 1
        assert "Figure 1" in (asset.caption or "")
        assert asset.asset_type in ("ARCHITECTURE_DIAGRAM", "FIGURE")
        assert len(asset.content_hash) == 64
        assert asset.storage_uri.endswith(".png")
        assert len(asset.entities) > 0

        # Check Chunks table includes the figure chunk
        chunks = session.execute(select(Chunk)).scalars().all()
        fig_chunks = [c for c in chunks if c.chunk_metadata.get("chunk_type") == "figure"]
        assert len(fig_chunks) == 1
        f_chunk = fig_chunks[0]
        assert "Figure:" in f_chunk.text
        assert "Components / Entities:" in f_chunk.text

        # Check ChunkMetadata table has asset_id linked
        meta_stmt = select(ChunkMetadata).where(ChunkMetadata.chunk_id == f_chunk.chunk_id)
        meta_row = session.execute(meta_stmt).scalar_one()
        assert meta_row.chunk_type == "figure"
        assert meta_row.asset_id == asset.asset_id

        # Check Embeddings exist for all chunks (including the figure chunk)
        embs = session.execute(select(Embedding)).scalars().all()
        assert len(embs) == len(chunks)
        fig_emb = [e for e in embs if e.chunk_id == f_chunk.chunk_id]
        assert len(fig_emb) == 1
        assert len(fig_emb[0].vector) == 1536
