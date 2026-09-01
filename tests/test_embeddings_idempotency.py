"""Unit tests for embedding idempotency and multi-model versioning."""

import uuid
import pytest
from sqlalchemy import select
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, Embedding
from rag.embeddings.providers.mock_provider import MockEmbeddingProvider
from rag.embeddings.service import EmbeddingService


@pytest.fixture
def db_with_chunks():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    doc_id = "doc_idemp_1"
    with db.session() as session:
        doc = Document(
            document_id=doc_id,
            tenant_id="default",
            source="LOCAL_FILE",
            source_uri="/docs/sample.pdf",
            filename="sample.pdf",
            mime_type="application/pdf",
            content_hash="hash_1",
            version=1,
        )
        session.add(doc)

        for i in range(3):
            ch = Chunk(
                chunk_id=f"chk_{i}",
                document_id=doc_id,
                version=1,
                chunk_index=i,
                text=f"Paragraph content number {i}",
            )
            session.add(ch)

    return db, doc_id


def test_embedding_idempotency_skips_existing(db_with_chunks):
    db, doc_id = db_with_chunks
    service = EmbeddingService(db_manager=db)

    # First embedding run
    sum1 = service.embed_document(document_id=doc_id)
    assert sum1.total_chunks == 3
    assert sum1.embedded_count == 3
    assert sum1.skipped_count == 0

    with db.session() as session:
        embs = session.execute(select(Embedding)).scalars().all()
        assert len(embs) == 3

    # Second embedding run (should skip all 3)
    sum2 = service.embed_document(document_id=doc_id)
    assert sum2.total_chunks == 3
    assert sum2.embedded_count == 0
    assert sum2.skipped_count == 3

    # Force re-embed run (should re-embed all 3)
    sum3 = service.embed_document(document_id=doc_id, force_reembed=True)
    assert sum3.embedded_count == 3
    assert sum3.skipped_count == 0


def test_embedding_multi_model_versioning(db_with_chunks):
    db, doc_id = db_with_chunks

    # Model A v1.0
    prov_a = MockEmbeddingProvider(model_name="model-alpha", model_version="v1.0", dimension=64)
    service_a = EmbeddingService(db_manager=db, provider=prov_a)
    sum_a = service_a.embed_document(document_id=doc_id)
    assert sum_a.embedded_count == 3

    # Model B v1.0 on the same chunks
    prov_b = MockEmbeddingProvider(model_name="model-beta", model_version="v1.0", dimension=128)
    service_b = EmbeddingService(db_manager=db, provider=prov_b)
    sum_b = service_b.embed_document(document_id=doc_id)
    assert sum_b.embedded_count == 3

    # Model A v2.0 on the same chunks
    prov_a_v2 = MockEmbeddingProvider(model_name="model-alpha", model_version="v2.0", dimension=64)
    service_a_v2 = EmbeddingService(db_manager=db, provider=prov_a_v2)
    sum_a_v2 = service_a_v2.embed_document(document_id=doc_id)
    assert sum_a_v2.embedded_count == 3

    # Total embeddings in DB should be 3 chunks * 3 model configs = 9 embeddings
    with db.session() as session:
        embs = session.execute(select(Embedding)).scalars().all()
        assert len(embs) == 9
