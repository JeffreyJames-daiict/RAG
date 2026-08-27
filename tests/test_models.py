"""Unit tests for RAG data models (SQLAlchemy ORM and Pydantic schemas)."""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import select
from rag.core.database import DatabaseManager
from rag.core.db_models import Chunk, Document, DocumentVersion, Embedding
from rag.core.models import (
        ChunkSchema,
        DocumentDescriptor,
        DocumentSchema,
        DocumentStatus,
        DocumentVersionSchema,
        EmbeddingSchema,
        SourceType,
    )



def test_pydantic_document_schema_validation():
    doc_id = uuid.uuid4()
    doc = DocumentSchema(
        document_id=doc_id,
        tenant_id="tenant_alpha",
        source=SourceType.LOCAL_FILE,
        source_uri="/data/reports/annual_2025.pdf",
        filename="annual_2025.pdf",
        mime_type="application/pdf",
        content_hash="abc123hash",
        version=1,
        status=DocumentStatus.DISCOVERED,
    )
    assert doc.tenant_id == "tenant_alpha"
    assert doc.source == SourceType.LOCAL_FILE
    assert doc.status == DocumentStatus.DISCOVERED
    assert doc.version == 1


def test_pydantic_chunk_and_embedding_schemas():
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    chunk = ChunkSchema(
        chunk_id=chunk_id,
        document_id=doc_id,
        version=1,
        chunk_index=0,
        text="This is an executive summary section.",
        section="Executive Summary",
        page=1,
        metadata={"tokens": 12, "author": "Alice"},
        chunking_version="semantic-v1",
    )
    assert chunk.chunk_id == chunk_id
    assert chunk.metadata["tokens"] == 12

    embedding = EmbeddingSchema(
        chunk_id=chunk_id,
        embedding_model="text-embedding-3-small",
        embedding_version="v1.0",
        vector=[0.12, -0.45, 0.99, 0.0],
    )
    assert len(embedding.vector) == 4


def test_sqlalchemy_orm_lifecycle():
    db = DatabaseManager(db_url="sqlite:///:memory:")
    db.create_tables()

    with db.session() as session:
        doc_id = str(uuid.uuid4())
        doc = Document(
            document_id=doc_id,
            tenant_id="org_1",
            source="LOCAL_FILE",
            source_uri="/tmp/doc1.pdf",
            filename="doc1.pdf",
            mime_type="application/pdf",
            content_hash="hash_v1",
            version=1,
            status=DocumentStatus.FETCHED.value,
        )
        session.add(doc)
        session.flush()

        v1 = DocumentVersion(
            document_id=doc_id,
            version=1,
            content_hash="hash_v1",
            raw_location="/storage/v1/doc1.pdf",
            parser_version="v1.0",
        )
        session.add(v1)

        chunk1 = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            version=1,
            chunk_index=0,
            text="First sentence.",
            section="Intro",
            page=1,
            chunk_metadata={"token_count": 2},
            chunking_version="v1.0",
        )
        session.add(chunk1)
        session.flush()

        emb1 = Embedding(
            chunk_id=chunk1.chunk_id,
            embedding_model="text-embedding-3-small",
            embedding_version="v1.0",
            vector=[0.1, 0.2, 0.3],
        )
        session.add(emb1)

    # Verify query and relationships
    with db.session() as session:
        queried_doc = session.execute(
            select(Document).where(Document.document_id == doc_id)
        ).scalar_one()

        assert queried_doc.filename == "doc1.pdf"
        assert len(queried_doc.versions) == 1
        assert queried_doc.versions[0].content_hash == "hash_v1"
        assert len(queried_doc.chunks) == 1
        assert queried_doc.chunks[0].text == "First sentence."
        assert len(queried_doc.chunks[0].embeddings) == 1
        assert queried_doc.chunks[0].embeddings[0].vector == [0.1, 0.2, 0.3]
