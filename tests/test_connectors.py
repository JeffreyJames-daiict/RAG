"""Unit tests for Source Connectors and Ingestion Pipeline."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import create_engine, text
from rag.connectors.database import DatabaseConnector
from rag.connectors.local_file import LocalFileConnector
from rag.connectors.s3 import S3Connector
from rag.connectors.web import WebConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Document, DocumentVersion
from rag.core.models import DocumentStatus, SourceType
from rag.ingestion.pipeline import IngestionPipeline


def test_local_file_connector(tmp_path: Path):
    # Setup sample files
    doc1 = tmp_path / "sample.pdf"
    doc1.write_bytes(b"%PDF-1.4 Mock PDF Content")
    
    doc2 = tmp_path / "notes.txt"
    doc2.write_text("Meeting notes for RAG pipeline")

    ignored = tmp_path / "binary.bin"
    ignored.write_bytes(b"\x00\x01\x02")

    connector = LocalFileConnector(
        base_dir=tmp_path,
        allowed_extensions={".pdf", ".txt"},
        tenant_id="tenant_test",
    )

    assert connector.validate_connection() is True
    descriptors = list(connector.discover())
    assert len(descriptors) == 2

    pdf_desc = next(d for d in descriptors if d.filename == "sample.pdf")
    assert pdf_desc.mime_type == "application/pdf"
    assert pdf_desc.source == SourceType.LOCAL_FILE

    payload = connector.fetch(pdf_desc)
    assert payload.content == b"%PDF-1.4 Mock PDF Content"
    assert len(payload.content_hash) == 64  # SHA-256 length


def test_s3_connector_discovery_and_fetch():
    mock_s3 = MagicMock()
    # Mock head_bucket
    mock_s3.head_bucket.return_value = {}

    # Mock list_objects_v2 paginator
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [
        {
            "Contents": [
                {
                    "Key": "docs/policy.pdf",
                    "Size": 1024,
                    "LastModified": datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    "ETag": '"etag123"',
                    "StorageClass": "STANDARD",
                }
            ]
        }
    ]
    mock_s3.get_paginator.return_value = mock_paginator

    # Mock get_object
    mock_body = MagicMock()
    mock_body.read.return_value = b"Mock S3 PDF Payload"
    mock_s3.get_object.return_value = {
        "Body": mock_body,
        "ContentType": "application/pdf",
        "VersionId": "v1",
    }

    connector = S3Connector(
        bucket_name="my-rag-bucket",
        prefix="docs/",
        s3_client=mock_s3,
        tenant_id="enterprise_1",
    )

    assert connector.validate_connection() is True
    descriptors = list(connector.discover())
    assert len(descriptors) == 1
    assert descriptors[0].source_uri == "s3://my-rag-bucket/docs/policy.pdf"
    assert descriptors[0].expected_hash == "etag123"

    payload = connector.fetch(descriptors[0])
    assert payload.content == b"Mock S3 PDF Payload"
    assert len(payload.content_hash) == 64


def test_web_connector_discovery_and_fetch():
    mock_session = MagicMock()
    
    # Mock HEAD
    head_response = MagicMock()
    head_response.status_code = 200
    head_response.headers = {
        "Content-Type": "text/html; charset=utf-8",
        "ETag": '"web_etag_1"',
        "Content-Length": "500",
    }
    mock_session.head.return_value = head_response

    # Mock GET
    get_response = MagicMock()
    get_response.status_code = 200
    get_response.headers = head_response.headers
    get_response.content = b"<html><body>Enterprise Documentation</body></html>"
    get_response.raise_for_status.return_value = None
    mock_session.get.return_value = get_response

    connector = WebConnector(
        seed_urls=["https://example.com/docs/api"],
        session=mock_session,
        tenant_id="web_tenant",
    )

    assert connector.validate_connection() is True
    descriptors = list(connector.discover())
    assert len(descriptors) == 1
    assert descriptors[0].filename == "api.html"
    assert descriptors[0].mime_type == "text/html"

    payload = connector.fetch(descriptors[0])
    assert b"Enterprise Documentation" in payload.content


def test_database_connector():
    # In-memory SQLite for test source database
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE knowledge_base (id INT PRIMARY KEY, title TEXT, body TEXT, updated_at TIMESTAMP)"))
        conn.execute(
            text("INSERT INTO knowledge_base VALUES (1, 'RAG Overview', 'RAG stands for Retrieval-Augmented Generation', '2025-01-01 10:00:00')")
        )
        conn.commit()

    connector = DatabaseConnector(
        connection_string="sqlite:///:memory:",
        query="SELECT id, title, body, updated_at FROM knowledge_base",
        id_column="id",
        timestamp_column="updated_at",
        table_name="knowledge_base",
        engine=engine,
    )

    assert connector.validate_connection() is True
    descriptors = list(connector.discover())
    assert len(descriptors) == 1
    assert descriptors[0].source_uri == "db://knowledge_base/1"

    payload = connector.fetch(descriptors[0])
    parsed_json = json.loads(payload.content.decode("utf-8"))
    assert parsed_json["title"] == "RAG Overview"
    assert parsed_json["id"] == 1


def test_ingestion_pipeline_end_to_end(tmp_path: Path):
    db_manager = DatabaseManager(db_url="sqlite:///:memory:")
    db_manager.create_tables()

    # Create local files
    doc_file = tmp_path / "handbook.txt"
    doc_file.write_text("Version 1 Handbook content")

    connector = LocalFileConnector(base_dir=tmp_path, allowed_extensions={".txt"})
    pipeline = IngestionPipeline(db_manager=db_manager)
    pipeline.register_connector("local_files", connector)

    # First Ingestion Run
    summary1 = pipeline.ingest_from_connector("local_files")
    assert summary1.discovered_count == 1
    assert summary1.fetched_count == 1
    assert summary1.skipped_count == 0

    with db_manager.session() as session:
        doc = session.query(Document).first()
        assert doc is not None
        assert doc.version == 1
        assert doc.status == DocumentStatus.FETCHED.value
        assert len(doc.versions) == 1

    # Second Ingestion Run (no modifications -> skipped)
    summary2 = pipeline.ingest_from_connector("local_files")
    assert summary2.discovered_count == 1
    assert summary2.fetched_count == 0
    assert summary2.skipped_count == 1

    # Update file -> triggers version bump
    doc_file.write_text("Version 2 Handbook content (updated)")
    summary3 = pipeline.ingest_from_connector("local_files")
    assert summary3.discovered_count == 1
    assert summary3.fetched_count == 1

    with db_manager.session() as session:
        doc = session.query(Document).first()
        assert doc.version == 2
        assert len(doc.versions) == 2
