"""Unit tests for incremental document discovery, change detection, and deletion handling."""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import select

from rag.connectors.local_file import LocalFileConnector
from rag.core.database import DatabaseManager
from rag.core.db_models import Document, DocumentVersion
from rag.core.models import DocumentChangeType, DocumentStatus, SourceType
from rag.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def db_manager():
    manager = DatabaseManager(db_url="sqlite:///:memory:")
    manager.create_tables()
    return manager


def test_incremental_discovery_lifecycle(tmp_path: Path, db_manager: DatabaseManager):
    """Test full document lifecycle:

    1. Initial discovery -> NEW
    2. Repeated discovery -> UNCHANGED (Exists)
    3. File edited -> CHANGED (version bumped)
    4. File deleted -> DELETED (tombstone & soft-delete)
    5. File resurrected -> NEW / Restored
    """
    file_a = tmp_path / "doc_a.txt"
    file_b = tmp_path / "doc_b.txt"

    file_a.write_text("Document A Content Initial")
    file_b.write_text("Document B Content Initial")

    connector = LocalFileConnector(base_dir=tmp_path, allowed_extensions={".txt"})
    pipeline = IngestionPipeline(db_manager=db_manager)
    pipeline.register_connector("local", connector)

    # -------------------------------------------------------------
    # Step 1: Initial Discovery & Ingestion (What is NEW)
    # -------------------------------------------------------------
    report1 = pipeline.discover_changes("local")
    assert report1.total_discovered == 2
    assert len(report1.new_items) == 2
    assert len(report1.changed_items) == 0
    assert len(report1.unchanged_items) == 0
    assert len(report1.deleted_items) == 0

    summary1 = pipeline.ingest_from_connector("local")
    assert summary1.new_count == 2
    assert summary1.fetched_count == 2
    assert summary1.deleted_count == 0

    with db_manager.session() as session:
        docs = session.execute(select(Document)).scalars().all()
        assert len(docs) == 2
        for d in docs:
            assert d.version == 1
            assert d.status == DocumentStatus.FETCHED.value
            assert d.deleted_at is None

    # -------------------------------------------------------------
    # Step 2: Idempotent Discovery (What EXISTS / UNCHANGED)
    # -------------------------------------------------------------
    report2 = pipeline.discover_changes("local")
    assert report2.total_discovered == 2
    assert len(report2.new_items) == 0
    assert len(report2.changed_items) == 0
    assert len(report2.unchanged_items) == 2
    assert len(report2.deleted_items) == 0

    summary2 = pipeline.ingest_from_connector("local")
    assert summary2.unchanged_count == 2
    assert summary2.fetched_count == 0
    assert summary2.skipped_count == 2

    # -------------------------------------------------------------
    # Step 3: File Content Modified (What is CHANGED)
    # -------------------------------------------------------------
    # Modify doc_a.txt
    file_a.write_text("Document A Content Revision 2")

    report3 = pipeline.discover_changes("local")
    assert report3.total_discovered == 2
    assert len(report3.changed_items) == 1
    assert report3.changed_items[0].descriptor.filename == "doc_a.txt"
    assert report3.changed_items[0].change_type == DocumentChangeType.CHANGED
    assert len(report3.unchanged_items) == 1
    assert report3.unchanged_items[0].descriptor.filename == "doc_b.txt"

    summary3 = pipeline.ingest_from_connector("local")
    assert summary3.changed_count == 1
    assert summary3.fetched_count == 1
    assert summary3.skipped_count == 1

    with db_manager.session() as session:
        doc_a = session.execute(select(Document).where(Document.filename == "doc_a.txt")).scalar_one()
        assert doc_a.version == 2
        versions = session.execute(
            select(DocumentVersion).where(DocumentVersion.document_id == doc_a.document_id)
        ).scalars().all()
        assert len(versions) == 2

    # -------------------------------------------------------------
    # Step 4: File Deleted from Source (What was DELETED)
    # -------------------------------------------------------------
    file_b.unlink()  # Remove doc_b.txt from disk

    report4 = pipeline.discover_changes("local")
    assert report4.total_discovered == 1
    assert len(report4.deleted_items) == 1
    assert report4.deleted_items[0].filename == "doc_b.txt"

    summary4 = pipeline.ingest_from_connector("local", handle_deletions=True)
    assert summary4.deleted_count == 1

    with db_manager.session() as session:
        doc_b = session.execute(select(Document).where(Document.filename == "doc_b.txt")).scalar_one()
        assert doc_b.status == DocumentStatus.DELETED.value
        assert doc_b.deleted_at is not None

    # -------------------------------------------------------------
    # Step 5: File Resurrected (Re-created after deletion)
    # -------------------------------------------------------------
    file_b.write_text("Document B Resurrected Content")

    report5 = pipeline.discover_changes("local")
    assert report5.total_discovered == 2
    assert len(report5.new_items) == 1
    assert report5.new_items[0].descriptor.filename == "doc_b.txt"
    assert "resurrected" in report5.new_items[0].reason.lower()

    summary5 = pipeline.ingest_from_connector("local")
    assert summary5.new_count == 1
    assert summary5.fetched_count == 1

    with db_manager.session() as session:
        doc_b = session.execute(select(Document).where(Document.filename == "doc_b.txt")).scalar_one()
        assert doc_b.status == DocumentStatus.FETCHED.value
        assert doc_b.deleted_at is None
        assert doc_b.version == 2


def test_dry_run_mode(tmp_path: Path, db_manager: DatabaseManager):
    """Test that dry_run mode returns full discovery diff without modifying DB."""
    file_1 = tmp_path / "report.txt"
    file_1.write_text("Quarterly Report Data")

    connector = LocalFileConnector(base_dir=tmp_path, allowed_extensions={".txt"})
    pipeline = IngestionPipeline(db_manager=db_manager)
    pipeline.register_connector("local", connector)

    summary = pipeline.ingest_from_connector("local", dry_run=True)
    assert summary.new_count == 1
    assert summary.fetched_count == 0  # No actual fetch or DB write in dry-run

    with db_manager.session() as session:
        docs = session.execute(select(Document)).scalars().all()
        assert len(docs) == 0


def test_timestamp_touch_without_content_change(tmp_path: Path, db_manager: DatabaseManager):
    """Test that if a file timestamp changes but content is identical, version is not bumped."""
    import time
    file_1 = tmp_path / "contract.txt"
    file_1.write_text("Signed Contract Terms")

    connector = LocalFileConnector(base_dir=tmp_path, allowed_extensions={".txt"})
    pipeline = IngestionPipeline(db_manager=db_manager)
    pipeline.register_connector("local", connector)

    # Ingest version 1
    summary1 = pipeline.ingest_from_connector("local")
    assert summary1.new_count == 1
    assert summary1.fetched_count == 1

    # Simulate touching file timestamp with new st_mtime but identical content
    future_time = time.time() + 100
    import os
    os.utime(file_1, (future_time, future_time))

    # Ingest again
    summary2 = pipeline.ingest_from_connector("local")
    assert summary2.changed_count == 1
    assert summary2.skipped_count == 1  # Content hash matched -> skipped version bump
    assert summary2.fetched_count == 0

    with db_manager.session() as session:
        doc = session.execute(select(Document).where(Document.filename == "contract.txt")).scalar_one()
        assert doc.version == 1
        versions = session.execute(
            select(DocumentVersion).where(DocumentVersion.document_id == doc.document_id)
        ).scalars().all()
        assert len(versions) == 1


def test_pipeline_error_logging_and_reason_display(tmp_path: Path, db_manager: DatabaseManager):
    """Test that when ingestion fails:

    1. Error reason is stored in `Document.error`.
    2. Status is updated to `DocumentStatus.FAILED`.
    3. Detailed structured log is recorded in `PipelineError` (`errors` table).
    4. Upon retry / resolution, error is cleared.
    """
    from unittest.mock import patch
    from rag.core.db_models import PipelineError
    from rag.core.exceptions import FetchError

    file_err = tmp_path / "broken.txt"
    file_err.write_text("Will fail during fetch")

    connector = LocalFileConnector(base_dir=tmp_path, allowed_extensions={".txt"})
    pipeline = IngestionPipeline(db_manager=db_manager)
    pipeline.register_connector("local", connector)

    # Simulate fetch failure
    with patch.object(connector, "fetch", side_effect=FetchError("Simulated disk I/O error")):
        summary = pipeline.ingest_from_connector("local")
        assert summary.failed_count == 1
        assert len(summary.errors) == 1
        assert "Simulated disk I/O error" in summary.errors[0]

    # Verify document and error table state
    with db_manager.session() as session:
        doc = session.execute(select(Document).where(Document.filename == "broken.txt")).scalar_one()
        assert doc.status == DocumentStatus.FAILED.value
        assert doc.error is not None
        assert "Simulated disk I/O error" in doc.error

        # Check errors table
        err_records = session.execute(
            select(PipelineError).where(PipelineError.document_id == doc.document_id)
        ).scalars().all()
        assert len(err_records) == 1
        assert err_records[0].error_type == "FetchError"
        assert "Simulated disk I/O error" in err_records[0].error_message
        assert err_records[0].stage == "FETCH"

    # Fix / Retry ingestion without failure
    summary_retry = pipeline.ingest_from_connector("local")
    assert summary_retry.fetched_count == 1
    assert summary_retry.failed_count == 0

    with db_manager.session() as session:
        doc = session.execute(select(Document).where(Document.filename == "broken.txt")).scalar_one()
        assert doc.status == DocumentStatus.FETCHED.value
        assert doc.error is None


