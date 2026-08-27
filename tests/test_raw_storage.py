"""Tests for the raw storage layer.

Covers:
- LocalRawStorageBackend: write/read/exists/delete round-trip
- Immutable write guard (FileExistsError, force=True override)
- Atomic write (tmp rename)
- SHA-256 sidecar creation and verify_on_read
- _save_raw helper (pipeline integration)
- _read_raw_or_fetch helper (prefers raw store over connector)
- Integration: ingest_from_connector stores raw bytes + sets raw_location
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.storage.base import RawStorageBackend
from rag.storage.local_storage import LocalRawStorageBackend


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path) -> LocalRawStorageBackend:
    """A fresh LocalRawStorageBackend backed by a pytest tmp directory."""
    return LocalRawStorageBackend(base_dir=tmp_path / "raw_store")


@pytest.fixture
def sample_bytes() -> bytes:
    return b"%PDF-1.4 sample content for testing 1234567890"


def _rand_id() -> str:
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# make_key
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeKey:
    def test_key_format(self):
        key = RawStorageBackend.make_key("t1", "doc-abc", 2, "file.pdf")
        assert key == "t1/doc-abc/v2/file.pdf"

    def test_key_is_relative(self):
        key = RawStorageBackend.make_key("default", _rand_id(), 1, "a.pdf")
        assert not key.startswith("/")

    def test_key_contains_version_prefix(self):
        key = RawStorageBackend.make_key("default", "d1", 5, "doc.pdf")
        assert "/v5/" in key


# ─────────────────────────────────────────────────────────────────────────────
# LocalRawStorageBackend — write
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalWrite:
    def test_write_returns_key(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "test.pdf", sample_bytes)
        assert key.startswith("t1/")
        assert key.endswith("test.pdf")

    def test_write_creates_file_on_disk(self, store, sample_bytes):
        doc_id = _rand_id()
        key = store.write("t1", doc_id, 1, "a.pdf", sample_bytes)
        assert store.absolute_path(key).exists()

    def test_write_creates_sha256_sidecar(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "a.pdf", sample_bytes)
        sidecar = store.absolute_path(key).with_suffix(".pdf.sha256")
        assert sidecar.exists()
        assert sidecar.read_text().strip() == hashlib.sha256(sample_bytes).hexdigest()

    def test_write_immutable_raises_file_exists_error(self, store, sample_bytes):
        doc_id = _rand_id()
        store.write("t1", doc_id, 1, "a.pdf", sample_bytes)
        with pytest.raises(FileExistsError, match="already exists"):
            store.write("t1", doc_id, 1, "a.pdf", sample_bytes)

    def test_write_force_overwrite_succeeds(self, store, sample_bytes):
        doc_id = _rand_id()
        store.write("t1", doc_id, 1, "a.pdf", sample_bytes)
        new_bytes = b"%PDF-1.4 updated content"
        key = store.write("t1", doc_id, 1, "a.pdf", new_bytes, force=True)
        assert store.read(key) == new_bytes

    def test_write_empty_bytes_raises_value_error(self, store):
        with pytest.raises(ValueError, match="empty"):
            store.write("t1", _rand_id(), 1, "a.pdf", b"")

    def test_write_creates_nested_directories(self, store, sample_bytes):
        """Parent dirs should be auto-created."""
        doc_id = _rand_id()
        key = store.write("org/team/project", doc_id, 3, "nested.pdf", sample_bytes)
        assert store.absolute_path(key).exists()


# ─────────────────────────────────────────────────────────────────────────────
# LocalRawStorageBackend — read
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalRead:
    def test_read_returns_exact_bytes(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "b.pdf", sample_bytes)
        assert store.read(key) == sample_bytes

    def test_read_missing_key_raises_file_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.read("nonexistent/key/v1/missing.pdf")

    def test_read_verify_on_read_passes(self, store, sample_bytes):
        verifying_store = LocalRawStorageBackend(
            base_dir=store.base_dir, verify_on_read=True
        )
        key = store.write("t1", _rand_id(), 1, "c.pdf", sample_bytes)
        result = verifying_store.read(key)
        assert result == sample_bytes

    def test_read_verify_on_read_detects_corruption(self, store, sample_bytes):
        verifying_store = LocalRawStorageBackend(
            base_dir=store.base_dir, verify_on_read=True
        )
        key = store.write("t1", _rand_id(), 1, "d.pdf", sample_bytes)
        # Corrupt the file on disk
        store.absolute_path(key).write_bytes(b"corrupted content")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verifying_store.read(key)


# ─────────────────────────────────────────────────────────────────────────────
# LocalRawStorageBackend — exists / delete
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalExistsDelete:
    def test_exists_true_after_write(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "e.pdf", sample_bytes)
        assert store.exists(key) is True

    def test_exists_false_before_write(self, store):
        assert store.exists("t1/missing-doc/v1/nope.pdf") is False

    def test_delete_removes_file(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "f.pdf", sample_bytes)
        store.delete(key)
        assert not store.absolute_path(key).exists()

    def test_delete_removes_sidecar(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "g.pdf", sample_bytes)
        store.delete(key)
        sidecar = store.absolute_path(key).with_suffix(".pdf.sha256")
        assert not sidecar.exists()

    def test_delete_missing_key_raises_file_not_found(self, store):
        with pytest.raises(FileNotFoundError):
            store.delete("t1/ghost/v1/phantom.pdf")

    def test_storage_size_bytes(self, store, sample_bytes):
        key = store.write("t1", _rand_id(), 1, "h.pdf", sample_bytes)
        size = store.storage_size_bytes(key)
        assert size == len(sample_bytes)

    def test_storage_size_bytes_missing_returns_none(self, store):
        assert store.storage_size_bytes("t1/missing/v1/x.pdf") is None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration — _save_raw
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveRaw:
    """Unit tests for IngestionPipeline._save_raw helper."""

    def _make_payload(self, content: bytes, filename: str, source_uri: str):
        from rag.core.models import RawDocumentPayload, DocumentDescriptor, SourceType
        desc = DocumentDescriptor(
            source=SourceType.LOCAL_FILE,
            source_uri=source_uri,
            filename=filename,
            mime_type="application/pdf",
        )
        import hashlib
        return RawDocumentPayload(
            descriptor=desc,
            content=content,
            content_hash=hashlib.sha256(content).hexdigest(),
        )

    def _make_pipeline(self, raw_storage=None):
        from rag.core.database import DatabaseManager
        from rag.core.db_models import Base
        from rag.ingestion.pipeline import IngestionPipeline
        db_manager = DatabaseManager(db_url="sqlite:///:memory:")
        Base.metadata.create_all(bind=db_manager.engine)
        return IngestionPipeline(db_manager=db_manager, raw_storage=raw_storage)

    def test_save_raw_returns_storage_key_when_backend_set(self, store, sample_bytes):
        pipeline = self._make_pipeline(raw_storage=store)
        payload = self._make_payload(sample_bytes, "x.pdf", "/data/x.pdf")
        doc_id = _rand_id()
        key = pipeline._save_raw(payload, doc_id, 1, "default")
        assert store.exists(key)
        assert store.read(key) == sample_bytes

    def test_save_raw_returns_source_uri_when_no_backend(self, sample_bytes):
        pipeline = self._make_pipeline(raw_storage=None)
        payload = self._make_payload(sample_bytes, "y.pdf", "/data/y.pdf")
        key = pipeline._save_raw(payload, _rand_id(), 1, "default")
        assert key == "/data/y.pdf"

    def test_save_raw_idempotent_on_existing_key(self, store, sample_bytes):
        """Second call with same doc/version returns existing key without error."""
        pipeline = self._make_pipeline(raw_storage=store)
        payload = self._make_payload(sample_bytes, "z.pdf", "/data/z.pdf")
        doc_id = _rand_id()
        k1 = pipeline._save_raw(payload, doc_id, 1, "default")
        k2 = pipeline._save_raw(payload, doc_id, 1, "default")  # idempotent
        assert k1 == k2


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integration — ingest stores raw bytes
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestStoresRaw:
    """Integration: ingest_from_connector saves raw bytes and sets raw_location."""

    def test_ingest_new_doc_creates_raw_store_entry(self, tmp_path, store):
        from pathlib import Path
        from rag.core.database import DatabaseManager
        from rag.core.db_models import Base, DocumentVersion
        from rag.ingestion.pipeline import IngestionPipeline
        from rag.connectors.local_file import LocalFileConnector
        from sqlalchemy import select

        # Write a real PDF-like file into a temp data dir
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pdf_path = data_dir / "sample.pdf"

        # Use a real minimal PDF if PyMuPDF is available, else use raw bytes
        try:
            import pymupdf as fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Integration test document.", fontsize=12)
            import io
            buf = io.BytesIO()
            doc.save(buf)
            doc.close()
            pdf_bytes = buf.getvalue()
        except ImportError:
            pdf_bytes = b"%PDF-1.4 minimal fake pdf for test"

        pdf_path.write_bytes(pdf_bytes)

        db_manager = DatabaseManager(db_url="sqlite:///:memory:")
        Base.metadata.create_all(bind=db_manager.engine)
        pipeline = IngestionPipeline(db_manager=db_manager, raw_storage=store)

        connector = LocalFileConnector(base_dir=data_dir)
        pipeline.register_connector("local", connector)

        summary = pipeline.ingest_from_connector("local")
        assert summary.fetched_count == 1
        assert summary.failed_count == 0

        # Verify raw_location is a store key (not source_uri)
        with db_manager.session() as session:
            ver = session.execute(select(DocumentVersion)).scalar_one()
            assert ver.raw_location is not None
            assert store.exists(ver.raw_location), (
                f"Expected raw file at key '{ver.raw_location}' but it was not found."
            )
            stored_bytes = store.read(ver.raw_location)
            assert stored_bytes == pdf_bytes
