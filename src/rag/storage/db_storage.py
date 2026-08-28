"""Database-backed raw document payload storage in dedicated 'raw' schema.

Implements RawStorageBackend by storing binary payloads directly in PostgreSQL
inside the ``raw.document_payloads`` table.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from rag.core.database import DatabaseManager
from rag.core.db_models import RawDocumentPayload
from rag.storage.base import RawStorageBackend

logger = logging.getLogger(__name__)


class DatabaseRawStorageBackend(RawStorageBackend):
    """Stores and retrieves raw immutable binary document payloads in the ``raw`` PostgreSQL schema."""

    def __init__(self, db_manager: DatabaseManager, schema_name: str = "raw"):
        self.db_manager = db_manager
        self.schema_name = schema_name
        logger.info("DatabaseRawStorageBackend initialised (schema=%s)", self.schema_name)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_key(key: str) -> tuple:
        """Split a canonical key ``tenant/doc_id/vN/filename`` into components."""
        parts = key.split("/", 3)
        if len(parts) < 4:
            raise ValueError(f"Invalid storage key: '{key}'")
        tenant_id = parts[0]
        doc_id = parts[1]
        ver_str = parts[2].lstrip("v")
        try:
            version = int(ver_str)
        except ValueError:
            version = 1
        filename = parts[3]
        return tenant_id, doc_id, version, filename

    # ── abstract interface implementation ────────────────────────────────────

    def write(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        filename: str,
        content: bytes,
        *,
        force: bool = False,
    ) -> str:
        """Store raw binary content into ``raw.document_payloads``."""
        if not content:
            raise ValueError("Cannot store empty content bytes in raw storage.")

        sha256_hash = hashlib.sha256(content).hexdigest()
        size_bytes = len(content)
        key = self.make_key(tenant_id, document_id, version, filename)

        with self.db_manager.session() as session:
            existing = session.execute(
                select(RawDocumentPayload).where(
                    RawDocumentPayload.document_id == document_id,
                    RawDocumentPayload.version == version,
                )
            ).scalars().first()

            if existing is not None:
                if not force:
                    raise FileExistsError(
                        f"Raw payload already exists for document '{document_id}' "
                        f"version {version} (key='{key}')."
                    )
                existing.payload = content
                existing.size_bytes = size_bytes
                existing.sha256 = sha256_hash
                existing.content_hash = sha256_hash
                existing.filename = filename
                existing.created_at = datetime.now(timezone.utc)
                logger.info("Overwrote raw payload in DB for doc %s v%d (%d bytes)", document_id, version, size_bytes)
            else:
                record = RawDocumentPayload(
                    payload_id=str(uuid.uuid4()),
                    document_id=document_id,
                    version=version,
                    tenant_id=tenant_id,
                    filename=filename,
                    mime_type="application/octet-stream",
                    content_hash=sha256_hash,
                    sha256=sha256_hash,
                    size_bytes=size_bytes,
                    payload=content,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(record)
                logger.info(
                    "Stored raw payload → raw.document_payloads doc=%s v%d (%d bytes, sha256=%s…)",
                    document_id, version, size_bytes, sha256_hash[:16],
                )

        return key

    def read(self, storage_key: str) -> bytes:
        """Fetch raw binary content from ``raw.document_payloads`` and verify SHA-256."""
        _, doc_id, version, _ = self._parse_key(storage_key)

        with self.db_manager.session() as session:
            record = session.execute(
                select(RawDocumentPayload).where(
                    RawDocumentPayload.document_id == doc_id,
                    RawDocumentPayload.version == version,
                )
            ).scalars().first()

            if record is None:
                raise FileNotFoundError(
                    f"Raw storage key not found in database: '{storage_key}' (doc={doc_id}, v={version})"
                )

            content = record.payload
            actual_sha = hashlib.sha256(content).hexdigest()
            if actual_sha != record.sha256:
                raise ValueError(
                    f"Data corruption detected for key '{storage_key}': "
                    f"stored sha256={record.sha256}, actual sha256={actual_sha}"
                )
            return content

    def exists(self, storage_key: str) -> bool:
        """Check if raw payload exists."""
        try:
            _, doc_id, version, _ = self._parse_key(storage_key)
        except ValueError:
            return False

        with self.db_manager.session() as session:
            row = session.execute(
                select(RawDocumentPayload.payload_id).where(
                    RawDocumentPayload.document_id == doc_id,
                    RawDocumentPayload.version == version,
                )
            ).first()
            return row is not None

    def delete(self, storage_key: str) -> None:
        """Delete raw document payload from the database."""
        _, doc_id, version, _ = self._parse_key(storage_key)

        with self.db_manager.session() as session:
            record = session.execute(
                select(RawDocumentPayload).where(
                    RawDocumentPayload.document_id == doc_id,
                    RawDocumentPayload.version == version,
                )
            ).scalars().first()

            if record is None:
                raise FileNotFoundError(f"Cannot delete: key '{storage_key}' not found in DB.")

            session.delete(record)
            logger.info("Deleted raw payload from DB for key '%s'", storage_key)

    def get_storage_size_bytes(self, storage_key: str) -> Optional[int]:
        """Return stored payload size in bytes, or None if not found."""
        try:
            _, doc_id, version, _ = self._parse_key(storage_key)
        except ValueError:
            return None

        with self.db_manager.session() as session:
            row = session.execute(
                select(RawDocumentPayload.size_bytes).where(
                    RawDocumentPayload.document_id == doc_id,
                    RawDocumentPayload.version == version,
                )
            ).scalars().first()
            return int(row) if row is not None else None
