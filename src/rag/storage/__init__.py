"""Raw storage package — immutable preservation of original document bytes.

Exports:
    RawStorageBackend          — abstract interface for all storage backends
    DatabaseRawStorageBackend  — PostgreSQL database-backed implementation (``raw`` schema)
    LocalRawStorageBackend     — local filesystem implementation (legacy)
    get_storage_backend        — factory helper
"""

from typing import Optional

from rag.core.config import Settings, get_settings
from rag.core.database import DatabaseManager
from rag.storage.base import RawStorageBackend
from rag.storage.db_storage import DatabaseRawStorageBackend
from rag.storage.local_storage import LocalRawStorageBackend


def get_storage_backend(
    settings: Optional[Settings] = None,
    db_manager: Optional[DatabaseManager] = None,
) -> RawStorageBackend:
    """Instantiate the configured raw storage backend.

    When ``backend_type`` is ``"database"`` (the default), raw payloads are
    stored in the PostgreSQL ``raw.document_payloads`` table.
    """
    cfg = settings or get_settings()
    backend_type = cfg.storage.backend_type.lower()

    if backend_type in ("database", "postgres", "postgresql", "db"):
        if db_manager is None:
            db_manager = DatabaseManager(cfg.database.url)
        return DatabaseRawStorageBackend(db_manager, schema_name=cfg.storage.raw_schema)
    else:
        return LocalRawStorageBackend(
            base_dir=cfg.storage.raw_storage_dir,
            verify_on_read=cfg.storage.verify_on_read,
        )


__all__ = [
    "RawStorageBackend",
    "DatabaseRawStorageBackend",
    "LocalRawStorageBackend",
    "get_storage_backend",
]
