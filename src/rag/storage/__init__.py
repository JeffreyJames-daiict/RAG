"""Raw storage package — immutable preservation of original document bytes.

Exports:
    RawStorageBackend          — abstract interface for all storage backends
    DatabaseRawStorageBackend  — PostgreSQL database-backed implementation (``raw`` schema)
    LocalRawStorageBackend     — local filesystem implementation (legacy)
    S3RawStorageBackend        — S3 / MinIO object storage implementation
    get_storage_backend        — factory helper
"""

import os
from typing import Optional

from rag.core.config import Settings, get_settings
from rag.core.database import DatabaseManager
from rag.storage.base import RawStorageBackend
from rag.storage.db_storage import DatabaseRawStorageBackend
from rag.storage.local_storage import LocalRawStorageBackend
from rag.storage.s3_storage import S3RawStorageBackend


def get_storage_backend(
    settings: Optional[Settings] = None,
    db_manager: Optional[DatabaseManager] = None,
) -> RawStorageBackend:
    """Instantiate the configured raw storage backend.

    Supported backend_type values:
        - ``"database"`` / ``"postgres"`` / ``"postgresql"`` / ``"db"``
          → :class:`DatabaseRawStorageBackend` (PostgreSQL ``raw.document_payloads``)
        - ``"s3"`` / ``"minio"``
          → :class:`S3RawStorageBackend` (AWS S3 or MinIO bucket)
        - any other value
          → :class:`LocalRawStorageBackend` (local filesystem)
    """
    cfg = settings or get_settings()
    backend_type = cfg.storage.backend_type.lower()

    if backend_type in ("database", "postgres", "postgresql", "db"):
        if db_manager is None:
            db_manager = DatabaseManager(cfg.database.url)
        return DatabaseRawStorageBackend(db_manager, schema_name=cfg.storage.raw_schema)

    elif backend_type in ("s3", "minio"):
        # Prefer config values; fall back to environment variables for MinIO
        bucket = (
            cfg.storage.s3_bucket
            or os.getenv("MINIO_BUCKET")
            or os.getenv("S3_BUCKET")
        )
        endpoint = (
            cfg.storage.s3_endpoint_url
            or _build_minio_endpoint()
        )
        access_key = (
            cfg.storage.s3_access_key
            or os.getenv("MINIO_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
        )
        secret_key = (
            cfg.storage.s3_secret_key
            or os.getenv("MINIO_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        if not bucket:
            raise ValueError(
                "S3/MinIO backend requires a bucket name. "
                "Set storage.s3_bucket in config.json or MINIO_BUCKET in .env"
            )
        return S3RawStorageBackend(
            bucket_name=bucket,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            region_name=cfg.storage.s3_region,
            verify_on_read=cfg.storage.verify_on_read,
            storage_prefix=cfg.storage.s3_storage_prefix,
        )

    else:
        return LocalRawStorageBackend(
            base_dir=cfg.storage.raw_storage_dir,
            verify_on_read=cfg.storage.verify_on_read,
        )


def get_asset_storage_backend(
    settings: Optional[Settings] = None,
    db_manager: Optional[DatabaseManager] = None,
) -> RawStorageBackend:
    """Instantiate the dedicated storage backend for visual assets/figures (MinIO by default)."""
    cfg = settings or get_settings()
    backend_type = (cfg.asset.storage_backend_type or cfg.storage.backend_type).lower()

    if backend_type in ("s3", "minio"):
        bucket = (
            cfg.storage.s3_bucket
            or os.getenv("MINIO_BUCKET")
            or os.getenv("S3_BUCKET")
            or "rag"
        )
        endpoint = (
            cfg.storage.s3_endpoint_url
            or _build_minio_endpoint()
            or "http://127.0.0.1:9000"
        )
        access_key = (
            cfg.storage.s3_access_key
            or os.getenv("MINIO_ACCESS_KEY")
            or os.getenv("AWS_ACCESS_KEY_ID")
            or "minioadmin"
        )
        secret_key = (
            cfg.storage.s3_secret_key
            or os.getenv("MINIO_SECRET_KEY")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
            or "minioadmin"
        )
        return S3RawStorageBackend(
            bucket_name=bucket,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            region_name=cfg.storage.s3_region,
            verify_on_read=cfg.storage.verify_on_read,
            storage_prefix=cfg.storage.s3_storage_prefix,
        )
    elif backend_type in ("database", "postgres", "postgresql", "db"):
        if db_manager is None:
            db_manager = DatabaseManager(cfg.database.url)
        return DatabaseRawStorageBackend(db_manager, schema_name=cfg.storage.raw_schema)
    else:
        return LocalRawStorageBackend(
            base_dir=cfg.storage.raw_storage_dir,
            verify_on_read=cfg.storage.verify_on_read,
        )


def _build_minio_endpoint() -> Optional[str]:
    """Build MinIO endpoint URL from MINIO_ENDPOINT and MINIO_SECURE env vars."""
    endpoint = os.getenv("MINIO_ENDPOINT")
    if not endpoint:
        return None
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    scheme = "https" if secure else "http"
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"{scheme}://{endpoint}"


__all__ = [
    "RawStorageBackend",
    "DatabaseRawStorageBackend",
    "LocalRawStorageBackend",
    "S3RawStorageBackend",
    "get_storage_backend",
    "get_asset_storage_backend",
]
