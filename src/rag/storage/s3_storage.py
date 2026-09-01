"""S3 / MinIO raw storage backend.

Implements :class:`RawStorageBackend` using any S3-compatible object store
(AWS S3, MinIO, etc.) via ``boto3``.

Storage key format (same as all backends):
    ``{tenant_id}/{document_id}/v{version}/{filename}``

Example:
    ``default/abc123/v1/report.pdf``
    ``default/abc123/v1/assets/figures/fig_001.png``
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from rag.storage.base import RawStorageBackend

logger = logging.getLogger(__name__)


class S3RawStorageBackend(RawStorageBackend):
    """Stores and retrieves raw document bytes in an S3-compatible bucket (AWS S3 / MinIO).

    All raw PDFs, figure images, and version artifacts are written
    under a structured deterministic key so the pipeline can retrieve
    them without touching the original source system again.
    """

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        verify_on_read: bool = True,
        storage_prefix: str = "storage",
    ):
        """
        Args:
            bucket_name:          S3 bucket to use.
            aws_access_key_id:    Access key (or MinIO access key).
            aws_secret_access_key: Secret key (or MinIO secret key).
            endpoint_url:         Override endpoint for MinIO (e.g. 'http://127.0.0.1:9000').
            region_name:          AWS region (not required for MinIO).
            verify_on_read:       SHA-256 integrity check on every read.
            storage_prefix:       Top-level prefix inside bucket, default 'storage'.
                                  Keeps pipeline artifacts separate from input documents.
        """
        self.bucket_name = bucket_name
        self.verify_on_read = verify_on_read
        self.storage_prefix = storage_prefix.strip("/")

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
        logger.info(
            "S3RawStorageBackend initialised (bucket=%s, prefix=%s, endpoint=%s)",
            bucket_name,
            storage_prefix,
            endpoint_url or "AWS",
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _object_key(self, storage_key: str) -> str:
        """Prepend bucket prefix to the canonical storage key."""
        if self.storage_prefix:
            return f"{self.storage_prefix}/{storage_key}"
        return storage_key

    def _canonical_key(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
        filename: str,
    ) -> str:
        """Build canonical storage key: ``{tenant}/{doc_id}/v{version}/{filename}``."""
        return f"{tenant_id}/{document_id}/v{version}/{filename}"

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    # ── RawStorageBackend interface ───────────────────────────────────────────

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
        if not content:
            raise ValueError("Cannot store empty content")

        canonical = self._canonical_key(tenant_id, document_id, version, filename)
        obj_key = self._object_key(canonical)

        # Idempotency: check if object already exists
        if not force:
            try:
                self.s3.head_object(Bucket=self.bucket_name, Key=obj_key)
                raise FileExistsError(
                    f"S3 object already exists: s3://{self.bucket_name}/{obj_key}"
                )
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                    raise

        # Compute SHA-256 and store as metadata tag
        content_hash = self._sha256(content)

        # Infer content type
        content_type = "application/octet-stream"
        fname_lower = filename.lower()
        if fname_lower.endswith(".pdf"):
            content_type = "application/pdf"
        elif fname_lower.endswith(".png"):
            content_type = "image/png"
        elif fname_lower.endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif fname_lower.endswith(".webp"):
            content_type = "image/webp"

        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=obj_key,
            Body=content,
            ContentType=content_type,
            Metadata={"sha256": content_hash},
        )
        logger.debug("S3 write: s3://%s/%s (%d bytes)", self.bucket_name, obj_key, len(content))
        return canonical

    def read(self, storage_key: str) -> bytes:
        obj_key = self._object_key(storage_key)
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=obj_key)
            data = response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                raise FileNotFoundError(
                    f"S3 object not found: s3://{self.bucket_name}/{obj_key}"
                ) from e
            raise

        if self.verify_on_read:
            stored_hash = response.get("Metadata", {}).get("sha256")
            if stored_hash:
                actual_hash = self._sha256(data)
                if actual_hash != stored_hash:
                    raise IOError(
                        f"SHA-256 integrity check failed for s3://{self.bucket_name}/{obj_key}. "
                        f"Expected {stored_hash}, got {actual_hash}"
                    )

        return data

    def exists(self, storage_key: str) -> bool:
        obj_key = self._object_key(storage_key)
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=obj_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, storage_key: str) -> None:
        obj_key = self._object_key(storage_key)
        if not self.exists(storage_key):
            raise FileNotFoundError(
                f"S3 object not found for deletion: s3://{self.bucket_name}/{obj_key}"
            )
        self.s3.delete_object(Bucket=self.bucket_name, Key=obj_key)
        logger.debug("S3 delete: s3://%s/%s", self.bucket_name, obj_key)

    def get_storage_size_bytes(self, storage_key: str) -> Optional[int]:
        obj_key = self._object_key(storage_key)
        try:
            response = self.s3.head_object(Bucket=self.bucket_name, Key=obj_key)
            return response.get("ContentLength")
        except ClientError:
            return None
