"""Local filesystem implementation of RawStorageBackend.

Stores document bytes under a structured, version-keyed directory tree::

    {base_dir}/{tenant_id}/{document_id}/v{version}/{filename}

Key properties:
- **Immutable by default**: writing a key that already exists raises
  ``FileExistsError`` unless ``force=True``.
- **Hash-verified reads**: optional SHA-256 verification on read-back.
- **Atomic writes**: bytes are written to a ``.tmp`` file first, then
  renamed — preventing partially-written files from being read.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from rag.storage.base import RawStorageBackend

logger = logging.getLogger(__name__)


class LocalRawStorageBackend(RawStorageBackend):
    """Immutable local-disk raw storage backend.

    Usage::

        backend = LocalRawStorageBackend(base_dir="/data/raw_store")
        key = backend.write("default", doc_id, 1, "report.pdf", pdf_bytes)
        # key = "default/{doc_id}/v1/report.pdf"

        raw = backend.read(key)  # returns exact original bytes
        assert backend.exists(key)
        backend.delete(key)      # GDPR removal

    Args:
        base_dir:       Root directory for all stored files.
                        Created automatically if it doesn't exist.
        verify_on_read: If True, re-compute SHA-256 after reading and
                        compare to the stored ``.sha256`` sidecar file.
                        Raises ``ValueError`` on mismatch (detects
                        silent disk corruption).
    """

    def __init__(self, base_dir: str | Path, *, verify_on_read: bool = False):
        self.base_dir = Path(base_dir).resolve()
        self.verify_on_read = verify_on_read
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LocalRawStorageBackend initialised at: %s", self.base_dir)

    # ------------------------------------------------------------------
    # RawStorageBackend interface
    # ------------------------------------------------------------------

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
        """Write bytes to disk and return the storage key.

        The file is written atomically via a ``.tmp`` rename so that
        a crash mid-write never leaves a corrupt file.

        Args:
            tenant_id:   Tenant scope.
            document_id: Document UUID string.
            version:     Version number (1, 2, 3 …).
            filename:    Original filename (leaf name on disk).
            content:     Raw bytes to persist.
            force:       Overwrite existing file. Defaults to False.

        Returns:
            The portable storage key (relative path string).

        Raises:
            FileExistsError: If the key already exists and ``force=False``.
            ValueError: If ``content`` is empty.
        """
        if not content:
            raise ValueError(
                f"Cannot write empty content for {tenant_id}/{document_id}/v{version}/{filename}."
            )

        key = self.make_key(tenant_id, document_id, version, filename)
        dest = self._abs_path(key)

        if dest.exists() and not force:
            raise FileExistsError(
                f"Raw storage key already exists (use force=True to overwrite): {key}"
            )

        # Ensure parent directory exists
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: tmp file → rename
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            tmp_path.write_bytes(content)
            tmp_path.replace(dest)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        # Write SHA-256 sidecar for integrity verification
        sha256 = hashlib.sha256(content).hexdigest()
        sidecar = dest.with_suffix(dest.suffix + ".sha256")
        sidecar.write_text(sha256)

        logger.info(
            "Stored %d bytes → %s  (sha256: %s…)",
            len(content), key, sha256[:16],
        )
        return key

    def read(self, storage_key: str) -> bytes:
        """Read and return the raw bytes stored at ``storage_key``.

        Args:
            storage_key: The key returned by :meth:`write`.

        Returns:
            Original raw bytes.

        Raises:
            FileNotFoundError: If ``storage_key`` does not exist.
            ValueError: If ``verify_on_read=True`` and the SHA-256
                        sidecar does not match the file content
                        (indicates disk corruption).
        """
        path = self._abs_path(storage_key)
        if not path.exists():
            raise FileNotFoundError(
                f"No raw content found at storage key: {storage_key}"
            )

        content = path.read_bytes()

        if self.verify_on_read:
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if sidecar.exists():
                expected = sidecar.read_text().strip()
                actual = hashlib.sha256(content).hexdigest()
                if actual != expected:
                    raise ValueError(
                        f"SHA-256 mismatch for '{storage_key}': "
                        f"expected {expected[:16]}…, got {actual[:16]}…. "
                        "File may be corrupted."
                    )
            else:
                logger.warning(
                    "verify_on_read=True but no .sha256 sidecar found for '%s'.",
                    storage_key,
                )

        logger.debug("Read %d bytes from %s", len(content), storage_key)
        return content

    def exists(self, storage_key: str) -> bool:
        """Return True if content exists at ``storage_key``."""
        return self._abs_path(storage_key).exists()

    def delete(self, storage_key: str) -> None:
        """Delete stored file and its SHA-256 sidecar.

        Raises:
            FileNotFoundError: If ``storage_key`` does not exist.
        """
        path = self._abs_path(storage_key)
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot delete — no raw content at storage key: {storage_key}"
            )
        path.unlink()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar.unlink(missing_ok=True)
        logger.info("Deleted raw storage key: %s", storage_key)

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    def absolute_path(self, storage_key: str) -> Path:
        """Return the absolute filesystem path for a storage key.

        Useful for debugging and monitoring tools.
        """
        return self._abs_path(storage_key)

    def storage_size_bytes(self, storage_key: str) -> Optional[int]:
        """Return file size in bytes, or None if not found."""
        path = self._abs_path(storage_key)
        return path.stat().st_size if path.exists() else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _abs_path(self, storage_key: str) -> Path:
        """Resolve a storage key to an absolute filesystem path."""
        return self.base_dir / storage_key
