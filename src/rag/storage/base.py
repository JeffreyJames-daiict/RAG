"""Abstract interface for immutable raw document storage.

All storage backends must implement this interface. This enables
swapping local disk storage for S3, GCS, or Azure Blob with zero
changes to the pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RawStorageBackend(ABC):
    """Immutable binary storage for original document payloads.

    Each document version is stored exactly once, addressed by a
    storage key of the form::

        {tenant_id}/{document_id}/v{version}/{filename}

    This key is persisted in ``DocumentVersion.raw_location`` and used
    by all downstream pipeline stages (parse, chunk, embed) to read
    the original bytes without touching the live source system again.
    """

    @abstractmethod
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
        """Write raw bytes and return the opaque storage key.

        Args:
            tenant_id:   Tenant scope.
            document_id: UUID string of the parent document.
            version:     Version number being stored (1, 2, 3 …).
            filename:    Original filename (used as the leaf name).
            content:     Raw bytes to persist immutably.
            force:       If True, overwrite an existing entry.
                         If False (default), raise ``FileExistsError``
                         when the key already exists.

        Returns:
            An opaque storage key that can be passed back to
            :meth:`read`, :meth:`exists`, and :meth:`delete`.
        """

    @abstractmethod
    def read(self, storage_key: str) -> bytes:
        """Return the raw bytes for a previously stored document version.

        Args:
            storage_key: The key returned by :meth:`write`.

        Raises:
            FileNotFoundError: If no content exists at ``storage_key``.
        """

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """Return True if content exists at the given storage key."""

    @abstractmethod
    def delete(self, storage_key: str) -> None:
        """Delete the stored content at ``storage_key`` (e.g. for GDPR).

        Raises:
            FileNotFoundError: If no content exists at ``storage_key``.
        """

    @staticmethod
    def make_key(tenant_id: str, document_id: str, version: int, filename: str) -> str:
        """Construct the canonical storage key for a document version.

        The key is a portable relative path — independent of the
        backend's physical base directory or bucket prefix::

            {tenant_id}/{document_id}/v{version}/{filename}

        Args:
            tenant_id:   Tenant scope.
            document_id: UUID string.
            version:     Version number.
            filename:    Original filename.

        Returns:
            A forward-slash separated relative key string.
        """
        return f"{tenant_id}/{document_id}/v{version}/{filename}"
