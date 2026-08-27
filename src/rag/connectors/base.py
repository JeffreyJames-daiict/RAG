"""Abstract Base Class for Source Connectors in the Enterprise RAG Pipeline."""

from __future__ import annotations
import abc
import hashlib
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional
from rag.core.exceptions import FetchError
from rag.core.models import DocumentDescriptor, RawDocumentPayload, SourceType


class BaseSourceConnector(abc.ABC):
    """Abstract interface for all data source connectors.

    A connector is responsible for:
    1. Validating connectivity/credentials (`validate_connection`)
    2. Discovering document manifests without loading full payloads (`discover`)
    3. Fetching raw bytes on-demand (`fetch`)
    4. Extracting source-specific metadata (`get_metadata`)
    """

    def __init__(self, tenant_id: str = "default"):
        self.tenant_id = tenant_id

    @property
    @abc.abstractmethod
    def source_type(self) -> SourceType:
        """Returns the connector's SourceType enum."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate_connection(self) -> bool:
        """Verify credentials, network accessibility, and permissions."""
        raise NotImplementedError

    @abc.abstractmethod
    def discover(self, since: Optional[datetime] = None) -> Iterator[DocumentDescriptor]:
        """Discover available documents in the source system.

        Args:
            since: Optional timestamp for incremental discovery (only newer items).

        Yields:
            DocumentDescriptor objects.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch(self, descriptor: DocumentDescriptor) -> RawDocumentPayload:
        """Fetch raw document content bytes for a given descriptor.

        Args:
            descriptor: The DocumentDescriptor identified during discovery.

        Returns:
            RawDocumentPayload containing bytes, computed hash, and metadata.
        """
        raise NotImplementedError

    def get_metadata(self, descriptor: DocumentDescriptor) -> Dict[str, Any]:
        """Retrieve source-specific metadata for a descriptor."""
        return descriptor.custom_metadata

    def fetch_batch(self, descriptors: List[DocumentDescriptor]) -> Iterator[RawDocumentPayload]:
        """Fetch a batch of documents sequentially or in subclasses with concurrency.

        Args:
            descriptors: List of descriptors to fetch.

        Yields:
            RawDocumentPayload for each successfully fetched document.
        """
        for desc in descriptors:
            try:
                yield self.fetch(desc)
            except Exception as e:
                raise FetchError(f"Failed to fetch document {desc.source_uri}: {e}") from e

    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Calculate SHA-256 hash of content bytes for change detection."""
        hasher = hashlib.sha256()
        hasher.update(content)
        return hasher.hexdigest()
