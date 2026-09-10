"""Protocols and interfaces for the Vector Retrieval subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from rag.retrieval.filters import RetrievalFilters
from rag.retrieval.models import RetrievalRequest, RetrievalResponse


@dataclass
class RetrievedCandidate:
    """Internal candidate representation returned by repository before service hydration."""

    chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    text: str
    similarity_score: float
    distance: float
    tenant_id: str = "default"
    title: str = ""
    section: Optional[str] = None
    page: Optional[int] = None
    parent_chunk_id: Optional[str] = None
    raw_score: float = 0.0
    normalized_score: float = 0.0
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    chunking_version: str = "v1.0"
    embedding_model: str = ""
    embedding_provider: str = ""
    embedding_dimension: int = 0
    # Document fields
    source_uri: str = ""
    filename: str = ""
    mime_type: str = ""
    content_hash: str = ""


class VectorRepository(Protocol):
    """Protocol for vector similarity search repositories."""

    @property
    def backend_name(self) -> str:
        """Name of the vector repository implementation (e.g. 'pgvector', 'python')."""
        ...

    def is_available(self) -> bool:
        """Check whether the repository backend is supported and healthy in current environment."""
        ...

    def search_vectors(
        self,
        query_vector: List[float],
        request: RetrievalRequest,
        model_name: str,
        model_version: str,
    ) -> Tuple[List[RetrievedCandidate], int]:
        """Perform vector similarity search for candidate chunks.

        Returns:
            Tuple of (list of candidates sorted by similarity desc, total candidates evaluated).
        """
        ...


class VectorRetriever(Protocol):
    """Protocol for high-level vector retrieval service supporting both async and sync operations."""

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResponse:
        """Asynchronously execute vector similarity search with query embedding and filtering."""
        ...

    def retrieve_sync(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResponse:
        """Synchronously execute vector similarity search with query embedding and filtering."""
        ...


class HybridRetriever(Protocol):
    """Protocol for high-level hybrid retrieval service combining dense and sparse search."""

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResponse:
        """Asynchronously execute hybrid retrieval combining dense vector and sparse BM25."""
        ...

    def retrieve_sync(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResponse:
        """Synchronously execute hybrid retrieval combining dense vector and sparse BM25."""
        ...
