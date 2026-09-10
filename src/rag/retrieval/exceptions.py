"""Custom typed exceptions for the Vector Retrieval subsystem."""

from __future__ import annotations

from rag.core.exceptions import RAGError


class RetrievalError(RAGError):
    """Base exception for all retrieval errors."""
    pass


class EmptyQueryError(RetrievalError):
    """Raised when retrieval query text is empty or whitespace-only."""
    pass


class QueryTooLongError(RetrievalError):
    """Raised when retrieval query text exceeds the maximum permitted character budget."""
    pass


class TenantIsolationError(RetrievalError):
    """Raised when tenant isolation is violated or unauthorized access is detected."""
    pass


class TenantContextMissingError(TenantIsolationError):
    """Raised when tenant_id is missing, empty, or not provided in retrieval context."""
    pass


class IncompatibleEmbeddingError(RetrievalError):
    """Raised when query embedding model, version, or dimension does not match indexed chunks."""
    pass


class EmbeddingDimensionMismatchError(IncompatibleEmbeddingError):
    """Raised when vector dimensionality does not match expected model dimension."""
    pass


class UnsupportedMetricError(RetrievalError):
    """Raised when an unsupported similarity or distance metric is requested."""
    pass


class MissingEmbeddingProviderError(RetrievalError):
    """Raised when the required embedding provider is not configured or unavailable."""
    pass


class DatabaseRetrievalError(RetrievalError):
    """Raised when an underlying database query, index scan, or connection fails during retrieval."""
    pass


class FilterValidationError(RetrievalError):
    """Raised when retrieval filters are invalid, malformed, or exceed permitted constraints."""
    pass


class BackendUnavailableError(RetrievalError):
    """Raised when the configured vector retrieval backend is unavailable or not permitted."""
    pass


class RetrievalTimeoutError(RetrievalError):
    """Raised when vector search exceeds the configured timeout threshold."""
    pass


class RerankingError(RetrievalError):
    """Base exception for all reranking-related errors."""
    pass


class RerankingTimeoutError(RerankingError):
    """Raised when the reranking provider exceeds the configured timeout."""
    pass


class RerankingProviderError(RerankingError):
    """Raised when the reranking provider returns an error, is unavailable, or returns invalid data."""
    pass
