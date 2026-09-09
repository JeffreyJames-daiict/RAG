"""Exceptions for the Embedding subsystem."""

from rag.core.exceptions import RAGError


class EmbeddingError(RAGError):
    """Base exception for all embedding generation and persistence errors."""
    pass


class EmbeddingProviderError(EmbeddingError):
    """Raised when an external embedding provider fails or returns an error."""
    pass


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Raised when an embedding provider returns HTTP 429 / rate limit exceeded."""
    pass


class EmbeddingTimeoutError(EmbeddingProviderError):
    """Raised when a request to an embedding provider times out."""
    pass


class EmbeddingValidationError(EmbeddingError):
    """Raised when an embedding vector fails quality, finiteness, or dimensional checks."""
    pass


class EmbeddingDimensionMismatchError(EmbeddingValidationError):
    """Raised when the generated vector dimension does not match the configured dimension."""
    pass


class EmbeddingInputTooLargeError(EmbeddingError):
    """Raised when chunk text exceeds the maximum token context of the embedding model."""
    pass


class EmbeddingPersistenceError(EmbeddingError):
    """Raised when persisting embedding records to the database fails."""
    pass


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when invalid or disallowed embedding configuration is detected (e.g. mock in production)."""
    pass


class EmbeddingZeroVectorError(EmbeddingValidationError):
    """Raised when an embedding vector contains all zeros."""
    pass


class EmbeddingConstantVectorError(EmbeddingValidationError):
    """Raised when an embedding vector contains identical/constant values."""
    pass


class EmbeddingMockDetectedError(EmbeddingError):
    """Raised when mock or dummy embeddings are unexpectedly detected."""
    pass
