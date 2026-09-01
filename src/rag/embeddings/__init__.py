"""Embedding subsystem package."""

from rag.embeddings.base import BaseEmbeddingProvider
from rag.embeddings.exceptions import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingInputTooLargeError,
    EmbeddingPersistenceError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    EmbeddingValidationError,
)
from rag.embeddings.models import (
    BatchEmbeddingResult,
    ChunkEmbeddingFailure,
    EmbeddingResult,
    EmbeddingStatus,
    EmbeddingSummary,
    EmbeddingTask,
)
from rag.embeddings.providers import (
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from rag.embeddings.service import EmbeddingService
from rag.embeddings.validator import EmbeddingValidator

__all__ = [
    "BaseEmbeddingProvider",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "get_embedding_provider",
    "EmbeddingService",
    "EmbeddingValidator",
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "ChunkEmbeddingFailure",
    "EmbeddingStatus",
    "EmbeddingSummary",
    "EmbeddingTask",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingRateLimitError",
    "EmbeddingTimeoutError",
    "EmbeddingValidationError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingInputTooLargeError",
    "EmbeddingPersistenceError",
]
