"""Core data structures, database models, and exceptions."""

from rag.core.models import (
        SourceType,
        DocumentStatus,
        DocumentDescriptor,
        RawDocumentPayload,
        DocumentSchema,
        DocumentVersionSchema,
        ChunkSchema,
        EmbeddingSchema,
        SyncSummary,
    )
from rag.core.db_models import (
        Base,
        Document,
        DocumentVersion,
        Chunk,
        Embedding,
    )
from rag.core.database import DatabaseManager
from rag.core.exceptions import (
        RAGError,
        ConnectorError,
        ConnectionValidationError,
        DiscoveryError,
        FetchError,
        AuthenticationError,
        RateLimitError,
        IngestionPipelineError,
    )

__all__ = [
    "SourceType",
    "DocumentStatus",
    "DocumentDescriptor",
    "RawDocumentPayload",
    "DocumentSchema",
    "DocumentVersionSchema",
    "ChunkSchema",
    "EmbeddingSchema",
    "SyncSummary",
    "Base",
    "Document",
    "DocumentVersion",
    "Chunk",
    "Embedding",
    "DatabaseManager",
    "RAGError",
    "ConnectorError",
    "ConnectionValidationError",
    "DiscoveryError",
    "FetchError",
    "AuthenticationError",
    "RateLimitError",
    "IngestionPipelineError",
]
