"""Custom exceptions for the Enterprise RAG pipeline."""


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""
    pass


class ConnectorError(RAGError):
    """Raised when an error occurs during source connection or discovery."""
    pass


class ConnectionValidationError(ConnectorError):
    """Raised when connector validation or health check fails."""
    pass


class DiscoveryError(ConnectorError):
    """Raised when discovery of source documents fails."""
    pass


class FetchError(ConnectorError):
    """Raised when fetching a specific document fails."""
    pass


class AuthenticationError(ConnectorError):
    """Raised when source authentication or credential check fails."""
    pass


class RateLimitError(ConnectorError):
    """Raised when a data source rate limit is exceeded."""
    pass


class IngestionPipelineError(RAGError):
    """Raised when an error occurs in the ingestion orchestrator."""
    pass
