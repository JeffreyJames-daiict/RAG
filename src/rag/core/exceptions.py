"""Custom exception taxonomy and typed error hierarchy for the Enterprise RAG pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCategory(str, Enum):
    """Broad functional categories for errors across the RAG system."""

    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    EMBEDDING = "embedding"
    DATABASE = "database"
    SEARCH = "search"
    RERANKING = "reranking"
    LLM_PROVIDER = "llm_provider"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONFIGURATION = "configuration"
    SERIALIZATION = "serialization"
    CIRCUIT_BREAKER = "circuit_breaker"
    CONNECTOR = "connector"
    INTERNAL = "internal"


class RAGError(Exception):
    """Base exception for all RAG pipeline errors with structured metadata and safe serialization."""

    def __init__(
        self,
        message: str = "An unexpected error occurred in the RAG pipeline.",
        error_code: str = "RAG_INTERNAL_ERROR",
        category: ErrorCategory = ErrorCategory.INTERNAL,
        is_retryable: bool = False,
        status_code: int = 500,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.is_retryable = is_retryable
        self.status_code = status_code
        self.tenant_id = tenant_id
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a safe, sanitized dictionary suitable for API responses and structured logs."""
        payload: Dict[str, Any] = {
            "error_code": self.error_code,
            "category": self.category.value if isinstance(self.category, ErrorCategory) else str(self.category),
            "message": self.message,
            "is_retryable": self.is_retryable,
            "status_code": self.status_code,
        }
        if self.tenant_id:
            payload["tenant_id"] = self.tenant_id
        if self.details:
            # Mask any potentially sensitive detail keys
            sanitized_details = {}
            for k, v in self.details.items():
                if any(sec in k.lower() for sec in ("key", "secret", "token", "password", "auth")):
                    sanitized_details[k] = "******"
                else:
                    sanitized_details[k] = v
            payload["details"] = sanitized_details
        return payload

    def __str__(self) -> str:
        tenant_part = f" [tenant={self.tenant_id}]" if self.tenant_id else ""
        return f"[{self.error_code}]{tenant_part} {self.message}"


# ---------------------------------------------------------------------------
# Typed Subsystem Exceptions
# ---------------------------------------------------------------------------

class ValidationError(RAGError):
    """Raised when input query, parameters, or schema validation fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_FAILED",
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            is_retryable=False,
            status_code=400,
            tenant_id=tenant_id,
            details=details,
        )


class ConnectorError(RAGError):
    """Raised when an error occurs during source connection, discovery, or ingestion."""

    def __init__(
        self,
        message: str = "Connector error",
        error_code: str = "CONNECTOR_ERROR",
        category: ErrorCategory = ErrorCategory.CONNECTOR,
        is_retryable: bool = False,
        status_code: int = 502,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=category,
            is_retryable=is_retryable,
            status_code=status_code,
            tenant_id=tenant_id,
            details=details,
        )


class AuthenticationError(ConnectorError):
    """Raised when authentication or credential verification fails."""

    def __init__(
        self,
        message: str = "Authentication failed.",
        error_code: str = "AUTH_FAILED",
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHENTICATION,
            is_retryable=False,
            status_code=401,
            tenant_id=tenant_id,
            details=details,
        )


class AuthorizationError(RAGError):
    """Raised when tenant or user lacks permission to access a resource."""

    def __init__(
        self,
        message: str = "Access denied for the requested resource.",
        error_code: str = "PERMISSION_DENIED",
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHORIZATION,
            is_retryable=False,
            status_code=403,
            tenant_id=tenant_id,
            details=details,
        )


class EmbeddingError(RAGError):
    """Raised when generating dense vector embeddings fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "EMBEDDING_GENERATION_FAILED",
        is_retryable: bool = True,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.EMBEDDING,
            is_retryable=is_retryable,
            status_code=502,
            tenant_id=tenant_id,
            details=details,
        )


class DatabaseError(RAGError):
    """Raised when database query, persistence, or connection encounters an error."""

    def __init__(
        self,
        message: str,
        error_code: str = "DATABASE_ERROR",
        is_retryable: bool = True,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.DATABASE,
            is_retryable=is_retryable,
            status_code=500,
            tenant_id=tenant_id,
            details=details,
        )


class SearchError(RAGError):
    """Raised when search execution encounters an error."""

    def __init__(
        self,
        message: str,
        error_code: str = "SEARCH_EXECUTION_FAILED",
        is_retryable: bool = False,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SEARCH,
            is_retryable=is_retryable,
            status_code=500,
            tenant_id=tenant_id,
            details=details,
        )


class RerankingError(RAGError):
    """Raised when candidate reranking encounters an error."""

    def __init__(
        self,
        message: str,
        error_code: str = "RERANKING_FAILED",
        is_retryable: bool = True,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.RERANKING,
            is_retryable=is_retryable,
            status_code=502,
            tenant_id=tenant_id,
            details=details,
        )


class LLMProviderError(RAGError):
    """Raised when an external LLM generation provider fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "LLM_PROVIDER_ERROR",
        is_retryable: bool = True,
        status_code: int = 502,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.LLM_PROVIDER,
            is_retryable=is_retryable,
            status_code=status_code,
            tenant_id=tenant_id,
            details=details,
        )


class TimeoutError(RAGError):
    """Raised when a pipeline stage or external service call exceeds its deadline."""

    def __init__(
        self,
        message: str = "Operation timed out.",
        error_code: str = "DEADLINE_EXCEEDED",
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.TIMEOUT,
            is_retryable=True,
            status_code=504,
            tenant_id=tenant_id,
            details=details,
        )


class RateLimitError(ConnectorError):
    """Raised when a provider or pipeline rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please retry after delay.",
        error_code: str = "RATE_LIMIT_EXCEEDED",
        retry_after_sec: Optional[float] = None,
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        det = details or {}
        if retry_after_sec is not None:
            det["retry_after_sec"] = retry_after_sec
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.RATE_LIMIT,
            is_retryable=True,
            status_code=429,
            tenant_id=tenant_id,
            details=det,
        )
        self.retry_after_sec = retry_after_sec


class ConfigurationError(RAGError):
    """Raised when invalid or missing system configuration is detected."""

    def __init__(
        self,
        message: str,
        error_code: str = "INVALID_CONFIGURATION",
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.CONFIGURATION,
            is_retryable=False,
            status_code=500,
            tenant_id=tenant_id,
            details=details,
        )


class SerializationError(RAGError):
    """Raised when serialization or deserialization of data models fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "SERIALIZATION_ERROR",
        tenant_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SERIALIZATION,
            is_retryable=False,
            status_code=400,
            tenant_id=tenant_id,
            details=details,
        )


class CircuitBreakerOpenError(RAGError):
    """Raised when a request is rejected because the external provider circuit is OPEN."""

    def __init__(
        self,
        provider_name: str,
        recovery_time_remaining_sec: float = 0.0,
        tenant_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Circuit breaker for provider '{provider_name}' is OPEN. Fast-failing request.",
            error_code="CIRCUIT_BREAKER_OPEN",
            category=ErrorCategory.CIRCUIT_BREAKER,
            is_retryable=False,
            status_code=503,
            tenant_id=tenant_id,
            details={"provider": provider_name, "recovery_time_remaining_sec": recovery_time_remaining_sec},
        )


# Legacy connector aliases
class ConnectionValidationError(ConnectorError):
    pass


class DiscoveryError(ConnectorError):
    pass


class FetchError(ConnectorError):
    pass


class IngestionPipelineError(RAGError):
    pass
