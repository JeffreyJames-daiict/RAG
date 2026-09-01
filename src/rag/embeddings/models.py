"""Pydantic schemas and result models for the embedding subsystem."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingStatus(str, Enum):
    """Execution status for an embedding task or batch."""
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class EmbeddingTask(BaseModel):
    """Represents a validated chunk prepared for embedding."""
    chunk_id: str
    text: str
    document_id: str
    version: int
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    """Container for a successfully generated and validated embedding vector."""
    chunk_id: str
    vector: List[float]
    model_name: str
    model_version: str = "v1.0"
    dimension: int
    provider_name: str
    prompt_tokens: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ChunkEmbeddingFailure(BaseModel):
    """Records failure details for a specific chunk embedding attempt."""
    chunk_id: str
    error_type: str
    error_message: str
    retry_count: int = 0


class BatchEmbeddingResult(BaseModel):
    """Aggregate result of a batch embedding operation."""
    status: EmbeddingStatus
    provider_name: str
    model_name: str
    model_version: str
    dimension: int
    successful_results: List[EmbeddingResult] = Field(default_factory=list)
    failed_chunks: List[ChunkEmbeddingFailure] = Field(default_factory=list)
    total_tokens: int = 0
    duration_ms: float = 0.0

    @property
    def total_count(self) -> int:
        return len(self.successful_results) + len(self.failed_chunks)

    @property
    def is_complete_success(self) -> bool:
        return len(self.failed_chunks) == 0 and len(self.successful_results) > 0


class EmbeddingSummary(BaseModel):
    """High-level summary of embedding generation across a document or connector."""
    document_id: Optional[str] = None
    total_chunks: int = 0
    embedded_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    model_name: str
    model_version: str
    provider_name: str
    duration_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)
